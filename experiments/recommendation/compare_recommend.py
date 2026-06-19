import argparse
import csv
import json
import statistics
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from experiments.user_based_cf.eval_recommend import (
    build_relevant_items_by_user,
    calculate_recommendation_metrics,
)
from models.baseline_recommenders.model import (
    GlobalMeanUnratedRecommender,
    RandomUnratedRecommender,
)
from models.user_based_cf.io import load_model, load_ratings_from_parquet
from models.user_based_cf.schema import TrainConfig


class ComparisonConfig(BaseModel):
    random_num_runs: int = Field(default=30, ge=1)
    min_item_rating_count: int = Field(default=5, ge=1)
    metrics_json_path: Path = Path("data/experiments/recommendation/comparison_metrics.json")
    metrics_csv_path: Path = Path("data/experiments/recommendation/comparison_metrics.csv")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _evaluate(
    relevant_items_by_user: dict[int, set[int]],
    recommendations_by_user: dict[int, list[int]],
    config: TrainConfig,
    num_users: int,
    catalog_size: int,
) -> dict[str, float | int]:
    assert catalog_size >= 1
    metrics = calculate_recommendation_metrics(
        relevant_items_by_user=relevant_items_by_user,
        recommended_items_by_user=recommendations_by_user,
        item_k=config.recommendation.item_k,
        neighbor_k=config.recommendation.neighbor_k,
        relevant_rating_min=config.recommendation.relevant_rating_min,
        min_neighbor_support=config.recommendation.min_neighbor_support,
        score_shrinkage=config.recommendation.score_shrinkage,
        num_users=num_users,
        num_fallback_users=0,
        neighbor_supports_by_user={user_id: [] for user_id in relevant_items_by_user},
    )
    result = metrics.model_dump()
    recommended_catalog = {
        item_id
        for recommendations in recommendations_by_user.values()
        for item_id in recommendations[:config.recommendation.item_k]
    }
    result["catalog_coverage"] = len(recommended_catalog) / catalog_size
    return result


def compare_recommenders(
    cfg: DictConfig,
    model_path: Path,
) -> list[dict[str, str | float | int]]:
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw_config, dict)
    config = TrainConfig.model_validate(raw_config)
    comparison = ComparisonConfig.model_validate(raw_config.get("comparison", {}))
    train_records = load_ratings_from_parquet(_resolve_path(config.dataset.train_path))
    test_records = load_ratings_from_parquet(_resolve_path(config.dataset.test_path))
    user_model = load_model(_resolve_path(model_path))
    relevant_items_by_user = build_relevant_items_by_user(
        records=test_records,
        model=user_model,
        relevant_rating_min=config.recommendation.relevant_rating_min,
    )
    num_users = len({record.user_id for record in test_records})
    user_ids = sorted(relevant_items_by_user)

    random_runs: list[dict[str, float | int]] = []
    for run_index in range(comparison.random_num_runs):
        random_model = RandomUnratedRecommender.fit(
            train_records,
            random_seed=config.recommendation.random_seed + run_index,
        )
        random_runs.append(
            _evaluate(
                relevant_items_by_user,
                {
                    user_id: [
                        row.item_id
                        for row in random_model.recommend_items(
                            user_id,
                            config.recommendation.item_k,
                        )
                    ]
                    for user_id in user_ids
                },
                config,
                num_users,
                len(user_model.item_ids),
            )
        )

    random_result: dict[str, str | float | int] = {
        "model_name": "random_unrated",
        "num_runs": comparison.random_num_runs,
    }
    for metric_name in (
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "coverage",
        "catalog_coverage",
    ):
        values = [float(run[metric_name]) for run in random_runs]
        random_result[f"{metric_name}_mean"] = statistics.fmean(values)
        random_result[f"{metric_name}_std"] = statistics.pstdev(values)
    random_result["num_evaluated_users"] = int(random_runs[0]["num_evaluated_users"])

    global_model = GlobalMeanUnratedRecommender.fit(
        train_records,
        min_item_rating_count=comparison.min_item_rating_count,
    )
    global_metrics = _evaluate(
        relevant_items_by_user,
        {
            user_id: [
                row.item_id
                for row in global_model.recommend_items(user_id, config.recommendation.item_k)
            ]
            for user_id in user_ids
        },
        config,
        num_users,
        len(user_model.item_ids),
    )
    global_result = {"model_name": "global_mean_unrated", **global_metrics}

    user_recommendations = {
        user_id: user_model.recommend_items(
            user_id=user_id,
            item_k=config.recommendation.item_k,
            neighbor_k=config.recommendation.neighbor_k,
            relevant_rating_min=config.recommendation.relevant_rating_min,
            min_neighbor_support=config.recommendation.min_neighbor_support,
            score_shrinkage=config.recommendation.score_shrinkage,
            fallback_pool_size=config.recommendation.fallback_pool_size,
            random_seed=config.recommendation.random_seed,
        )
        for user_id in user_ids
    }
    user_metrics = calculate_recommendation_metrics(
        relevant_items_by_user=relevant_items_by_user,
        recommended_items_by_user={
            user_id: [row.item_id for row in rows]
            for user_id, rows in user_recommendations.items()
        },
        item_k=config.recommendation.item_k,
        neighbor_k=config.recommendation.neighbor_k,
        relevant_rating_min=config.recommendation.relevant_rating_min,
        min_neighbor_support=config.recommendation.min_neighbor_support,
        score_shrinkage=config.recommendation.score_shrinkage,
        num_users=num_users,
        num_fallback_users=sum(
            bool(rows) and rows[0].source == "fallback"
            for rows in user_recommendations.values()
        ),
        neighbor_supports_by_user={
            user_id: [row.neighbor_support for row in rows]
            for user_id, rows in user_recommendations.items()
        },
    ).model_dump()
    user_catalog = {
        row.item_id
        for rows in user_recommendations.values()
        for row in rows[:config.recommendation.item_k]
    }
    user_metrics["catalog_coverage"] = len(user_catalog) / len(user_model.item_ids)
    return [random_result, global_result, {"model_name": "user_based_cf", **user_metrics}]


def save_results(
    results: list[dict[str, str | float | int]],
    json_path: Path,
    csv_path: Path,
) -> None:
    assert results
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for result in results for key in result})
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--model", required=True, type=Path)
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw_config, dict)
    comparison = ComparisonConfig.model_validate(raw_config.get("comparison", {}))
    results = compare_recommenders(cfg, args.model)
    save_results(
        results,
        _resolve_path(comparison.metrics_json_path),
        _resolve_path(comparison.metrics_csv_path),
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
