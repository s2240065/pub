import argparse
import json
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from models.user_based_cf.io import load_model, load_ratings_from_parquet
from models.user_based_cf.model import UserBasedCFModel
from models.user_based_cf.schema import RatingRecord, TrainConfig


class RecommendationMetrics(BaseModel):
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    num_users: int = Field(ge=0)
    num_evaluated_users: int = Field(ge=0)
    num_skipped_users: int = Field(ge=0)
    num_empty_recommendations: int = Field(ge=0)
    item_k: int = Field(ge=1)
    neighbor_k: int = Field(ge=1)
    relevant_rating_min: float = Field(ge=1, le=5)
    min_neighbor_support: int = Field(ge=1)
    score_shrinkage: float = Field(ge=0)
    num_fallback_users: int = Field(ge=0)
    fallback_user_rate: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    avg_recommendation_count: float = Field(ge=0)
    avg_neighbor_support: float = Field(ge=0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _repo_root() / path


def build_relevant_items_by_user(
    records: list[RatingRecord],
    model: UserBasedCFModel,
    relevant_rating_min: float,
) -> dict[int, set[int]]:
    assert records, "records must not be empty"
    assert 1.0 <= relevant_rating_min <= 5.0
    relevant_items: dict[int, set[int]] = {}

    for record in records:
        if record.rating < relevant_rating_min:
            continue
        if record.user_id not in model.user_index or record.item_id not in model.item_index:
            continue
        user_row = model.user_index[record.user_id]
        item_column = model.item_index[record.item_id]
        if model.ratings[user_row][item_column] is not None:
            continue
        relevant_items.setdefault(record.user_id, set()).add(record.item_id)

    return relevant_items


def calculate_recommendation_metrics(
    relevant_items_by_user: dict[int, set[int]],
    recommended_items_by_user: dict[int, list[int]],
    item_k: int,
    neighbor_k: int,
    relevant_rating_min: float,
    min_neighbor_support: int,
    score_shrinkage: float,
    num_users: int,
    num_fallback_users: int,
    neighbor_supports_by_user: dict[int, list[int]],
) -> RecommendationMetrics:
    assert item_k >= 1
    assert neighbor_k >= 1
    assert min_neighbor_support >= 1
    assert score_shrinkage >= 0.0
    assert num_users >= len(relevant_items_by_user)
    assert 0 <= num_fallback_users <= len(relevant_items_by_user)
    assert set(recommended_items_by_user) == set(relevant_items_by_user)
    assert set(neighbor_supports_by_user) == set(relevant_items_by_user)
    assert all(items for items in relevant_items_by_user.values())

    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    num_empty_recommendations = 0

    for user_id, relevant_items in relevant_items_by_user.items():
        recommended_items = recommended_items_by_user[user_id][:item_k]
        assert len(recommended_items) == len(set(recommended_items))
        if not recommended_items:
            num_empty_recommendations += 1

        hits = [1 if item_id in relevant_items else 0 for item_id in recommended_items]
        precision_values.append(sum(hits) / item_k)
        recall_values.append(sum(hits) / len(relevant_items))
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        ideal_hits = min(len(relevant_items), item_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        assert idcg > 0.0
        ndcg_values.append(dcg / idcg)

    num_evaluated_users = len(relevant_items_by_user)
    assert num_evaluated_users > 0, "No users with relevant test items"
    recommendation_counts = [len(items[:item_k]) for items in recommended_items_by_user.values()]
    neighbor_supports = [
        support
        for supports in neighbor_supports_by_user.values()
        for support in supports[:item_k]
        if support > 0
    ]
    return RecommendationMetrics(
        precision_at_k=sum(precision_values) / num_evaluated_users,
        recall_at_k=sum(recall_values) / num_evaluated_users,
        ndcg_at_k=sum(ndcg_values) / num_evaluated_users,
        num_users=num_users,
        num_evaluated_users=num_evaluated_users,
        num_skipped_users=num_users - num_evaluated_users,
        num_empty_recommendations=num_empty_recommendations,
        item_k=item_k,
        neighbor_k=neighbor_k,
        relevant_rating_min=relevant_rating_min,
        min_neighbor_support=min_neighbor_support,
        score_shrinkage=score_shrinkage,
        num_fallback_users=num_fallback_users,
        fallback_user_rate=num_fallback_users / num_evaluated_users,
        coverage=(num_evaluated_users - num_empty_recommendations) / num_evaluated_users,
        avg_recommendation_count=sum(recommendation_counts) / num_evaluated_users,
        avg_neighbor_support=sum(neighbor_supports) / len(neighbor_supports) if neighbor_supports else 0.0,
    )


def eval_recommend(cfg: DictConfig, model_path: Path) -> dict[str, float | int | str]:
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    resolved_model_path = _resolve_path(model_path)
    model = load_model(resolved_model_path)
    records = load_ratings_from_parquet(_resolve_path(config.dataset.test_path))
    relevant_items_by_user = build_relevant_items_by_user(
        records=records,
        model=model,
        relevant_rating_min=config.recommendation.relevant_rating_min,
    )
    recommendations_by_user = {
        user_id: model.recommend_items(
            user_id=user_id,
            item_k=config.recommendation.item_k,
            neighbor_k=config.recommendation.neighbor_k,
            relevant_rating_min=config.recommendation.relevant_rating_min,
            min_neighbor_support=config.recommendation.min_neighbor_support,
            score_shrinkage=config.recommendation.score_shrinkage,
            fallback_pool_size=config.recommendation.fallback_pool_size,
            random_seed=config.recommendation.random_seed,
        )
        for user_id in relevant_items_by_user
    }
    recommended_items_by_user = {
        user_id: [recommendation.item_id for recommendation in recommendations]
        for user_id, recommendations in recommendations_by_user.items()
    }
    num_fallback_users = sum(
        bool(recommendations) and recommendations[0].source == "fallback"
        for recommendations in recommendations_by_user.values()
    )
    metrics = calculate_recommendation_metrics(
        relevant_items_by_user=relevant_items_by_user,
        recommended_items_by_user=recommended_items_by_user,
        item_k=config.recommendation.item_k,
        neighbor_k=config.recommendation.neighbor_k,
        relevant_rating_min=config.recommendation.relevant_rating_min,
        min_neighbor_support=config.recommendation.min_neighbor_support,
        score_shrinkage=config.recommendation.score_shrinkage,
        num_users=len({record.user_id for record in records}),
        num_fallback_users=num_fallback_users,
        neighbor_supports_by_user={
            user_id: [recommendation.neighbor_support for recommendation in recommendations]
            for user_id, recommendations in recommendations_by_user.items()
        },
    )
    return {"model_path": resolved_model_path.as_posix(), **metrics.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--model", required=True, type=Path)
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(
        config_dir=str(Path(args.config_dir).resolve()),
        version_base=None,
    ):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    print(json.dumps(eval_recommend(cfg, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
