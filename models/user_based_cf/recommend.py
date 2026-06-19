import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from models.user_based_cf.io import load_model
from models.user_based_cf.schema import TrainConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _repo_root() / path


def recommend_items(cfg: DictConfig, model_path: Path) -> dict[str, str | list[dict[str, float | int | str]]]:
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    resolved_model_path = _resolve_path(model_path)
    model = load_model(resolved_model_path)
    rows = model.recommend_items(
        user_id=config.query.user_id,
        item_k=config.recommendation.item_k,
        neighbor_k=config.recommendation.neighbor_k,
        relevant_rating_min=config.recommendation.relevant_rating_min,
        min_neighbor_support=config.recommendation.min_neighbor_support,
        score_shrinkage=config.recommendation.score_shrinkage,
        fallback_pool_size=config.recommendation.fallback_pool_size,
        random_seed=config.recommendation.random_seed,
    )
    return {
        "model_path": resolved_model_path.as_posix(),
        "items": [row.model_dump() for row in rows],
    }


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
    print(json.dumps(recommend_items(cfg, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
