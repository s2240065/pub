import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from models.user_based_cf.io import load_model
from models.user_based_cf.schema import TrainConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _repo_root() / path


def predict_rating(cfg: DictConfig) -> dict[str, float | int | None]:
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    model = load_model(_resolve_path(config.output.model_path))
    predicted_rating = model.predict_rating(
        user_id=config.query.user_id,
        item_id=config.query.item_id,
        k=config.recommendation.neighbor_k,
    )
    return {
        "user_id": config.query.user_id,
        "item_id": config.query.item_id,
        "predicted_rating": predicted_rating,
    }


@hydra.main(version_base=None, config_path=None, config_name="default")
def main(cfg: DictConfig) -> None:
    print(json.dumps(predict_rating(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
