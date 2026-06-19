import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from models.user_based_cf.io import load_ratings_from_parquet, save_model
from models.user_based_cf.model import UserBasedCFModel
from models.user_based_cf.schema import TrainConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _repo_root() / path


def train(cfg: DictConfig) -> dict[str, int | str]:
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    assert config.model.name == "user_based_cf"
    assert config.model.similarity == "pearson"
    train_path = _resolve_path(config.dataset.train_path)
    model_path = _resolve_path(config.output.model_path)
    assert train_path.exists(), f"Train parquet not found: {train_path}"
    records = load_ratings_from_parquet(train_path)
    model = UserBasedCFModel.fit(
        records=records,
        min_common_items=config.model.min_common_items,
        similarity_shrinkage=config.model.similarity_shrinkage,
    )
    assert model.user_ids
    assert model.item_ids
    save_model(model, model_path)
    assert model_path.exists(), f"Model file was not created: {model_path}"
    return {
        "model_path": model_path.as_posix(),
        "num_ratings": len(records),
        "num_users": len(model.user_ids),
        "num_items": len(model.item_ids),
    }


@hydra.main(version_base=None, config_path=None, config_name="default")
def main(cfg: DictConfig) -> None:
    print(json.dumps(train(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
