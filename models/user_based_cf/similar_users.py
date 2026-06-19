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


def similar_users(cfg: DictConfig) -> list[dict[str, float | int]]:
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    user_id = int(cfg.query.user_id)
    model = load_model(_resolve_path(config.output.model_path))
    rows = model.similar_users(user_id=user_id, k=config.model.top_k)
    return [row.model_dump() for row in rows]


@hydra.main(version_base=None, config_path=None, config_name="default")
def main(cfg: DictConfig) -> None:
    print(json.dumps(similar_users(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
