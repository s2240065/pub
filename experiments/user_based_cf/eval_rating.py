import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from models.user_based_cf.io import load_model, load_ratings_from_parquet
from models.user_based_cf.schema import RatingPrediction, RatingRecord, TrainConfig


class RatingMetrics(BaseModel):
    rmse: float = Field(ge=0)
    mae: float = Field(ge=0)
    num_predicted: int = Field(ge=0)
    num_skipped: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _repo_root() / path


def calculate_rating_metrics(records: list[RatingRecord], predictions: list[RatingPrediction]) -> RatingMetrics:
    assert records, "records must not be empty"
    assert predictions, "No predictable ratings"
    squared_errors = [
        (prediction.actual_rating - prediction.predicted_rating) ** 2 for prediction in predictions
    ]
    absolute_errors = [
        abs(prediction.actual_rating - prediction.predicted_rating) for prediction in predictions
    ]
    return RatingMetrics(
        rmse=(sum(squared_errors) / len(squared_errors)) ** 0.5,
        mae=sum(absolute_errors) / len(absolute_errors),
        num_predicted=len(predictions),
        num_skipped=len(records) - len(predictions),
        coverage=len(predictions) / len(records),
    )


def eval_rating(cfg: DictConfig) -> dict[str, float | int]:
    config = TrainConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    model = load_model(_resolve_path(config.output.model_path))
    records = load_ratings_from_parquet(_resolve_path(config.dataset.test_path))
    predictions = model.predict_many(records, k=config.recommendation.neighbor_k)
    metrics = calculate_rating_metrics(records, predictions)
    return metrics.model_dump()


@hydra.main(version_base=None, config_path=None, config_name="default")
def main(cfg: DictConfig) -> None:
    print(json.dumps(eval_rating(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
