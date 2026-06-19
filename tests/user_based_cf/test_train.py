from pathlib import Path

import duckdb
from omegaconf import OmegaConf

from models.user_based_cf.io import load_model, load_ratings_from_parquet
from models.user_based_cf.train import train


def _write_ratings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.connect(database=":memory:").execute(
        f"""
        COPY (
            SELECT *
            FROM (
                VALUES
                    (1, 1, 5.0, 10),
                    (1, 2, 1.0, 11),
                    (2, 1, 4.0, 12),
                    (2, 2, 2.0, 13),
                    (3, 1, 1.0, 14),
                    (3, 2, 5.0, 15)
            ) AS ratings(user_id, item_id, rating, timestamp)
        ) TO '{path.as_posix()}' (FORMAT PARQUET)
        """
    )


def test_load_ratings_from_parquet_ignores_timestamp(tmp_path: Path) -> None:
    train_path = tmp_path / "ratings_train.parquet"
    _write_ratings(train_path)

    records = load_ratings_from_parquet(train_path)

    assert len(records) == 6
    assert records[0].user_id == 1
    assert records[0].item_id == 1
    assert records[0].rating == 5.0


def test_train_saves_reloadable_model(tmp_path: Path) -> None:
    train_path = tmp_path / "ratings_train.parquet"
    test_path = tmp_path / "ratings_test.parquet"
    model_path = tmp_path / "models" / "user_based_cf.pkl"
    _write_ratings(train_path)
    _write_ratings(test_path)
    cfg = OmegaConf.create(
        {
            "dataset": {"train_path": str(train_path), "test_path": str(test_path)},
            "model": {"name": "user_based_cf", "similarity": "pearson", "min_common_items": 2, "top_k": 2},
            "output": {"model_path": str(model_path)},
        }
    )

    stats = train(cfg)
    model = load_model(model_path)

    assert stats["num_ratings"] == 6
    assert stats["num_users"] == 3
    assert stats["num_items"] == 2
    assert model_path.exists()
    assert model.user_ids == [1, 2, 3]
    assert model.item_ids == [1, 2]


def test_trained_model_can_run_inference_after_reload(tmp_path: Path) -> None:
    train_path = tmp_path / "ratings_train.parquet"
    test_path = tmp_path / "ratings_test.parquet"
    model_path = tmp_path / "models" / "user_based_cf.pkl"
    _write_ratings(train_path)
    _write_ratings(test_path)
    cfg = OmegaConf.create(
        {
            "dataset": {"train_path": str(train_path), "test_path": str(test_path)},
            "model": {"name": "user_based_cf", "similarity": "pearson", "min_common_items": 2, "top_k": 2},
            "output": {"model_path": str(model_path)},
        }
    )

    train(cfg)
    model = load_model(model_path)

    assert model.similar_users(user_id=1, k=1)
    assert model.predict_rating(user_id=1, item_id=2, k=1) == 2.0
