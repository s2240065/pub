import pickle
from pathlib import Path

import duckdb

from models.user_based_cf.model import UserBasedCFModel
from models.user_based_cf.schema import RatingRecord


def load_ratings_from_parquet(path: Path) -> list[RatingRecord]:
    assert path.exists(), f"Ratings parquet not found: {path}"
    rows = duckdb.connect(database=":memory:").execute(
        f"""
        SELECT
            CAST(user_id AS INTEGER) AS user_id,
            CAST(item_id AS INTEGER) AS item_id,
            CAST(rating AS DOUBLE) AS rating
        FROM read_parquet('{path.as_posix()}')
        ORDER BY user_id, item_id
        """
    ).fetchall()
    assert rows, f"Ratings parquet is empty: {path}"
    return [RatingRecord(user_id=row[0], item_id=row[1], rating=row[2]) for row in rows]


def save_model(model: UserBasedCFModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def load_model(path: Path) -> UserBasedCFModel:
    assert path.exists(), f"Model file not found: {path}"
    with path.open("rb") as file:
        model = pickle.load(file)
    assert isinstance(model, UserBasedCFModel)
    return model
