import json
import re
import shutil
import zipfile
from pathlib import Path

import duckdb
import hydra
from omegaconf import DictConfig

from datasets.movielens.schema import GenreRecord, ItemRecord, RatingRecord, UserRecord


RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _extract_zip_if_needed(zip_path: Path, raw_dir: Path) -> None:
    if raw_dir.exists() and any(raw_dir.iterdir()):
        return
    assert zip_path.exists(), f"MovieLens zip not found: {zip_path}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.startswith("ml-100k/") and not name.endswith("/")]
        assert members, "MovieLens zip does not contain ml-100k files"
        for member in members:
            target = raw_dir / Path(member).name
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _assert_required_files(paths: list[Path]) -> None:
    for path in paths:
        assert path.exists(), f"Required raw file not found: {path}"


def _write_ratings(con: duckdb.DuckDBPyConnection, source: Path, output: Path) -> int:
    query = f"""
        SELECT
            CAST(user_id AS INTEGER) AS user_id,
            CAST(item_id AS INTEGER) AS item_id,
            CAST(rating AS DOUBLE) AS rating,
            CAST(timestamp AS BIGINT) AS timestamp
        FROM read_csv(
            '{source.as_posix()}',
            delim='\\t',
            header=false,
            columns={{'user_id': 'INTEGER', 'item_id': 'INTEGER', 'rating': 'DOUBLE', 'timestamp': 'BIGINT'}}
        )
    """
    stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            MIN(rating) AS rating_min,
            MAX(rating) AS rating_max,
            SUM(CASE WHEN user_id IS NULL OR item_id IS NULL OR rating IS NULL OR timestamp IS NULL THEN 1 ELSE 0 END) AS nulls
        FROM ({query})
        """
    ).fetchone()
    assert stats is not None
    assert stats[0] > 0
    assert stats[1] >= 1
    assert stats[2] <= 5
    assert stats[3] == 0
    sample = con.execute(f"SELECT * FROM ({query}) LIMIT 1").fetchone()
    assert sample is not None
    RatingRecord.model_validate(dict(zip(RATING_COLUMNS, sample, strict=True)))
    con.execute(f"COPY ({query}) TO '{output.as_posix()}' (FORMAT PARQUET)")
    return int(stats[0])


def _read_genres(con: duckdb.DuckDBPyConnection, source: Path) -> list[tuple[str, int]]:
    genres = con.execute(
        f"""
        SELECT genre, CAST(genre_id AS INTEGER) AS genre_id
        FROM read_csv(
            '{source.as_posix()}',
            delim='|',
            header=false,
            columns={{'genre': 'VARCHAR', 'genre_id': 'INTEGER'}}
        )
        ORDER BY genre_id
        """
    ).fetchall()
    assert genres
    for genre, genre_id in genres:
        GenreRecord.model_validate({"genre": genre, "genre_id": genre_id})
    return [(str(genre), int(genre_id)) for genre, genre_id in genres]


def _genre_column_name(genre: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
    return f"genre_{normalized}"


def _write_genres(con: duckdb.DuckDBPyConnection, source: Path, output: Path) -> list[tuple[str, int]]:
    genres = _read_genres(con, source)
    con.execute(
        f"""
        COPY (
            SELECT genre, CAST(genre_id AS INTEGER) AS genre_id
            FROM read_csv(
                '{source.as_posix()}',
                delim='|',
                header=false,
                columns={{'genre': 'VARCHAR', 'genre_id': 'INTEGER'}}
            )
            ORDER BY genre_id
        ) TO '{output.as_posix()}' (FORMAT PARQUET)
        """
    )
    return genres


def _write_users(con: duckdb.DuckDBPyConnection, source: Path, output: Path) -> int:
    query = f"""
        SELECT
            CAST(user_id AS INTEGER) AS user_id,
            CAST(age AS INTEGER) AS age,
            gender,
            occupation,
            zip_code
        FROM read_csv(
            '{source.as_posix()}',
            delim='|',
            header=false,
            columns={{'user_id': 'INTEGER', 'age': 'INTEGER', 'gender': 'VARCHAR', 'occupation': 'VARCHAR', 'zip_code': 'VARCHAR'}}
        )
    """
    rows = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
    assert rows > 0
    sample = con.execute(f"SELECT * FROM ({query}) LIMIT 1").fetchone()
    assert sample is not None
    UserRecord.model_validate(dict(zip(USER_COLUMNS, sample, strict=True)))
    con.execute(f"COPY ({query}) TO '{output.as_posix()}' (FORMAT PARQUET)")
    return int(rows)


def _write_items(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    output: Path,
    genres: list[tuple[str, int]],
) -> int:
    genre_columns = [_genre_column_name(genre) for genre, _ in genres]
    raw_columns = {
        "item_id": "INTEGER",
        "title": "VARCHAR",
        "release_date": "VARCHAR",
        "video_release_date": "VARCHAR",
        "imdb_url": "VARCHAR",
    }
    raw_columns.update({column: "INTEGER" for column in genre_columns})
    column_spec = "{" + ", ".join(f"'{name}': '{kind}'" for name, kind in raw_columns.items()) + "}"
    selected_genres = ",\n            ".join(f"CAST({column} AS INTEGER) AS {column}" for column in genre_columns)
    query = f"""
        SELECT
            CAST(item_id AS INTEGER) AS item_id,
            title,
            release_date,
            imdb_url,
            {selected_genres}
        FROM read_csv(
            '{source.as_posix()}',
            delim='|',
            header=false,
            columns={column_spec},
            encoding='latin-1'
        )
    """
    rows = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
    assert rows > 0
    sample = con.execute(f"SELECT item_id, title, release_date, imdb_url FROM ({query}) LIMIT 1").fetchone()
    assert sample is not None
    ItemRecord.model_validate(
        {"item_id": sample[0], "title": sample[1], "release_date": sample[2], "imdb_url": sample[3]}
    )
    con.execute(f"COPY ({query}) TO '{output.as_posix()}' (FORMAT PARQUET)")
    return int(rows)


def _dataset_stats(con: duckdb.DuckDBPyConnection, processed_dir: Path) -> dict[str, int | float]:
    return {
        "num_train_ratings": con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{(processed_dir / 'ratings_train.parquet').as_posix()}')"
        ).fetchone()[0],
        "num_test_ratings": con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{(processed_dir / 'ratings_test.parquet').as_posix()}')"
        ).fetchone()[0],
        "num_users_train": con.execute(
            f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{(processed_dir / 'ratings_train.parquet').as_posix()}')"
        ).fetchone()[0],
        "num_items_train": con.execute(
            f"SELECT COUNT(DISTINCT item_id) FROM read_parquet('{(processed_dir / 'ratings_train.parquet').as_posix()}')"
        ).fetchone()[0],
        "num_users_test": con.execute(
            f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{(processed_dir / 'ratings_test.parquet').as_posix()}')"
        ).fetchone()[0],
        "num_items_test": con.execute(
            f"SELECT COUNT(DISTINCT item_id) FROM read_parquet('{(processed_dir / 'ratings_test.parquet').as_posix()}')"
        ).fetchone()[0],
        "rating_min": con.execute(
            f"""
            SELECT MIN(rating)
            FROM (
                SELECT rating FROM read_parquet('{(processed_dir / 'ratings_train.parquet').as_posix()}')
                UNION ALL
                SELECT rating FROM read_parquet('{(processed_dir / 'ratings_test.parquet').as_posix()}')
            )
            """
        ).fetchone()[0],
        "rating_max": con.execute(
            f"""
            SELECT MAX(rating)
            FROM (
                SELECT rating FROM read_parquet('{(processed_dir / 'ratings_train.parquet').as_posix()}')
                UNION ALL
                SELECT rating FROM read_parquet('{(processed_dir / 'ratings_test.parquet').as_posix()}')
            )
            """
        ).fetchone()[0],
    }


def preprocess(cfg: DictConfig) -> dict[str, int | float]:
    raw_dir = _resolve_path(str(cfg.dataset.raw_dir))
    processed_dir = _resolve_path(str(cfg.dataset.processed_dir))
    zip_path = _resolve_path(str(cfg.dataset.zip_path))

    _extract_zip_if_needed(zip_path, raw_dir)
    assert raw_dir.exists()
    processed_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        processed_dir / "ratings_train.parquet",
        processed_dir / "ratings_test.parquet",
        processed_dir / "items.parquet",
        processed_dir / "users.parquet",
        processed_dir / "genres.parquet",
        processed_dir / "dataset_stats.json",
    ]
    if not bool(cfg.preprocess.overwrite):
        for output in outputs:
            assert not output.exists(), f"Output already exists: {output}"

    train_path = raw_dir / str(cfg.dataset.ratings_train_file)
    test_path = raw_dir / str(cfg.dataset.ratings_test_file)
    items_path = raw_dir / str(cfg.dataset.items_file)
    users_path = raw_dir / str(cfg.dataset.users_file)
    genres_path = raw_dir / str(cfg.dataset.genres_file)
    _assert_required_files([train_path, test_path, items_path, users_path, genres_path])

    con = duckdb.connect(database=":memory:")
    _write_ratings(con, train_path, processed_dir / "ratings_train.parquet")
    _write_ratings(con, test_path, processed_dir / "ratings_test.parquet")
    genres = _write_genres(con, genres_path, processed_dir / "genres.parquet")
    _write_users(con, users_path, processed_dir / "users.parquet")
    _write_items(con, items_path, processed_dir / "items.parquet", genres)

    stats = _dataset_stats(con, processed_dir)
    assert stats["num_train_ratings"] > 0
    assert stats["num_test_ratings"] > 0
    assert stats["rating_min"] >= cfg["schema"].rating_min
    assert stats["rating_max"] <= cfg["schema"].rating_max
    (processed_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


@hydra.main(version_base=None, config_path=None, config_name="default")
def main(cfg: DictConfig) -> None:
    stats = preprocess(cfg)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
