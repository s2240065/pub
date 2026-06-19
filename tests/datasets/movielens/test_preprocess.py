import json
import zipfile
from pathlib import Path

import duckdb
from omegaconf import OmegaConf

from datasets.movielens.preprocess import preprocess


def _write_fixture_zip(path: Path) -> None:
    files = {
        "ml-100k/u1.base": "1\t1\t5\t10\n1\t2\t3\t11\n2\t1\t4\t12\n",
        "ml-100k/u1.test": "2\t2\t2\t13\n",
        "ml-100k/u.genre": "unknown|0\nAction|1\nComedy|2\n",
        "ml-100k/u.user": "1|24|M|technician|85711\n2|53|F|other|94043\n",
        "ml-100k/u.item": (
            "1|Toy Story (1995)|01-Jan-1995||http://example.test/1|0|1|1\n"
            "2|GoldenEye (1995)|01-Jan-1995||http://example.test/2|0|1|0\n"
        ),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_preprocess_writes_parquet_files_and_stats(tmp_path: Path) -> None:
    zip_path = tmp_path / "ml-100k.zip"
    raw_dir = tmp_path / "raw" / "movielens"
    processed_dir = tmp_path / "processed" / "movielens"
    _write_fixture_zip(zip_path)
    cfg = OmegaConf.create(
        {
            "dataset": {
                "zip_path": str(zip_path),
                "raw_dir": str(raw_dir),
                "processed_dir": str(processed_dir),
                "ratings_train_file": "u1.base",
                "ratings_test_file": "u1.test",
                "items_file": "u.item",
                "users_file": "u.user",
                "genres_file": "u.genre",
            },
            "preprocess": {"overwrite": True},
            "schema": {"rating_min": 1, "rating_max": 5},
        }
    )

    stats = preprocess(cfg)

    for filename in [
        "ratings_train.parquet",
        "ratings_test.parquet",
        "items.parquet",
        "users.parquet",
        "genres.parquet",
        "dataset_stats.json",
    ]:
        assert (processed_dir / filename).exists()

    con = duckdb.connect(database=":memory:")
    columns = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{(processed_dir / 'ratings_train.parquet').as_posix()}')"
    ).fetchall()
    assert [column[0] for column in columns] == ["user_id", "item_id", "rating", "timestamp"]
    assert stats["num_train_ratings"] == 3
    assert stats["num_test_ratings"] == 1
    assert json.loads((processed_dir / "dataset_stats.json").read_text()) == stats

    item_columns = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{(processed_dir / 'items.parquet').as_posix()}')"
    ).fetchall()
    assert [column[0] for column in item_columns] == [
        "item_id",
        "title",
        "release_date",
        "imdb_url",
        "genre_unknown",
        "genre_action",
        "genre_comedy",
    ]
