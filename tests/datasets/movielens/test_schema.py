import pytest
from pydantic import ValidationError

from datasets.movielens.schema import RatingRecord


def test_rating_record_accepts_valid_record() -> None:
    record = RatingRecord(user_id=1, item_id=1, rating=4.0, timestamp=874965758)

    assert record.user_id == 1
    assert record.rating == 4.0


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": 1, "item_id": 1, "rating": 0, "timestamp": 1},
        {"user_id": 1, "item_id": 1, "rating": 6, "timestamp": 1},
        {"user_id": 0, "item_id": 1, "rating": 3, "timestamp": 1},
        {"user_id": 1, "item_id": 1, "rating": 3, "timestamp": -1},
    ],
)
def test_rating_record_rejects_invalid_record(payload: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RatingRecord.model_validate(payload)
