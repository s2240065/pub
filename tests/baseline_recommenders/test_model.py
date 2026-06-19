from models.baseline_recommenders.model import (
    GlobalMeanUnratedRecommender,
    RandomUnratedRecommender,
)
from models.user_based_cf.schema import RatingRecord


def _records() -> list[RatingRecord]:
    return [
        RatingRecord(user_id=1, item_id=1, rating=5),
        RatingRecord(user_id=1, item_id=2, rating=3),
        RatingRecord(user_id=2, item_id=2, rating=5),
        RatingRecord(user_id=2, item_id=3, rating=5),
        RatingRecord(user_id=3, item_id=3, rating=5),
        RatingRecord(user_id=3, item_id=4, rating=4),
    ]


def test_random_unrated_is_deterministic_and_excludes_seen_items() -> None:
    model = RandomUnratedRecommender.fit(_records(), random_seed=42)

    first = model.recommend_items(user_id=1, item_k=10)
    second = model.recommend_items(user_id=1, item_k=10)

    assert [row.item_id for row in first] == [row.item_id for row in second]
    assert {row.item_id for row in first}.isdisjoint({1, 2})


def test_global_mean_unrated_sorts_by_mean_count_and_item_id() -> None:
    model = GlobalMeanUnratedRecommender.fit(_records(), min_item_rating_count=1)

    recommendations = model.recommend_items(user_id=1, item_k=10)

    assert [row.item_id for row in recommendations] == [3, 4]
    assert all(row.source == "global_mean_unrated" for row in recommendations)


def test_global_mean_unrated_filters_items_below_minimum_count() -> None:
    model = GlobalMeanUnratedRecommender.fit(_records(), min_item_rating_count=2)

    assert [row.item_id for row in model.recommend_items(user_id=1, item_k=10)] == [3]
