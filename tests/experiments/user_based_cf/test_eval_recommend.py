import math

import pytest

from experiments.user_based_cf.eval_recommend import (
    build_relevant_items_by_user,
    calculate_recommendation_metrics,
)
from models.user_based_cf.model import UserBasedCFModel
from models.user_based_cf.schema import RatingRecord


def test_calculate_recommendation_metrics_uses_binary_relevance() -> None:
    metrics = calculate_recommendation_metrics(
        relevant_items_by_user={1: {3, 5}},
        recommended_items_by_user={1: [3, 4, 5]},
        item_k=3,
        neighbor_k=2,
        relevant_rating_min=4.0,
        min_neighbor_support=2,
        score_shrinkage=5.0,
        num_users=2,
        num_fallback_users=1,
        neighbor_supports_by_user={1: [3, 2, 1]},
    )

    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1.0
    assert metrics.ndcg_at_k == pytest.approx(expected_dcg / expected_idcg)
    assert metrics.num_evaluated_users == 1
    assert metrics.num_skipped_users == 1
    assert metrics.num_empty_recommendations == 0
    assert metrics.num_fallback_users == 1
    assert metrics.fallback_user_rate == 1.0
    assert metrics.coverage == 1.0
    assert metrics.avg_recommendation_count == 3.0
    assert metrics.avg_neighbor_support == 2.0


def test_empty_recommendations_are_included_as_zero_scores() -> None:
    metrics = calculate_recommendation_metrics(
        relevant_items_by_user={1: {3}, 2: {4}},
        recommended_items_by_user={1: [3], 2: []},
        item_k=2,
        neighbor_k=1,
        relevant_rating_min=4.0,
        min_neighbor_support=2,
        score_shrinkage=5.0,
        num_users=2,
        num_fallback_users=1,
        neighbor_supports_by_user={1: [2], 2: []},
    )

    assert metrics.precision_at_k == 0.25
    assert metrics.recall_at_k == 0.5
    assert metrics.ndcg_at_k == 0.5
    assert metrics.num_empty_recommendations == 1
    assert metrics.coverage == 0.5


def test_build_relevant_items_filters_ratings_unknown_ids_and_seen_items() -> None:
    model = UserBasedCFModel.fit(
        records=[
            RatingRecord(user_id=1, item_id=1, rating=5),
            RatingRecord(user_id=1, item_id=2, rating=1),
            RatingRecord(user_id=2, item_id=1, rating=4),
            RatingRecord(user_id=2, item_id=2, rating=2),
            RatingRecord(user_id=2, item_id=3, rating=5),
        ],
        min_common_items=2,
    )
    records = [
        RatingRecord(user_id=1, item_id=3, rating=4),
        RatingRecord(user_id=1, item_id=2, rating=5),
        RatingRecord(user_id=1, item_id=1, rating=3),
        RatingRecord(user_id=999, item_id=3, rating=5),
        RatingRecord(user_id=1, item_id=999, rating=5),
    ]

    assert build_relevant_items_by_user(records, model, relevant_rating_min=4.0) == {1: {3}}
