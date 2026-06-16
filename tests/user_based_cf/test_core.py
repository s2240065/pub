from models.user_based_cf.model import UserBasedCFModel, _build_rating_matrix, _pearson_similarity
from models.user_based_cf.schema import RatingRecord


def test_build_rating_matrix_sorts_ids_and_places_ratings() -> None:
    records = [
        RatingRecord(user_id=2, item_id=10, rating=4),
        RatingRecord(user_id=1, item_id=20, rating=5),
        RatingRecord(user_id=1, item_id=10, rating=3),
    ]

    user_ids, item_ids, user_index, item_index, ratings = _build_rating_matrix(records)

    assert user_ids == [1, 2]
    assert item_ids == [10, 20]
    assert user_index == {1: 0, 2: 1}
    assert item_index == {10: 0, 20: 1}
    assert ratings == [[3.0, 5.0], [4.0, None]]


def test_pearson_similarity_uses_common_items_only() -> None:
    ratings_a = [1.0, 2.0, None, 3.0]
    ratings_b = [1.0, 2.0, 5.0, 3.0]

    assert _pearson_similarity(ratings_a, ratings_b, min_common_items=2) == 1.0


def test_pearson_similarity_returns_zero_without_enough_variance_or_overlap() -> None:
    assert _pearson_similarity([5.0, None], [5.0, 4.0], min_common_items=2) == 0.0
    assert _pearson_similarity([3.0, 3.0], [4.0, 5.0], min_common_items=2) == 0.0


def test_model_fit_computes_user_similarity() -> None:
    model = UserBasedCFModel.fit(
        records=[
            RatingRecord(user_id=1, item_id=1, rating=5),
            RatingRecord(user_id=1, item_id=2, rating=1),
            RatingRecord(user_id=2, item_id=1, rating=4),
            RatingRecord(user_id=2, item_id=2, rating=2),
            RatingRecord(user_id=3, item_id=1, rating=1),
            RatingRecord(user_id=3, item_id=2, rating=5),
        ],
        min_common_items=2,
    )

    assert model.user_similarity[0][0] == 1.0
    assert model.user_similarity[0][1] == 1.0
    assert model.user_similarity[0][2] == -1.0
    assert [row.model_dump() for row in model.similar_users(user_id=1, k=2)] == [
        {"user_id": 2, "similarity": 1.0},
    ]


def test_model_predicts_ratings() -> None:
    model = UserBasedCFModel.fit(
        records=[
            RatingRecord(user_id=1, item_id=1, rating=5),
            RatingRecord(user_id=1, item_id=2, rating=1),
            RatingRecord(user_id=2, item_id=1, rating=4),
            RatingRecord(user_id=2, item_id=2, rating=2),
            RatingRecord(user_id=3, item_id=1, rating=1),
            RatingRecord(user_id=3, item_id=2, rating=5),
        ],
        min_common_items=2,
    )

    predicted = model.predict_rating(user_id=1, item_id=2, k=1)

    assert predicted == 2.0
