from experiments.user_based_cf.eval_rating import calculate_rating_metrics
from models.user_based_cf.model import UserBasedCFModel
from models.user_based_cf.schema import RatingRecord


def test_calculate_rating_metrics_uses_predictable_ratings_only() -> None:
    records = [
        RatingRecord(user_id=1, item_id=2, rating=2),
        RatingRecord(user_id=999, item_id=2, rating=4),
        RatingRecord(user_id=1, item_id=999, rating=4),
    ]
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

    predictions = model.predict_many(records, k=1)
    metrics = calculate_rating_metrics(records, predictions)

    assert len(predictions) == 1
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.num_predicted == 1
    assert metrics.num_skipped == 2
    assert metrics.num_predicted + metrics.num_skipped == len(records)
    assert metrics.coverage == 1 / 3
