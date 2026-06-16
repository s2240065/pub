from pydantic import BaseModel

from models.user_based_cf.schema import (
    ItemRecommendation,
    RatingPrediction,
    RatingRecord,
    SimilarUserResult,
)


RatingMatrix = list[list[float | None]]


class UserBasedCFModel(BaseModel):
    user_ids: list[int]
    item_ids: list[int]
    user_index: dict[int, int]
    item_index: dict[int, int]
    ratings: RatingMatrix
    user_similarity: list[list[float]]
    min_common_items: int

    @classmethod
    def fit(cls, records: list[RatingRecord], min_common_items: int) -> "UserBasedCFModel":
        assert min_common_items >= 1
        user_ids, item_ids, user_index, item_index, ratings = _build_rating_matrix(records)
        return cls(
            user_ids=user_ids,
            item_ids=item_ids,
            user_index=user_index,
            item_index=item_index,
            ratings=ratings,
            user_similarity=_compute_similarity_matrix(ratings, min_common_items),
            min_common_items=min_common_items,
        )

    def similar_users(self, user_id: int, k: int) -> list[SimilarUserResult]:
        rows = _top_k_positive_similar_users(user_id, self.user_ids, self.user_similarity, k)
        return [SimilarUserResult(user_id=row[0], similarity=row[1]) for row in rows]

    def predict_rating(self, user_id: int, item_id: int, k: int) -> float | None:
        assert k >= 1
        assert user_id in self.user_index, f"Unknown user_id: {user_id}"
        assert item_id in self.item_index, f"Unknown item_id: {item_id}"
        user_row = self.user_index[user_id]
        item_column = self.item_index[item_id]
        neighbors: list[tuple[float, float]] = []
        for other_row, similarity in enumerate(self.user_similarity[user_row]):
            if other_row == user_row or similarity <= 0.0:
                continue
            rating = self.ratings[other_row][item_column]
            if rating is not None:
                neighbors.append((similarity, rating))

        top_neighbors = sorted(neighbors, key=lambda row: -row[0])[:k]
        denominator = sum(abs(similarity) for similarity, _ in top_neighbors)
        if denominator == 0.0:
            return None
        prediction = sum(similarity * rating for similarity, rating in top_neighbors) / denominator
        return min(5.0, max(1.0, prediction))

    def predict_many(self, records: list[RatingRecord], k: int) -> list[RatingPrediction]:
        predictions: list[RatingPrediction] = []
        for record in records:
            if record.user_id not in self.user_index or record.item_id not in self.item_index:
                continue
            predicted_rating = self.predict_rating(record.user_id, record.item_id, k)
            if predicted_rating is None:
                continue
            predictions.append(
                RatingPrediction(
                    user_id=record.user_id,
                    item_id=record.item_id,
                    actual_rating=record.rating,
                    predicted_rating=predicted_rating,
                )
            )
        return predictions

    def recommend_items(self, user_id: int, item_k: int, neighbor_k: int) -> list[ItemRecommendation]:
        assert item_k >= 1
        assert neighbor_k >= 1
        assert user_id in self.user_index, f"Unknown user_id: {user_id}"
        user_row = self.user_index[user_id]
        recommendations: list[ItemRecommendation] = []
        for item_column, item_id in enumerate(self.item_ids):
            if self.ratings[user_row][item_column] is not None:
                continue
            predicted_rating = self.predict_rating(user_id=user_id, item_id=item_id, k=neighbor_k)
            if predicted_rating is None:
                continue
            recommendations.append(ItemRecommendation(item_id=item_id, predicted_rating=predicted_rating))
        return sorted(recommendations, key=lambda row: (-row.predicted_rating, row.item_id))[:item_k]


def _build_rating_matrix(
    records: list[RatingRecord],
) -> tuple[list[int], list[int], dict[int, int], dict[int, int], RatingMatrix]:
    assert records, "records must not be empty"
    user_ids = sorted({record.user_id for record in records})
    item_ids = sorted({record.item_id for record in records})
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    ratings: RatingMatrix = [[None for _ in item_ids] for _ in user_ids]

    for record in records:
        row = user_index[record.user_id]
        column = item_index[record.item_id]
        assert ratings[row][column] is None, f"Duplicate rating: user={record.user_id}, item={record.item_id}"
        ratings[row][column] = record.rating

    return user_ids, item_ids, user_index, item_index, ratings


def _pearson_similarity(
    ratings_a: list[float | None],
    ratings_b: list[float | None],
    min_common_items: int,
) -> float:
    assert min_common_items >= 1
    assert len(ratings_a) == len(ratings_b)
    pairs = [(a, b) for a, b in zip(ratings_a, ratings_b, strict=True) if a is not None and b is not None]
    if len(pairs) < min_common_items:
        return 0.0

    values_a = [pair[0] for pair in pairs]
    values_b = [pair[1] for pair in pairs]
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    numerator = sum(a * b for a, b in zip(centered_a, centered_b, strict=True))
    denominator_a = sum(a * a for a in centered_a)
    denominator_b = sum(b * b for b in centered_b)
    if denominator_a == 0.0 or denominator_b == 0.0:
        return 0.0
    return numerator / ((denominator_a * denominator_b) ** 0.5)


def _compute_similarity_matrix(ratings: RatingMatrix, min_common_items: int) -> list[list[float]]:
    assert ratings, "ratings must not be empty"
    size = len(ratings)
    similarity = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        similarity[row][row] = 1.0
        for column in range(row + 1, size):
            score = _pearson_similarity(ratings[row], ratings[column], min_common_items)
            similarity[row][column] = score
            similarity[column][row] = score
    return similarity


def _top_k_positive_similar_users(
    user_id: int,
    user_ids: list[int],
    similarity: list[list[float]],
    k: int,
) -> list[tuple[int, float]]:
    assert k >= 1
    index_by_id = {value: index for index, value in enumerate(user_ids)}
    assert user_id in index_by_id, f"Unknown user_id: {user_id}"
    row_index = index_by_id[user_id]
    scores = [
        (candidate_user_id, similarity[row_index][candidate_index])
        for candidate_index, candidate_user_id in enumerate(user_ids)
        if candidate_index != row_index and similarity[row_index][candidate_index] > 0.0
    ]
    return sorted(scores, key=lambda score: (-score[1], score[0]))[:k]
