import random

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
    user_common_counts: list[list[int]]
    user_means: list[float]
    global_item_mean_ratings: dict[int, float]
    global_item_rating_counts: dict[int, int]
    global_mean_rating: float
    min_common_items: int
    similarity_shrinkage: float

    @classmethod
    def fit(
        cls,
        records: list[RatingRecord],
        min_common_items: int,
        similarity_shrinkage: float = 0.0,
    ) -> "UserBasedCFModel":
        assert min_common_items >= 1
        assert similarity_shrinkage >= 0.0
        user_ids, item_ids, user_index, item_index, ratings = _build_rating_matrix(records)
        user_similarity, user_common_counts = _compute_similarity_matrices(
            ratings,
            min_common_items,
            similarity_shrinkage,
        )
        item_values = {
            item_id: [
                float(row[item_index[item_id]])
                for row in ratings
                if row[item_index[item_id]] is not None
            ]
            for item_id in item_ids
        }
        return cls(
            user_ids=user_ids,
            item_ids=item_ids,
            user_index=user_index,
            item_index=item_index,
            ratings=ratings,
            user_similarity=user_similarity,
            user_common_counts=user_common_counts,
            user_means=[_mean_rating(row) for row in ratings],
            global_item_mean_ratings={
                item_id: sum(values) / len(values) for item_id, values in item_values.items()
            },
            global_item_rating_counts={
                item_id: len(values) for item_id, values in item_values.items()
            },
            global_mean_rating=sum(record.rating for record in records) / len(records),
            min_common_items=min_common_items,
            similarity_shrinkage=similarity_shrinkage,
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
        neighbors = [
            (similarity, float(rating), other_row)
            for other_row, similarity in enumerate(self.user_similarity[user_row])
            if other_row != user_row
            and similarity > 0.0
            and (rating := self.ratings[other_row][item_column]) is not None
        ]
        top_neighbors = sorted(neighbors, key=lambda row: -row[0])[:k]
        denominator = sum(abs(similarity) for similarity, _, _ in top_neighbors)
        if denominator == 0.0:
            return None
        prediction = self.user_means[user_row] + sum(
            similarity * (rating - self.user_means[other_row])
            for similarity, rating, other_row in top_neighbors
        ) / denominator
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

    def recommend_items(
        self,
        user_id: int,
        item_k: int,
        neighbor_k: int,
        relevant_rating_min: float,
        min_neighbor_support: int,
        score_shrinkage: float,
        fallback_pool_size: int,
        random_seed: int,
    ) -> list[ItemRecommendation]:
        assert item_k >= 1
        assert neighbor_k >= 1
        assert 1.0 <= relevant_rating_min <= 5.0
        assert min_neighbor_support >= 1
        assert score_shrinkage >= 0.0
        assert fallback_pool_size >= 1
        assert user_id in self.user_index, f"Unknown user_id: {user_id}"
        user_row = self.user_index[user_id]
        user_ratings = self.ratings[user_row]
        recommendations: list[ItemRecommendation] = []

        for item_column, item_id in enumerate(self.item_ids):
            if user_ratings[item_column] is not None:
                continue
            neighbors = [
                (similarity, float(rating), other_row)
                for other_row, similarity in enumerate(self.user_similarity[user_row])
                if other_row != user_row
                and similarity > 0.0
                and (rating := self.ratings[other_row][item_column]) is not None
            ]
            top_neighbors = sorted(neighbors, key=lambda row: -row[0])[:neighbor_k]
            support = len(top_neighbors)
            if support < min_neighbor_support:
                continue
            denominator = sum(abs(similarity) for similarity, _, _ in top_neighbors)
            assert denominator > 0.0
            predicted_rating = self.user_means[user_row] + sum(
                similarity * (rating - self.user_means[other_row])
                for similarity, rating, other_row in top_neighbors
            ) / denominator
            predicted_rating = min(5.0, max(1.0, predicted_rating))
            confidence = support / (support + score_shrinkage)
            rank_score = (
                confidence * predicted_rating
                + (1.0 - confidence) * self.global_item_mean_ratings[item_id]
            )
            recommendations.append(
                ItemRecommendation(
                    item_id=item_id,
                    predicted_rating=predicted_rating,
                    rank_score=rank_score,
                    neighbor_support=support,
                    source="neighbors",
                )
            )

        if recommendations:
            return sorted(recommendations, key=lambda row: (-row.rank_score, row.item_id))[:item_k]
        return self._fallback_global_high_rating_items(
            user_id=user_id,
            item_k=item_k,
            relevant_rating_min=relevant_rating_min,
            fallback_pool_size=fallback_pool_size,
            random_seed=random_seed,
        )

    def _fallback_global_high_rating_items(
        self,
        user_id: int,
        item_k: int,
        relevant_rating_min: float,
        fallback_pool_size: int,
        random_seed: int,
    ) -> list[ItemRecommendation]:
        user_ratings = self.ratings[self.user_index[user_id]]
        candidates = sorted(
            [
                ItemRecommendation(
                    item_id=item_id,
                    predicted_rating=self.global_item_mean_ratings[item_id],
                    rank_score=self.global_item_mean_ratings[item_id],
                    neighbor_support=0,
                    source="fallback",
                )
                for item_column, item_id in enumerate(self.item_ids)
                if user_ratings[item_column] is None
                and self.global_item_mean_ratings[item_id] >= relevant_rating_min
            ],
            key=lambda row: (
                -row.rank_score,
                -self.global_item_rating_counts[row.item_id],
                row.item_id,
            ),
        )[:fallback_pool_size]
        random.Random(random_seed + user_id).shuffle(candidates)
        return candidates[:item_k]


def _mean_rating(ratings: list[float | None]) -> float:
    values = [rating for rating in ratings if rating is not None]
    assert values, "ratings must contain at least one value"
    return sum(values) / len(values)


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
    similarity_shrinkage: float = 0.0,
) -> float:
    assert min_common_items >= 1
    assert similarity_shrinkage >= 0.0
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
    pearson = numerator / ((denominator_a * denominator_b) ** 0.5)
    return pearson * len(pairs) / (len(pairs) + similarity_shrinkage)


def _compute_similarity_matrices(
    ratings: RatingMatrix,
    min_common_items: int,
    similarity_shrinkage: float,
) -> tuple[list[list[float]], list[list[int]]]:
    assert ratings, "ratings must not be empty"
    size = len(ratings)
    similarity = [[0.0 for _ in range(size)] for _ in range(size)]
    common_counts = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        similarity[row][row] = 1.0
        common_counts[row][row] = sum(rating is not None for rating in ratings[row])
        for column in range(row + 1, size):
            common_count = sum(
                rating_a is not None and rating_b is not None
                for rating_a, rating_b in zip(ratings[row], ratings[column], strict=True)
            )
            score = _pearson_similarity(
                ratings[row],
                ratings[column],
                min_common_items,
                similarity_shrinkage,
            )
            similarity[row][column] = score
            similarity[column][row] = score
            common_counts[row][column] = common_count
            common_counts[column][row] = common_count
    return similarity, common_counts


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
