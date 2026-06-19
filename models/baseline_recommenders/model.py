import random

from pydantic import BaseModel

from models.user_based_cf.schema import ItemRecommendation, RatingRecord


class BaselineRecommender(BaseModel):
    item_ids: list[int]
    user_seen_items: dict[int, set[int]]
    global_mean_rating: float

    @classmethod
    def _training_state(
        cls,
        records: list[RatingRecord],
    ) -> tuple[list[int], dict[int, set[int]], float]:
        assert records, "records must not be empty"
        item_ids = sorted({record.item_id for record in records})
        user_seen_items: dict[int, set[int]] = {}
        for record in records:
            user_seen_items.setdefault(record.user_id, set()).add(record.item_id)
        return (
            item_ids,
            user_seen_items,
            sum(record.rating for record in records) / len(records),
        )

    def _unseen_items(self, user_id: int) -> list[int]:
        assert user_id in self.user_seen_items, f"Unknown user_id: {user_id}"
        return [item_id for item_id in self.item_ids if item_id not in self.user_seen_items[user_id]]


class RandomUnratedRecommender(BaselineRecommender):
    random_seed: int

    @classmethod
    def fit(
        cls,
        records: list[RatingRecord],
        random_seed: int,
    ) -> "RandomUnratedRecommender":
        item_ids, user_seen_items, global_mean_rating = cls._training_state(records)
        return cls(
            item_ids=item_ids,
            user_seen_items=user_seen_items,
            global_mean_rating=global_mean_rating,
            random_seed=random_seed,
        )

    def recommend_items(self, user_id: int, item_k: int) -> list[ItemRecommendation]:
        assert item_k >= 1
        unseen_items = self._unseen_items(user_id)
        rng = random.Random(self.random_seed + user_id)
        selected = rng.sample(unseen_items, k=min(item_k, len(unseen_items)))
        return [
            ItemRecommendation(
                item_id=item_id,
                predicted_rating=self.global_mean_rating,
                rank_score=self.global_mean_rating,
                neighbor_support=0,
                source="random_unrated",
            )
            for item_id in selected
        ]


class GlobalMeanUnratedRecommender(BaselineRecommender):
    ranked_item_ids: list[int]
    item_mean_ratings: dict[int, float]
    item_rating_counts: dict[int, int]
    min_item_rating_count: int

    @classmethod
    def fit(
        cls,
        records: list[RatingRecord],
        min_item_rating_count: int,
    ) -> "GlobalMeanUnratedRecommender":
        assert min_item_rating_count >= 1
        item_ids, user_seen_items, global_mean_rating = cls._training_state(records)
        ratings_by_item: dict[int, list[float]] = {}
        for record in records:
            ratings_by_item.setdefault(record.item_id, []).append(record.rating)
        item_mean_ratings = {
            item_id: sum(ratings) / len(ratings)
            for item_id, ratings in ratings_by_item.items()
        }
        item_rating_counts = {
            item_id: len(ratings)
            for item_id, ratings in ratings_by_item.items()
        }
        ranked_item_ids = sorted(
            (
                item_id
                for item_id in item_ids
                if item_rating_counts[item_id] >= min_item_rating_count
            ),
            key=lambda item_id: (
                -item_mean_ratings[item_id],
                -item_rating_counts[item_id],
                item_id,
            ),
        )
        return cls(
            item_ids=item_ids,
            user_seen_items=user_seen_items,
            global_mean_rating=global_mean_rating,
            ranked_item_ids=ranked_item_ids,
            item_mean_ratings=item_mean_ratings,
            item_rating_counts=item_rating_counts,
            min_item_rating_count=min_item_rating_count,
        )

    def recommend_items(self, user_id: int, item_k: int) -> list[ItemRecommendation]:
        assert item_k >= 1
        assert user_id in self.user_seen_items, f"Unknown user_id: {user_id}"
        seen_items = self.user_seen_items[user_id]
        selected = [
            item_id
            for item_id in self.ranked_item_ids
            if item_id not in seen_items
        ][:item_k]
        return [
            ItemRecommendation(
                item_id=item_id,
                predicted_rating=self.item_mean_ratings[item_id],
                rank_score=self.item_mean_ratings[item_id],
                neighbor_support=0,
                source="global_mean_unrated",
            )
            for item_id in selected
        ]
