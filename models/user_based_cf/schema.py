from pathlib import Path

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    train_path: Path
    test_path: Path


class ModelConfig(BaseModel):
    name: str = "user_based_cf"
    similarity: str = "pearson"
    min_common_items: int = Field(ge=1)
    top_k: int = Field(ge=1)
    similarity_shrinkage: float = Field(default=10.0, ge=0)


class OutputConfig(BaseModel):
    model_path: Path


class QueryConfig(BaseModel):
    user_id: int = Field(default=1, ge=1)
    item_id: int = Field(default=1, ge=1)


class RecommendationConfig(BaseModel):
    item_k: int = Field(default=10, ge=1)
    neighbor_k: int = Field(default=40, ge=1)
    relevant_rating_min: float = Field(default=4.0, ge=1, le=5)
    min_neighbor_support: int = Field(default=2, ge=1)
    score_shrinkage: float = Field(default=5.0, ge=0)
    fallback_pool_size: int = Field(default=100, ge=1)
    random_seed: int = 42


class TrainConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    output: OutputConfig
    query: QueryConfig = QueryConfig()
    recommendation: RecommendationConfig = RecommendationConfig()


class RatingRecord(BaseModel):
    user_id: int = Field(ge=1)
    item_id: int = Field(ge=1)
    rating: float = Field(ge=1, le=5)


class SimilarUserResult(BaseModel):
    user_id: int = Field(ge=1)
    similarity: float


class RatingPrediction(BaseModel):
    user_id: int = Field(ge=1)
    item_id: int = Field(ge=1)
    actual_rating: float = Field(ge=1, le=5)
    predicted_rating: float = Field(ge=1, le=5)


class ItemRecommendation(BaseModel):
    item_id: int = Field(ge=1)
    predicted_rating: float = Field(ge=1, le=5)
    rank_score: float = Field(ge=1, le=5)
    neighbor_support: int = Field(ge=0)
    source: str
