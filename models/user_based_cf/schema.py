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


class OutputConfig(BaseModel):
    model_path: Path


class QueryConfig(BaseModel):
    user_id: int = Field(default=1, ge=1)
    item_id: int = Field(default=1, ge=1)


class RecommendationConfig(BaseModel):
    item_k: int = Field(default=10, ge=1)
    neighbor_k: int = Field(default=20, ge=1)


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
