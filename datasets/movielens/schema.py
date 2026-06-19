from pydantic import BaseModel, Field


class RatingRecord(BaseModel):
    user_id: int = Field(ge=1)
    item_id: int = Field(ge=1)
    rating: float = Field(ge=1, le=5)
    timestamp: int = Field(ge=0)


class UserRecord(BaseModel):
    user_id: int = Field(ge=1)
    age: int = Field(ge=0)
    gender: str = Field(min_length=1)
    occupation: str = Field(min_length=1)
    zip_code: str = Field(min_length=1)


class GenreRecord(BaseModel):
    genre: str = Field(min_length=1)
    genre_id: int = Field(ge=0)


class ItemRecord(BaseModel):
    item_id: int = Field(ge=1)
    title: str = Field(min_length=1)
    release_date: str | None = None
    imdb_url: str | None = None
