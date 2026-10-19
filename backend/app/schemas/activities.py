"""Activities agent: discover activities and attractions."""
from datetime import time
from enum import Enum
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from app.schemas.common import Money


class ActivityCategory(str, Enum):
    CULTURE = "culture"
    NATURE = "nature"
    FOOD = "food"
    ADVENTURE = "adventure"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    RELAXATION = "relaxation"
    FAMILY = "family"


class ActivitySearchInput(BaseModel):
    destination: str
    interests: list[ActivityCategory] = Field(default_factory=list)
    num_days: int = Field(ge=1)
    budget_per_day: Money | None = None
    travelers: int = Field(default=1, ge=1)
    pace: str = Field(default="balanced", description="relaxed | balanced | packed")
    accessibility_needs: list[str] = Field(default_factory=list)


class Activity(BaseModel):
    id: str
    name: str
    category: ActivityCategory
    description: str
    location: str
    latitude: float | None = None
    longitude: float | None = None
    duration_minutes: int = Field(gt=0)
    price: Money | None = None
    opening_hours: str | None = None
    best_time_of_day: str | None = Field(default=None, description="morning | afternoon | evening | any")
    booking_required: bool = False
    rating: float | None = Field(default=None, ge=0, le=5)


class ActivitySearchOutput(BaseModel):
    activities: list[Activity]
    must_do_ids: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, description="Seasonal tips, closures, local advice")