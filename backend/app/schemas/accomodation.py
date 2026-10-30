"""Hotels agent: find and compare accommodation options."""
from datetime import date
from enum import Enum
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from app.schemas.common import Money, Priority


class AccommodationType(str, Enum):
    HOTEL = "hotel"
    HOSTEL = "hostel"
    APARTMENT = "apartment"
    GUESTHOUSE = "guesthouse"
    RESORT = "resort"


class HotelSearchInput(BaseModel):
    destination: str
    check_in: date
    check_out: date
    guests: int = Field(default=1, ge=1)
    rooms: int = Field(default=1, ge=1)
    max_price_per_night: Money | None = None
    min_rating: float | None = Field(default=None, ge=0, le=10)
    types: list[AccommodationType] = Field(default_factory=list)
    must_have_amenities: list[str] = Field(default_factory=list)
    near: str | None = Field(default=None, description="Landmark or area to stay close to")
    priority: Priority = Priority.BEST_VALUE


class HotelOption(BaseModel):
    id: str
    name: str
    type: AccommodationType
    address: str
    neighborhood: str | None = None
    rating: float | None = Field(default=None, ge=0, le=10)
    price_per_night: Money
    total_price: Money
    amenities: list[str] = Field(default_factory=list)
    distance_to_center_km: float | None = None
    free_cancellation: bool = False
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class HotelSearchOutput(BaseModel):
    options: list[HotelOption]
    recommended_id: str
    comparison_summary: str