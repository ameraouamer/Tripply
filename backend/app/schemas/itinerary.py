"""Itinerary agent: assemble flights, hotel and activities into the final plan."""
from datetime import date, time
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from app.schemas.common import Money
# pyrefly: ignore [missing-import]
from app.schemas.flights import FlightOption
# pyrefly: ignore [missing-import]
from app.schemas.Hotels import HotelOption
# pyrefly: ignore [missing-import]
from app.schemas.activities import Activity


class ItineraryInput(BaseModel):
    destination: str
    start_date: date
    end_date: date
    flight: FlightOption | None = None
    hotel: HotelOption | None = None
    activities: list[Activity] = Field(default_factory=list)
    pace: str = "balanced"
    total_budget: Money | None = None


class ItineraryItem(BaseModel):
    start_time: time
    end_time: time | None = None
    title: str
    kind: str = Field(description="flight | hotel | activity | meal | transit | free_time")
    location: str | None = None
    activity_id: str | None = Field(default=None, description="Links back to an Activity")
    estimated_cost: Money | None = None
    notes: str | None = None


class DayPlan(BaseModel):
    day: int = Field(ge=1)
    date: date
    theme: str | None = None
    items: list[ItineraryItem]
    daily_cost: Money | None = None


class Itinerary(BaseModel):
    destination: str
    start_date: date
    end_date: date
    days: list[DayPlan]
    total_cost: Money
    within_budget: bool | None = None
    packing_tips: list[str] = Field(default_factory=list)
    summary: str