"""Flights agent: find and compare flight options."""
from datetime import date, datetime
from enum import Enum
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from app.schemas.common import Money, Priority 


class CabinClass(str, Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Departure city or IATA code")
    destination: str = Field(description="Arrival city or IATA code")
    departure_date: date
    return_date: date | None = Field(default=None, description="None means one-way")
    passengers: int = Field(default=1, ge=1)
    cabin: CabinClass = CabinClass.ECONOMY
    max_price: Money | None = None
    max_stops: int | None = Field(default=None, ge=0)
    priority: Priority = Priority.BEST_VALUE


class FlightSegment(BaseModel):
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: datetime
    arrival_time: datetime


class FlightOption(BaseModel):
    id: str
    outbound: list[FlightSegment]
    inbound: list[FlightSegment] = Field(default_factory=list)
    total_price: Money
    total_duration_minutes: int
    stops: int
    cabin: CabinClass

class FlightSearchOutput(BaseModel):
    options: list[FlightOption]
    recommended_id: str = Field(description="id of the best option")
    comparison_summary: str = Field(description="Short explanation of the trade-offs")