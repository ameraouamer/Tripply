import operator
from pydantic import BaseModel, Field
from enum import Enum
from app.schemas.common import Priority , Money 
from app.schemas.flights import CabinClass, FlightSearchInput, FlightSearchOutput
from app.schemas.accomodation import HotelSearchInput, HotelSearchOutput
from itinerary import Itinerary, ItineraryInput
from datetime import date
from typing import Annotated
from activities import (
    ActivityCategory, ActivitySearchInput, ActivitySearchOutput,
)

class TripRequest(BaseModel):
    """What the user asks for. The planner turns this into per-agent inputs."""
    origin: str
    destination: str
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1)
    total_budget: Money | None = None
    cabin: CabinClass = CabinClass.ECONOMY
    interests: list[ActivityCategory] = Field(default_factory=list)
    pace: str = "balanced"
    priority: Priority = Priority.BEST_VALUE
    notes: str | None = Field(default=None, description="Free-text preferences")
 
 
class Stage(str, Enum):
    PLANNING = "planning"
    SEARCHING = "searching"
    ITINERARY = "itinerary"
    DONE = "done"
    FAILED = "failed"
 
 
class AgentError(BaseModel):
    agent: str
    message: str
 
 
class TripState(BaseModel):
    """Shared graph state. `errors` has a reducer because the three search
    nodes run in parallel and may each append to it."""
    request: TripRequest
    stage: Stage = Stage.PLANNING
 
    flight_query: FlightSearchInput | None = None
    hotel_query: HotelSearchInput | None = None
    activity_query: ActivitySearchInput | None = None
 
    flights: FlightSearchOutput | None = None
    hotels: HotelSearchOutput | None = None
    activities: ActivitySearchOutput | None = None
 
    itinerary: Itinerary | None = None
    errors: Annotated[list[AgentError], operator.add] = Field(default_factory=list)
