from collections.abc import Awaitable, Callable
from dataclasses import dataclass
 
from langgraph.graph import END, START, StateGraph
 
from app.schemas.activities import ActivitySearchInput, ActivitySearchOutput
from app.schemas.common import Money
from app.schemas.flights import FlightSearchInput, FlightSearchOutput
from app.schemas.Hotels import HotelSearchInput, HotelSearchOutput
from app.schemas.itinerary import Itinerary, ItineraryInput
from app.schemas.planner import AgentError, Stage, TripRequest, TripState


# Agent interface

@dataclass(frozen=True)
class Agents:
    flights: Callable[[FlightSearchInput], Awaitable[FlightSearchOutput]]
    hotels: Callable[[HotelSearchInput], Awaitable[HotelSearchOutput]]
    activities: Callable[[ActivitySearchInput], Awaitable[ActivitySearchOutput]]
    itinerary: Callable[[ItineraryInput], Awaitable[Itinerary]]

    # Share of the total budget given to each agent. Tune freely.
BUDGET_SPLIT = {"flights": 0.35, "hotels": 0.40, "activities": 0.25}

def _share(budget: Money | None, key: str, divisor: int = 1) -> Money | None:
    if budget is None:
        return None
    amount = budget.amount * BUDGET_SPLIT[key] / divisor
    return Money(amount=round(amount, 2), currency=budget.currency)

def _pick(options: list, recommended_id: str):
    """Return the recommended option, falling back to the first one."""
    for option in options:
        if option.id == recommended_id:
            return option
    return options[0] if options else None

async def _call(name: str, fn, arg, retries: int):
    """Run an agent, retrying on failure. Returns (result, error)."""
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return await fn(arg), None
        except Exception as exc:  # noqa: BLE001 - one failing agent must not kill the run
            last = exc
    return None, AgentError(agent=name, message=f"{type(last).__name__}: {last}")
 
# Graph

def build_graph(agents: Agents, retries: int = 1):
    # --- nodes ------------------------------------------------------------- #
    def plan(state: TripState) -> dict:
        r = state.request
        if r.end_date <= r.start_date:
            return {
                "stage": Stage.FAILED,
                "errors": [AgentError(agent="planner", message="end_date must be after start_date")],
            }
        nights = (r.end_date - r.start_date).days
        num_days = nights + 1
        return {
            "stage": Stage.SEARCHING,
            "flight_query": FlightSearchInput(
                origin=r.origin,
                destination=r.destination,
                departure_date=r.start_date,
                return_date=r.end_date,
                passengers=r.travelers,
                cabin=r.cabin,
                max_price=_share(r.total_budget, "flights"),
                priority=r.priority,
            ),
            "hotel_query": HotelSearchInput(
                destination=r.destination,
                check_in=r.start_date,
                check_out=r.end_date,
                guests=r.travelers,
                max_price_per_night=_share(r.total_budget, "hotels", divisor=nights),
                priority=r.priority,
            ),
            "activity_query": ActivitySearchInput(
                destination=r.destination,
                interests=r.interests,
                num_days=num_days,
                budget_per_day=_share(r.total_budget, "activities", divisor=num_days),
                travelers=r.travelers,
                pace=r.pace,
            ),
        }



