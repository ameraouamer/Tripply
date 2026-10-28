"""Planner: turns a TripRequest into agent queries, runs the agents, and hands
their results to the itinerary agent. Schemas live in app/schemas/planner.py.

Flow:
    START -> plan -> [flights | hotels | activities]  (parallel)
                          -> collect -> itinerary -> END
    Any stage can short-circuit to END with stage=FAILED.

Each agent is injected as an async callable (typed input -> typed output), so the
graph can be tested with fakes and the agents stay independent of this file.
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from app.schemas.activities import ActivitySearchInput, ActivitySearchOutput
from app.schemas.common import Money
from app.schemas.flights import FlightSearchInput, FlightSearchOutput
from app.schemas.Hotels import HotelSearchInput, HotelSearchOutput
from app.schemas.itinerary import Itinerary, ItineraryInput
from app.schemas.planner import AgentError, Stage, TripRequest, TripState

# --------------------------------------------------------------------------- #
# Agent interface
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
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

    async def flights(state: TripState) -> dict:
        result, error = await _call("flights", agents.flights, state.flight_query, retries)
        return {"flights": result} if result else {"errors": [error]}

    async def hotels(state: TripState) -> dict:
        result, error = await _call("hotels", agents.hotels, state.hotel_query, retries)
        return {"hotels": result} if result else {"errors": [error]}

    async def activities(state: TripState) -> dict:
        result, error = await _call("activities", agents.activities, state.activity_query, retries)
        return {"activities": result} if result else {"errors": [error]}

    def collect(state: TripState) -> dict:
        """Join point: runs once all three search nodes have finished."""
        if state.flights or state.hotels or state.activities:
            return {"stage": Stage.ITINERARY}
        return {"stage": Stage.FAILED}

    async def itinerary(state: TripState) -> dict:
        r = state.request
        payload = ItineraryInput(
            destination=r.destination,
            start_date=r.start_date,
            end_date=r.end_date,
            flight=_pick(state.flights.options, state.flights.recommended_id) if state.flights else None,
            hotel=_pick(state.hotels.options, state.hotels.recommended_id) if state.hotels else None,
            activities=state.activities.activities if state.activities else [],
            pace=r.pace,
            total_budget=r.total_budget,
        )
        result, error = await _call("itinerary", agents.itinerary, payload, retries)
        if result is None:
            return {"stage": Stage.FAILED, "errors": [error]}
        return {"stage": Stage.DONE, "itinerary": result}

    # --- routing ----------------------------------------------------------- #
    def after_plan(state: TripState) -> list[str]:
        if state.stage == Stage.FAILED:
            return [END]
        return ["flights", "hotels", "activities"]

    def after_collect(state: TripState) -> str:
        return END if state.stage == Stage.FAILED else "itinerary"

    # --- wiring ------------------------------------------------------------ #
    g = StateGraph(TripState)
    g.add_node("plan", plan)
    g.add_node("flights", flights)
    g.add_node("hotels", hotels)
    g.add_node("activities", activities)
    g.add_node("collect", collect)
    g.add_node("itinerary", itinerary)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["flights", "hotels", "activities", END])
    g.add_edge(["flights", "hotels", "activities"], "collect")  # waits for all three
    g.add_conditional_edges("collect", after_collect, ["itinerary", END])
    g.add_edge("itinerary", END)
    return g.compile()


async def plan_trip(request: TripRequest, agents: Agents, retries: int = 1) -> TripState:
    graph = build_graph(agents, retries)
    result = await graph.ainvoke(TripState(request=request))
    return TripState.model_validate(result)