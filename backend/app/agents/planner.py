"""Planner: turns a TripRequest into agent queries, runs the agents, and hands
their results to the itinerary agent. Schemas live in app/schemas/planner.py.

Each agent is injected as an async callable (typed input -> typed output), so the
graph can be tested with fakes and the agents stay independent of this file.
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from langgraph.graph import END, START, StateGraph
import asyncio
from app.schemas.activities import Activity, ActivitySearchInput, ActivitySearchOutput
from app.schemas.common import Money
from app.schemas.flights import FlightSearchInput, FlightSearchOutput
from app.schemas.accomodation import HotelSearchInput, HotelSearchOutput
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

# Max activities handed to the itinerary agent, per day, by pace.
ACTIVITIES_PER_DAY = {"relaxed": 2, "balanced": 3, "packed": 5}

SEARCH_NODES = ["search_flights", "search_hotels", "search_activities"]


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


def _select_activities(
    out: ActivitySearchOutput | None, num_days: int, pace: str
) -> list[Activity]:
    """The itinerary agent must schedule EVERY activity it gets, so trim here:
    all must-dos first, then best-rated, capped by pace * days."""
    if out is None:
        return []
    must_ids = set(out.must_do_ids)
    must = [a for a in out.activities if a.id in must_ids]
    rest = sorted(
        (a for a in out.activities if a.id not in must_ids),
        key=lambda a: a.rating or 0,
        reverse=True,
    )
    cap = ACTIVITIES_PER_DAY.get(pace, 3) * num_days
    return (must + rest)[: max(cap, len(must))]


async def _run(fn, arg):
    """Await async agents; run sync (blocking) agents in a worker thread."""
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)   # callable objects like ItineraryAgent
    )
    return await fn(arg) if is_async else await asyncio.to_thread(fn, arg)


async def _call(name: str, fn, arg, retries: int):
    """Run an agent, retrying on failure. Returns (result, error)."""
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return await _run(fn, arg), None
        except Exception as exc:  # noqa: BLE001 - one failing agent must not kill the run
            last = exc
    return None, AgentError(agent=name, message=f"{type(last).__name__}: {last}")

# Graph
#
#   START -> plan --(ok)--> search_flights    --\
#              |            search_hotels     ---+--> itinerary -> END
#              |            search_activities --/
#              +--(failed)-> END

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

    async def search_flights(state: TripState) -> dict:
        out, err = await _call("flights", agents.flights, state.flight_query, retries)
        return {"flights": out, "errors": [err] if err else []}

    async def search_hotels(state: TripState) -> dict:
        out, err = await _call("hotels", agents.hotels, state.hotel_query, retries)
        return {"hotels": out, "errors": [err] if err else []}

    async def search_activities(state: TripState) -> dict:
        out, err = await _call("activities", agents.activities, state.activity_query, retries)
        return {"activities": out, "errors": [err] if err else []}

    async def build_itinerary(state: TripState) -> dict:
        r = state.request
        if not (state.flights or state.hotels or state.activities):
            return {
                "stage": Stage.FAILED,
                "errors": [AgentError(
                    agent="planner",
                    message="All search agents failed; nothing to build an itinerary from.",
                )],
            }

        flight = _pick(state.flights.options, state.flights.recommended_id) if state.flights else None
        hotel = _pick(state.hotels.options, state.hotels.recommended_id) if state.hotels else None
        num_days = (r.end_date - r.start_date).days + 1

        inp = ItineraryInput(
            destination=r.destination,
            start_date=r.start_date,
            end_date=r.end_date,
            flight=flight,
            hotel=hotel,
            activities=_select_activities(state.activities, num_days, r.pace),
            pace=r.pace,
            total_budget=r.total_budget,
            travelers=r.travelers,
        )
        # retries=0: the itinerary agent already has its own generate/validate loop.
        out, err = await _call("itinerary", agents.itinerary, inp, retries=0)
        if err:
            return {"stage": Stage.FAILED, "errors": [err]}
        return {"stage": Stage.DONE, "itinerary": out}

    # --- edges ------------------------------------------------------------- #
    def after_plan(state: TripState) -> list[str]:
        return [END] if state.stage == Stage.FAILED else SEARCH_NODES

    g = StateGraph(TripState)
    g.add_node("plan", plan)
    g.add_node("search_flights", search_flights)
    g.add_node("search_hotels", search_hotels)
    g.add_node("search_activities", search_activities)
    g.add_node("itinerary", build_itinerary)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, SEARCH_NODES + [END])  # fan-out (parallel)
    g.add_edge(SEARCH_NODES, "itinerary")                              # join: waits for all three
    g.add_edge("itinerary", END)
    return g.compile()


async def plan_trip(request: TripRequest, agents: Agents, retries: int = 1) -> TripState:
    graph = build_graph(agents, retries)
    result = await graph.ainvoke(TripState(request=request))
    return TripState.model_validate(result)