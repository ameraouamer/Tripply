"""
Itinerary agent: the LLM drafts the plan, code fixes the numbers and validates it.

    START -> generate -> check --(errors and attempts left)--> generate
                           |
                           +--(clean, or out of attempts)----> END

* generate : LLM call with with_structured_output(Itinerary). On a retry it also
             receives its previous draft plus the list of validation errors.
* check    : deterministic. Recomputes costs, sorts items, then validates.
* route    : the conditional edge that makes the loop agentic.
"""
import logging
from datetime import date, datetime, time, timedelta
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.schemas.common import Money
from app.schemas.itinerary import Itinerary, ItineraryInput

log = logging.getLogger(__name__)

ARRIVAL_BUFFER = timedelta(minutes=90)   # landing -> immigration, bags, transfer
DEPARTURE_BUFFER = timedelta(hours=3)    # be done sightseeing this long before takeoff
MAX_ERRORS_IN_PROMPT = 15

SYSTEM_PROMPT = """You are an expert travel itinerary planner. Return the plan as the \
requested structured object.

Rules:
- Exactly one DayPlan per calendar day from start_date to end_date inclusive. `day` starts at 1.
- Use ONLY the activities provided. Schedule EVERY one of them exactly once and set \
`activity_id` to its id. Never invent attractions. You may add meals, transit and free_time \
items (no activity_id).
- Group activities that are close together (use latitude/longitude/location) on the same \
day and order them to avoid backtracking.
- Respect each activity's opening_hours, best_time_of_day and duration_minutes. Leave \
realistic transit time between locations.
- Respect pace: relaxed = about 2 main activities per day with long breaks, balanced = about 3, \
packed = 4-5.
- Respect the flight constraints you are given: nothing (except the flight itself) before \
`earliest_activity_start` or after `latest_activity_end`. Add a `flight` item for the \
arrival and for the departure. Add a hotel check-in item after arrival and a check-out \
item on the last morning (if a hotel is given).
- Times are 24h and local. Items within a day must not overlap. No item may cross midnight.
- `estimated_cost` on meal/transit/free_time items is for the WHOLE group, in the given currency. \
Leave it null on activity items (it is filled in later).
- Give each day a short `theme`, write a 2-3 sentence `summary`, and add practical \
`packing_tips` for the destination and season.
- total_cost, daily_cost and within_budget are recomputed by the system: put 0 / null there.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class ItineraryValidationError(Exception):
    """Raised when the LLM could not produce a valid itinerary within max_attempts."""

    def __init__(self, errors: list[str], draft: Itinerary | None):
        super().__init__("Itinerary failed validation:\n- " + "\n- ".join(errors))
        self.errors = errors
        self.draft = draft


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


def _at(d: date, t: time) -> datetime:
    return datetime.combine(d, t)


def _currency(inp: ItineraryInput) -> str:
    for money in (
        inp.total_budget,
        inp.flight.total_price if inp.flight else None,
        inp.hotel.total_price if inp.hotel else None,
    ):
        if money:
            return money.currency
    return "USD"


def _amount(m: Money, cur: str) -> float:
    if m.currency != cur:
        # No FX in the pipeline yet: fail loudly rather than add EUR to USD.
        raise ValueError(f"Mixed currencies ({m.currency} vs {cur}); convert before the itinerary step.")
    return m.amount


def bounds(inp: ItineraryInput) -> tuple[datetime | None, datetime | None]:
    """(earliest moment an activity may start, latest moment one may end)."""
    earliest = latest = None
    f = inp.flight
    if f and f.outbound:
        earliest = _naive(f.outbound[-1].arrival_time) + ARRIVAL_BUFFER
    if f and f.inbound:
        latest = _naive(f.inbound[0].departure_time) - DEPARTURE_BUFFER
    return earliest, latest


# --------------------------------------------------------------------------- #
# Step 1: recompute everything numeric (never trust the LLM's arithmetic)
# --------------------------------------------------------------------------- #
def finalize(inp: ItineraryInput, draft: Itinerary) -> Itinerary:
    """Return a copy of `draft` with items sorted and all costs recomputed.

    Conventions:
    * Activity price is per person -> price * inp.travelers, taken from the source
      Activity (the LLM's number is discarded).
    * Flight and hotel are priced once at trip level (from their own totals).
    * daily_cost = on-the-ground items + that night's share of the hotel.
      The flight is NOT in daily_cost; it is only in total_cost.
    """
    it = draft.model_copy(deep=True)
    cur = _currency(inp)
    nights = (inp.end_date - inp.start_date).days
    acts = {a.id: a for a in inp.activities}

    flight_cost = _amount(inp.flight.total_price, cur) if inp.flight else 0.0
    hotel_cost = _amount(inp.hotel.total_price, cur) if inp.hotel else 0.0
    nightly = hotel_cost / nights if nights > 0 else 0.0

    # Top-level fields come from the request, not from the LLM.
    it.destination, it.start_date, it.end_date = inp.destination, inp.start_date, inp.end_date

    right_count = len(it.days) == nights + 1
    ground_total = 0.0
    for i, day in enumerate(it.days):
        day.items.sort(key=lambda x: x.start_time)
        if right_count:  # numbering and dates are deterministic, so just set them
            day.day, day.date = i + 1, inp.start_date + timedelta(days=i)

        spend = 0.0
        for item in day.items:
            act = acts.get(item.activity_id) if item.activity_id else None
            if act is not None:
                item.estimated_cost = (
                    Money(amount=round(_amount(act.price, cur) * inp.travelers, 2), currency=cur)
                    if act.price else None
                )
            elif item.kind in ("flight", "hotel"):
                item.estimated_cost = None  # counted at trip level
            elif item.estimated_cost:
                item.estimated_cost = Money(amount=item.estimated_cost.amount, currency=cur)
            if item.estimated_cost:
                spend += item.estimated_cost.amount

        ground_total += spend
        if i < nights:  # one hotel night per day except the last
            spend += nightly
        day.daily_cost = Money(amount=round(spend, 2), currency=cur)

    total = flight_cost + hotel_cost + ground_total
    it.total_cost = Money(amount=round(total, 2), currency=cur)
    it.within_budget = (
        total <= _amount(inp.total_budget, cur) if inp.total_budget else None
    )
    return it


# --------------------------------------------------------------------------- #
# Step 2: validate
# --------------------------------------------------------------------------- #
def validate(inp: ItineraryInput, it: Itinerary) -> list[str]:
    """Return a list of human-readable problems (empty list = valid).

    Expects `it` to have gone through finalize() (items sorted by start_time).
    Not checked: opening_hours (free text) and pace; those are left to the prompt.
    """
    errs: list[str] = []
    expected_days = (inp.end_date - inp.start_date).days + 1
    if len(it.days) != expected_days:
        errs.append(
            f"Expected {expected_days} days ({inp.start_date} to {inp.end_date}), got {len(it.days)}."
        )

    earliest, latest = bounds(inp)
    known = {a.id for a in inp.activities}
    seen: dict[str, int] = {}

    for day in it.days:
        label = f"Day {day.day} ({day.date})"
        if not day.items:
            errs.append(f"{label} has no items.")
        busy_until: datetime | None = None
        busy_title = ""

        for item in day.items:
            start = _at(day.date, item.start_time)
            end = _at(day.date, item.end_time) if item.end_time else None
            what = f"{label}: '{item.title}' ({item.start_time:%H:%M})"

            if end and end <= start:
                errs.append(f"{what} ends at or before it starts. Items must not cross midnight.")

            if item.activity_id:
                if item.activity_id not in known:
                    errs.append(f"{what} uses unknown activity_id '{item.activity_id}'.")
                else:
                    seen[item.activity_id] = seen.get(item.activity_id, 0) + 1

            # flight window (the flight items themselves are exempt)
            if item.kind != "flight":
                if earliest and start < earliest:
                    errs.append(
                        f"{what} starts before you can be there; earliest possible start is "
                        f"{earliest:%Y-%m-%d %H:%M} (arrival + buffer)."
                    )
                if latest and (end or start) > latest:
                    errs.append(
                        f"{what} ends after you must leave for the airport; latest end is "
                        f"{latest:%Y-%m-%d %H:%M}."
                    )

            # overlaps
            if busy_until and start < busy_until:
                errs.append(f"{label}: '{busy_title}' overlaps '{item.title}'.")
            this_end = end or start
            if busy_until is None or this_end >= busy_until:
                busy_until, busy_title = this_end, item.title

    # missing / duplicated activities
    for a in inp.activities:
        n = seen.get(a.id, 0)
        if n == 0:
            errs.append(f"Activity '{a.name}' (id={a.id}) is not scheduled.")
        elif n > 1:
            errs.append(f"Activity '{a.name}' (id={a.id}) is scheduled {n} times.")

    # flight items present on arrival and departure days
    f = inp.flight
    if f:
        def has_flight_item(d: date) -> bool:
            return any(i.kind == "flight" for dp in it.days if dp.date == d for i in dp.items)

        if f.outbound and not has_flight_item(_naive(f.outbound[-1].arrival_time).date()):
            errs.append("Missing a 'flight' item on the arrival day.")
        if f.inbound and not has_flight_item(_naive(f.inbound[0].departure_time).date()):
            errs.append("Missing a 'flight' item on the departure day.")

    return errs


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class State(TypedDict):
    inp: ItineraryInput
    draft: Itinerary | None
    errors: list[str]
    attempts: int


def _build_prompt(inp: ItineraryInput, draft: Itinerary | None, errors: list[str]) -> list:
    earliest, latest = bounds(inp)
    fmt = lambda dt: f"{dt:%Y-%m-%d %H:%M}" if dt else "none"
    context = (
        f"Trip request (JSON):\n{inp.model_dump_json()}\n\n"
        f"currency: {_currency(inp)}\n"
        f"travelers: {inp.travelers}\n"
        f"earliest_activity_start: {fmt(earliest)}\n"
        f"latest_activity_end: {fmt(latest)}\n"
    )
    if errors and draft is not None:
        context += (
            "\nYour previous attempt had problems.\n"
            f"Previous attempt:\n{draft.model_dump_json()}\n\n"
            "Fix ALL of these and return the full corrected itinerary:\n- "
            + "\n- ".join(errors[:MAX_ERRORS_IN_PROMPT])
        )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)]


class ItineraryAgent:
    """Callable matching `Agents.itinerary`: async (ItineraryInput) -> Itinerary."""

    def __init__(self, llm: BaseChatModel, max_attempts: int = 3) -> None:
        self._llm = llm.with_structured_output(Itinerary)
        self._max_attempts = max_attempts
        self._graph = self._build()

    # nodes ------------------------------------------------------------------
    async def _generate(self, state: State) -> dict:
        messages = _build_prompt(state["inp"], state["draft"], state["errors"])
        draft = await self._llm.ainvoke(messages)
        return {"draft": draft, "attempts": state["attempts"] + 1}

    def _check(self, state: State) -> dict:
        fixed = finalize(state["inp"], state["draft"])
        errors = validate(state["inp"], fixed)
        if errors:
            log.warning("Itinerary attempt %d: %d problem(s)", state["attempts"], len(errors))
        return {"draft": fixed, "errors": errors}

    # conditional edge ---------------------------------------------------------
    def _route(self, state: State) -> str:
        if state["errors"] and state["attempts"] < self._max_attempts:
            return "generate"
        return END

    def _build(self):
        g = StateGraph(State)
        g.add_node("generate", self._generate)
        g.add_node("check", self._check)
        g.add_edge(START, "generate")
        g.add_edge("generate", "check")
        g.add_conditional_edges("check", self._route, {"generate": "generate", END: END})
        return g.compile()

    # public API ---------------------------------------------------------------
    async def __call__(self, inp: ItineraryInput) -> Itinerary:
        final = await self._graph.ainvoke(
            {"inp": inp, "draft": None, "errors": [], "attempts": 0}
        )
        if final["errors"]:
            raise ItineraryValidationError(final["errors"], final["draft"])
        return final["draft"]