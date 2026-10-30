from datetime import date, datetime, time

from app.agents.itinerary_agent import finalize, validate
from app.schemas.activities import Activity, ActivityCategory
from app.schemas.common import Money
from app.schemas.flights import CabinClass, FlightOption, FlightSegment
from app.schemas.itinerary import DayPlan, Itinerary, ItineraryInput, ItineraryItem

D1, D2, D3 = date(2026, 12, 20), date(2026, 12, 21), date(2026, 12, 22)


def _flight() -> FlightOption:
    out = FlightSegment(airline="AF", flight_number="AF1", origin="ALG", destination="CDG",
                        departure_time=datetime(2026, 12, 20, 11, 0),
                        arrival_time=datetime(2026, 12, 20, 14, 0))   # earliest start 15:30
    back = FlightSegment(airline="AF", flight_number="AF2", origin="CDG", destination="ALG",
                         departure_time=datetime(2026, 12, 22, 18, 0),
                         arrival_time=datetime(2026, 12, 22, 21, 0))  # latest end 15:00
    return FlightOption(id="f1", outbound=[out], inbound=[back],
                        total_price=Money(amount=300, currency="EUR"),
                        total_duration_minutes=360, stops=0, cabin=CabinClass.ECONOMY)


def _inp(**kw) -> ItineraryInput:
    acts = [
        Activity(id="a1", name="Louvre", category=ActivityCategory.CULTURE, description="",
                 location="Paris", duration_minutes=120, price=Money(amount=20, currency="EUR")),
        Activity(id="a2", name="Seine cruise", category=ActivityCategory.RELAXATION, description="",
                 location="Paris", duration_minutes=60),
    ]
    return ItineraryInput(destination="Paris", start_date=D1, end_date=D3, flight=_flight(),
                          activities=acts, travelers=2,
                          total_budget=Money(amount=500, currency="EUR"), **kw)


def _item(title, start, end=None, kind="activity", activity_id=None, cost=None):
    return ItineraryItem(start_time=start, end_time=end, title=title, kind=kind,
                         activity_id=activity_id, estimated_cost=cost)


def _itinerary(days: list[list[ItineraryItem]]) -> Itinerary:
    dates = [D1, D2, D3]
    return Itinerary(
        destination="Paris", start_date=D1, end_date=D3,
        days=[DayPlan(day=i + 1, date=dates[i], items=items) for i, items in enumerate(days)],
        total_cost=Money(amount=0, currency="EUR"), summary="x",
    )


def test_validate_flags_all_problem_types():
    bad = _itinerary([
        [_item("Louvre", time(10), time(12), activity_id="a1")],           # before arrival, no flight item
        [_item("A", time(9), time(11)), _item("B", time(10), time(12))],   # overlap
        [_item("Late", time(16), time(17))],                               # after departure cutoff
    ])
    errs = " | ".join(validate(_inp(), finalize(_inp(), bad)))
    assert "starts before you can be there" in errs
    assert "overlaps" in errs
    assert "ends after you must leave" in errs
    assert "Seine cruise" in errs and "not scheduled" in errs
    assert "Missing a 'flight' item" in errs


def test_finalize_recomputes_costs_in_code():
    draft = _itinerary([
        [_item("Louvre", time(16), time(18), activity_id="a1", cost=Money(amount=999, currency="EUR"))],
        [], [],
    ])
    fixed = finalize(_inp(), draft)
    assert fixed.days[0].items[0].estimated_cost.amount == 40   # 20 EUR x 2 travelers
    assert fixed.days[0].daily_cost.amount == 40                # no hotel in this input
    assert fixed.total_cost.amount == 340                       # 300 flight + 40
    assert fixed.within_budget is True                          # budget 500