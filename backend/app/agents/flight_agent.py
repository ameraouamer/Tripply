"""
Flights agent: fetch live offers (Ignav) and return them in the app's own schema.

.env
    IGNAV_API_KEY=...        # required (1,000 free requests, then pay as you go)
    IGNAV_MARKET=US          # optional, 2-letter country code; decides price currency
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import requests
from dotenv import load_dotenv
from app.schemas.common import Money, Priority

from app.schemas.flights import (
    CabinClass,
    FlightOption,
    FlightSearchInput,
    FlightSearchOutput,
    FlightSegment,
)

load_dotenv()


class FlightsAgent:
    BASE = "https://ignav.com/api"

    def __init__(self) -> None:
        self._key = os.environ["IGNAV_API_KEY"]
        self._market = os.getenv("IGNAV_MARKET", "US")

    # ------------------------------------------------------------------ auth
    @property
    def _headers(self) -> dict:
        # No OAuth flow, no token expiry: just a static header.
        return {"X-Api-Key": self._key, "Content-Type": "application/json"}

    # ------------------------------------------------- city name -> IATA code
    def _resolve_iata(self, place: str) -> str:
        place = place.strip()
        if re.fullmatch(r"[A-Za-z]{3}", place):
            return place.upper()
        r = requests.get(
            f"{self.BASE}/airports",
            headers=self._headers,
            params={"q": place, "limit": 1},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()                       # a plain list: [{"code": "ALG", ...}]
        if not data:
            raise ValueError(f"Could not resolve '{place}' to an airport/city code")
        return data[0]["code"]

    # ---------------------------------------------------------------- fetch
    def _fetch(self, inp: FlightSearchInput) -> dict:
        body = {
            "origin": self._resolve_iata(inp.origin),
            "destination": self._resolve_iata(inp.destination),
            "departure_date": inp.departure_date.isoformat(),
            "adults": inp.passengers,
            "cabin_class": inp.cabin.value,   # your enum values already match Ignav's
            "market": self._market,
        }
        if inp.max_stops is not None and inp.max_stops <= 2:   # Ignav accepts 0, 1 or 2
            body["max_stops"] = inp.max_stops

        if inp.return_date:
            body["return_date"] = inp.return_date.isoformat()
            endpoint = "round-trip"
        else:
            endpoint = "one-way"

        r = requests.post(f"{self.BASE}/fares/{endpoint}",
                          headers=self._headers, json=body, timeout=40)
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------------- parse
    @staticmethod
    def _segments(leg: dict) -> list[FlightSegment]:
        return [
            FlightSegment(
                airline=s.get("operating_carrier_name") or leg.get("carrier")
                        or s.get("marketing_carrier_code") or "Unknown",
                flight_number=f'{s.get("marketing_carrier_code") or ""}{s.get("flight_number") or ""}',
                origin=s["departure_airport"],
                destination=s["arrival_airport"],
                departure_time=datetime.fromisoformat(s["departure_time_local"]),
                arrival_time=datetime.fromisoformat(s["arrival_time_local"]),
            )
            for s in leg["segments"]
        ]

    @staticmethod
    def _leg_minutes(leg: dict) -> int:
        return leg.get("duration_minutes") or sum(s["duration_minutes"] for s in leg["segments"])

    def _to_option(self, raw: dict, inp: FlightSearchInput) -> FlightOption:
        legs = [raw["outbound"]] + ([raw["inbound"]] if raw.get("inbound") else [])
        return FlightOption(
            id=raw["ignav_id"],               # also the key for Ignav's booking-links endpoint
            outbound=self._segments(raw["outbound"]),
            inbound=self._segments(raw["inbound"]) if raw.get("inbound") else [],
            total_price=Money(amount=float(raw["price"]["amount"]),
                              currency=raw["price"]["currency"]),
            total_duration_minutes=sum(self._leg_minutes(l) for l in legs),
            stops=max(len(l["segments"]) - 1 for l in legs),   # worst direction
            cabin=inp.cabin,
        )

    # -------------------------------------------------------------- ranking
    @staticmethod
    def _mode(priority: Priority) -> str:
        key = f"{getattr(priority, 'name', '')} {getattr(priority, 'value', priority)}".lower()
        if "cheap" in key or "price" in key:
            return "cheapest"
        if "fast" in key or "quick" in key or "duration" in key:
            return "fastest"
        return "value"

    def _rank(self, options: list[FlightOption], priority: Priority) -> list[FlightOption]:
        mode = self._mode(priority)
        price = lambda o: o.total_price.amount
        if mode == "cheapest":
            return sorted(options, key=lambda o: (price(o), o.stops, o.total_duration_minutes))
        if mode == "fastest":
            return sorted(options, key=lambda o: (o.total_duration_minutes, price(o)))

        lo_p, hi_p = min(map(price, options)), max(map(price, options))
        lo_d = min(o.total_duration_minutes for o in options)
        hi_d = max(o.total_duration_minutes for o in options)
        norm = lambda v, lo, hi: 0 if hi == lo else (v - lo) / (hi - lo)

        def score(o: FlightOption) -> float:   # lower is better
            return (0.55 * norm(price(o), lo_p, hi_p)
                    + 0.30 * norm(o.total_duration_minutes, lo_d, hi_d)
                    + 0.15 * min(o.stops, 2) / 2)
        return sorted(options, key=score)

    # ------------------------------------------------------- pros / cons
    @staticmethod
    def _annotate(options: list[FlightOption]) -> None:
        cheapest = min(options, key=lambda o: o.total_price.amount)
        fastest = min(options, key=lambda o: o.total_duration_minutes)
        for o in options:
            if o is cheapest:
                o.pros.append("Cheapest option")
            else:
                diff = o.total_price.amount - cheapest.total_price.amount
                o.cons.append(f"{diff:.0f} {o.total_price.currency} more than the cheapest")
            if o is fastest:
                o.pros.append("Shortest total travel time")
            if o.stops == 0:
                o.pros.append("Direct flight")
            elif o.stops >= 2:
                o.cons.append(f"{o.stops} stops")

    # ------------------------------------------------------------ public API
    def search(self, inp: FlightSearchInput, top_n: int = 5) -> FlightSearchOutput:
        payload = self._fetch(inp)
        options = [self._to_option(r, inp) for r in payload.get("itineraries", [])]

        if inp.max_stops is not None:
            options = [o for o in options if o.stops <= inp.max_stops]

        # Ignav prices in the market's currency (no currency parameter), so a
        # budget in another currency can't be compared without FX: skip it and say so.
        note = ""
        if inp.max_price is not None and options:
            if options[0].total_price.currency == inp.max_price.currency:
                options = [o for o in options if o.total_price.amount <= inp.max_price.amount]
            else:
                note = (f" Note: budget is in {inp.max_price.currency} but prices are in "
                        f"{options[0].total_price.currency}; max price was not applied.")

        if not options:
            return FlightSearchOutput(
                options=[], recommended_id="",
                comparison_summary="No flights match your dates and constraints. "
                                   "Try other dates or loosen max price / max stops.",
            )

        ranked = self._rank(options, inp.priority)
        top = ranked[:top_n]
        cheapest = min(options, key=lambda o: o.total_price.amount)
        if cheapest not in top:               # always show the cheapest offer
            top = top[:-1] + [cheapest]
        self._annotate(top)

        best = top[0]
        summary = (
            f"Recommended: {best.total_price.amount:.0f} {best.total_price.currency}, "
            f"{best.total_duration_minutes // 60}h{best.total_duration_minutes % 60:02d}, "
            f"{best.stops} stop(s). Cheapest overall: "
            f"{cheapest.total_price.amount:.0f} {cheapest.total_price.currency} "
            f"({cheapest.stops} stop(s), {cheapest.total_duration_minutes // 60}h"
            f"{cheapest.total_duration_minutes % 60:02d})." + note
        )
        return FlightSearchOutput(options=top, recommended_id=best.id, comparison_summary=summary)


## test
if __name__ == "__main__":
    from datetime import date
    out = FlightsAgent().search(FlightSearchInput(
        origin="Algiers", destination="Paris",
        departure_date=date(2026, 12, 20), return_date=date(2027, 1, 3),
        priority=next(p for p in Priority if "cheap" in p.name.lower()),
    ))
    print(out.model_dump_json(indent=2))