"""
Flights agent: fetch live offers (Amadeus) and return them in the app's own schema.
"""
from __future__ import annotations
import os
import re
import time
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

CABIN_TO_AMADEUS = {
    CabinClass.ECONOMY: "ECONOMY",
    CabinClass.PREMIUM_ECONOMY: "PREMIUM_ECONOMY",
    CabinClass.BUSINESS: "BUSINESS",
    CabinClass.FIRST: "FIRST",
}


class FlightsAgent:
    def __init__(self) -> None:
        env = os.getenv("AMADEUS_ENV", "test")
        self.base = "https://test.api.amadeus.com" if env == "test" else "https://api.amadeus.com"
        self._id = os.environ["AMADEUS_CLIENT_ID"]
        self._secret = os.environ["AMADEUS_CLIENT_SECRET"]
        self._token: str | None = None
        self._expiry = 0.0

    # ------------------------------------------------------------------ auth
    def _headers(self) -> dict:
        if not self._token or time.time() > self._expiry - 30:
            r = requests.post(
                f"{self.base}/v1/security/oauth2/token",
                data={"grant_type": "client_credentials",
                      "client_id": self._id, "client_secret": self._secret},
                timeout=20,
            )
            r.raise_for_status()
            d = r.json()
            self._token, self._expiry = d["access_token"], time.time() + int(d["expires_in"])
        return {"Authorization": f"Bearer {self._token}"}

    # ------------------------------------------------- city name -> IATA code
    def _resolve_iata(self, place: str) -> str:
        place = place.strip()
        if re.fullmatch(r"[A-Za-z]{3}", place):
            return place.upper()
        r = requests.get(
            f"{self.base}/v1/reference-data/locations",
            headers=self._headers(),
            params={"subType": "CITY,AIRPORT", "keyword": place, "page[limit]": 1},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise ValueError(f"Could not resolve '{place}' to an airport/city code")
        return data[0]["iataCode"]

    # ---------------------------------------------------------------- fetch
    def _fetch(self, inp: FlightSearchInput, currency: str) -> dict:
        params = {
            "originLocationCode": self._resolve_iata(inp.origin),
            "destinationLocationCode": self._resolve_iata(inp.destination),
            "departureDate": inp.departure_date.isoformat(),
            "adults": inp.passengers,
            "travelClass": CABIN_TO_AMADEUS[inp.cabin],
            "currencyCode": currency,
            "max": 50,
        }
        if inp.return_date:
            params["returnDate"] = inp.return_date.isoformat()
        if inp.max_stops == 0:
            params["nonStop"] = "true"
        r = requests.get(f"{self.base}/v2/shopping/flight-offers",
                         headers=self._headers(), params=params, timeout=40)
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------------- parse
    @staticmethod
    def _minutes(iso: str) -> int:
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso)
        return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)

    @staticmethod
    def _segments(itinerary: dict, carriers: dict) -> list[FlightSegment]:
        return [
            FlightSegment(
                airline=carriers.get(s["carrierCode"], s["carrierCode"]),
                flight_number=f'{s["carrierCode"]}{s["number"]}',
                origin=s["departure"]["iataCode"],
                destination=s["arrival"]["iataCode"],
                departure_time=datetime.fromisoformat(s["departure"]["at"]),
                arrival_time=datetime.fromisoformat(s["arrival"]["at"]),
            )
            for s in itinerary["segments"]
        ]

    def _to_option(self, raw: dict, carriers: dict, inp: FlightSearchInput) -> FlightOption:
        its = raw["itineraries"]
        return FlightOption(
            id=raw["id"],
            outbound=self._segments(its[0], carriers),
            inbound=self._segments(its[1], carriers) if len(its) > 1 else [],
            total_price=Money(amount=float(raw["price"]["grandTotal"]),   # <- Money fields
                              currency=raw["price"]["currency"]),
            total_duration_minutes=sum(self._minutes(i["duration"]) for i in its),
            stops=max(len(i["segments"]) - 1 for i in its),   # worst direction
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
        price = lambda o: o.total_price.amount                         # <- Money field
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
        currency = inp.max_price.currency if inp.max_price else "EUR"
        payload = self._fetch(inp, currency)
        carriers = payload.get("dictionaries", {}).get("carriers", {})

        options = [self._to_option(r, carriers, inp) for r in payload.get("data", [])]
        if inp.max_stops is not None:
            options = [o for o in options if o.stops <= inp.max_stops]
        if inp.max_price is not None:
            options = [o for o in options if o.total_price.amount <= inp.max_price.amount]

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
            f"{cheapest.total_duration_minutes % 60:02d})."
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