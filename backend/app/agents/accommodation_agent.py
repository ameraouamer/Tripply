"""
Accommodation agent: fetch live hotel rates (LiteAPI / Nuitée) and return them
in the app's own schema (HotelSearchInput -> HotelSearchOutput).

Layout
- LiteApiProvider : talks to LiteAPI and maps its JSON to HotelOption.
- HotelsAgent     : provider-independent logic (filters, ranking, pros/cons,
                    summary). Swap the provider without touching this part.

Env vars
- LITEAPI_KEY                 (required) sandbox keys start with "sand_"/"sandbox_"
- LITEAPI_GUEST_NATIONALITY   (optional, ISO-2, default "US") - rates can depend on it
"""
from __future__ import annotations

import asyncio
import os
from typing import Protocol

import requests
from dotenv import load_dotenv

from app.schemas.common import Money, Priority
from app.schemas.accomodation import (
    AccommodationType,
    HotelOption,
    HotelSearchInput,
    HotelSearchOutput,
)

load_dotenv()


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class HotelProvider(Protocol):
    def search(self, inp: HotelSearchInput) -> list[HotelOption]:
        """Return one priced option per available hotel (cheapest rate)."""
        ...


# --------------------------------------------------------------------------- #
# LiteAPI provider
# --------------------------------------------------------------------------- #
class LiteApiProvider:
    BASE = "https://api.liteapi.travel/v3.0"

    def __init__(self) -> None:
        self._key = os.environ["LITEAPI_KEY"]
        self._nationality = os.getenv("LITEAPI_GUEST_NATIONALITY", "US")

    @property
    def _headers(self) -> dict:
        return {
            "X-API-Key": self._key,
            "accept": "application/json",
            "content-type": "application/json",
        }

    # ---------------------------------------------------- destination -> placeId
    def _place_id(self, destination: str) -> str:
        r = requests.get(
            f"{self.BASE}/data/places",
            headers=self._headers,
            params={"textQuery": destination},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise ValueError(f"Could not resolve '{destination}' to a place")
        return data[0]["placeId"]

    # ------------------------------------------------------------------ fetch
    @staticmethod
    def _occupancies(guests: int, rooms: int) -> list[dict]:
        """Spread guests over rooms (no empty rooms): 5 guests, 2 rooms -> [3, 2]."""
        rooms = max(1, min(rooms, guests))
        base, extra = divmod(guests, rooms)
        return [{"adults": base + (1 if i < extra else 0)} for i in range(rooms)]

    def _fetch(self, inp: HotelSearchInput, currency: str) -> dict:
        body = {
            "placeId": self._place_id(inp.destination),
            "occupancies": self._occupancies(inp.guests, inp.rooms),
            "currency": currency,
            "guestNationality": self._nationality,
            "checkin": inp.check_in.isoformat(),
            "checkout": inp.check_out.isoformat(),
            "maxRatesPerHotel": 1,      # one (cheapest) rate per hotel
            "includeHotelData": True,   # name / address / rating / photo
        }
        r = requests.post(f"{self.BASE}/hotels/rates", headers=self._headers,
                          json=body, timeout=40)
        if r.status_code == 404:        # assumption: "no availability" -> 404; verify in sandbox
            return {}
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _to_option(entry: dict, info: dict, nights: int) -> HotelOption | None:
        rates = [rate for rt in entry.get("roomTypes", []) for rate in rt.get("rates", [])]
        rates = [r for r in rates if r.get("retailRate", {}).get("total")]
        if not rates:
            return None

        rate = min(rates, key=lambda r: float(r["retailRate"]["total"][0]["amount"]))
        t = rate["retailRate"]["total"][0]
        total, cur = float(t["amount"]), t["currency"]

        board = rate.get("boardName") or ""
        refundable = (rate.get("cancellationPolicies") or {}).get("refundableTag") == "RFN"
        taxes = rate["retailRate"].get("taxesAndFees") or []
        taxes_extra = any(not tx.get("included", True) for tx in taxes)

        pros, cons = [], []
        if "breakfast" in board.lower():
            pros.append("Breakfast included")
        if taxes_extra:
            cons.append("Taxes/fees not included in the price")

        rating = info.get("rating")
        return HotelOption(
            id=entry["hotelId"],
            name=info.get("name") or entry["hotelId"],
            type=AccommodationType.HOTEL,   # LiteAPI rates don't classify the property type
            address=info.get("address") or "",
            rating=float(rating) if rating else None,   # LiteAPI rating is already 0-10
            price_per_night=Money(amount=round(total / nights, 2), currency=cur),
            total_price=Money(amount=total, currency=cur),
            amenities=[board] if board else [],
            free_cancellation=refundable,
            pros=pros,
            cons=cons,
        )

    # ------------------------------------------------------------------ public
    def search(self, inp: HotelSearchInput) -> list[HotelOption]:
        nights = (inp.check_out - inp.check_in).days
        currency = inp.max_price_per_night.currency if inp.max_price_per_night else "USD"
        payload = self._fetch(inp, currency)

        info_by_id = {h.get("id"): h for h in payload.get("hotels", [])}
        options = []
        for entry in payload.get("data", []):
            opt = self._to_option(entry, info_by_id.get(entry["hotelId"], {}), nights)
            if opt is not None:
                options.append(opt)
        return options


# --------------------------------------------------------------------------- #
# Agent (provider-independent)
# --------------------------------------------------------------------------- #
class HotelsAgent:
    def __init__(self, provider: HotelProvider | None = None) -> None:
        self.provider = provider or LiteApiProvider()

    # ---------------------------------------------------------------- ranking
    @staticmethod
    def _rank(options: list[HotelOption], priority: Priority) -> list[HotelOption]:
        price = lambda o: o.price_per_night.amount
        rating = lambda o: o.rating or 0.0

        if priority == Priority.CHEAPEST:
            return sorted(options, key=lambda o: (price(o), -rating(o)))
        if priority == Priority.COMFORT:
            return sorted(options, key=lambda o: (-rating(o), price(o)))

        # BEST_VALUE (FASTEST has no meaning for hotels, so it falls through here)
        lo, hi = min(map(price, options)), max(map(price, options))
        norm = lambda v: 0 if hi == lo else (v - lo) / (hi - lo)

        def score(o: HotelOption) -> float:   # lower is better
            return 0.6 * norm(price(o)) + 0.4 * (1 - rating(o) / 10)
        return sorted(options, key=score)

    # ------------------------------------------------------------- pros / cons
    @staticmethod
    def _annotate(options: list[HotelOption]) -> None:
        cheapest = min(options, key=lambda o: o.price_per_night.amount)
        rated = [o for o in options if o.rating]
        best_rated = max(rated, key=lambda o: o.rating) if rated else None

        for o in options:
            if o is cheapest:
                o.pros.append("Cheapest option")
            else:
                diff = o.price_per_night.amount - cheapest.price_per_night.amount
                o.cons.append(f"{diff:.0f} {o.price_per_night.currency} more per night than the cheapest")
            if o is best_rated:
                o.pros.append(f"Highest rated ({o.rating:.1f}/10)")
            if o.free_cancellation:
                o.pros.append("Free cancellation")
            elif o.rating is None:
                o.cons.append("No guest rating available")

    # ------------------------------------------------------------- public API
    def search(self, inp: HotelSearchInput, top_n: int = 5) -> HotelSearchOutput:
        if inp.check_out <= inp.check_in:
            raise ValueError("check_out must be after check_in")

        options = self.provider.search(inp)
        if inp.max_price_per_night is not None:
            options = [o for o in options
                       if o.price_per_night.amount <= inp.max_price_per_night.amount]
        if inp.min_rating is not None:
            options = [o for o in options if (o.rating or 0) >= inp.min_rating]
        # NOTE v1: inp.types, inp.must_have_amenities and inp.near are not applied yet.

        if not options:
            return HotelSearchOutput(
                options=[], recommended_id="",
                comparison_summary="No hotels match your dates and constraints. "
                                   "Try other dates or loosen max price / min rating.",
            )

        ranked = self._rank(options, inp.priority)
        top = ranked[:top_n]
        cheapest = min(options, key=lambda o: o.price_per_night.amount)
        if cheapest not in top:                # always show the cheapest offer
            top = top[:-1] + [cheapest]
        self._annotate(top)

        best = top[0]
        cur = best.price_per_night.currency
        summary = (
            f"Recommended: {best.name}, {best.price_per_night.amount:.0f} {cur}/night "
            f"({best.total_price.amount:.0f} {cur} total)"
            + (f", rated {best.rating:.1f}/10" if best.rating else "")
            + f". Cheapest overall: {cheapest.name}, "
              f"{cheapest.price_per_night.amount:.0f} {cheapest.price_per_night.currency}/night."
        )
        return HotelSearchOutput(options=top, recommended_id=best.id, comparison_summary=summary)

    async def asearch(self, inp: HotelSearchInput, top_n: int = 5) -> HotelSearchOutput:
        """Async wrapper so it fits the planner's `Callable[..., Awaitable[...]]` contract."""
        return await asyncio.to_thread(self.search, inp, top_n)


# --------------------------------------------------------------------------- #
# Manual test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from datetime import date

    out = HotelsAgent().search(HotelSearchInput(
        destination="Paris",
        check_in=date(2026, 12, 20),
        check_out=date(2027, 1, 3),
        guests=2,
        priority=Priority.CHEAPEST,
    ))
    print(out.model_dump_json(indent=2))