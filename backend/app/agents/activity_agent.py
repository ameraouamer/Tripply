"""
Activities agent: real candidates from OpenTripMap, ranked/filtered by an LLM.

The LLM only SELECTS from the candidate list (by id) and estimates duration /
best time / booking. Names, locations and descriptions come from the data
source, so places can't be hallucinated. Opening hours and prices are not
provided by OpenTripMap, so they stay None.

Without OPENTRIPMAP_API_KEY it falls back to LLM-generated activities,
clearly flagged as unverified in `notes`.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Literal

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas.activities import (
    Activity,
    ActivityCategory,
    ActivitySearchInput,
    ActivitySearchOutput,
)

load_dotenv()
log = logging.getLogger(__name__)

OTM = "https://api.opentripmap.com/0.1/en"
RADIUS_M = 10_000
PER_CATEGORY = 20          # candidates kept per interest
EXTRA_PICKS = 2            # spare activities so the itinerary agent has choice
PER_DAY = {"relaxed": 2, "balanced": 3, "packed": 4}

# OpenTripMap "kinds" per category (check names against OTM's kinds list)
KINDS = {
    ActivityCategory.CULTURE: "cultural,historic,architecture",
    ActivityCategory.NATURE: "natural",
    ActivityCategory.FOOD: "foods",
    ActivityCategory.ADVENTURE: "sport,amusements",
    ActivityCategory.NIGHTLIFE: "bars,pubs,nightclubs",
    ActivityCategory.SHOPPING: "shops",
    ActivityCategory.RELAXATION: "gardens_and_parks,baths_and_saunas",
    ActivityCategory.FAMILY: "amusements,zoos",
}
DEFAULT_INTERESTS = [ActivityCategory.CULTURE, ActivityCategory.NATURE, ActivityCategory.FOOD]


# ------------------------------------------------------------ LLM I/O models
class Pick(BaseModel):
    id: str = Field(description="id copied EXACTLY from the candidate list")
    duration_minutes: int = Field(gt=0, le=480, description="realistic visit length")
    best_time_of_day: Literal["morning", "afternoon", "evening", "any"]
    booking_required: bool = Field(description="true only if you are fairly sure")
    must_do: bool = Field(description="a highlight of the destination")


class Selection(BaseModel):
    picks: list[Pick]
    notes: str | None = Field(default=None, description="short local advice, no invented facts")


class GeneratedActivity(BaseModel):          # fallback mode only
    name: str
    category: ActivityCategory
    description: str
    location: str
    duration_minutes: int = Field(gt=0, le=480)
    best_time_of_day: Literal["morning", "afternoon", "evening", "any"]
    booking_required: bool


class GeneratedList(BaseModel):
    activities: list[GeneratedActivity]


@dataclass
class Candidate:
    xid: str
    name: str
    category: ActivityCategory
    kinds: str
    rate: int
    dist_m: float


SELECT_SYSTEM = """You are a travel activity planner. You receive a numbered list of REAL places.
Choose exactly {n} of them for a {days}-day trip ({pace} pace, {travelers} traveler(s)).
Rules:
- Use ONLY ids from the list. Never invent or rename places.
- Match the traveler's interests and spread picks across categories.
- Prefer higher-rated places, but avoid choosing everything in one area when distances differ widely.
- Respect accessibility needs: {access}. Skip places that are clearly unsuitable (e.g. hikes, stairs-heavy sites).
- Budget hint per day: {budget}. Prefer free/cheap kinds when the budget is tight.
- duration_minutes, best_time_of_day and booking_required are estimates: be conservative.
- Mark at most 3 picks as must_do."""


class ActivitiesAgent:
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._key = os.getenv("OPENTRIPMAP_API_KEY")
        self._llm = llm or ChatOpenAI(
            model=os.getenv("ACTIVITIES_LLM_MODEL", "gpt-4o-mini"), temperature=0
        )

    # ------------------------------------------------------------ retrieval
    async def _geocode(self, http: httpx.AsyncClient, place: str) -> tuple[float, float]:
        r = await http.get(f"{OTM}/places/geoname", params={"name": place, "apikey": self._key})
        r.raise_for_status()
        d = r.json()
        if d.get("status") != "OK":
            raise ValueError(f"Could not geocode '{place}'")
        return d["lat"], d["lon"]

    async def _candidates(self, http, lat, lon, interests) -> list[Candidate]:
        async def one(cat: ActivityCategory):
            r = await http.get(f"{OTM}/places/radius", params={
                "radius": RADIUS_M, "lon": lon, "lat": lat, "kinds": KINDS[cat],
                "limit": 50, "format": "json", "apikey": self._key,
            })
            r.raise_for_status()
            return cat, r.json()

        results = await asyncio.gather(*(one(c) for c in interests))
        seen: dict[str, Candidate] = {}
        for cat, items in results:
            found = []
            for p in items:
                xid, name = p.get("xid"), (p.get("name") or "").strip()
                if not xid or not name or xid in seen:
                    continue
                rate = int(str(p.get("rate", "0"))[:1] or 0)
                found.append(Candidate(xid, name, cat, p.get("kinds", ""), rate, p.get("dist", 0)))
            found.sort(key=lambda c: (-c.rate, c.dist_m))
            for c in found[:PER_CATEGORY]:
                seen[c.xid] = c
        return list(seen.values())

    async def _details(self, http: httpx.AsyncClient, xid: str) -> dict:
        try:
            r = await http.get(f"{OTM}/places/xid/{xid}", params={"apikey": self._key})
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 - details are optional enrichment
            log.warning("details failed for %s: %s", xid, exc)
            return {}

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _describe(detail: dict, cand: Candidate, destination: str) -> str:
        text = (detail.get("wikipedia_extracts") or {}).get("text", "")
        if text:
            out = ""
            for sentence in text.split(". "):
                if len(out) + len(sentence) > 220:
                    break
                out += sentence.strip() + ". "
            if out:
                return out.strip()
        label = cand.kinds.split(",")[0].replace("_", " ") or cand.category.value
        return f"{label.capitalize()} in {destination}."

    @staticmethod
    def _location(detail: dict, cand: Candidate, destination: str) -> str:
        a = detail.get("address") or {}
        street = " ".join(x for x in (a.get("road"), a.get("house_number")) if x)
        parts = [cand.name, street, a.get("city") or a.get("town") or destination]
        return ", ".join(p for p in parts if p)

    # --------------------------------------------------------------- public
    async def search(self, inp: ActivitySearchInput) -> ActivitySearchOutput:
        if not self._key:
            return await self._unverified(inp)

        interests = inp.interests or DEFAULT_INTERESTS
        async with httpx.AsyncClient(timeout=20) as http:
            lat, lon = await self._geocode(http, inp.destination)
            cands = await self._candidates(http, lat, lon, interests)
            if not cands:
                return ActivitySearchOutput(
                    activities=[],
                    notes=f"No places found for '{inp.destination}' with these interests.",
                )

            n = min(PER_DAY.get(inp.pace.lower(), 3) * inp.num_days + EXTRA_PICKS, len(cands))
            selection = await self._select(inp, cands, n)

            by_id = {c.xid: c for c in cands}
            picks = [p for p in selection.picks if p.id in by_id][:n]   # drop hallucinated ids
            details = await asyncio.gather(*(self._details(http, p.id) for p in picks))

        activities, must_do = [], []
        for p, d in zip(picks, details):
            c = by_id[p.id]
            activities.append(Activity(
                id=c.xid,
                name=c.name,
                category=c.category,
                description=self._describe(d, c, inp.destination),
                location=self._location(d, c, inp.destination),
                duration_minutes=p.duration_minutes,
                price=None,                      # not available from OTM
                opening_hours=None,              # not available from OTM
                best_time_of_day=p.best_time_of_day,
                booking_required=p.booking_required,
                rating=None,
            ))
            if p.must_do:
                must_do.append(c.xid)

        notes = [
            "Places are real (OpenTripMap). Opening hours and prices are NOT verified: check before going.",
            "Duration, best time of day and booking_required are LLM estimates.",
        ]
        if inp.accessibility_needs:
            notes.append(
                f"Accessibility ({', '.join(inp.accessibility_needs)}) was applied by LLM judgement only; "
                "the data source has no accessibility information."
            )
        if selection.notes:
            notes.append(selection.notes)
        return ActivitySearchOutput(activities=activities, must_do_ids=must_do, notes=" ".join(notes))

    async def _select(self, inp, cands: list[Candidate], n: int) -> Selection:
        lines = "\n".join(
            f"{c.xid} | {c.name} | {c.category.value} | {c.kinds.split(',')[0]} | rate {c.rate} | {c.dist_m / 1000:.1f} km"
            for c in cands
        )
        budget = f"{inp.budget_per_day.amount:.0f} {inp.budget_per_day.currency}" if inp.budget_per_day else "none"
        system = SELECT_SYSTEM.format(
            n=n, days=inp.num_days, pace=inp.pace, travelers=inp.travelers,
            access=", ".join(inp.accessibility_needs) or "none", budget=budget,
        )
        chain = self._llm.with_structured_output(Selection)
        return await chain.ainvoke([
            ("system", system),
            ("human", f"Interests: {[i.value for i in (inp.interests or DEFAULT_INTERESTS)]}\n\nCandidates:\n{lines}"),
        ])

    # ------------------------------------------------ fallback: no data source
    async def _unverified(self, inp: ActivitySearchInput) -> ActivitySearchOutput:
        n = PER_DAY.get(inp.pace.lower(), 3) * inp.num_days + EXTRA_PICKS
        chain = self._llm.with_structured_output(GeneratedList)
        out: GeneratedList = await chain.ainvoke([
            ("system", "You suggest well-known, real attractions only. If unsure a place exists, leave it out. "
                       "Never state opening hours or prices."),
            ("human", f"Suggest {n} activities in {inp.destination} for interests "
                      f"{[i.value for i in inp.interests] or 'any'}, {inp.pace} pace, "
                      f"accessibility needs: {inp.accessibility_needs or 'none'}."),
        ])
        acts = [Activity(id=f"llm-{i}", **a.model_dump()) for i, a in enumerate(out.activities, 1)]
        return ActivitySearchOutput(
            activities=acts,
            must_do_ids=[],
            notes="UNVERIFIED: generated by an LLM without a places data source. "
                  "Confirm each place exists and check hours/prices before booking.",
        )


# ---------------------------------------------------------------- manual test
if __name__ == "__main__":
    inp = ActivitySearchInput(
        destination="Paris", num_days=3, pace="balanced",
        interests=[ActivityCategory.CULTURE, ActivityCategory.FOOD],
    )
    print(asyncio.run(ActivitiesAgent().search(inp)).model_dump_json(indent=2))