from fastapi import APIRouter, Depends, HTTPException

from app.agents.planner import Agents, plan_trip
from app.api.deps import get_agents
from app.schemas.planner import Stage, TripRequest, TripResponse

router = APIRouter(prefix="/trips", tags=["trips"])


@router.post("/plan", response_model=TripResponse)
async def plan(body: TripRequest, agents: Agents = Depends(get_agents)):
    state = await plan_trip(body, agents)
    if state.stage == Stage.FAILED:
        raise HTTPException(status_code=502, detail=[e.model_dump() for e in state.errors])
    return TripResponse(
        itinerary=state.itinerary,
        flights=state.flights,
        hotels=state.hotels,
        activities=state.activities,
        warnings=state.errors,   # partial failures (e.g. hotels failed but itinerary built)
    )