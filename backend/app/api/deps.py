# app/api/deps.pyOI
import os
from fastapi import Request
from langchain_openai import ChatOpenAI

from app.agents.planner import Agents
from app.agents.flight_agent import FlightsAgent
from app.agents.accommodation_agent import HotelsAgent
from app.agents.activity_agent import ActivitiesAgent
from app.agents.itinerary_agent import ItineraryAgent

# Wiring the agents together. This is the single place where we instantiate the agents and their dependencies.
def build_agents() -> Agents:
    llm = ChatOpenAI(model=os.getenv("OPENAI_API_KEY", "gpt-4o"), temperature=0)
    return Agents(
        flights=FlightsAgent().search,        # sync -> planner's _run uses a worker thread
        hotels=HotelsAgent().asearch,         # async wrapper already exists
        activities=ActivitiesAgent().search,  # async
        itinerary=ItineraryAgent(llm),        # callable async object
    )

def get_agents(request: Request) -> Agents:
    return request.app.state.agents