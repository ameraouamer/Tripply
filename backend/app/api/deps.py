import os
from fastapi import Request
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agents.planner import Agents
from app.agents.flight_agent import FlightsAgent
from app.agents.accommodation_agent import HotelsAgent
from app.agents.activity_agent import ActivitiesAgent
from app.agents.itinerary_agent import ItineraryAgent


def get_model(
    model: str | None = None,
    temperature: float = 0,
) -> ChatGoogleGenerativeAI:
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


# Wiring the agents together. This is the single place where we instantiate the agents and their dependencies.
def build_agents() -> Agents:
    llm = get_model()
    return Agents(
        flights=FlightsAgent().search,        # sync -> planner's _run uses a worker thread
        hotels=HotelsAgent().asearch,         # async wrapper already exists
        activities=ActivitiesAgent(llm=llm).search,  # async
        itinerary=ItineraryAgent(llm),        # callable async object
    )


def get_agents(request: Request) -> Agents:
    return request.app.state.agents
