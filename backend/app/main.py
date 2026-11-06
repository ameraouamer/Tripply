# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import build_agents
from app.api.routes import trips


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agents = build_agents()   # built once, crashes early if env is missing
    yield


app = FastAPI(title="Tripply", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # your frontend dev origin
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(trips.router, prefix="/api")