"""Shared types used by several agents."""
from enum import Enum
from pydantic import BaseModel, Field
 
 
class Money(BaseModel):
    amount: float = Field(ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3, description="ISO 4217 code")
 
 
class Priority(str, Enum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    BEST_VALUE = "best_value"
    COMFORT = "comfort"
