"""routes - API route registration."""

from .aerospace import router as aerospace_router
from .food import router as food_router
from .medical import router as medical_router

__all__ = ["aerospace_router", "food_router", "medical_router"]
