from fastapi import APIRouter
from app.services.health_service import get_health_status


router = APIRouter()


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    return get_health_status()


