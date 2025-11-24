from fastapi import APIRouter

from app.api.v1.endpoints import (
    health, patients, doctors, visits, labs, 
    unified_inference, progression_report
)

api_router = APIRouter()

# Health check
api_router.include_router(health.router, tags=["health"])

# Patient management
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])

# Doctor management
api_router.include_router(doctors.router, prefix="/doctors", tags=["doctors"])

# Visit management
api_router.include_router(visits.router, prefix="/visits", tags=["visits"])

# Lab management
api_router.include_router(labs.router, prefix="/labs", tags=["labs"])

# ML Inference - Unified (supports all diseases)
api_router.include_router(unified_inference.router, prefix="/ml", tags=["ml-inference"])

# Progression Reports
api_router.include_router(progression_report.router, prefix="/reports", tags=["progression-reports"])


