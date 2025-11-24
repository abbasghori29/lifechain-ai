"""
Pydantic schemas for Disease-related API endpoints
"""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, Field
from uuid import UUID

if TYPE_CHECKING:
    pass


class DiseaseProgressionBase(BaseModel):
    disease_name: str = Field(..., min_length=1, max_length=200)
    progression_stage: str = Field(..., min_length=1, max_length=100)
    assessed_date: datetime
    severity_score: Optional[float] = Field(None, ge=0, le=10)
    notes: Optional[str] = Field(None, max_length=2000)


class DiseaseProgressionCreate(DiseaseProgressionBase):
    pass


class DiseaseProgressionUpdate(BaseModel):
    progression_stage: Optional[str] = Field(None, min_length=1, max_length=100)
    assessed_date: Optional[datetime] = None
    severity_score: Optional[float] = Field(None, ge=0, le=10)
    notes: Optional[str] = Field(None, max_length=2000)


class DiseaseProgression(DiseaseProgressionBase):
    progression_id: UUID
    patient_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class MLPredictionBase(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    prediction_type: str = Field(..., pattern='^(diagnosis|progression|risk_assessment)$')
    predicted_value: str = Field(..., min_length=1, max_length=200)
    confidence_score: float = Field(..., ge=0, le=1)
    input_features: dict
    model_version: str = Field(..., min_length=1, max_length=50)


class MLPredictionCreate(MLPredictionBase):
    pass


class MLPrediction(MLPredictionBase):
    prediction_id: UUID
    patient_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class ProgressionReport(BaseModel):
    patient_id: UUID
    patient_name: str
    disease_name: str
    progression_timeline: List[dict]
    current_stage: str
    risk_factors: List[str]
    recommendations: List[str]
    predicted_progression: str
    confidence_score: float
    generated_at: datetime

    class Config:
        from_attributes = True
