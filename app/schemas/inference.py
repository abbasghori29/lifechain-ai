"""
Pydantic schemas for ML Inference API endpoints
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class DiagnosisRequest(BaseModel):
    fasting_glucose: float = Field(..., ge=0, le=1000, description="Fasting glucose level (mg/dL)")
    hba1c: float = Field(..., ge=0, le=20, description="HbA1c level (%)")
    hdl: float = Field(..., ge=0, le=200, description="HDL cholesterol (mg/dL)")
    ldl: float = Field(..., ge=0, le=500, description="LDL cholesterol (mg/dL)")
    triglycerides: float = Field(..., ge=0, le=2000, description="Triglycerides (mg/dL)")
    total_cholesterol: float = Field(..., ge=0, le=500, description="Total cholesterol (mg/dL)")
    creatinine: float = Field(..., ge=0, le=10, description="Creatinine level (mg/dL)")
    bmi: float = Field(..., ge=10, le=100, description="Body Mass Index")
    systolic_bp: float = Field(..., ge=50, le=300, description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=30, le=200, description="Diastolic blood pressure (mmHg)")


class DiagnosisResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted diagnosis class")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for all classes")
    prediction_confidence: float = Field(..., ge=0, le=1, description="Overall prediction confidence")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class ProgressionRequest(BaseModel):
    visit_sequence: List[Dict[str, float]] = Field(..., min_items=1, max_items=50, description="Sequence of visit data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "visit_sequence": [
                    {
                        "fasting_glucose": 100.0,
                        "hba1c": 5.5,
                        "hdl": 50.0,
                        "ldl": 100.0,
                        "triglycerides": 120.0,
                        "total_cholesterol": 180.0,
                        "creatinine": 0.9,
                        "bmi": 25.0,
                        "systolic_bp": 120.0,
                        "diastolic_bp": 80.0
                    },
                    {
                        "fasting_glucose": 110.0,
                        "hba1c": 5.8,
                        "hdl": 48.0,
                        "ldl": 110.0,
                        "triglycerides": 130.0,
                        "total_cholesterol": 190.0,
                        "creatinine": 0.95,
                        "bmi": 26.0,
                        "systolic_bp": 125.0,
                        "diastolic_bp": 82.0
                    }
                ]
            }
        }


class ProgressionResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted progression class")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for all classes")
    prediction_confidence: float = Field(..., ge=0, le=1, description="Overall prediction confidence")
    model_used: str = Field(..., description="Model used for prediction")
    sequence_length: int = Field(..., description="Length of input sequence")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchDiagnosisRequest(BaseModel):
    patient_id: UUID = Field(..., description="Patient ID")
    lab_test_results: List[Dict[str, float]] = Field(..., min_items=1, description="List of lab test results")


class BatchDiagnosisResponse(BaseModel):
    patient_id: UUID
    predictions: List[DiagnosisResponse]
    batch_timestamp: datetime


class ModelInfo(BaseModel):
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    device: str = Field(..., description="Device used for inference")
    models_directory: str = Field(..., description="Directory containing models")
    diagnosis_features: Optional[List[str]] = Field(None, description="Features used by diagnosis model")
    progression_features: Optional[List[str]] = Field(None, description="Features used by progression model")
    progression_classes: Optional[List[str]] = Field(None, description="Classes predicted by progression model")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
