"""
ML Inference API endpoints
"""

from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_db
from app.schemas.inference import (
    DiagnosisRequest, DiagnosisResponse,
    ProgressionRequest, ProgressionResponse,
    BatchDiagnosisRequest, BatchDiagnosisResponse,
    ModelInfo, ErrorResponse
)
from app.services.inference_service import InferenceService

router = APIRouter()

# Initialize inference service (models will be loaded on first use)
inference_service = None

def get_inference_service():
    """Get or create the inference service instance"""
    global inference_service
    if inference_service is None:
        inference_service = InferenceService()
        inference_service.load_models()
    return inference_service

@router.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded ML models"""
    try:
        service = get_inference_service()
        info = service.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/diagnosis/predict", response_model=DiagnosisResponse)
async def predict_diagnosis(request: DiagnosisRequest):
    """Predict diabetes diagnosis from lab test results"""
    try:
        service = get_inference_service()
        # Convert request to dict
        test_data = request.dict()
        
        # Make prediction
        result = service.predict_diagnosis(test_data)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return DiagnosisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/progression/predict", response_model=ProgressionResponse)
async def predict_progression(request: ProgressionRequest):
    """Predict diabetes progression from visit sequence"""
    try:
        service = get_inference_service()
        # Make prediction
        result = service.predict_progression(request.visit_sequence)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return ProgressionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/diagnosis/batch", response_model=BatchDiagnosisResponse)
async def batch_diagnosis_predict(request: BatchDiagnosisRequest):
    """Predict diabetes diagnosis for multiple lab test results"""
    try:
        service = get_inference_service()
        predictions = []
        
        for lab_data in request.lab_test_results:
            result = service.predict_diagnosis(lab_data)
            
            if 'error' in result:
                raise HTTPException(status_code=400, detail=f"Prediction failed: {result['error']}")
            
            predictions.append(DiagnosisResponse(**result))
        
        return BatchDiagnosisResponse(
            patient_id=request.patient_id,
            predictions=predictions,
            batch_timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/diagnosis/patient/{patient_id}")
async def predict_patient_diagnosis(
    patient_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Predict diagnosis for a specific patient using their latest lab results"""
    try:
        service = get_inference_service()
        # Get patient's latest lab results
        lab_data = await service.get_patient_latest_lab_data(patient_id, db)
        
        if not lab_data:
            raise HTTPException(status_code=404, detail="No lab data found for patient")
        
        # Make prediction
        result = service.predict_diagnosis(lab_data)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return DiagnosisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient diagnosis prediction failed: {str(e)}")

@router.post("/progression/patient/{patient_id}")
async def predict_patient_progression(
    patient_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Predict progression for a specific patient using their visit history"""
    try:
        service = get_inference_service()
        # Get patient's visit sequence
        visit_sequence = await service.get_patient_visit_sequence(patient_id, db)
        
        if not visit_sequence:
            raise HTTPException(status_code=404, detail="No visit data found for patient")
        
        # Make prediction
        result = service.predict_progression(visit_sequence)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return ProgressionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient progression prediction failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for inference service"""
    try:
        service = get_inference_service()
        info = service.get_model_info()
        models_loaded = info.get('models_loaded', {})
        
        if not any(models_loaded.values()):
            raise HTTPException(status_code=503, detail="No models loaded")
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
