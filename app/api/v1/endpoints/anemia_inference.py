"""
Iron Deficiency Anemia Inference API Endpoints
Handles diagnosis and progression predictions for anemia patients
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
from uuid import UUID
from pydantic import BaseModel

from app.db.session import get_db
from app.services.multi_disease_inference import multi_disease_inference
from app.models import Patient, LabTestResult, LabReport, DoctorVisit

router = APIRouter()

# Simple request schemas
class AnemiaFeaturesRequest(BaseModel):
    features: Dict[str, float]

class AnemiaSequenceRequest(BaseModel):
    sequence: List[Dict[str, float]]

@router.post("/diagnosis", response_model=Dict[str, Any])
async def predict_anemia_diagnosis(
    request: AnemiaFeaturesRequest
):
    """
    Predict Iron Deficiency Anemia diagnosis from lab test results
    
    **Required Features:**
    - hemoglobin (g/dL)
    - hematocrit (%)
    - mcv (fL)
    - mch (pg)
    - mchc (g/dL)
    - rdw (%)
    - serum_iron (μg/dL)
    - ferritin (ng/mL)
    - tibc (μg/dL)
    - transferrin_saturation (%)
    - reticulocyte_count (%)
    - wbc (cells/μL)
    - platelet_count (cells/μL)
    - esr (mm/hr)
    - bmi (kg/m²)
    - systolic_bp (mmHg)
    - diastolic_bp (mmHg)
    
    **Returns:**
    - diagnosis: Normal | Iron Deficiency without Anemia | Mild/Moderate/Severe Iron Deficiency Anemia
    - confidence: 0-1 score
    - probabilities: Probability for each class
    """
    try:
        # Load models if not already loaded
        if not multi_disease_inference._models_loaded:
            multi_disease_inference.load_models()
        
        # Make prediction
        result = multi_disease_inference.predict_anemia_diagnosis(request.features)
        
        return {
            "disease": "iron_deficiency_anemia",
            "prediction_type": "diagnosis",
            **result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/progression", response_model=Dict[str, Any])
async def predict_anemia_progression(
    request: AnemiaSequenceRequest
):
    """
    Predict Iron Deficiency Anemia progression from patient visit sequence
    
    **Required:** List of visits with lab values (same features as diagnosis)
    
    **Returns:**
    - progression: Resolved | Improving | Stable | Worsening | Severe
    - confidence: 0-1 score
    - probabilities: Probability for each stage
    """
    try:
        # Load models if not already loaded
        if not multi_disease_inference._models_loaded:
            multi_disease_inference.load_models()
        
        # Make prediction
        result = multi_disease_inference.predict_anemia_progression(request.sequence)
        
        return {
            "disease": "iron_deficiency_anemia",
            "prediction_type": "progression",
            **result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/patient/{patient_id}/diagnosis")
async def predict_anemia_diagnosis_for_patient(
    patient_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict anemia diagnosis for a patient using their latest lab results
    """
    try:
        # Get latest lab test results for patient
        query = select(
            LabTestResult.test_name,
            LabTestResult.test_value
        ).join(
            LabReport, LabTestResult.report_id == LabReport.report_id
        ).join(
            DoctorVisit, LabReport.visit_id == DoctorVisit.visit_id
        ).where(
            DoctorVisit.patient_id == patient_id
        ).order_by(
            DoctorVisit.visit_date.desc()
        ).limit(20)
        
        result = await db.execute(query)
        rows = result.all()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No lab results found for patient")
        
        # Convert to dict
        lab_data = {}
        for row in rows:
            lab_data[row[0]] = float(row[1])
        
        # Required features for anemia
        required_features = [
            'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
            'serum_iron', 'ferritin', 'tibc', 'transferrin_saturation',
            'reticulocyte_count', 'wbc', 'platelet_count', 'esr',
            'bmi', 'systolic_bp', 'diastolic_bp'
        ]
        
        # Fill missing features with 0
        for feature in required_features:
            if feature not in lab_data:
                lab_data[feature] = 0.0
        
        # Load models if not already loaded
        if not multi_disease_inference._models_loaded:
            multi_disease_inference.load_models()
        
        # Make prediction
        prediction = multi_disease_inference.predict_anemia_diagnosis(lab_data)
        
        return {
            "patient_id": str(patient_id),
            "disease": "iron_deficiency_anemia",
            "prediction_type": "diagnosis",
            **prediction
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/patient/{patient_id}/progression")
async def predict_anemia_progression_for_patient(
    patient_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict anemia progression for a patient using their visit history
    """
    try:
        # Get visit sequence for patient
        query = select(
            DoctorVisit.visit_date,
            LabTestResult.test_name,
            LabTestResult.test_value
        ).join(
            LabReport, DoctorVisit.visit_id == LabReport.visit_id
        ).join(
            LabTestResult, LabReport.report_id == LabTestResult.report_id
        ).where(
            DoctorVisit.patient_id == patient_id
        ).order_by(
            DoctorVisit.visit_date.asc()
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No visit history found for patient")
        
        # Group by visit
        visits_data = {}
        for row in rows:
            visit_date = row[0]
            if visit_date not in visits_data:
                visits_data[visit_date] = {}
            visits_data[visit_date][row[1]] = float(row[2])
        
        # Required features
        required_features = [
            'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
            'serum_iron', 'ferritin', 'tibc', 'transferrin_saturation',
            'reticulocyte_count', 'bmi', 'systolic_bp', 'diastolic_bp'
        ]
        
        # Convert to sequence
        sequence = []
        for visit_date in sorted(visits_data.keys()):
            visit_features = {}
            for feature in required_features:
                visit_features[feature] = visits_data[visit_date].get(feature, 0.0)
            sequence.append(visit_features)
        
        if len(sequence) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient visit history (minimum 3 visits required)"
            )
        
        # Load models if not already loaded
        if not multi_disease_inference._models_loaded:
            multi_disease_inference.load_models()
        
        # Make prediction
        prediction = multi_disease_inference.predict_anemia_progression(sequence)
        
        return {
            "patient_id": str(patient_id),
            "disease": "iron_deficiency_anemia",
            "prediction_type": "progression",
            **prediction
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/model-info")
async def get_anemia_model_info():
    """Get information about anemia models"""
    try:
        if not multi_disease_inference._models_loaded:
            multi_disease_inference.load_models()
        
        info = multi_disease_inference.get_model_info()
        
        return {
            "disease": "iron_deficiency_anemia",
            "models": info['anemia'],
            "features": {
                "diagnosis": [
                    "hemoglobin", "hematocrit", "mcv", "mch", "mchc", "rdw",
                    "serum_iron", "ferritin", "tibc", "transferrin_saturation",
                    "reticulocyte_count", "wbc", "platelet_count", "esr",
                    "bmi", "systolic_bp", "diastolic_bp"
                ],
                "progression": [
                    "hemoglobin", "hematocrit", "mcv", "mch", "mchc", "rdw",
                    "serum_iron", "ferritin", "tibc", "transferrin_saturation",
                    "reticulocyte_count", "bmi", "systolic_bp", "diastolic_bp"
                ]
            },
            "diagnosis_classes": [
                "Normal",
                "Iron Deficiency without Anemia",
                "Mild Iron Deficiency Anemia",
                "Moderate Iron Deficiency Anemia",
                "Severe Iron Deficiency Anemia"
            ],
            "progression_classes": [
                "Resolved",
                "Improving",
                "Stable",
                "Worsening",
                "Severe"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

