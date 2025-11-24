"""
Inference Service for ML Model Predictions
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from uuid import UUID

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import the LSTM model class before importing ModelInference
from train_models import ProgressionBiLSTM
from model_inference import ModelInference
from app.models import Patient, DoctorVisit, LabTestResult, LabReport

class InferenceService:
    """Service for ML model inference operations"""
    
    def __init__(self):
        self.inference = ModelInference()
        self._models_loaded = False
    
    def load_models(self):
        """Load ML models (called lazily on first use)"""
        if not self._models_loaded:
            self.inference.load_models()
            self._models_loaded = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return self.inference.get_model_info()
    
    def predict_diagnosis(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict diabetes diagnosis from single visit data"""
        return self.inference.predict_diagnosis(patient_data)
    
    def predict_progression(self, patient_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """Predict diabetes progression from patient visit sequence"""
        return self.inference.predict_progression(patient_sequence)
    
    async def get_patient_latest_lab_data(self, patient_id: UUID, db: AsyncSession) -> Optional[Dict[str, float]]:
        """Get latest lab test results for a patient"""
        try:
            # Query to get latest lab test results for a patient using SQLAlchemy
            from sqlalchemy import select
            
            query = select(
                LabTestResult.test_name,
                LabTestResult.test_value
            ).join(
                LabReport, LabTestResult.report_id == LabReport.report_id
            ).join(
                DoctorVisit, LabReport.visit_id == DoctorVisit.visit_id
            ).join(
                Patient, DoctorVisit.patient_id == Patient.patient_id
            ).where(
                Patient.patient_id == patient_id
            ).where(
                LabTestResult.test_name.in_([
                    'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
                    'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
                ])
            ).order_by(
                DoctorVisit.visit_date.desc()
            ).limit(10)
            
            result = await db.execute(query)
            rows = result.all()
            
            if not rows:
                return None
            
            # Convert to dict
            lab_data = {}
            for row in rows:
                lab_data[row[0]] = float(row[1])
            
            # Ensure all required features are present
            required_features = [
                'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
                'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
            ]
            
            for feature in required_features:
                if feature not in lab_data:
                    lab_data[feature] = 0.0  # Default value for missing features
            
            return lab_data
            
        except Exception as e:
            print(f"Error getting patient lab data: {e}")
            return None
    
    async def get_patient_visit_sequence(self, patient_id: UUID, db: AsyncSession) -> Optional[List[Dict[str, float]]]:
        """Get visit sequence for a patient for progression prediction"""
        try:
            # Query to get visit sequence with lab test results using SQLAlchemy
            from sqlalchemy import select
            
            query = select(
                DoctorVisit.visit_date,
                LabTestResult.test_name,
                LabTestResult.test_value
            ).join(
                LabReport, DoctorVisit.visit_id == LabReport.visit_id
            ).join(
                LabTestResult, LabReport.report_id == LabTestResult.report_id
            ).join(
                Patient, DoctorVisit.patient_id == Patient.patient_id
            ).where(
                Patient.patient_id == patient_id
            ).where(
                LabTestResult.test_name.in_([
                    'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
                    'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
                ])
            ).order_by(
                DoctorVisit.visit_date.asc()
            )
            
            result = await db.execute(query)
            rows = result.all()
            
            if not rows:
                return None
            
            # Group by visit date
            visits = {}
            for row in rows:
                visit_date = row[0]
                test_name = row[1]
                test_value = float(row[2])
                
                if visit_date not in visits:
                    visits[visit_date] = {}
                visits[visit_date][test_name] = test_value
            
            # Convert to list of visit data
            visit_sequence = []
            for visit_date in sorted(visits.keys()):
                visit_data = visits[visit_date]
                
                # Ensure all required features are present
                required_features = [
                    'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
                    'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
                ]
                
                complete_visit = {}
                for feature in required_features:
                    complete_visit[feature] = visit_data.get(feature, 0.0)
                
                visit_sequence.append(complete_visit)
            
            return visit_sequence if visit_sequence else None
            
        except Exception as e:
            print(f"Error getting patient visit sequence: {e}")
            return None
