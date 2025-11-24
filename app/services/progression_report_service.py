"""
Progression Report Service for generating comprehensive patient reports
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func, and_, or_
from uuid import UUID
import os

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Import the LSTM model class before importing InferenceService
from train_models import ProgressionBiLSTM
from app.services.inference_service import InferenceService
from app.models import (
    Patient, DoctorVisit, LabTestResult, LabReport, 
    DiseaseProgression, FamilyDiseaseHistory, FamilyRelationship
)

# Gemini LLM for intelligent recommendations
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from app.core.config import get_settings

# Get settings and set Google API key from .env
settings = get_settings()
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

class ProgressionReportService:
    """Service for generating comprehensive progression reports"""
    
    def __init__(self):
        self.inference = InferenceService()
        self.inference.load_models()
    
    async def generate_progression_report(
        self, 
        patient_id: UUID, 
        patient_name: str, 
        disease_name: str,
        months_back: int = 12,
        db: AsyncSession = None
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive progression report for a patient"""
        try:
            # Get progression timeline
            timeline = await self.get_progression_timeline(patient_id, disease_name, months_back, db)
            
            if not timeline:
                return None
            
            # Get current progression stage
            current_stage = timeline[-1]['progression_stage'] if timeline else 'Unknown'
            
            # Get risk factors
            risk_factors = await self.get_risk_factors(patient_id, disease_name, db)
            
            # Get recommendations
            recommendations = await self.get_recommendations(patient_id, disease_name, db)
            
            # Predict future progression
            future_prediction = await self.predict_future_progression(patient_id, disease_name, 6, db)
            
            # Generate report
            report = {
                'patient_id': patient_id,
                'patient_name': patient_name,
                'disease_name': disease_name,
                'progression_timeline': timeline,
                'current_stage': current_stage,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'predicted_progression': future_prediction.get('predicted_stage', 'Unknown') if future_prediction else 'Unknown',
                'confidence_score': future_prediction.get('confidence_score', 0.0) if future_prediction else 0.0,
                'generated_at': datetime.now()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating progression report: {e}")
            return None
    
    async def get_progression_timeline(
        self, 
        patient_id: UUID, 
        disease_name: str, 
        months_back: int,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get progression timeline for a patient"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30)
            
            # Query progression data using SQLAlchemy
            from sqlalchemy.orm import aliased
            
            query = select(
                DiseaseProgression.assessed_date,
                DiseaseProgression.progression_stage,
                DiseaseProgression.notes,
                DoctorVisit.visit_date,
                DoctorVisit.visit_type,
                DoctorVisit.doctor_notes
            ).outerjoin(
                DoctorVisit,
                and_(
                    DiseaseProgression.patient_id == DoctorVisit.patient_id,
                    func.date(DiseaseProgression.assessed_date) == func.date(DoctorVisit.visit_date)
                )
            ).where(
                and_(
                    DiseaseProgression.patient_id == patient_id,
                    DiseaseProgression.disease_name.ilike(disease_name),
                    DiseaseProgression.assessed_date >= start_date
                )
            ).order_by(DiseaseProgression.assessed_date.asc())
            
            result = await db.execute(query)
            rows = result.all()
            
            timeline = []
            for row in rows:
                timeline.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'progression_stage': row[1],
                    'severity_score': None,  # column not present in DB
                    'notes': row[2],
                    'visit_date': row[3].isoformat() if row[3] else None,
                    'visit_type': row[4],
                    'doctor_notes': row[5]
                })
            
            return timeline
            
        except Exception as e:
            print(f"Error getting progression timeline: {e}")
            return []
    
    async def get_risk_factors(
        self, 
        patient_id: UUID, 
        disease_name: str,
        db: AsyncSession
    ) -> List[str]:
        """Get risk factors for a patient"""
        try:
            risk_factors = []
            
            # Get family history using SQLAlchemy
            family_query = select(
                FamilyDiseaseHistory.disease_name,
                FamilyDiseaseHistory.diagnosed_at,
                FamilyDiseaseHistory.severity,
                FamilyDiseaseHistory.notes
            ).where(
                and_(
                    FamilyDiseaseHistory.patient_id == patient_id,
                    FamilyDiseaseHistory.disease_name.ilike(disease_name)
                )
            ).distinct()
            
            family_result = await db.execute(family_query)
            family_rows = family_result.all()
            
            for row in family_rows:
                diagnosed_at = row[1]
                severity = row[2]
                if diagnosed_at is not None and severity is not None:
                    risk_factors.append(f"Family history present ({disease_name}, severity {severity})")
                else:
                    risk_factors.append(f"Family history present ({disease_name})")
            
            # Get recent lab values for risk assessment using SQLAlchemy
            recent_date = datetime.now() - timedelta(days=90)
            
            lab_query = select(
                LabTestResult.test_name,
                LabTestResult.test_value,
                LabTestResult.reference_range_min,
                LabTestResult.reference_range_max,
                LabTestResult.is_abnormal
            ).join(
                LabReport, LabTestResult.report_id == LabReport.report_id
            ).join(
                DoctorVisit, LabReport.visit_id == DoctorVisit.visit_id
            ).join(
                Patient, DoctorVisit.patient_id == Patient.patient_id
            ).where(
                and_(
                    Patient.patient_id == patient_id,
                    LabTestResult.test_name.in_(['fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides', 'bmi']),
                    DoctorVisit.visit_date >= recent_date
                )
            ).order_by(
                DoctorVisit.visit_date.desc()
            ).limit(20)
            
            lab_result = await db.execute(lab_query)
            lab_rows = lab_result.all()
            
            # Analyze lab values for risk factors
            lab_values = {}
            for row in lab_rows:
                test_name = row[0]
                test_value = float(row[1])
                is_abnormal = row[4]
                
                if test_name not in lab_values:
                    lab_values[test_name] = []
                lab_values[test_name].append(test_value)
            
            # Check for abnormal values
            for test_name, values in lab_values.items():
                avg_value = sum(values) / len(values)
                
                if test_name == 'fasting_glucose' and avg_value > 126:
                    risk_factors.append(f"Elevated fasting glucose: {avg_value:.1f} mg/dL")
                elif test_name == 'hba1c' and avg_value > 6.5:
                    risk_factors.append(f"Elevated HbA1c: {avg_value:.1f}%")
                elif test_name == 'hdl' and avg_value < 40:
                    risk_factors.append(f"Low HDL cholesterol: {avg_value:.1f} mg/dL")
                elif test_name == 'ldl' and avg_value > 160:
                    risk_factors.append(f"High LDL cholesterol: {avg_value:.1f} mg/dL")
                elif test_name == 'bmi' and avg_value > 30:
                    risk_factors.append(f"Obesity: BMI {avg_value:.1f}")
            
            return risk_factors
            
        except Exception as e:
            print(f"Error getting risk factors: {e}")
            return []
    
    async def get_recommendations(
        self, 
        patient_id: UUID, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get AI-powered personalized recommendations for a patient using Gemini LLM"""
        try:
            # Get patient info
            patient_query = select(Patient).where(Patient.patient_id == patient_id)
            patient_result = await db.execute(patient_query)
            patient = patient_result.scalar_one_or_none()
            
            if not patient:
                return {"error": "Patient not found"}
            
            # Find ALL diseases/conditions the patient has from disease_progressions
            diseases_query = select(DiseaseProgression.disease_name).where(
                DiseaseProgression.patient_id == patient_id
            ).distinct()
            diseases_result = await db.execute(diseases_query)
            all_diseases = [row[0] for row in diseases_result.all()]
            
            # Get progression data for all diseases (last 3 months)
            recent_date = datetime.now() - timedelta(days=90)
            all_progressions_query = select(DiseaseProgression).where(
                and_(
                    DiseaseProgression.patient_id == patient_id,
                    DiseaseProgression.assessed_date >= recent_date
                )
            ).order_by(DiseaseProgression.assessed_date.desc())
            
            progressions_result = await db.execute(all_progressions_query)
            all_progressions = progressions_result.scalars().all()
            
            # Group progressions by disease
            progressions_by_disease = {}
            for prog in all_progressions:
                if prog.disease_name not in progressions_by_disease:
                    progressions_by_disease[prog.disease_name] = []
                progressions_by_disease[prog.disease_name].append({
                    'date': prog.assessed_date.isoformat(),
                    'stage': prog.progression_stage,
                    'notes': prog.notes,
                    'confidence': prog.confidence_score
                })
            
            # Get most recent lab test results
            lab_query = select(
                LabTestResult.test_name,
                LabTestResult.test_value,
                LabTestResult.unit,
                LabTestResult.reference_range_min,
                LabTestResult.reference_range_max,
                LabTestResult.is_abnormal,
                DoctorVisit.visit_date
            ).join(
                LabReport, LabTestResult.report_id == LabReport.report_id
            ).join(
                DoctorVisit, LabReport.visit_id == DoctorVisit.visit_id
            ).where(
                DoctorVisit.patient_id == patient_id
            ).where(
                DoctorVisit.visit_date >= recent_date
            ).order_by(
                DoctorVisit.visit_date.desc()
            )
            
            lab_result = await db.execute(lab_query)
            lab_tests = lab_result.all()
            
            # Get most recent test results (very last visit)
            latest_tests = {}
            if lab_tests:
                latest_visit_date = lab_tests[0][6]
                for test in lab_tests:
                    if test[6] == latest_visit_date:
                        latest_tests[test[0]] = {
                            'value': test[1],
                            'unit': test[2],
                            'normal_min': test[3],
                            'normal_max': test[4],
                            'is_abnormal': test[5]
                        }
            
            # Get future progression predictions for each disease
            future_predictions = {}
            for disease in all_diseases:
                try:
                    prediction = await self.predict_future_progression(
                        patient_id=patient_id,
                        disease_name=disease,
                        months_ahead=6,
                        db=db
                    )
                    if prediction:
                        future_predictions[disease] = prediction
                except Exception as e:
                    print(f"Could not predict progression for {disease}: {e}")
            
            # Determine scenario
            has_recent_data = len(all_progressions) > 0
            has_lab_tests = len(lab_tests) > 0
            
            # Get current stages for all diseases
            current_stages = {}
            for disease, progs in progressions_by_disease.items():
                if progs:
                    current_stages[disease] = progs[0]['stage']
            
            # Build context for Gemini
            context = self._build_comprehensive_context(
                patient=patient,
                all_diseases=all_diseases,
                progressions_by_disease=progressions_by_disease,
                current_stages=current_stages,
                latest_tests=latest_tests,
                future_predictions=future_predictions,
                has_recent_data=has_recent_data,
                has_lab_tests=has_lab_tests
            )
            
            # Generate recommendations using Gemini
            recommendations = self._generate_recommendations_with_gemini(context)
            
            return {
                "patient_id": str(patient_id),
                "patient_name": f"{patient.first_name} {patient.last_name}",
                "conditions_identified": all_diseases,
                "current_stages": current_stages,
                "future_predictions": future_predictions,
                "has_recent_data": has_recent_data,
                "has_lab_tests": has_lab_tests,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "recommendations": [
                    "Unable to generate personalized recommendations at this time",
                    "Please consult with your healthcare provider"
                ]
            }
    
    def _build_comprehensive_context(
        self,
        patient: Patient,
        all_diseases: List[str],
        progressions_by_disease: Dict[str, List[Dict]],
        current_stages: Dict[str, str],
        latest_tests: Dict,
        future_predictions: Dict,
        has_recent_data: bool,
        has_lab_tests: bool
    ) -> str:
        """Build comprehensive context string for Gemini LLM"""
        
        age = (datetime.now().date() - patient.date_of_birth).days // 365
        
        context = f"""Patient Information:
- Age: {age} years
- Gender: {patient.gender}
- Patient ID: {patient.patient_id}

"""
        
        # List all conditions
        if all_diseases:
            context += f"📋 Active Conditions ({len(all_diseases)}):\n"
            for disease in all_diseases:
                stage = current_stages.get(disease, 'Unknown')
                context += f"  • {disease.title()} - Current Stage: {stage}\n"
            context += "\n"
        else:
            context += "📋 No active conditions found in system.\n\n"
        
        # Progression history by disease
        if has_recent_data and progressions_by_disease:
            context += "📊 Progression History (Last 3 Months):\n"
            for disease, progressions in progressions_by_disease.items():
                context += f"\n  {disease.title()}:\n"
                for prog in progressions[:5]:  # Last 5 entries
                    context += f"    - {prog['date'][:10]}: {prog['stage']}"
                    if prog['confidence']:
                        context += f" (Confidence: {prog['confidence']:.2%})"
                    if prog['notes']:
                        context += f" - {prog['notes'][:100]}"
                    context += "\n"
            context += "\n"
        else:
            context += "⚠️ No progression data found in the last 3 months.\n\n"
        
        # Latest lab test results
        if has_lab_tests and latest_tests:
            context += "🔬 Most Recent Lab Test Results:\n"
            abnormal_tests = []
            normal_tests = []
            
            for test_name, test_data in latest_tests.items():
                if test_data['is_abnormal']:
                    abnormal_tests.append(f"  ⚠️ {test_name}: {test_data['value']} {test_data['unit']} (Normal: {test_data['normal_min']}-{test_data['normal_max']})")
                else:
                    normal_tests.append(f"  ✓ {test_name}: {test_data['value']} {test_data['unit']}")
            
            if abnormal_tests:
                context += "\n  ABNORMAL VALUES:\n" + "\n".join(abnormal_tests) + "\n"
            if normal_tests:
                context += "\n  NORMAL VALUES:\n" + "\n".join(normal_tests) + "\n"
            context += "\n"
        else:
            context += "⚠️ No lab test results found in the last 3 months.\n\n"
        
        # Future predictions
        if future_predictions:
            context += "🔮 AI-Predicted Future Progression (Next 6 Months):\n"
            for disease, prediction in future_predictions.items():
                predicted_stage = prediction.get('predicted_stage', 'Unknown')
                confidence = prediction.get('confidence_score', 0)
                context += f"  • {disease.title()}: Predicted to be '{predicted_stage}' (Confidence: {confidence:.2%})\n"
            context += "\n"
        
        # Treatment adherence warning
        if not has_recent_data and not has_lab_tests:
            context += "🚨 CRITICAL ALERT: Patient appears to have discontinued treatment (no activity in 3 months).\n"
            context += "   This is a HIGH RISK situation requiring immediate medical attention.\n\n"
        
        # Check for cured conditions
        cured_conditions = [disease for disease, stage in current_stages.items() if stage.lower() in ['cured', 'normal']]
        if cured_conditions:
            context += f"✅ Recovered/Controlled Conditions: {', '.join(cured_conditions)}\n\n"
        
        return context
    
    def _generate_recommendations_with_gemini(self, context: str) -> List[str]:
        """Generate recommendations using Gemini LLM"""
        try:
            # Define structured output
            class Recommendations(BaseModel):
                """Medical recommendations for a patient"""
                recommendations: List[str] = Field(
                    ..., 
                    description="List of specific, actionable medical recommendations (5-8 items)"
                )
            
            # Initialize Gemini
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.3,
                max_retries=2
            )
            
            structured_llm = llm.with_structured_output(Recommendations, method="json_schema")
            
            # Create prompt
            prompt = f"""You are a medical AI assistant providing personalized healthcare recommendations.

{context}

Based on the above patient information, provide 5-8 specific, actionable recommendations:

RULES:
1. If patient has abnormal test results, prioritize recommendations to address those specific issues
2. If patient shows "Cured" status, provide maintenance and prevention recommendations
3. If patient has no recent data (discontinued treatment), strongly emphasize importance of:
   - Not abandoning treatment
   - Scheduling immediate medical evaluation
   - Potential risks of treatment interruption
4. Be specific about lab values that need attention
5. Include lifestyle, medication, and monitoring recommendations
6. Use clear, patient-friendly language
7. Each recommendation should be 1-2 sentences

Provide recommendations as a numbered list."""

            # Generate recommendations
            result = structured_llm.invoke(prompt)
            
            return result.recommendations
            
        except Exception as e:
            print(f"Error generating recommendations with Gemini: {e}")
            # Fallback recommendations
            return [
                "Schedule a follow-up appointment with your healthcare provider",
                "Maintain regular health monitoring and checkups",
                "Follow prescribed treatment plans consistently",
                "Adopt healthy lifestyle habits including balanced diet and regular exercise",
                "Keep track of your symptoms and report any changes to your doctor"
            ]
    
    async def get_risk_assessment(
        self, 
        patient_id: UUID, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get risk assessment for a patient based on ancestors' diseases (genetic risk)"""
        try:
            # Get only ancestor blood relatives (not descendants)
            # Genetic risk comes from: parents, grandparents, siblings, aunts/uncles, cousins
            # NOT from: children, grandchildren, nieces/nephews
            from app.models.family import RelationshipTypeEnum
            
            ancestor_relationships = [
                RelationshipTypeEnum.PARENT,
                RelationshipTypeEnum.GRANDPARENT,
                RelationshipTypeEnum.SIBLING,
                RelationshipTypeEnum.AUNT_UNCLE,
                RelationshipTypeEnum.COUSIN
            ]
            
            blood_relatives_query = select(FamilyRelationship).where(
                and_(
                    FamilyRelationship.patient_id == patient_id,
                    FamilyRelationship.is_blood_relative == True,
                    FamilyRelationship.relationship_type.in_(ancestor_relationships)
                )
            )
            blood_relatives_result = await db.execute(blood_relatives_query)
            blood_relatives = blood_relatives_result.scalars().all()
            
            if not blood_relatives:
                return {
                    'status': 'negative',
                    'message': 'No ancestors or blood relatives found',
                    'ancestors_count': 0,
                    'diseases_found': [],
                    'assessment_date': datetime.now().isoformat()
                }
            
            # Get relative patient IDs
            relative_ids = [rel.relative_patient_id for rel in blood_relatives]
            
            # Search for diseases in blood relatives from FamilyDiseaseHistory
            diseases_query = select(FamilyDiseaseHistory).where(
                FamilyDiseaseHistory.patient_id.in_(relative_ids)
            )
            diseases_result = await db.execute(diseases_query)
            diseases = diseases_result.scalars().all()
            
            # Also check DiseaseProgression table for blood relatives
            progression_query = select(DiseaseProgression).where(
                DiseaseProgression.patient_id.in_(relative_ids)
            )
            progression_result = await db.execute(progression_query)
            progressions = progression_result.scalars().all()
            
            # Collect disease information
            diseases_found = []
            relative_disease_map = {}
            
            # From FamilyDiseaseHistory
            for disease in diseases:
                relative_id = str(disease.patient_id)
                if relative_id not in relative_disease_map:
                    relative_disease_map[relative_id] = []
                
                disease_info = {
                    'disease_name': disease.disease_name,
                    'diagnosed_at': disease.diagnosed_at.isoformat() if disease.diagnosed_at else None,
                    'severity': disease.severity.value if disease.severity else None,
                    'source': 'family_history'
                }
                relative_disease_map[relative_id].append(disease_info)
                diseases_found.append(disease.disease_name)
            
            # From DiseaseProgression
            for progression in progressions:
                relative_id = str(progression.patient_id)
                if relative_id not in relative_disease_map:
                    relative_disease_map[relative_id] = []
                
                disease_info = {
                    'disease_name': progression.disease_name,
                    'assessed_date': progression.assessed_date.isoformat() if progression.assessed_date else None,
                    'progression_stage': progression.progression_stage,
                    'source': 'disease_progression'
                }
                relative_disease_map[relative_id].append(disease_info)
                diseases_found.append(progression.disease_name)
            
            # Get relative details
            relatives_with_diseases = []
            for relative_id, diseases_list in relative_disease_map.items():
                # Get relative info
                relative_patient_query = select(Patient).where(Patient.patient_id == UUID(relative_id))
                relative_patient_result = await db.execute(relative_patient_query)
                relative_patient = relative_patient_result.scalar_one_or_none()
                
                # Get relationship type
                relationship = next((rel for rel in blood_relatives if str(rel.relative_patient_id) == relative_id), None)
                
                if relative_patient:
                    relatives_with_diseases.append({
                        'relative_id': relative_id,
                        'relative_name': f"{relative_patient.first_name} {relative_patient.last_name}",
                        'relationship_type': relationship.relationship_type.value if relationship else 'unknown',
                        'diseases': diseases_list
                    })
            
            # Determine status
            status = 'positive' if diseases_found else 'negative'
            unique_diseases = list(set(diseases_found))
            
            return {
                'status': status,
                'message': f'Found {len(unique_diseases)} disease(s) in {len(relatives_with_diseases)} ancestor(s)/blood relative(s)' if status == 'positive' else 'No diseases found in ancestors or blood relatives',
                'ancestors_count': len(blood_relatives),
                'ancestors_with_diseases_count': len(relatives_with_diseases),
                'unique_diseases': unique_diseases,
                'total_disease_records': len(diseases_found),
                'relatives_with_diseases': relatives_with_diseases,
                'assessment_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting risk assessment: {e}")
            return {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'assessment_date': datetime.now().isoformat()
            }
    
    async def get_family_disease_history(
        self, 
        patient_id: UUID, 
        disease_name: str,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get family disease history for a patient"""
        try:
            # Get family disease history records using SQLAlchemy
            # Note: FamilyDiseaseHistory stores disease for the patient, not relatives
            # We need to get family relationships and check their disease history
            
            # Get blood relatives
            relatives_query = select(FamilyRelationship).where(
                and_(
                    FamilyRelationship.patient_id == patient_id,
                    FamilyRelationship.is_blood_relative == True
                )
            )
            relatives_result = await db.execute(relatives_query)
            relatives = relatives_result.scalars().all()
            
            if not relatives:
                return []
            
            relative_ids = [rel.relative_patient_id for rel in relatives]
            
            # Get disease history for blood relatives
            disease_query = select(
                FamilyDiseaseHistory,
                Patient.first_name,
                Patient.last_name
            ).join(
                Patient, FamilyDiseaseHistory.patient_id == Patient.patient_id
            ).where(
                and_(
                    FamilyDiseaseHistory.patient_id.in_(relative_ids),
                    FamilyDiseaseHistory.disease_name.ilike(disease_name)
                )
            )
            
            disease_result = await db.execute(disease_query)
            disease_rows = disease_result.all()
            
            family_history = []
            for row in disease_rows:
                disease_record = row[0]
                first_name = row[1]
                last_name = row[2]
                
                # Find relationship type
                relationship_type = 'unknown'
                for rel in relatives:
                    if rel.relative_patient_id == disease_record.patient_id:
                        relationship_type = rel.relationship_type.value
                        break
                
                family_history.append({
                    'relative_name': f"{first_name} {last_name}",
                    'relationship': relationship_type,
                    'diagnosed_at': disease_record.diagnosed_at.isoformat() if disease_record.diagnosed_at else None,
                    'severity': disease_record.severity.value if disease_record.severity else None
                })
            
            return family_history
            
        except Exception as e:
            print(f"Error getting family disease history: {e}")
            return []
    
    async def predict_future_progression(
        self, 
        patient_id: UUID, 
        disease_name: str, 
        months_ahead: int,
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Predict future progression for a specific disease"""
        try:
            # Get patient's visit sequence
            visit_sequence = await self.inference.get_patient_visit_sequence(patient_id, db)
            
            if not visit_sequence:
                return None
            
            # Make prediction using ML model
            prediction_result = self.inference.predict_progression(visit_sequence)
            
            if 'error' in prediction_result:
                return None
            
            return {
                'predicted_stage': prediction_result['predicted_class'],
                'confidence_score': prediction_result['prediction_confidence'],
                'prediction_date': datetime.now().isoformat(),
                'months_ahead': months_ahead
            }
            
        except Exception as e:
            print(f"Error predicting future progression: {e}")
            return None
    
    async def predict_all_conditions_progression(
        self,
        patient_id: UUID,
        months_ahead: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Predict future progression for all patient conditions automatically"""
        try:
            # Get patient info
            patient_query = select(Patient).where(Patient.patient_id == patient_id)
            patient_result = await db.execute(patient_query)
            patient = patient_result.scalar_one_or_none()
            
            if not patient:
                return {"error": "Patient not found"}
            
            # Find ALL diseases/conditions the patient has
            diseases_query = select(DiseaseProgression.disease_name).where(
                DiseaseProgression.patient_id == patient_id
            ).distinct()
            diseases_result = await db.execute(diseases_query)
            all_diseases = [row[0] for row in diseases_result.all()]
            
            if not all_diseases:
                return {
                    "error": "No disease history found for patient",
                    "patient_id": str(patient_id),
                    "patient_name": f"{patient.first_name} {patient.last_name}"
                }
            
            # Get current progression status for each disease
            current_status = {}
            recent_date = datetime.now() - timedelta(days=90)
            
            for disease in all_diseases:
                status_query = select(DiseaseProgression).where(
                    and_(
                        DiseaseProgression.patient_id == patient_id,
                        DiseaseProgression.disease_name == disease
                    )
                ).order_by(DiseaseProgression.assessed_date.desc()).limit(1)
                
                status_result = await db.execute(status_query)
                latest_status = status_result.scalar_one_or_none()
                
                if latest_status:
                    current_status[disease] = {
                        'current_stage': latest_status.progression_stage,
                        'assessed_date': latest_status.assessed_date.isoformat(),
                        'confidence': latest_status.confidence_score
                    }
            
            # Get lab test results for predictions
            lab_query = select(
                LabTestResult.test_name,
                LabTestResult.test_value,
                LabTestResult.is_abnormal,
                DoctorVisit.visit_date
            ).join(
                LabReport, LabTestResult.report_id == LabReport.report_id
            ).join(
                DoctorVisit, LabReport.visit_id == DoctorVisit.visit_id
            ).where(
                DoctorVisit.patient_id == patient_id
            ).where(
                DoctorVisit.visit_date >= recent_date
            ).order_by(
                DoctorVisit.visit_date.desc()
            ).limit(50)
            
            lab_result = await db.execute(lab_query)
            lab_tests = lab_result.all()
            
            # Make predictions for each disease
            predictions = {}
            for disease in all_diseases:
                try:
                    # Get visit sequence for ML prediction
                    visit_sequence = await self.inference.get_patient_visit_sequence(patient_id, db)
                    
                    if visit_sequence and len(visit_sequence) >= 2:
                        # Make prediction using ML model
                        ml_prediction = self.inference.predict_progression(visit_sequence)
                        
                        if 'error' not in ml_prediction:
                            predictions[disease] = {
                                'disease_name': disease,
                                'current_stage': current_status.get(disease, {}).get('current_stage', 'Unknown'),
                                'predicted_stage': ml_prediction['predicted_class'],
                                'confidence_score': ml_prediction['prediction_confidence'],
                                'model_used': ml_prediction['model_used'],
                                'prediction_basis': 'ML model trained on visit sequence',
                                'months_ahead': months_ahead,
                                'risk_level': self._calculate_risk_level(ml_prediction['predicted_class'])
                            }
                        else:
                            # Fallback to rule-based prediction
                            predictions[disease] = self._rule_based_prediction(
                                disease, 
                                current_status.get(disease, {}),
                                lab_tests,
                                months_ahead
                            )
                    else:
                        # Not enough data for ML, use rule-based
                        predictions[disease] = self._rule_based_prediction(
                            disease,
                            current_status.get(disease, {}),
                            lab_tests,
                            months_ahead
                        )
                        
                except Exception as e:
                    print(f"Error predicting progression for {disease}: {e}")
                    predictions[disease] = {
                        'disease_name': disease,
                        'error': f'Unable to predict: {str(e)}',
                        'current_stage': current_status.get(disease, {}).get('current_stage', 'Unknown')
                    }
            
            # Calculate overall health trajectory
            trajectory = self._calculate_overall_trajectory(predictions)
            
            return {
                'patient_id': str(patient_id),
                'patient_name': f"{patient.first_name} {patient.last_name}",
                'conditions_analyzed': len(all_diseases),
                'diseases': all_diseases,
                'predictions': predictions,
                'overall_trajectory': trajectory,
                'months_ahead': months_ahead,
                'prediction_date': datetime.now().isoformat(),
                'data_quality': {
                    'has_lab_tests': len(lab_tests) > 0,
                    'lab_test_count': len(lab_tests),
                    'conditions_with_ml_prediction': len([p for p in predictions.values() if 'model_used' in p]),
                    'conditions_with_rule_based': len([p for p in predictions.values() if 'prediction_basis' in p and 'rule-based' in p['prediction_basis']])
                }
            }
            
        except Exception as e:
            print(f"Error in predict_all_conditions_progression: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'patient_id': str(patient_id)
            }
    
    def _rule_based_prediction(
        self,
        disease: str,
        current_status: Dict,
        lab_tests: List,
        months_ahead: int
    ) -> Dict[str, Any]:
        """Rule-based prediction when ML model can't be used"""
        current_stage = current_status.get('current_stage', 'Unknown')
        
        # Analyze lab trends
        abnormal_count = sum(1 for test in lab_tests if test[2])  # is_abnormal
        total_tests = len(lab_tests)
        abnormal_ratio = abnormal_count / total_tests if total_tests > 0 else 0
        
        # Predict based on current stage and lab results
        if current_stage.lower() in ['cured', 'normal']:
            if abnormal_ratio > 0.3:
                predicted = 'stable_with_monitoring'
                confidence = 0.65
            else:
                predicted = 'likely_stable'
                confidence = 0.75
        elif current_stage.lower() in ['controlled', 'mild']:
            if abnormal_ratio > 0.5:
                predicted = 'possible_worsening'
                confidence = 0.60
            else:
                predicted = 'likely_stable'
                confidence = 0.70
        else:  # Complicated, severe, etc.
            if abnormal_ratio > 0.6:
                predicted = 'high_risk_progression'
                confidence = 0.55
            else:
                predicted = 'requires_monitoring'
                confidence = 0.60
        
        return {
            'disease_name': disease,
            'current_stage': current_stage,
            'predicted_stage': predicted,
            'confidence_score': confidence,
            'prediction_basis': 'rule-based (insufficient visit data for ML)',
            'abnormal_test_ratio': abnormal_ratio,
            'months_ahead': months_ahead,
            'risk_level': self._calculate_risk_level(predicted)
        }
    
    def _calculate_risk_level(self, predicted_stage: str) -> str:
        """Calculate risk level from predicted stage"""
        stage_lower = predicted_stage.lower()
        
        if any(word in stage_lower for word in ['worsening', 'high_risk', 'progression', 'complicated']):
            return 'HIGH'
        elif any(word in stage_lower for word in ['possible', 'monitoring', 'moderate']):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_overall_trajectory(self, predictions: Dict) -> Dict[str, Any]:
        """Calculate overall health trajectory"""
        risk_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for pred in predictions.values():
            if 'risk_level' in pred:
                risk_counts[pred['risk_level']] += 1
        
        total = sum(risk_counts.values())
        
        if risk_counts['HIGH'] > 0:
            overall = 'CONCERNING'
            message = f"{risk_counts['HIGH']} condition(s) at high risk of worsening"
        elif risk_counts['MEDIUM'] > total / 2:
            overall = 'MODERATE'
            message = "Multiple conditions require close monitoring"
        else:
            overall = 'STABLE'
            message = "Conditions appear stable or improving"
        
        return {
            'status': overall,
            'message': message,
            'risk_distribution': risk_counts
        }
    
    async def get_lab_measurements_timeline(
        self,
        patient_id: UUID,
        test_name: Optional[str] = None,
        months_back: int = 12,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get lab test measurements over time for graphing"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_back * 30)
            
            # Query lab test results with report dates
            query = select(
                LabTestResult.test_name,
                LabTestResult.test_value,
                LabTestResult.unit,
                LabTestResult.reference_range_min,
                LabTestResult.reference_range_max,
                LabTestResult.is_abnormal,
                LabReport.report_date
            ).join(
                LabReport, LabTestResult.report_id == LabReport.report_id
            ).where(
                and_(
                    LabReport.patient_id == patient_id,
                    LabReport.report_date >= start_date
                )
            )
            
            # Filter by test name if provided
            if test_name:
                query = query.where(LabTestResult.test_name.ilike(f"%{test_name}%"))
            
            query = query.order_by(LabReport.report_date.asc(), LabTestResult.test_name.asc())
            
            result = await db.execute(query)
            rows = result.all()
            
            if not rows:
                return {
                    'patient_id': str(patient_id),
                    'test_name': test_name,
                    'months_back': months_back,
                    'measurements': {},
                    'available_tests': []
                }
            
            # Group measurements by test name
            measurements_by_test = {}
            available_tests = set()
            
            for row in rows:
                test_name_val = row[0]
                test_value = float(row[1])
                unit = row[2]
                ref_min = float(row[3]) if row[3] is not None else None
                ref_max = float(row[4]) if row[4] is not None else None
                is_abnormal = row[5]
                report_date = row[6]
                
                available_tests.add(test_name_val)
                
                if test_name_val not in measurements_by_test:
                    measurements_by_test[test_name_val] = {
                        'test_name': test_name_val,
                        'unit': unit,
                        'reference_range_min': ref_min,
                        'reference_range_max': ref_max,
                        'data_points': []
                    }
                
                measurements_by_test[test_name_val]['data_points'].append({
                    'date': report_date.isoformat() if report_date else None,
                    'value': test_value,
                    'is_abnormal': is_abnormal
                })
            
            return {
                'patient_id': str(patient_id),
                'test_name': test_name,
                'months_back': months_back,
                'measurements': measurements_by_test,
                'available_tests': sorted(list(available_tests))
            }
            
        except Exception as e:
            print(f"Error getting lab measurements timeline: {e}")
            import traceback
            traceback.print_exc()
            return {
                'patient_id': str(patient_id),
                'error': str(e),
                'measurements': {},
                'available_tests': []
            }