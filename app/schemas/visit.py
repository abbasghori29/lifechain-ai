"""
Pydantic schemas for Visit-related API endpoints
"""

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, Field
from uuid import UUID

if TYPE_CHECKING:
    from app.schemas.patient import Patient
    from app.schemas.doctor import Doctor
    from app.schemas.lab import LabReport


class DoctorVisitBase(BaseModel):
    patient_id: UUID
    doctor_patient_id: UUID = Field(..., description="ID of the patient who is the doctor")
    visit_date: datetime
    visit_type: str = Field(..., pattern='^(consultation|follow_up|routine_checkup|lab_review|emergency)$')
    chief_complaint: Optional[str] = Field(None, max_length=1000)
    doctor_notes: Optional[str] = Field(None, max_length=5000)
    vital_signs: Optional[dict] = Field(None, description="Vital signs measurements (temperature, blood pressure, heart rate, etc.)")


class DoctorVisitCreate(DoctorVisitBase):
    pass


class DoctorVisitUpdate(BaseModel):
    visit_date: Optional[datetime] = None
    visit_type: Optional[str] = Field(None, pattern='^(consultation|follow_up|routine_checkup|lab_review|emergency)$')
    chief_complaint: Optional[str] = Field(None, max_length=1000)
    doctor_notes: Optional[str] = Field(None, max_length=5000)
    vital_signs: Optional[dict] = Field(None, description="Vital signs measurements (temperature, blood pressure, heart rate, etc.)")


class DoctorVisit(DoctorVisitBase):
    visit_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DoctorVisitWithRelations(DoctorVisit):
    patient: Optional['Patient'] = None
    doctor_patient: Optional['Patient'] = None
    symptoms: List['Symptom'] = []
    diagnoses: List['Diagnosis'] = []
    prescriptions: List['Prescription'] = []
    lab_reports: List['LabReport'] = []

    class Config:
        from_attributes = True


class SymptomBase(BaseModel):
    symptom_name: str = Field(..., min_length=1, max_length=200)
    severity: Optional[int] = Field(None, ge=1, le=10, description="Severity on a scale of 1-10")
    duration_days: Optional[int] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=500)


class SymptomCreate(SymptomBase):
    pass


class Symptom(SymptomBase):
    id: UUID
    visit_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class DiagnosisBase(BaseModel):
    disease_name: str = Field(..., min_length=1, max_length=200)
    diagnosis_date: datetime
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score between 0 and 1")
    ml_model_used: Optional[str] = Field(None, max_length=100)
    status: str = Field(..., pattern='^(suspected|confirmed)$')
    notes: Optional[str] = Field(None, max_length=2000)


class DiagnosisCreate(DiagnosisBase):
    pass


class Diagnosis(DiagnosisBase):
    diagnosis_id: UUID
    visit_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class PrescriptionBase(BaseModel):
    medication_name: str = Field(..., min_length=1, max_length=200)
    dosage: str = Field(..., min_length=1, max_length=100)
    frequency: str = Field(..., min_length=1, max_length=100)
    duration_days: Optional[int] = Field(None, ge=1)
    instructions: Optional[str] = Field(None, max_length=1000)


class PrescriptionCreate(PrescriptionBase):
    pass


class Prescription(PrescriptionBase):
    prescription_id: UUID
    visit_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Forward references are handled by string annotations
