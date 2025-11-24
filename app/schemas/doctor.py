"""
Pydantic schemas for Doctor-related API endpoints
Note: Doctors are patients with additional fields. A doctor must first exist as a patient.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from uuid import UUID
from app.schemas.patient import Patient


class DoctorCreate(BaseModel):
    """Create a doctor by adding doctor-specific fields to an existing patient"""
    patient_id: UUID = Field(..., description="ID of the existing patient to convert to doctor")
    specialization: str = Field(..., min_length=1, max_length=100)
    license_number: str = Field(..., min_length=1, max_length=50)
    hospital_affiliation: Optional[str] = Field(None, min_length=1, max_length=200)


class DoctorUpdate(BaseModel):
    """Update doctor-specific fields"""
    specialization: Optional[str] = Field(None, min_length=1, max_length=100)
    license_number: Optional[str] = Field(None, min_length=1, max_length=50)
    hospital_affiliation: Optional[str] = Field(None, min_length=1, max_length=200)


class Doctor(Patient):
    """Doctor schema extends Patient with doctor-specific fields"""
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    hospital_affiliation: Optional[str] = None
    is_doctor: bool = True

    class Config:
        from_attributes = True
