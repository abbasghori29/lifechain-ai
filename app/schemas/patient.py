"""
Pydantic schemas for Patient-related API endpoints
"""

from datetime import datetime, date
from typing import Optional, List, TYPE_CHECKING
from pydantic import BaseModel, Field
from uuid import UUID

if TYPE_CHECKING:
    from app.schemas.lab import LabReport
    from app.schemas.visit import DoctorVisit
    from app.schemas.disease import DiseaseProgression


class PatientBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    cnic: str = Field(..., pattern=r'^\d{5}-\d{7}-\d{1}$')
    date_of_birth: date
    gender: str = Field(..., pattern='^(male|female|other)$')
    blood_group: str = Field(..., pattern='^(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)$')
    phone: str = Field(..., min_length=10, max_length=15)
    email: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = Field(None, max_length=500)


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    cnic: Optional[str] = Field(None, pattern=r'^\d{5}-\d{7}-\d{1}$')
    date_of_birth: Optional[date] = None
    gender: Optional[str] = Field(None, pattern='^(male|female|other)$')
    blood_group: Optional[str] = Field(None, pattern='^(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)$')
    phone: Optional[str] = Field(None, min_length=10, max_length=15)
    email: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = Field(None, max_length=500)


class Patient(PatientBase):
    patient_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatientWithRelations(Patient):
    family_relationships: List['FamilyRelationship'] = []
    lab_reports: List['LabReport'] = []
    doctor_visits: List['DoctorVisit'] = []
    disease_progressions: List['DiseaseProgression'] = []

    class Config:
        from_attributes = True


class FamilyRelationshipBase(BaseModel):
    relative_patient_id: UUID
    relationship_type: str = Field(..., pattern='^(parent|child|sibling|spouse|grandparent|grandchild|aunt_uncle|niece_nephew|cousin)$')
    is_blood_relative: bool = True


class FamilyRelationshipCreate(FamilyRelationshipBase):
    pass


class FamilyRelationshipAutoCreate(BaseModel):
    relative_patient_id: UUID
    relationship_type: str = Field(..., pattern='^(parent|child|sibling|spouse|grandparent|grandchild|aunt_uncle|niece_nephew|cousin)$')
    is_blood_relative: bool = True
    max_depth: int = Field(10, ge=1, le=20, description="Maximum depth for automatic inference")
    auto_infer: bool = Field(True, description="Enable automatic relationship inference")


class FamilyRelationship(FamilyRelationshipBase):
    id: UUID
    patient_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class FamilyDiseaseHistoryBase(BaseModel):
    relative_patient_id: UUID
    disease_name: str = Field(..., min_length=1, max_length=100)
    relationship_type: str = Field(..., pattern='^(Parent|Child|Sibling|Spouse|Grandparent|Grandchild|Uncle|Aunt|Cousin)$')
    age_of_onset: Optional[int] = Field(None, ge=0, le=120)
    is_genetic: bool = False


class FamilyDiseaseHistoryCreate(FamilyDiseaseHistoryBase):
    pass


class FamilyDiseaseHistory(FamilyDiseaseHistoryBase):
    history_id: UUID
    patient_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Forward references are handled by string annotations
