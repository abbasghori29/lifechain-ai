"""
Visit CRUD API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from uuid import UUID
from datetime import datetime, date

from app.db.session import get_db
from app.models import DoctorVisit, Symptom, Diagnosis, Prescription, Patient
from app.schemas.visit import (
    DoctorVisit as DoctorVisitSchema,
    DoctorVisitCreate,
    DoctorVisitUpdate,
    Symptom as SymptomSchema,
    SymptomCreate,
    Diagnosis as DiagnosisSchema,
    DiagnosisCreate,
    Prescription as PrescriptionSchema,
    PrescriptionCreate
)
from app.api.v1.dependencies import get_translation_language, apply_translation

router = APIRouter()

@router.post("/", response_model=DoctorVisitSchema)
async def create_visit(
    visit: DoctorVisitCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new doctor visit"""
    try:
        # Verify patient exists
        patient_result = await db.execute(
            select(Patient).where(Patient.patient_id == visit.patient_id)
        )
        if not patient_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Verify doctor (patient with is_doctor=True) exists
        doctor_result = await db.execute(
            select(Patient).where(
                Patient.patient_id == visit.doctor_patient_id,
                Patient.is_doctor == True
            )
        )
        if not doctor_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Doctor not found")
        
        # Create visit data and handle timezone conversion
        visit_data = visit.dict()
        # Map doctor_patient_id to the model field
        if 'doctor_patient_id' in visit_data:
            visit_data['doctor_patient_id'] = visit_data['doctor_patient_id']
        
        # Convert timezone-aware datetime to timezone-naive for database storage
        if visit_data.get('visit_date') and visit_data['visit_date'].tzinfo is not None:
            visit_data['visit_date'] = visit_data['visit_date'].replace(tzinfo=None)
        
        db_visit = DoctorVisit(**visit_data)
        db.add(db_visit)
        await db.commit()
        await db.refresh(db_visit)
        
        return db_visit
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create visit: {str(e)}")

@router.get("/", response_model=List[DoctorVisitSchema])
async def get_visits(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    patient_id: Optional[UUID] = Query(None, description="Filter by patient ID"),
    doctor_patient_id: Optional[UUID] = Query(None, description="Filter by doctor patient ID"),
    visit_type: Optional[str] = Query(None, description="Filter by visit type"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date"),
    lang: str = Depends(get_translation_language),
    db: AsyncSession = Depends(get_db)
):
    """Get list of visits with filters"""
    try:
        query = select(DoctorVisit)
        
        if patient_id:
            query = query.where(DoctorVisit.patient_id == patient_id)
        
        if doctor_patient_id:
            query = query.where(DoctorVisit.doctor_patient_id == doctor_patient_id)
        
        if visit_type:
            query = query.where(DoctorVisit.visit_type == visit_type)
        
        if start_date:
            query = query.where(DoctorVisit.visit_date >= start_date)
        
        if end_date:
            query = query.where(DoctorVisit.visit_date <= end_date)
        
        query = query.order_by(DoctorVisit.visit_date.desc())
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        visits = result.scalars().all()
        
        # Apply translation if needed
        translated_visits = await apply_translation(visits, "visit", lang)
        return translated_visits
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get visits: {str(e)}")

@router.get("/{visit_id}", response_model=DoctorVisitSchema)
async def get_visit(
    visit_id: UUID,
    lang: str = Depends(get_translation_language),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific visit by ID"""
    try:
        result = await db.execute(
            select(DoctorVisit).where(DoctorVisit.visit_id == visit_id)
        )
        visit = result.scalar_one_or_none()
        
        if not visit:
            raise HTTPException(status_code=404, detail="Visit not found")
        
        # Apply translation if needed
        translated_visit = await apply_translation(visit, "visit", lang)
        return translated_visit
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get visit: {str(e)}")

@router.put("/{visit_id}", response_model=DoctorVisitSchema)
async def update_visit(
    visit_id: UUID,
    visit_update: DoctorVisitUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a visit"""
    try:
        result = await db.execute(
            select(DoctorVisit).where(DoctorVisit.visit_id == visit_id)
        )
        db_visit = result.scalar_one_or_none()
        
        if not db_visit:
            raise HTTPException(status_code=404, detail="Visit not found")
        
        # Update fields
        update_data = visit_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_visit, field, value)
        
        await db.commit()
        await db.refresh(db_visit)
        
        return db_visit
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update visit: {str(e)}")

@router.delete("/{visit_id}")
async def delete_visit(
    visit_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a visit"""
    try:
        result = await db.execute(
            select(DoctorVisit).where(DoctorVisit.visit_id == visit_id)
        )
        db_visit = result.scalar_one_or_none()
        
        if not db_visit:
            raise HTTPException(status_code=404, detail="Visit not found")
        
        await db.delete(db_visit)
        await db.commit()
        
        return {"message": "Visit deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete visit: {str(e)}")

# Symptom endpoints
@router.post("/{visit_id}/symptoms", response_model=SymptomSchema)
async def create_symptom(
    visit_id: UUID,
    symptom: SymptomCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add a symptom to a visit"""
    try:
        # Verify visit exists
        visit_result = await db.execute(
            select(DoctorVisit).where(DoctorVisit.visit_id == visit_id)
        )
        if not visit_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Visit not found")
        
        # Create symptom
        db_symptom = Symptom(visit_id=visit_id, **symptom.dict())
        db.add(db_symptom)
        await db.commit()
        await db.refresh(db_symptom)
        
        return db_symptom
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create symptom: {str(e)}")

@router.get("/{visit_id}/symptoms", response_model=List[SymptomSchema])
async def get_visit_symptoms(
    visit_id: UUID,
    lang: str = Depends(get_translation_language),
    db: AsyncSession = Depends(get_db)
):
    """Get symptoms for a visit"""
    try:
        result = await db.execute(
            select(Symptom).where(Symptom.visit_id == visit_id)
        )
        symptoms = result.scalars().all()
        
        # Apply translation if needed
        translated_symptoms = await apply_translation(symptoms, "symptom", lang)
        return translated_symptoms
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get symptoms: {str(e)}")

# Diagnosis endpoints
@router.post("/{visit_id}/diagnoses", response_model=DiagnosisSchema)
async def create_diagnosis(
    visit_id: UUID,
    diagnosis: DiagnosisCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add a diagnosis to a visit"""
    try:
        # Verify visit exists
        visit_result = await db.execute(
            select(DoctorVisit).where(DoctorVisit.visit_id == visit_id)
        )
        if not visit_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Visit not found")
        
        # Create diagnosis
        db_diagnosis = Diagnosis(visit_id=visit_id, **diagnosis.dict())
        db.add(db_diagnosis)
        await db.commit()
        await db.refresh(db_diagnosis)
        
        return db_diagnosis
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create diagnosis: {str(e)}")

@router.get("/{visit_id}/diagnoses", response_model=List[DiagnosisSchema])
async def get_visit_diagnoses(
    visit_id: UUID,
    lang: str = Depends(get_translation_language),
    db: AsyncSession = Depends(get_db)
):
    """Get diagnoses for a visit"""
    try:
        result = await db.execute(
            select(Diagnosis).where(Diagnosis.visit_id == visit_id)
        )
        diagnoses = result.scalars().all()
        
        # Apply translation if needed
        translated_diagnoses = await apply_translation(diagnoses, "diagnosis", lang)
        return translated_diagnoses
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get diagnoses: {str(e)}")

# Prescription endpoints
@router.post("/{visit_id}/prescriptions", response_model=PrescriptionSchema)
async def create_prescription(
    visit_id: UUID,
    prescription: PrescriptionCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add a prescription to a visit"""
    try:
        # Verify visit exists
        visit_result = await db.execute(
            select(DoctorVisit).where(DoctorVisit.visit_id == visit_id)
        )
        if not visit_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Visit not found")
        
        # Create prescription
        db_prescription = Prescription(visit_id=visit_id, **prescription.dict())
        db.add(db_prescription)
        await db.commit()
        await db.refresh(db_prescription)
        
        return db_prescription
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create prescription: {str(e)}")

@router.get("/{visit_id}/prescriptions", response_model=List[PrescriptionSchema])
async def get_visit_prescriptions(
    visit_id: UUID,
    lang: str = Depends(get_translation_language),
    db: AsyncSession = Depends(get_db)
):
    """Get prescriptions for a visit"""
    try:
        result = await db.execute(
            select(Prescription).where(Prescription.visit_id == visit_id)
        )
        prescriptions = result.scalars().all()
        
        # Apply translation if needed
        translated_prescriptions = await apply_translation(prescriptions, "prescription", lang)
        return translated_prescriptions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prescriptions: {str(e)}")
