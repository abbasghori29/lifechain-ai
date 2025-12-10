"""
Create sample disease progression timeline data for testing charts
Inserts multiple progression records over time for one patient
"""
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.disease import DiseaseProgression
from app.models.patient import Patient
import uuid
import random

async def create_ckd_timeline():
    """Create a realistic CKD progression timeline for one patient"""
    
    async with AsyncSessionLocal() as db:
        # Find a patient with CKD
        query = select(DiseaseProgression.patient_id).where(
            DiseaseProgression.disease_name.ilike('%chronic%kidney%')
        ).limit(1)
        result = await db.execute(query)
        row = result.first()
        
        if not row:
            print("❌ No CKD patients found. Looking for any patient...")
            query = select(Patient.patient_id).limit(1)
            result = await db.execute(query)
            row = result.first()
        
        if not row:
            print("❌ No patients found in database!")
            return
        
        patient_id = row[0]
        print(f"📋 Creating timeline for patient: {patient_id}")
        
        # CKD progression stages over 2 years (realistic timeline)
        # Stage progression: Normal → Stage 1 → Stage 2 → Stage 3a → Stage 3b → Stage 4 → Stage 5/ESRD
        ckd_timeline = [
            # Year 1 - Early detection and monitoring
            {"stage": "Stage 1", "days_ago": 730, "confidence": 0.85, "notes": "Initial screening - mild kidney damage detected"},
            {"stage": "Stage 1", "days_ago": 640, "confidence": 0.88, "notes": "Follow-up - stable condition"},
            {"stage": "Stage 2", "days_ago": 550, "confidence": 0.82, "notes": "Slight decline in eGFR noted"},
            {"stage": "Stage 2", "days_ago": 460, "confidence": 0.86, "notes": "Diet modification recommended"},
            
            # Year 2 - Progression
            {"stage": "Stage 3a", "days_ago": 365, "confidence": 0.91, "notes": "Moderate decrease in kidney function"},
            {"stage": "Stage 3a", "days_ago": 275, "confidence": 0.89, "notes": "Started ACE inhibitor therapy"},
            {"stage": "Stage 3b", "days_ago": 180, "confidence": 0.93, "notes": "Further decline despite treatment"},
            {"stage": "Stage 3b", "days_ago": 120, "confidence": 0.90, "notes": "Referral to nephrologist"},
            {"stage": "Stage 4", "days_ago": 60, "confidence": 0.94, "notes": "Severe kidney function loss - dialysis planning"},
            {"stage": "Stage 4", "days_ago": 30, "confidence": 0.92, "notes": "Pre-dialysis education completed"},
            {"stage": "ESRD", "days_ago": 7, "confidence": 0.96, "notes": "End-stage renal disease - dialysis initiated"},
        ]
        
        base_date = datetime.now()
        created_count = 0
        
        for entry in ckd_timeline:
            assessed_date = base_date - timedelta(days=entry["days_ago"])
            
            # Check if this exact record already exists
            check_query = select(DiseaseProgression).where(
                DiseaseProgression.patient_id == patient_id,
                DiseaseProgression.disease_name == "chronic_kidney_disease",
                DiseaseProgression.assessed_date == assessed_date
            )
            existing = await db.execute(check_query)
            if existing.first():
                continue
            
            progression = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=patient_id,
                disease_name="chronic_kidney_disease",
                progression_stage=entry["stage"],
                assessed_date=assessed_date,
                ml_model_used="ckd_progression_lstm",
                confidence_score=entry["confidence"],
                notes=entry["notes"],
                created_at=datetime.now()
            )
            db.add(progression)
            created_count += 1
        
        await db.commit()
        
        print(f"\n✅ Created {created_count} CKD progression records for patient {patient_id}")
        print(f"\n📊 Timeline spans from {(base_date - timedelta(days=730)).strftime('%Y-%m-%d')} to {base_date.strftime('%Y-%m-%d')}")
        
        # Show what was created
        verify_query = select(
            DiseaseProgression.assessed_date,
            DiseaseProgression.progression_stage,
            DiseaseProgression.confidence_score
        ).where(
            DiseaseProgression.patient_id == patient_id,
            DiseaseProgression.disease_name == "chronic_kidney_disease"
        ).order_by(DiseaseProgression.assessed_date.asc())
        
        verify_result = await db.execute(verify_query)
        records = verify_result.all()
        
        print(f"\n📈 Complete timeline ({len(records)} records):")
        for record in records:
            print(f"   {record[0].strftime('%Y-%m-%d')} | {record[1]:15} | Confidence: {record[2]:.0%}")
        
        print(f"\n📌 Test with this curl command:")
        print(f"""
curl -X 'GET' \\
  'http://localhost:8080/api/v1/reports/patient/{patient_id}/progression-timeline?disease_name=ckd&months_back=36' \\
  -H 'accept: application/json'
""")
        
        return patient_id

if __name__ == "__main__":
    asyncio.run(create_ckd_timeline())

