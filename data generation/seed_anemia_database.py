"""
Database Seeding Script for Iron Deficiency Anemia Data
Populates Supabase PostgreSQL with generated anemia data
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.core.config import get_settings
from app.models import (
    Patient, Lab, FamilyRelationship, FamilyDiseaseHistory,
    DoctorVisit, Symptom, Diagnosis, Prescription, LabReport, LabTestResult,
    DiseaseProgression, MLPrediction
)

def convert_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string timestamps and dates back to proper objects"""
    from datetime import date
    
    timestamp_fields = [
        'created_at', 'updated_at', 'visit_date', 'diagnosis_date', 
        'report_date', 'assessed_date', 'diagnosed_at'
    ]
    
    date_fields = [
        'date_of_birth'
    ]
    
    for field in timestamp_fields:
        if field in data and isinstance(data[field], str):
            try:
                # Handle different timestamp formats
                if 'T' in data[field]:
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                else:
                    data[field] = datetime.fromisoformat(data[field])
            except ValueError:
                # If conversion fails, keep as string
                pass
    
    for field in date_fields:
        if field in data and isinstance(data[field], str):
            try:
                data[field] = date.fromisoformat(data[field])
            except ValueError:
                # If conversion fails, keep as string
                pass
    
    return data

async def seed_anemia_database():
    """Seed the database with anemia data"""
    settings = get_settings()
    
    # Use DATABASE_URL directly from .env
    engine = create_async_engine(
        settings.DATABASE_URL, 
        echo=True,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0
        }
    )
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        async with async_session() as session:
            print("\n" + "="*60)
            print("SEEDING IRON DEFICIENCY ANEMIA DATA")
            print("="*60)
            
            # Load generated anemia data
            data_files = [
                "generated_anemia_patients.json", "generated_anemia_doctors.json", "generated_anemia_labs.json",
                "generated_anemia_family_relationships.json", "generated_anemia_family_disease_history.json",
                "generated_anemia_doctor_visits.json", "generated_anemia_symptoms.json", "generated_anemia_diagnoses.json",
                "generated_anemia_prescriptions.json", "generated_anemia_lab_reports.json", 
                "generated_anemia_lab_test_results.json", "generated_anemia_disease_progressions.json"
            ]
            
            # Get the directory where this script is located
            script_dir = Path(__file__).resolve().parent
            
            data = {}
            for file_name in data_files:
                try:
                    # Look for file in the same directory as the script
                    file_path = script_dir / file_name
                    with open(file_path, 'r') as f:
                        table_name = file_name.replace("generated_anemia_", "").replace(".json", "")
                        data[table_name] = json.load(f)
                        print(f"✅ Loaded {len(data[table_name])} records from {file_name}")
                except FileNotFoundError:
                    print(f"⚠️  Warning: {file_name} not found, skipping...")
                    continue
            
            # Seed in order (respecting foreign key constraints)
            
            # 1. Seed Doctors (as Patients with doctor fields)
            # Note: Doctors must be created as patients first with all patient fields
            # For now, skip if doctors data doesn't have patient fields (CNIC, date_of_birth, etc.)
            if "doctors" in data:
                print("\n⚠️  Skipping anemia doctors seeding - doctors should be created as patients first")
                print("   Use the /doctors POST endpoint to convert existing patients to doctors")
            
            # 2. Seed Labs
            if "labs" in data:
                print("\n🔬 Seeding anemia labs...")
                for lab_data in data["labs"]:
                    lab_data = convert_timestamps(lab_data)
                    lab = Lab(**lab_data)
                    session.add(lab)
                await session.commit()
                print(f"✅ Seeded {len(data['labs'])} anemia labs")
            
            # 3. Seed Patients
            if "patients" in data:
                print("\n👥 Seeding anemia patients...")
                from app.models.patient import GenderEnum
                batch_size = 100
                for i in range(0, len(data["patients"]), batch_size):
                    batch = data["patients"][i:i+batch_size]
                    for patient_data in batch:
                        patient_data = convert_timestamps(patient_data)
                        # Normalize gender to lowercase and convert to enum (enum expects lowercase)
                        if "gender" in patient_data and isinstance(patient_data["gender"], str):
                            gender_str = patient_data["gender"].lower()
                            try:
                                patient_data["gender"] = GenderEnum(gender_str)
                            except ValueError:
                                # If invalid, default to "other"
                                patient_data["gender"] = GenderEnum.OTHER
                        patient = Patient(**patient_data)
                        session.add(patient)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['patients']))}/{len(data['patients'])} patients")
                print(f"✅ Seeded {len(data['patients'])} anemia patients")
            
            # 4. Seed Family Relationships
            if "family_relationships" in data:
                print("\n👨‍👩‍👧‍👦 Seeding anemia family relationships...")
                for rel_data in data["family_relationships"]:
                    rel_data = convert_timestamps(rel_data)
                    relationship = FamilyRelationship(**rel_data)
                    session.add(relationship)
                await session.commit()
                print(f"✅ Seeded {len(data['family_relationships'])} anemia family relationships")
            
            # 5. Seed Family Disease History
            if "family_disease_history" in data:
                print("\n🧬 Seeding anemia family disease history...")
                for disease_data in data["family_disease_history"]:
                    disease_data = convert_timestamps(disease_data)
                    disease = FamilyDiseaseHistory(**disease_data)
                    session.add(disease)
                await session.commit()
                print(f"✅ Seeded {len(data['family_disease_history'])} anemia family disease records")
            
            # 6. Seed Doctor Visits
            if "doctor_visits" in data:
                print("\n📋 Seeding anemia doctor visits...")
                batch_size = 1000
                for i in range(0, len(data["doctor_visits"]), batch_size):
                    batch = data["doctor_visits"][i:i+batch_size]
                    for visit_data in batch:
                        visit_data = convert_timestamps(visit_data)
                        # Map doctor_id to doctor_patient_id for new schema
                        if "doctor_id" in visit_data:
                            visit_data["doctor_patient_id"] = visit_data.pop("doctor_id")
                        visit = DoctorVisit(**visit_data)
                        session.add(visit)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['doctor_visits']))}/{len(data['doctor_visits'])} visits")
                print(f"✅ Seeded {len(data['doctor_visits'])} anemia doctor visits")
            
            # 7. Seed Symptoms
            if "symptoms" in data:
                print("\n🤒 Seeding anemia symptoms...")
                batch_size = 1000
                for i in range(0, len(data["symptoms"]), batch_size):
                    batch = data["symptoms"][i:i+batch_size]
                    for symptom_data in batch:
                        symptom_data = convert_timestamps(symptom_data)
                        symptom = Symptom(**symptom_data)
                        session.add(symptom)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['symptoms']))}/{len(data['symptoms'])} symptoms")
                print(f"✅ Seeded {len(data['symptoms'])} anemia symptoms")
            
            # 8. Seed Diagnoses
            if "diagnoses" in data:
                print("\n🩺 Seeding anemia diagnoses...")
                batch_size = 1000
                for i in range(0, len(data["diagnoses"]), batch_size):
                    batch = data["diagnoses"][i:i+batch_size]
                    for diagnosis_data in batch:
                        diagnosis_data = convert_timestamps(diagnosis_data)
                        diagnosis = Diagnosis(**diagnosis_data)
                        session.add(diagnosis)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['diagnoses']))}/{len(data['diagnoses'])} diagnoses")
                print(f"✅ Seeded {len(data['diagnoses'])} anemia diagnoses")
            
            # 9. Seed Prescriptions
            if "prescriptions" in data:
                print("\n💊 Seeding anemia prescriptions...")
                batch_size = 1000
                for i in range(0, len(data["prescriptions"]), batch_size):
                    batch = data["prescriptions"][i:i+batch_size]
                    for prescription_data in batch:
                        prescription_data = convert_timestamps(prescription_data)
                        prescription = Prescription(**prescription_data)
                        session.add(prescription)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['prescriptions']))}/{len(data['prescriptions'])} prescriptions")
                print(f"✅ Seeded {len(data['prescriptions'])} anemia prescriptions")
            
            # 10. Seed Lab Reports
            if "lab_reports" in data:
                print("\n📊 Seeding anemia lab reports...")
                batch_size = 1000
                for i in range(0, len(data["lab_reports"]), batch_size):
                    batch = data["lab_reports"][i:i+batch_size]
                    for report_data in batch:
                        report_data = convert_timestamps(report_data)
                        report = LabReport(**report_data)
                        session.add(report)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['lab_reports']))}/{len(data['lab_reports'])} reports")
                print(f"✅ Seeded {len(data['lab_reports'])} anemia lab reports")
            
            # 11. Seed Lab Test Results
            if "lab_test_results" in data:
                print("\n🧪 Seeding anemia lab test results...")
                batch_size = 5000
                for i in range(0, len(data["lab_test_results"]), batch_size):
                    batch = data["lab_test_results"][i:i+batch_size]
                    for result_data in batch:
                        result_data = convert_timestamps(result_data)
                        result = LabTestResult(**result_data)
                        session.add(result)
                    await session.commit()
                    print(f"   Seeded {min(i+batch_size, len(data['lab_test_results']))}/{len(data['lab_test_results'])} results")
                print(f"✅ Seeded {len(data['lab_test_results'])} anemia lab test results")
            
            # 12. Seed Disease Progressions
            if "disease_progressions" in data:
                print("\n📈 Seeding anemia disease progressions...")
                for progression_data in data["disease_progressions"]:
                    progression_data = convert_timestamps(progression_data)
                    progression = DiseaseProgression(**progression_data)
                    session.add(progression)
                await session.commit()
                print(f"✅ Seeded {len(data['disease_progressions'])} anemia disease progressions")
            
            print("\n" + "="*60)
            print("✅ ANEMIA DATABASE SEEDING COMPLETED SUCCESSFULLY!")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ Error during seeding: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed LifeChain AI database with anemia data")
    parser.add_argument("--seed", action="store_true", help="Seed database with anemia data")
    
    args = parser.parse_args()
    
    if args.seed:
        print("Seeding anemia database...")
        asyncio.run(seed_anemia_database())
    else:
        print("Use --seed to seed database with anemia data")
        print("Example: python seed_anemia_database.py --seed")

