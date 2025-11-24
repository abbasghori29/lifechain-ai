"""
Database Seeding Script for LifeChain AI Healthcare System
Populates Supabase PostgreSQL with generated data
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

async def seed_database():
    """Seed the database with generated data"""
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
            print("Starting database seeding...")
            
            # Load generated data
            data_files = [
                "generated_patients.json", "generated_doctors.json", "generated_labs.json",
                "generated_family_relationships.json", "generated_family_disease_history.json",
                "generated_doctor_visits.json", "generated_symptoms.json", "generated_diagnoses.json",
                "generated_prescriptions.json", "generated_lab_reports.json", 
                "generated_lab_test_results.json", "generated_disease_progressions.json"
            ]
            
            # Get the directory where this script is located
            script_dir = Path(__file__).resolve().parent
            
            data = {}
            for file_name in data_files:
                try:
                    # Look for file in the same directory as the script
                    file_path = script_dir / file_name
                    with open(file_path, 'r') as f:
                        table_name = file_name.replace("generated_", "").replace(".json", "")
                        data[table_name] = json.load(f)
                        print(f"Loaded {len(data[table_name])} records from {file_name}")
                except FileNotFoundError:
                    print(f"Warning: {file_name} not found, skipping...")
                    continue
            
            # Seed in order (respecting foreign key constraints)
            
            # 1. Seed Doctors (as Patients with doctor fields)
            # Note: Doctors must be created as patients first with all patient fields
            # For now, skip if doctors data doesn't have patient fields (CNIC, date_of_birth, etc.)
            if "doctors" in data:
                print("⚠️  Skipping doctors seeding - doctors should be created as patients first")
                print("   Use the /doctors POST endpoint to convert existing patients to doctors")
            
            # 2. Seed Labs
            if "labs" in data:
                print("Seeding labs...")
                for lab_data in data["labs"]:
                    lab_data = convert_timestamps(lab_data)
                    lab = Lab(**lab_data)
                    session.add(lab)
                await session.commit()
                print(f"Seeded {len(data['labs'])} labs")
            
            # 3. Seed Patients
            if "patients" in data:
                print("Seeding patients...")
                from app.models.patient import GenderEnum
                for patient_data in data["patients"]:
                    patient_data = convert_timestamps(patient_data)
                    # Normalize gender to lowercase and convert to enum
                    if "gender" in patient_data and isinstance(patient_data["gender"], str):
                        gender_str = patient_data["gender"].lower()
                        try:
                            patient_data["gender"] = GenderEnum(gender_str)
                        except ValueError:
                            patient_data["gender"] = GenderEnum.OTHER
                    
                    # Create patient - SQLAlchemy should serialize enum.value
                    patient = Patient(**patient_data)
                    # Explicitly ensure gender is set as enum value for serialization
                    # This is a workaround for asyncpg enum serialization
                    if hasattr(patient.gender, 'value'):
                        # Force SQLAlchemy to use the enum value by accessing it
                        # The enum should serialize correctly, but we ensure it's the right type
                        pass
                    session.add(patient)
                await session.commit()
                print(f"Seeded {len(data['patients'])} patients")
            
            # 4. Seed Family Relationships
            if "family_relationships" in data:
                print("Seeding family relationships...")
                for rel_data in data["family_relationships"]:
                    rel_data = convert_timestamps(rel_data)
                    relationship = FamilyRelationship(**rel_data)
                    session.add(relationship)
                await session.commit()
                print(f"Seeded {len(data['family_relationships'])} family relationships")
            
            # 5. Seed Family Disease History
            if "family_disease_history" in data:
                print("Seeding family disease history...")
                for disease_data in data["family_disease_history"]:
                    disease_data = convert_timestamps(disease_data)
                    disease = FamilyDiseaseHistory(**disease_data)
                    session.add(disease)
                await session.commit()
                print(f"Seeded {len(data['family_disease_history'])} family disease records")
            
            # 6. Seed Doctor Visits
            if "doctor_visits" in data:
                print("Seeding doctor visits...")
                for visit_data in data["doctor_visits"]:
                    visit_data = convert_timestamps(visit_data)
                    # Map doctor_id to doctor_patient_id for new schema
                    if "doctor_id" in visit_data:
                        visit_data["doctor_patient_id"] = visit_data.pop("doctor_id")
                    visit = DoctorVisit(**visit_data)
                    session.add(visit)
                await session.commit()
                print(f"Seeded {len(data['doctor_visits'])} doctor visits")
            
            # 7. Seed Symptoms
            if "symptoms" in data:
                print("Seeding symptoms...")
                for symptom_data in data["symptoms"]:
                    symptom_data = convert_timestamps(symptom_data)
                    symptom = Symptom(**symptom_data)
                    session.add(symptom)
                await session.commit()
                print(f"Seeded {len(data['symptoms'])} symptoms")
            
            # 8. Seed Diagnoses
            if "diagnoses" in data:
                print("Seeding diagnoses...")
                for diagnosis_data in data["diagnoses"]:
                    diagnosis_data = convert_timestamps(diagnosis_data)
                    diagnosis = Diagnosis(**diagnosis_data)
                    session.add(diagnosis)
                await session.commit()
                print(f"Seeded {len(data['diagnoses'])} diagnoses")
            
            # 9. Seed Prescriptions
            if "prescriptions" in data:
                print("Seeding prescriptions...")
                for prescription_data in data["prescriptions"]:
                    prescription_data = convert_timestamps(prescription_data)
                    prescription = Prescription(**prescription_data)
                    session.add(prescription)
                await session.commit()
                print(f"Seeded {len(data['prescriptions'])} prescriptions")
            
            # 10. Seed Lab Reports
            if "lab_reports" in data:
                print("Seeding lab reports...")
                for report_data in data["lab_reports"]:
                    report_data = convert_timestamps(report_data)
                    report = LabReport(**report_data)
                    session.add(report)
                await session.commit()
                print(f"Seeded {len(data['lab_reports'])} lab reports")
            
            # 11. Seed Lab Test Results
            if "lab_test_results" in data:
                print("Seeding lab test results...")
                for result_data in data["lab_test_results"]:
                    result_data = convert_timestamps(result_data)
                    result = LabTestResult(**result_data)
                    session.add(result)
                await session.commit()
                print(f"Seeded {len(data['lab_test_results'])} lab test results")
            
            # 12. Seed Disease Progressions
            if "disease_progressions" in data:
                print("Seeding disease progressions...")
                for progression_data in data["disease_progressions"]:
                    progression_data = convert_timestamps(progression_data)
                    progression = DiseaseProgression(**progression_data)
                    session.add(progression)
                await session.commit()
                print(f"Seeded {len(data['disease_progressions'])} disease progressions")
            
            print("Database seeding completed successfully!")
            
    except Exception as e:
        print(f"Error during seeding: {e}")
        raise
    finally:
        await engine.dispose()

async def clear_database():
    """Clear all data from database (for testing)"""
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
    
    try:
        async with engine.begin() as conn:
            # Delete in reverse order of dependencies (doctors table no longer exists)
            tables = [
                "lab_test_results", "lab_reports", "prescriptions", "diagnoses", 
                "symptoms", "doctor_visits", "family_disease_history", 
                "family_relationships", "disease_progressions", "ml_predictions",
                "patients", "labs"
            ]
            
            for table in tables:
                await conn.execute(text(f"DELETE FROM {table}"))
                print(f"Cleared {table}")
            
            print("Database cleared successfully!")
            
    except Exception as e:
        print(f"Error clearing database: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed LifeChain AI database")
    parser.add_argument("--clear", action="store_true", help="Clear database before seeding")
    parser.add_argument("--seed", action="store_true", help="Seed database with generated data")
    
    args = parser.parse_args()
    
    if args.clear:
        print("Clearing database...")
        asyncio.run(clear_database())
    
    if args.seed:
        print("Seeding database...")
        asyncio.run(seed_database())
    
    if not args.clear and not args.seed:
        print("Use --clear to clear database or --seed to seed database")
        print("Example: python seed_database.py --clear --seed")
