"""
Seed Demo Patient Script
Creates ONE comprehensive patient with ALL data types for full system demonstration
- Complete family tree (4 generations)
- Multiple doctor visits over 2 years
- Lab reports with test results
- Multiple disease diagnoses
- Disease progression timelines
- ML predictions
- Prescriptions
- Symptoms
"""
import asyncio
from datetime import datetime, date, timedelta
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.patient import Patient, GenderEnum
from app.models.family import FamilyRelationship, FamilyDiseaseHistory, RelationshipTypeEnum, SeverityEnum
from app.models.disease import DiseaseProgression, MLPrediction
from app.models.visit import DoctorVisit, Diagnosis, Prescription, Symptom, VisitTypeEnum, DiagnosisStatusEnum
from app.models.lab import Lab, LabReport, LabTestResult, ReportStatusEnum
import uuid
import random
import json

async def generate_cnic():
    """Generate unique CNIC"""
    return f"{random.randint(10000, 99999)}-{random.randint(1000000, 9999999)}-{random.randint(0, 9)}"

async def seed_demo_patient():
    """Create ONE demo patient with comprehensive data for all features"""
    
    async with AsyncSessionLocal() as db:
        print("\n" + "="*80)
        print("🎯 CREATING DEMO PATIENT WITH COMPLETE DATA")
        print("="*80)
        
        # ============================================================
        # STEP 1: CREATE DEMO PATIENT
        # ============================================================
        print("\n📍 Step 1: Creating Demo Patient...")
        
        demo_patient = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Muhammad",
            last_name="Ali",
            date_of_birth=date(1990, 5, 15),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0300-1234567",
            email="muhammad.ali@demo.com",
            address="House 123, Model Town, Islamabad, Pakistan"
        )
        db.add(demo_patient)
        await db.flush()
        print(f"   ✅ Demo Patient: {demo_patient.first_name} {demo_patient.last_name}")
        print(f"   📋 Patient ID: {demo_patient.patient_id}")
        
        # ============================================================
        # STEP 2: CREATE DOCTOR
        # ============================================================
        print("\n📍 Step 2: Creating Doctor...")
        
        doctor = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Dr. Ahmed",
            last_name="Khan",
            date_of_birth=date(1975, 3, 20),
            gender=GenderEnum.MALE,
            blood_group="O+",
            phone="0301-9876543",
            email="dr.ahmed.khan@hospital.com",
            address="City Hospital, Islamabad",
            is_doctor=True,
            specialization="Internal Medicine & Diabetology",
            license_number=f"PMC-{random.randint(10000, 99999)}",
            hospital_affiliation="City General Hospital"
        )
        db.add(doctor)
        await db.flush()
        print(f"   ✅ Doctor: {doctor.first_name} {doctor.last_name}")
        
        # ============================================================
        # STEP 3: CREATE LAB
        # ============================================================
        print("\n📍 Step 3: Creating Lab...")
        
        lab = Lab(
            lab_id=uuid.uuid4(),
            lab_name="LifeChain Diagnostics",
            lab_location="F-8 Markaz, Islamabad",
            accreditation_number="ISO-15189-2024",
            phone="051-1234567",
            email="info@lifechaindx.com"
        )
        db.add(lab)
        await db.flush()
        print(f"   ✅ Lab: {lab.lab_name}")
        
        # ============================================================
        # STEP 4: CREATE FAMILY MEMBERS (4 Generations)
        # ============================================================
        print("\n📍 Step 4: Creating Family Tree (4 Generations)...")
        
        family_members = {}
        
        # Grandfather (Paternal)
        grandfather = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Abdul",
            last_name="Rahman",
            date_of_birth=date(1940, 8, 10),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0302-1111111"
        )
        db.add(grandfather)
        family_members['grandfather'] = grandfather
        
        # Grandmother (Paternal)
        grandmother = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Fatima",
            last_name="Bibi",
            date_of_birth=date(1945, 12, 5),
            gender=GenderEnum.FEMALE,
            blood_group="B+",
            phone="0302-2222222"
        )
        db.add(grandmother)
        family_members['grandmother'] = grandmother
        
        # Father
        father = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Imran",
            last_name="Ali",
            date_of_birth=date(1965, 6, 20),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0303-3333333"
        )
        db.add(father)
        family_members['father'] = father
        
        # Mother
        mother = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Ayesha",
            last_name="Imran",
            date_of_birth=date(1970, 9, 15),
            gender=GenderEnum.FEMALE,
            blood_group="O+",
            phone="0303-4444444"
        )
        db.add(mother)
        family_members['mother'] = mother
        
        # Spouse
        spouse = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Maryam",
            last_name="Ali",
            date_of_birth=date(1992, 11, 25),
            gender=GenderEnum.FEMALE,
            blood_group="AB+",
            phone="0304-5555555"
        )
        db.add(spouse)
        family_members['spouse'] = spouse
        
        # Child 1
        child1 = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Ahmed",
            last_name="Ali",
            date_of_birth=date(2018, 3, 10),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0305-6666666"
        )
        db.add(child1)
        family_members['child1'] = child1
        
        # Child 2
        child2 = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Zainab",
            last_name="Ali",
            date_of_birth=date(2020, 7, 22),
            gender=GenderEnum.FEMALE,
            blood_group="O+",
            phone="0305-7777777"
        )
        db.add(child2)
        family_members['child2'] = child2
        
        # Brother
        brother = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Hassan",
            last_name="Ali",
            date_of_birth=date(1993, 2, 14),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0306-8888888"
        )
        db.add(brother)
        family_members['brother'] = brother
        
        await db.flush()
        print(f"   ✅ Created {len(family_members)} family members")
        
        # Create Family Relationships
        relationships = [
            # Grandparents
            (demo_patient.patient_id, grandfather.patient_id, RelationshipTypeEnum.GRANDPARENT, True),
            (demo_patient.patient_id, grandmother.patient_id, RelationshipTypeEnum.GRANDPARENT, True),
            # Parents
            (demo_patient.patient_id, father.patient_id, RelationshipTypeEnum.PARENT, True),
            (demo_patient.patient_id, mother.patient_id, RelationshipTypeEnum.PARENT, True),
            # Spouse
            (demo_patient.patient_id, spouse.patient_id, RelationshipTypeEnum.SPOUSE, False),
            # Children
            (demo_patient.patient_id, child1.patient_id, RelationshipTypeEnum.CHILD, True),
            (demo_patient.patient_id, child2.patient_id, RelationshipTypeEnum.CHILD, True),
            # Sibling
            (demo_patient.patient_id, brother.patient_id, RelationshipTypeEnum.SIBLING, True),
            # Reverse relationships
            (grandfather.patient_id, demo_patient.patient_id, RelationshipTypeEnum.GRANDCHILD, True),
            (grandmother.patient_id, demo_patient.patient_id, RelationshipTypeEnum.GRANDCHILD, True),
            (father.patient_id, demo_patient.patient_id, RelationshipTypeEnum.CHILD, True),
            (mother.patient_id, demo_patient.patient_id, RelationshipTypeEnum.CHILD, True),
            (spouse.patient_id, demo_patient.patient_id, RelationshipTypeEnum.SPOUSE, False),
            (child1.patient_id, demo_patient.patient_id, RelationshipTypeEnum.PARENT, True),
            (child2.patient_id, demo_patient.patient_id, RelationshipTypeEnum.PARENT, True),
            (brother.patient_id, demo_patient.patient_id, RelationshipTypeEnum.SIBLING, True),
        ]
        
        for patient_id, relative_id, rel_type, is_blood in relationships:
            rel = FamilyRelationship(
                id=uuid.uuid4(),
                patient_id=patient_id,
                relative_patient_id=relative_id,
                relationship_type=rel_type,
                is_blood_relative=is_blood
            )
            db.add(rel)
        
        print(f"   ✅ Created {len(relationships)} family relationships")
        
        # ============================================================
        # STEP 5: CREATE FAMILY DISEASE HISTORY
        # ============================================================
        print("\n📍 Step 5: Creating Family Disease History...")
        
        family_diseases = [
            (grandfather.patient_id, "diabetes", SeverityEnum.SEVERE, date(1980, 5, 10), "Type 2 Diabetes - 40+ years"),
            (grandfather.patient_id, "hypertension", SeverityEnum.SEVERE, date(1975, 3, 15), "Chronic hypertension"),
            (grandfather.patient_id, "heart_disease", SeverityEnum.MODERATE, date(2000, 8, 20), "Coronary artery disease"),
            (grandmother.patient_id, "diabetes", SeverityEnum.MODERATE, date(1990, 11, 5), "Type 2 Diabetes"),
            (grandmother.patient_id, "chronic_kidney_disease", SeverityEnum.MILD, date(2010, 6, 12), "CKD Stage 2"),
            (father.patient_id, "diabetes", SeverityEnum.MILD, date(2015, 4, 8), "Prediabetes"),
            (father.patient_id, "hypertension", SeverityEnum.MILD, date(2018, 9, 22), "Stage 1 hypertension"),
            (mother.patient_id, "iron_deficiency_anemia", SeverityEnum.MODERATE, date(2005, 7, 30), "Chronic anemia"),
            (mother.patient_id, "thyroid_disease", SeverityEnum.MILD, date(2012, 2, 18), "Hypothyroidism"),
            (brother.patient_id, "diabetes", SeverityEnum.MILD, date(2023, 3, 10), "Prediabetes - recently diagnosed"),
        ]
        
        for patient_id, disease, severity, diagnosed, notes in family_diseases:
            fdh = FamilyDiseaseHistory(
                id=uuid.uuid4(),
                patient_id=patient_id,
                disease_name=disease,
                severity=severity,
                diagnosed_at=diagnosed,
                notes=notes
            )
            db.add(fdh)
        
        print(f"   ✅ Created {len(family_diseases)} family disease records")
        
        # ============================================================
        # STEP 6: CREATE DOCTOR VISITS (24 months of visits)
        # ============================================================
        print("\n📍 Step 6: Creating Doctor Visits (2 years)...")
        
        base_date = datetime.now()
        visits = []
        
        visit_data = [
            # Initial consultation - 24 months ago
            (730, VisitTypeEnum.CONSULTATION, "Routine checkup, feeling fatigued", "Initial assessment. Ordered comprehensive metabolic panel."),
            # Follow-up visits
            (700, VisitTypeEnum.LAB_REVIEW, "Lab results review", "HbA1c elevated at 6.2%. Prediabetes diagnosed. Lifestyle modifications recommended."),
            (640, VisitTypeEnum.FOLLOW_UP, "Follow-up for prediabetes", "Started metformin 500mg. Diet and exercise plan discussed."),
            (580, VisitTypeEnum.LAB_REVIEW, "3-month lab review", "HbA1c improved to 5.9%. Continue current treatment."),
            (520, VisitTypeEnum.ROUTINE_CHECKUP, "Routine checkup", "Blood pressure slightly elevated. Monitoring advised."),
            (460, VisitTypeEnum.LAB_REVIEW, "6-month comprehensive review", "Kidney function tests ordered due to family history of CKD."),
            (400, VisitTypeEnum.FOLLOW_UP, "Kidney function review", "eGFR 85. Early CKD Stage 1 detected. Nephrology referral."),
            (340, VisitTypeEnum.CONSULTATION, "Nephrology consultation", "CKD management plan initiated. ACE inhibitor started."),
            (280, VisitTypeEnum.LAB_REVIEW, "Quarterly monitoring", "Stable kidney function. Diabetes well controlled."),
            (220, VisitTypeEnum.FOLLOW_UP, "Follow-up visit", "Patient feeling well. Continue current medications."),
            (160, VisitTypeEnum.LAB_REVIEW, "Lab results review", "HbA1c 5.7%. Excellent control. eGFR stable at 82."),
            (100, VisitTypeEnum.ROUTINE_CHECKUP, "Annual physical exam", "Overall health improved. Weight loss of 5kg noted."),
            (60, VisitTypeEnum.LAB_REVIEW, "Recent lab review", "All markers within acceptable range."),
            (30, VisitTypeEnum.FOLLOW_UP, "Monthly check-in", "Patient reports increased energy. Exercise routine consistent."),
            (7, VisitTypeEnum.CONSULTATION, "Current visit", "Comprehensive review. Patient doing well overall."),
        ]
        
        for days_ago, visit_type, complaint, notes in visit_data:
            visit = DoctorVisit(
                visit_id=uuid.uuid4(),
                patient_id=demo_patient.patient_id,
                doctor_patient_id=doctor.patient_id,
                visit_date=base_date - timedelta(days=days_ago),
                visit_type=visit_type,
                chief_complaint=complaint,
                doctor_notes=notes,
                vital_signs={
                    "blood_pressure": f"{random.randint(115, 135)}/{random.randint(70, 85)}",
                    "heart_rate": random.randint(68, 82),
                    "temperature": round(36.5 + random.random() * 0.8, 1),
                    "weight_kg": round(78 - (days_ago / 100), 1),
                    "height_cm": 175
                }
            )
            db.add(visit)
            visits.append(visit)
        
        await db.flush()
        print(f"   ✅ Created {len(visits)} doctor visits")
        
        # ============================================================
        # STEP 7: CREATE SYMPTOMS FOR VISITS
        # ============================================================
        print("\n📍 Step 7: Creating Symptoms...")
        
        symptoms_data = [
            (visits[0].visit_id, "Fatigue", 6, 30, "Feeling tired throughout the day"),
            (visits[0].visit_id, "Increased thirst", 4, 14, "Drinking more water than usual"),
            (visits[0].visit_id, "Frequent urination", 5, 14, "Especially at night"),
            (visits[2].visit_id, "Mild headache", 3, 7, "Occasional morning headaches"),
            (visits[4].visit_id, "Dizziness", 2, 3, "When standing up quickly"),
            (visits[6].visit_id, "Lower back discomfort", 3, 14, "Mild aching"),
            (visits[10].visit_id, "Improved energy", 1, 30, "Patient reports feeling better"),
        ]
        
        for visit_id, symptom_name, severity, duration, notes in symptoms_data:
            symptom = Symptom(
                id=uuid.uuid4(),
                visit_id=visit_id,
                symptom_name=symptom_name,
                severity=severity,
                duration_days=duration,
                notes=notes
            )
            db.add(symptom)
        
        print(f"   ✅ Created {len(symptoms_data)} symptoms")
        
        # ============================================================
        # STEP 8: CREATE DIAGNOSES
        # ============================================================
        print("\n📍 Step 8: Creating Diagnoses...")
        
        diagnoses_data = [
            (visits[1].visit_id, "Prediabetes", DiagnosisStatusEnum.CONFIRMED, 0.89, "xgb_diabetes_v1"),
            (visits[4].visit_id, "Hypertension Stage 1", DiagnosisStatusEnum.SUSPECTED, 0.72, "clinical_assessment"),
            (visits[6].visit_id, "Chronic Kidney Disease Stage 1", DiagnosisStatusEnum.CONFIRMED, 0.85, "xgb_ckd_v1"),
            (visits[10].visit_id, "Prediabetes - Controlled", DiagnosisStatusEnum.CONFIRMED, 0.92, "xgb_diabetes_v1"),
        ]
        
        for visit_id, disease, status, confidence, model in diagnoses_data:
            diagnosis = Diagnosis(
                diagnosis_id=uuid.uuid4(),
                visit_id=visit_id,
                disease_name=disease,
                diagnosis_date=datetime.now() - timedelta(days=random.randint(1, 30)),
                status=status,
                confidence_score=confidence,
                ml_model_used=model,
                notes=f"{'ML-assisted' if 'xgb' in model else 'Clinical'} diagnosis with {confidence:.0%} confidence"
            )
            db.add(diagnosis)
        
        print(f"   ✅ Created {len(diagnoses_data)} diagnoses")
        
        # ============================================================
        # STEP 9: CREATE PRESCRIPTIONS
        # ============================================================
        print("\n📍 Step 9: Creating Prescriptions...")
        
        prescriptions_data = [
            (visits[2].visit_id, "Metformin", "500mg", "Once daily", 90, "Take with breakfast"),
            (visits[2].visit_id, "Vitamin D3", "2000 IU", "Once daily", 90, "For vitamin D deficiency"),
            (visits[7].visit_id, "Lisinopril", "5mg", "Once daily", 30, "ACE inhibitor for kidney protection"),
            (visits[7].visit_id, "Metformin", "500mg", "Twice daily", 90, "Increased dose - take with meals"),
            (visits[12].visit_id, "Metformin", "500mg", "Twice daily", 90, "Continue current dose"),
            (visits[12].visit_id, "Lisinopril", "5mg", "Once daily", 30, "Continue for BP and kidney"),
        ]
        
        for visit_id, med, dosage, freq, duration, instructions in prescriptions_data:
            prescription = Prescription(
                prescription_id=uuid.uuid4(),
                visit_id=visit_id,
                medication_name=med,
                dosage=dosage,
                frequency=freq,
                duration_days=duration,
                instructions=instructions
            )
            db.add(prescription)
        
        print(f"   ✅ Created {len(prescriptions_data)} prescriptions")
        
        # ============================================================
        # STEP 10: CREATE LAB REPORTS WITH TEST RESULTS
        # ============================================================
        print("\n📍 Step 10: Creating Lab Reports & Test Results...")
        
        lab_reports_data = [
            # Initial labs - 24 months ago
            (visits[0].visit_id, 730, "Comprehensive Metabolic Panel", [
                ("fasting_glucose", 118, "mg/dL", 70, 100),
                ("hba1c", 6.2, "%", 4.0, 5.6),
                ("hdl", 42, "mg/dL", 40, 60),
                ("ldl", 135, "mg/dL", 0, 100),
                ("triglycerides", 165, "mg/dL", 0, 150),
                ("total_cholesterol", 210, "mg/dL", 0, 200),
                ("creatinine", 1.1, "mg/dL", 0.7, 1.3),
                ("bun", 18, "mg/dL", 7, 20),
            ]),
            # 3-month follow-up
            (visits[3].visit_id, 580, "Diabetes Panel", [
                ("fasting_glucose", 105, "mg/dL", 70, 100),
                ("hba1c", 5.9, "%", 4.0, 5.6),
                ("hemoglobin", 14.2, "g/dL", 13.5, 17.5),
            ]),
            # 6-month - kidney function
            (visits[5].visit_id, 460, "Renal Function Panel", [
                ("fasting_glucose", 102, "mg/dL", 70, 100),
                ("hba1c", 5.8, "%", 4.0, 5.6),
                ("creatinine", 1.15, "mg/dL", 0.7, 1.3),
                ("egfr", 85, "mL/min", 90, 120),
                ("bun", 20, "mg/dL", 7, 20),
                ("uacr", 35, "mg/g", 0, 30),
                ("potassium", 4.3, "mEq/L", 3.5, 5.0),
                ("sodium", 140, "mEq/L", 136, 145),
            ]),
            # 9-month check
            (visits[8].visit_id, 280, "Quarterly Panel", [
                ("fasting_glucose", 98, "mg/dL", 70, 100),
                ("hba1c", 5.7, "%", 4.0, 5.6),
                ("creatinine", 1.12, "mg/dL", 0.7, 1.3),
                ("egfr", 82, "mL/min", 90, 120),
                ("hemoglobin", 14.5, "g/dL", 13.5, 17.5),
            ]),
            # 12-month annual
            (visits[10].visit_id, 160, "Annual Comprehensive Panel", [
                ("fasting_glucose", 95, "mg/dL", 70, 100),
                ("hba1c", 5.7, "%", 4.0, 5.6),
                ("hdl", 48, "mg/dL", 40, 60),
                ("ldl", 115, "mg/dL", 0, 100),
                ("triglycerides", 140, "mg/dL", 0, 150),
                ("total_cholesterol", 188, "mg/dL", 0, 200),
                ("creatinine", 1.1, "mg/dL", 0.7, 1.3),
                ("egfr", 84, "mL/min", 90, 120),
                ("bun", 17, "mg/dL", 7, 20),
                ("hemoglobin", 14.8, "g/dL", 13.5, 17.5),
                ("ferritin", 85, "ng/mL", 30, 400),
                ("vitamin_d", 42, "ng/mL", 30, 100),
            ]),
            # Recent labs
            (visits[12].visit_id, 60, "Recent Checkup", [
                ("fasting_glucose", 92, "mg/dL", 70, 100),
                ("hba1c", 5.6, "%", 4.0, 5.6),
                ("creatinine", 1.08, "mg/dL", 0.7, 1.3),
                ("egfr", 86, "mL/min", 90, 120),
            ]),
            # Latest labs - 1 week ago
            (visits[14].visit_id, 7, "Current Status", [
                ("fasting_glucose", 94, "mg/dL", 70, 100),
                ("hba1c", 5.5, "%", 4.0, 5.6),
                ("hdl", 52, "mg/dL", 40, 60),
                ("ldl", 105, "mg/dL", 0, 100),
                ("triglycerides", 125, "mg/dL", 0, 150),
                ("total_cholesterol", 175, "mg/dL", 0, 200),
                ("creatinine", 1.05, "mg/dL", 0.7, 1.3),
                ("egfr", 88, "mL/min", 90, 120),
                ("bun", 16, "mg/dL", 7, 20),
                ("uacr", 28, "mg/g", 0, 30),
                ("hemoglobin", 15.0, "g/dL", 13.5, 17.5),
                ("potassium", 4.2, "mEq/L", 3.5, 5.0),
                ("sodium", 141, "mEq/L", 136, 145),
            ]),
        ]
        
        lab_report_count = 0
        test_result_count = 0
        
        for visit_id, days_ago, report_type, tests in lab_reports_data:
            report = LabReport(
                report_id=uuid.uuid4(),
                patient_id=demo_patient.patient_id,
                lab_id=lab.lab_id,
                visit_id=visit_id,
                report_date=base_date - timedelta(days=days_ago),
                report_type=report_type,
                status=ReportStatusEnum.COMPLETED,
                test_name=report_type
            )
            db.add(report)
            await db.flush()
            lab_report_count += 1
            
            for test_name, value, unit, ref_min, ref_max in tests:
                is_abnormal = value < ref_min or value > ref_max
                test_result = LabTestResult(
                    result_id=uuid.uuid4(),
                    report_id=report.report_id,
                    test_name=test_name,
                    test_value=value,
                    unit=unit,
                    reference_range_min=ref_min,
                    reference_range_max=ref_max,
                    is_abnormal=is_abnormal
                )
                db.add(test_result)
                test_result_count += 1
        
        print(f"   ✅ Created {lab_report_count} lab reports with {test_result_count} test results")
        
        # ============================================================
        # STEP 11: CREATE DISEASE PROGRESSIONS (Timelines)
        # ============================================================
        print("\n📍 Step 11: Creating Disease Progression Timelines...")
        
        # Diabetes Progression (improving over 2 years)
        diabetes_progression = [
            ("Prediabetes", 730, 0.89, "Initial diagnosis - elevated HbA1c"),
            ("Prediabetes", 580, 0.85, "3-month check - slight improvement with lifestyle changes"),
            ("Prediabetes", 460, 0.82, "6-month check - continued improvement"),
            ("Controlled", 280, 0.88, "9-month check - well controlled on metformin"),
            ("Controlled", 160, 0.91, "12-month check - excellent control"),
            ("Controlled", 60, 0.93, "18-month check - stable"),
            ("Normal", 7, 0.87, "Current - HbA1c normalized, monitoring continues"),
        ]
        
        # CKD Progression (stable)
        ckd_progression = [
            ("Stage 1", 460, 0.85, "Initial CKD diagnosis - eGFR 85"),
            ("Stage 1", 280, 0.82, "Stable - eGFR 82"),
            ("Stage 1", 160, 0.86, "Stable with treatment - eGFR 84"),
            ("Stage 1", 60, 0.88, "Improving - eGFR 86"),
            ("Stage 1", 7, 0.90, "Current - eGFR 88, near normal range"),
        ]
        
        progression_count = 0
        for stage, days_ago, confidence, notes in diabetes_progression:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=demo_patient.patient_id,
                disease_name="diabetes",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="diabetes_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        for stage, days_ago, confidence, notes in ckd_progression:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=demo_patient.patient_id,
                disease_name="chronic_kidney_disease",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="ckd_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        print(f"   ✅ Created {progression_count} progression records")
        
        # ============================================================
        # STEP 12: CREATE ML PREDICTIONS
        # ============================================================
        print("\n📍 Step 12: Creating ML Predictions...")
        
        predictions_data = [
            ("xgb_diabetes_v1", "v1.0", "diagnosis", {"predicted_class": "Prediabetes", "confidence": 0.89}, 730),
            ("diabetes_progression_lstm", "v1.0", "progression", {"predicted_class": "Improving", "confidence": 0.85}, 580),
            ("xgb_ckd_v1", "v1.0", "diagnosis", {"predicted_class": "Stage 1", "confidence": 0.85}, 460),
            ("ckd_progression_lstm", "v1.0", "progression", {"predicted_class": "Stable", "confidence": 0.88}, 280),
            ("xgb_diabetes_v1", "v1.0", "diagnosis", {"predicted_class": "Controlled", "confidence": 0.91}, 160),
            ("xgb_diabetes_v1", "v1.0", "diagnosis", {"predicted_class": "Normal", "confidence": 0.87}, 7),
        ]
        
        for model, version, pred_type, result, days_ago in predictions_data:
            prediction = MLPrediction(
                prediction_id=uuid.uuid4(),
                patient_id=demo_patient.patient_id,
                model_name=model,
                model_version=version,
                input_features={"source": "lab_test_results", "visit_based": True},
                prediction_result=result,
                confidence_score=result.get("confidence"),
                prediction_date=base_date - timedelta(days=days_ago)
            )
            db.add(prediction)
        
        print(f"   ✅ Created {len(predictions_data)} ML predictions")
        
        # ============================================================
        # COMMIT ALL DATA
        # ============================================================
        await db.commit()
        
        # ============================================================
        # PRINT SUMMARY
        # ============================================================
        print("\n" + "="*80)
        print("✅ DEMO PATIENT CREATED SUCCESSFULLY!")
        print("="*80)
        
        print(f"""
🎯 DEMO PATIENT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Patient ID: {demo_patient.patient_id}
👤 Name: {demo_patient.first_name} {demo_patient.last_name}
🆔 CNIC: {demo_patient.cnic}
📧 Email: {demo_patient.email}

📊 DATA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Family Members: {len(family_members)} (4 generations)
• Family Relationships: {len(relationships)}
• Family Disease History: {len(family_diseases)}
• Doctor Visits: {len(visits)} (over 2 years)
• Symptoms: {len(symptoms_data)}
• Diagnoses: {len(diagnoses_data)}
• Prescriptions: {len(prescriptions_data)}
• Lab Reports: {lab_report_count}
• Lab Test Results: {test_result_count}
• Disease Progressions: {progression_count} (Diabetes + CKD)
• ML Predictions: {len(predictions_data)}

👨‍⚕️ DOCTOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Doctor ID: {doctor.patient_id}
• Name: {doctor.first_name} {doctor.last_name}

🏥 LAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Lab ID: {lab.lab_id}
• Name: {lab.lab_name}
""")

        print("""
📌 TEST API ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        print(f"""
# Get Patient Details
curl 'http://localhost:8080/api/v1/patients/{demo_patient.patient_id}'

# Get Family Tree
curl 'http://localhost:8080/api/v1/patients/{demo_patient.patient_id}/family-tree'

# Get Family Disease History
curl 'http://localhost:8080/api/v1/patients/{demo_patient.patient_id}/family-disease-history'

# Get Visits
curl 'http://localhost:8080/api/v1/visits?patient_id={demo_patient.patient_id}'

# Get Progression Timeline (Diabetes)
curl 'http://localhost:8080/api/v1/reports/patient/{demo_patient.patient_id}/progression-timeline?disease_name=diabetes&months_back=36'

# Get Progression Timeline (CKD)
curl 'http://localhost:8080/api/v1/reports/patient/{demo_patient.patient_id}/progression-timeline?disease_name=ckd&months_back=36'

# Get Risk Assessment
curl 'http://localhost:8080/api/v1/reports/patient/{demo_patient.patient_id}/risk-assessment'

# Get AI Recommendations
curl 'http://localhost:8080/api/v1/reports/patient/{demo_patient.patient_id}/recommendations'

# Get Lab Reports
curl 'http://localhost:8080/api/v1/labs/reports?patient_id={demo_patient.patient_id}'

# Get Lab Measurements Timeline
curl 'http://localhost:8080/api/v1/reports/patient/{demo_patient.patient_id}/lab-measurements-timeline?months_back=36'

# Run ML Diagnosis
curl -X POST 'http://localhost:8080/api/v1/ml/diagnosis/patient/{demo_patient.patient_id}?disease_name=diabetes&auto_save=false'

# Run ML Progression Prediction
curl -X POST 'http://localhost:8080/api/v1/ml/progression/patient/{demo_patient.patient_id}?disease_name=diabetes'
""")
        
        # Save patient ID to file for easy access
        with open("demo_patient_id.txt", "w") as f:
            f.write(str(demo_patient.patient_id))
        
        print(f"\n💾 Patient ID saved to: demo_patient_id.txt")
        
        return demo_patient.patient_id

if __name__ == "__main__":
    patient_id = asyncio.run(seed_demo_patient())
    print(f"\n🎯 Demo Patient ID: {patient_id}")

