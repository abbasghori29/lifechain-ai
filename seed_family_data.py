"""
Seed Family Data Script
Creates a comprehensive 4-generation family with disease history for frontend testing
"""
import asyncio
from datetime import datetime, date, timedelta
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.patient import Patient, GenderEnum
from app.models.family import FamilyRelationship, FamilyDiseaseHistory, RelationshipTypeEnum, SeverityEnum
from app.models.disease import DiseaseProgression
import uuid
import random

# ============================================================================
# FAMILY STRUCTURE (15 Members, 4 Generations)
# ============================================================================
#
#                     GENERATION 1 (Grandparents - 4 people)
#                 ┌─────────────────────────────────────────┐
#           Grandfather ═══ Grandmother        Grandfather ═══ Grandmother
#           (Paternal)      (Paternal)         (Maternal)      (Maternal)
#           DADA            DADI               NANA            NANI
#                 │                                   │
#                 └──────────────┬────────────────────┘
#                                │
#                     GENERATION 2 (Parents + Siblings - 4 people)
#                 ┌──────────────┼──────────────────────┐
#           Uncle (Chacha)   Father ═══ Mother      Aunt (Khala)
#           (Dad's brother)  (MAIN)     (MAIN)      (Mom's sister)
#                 │              │
#                 │              └────────────────────────┐
#                 │                                       │
#                     GENERATION 3 (Patient + Siblings + Cousins - 5 people)
#                 ┌──────────────┼──────────────────────┐
#           Cousin         ⭐ MAIN PATIENT ═══ Spouse   Brother    Sister
#           (Chacha's son)                   (Not blood)
#                                │
#                                │
#                     GENERATION 4 (Children - 2 people)
#                          ┌─────┴─────┐
#                       Son           Daughter
#
# ============================================================================

async def generate_cnic():
    """Generate unique CNIC"""
    return f"{random.randint(10000, 99999)}-{random.randint(1000000, 9999999)}-{random.randint(0, 9)}"

async def seed_family_data():
    """Create complete family structure with disease history"""
    
    async with AsyncSessionLocal() as db:
        print("\n" + "="*80)
        print("🏠 SEEDING FAMILY DATA - 4 Generations, 15 Members")
        print("="*80)
        
        # Store all created patients
        family_members = {}
        
        # ============================================================
        # GENERATION 1: GRANDPARENTS (4 people)
        # ============================================================
        print("\n📍 Creating Generation 1: Grandparents...")
        
        # Paternal Grandfather (DADA)
        dada = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Muhammad",
            last_name="Ahmad",
            date_of_birth=date(1945, 3, 15),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0300-1234567",
            address="House 1, Street 5, Lahore"
        )
        db.add(dada)
        family_members['dada'] = dada
        
        # Paternal Grandmother (DADI)
        dadi = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Fatima",
            last_name="Ahmad",
            date_of_birth=date(1948, 7, 22),
            gender=GenderEnum.FEMALE,
            blood_group="B+",
            phone="0300-1234568",
            address="House 1, Street 5, Lahore"
        )
        db.add(dadi)
        family_members['dadi'] = dadi
        
        # Maternal Grandfather (NANA)
        nana = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Abdul",
            last_name="Rashid",
            date_of_birth=date(1943, 11, 8),
            gender=GenderEnum.MALE,
            blood_group="O+",
            phone="0301-2345678",
            address="House 22, Block C, Karachi"
        )
        db.add(nana)
        family_members['nana'] = nana
        
        # Maternal Grandmother (NANI)
        nani = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Khadija",
            last_name="Rashid",
            date_of_birth=date(1947, 5, 30),
            gender=GenderEnum.FEMALE,
            blood_group="A-",
            phone="0301-2345679",
            address="House 22, Block C, Karachi"
        )
        db.add(nani)
        family_members['nani'] = nani
        
        # ============================================================
        # GENERATION 2: PARENTS + AUNTS/UNCLES (4 people)
        # ============================================================
        print("📍 Creating Generation 2: Parents & Aunts/Uncles...")
        
        # Father
        father = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Imran",
            last_name="Ahmad",
            date_of_birth=date(1970, 6, 12),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0302-3456789",
            address="House 45, Model Town, Islamabad"
        )
        db.add(father)
        family_members['father'] = father
        
        # Mother
        mother = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Ayesha",
            last_name="Imran",
            date_of_birth=date(1975, 9, 25),
            gender=GenderEnum.FEMALE,
            blood_group="B+",
            phone="0302-3456790",
            address="House 45, Model Town, Islamabad"
        )
        db.add(mother)
        family_members['mother'] = mother
        
        # Uncle (Chacha - Father's brother)
        chacha = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Usman",
            last_name="Ahmad",
            date_of_birth=date(1968, 2, 18),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0303-4567890",
            address="House 12, Gulberg, Lahore"
        )
        db.add(chacha)
        family_members['chacha'] = chacha
        
        # Aunt (Khala - Mother's sister)
        khala = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Sana",
            last_name="Hussain",
            date_of_birth=date(1978, 12, 3),
            gender=GenderEnum.FEMALE,
            blood_group="O+",
            phone="0304-5678901",
            address="House 8, DHA Phase 5, Karachi"
        )
        db.add(khala)
        family_members['khala'] = khala
        
        # ============================================================
        # GENERATION 3: MAIN PATIENT + SIBLINGS + COUSINS (5 people)
        # ============================================================
        print("📍 Creating Generation 3: Main Patient, Siblings & Cousins...")
        
        # ⭐ MAIN PATIENT
        main_patient = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Ali",
            last_name="Imran",
            date_of_birth=date(1995, 4, 10),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0305-6789012",
            email="ali.imran@email.com",
            address="House 45, Model Town, Islamabad"
        )
        db.add(main_patient)
        family_members['main_patient'] = main_patient
        
        # Spouse (Not blood relative)
        spouse = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Maryam",
            last_name="Ali",
            date_of_birth=date(1997, 8, 14),
            gender=GenderEnum.FEMALE,
            blood_group="AB+",
            phone="0305-6789013",
            email="maryam.ali@email.com",
            address="House 45, Model Town, Islamabad"
        )
        db.add(spouse)
        family_members['spouse'] = spouse
        
        # Brother
        brother = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Hassan",
            last_name="Imran",
            date_of_birth=date(1998, 1, 20),
            gender=GenderEnum.MALE,
            blood_group="B+",
            phone="0306-7890123",
            address="House 45, Model Town, Islamabad"
        )
        db.add(brother)
        family_members['brother'] = brother
        
        # Sister
        sister = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Zainab",
            last_name="Imran",
            date_of_birth=date(2000, 11, 5),
            gender=GenderEnum.FEMALE,
            blood_group="A+",
            phone="0306-7890124",
            address="House 45, Model Town, Islamabad"
        )
        db.add(sister)
        family_members['sister'] = sister
        
        # Cousin (Chacha's son)
        cousin = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Bilal",
            last_name="Usman",
            date_of_birth=date(1996, 7, 28),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0307-8901234",
            address="House 12, Gulberg, Lahore"
        )
        db.add(cousin)
        family_members['cousin'] = cousin
        
        # ============================================================
        # GENERATION 4: CHILDREN (2 people)
        # ============================================================
        print("📍 Creating Generation 4: Children...")
        
        # Son
        son = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Ahmed",
            last_name="Ali",
            date_of_birth=date(2020, 3, 15),
            gender=GenderEnum.MALE,
            blood_group="A+",
            phone="0308-9012345",
            address="House 45, Model Town, Islamabad"
        )
        db.add(son)
        family_members['son'] = son
        
        # Daughter
        daughter = Patient(
            patient_id=uuid.uuid4(),
            cnic=await generate_cnic(),
            first_name="Hira",
            last_name="Ali",
            date_of_birth=date(2022, 9, 8),
            gender=GenderEnum.FEMALE,
            blood_group="B+",
            phone="0308-9012346",
            address="House 45, Model Town, Islamabad"
        )
        db.add(daughter)
        family_members['daughter'] = daughter
        
        await db.flush()
        print(f"   ✅ Created {len(family_members)} family members")
        
        # ============================================================
        # CREATE FAMILY RELATIONSHIPS (Bidirectional)
        # ============================================================
        print("\n📍 Creating Family Relationships...")
        
        relationships = []
        
        # Helper function to create bidirectional relationship
        def add_relationship(person1_key, person2_key, rel_type1, rel_type2, is_blood=True):
            p1 = family_members[person1_key]
            p2 = family_members[person2_key]
            relationships.append((p1.patient_id, p2.patient_id, rel_type1, is_blood))
            relationships.append((p2.patient_id, p1.patient_id, rel_type2, is_blood))
        
        # Grandparent-Parent relationships
        add_relationship('dada', 'father', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dadi', 'father', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dada', 'chacha', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dadi', 'chacha', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nana', 'mother', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nani', 'mother', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nana', 'khala', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nani', 'khala', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        
        # Spouse relationships (Generation 1)
        add_relationship('dada', 'dadi', RelationshipTypeEnum.SPOUSE, RelationshipTypeEnum.SPOUSE, is_blood=False)
        add_relationship('nana', 'nani', RelationshipTypeEnum.SPOUSE, RelationshipTypeEnum.SPOUSE, is_blood=False)
        
        # Parent-Child relationships
        add_relationship('father', 'main_patient', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('mother', 'main_patient', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('father', 'brother', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('mother', 'brother', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('father', 'sister', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('mother', 'sister', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('chacha', 'cousin', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        
        # Spouse relationship (Generation 2)
        add_relationship('father', 'mother', RelationshipTypeEnum.SPOUSE, RelationshipTypeEnum.SPOUSE, is_blood=False)
        
        # Sibling relationships
        add_relationship('father', 'chacha', RelationshipTypeEnum.SIBLING, RelationshipTypeEnum.SIBLING)
        add_relationship('mother', 'khala', RelationshipTypeEnum.SIBLING, RelationshipTypeEnum.SIBLING)
        add_relationship('main_patient', 'brother', RelationshipTypeEnum.SIBLING, RelationshipTypeEnum.SIBLING)
        add_relationship('main_patient', 'sister', RelationshipTypeEnum.SIBLING, RelationshipTypeEnum.SIBLING)
        add_relationship('brother', 'sister', RelationshipTypeEnum.SIBLING, RelationshipTypeEnum.SIBLING)
        
        # Spouse relationship (Generation 3)
        add_relationship('main_patient', 'spouse', RelationshipTypeEnum.SPOUSE, RelationshipTypeEnum.SPOUSE, is_blood=False)
        
        # Aunt/Uncle relationships
        add_relationship('chacha', 'main_patient', RelationshipTypeEnum.AUNT_UNCLE, RelationshipTypeEnum.NIECE_NEPHEW)
        add_relationship('chacha', 'brother', RelationshipTypeEnum.AUNT_UNCLE, RelationshipTypeEnum.NIECE_NEPHEW)
        add_relationship('chacha', 'sister', RelationshipTypeEnum.AUNT_UNCLE, RelationshipTypeEnum.NIECE_NEPHEW)
        add_relationship('khala', 'main_patient', RelationshipTypeEnum.AUNT_UNCLE, RelationshipTypeEnum.NIECE_NEPHEW)
        add_relationship('khala', 'brother', RelationshipTypeEnum.AUNT_UNCLE, RelationshipTypeEnum.NIECE_NEPHEW)
        add_relationship('khala', 'sister', RelationshipTypeEnum.AUNT_UNCLE, RelationshipTypeEnum.NIECE_NEPHEW)
        
        # Cousin relationships
        add_relationship('main_patient', 'cousin', RelationshipTypeEnum.COUSIN, RelationshipTypeEnum.COUSIN)
        add_relationship('brother', 'cousin', RelationshipTypeEnum.COUSIN, RelationshipTypeEnum.COUSIN)
        add_relationship('sister', 'cousin', RelationshipTypeEnum.COUSIN, RelationshipTypeEnum.COUSIN)
        
        # Grandparent-Grandchild relationships (Main patient's grandparents)
        add_relationship('dada', 'main_patient', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dadi', 'main_patient', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nana', 'main_patient', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nani', 'main_patient', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dada', 'brother', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dadi', 'brother', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nana', 'brother', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nani', 'brother', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dada', 'sister', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dadi', 'sister', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nana', 'sister', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('nani', 'sister', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dada', 'cousin', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('dadi', 'cousin', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        
        # Parent-Child (Generation 3-4)
        add_relationship('main_patient', 'son', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('spouse', 'son', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('main_patient', 'daughter', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        add_relationship('spouse', 'daughter', RelationshipTypeEnum.PARENT, RelationshipTypeEnum.CHILD)
        
        # Sibling (Generation 4)
        add_relationship('son', 'daughter', RelationshipTypeEnum.SIBLING, RelationshipTypeEnum.SIBLING)
        
        # Grandparent-Grandchild (Generation 2-4)
        add_relationship('father', 'son', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('mother', 'son', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('father', 'daughter', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        add_relationship('mother', 'daughter', RelationshipTypeEnum.GRANDPARENT, RelationshipTypeEnum.GRANDCHILD)
        
        # Create all relationships
        for patient_id, relative_id, rel_type, is_blood in relationships:
            rel = FamilyRelationship(
                id=uuid.uuid4(),
                patient_id=patient_id,
                relative_patient_id=relative_id,
                relationship_type=rel_type,
                is_blood_relative=is_blood
            )
            db.add(rel)
        
        await db.flush()
        print(f"   ✅ Created {len(relationships)} relationship records")
        
        # ============================================================
        # CREATE DISEASE HISTORY (FamilyDiseaseHistory)
        # ============================================================
        print("\n📍 Creating Disease History Records...")
        
        disease_records = [
            # Generation 1 - Grandparents (older, more diseases)
            ('dada', 'diabetes', SeverityEnum.SEVERE, date(1990, 5, 10), "Type 2 diabetes, on insulin"),
            ('dada', 'heart_disease', SeverityEnum.MODERATE, date(2005, 3, 15), "Coronary artery disease"),
            ('dada', 'hypertension', SeverityEnum.SEVERE, date(1985, 8, 20), "Chronic hypertension"),
            ('dadi', 'hypertension', SeverityEnum.MODERATE, date(1995, 11, 5), "Essential hypertension"),
            ('dadi', 'arthritis', SeverityEnum.MILD, date(2010, 6, 12), "Osteoarthritis in knees"),
            ('nana', 'chronic_kidney_disease', SeverityEnum.SEVERE, date(2000, 4, 8), "CKD Stage 4, on dialysis"),
            ('nana', 'diabetes', SeverityEnum.MODERATE, date(1988, 9, 22), "Type 2 diabetes"),
            ('nana', 'hypertension', SeverityEnum.SEVERE, date(1982, 1, 15), "Hypertensive nephropathy"),
            ('nani', 'diabetes', SeverityEnum.MILD, date(2005, 7, 30), "Type 2 diabetes, diet controlled"),
            ('nani', 'iron_deficiency_anemia', SeverityEnum.MODERATE, date(2015, 2, 18), "Chronic iron deficiency"),
            
            # Generation 2 - Parents/Aunts/Uncles
            ('father', 'diabetes', SeverityEnum.MILD, date(2018, 3, 10), "Prediabetes, lifestyle management"),
            ('father', 'hypertension', SeverityEnum.MILD, date(2020, 6, 5), "Stage 1 hypertension"),
            ('mother', 'iron_deficiency_anemia', SeverityEnum.MODERATE, date(2010, 11, 20), "Chronic anemia"),
            ('mother', 'thyroid_disease', SeverityEnum.MILD, date(2015, 8, 12), "Hypothyroidism"),
            ('chacha', 'diabetes', SeverityEnum.MODERATE, date(2012, 4, 15), "Type 2 diabetes on metformin"),
            ('chacha', 'chronic_kidney_disease', SeverityEnum.MODERATE, date(2019, 9, 8), "CKD Stage 3"),
            ('khala', 'chronic_kidney_disease', SeverityEnum.MILD, date(2021, 5, 22), "CKD Stage 2"),
            ('khala', 'hypertension', SeverityEnum.MODERATE, date(2016, 12, 3), "Essential hypertension"),
            
            # Generation 3 - Main patient & siblings
            ('main_patient', 'chronic_kidney_disease', SeverityEnum.MILD, date(2023, 6, 15), "CKD Stage 1, early detection"),
            ('main_patient', 'hypertension', SeverityEnum.MILD, date(2024, 1, 10), "Borderline hypertension"),
            ('brother', 'diabetes', SeverityEnum.MILD, date(2024, 3, 20), "Prediabetes"),
            ('brother', 'iron_deficiency_anemia', SeverityEnum.MILD, date(2023, 8, 5), "Mild anemia"),
            ('sister', 'thyroid_disease', SeverityEnum.MILD, date(2024, 5, 12), "Subclinical hypothyroidism"),
            ('cousin', 'diabetes', SeverityEnum.MODERATE, date(2022, 7, 8), "Type 2 diabetes"),
            ('spouse', None, None, None, None),  # Healthy - no diseases
            
            # Generation 4 - Children (mostly healthy, some early signs)
            ('son', None, None, None, None),  # Healthy
            ('daughter', 'iron_deficiency_anemia', SeverityEnum.MILD, date(2024, 9, 1), "Mild anemia, dietary iron supplementation"),
        ]
        
        disease_count = 0
        for member_key, disease, severity, diagnosed, notes in disease_records:
            if disease is None:
                continue
            
            patient = family_members[member_key]
            disease_history = FamilyDiseaseHistory(
                id=uuid.uuid4(),
                patient_id=patient.patient_id,
                disease_name=disease,
                severity=severity,
                diagnosed_at=diagnosed,
                notes=notes
            )
            db.add(disease_history)
            disease_count += 1
        
        await db.flush()
        print(f"   ✅ Created {disease_count} disease history records")
        
        # ============================================================
        # CREATE DISEASE PROGRESSIONS (for timeline charts)
        # ============================================================
        print("\n📍 Creating Disease Progression Timelines...")
        
        progression_count = 0
        base_date = datetime.now()
        
        # Main Patient CKD Timeline (2 years)
        ckd_stages = [
            ("Normal", 730, 0.85, "Baseline screening - normal kidney function"),
            ("Stage 1", 550, 0.88, "Early kidney damage detected"),
            ("Stage 1", 365, 0.90, "Stable Stage 1 CKD"),
            ("Stage 1", 180, 0.87, "Continued monitoring"),
            ("Stage 2", 90, 0.92, "Slight progression to Stage 2"),
            ("Stage 2", 30, 0.89, "Current status - Stage 2 CKD"),
        ]
        
        for stage, days_ago, confidence, notes in ckd_stages:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=family_members['main_patient'].patient_id,
                disease_name="chronic_kidney_disease",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="ckd_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        # Nana CKD Timeline (severe case, 5 years)
        nana_ckd = [
            ("Stage 2", 1825, 0.82, "Initial diagnosis"),
            ("Stage 3a", 1460, 0.85, "Progression noted"),
            ("Stage 3b", 1095, 0.88, "Further decline"),
            ("Stage 4", 730, 0.91, "Severe CKD"),
            ("Stage 4", 365, 0.89, "Pre-dialysis care"),
            ("ESRD", 180, 0.94, "Started dialysis"),
            ("ESRD", 30, 0.92, "Ongoing dialysis"),
        ]
        
        for stage, days_ago, confidence, notes in nana_ckd:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=family_members['nana'].patient_id,
                disease_name="chronic_kidney_disease",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="ckd_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        # Father Diabetes Timeline (3 years)
        father_diabetes = [
            ("Normal", 1095, 0.85, "Baseline - normal glucose"),
            ("Prediabetes", 730, 0.88, "Elevated HbA1c detected"),
            ("Prediabetes", 365, 0.86, "Diet and exercise program"),
            ("Prediabetes", 180, 0.89, "Improved but still prediabetic"),
            ("Controlled", 30, 0.91, "Well controlled with lifestyle"),
        ]
        
        for stage, days_ago, confidence, notes in father_diabetes:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=family_members['father'].patient_id,
                disease_name="diabetes",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="diabetes_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        # Mother Anemia Timeline (2 years)
        mother_anemia = [
            ("Moderate", 730, 0.84, "Chronic iron deficiency"),
            ("Moderate", 545, 0.86, "Started iron supplements"),
            ("Mild", 365, 0.88, "Improving with treatment"),
            ("Mild", 180, 0.90, "Continued improvement"),
            ("Stable", 30, 0.92, "Well controlled"),
        ]
        
        for stage, days_ago, confidence, notes in mother_anemia:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=family_members['mother'].patient_id,
                disease_name="iron_deficiency_anemia",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="anemia_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        # Dada Diabetes Timeline (20+ years, severe)
        dada_diabetes = [
            ("Diabetes", 7300, 0.75, "Initial diagnosis"),
            ("Uncontrolled", 5475, 0.78, "Poor glycemic control"),
            ("Complicated", 3650, 0.82, "Diabetic complications developing"),
            ("Complicated", 1825, 0.85, "Retinopathy diagnosed"),
            ("Severe", 730, 0.88, "Multiple complications"),
            ("Severe", 30, 0.90, "Ongoing management"),
        ]
        
        for stage, days_ago, confidence, notes in dada_diabetes:
            prog = DiseaseProgression(
                progression_id=uuid.uuid4(),
                patient_id=family_members['dada'].patient_id,
                disease_name="diabetes",
                progression_stage=stage,
                assessed_date=base_date - timedelta(days=days_ago),
                ml_model_used="diabetes_progression_lstm",
                confidence_score=confidence,
                notes=notes
            )
            db.add(prog)
            progression_count += 1
        
        await db.flush()
        print(f"   ✅ Created {progression_count} disease progression records")
        
        # Commit all changes
        await db.commit()
        
        # ============================================================
        # PRINT SUMMARY FOR FRONTEND DEVELOPER
        # ============================================================
        print("\n" + "="*80)
        print("📊 SUMMARY FOR FRONTEND DEVELOPER")
        print("="*80)
        
        print("\n🏠 FAMILY MEMBERS WITH PATIENT IDs:")
        print("-"*80)
        
        # Query relationship counts for each member
        summary_data = []
        for key, patient in family_members.items():
            # Count relationships
            rel_query = select(func.count(FamilyRelationship.id)).where(
                FamilyRelationship.patient_id == patient.patient_id
            )
            rel_count = (await db.execute(rel_query)).scalar()
            
            # Count diseases
            disease_query = select(func.count(FamilyDiseaseHistory.id)).where(
                FamilyDiseaseHistory.patient_id == patient.patient_id
            )
            disease_count = (await db.execute(disease_query)).scalar()
            
            # Count progressions
            prog_query = select(func.count(DiseaseProgression.progression_id)).where(
                DiseaseProgression.patient_id == patient.patient_id
            )
            prog_count = (await db.execute(prog_query)).scalar()
            
            summary_data.append({
                'key': key,
                'patient': patient,
                'rel_count': rel_count,
                'disease_count': disease_count,
                'prog_count': prog_count
            })
        
        # Print organized by generation
        generations = {
            'Generation 1 (Grandparents)': ['dada', 'dadi', 'nana', 'nani'],
            'Generation 2 (Parents/Aunts/Uncles)': ['father', 'mother', 'chacha', 'khala'],
            'Generation 3 (Patient/Siblings/Cousins)': ['main_patient', 'spouse', 'brother', 'sister', 'cousin'],
            'Generation 4 (Children)': ['son', 'daughter']
        }
        
        for gen_name, members in generations.items():
            print(f"\n📍 {gen_name}:")
            for key in members:
                data = next(d for d in summary_data if d['key'] == key)
                patient = data['patient']
                star = "⭐ " if key == 'main_patient' else "   "
                blood = "🩸" if key != 'spouse' else "💍"
                print(f"{star}{blood} {patient.first_name} {patient.last_name} ({key.upper()})")
                print(f"      Patient ID: {patient.patient_id}")
                print(f"      Relationships: {data['rel_count']} | Diseases: {data['disease_count']} | Progressions: {data['prog_count']}")
        
        # Print JSON-friendly format
        print("\n" + "="*80)
        print("📋 JSON FORMAT FOR FRONTEND:")
        print("="*80)
        print("\n{")
        print('  "family_members": {')
        for i, (key, patient) in enumerate(family_members.items()):
            data = next(d for d in summary_data if d['key'] == key)
            comma = "," if i < len(family_members) - 1 else ""
            print(f'    "{key}": {{')
            print(f'      "patient_id": "{patient.patient_id}",')
            print(f'      "name": "{patient.first_name} {patient.last_name}",')
            print(f'      "gender": "{patient.gender.value}",')
            print(f'      "relationships_count": {data["rel_count"]},')
            print(f'      "diseases_count": {data["disease_count"]},')
            print(f'      "progressions_count": {data["prog_count"]}')
            print(f'    }}{comma}')
        print('  },')
        print(f'  "total_members": {len(family_members)},')
        print(f'  "total_relationships": {len(relationships)},')
        print(f'  "main_patient_id": "{family_members["main_patient"].patient_id}"')
        print("}")
        
        print("\n" + "="*80)
        print("✅ SEEDING COMPLETE!")
        print("="*80)
        print(f"\n🎯 Main Patient ID for testing: {family_members['main_patient'].patient_id}")
        print(f"\n📌 Test endpoints:")
        print(f"""
# Family Tree
curl 'http://localhost:8080/api/v1/patients/{family_members['main_patient'].patient_id}/family-disease-history'

# Progression Timeline (CKD)
curl 'http://localhost:8080/api/v1/reports/patient/{family_members['main_patient'].patient_id}/progression-timeline?disease_name=ckd&months_back=36'

# Risk Assessment
curl 'http://localhost:8080/api/v1/reports/patient/{family_members['main_patient'].patient_id}/risk-assessment'

# Recommendations
curl 'http://localhost:8080/api/v1/reports/patient/{family_members['main_patient'].patient_id}/recommendations'
""")
        
        return family_members

if __name__ == "__main__":
    asyncio.run(seed_family_data())

