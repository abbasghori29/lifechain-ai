"""
Chronic Kidney Disease (CKD) Data Generation for LifeChain AI Healthcare System
Generates realistic, unbiased data for CKD diagnosis and progression ML models
"""

import json
import random
import uuid
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker for realistic data
fake = Faker()

# Load names from JSON file
def load_names_from_json(file_path: str) -> List[Dict[str, str]]:
    """Load names from names.json file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Fixed real doctor notes templates for CKD
CKD_DOCTOR_NOTES = [
    "Patient presents with elevated creatinine and reduced eGFR. Urine albumin-to-creatinine ratio elevated. CKD stage 3a diagnosed. Recommend ACE inhibitor, blood pressure control, and dietary modifications.",
    "Follow-up visit for CKD management. Patient shows stable kidney function. Continue current medication regimen. Monitor electrolytes and blood pressure regularly.",
    "Patient complains of fatigue, swelling in legs, and decreased urine output. Lab results show advanced CKD stage 4. Nephrology referral initiated. Discussed dialysis preparation.",
    "Routine check-up for CKD stage 2. Patient maintaining good control with medication compliance. eGFR stable. Continue monitoring every 3-6 months.",
    "Patient presents with hypertension and proteinuria. Initial workup reveals early CKD. Started on ACE inhibitor and lifestyle modifications. Emphasized importance of blood pressure control.",
    "Laboratory results confirm CKD stage 3b. Patient asymptomatic but requires close monitoring. Phosphate and PTH levels elevated. Started on phosphate binders and vitamin D supplementation.",
    "Patient reports nausea, loss of appetite, and metallic taste. Advanced CKD stage 4-5 suspected. Comprehensive metabolic panel ordered. Discussed renal replacement therapy options.",
    "Follow-up after 6 months of CKD management. Patient showing improvement with eGFR trending upward. Medication compliance excellent. Continue current treatment plan.",
    "Elderly patient with diabetes and hypertension presenting with declining kidney function. CKD stage 3a diagnosed. Multidisciplinary approach with endocrinology and cardiology coordination.",
    "Patient with history of recurrent kidney infections now showing signs of CKD. Imaging reveals kidney scarring. Stage 2 CKD diagnosed. Long-term antibiotic prophylaxis considered.",
    "Young patient with IgA nephropathy presenting with persistent proteinuria. CKD stage 1-2. Renal biopsy recommended for definitive diagnosis and treatment planning.",
    "Patient with polycystic kidney disease showing progressive decline in kidney function. CKD stage 3b. Genetic counseling and family screening recommended.",
    "Post-operative patient with acute kidney injury that has progressed to CKD. Stage 3a CKD. Close monitoring and nephrology follow-up scheduled.",
    "Patient with lupus nephritis showing signs of CKD progression. Stage 3 CKD. Immunosuppressive therapy adjusted. Rheumatology and nephrology coordination ongoing.",
    "Patient with chronic glomerulonephritis. CKD stage 4. Preparing for dialysis. Vascular access evaluation scheduled. Dietary restrictions discussed."
]

# Fixed real prescription instructions for CKD medications
CKD_PRESCRIPTION_INSTRUCTIONS = {
    "Lisinopril": "Take 10mg once daily in the morning. May cause dry cough. Monitor blood pressure and kidney function. Avoid potassium supplements unless prescribed.",
    "Losartan": "Take 50mg once daily. Helps protect kidney function and control blood pressure. Drink plenty of water. Report any dizziness or swelling.",
    "Furosemide": "Take 40mg once or twice daily as directed. Take in the morning to avoid nighttime urination. Monitor for dehydration and electrolyte imbalance.",
    "Spironolactone": "Take 25mg once daily. Do not take if potassium levels are high. Avoid potassium-rich foods. Regular blood tests required.",
    "Calcium Carbonate": "Take 500mg with meals to bind phosphate. Take with food to reduce stomach upset. Do not take within 2 hours of other medications.",
    "Sevelamer": "Take 800mg three times daily with meals. Helps control phosphate levels. Swallow whole, do not crush. May cause constipation.",
    "Calcitriol": "Take 0.25mcg once daily. Helps manage bone health in CKD. Take with food. Regular monitoring of calcium and phosphate levels required.",
    "Erythropoietin": "Injection administered at clinic/hospital. 2000 units subcutaneously weekly. Helps treat anemia in CKD. Monitor hemoglobin levels.",
    "Iron Sucrose IV": "Intravenous infusion at hospital/clinic. 200mg over 2-3 hours. Treats iron deficiency anemia in CKD. Monitor for allergic reactions.",
    "Sodium Bicarbonate": "Take 650mg twice daily. Helps correct acidosis in advanced CKD. Take with meals. May cause bloating or gas.",
    "Cholecalciferol": "Take 1000 IU once daily. Vitamin D supplement for bone health. Take with food containing fat for better absorption.",
    "Amlodipine": "Take 5mg once daily for blood pressure control. May cause ankle swelling. Do not stop suddenly. Regular blood pressure monitoring."
}

# Chief complaints for CKD patients
CKD_CHIEF_COMPLAINTS = [
    "Routine check-up for kidney function monitoring",
    "Fatigue and decreased energy levels",
    "Swelling in legs and ankles",
    "Decreased urine output",
    "Foamy or bubbly urine",
    "High blood pressure concerns",
    "Follow-up after abnormal kidney function tests",
    "Nausea and loss of appetite",
    "Metallic taste in mouth",
    "Itching and dry skin",
    "Muscle cramps and weakness",
    "Shortness of breath",
    "Chest pain and fluid retention",
    "Difficulty sleeping due to restless legs",
    "Bone pain and fractures",
    "Follow-up for CKD management"
]

# CNIC generation (Pakistani format)
def generate_cnic() -> str:
    """Generate valid Pakistani CNIC format: 42101-5819341-7"""
    area_code = random.choice(["42101", "42102", "42103", "42104", "42105"])
    middle_part = f"{random.randint(1000000, 9999999)}"
    last_digit = str(random.randint(0, 9))
    return f"{area_code}-{middle_part}-{last_digit}"

# Blood group generation
def generate_blood_group() -> str:
    """Generate blood group with realistic distribution"""
    blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    weights = [0.25, 0.05, 0.20, 0.04, 0.08, 0.02, 0.30, 0.06]
    return random.choices(blood_groups, weights=weights)[0]

# Pakistani phone number generation
def generate_phone() -> str:
    """Generate Pakistani phone number"""
    prefixes = ["0300", "0301", "0302", "0303", "0304", "0305", "0306", "0307", "0308", "0309",
                "0310", "0311", "0312", "0313", "0314", "0315", "0316", "0317", "0318", "0319",
                "0320", "0321", "0322", "0323", "0324", "0325", "0326", "0327", "0328", "0329",
                "0330", "0331", "0332", "0333", "0334", "0335", "0336", "0337", "0338", "0339",
                "0340", "0341", "0342", "0343", "0344", "0345", "0346", "0347", "0348", "0349",
                "0350", "0351", "0352", "0353", "0354", "0355", "0356", "0357", "0358", "0359"]
    prefix = random.choice(prefixes)
    number = f"{random.randint(1000000, 9999999)}"
    return f"{prefix}{number}"

# Lab names (Pakistani labs)
LAB_NAMES = [
    "Chughtai Lab", "Shaukat Khanum Lab", "Agha Khan Lab", "Dow Lab", "Jinnah Lab",
    "Services Lab", "Al-Shifa Lab", "Medicare Lab", "Pathology Lab", "Diagnostic Lab"
]

# Doctor specializations relevant to CKD
DOCTOR_SPECIALIZATIONS = [
    "Nephrologist", "General Physician", "Internal Medicine", "Cardiologist",
    "Endocrinologist", "Family Medicine", "Urologist", "Hypertension Specialist"
]

# Medication names for CKD
CKD_MEDICATIONS = [
    "Lisinopril", "Losartan", "Furosemide", "Spironolactone", "Calcium Carbonate",
    "Sevelamer", "Calcitriol", "Erythropoietin", "Iron Sucrose IV", "Sodium Bicarbonate",
    "Cholecalciferol", "Amlodipine"
]

# Common symptoms of CKD
CKD_SYMPTOMS = [
    "Fatigue", "Weakness", "Swelling in legs", "Decreased urine output", "Foamy urine",
    "High blood pressure", "Nausea", "Loss of appetite", "Metallic taste", "Itching",
    "Muscle cramps", "Shortness of breath", "Chest pain", "Restless legs", "Bone pain",
    "Difficulty concentrating", "Sleep problems", "Dry skin", "Headaches", "Dizziness"
]

def generate_ckd_data(names_list: List[Dict[str, str]], num_patients: int = None) -> Dict[str, Any]:
    """
    Generate comprehensive healthcare data for Chronic Kidney Disease (CKD)
    Creates realistic, unbiased data for ML model training
    """
    if num_patients is None:
        num_patients = len(names_list)  # Use ALL names from JSON file
    
    print(f"Generating CKD data for {num_patients} patients...")
    
    # Lab value ranges for CKD (realistic clinical ranges)
    ckd_ranges = {
        "serum_creatinine": (0.5, 8.0),          # mg/dL (normal: 0.6-1.2, CKD: up to 8+)
        "egfr": (8, 120),                        # mL/min/1.73 m² (normal: >90, CKD: <90)
        "uacr": (5, 3000),                       # mg/g (normal: <30, CKD: >30)
        "bun": (5, 120),                         # mg/dL (normal: 7-20, CKD: elevated)
        "sodium": (125, 145),                    # mEq/L (normal: 136-145)
        "potassium": (3.0, 6.5),                 # mEq/L (normal: 3.5-5.0)
        "calcium": (7.0, 11.0),                  # mg/dL (normal: 8.5-10.5)
        "phosphorus": (2.0, 8.0),                # mg/dL (normal: 2.5-4.5)
        "hemoglobin": (7.0, 16.0),               # g/dL (anemia common in CKD)
        "pth": (10, 800),                         # pg/mL (normal: 10-65, elevated in CKD)
        "bicarbonate": (15, 30),                 # mEq/L (normal: 22-28, low in advanced CKD)
        "albumin": (2.5, 5.0),                    # g/dL (normal: 3.5-5.0)
        "bmi": (18, 40),
        "systolic_bp": (90, 180),
        "diastolic_bp": (50, 120)
    }

    def generate_trend_v2(base_value, visits, feature_name, direction, rate):
        """Generate trend for lab values over visits"""
        values = [base_value]
        for _ in range(1, visits):
            noise = random.gauss(0, 0.05 * base_value)
            change = (base_value * 0.05 * rate) + noise
            if direction == 'improving': 
                # For CKD: decreasing creatinine, increasing eGFR is improvement
                if feature_name in ['serum_creatinine', 'bun', 'uacr', 'phosphorus', 'pth']:
                    base_value -= abs(change)  # Decrease is improvement
                else:
                    base_value += abs(change)  # Increase is improvement
            elif direction == 'worsening': 
                # For CKD: increasing creatinine, decreasing eGFR is worsening
                if feature_name in ['serum_creatinine', 'bun', 'uacr', 'phosphorus', 'pth']:
                    base_value += abs(change)  # Increase is worsening
                else:
                    base_value -= abs(change)  # Decrease is worsening
            else: 
                base_value += random.choice([-1, 1]) * noise
            min_val, max_val = ckd_ranges[feature_name]
            base_value = max(min_val, min(max_val, base_value))
            values.append(round(base_value, 2))
        return values
        
    def calculate_egfr_from_creatinine(creatinine, age, gender):
        """Calculate eGFR using simplified MDRD formula"""
        # Simplified MDRD: eGFR = 175 × (Scr)^-1.154 × (Age)^-0.203 × (0.742 if female)
        gender_factor = 0.742 if gender == 'female' else 1.0
        egfr = 175 * (creatinine ** -1.154) * (age ** -0.203) * gender_factor
        return max(8, min(120, egfr))  # Clamp to realistic range
    
    def determine_ckd_stage(egfr, uacr):
        """Determine CKD stage based on eGFR and UACR (KDIGO classification)"""
        if egfr >= 90 and uacr < 30:
            return "Normal"
        elif egfr >= 90 and uacr >= 30:
            return "CKD Stage 1"
        elif egfr >= 60 and uacr >= 30:
            return "CKD Stage 2"
        elif egfr >= 45:
            return "CKD Stage 3a"
        elif egfr >= 30:
            return "CKD Stage 3b"
        elif egfr >= 15:
            return "CKD Stage 4"
        else:
            return "CKD Stage 5 (ESRD)"
    
    def determine_ckd_diagnosis(egfr, uacr, creatinine):
        """Determine detailed CKD diagnosis"""
        stage = determine_ckd_stage(egfr, uacr)
        if stage == "Normal":
            return "Normal Kidney Function"
        elif "Stage 1" in stage or "Stage 2" in stage:
            return f"Early {stage}"
        elif "Stage 3a" in stage or "Stage 3b" in stage:
            return f"Moderate {stage}"
        elif "Stage 4" in stage:
            return f"Advanced {stage}"
        else:
            return f"End Stage Renal Disease (ESRD)"
    
    # Progression outcomes for CKD (added Normal for better balance)
    progression_outcomes = ['Normal', 'Stable', 'Slowly Progressing', 'Rapidly Progressing', 'Improving', 'ESRD']
    
    # Initialize data containers
    patients_data = []
    doctors_data = []
    labs_data = []
    family_relationships_data = []
    family_disease_history_data = []
    doctor_visits_data = []
    symptoms_data = []
    diagnoses_data = []
    prescriptions_data = []
    lab_reports_data = []
    lab_test_results_data = []
    disease_progressions_data = []
    
    # Generate doctors as patients (5-10 doctors)
    num_doctors = random.randint(5, 10)
    doctor_patient_ids = []
    for i in range(num_doctors):
        doctor_patient_id = str(uuid.uuid4())
        doctor_patient_ids.append(doctor_patient_id)
        
        age = random.randint(30, 65)
        birth_date = fake.date_of_birth(minimum_age=age, maximum_age=age)
        
        full_name = fake.name().split()
        first_name = full_name[0] if len(full_name) > 0 else "Dr."
        last_name = " ".join(full_name[1:]) if len(full_name) > 1 else fake.last_name()
        
        doctor_patient = {
            "patient_id": doctor_patient_id,
            "cnic": generate_cnic(),
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": birth_date,
            "gender": random.choice(["male", "female"]),
            "blood_group": generate_blood_group(),
            "phone": generate_phone(),
            "email": fake.email(),
            "address": fake.address(),
            "created_at": fake.date_time_between(start_date='-2y', end_date='now'),
            "updated_at": fake.date_time_between(start_date='-1y', end_date='now'),
            "is_doctor": True,
            "specialization": random.choice(DOCTOR_SPECIALIZATIONS),
            "license_number": f"PMDC-{random.randint(10000, 99999)}",
            "hospital_affiliation": fake.company()
        }
        patients_data.append(doctor_patient)
        
        doctors_data.append({
            "doctor_id": doctor_patient_id,
            "patient_id": doctor_patient_id,
            "name": f"{first_name} {last_name}",
            "specialization": doctor_patient["specialization"],
            "license_number": doctor_patient["license_number"],
            "phone": doctor_patient["phone"],
            "email": doctor_patient["email"],
            "hospital_affiliation": doctor_patient["hospital_affiliation"],
            "created_at": doctor_patient["created_at"],
            "updated_at": doctor_patient["updated_at"]
        })
    
    # Generate labs (3-5 labs)
    num_labs = random.randint(3, 5)
    for i in range(num_labs):
        lab = {
            "lab_id": str(uuid.uuid4()),
            "lab_name": random.choice(LAB_NAMES),
            "lab_location": fake.city() + ", Pakistan",
            "accreditation_number": f"PAL-{random.randint(1000, 9999)}",
            "phone": generate_phone(),
            "email": fake.email(),
            "created_at": fake.date_time_between(start_date='-2y', end_date='now')
        }
        labs_data.append(lab)
    
    # Generate patients and all related data
    for i in range(num_patients):
        person = names_list[i % len(names_list)]
        
        print(f"Processing patient {i+1}/{num_patients} - {person['first_name']} {person['last_name']}")
        
        patient_id = str(uuid.uuid4())
        
        age = random.randint(25, 85)
        birth_date = fake.date_of_birth(minimum_age=age, maximum_age=age)
        
        patient = {
            "patient_id": patient_id,
            "cnic": generate_cnic(),
            "first_name": person['first_name'],
            "last_name": person['last_name'],
            "date_of_birth": birth_date,
            "gender": person['gender'].lower(),
            "blood_group": generate_blood_group(),
            "phone": generate_phone(),
            "email": fake.email(),
            "address": fake.address(),
            "created_at": fake.date_time_between(start_date='-2y', end_date='now'),
            "updated_at": fake.date_time_between(start_date='-1y', end_date='now')
        }
        patients_data.append(patient)
        
        # Generate family relationships (20% chance)
        if random.random() < 0.2 and i > 0:
            relative_idx = random.randint(0, i - 1)
            relationship_type = random.choice(["parent", "child", "sibling", "spouse"])
            
            blood_relatives = ["parent", "child", "sibling", "grandparent", "grandchild", 
                              "aunt_uncle", "niece_nephew", "cousin"]
            is_blood_relative = relationship_type in blood_relatives
            
            relationship = {
                "id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "relative_patient_id": patients_data[relative_idx]["patient_id"],
                "relationship_type": relationship_type,
                "is_blood_relative": is_blood_relative,
                "created_at": fake.date_time_between(start_date='-1y', end_date='now')
            }
            family_relationships_data.append(relationship)
        
        # Generate family disease history (30% chance)
        if random.random() < 0.3:
            diseases = ["Diabetes", "Hypertension", "Kidney Disease", "Heart Disease", "Polycystic Kidney Disease"]
            disease = {
                "id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "disease_name": random.choice(diseases),
                "diagnosed_at": fake.date_between(start_date='-10y', end_date='now'),
                "severity": random.choice(["mild", "moderate", "severe"]),
                "notes": fake.text(max_nb_chars=200),
                "created_at": fake.date_time_between(start_date='-1y', end_date='now')
            }
            family_disease_history_data.append(disease)
        
        # Determine final progression outcome
        # Increase probability of Normal cases for better balance
        outcome_weights = [0.20, 0.18, 0.18, 0.15, 0.15, 0.14]  # Normal, Stable, Slow, Rapid, Improving, ESRD
        final_outcome = random.choices(progression_outcomes, weights=outcome_weights)[0]
        visit_count = random.randint(10, 25)
        
        # Generate base values based on final outcome
        base_values = {}
        
        if final_outcome == 'ESRD':
            # End stage: very high creatinine, very low eGFR
            base_values['serum_creatinine'] = random.uniform(5.0, 8.0)
            base_values['egfr'] = random.uniform(8, 15)
            base_values['uacr'] = random.uniform(500, 3000)
            base_values['bun'] = random.uniform(60, 120)
        elif final_outcome == 'Rapidly Progressing':
            # Rapid decline: high creatinine, low eGFR, trending worse
            base_values['serum_creatinine'] = random.uniform(2.5, 5.0)
            base_values['egfr'] = random.uniform(15, 30)
            base_values['uacr'] = random.uniform(200, 1000)
            base_values['bun'] = random.uniform(40, 80)
        elif final_outcome == 'Slowly Progressing':
            # Slow decline: moderate values
            base_values['serum_creatinine'] = random.uniform(1.5, 2.5)
            base_values['egfr'] = random.uniform(30, 60)
            base_values['uacr'] = random.uniform(100, 500)
            base_values['bun'] = random.uniform(25, 50)
        elif final_outcome == 'Stable':
            # Stable: moderate CKD, not changing much
            base_values['serum_creatinine'] = random.uniform(1.2, 2.0)
            base_values['egfr'] = random.uniform(45, 75)
            base_values['uacr'] = random.uniform(30, 200)
            base_values['bun'] = random.uniform(20, 40)
        elif final_outcome == 'Normal':
            # Normal: healthy kidney function
            base_values['serum_creatinine'] = random.uniform(0.6, 1.2)
            base_values['egfr'] = random.uniform(90, 120)
            base_values['uacr'] = random.uniform(5, 30)
            base_values['bun'] = random.uniform(7, 20)
        else:  # Improving
            # Improving: starting high, trending down
            base_values['serum_creatinine'] = random.uniform(1.8, 3.0)
            base_values['egfr'] = random.uniform(30, 50)
            base_values['uacr'] = random.uniform(150, 400)
            base_values['bun'] = random.uniform(30, 60)
        
        # Generate other base values
        for key in ['sodium', 'potassium', 'calcium', 'phosphorus', 'hemoglobin', 
                   'pth', 'bicarbonate', 'albumin', 'bmi', 'systolic_bp', 'diastolic_bp']:
            base_values[key] = random.uniform(*ckd_ranges[key])
        
        # Adjust values based on CKD stage
        if base_values['egfr'] < 30:
            # Advanced CKD: adjust electrolytes and minerals
            base_values['phosphorus'] = random.uniform(4.5, 8.0)  # Elevated
            base_values['pth'] = random.uniform(150, 800)  # Elevated
            base_values['hemoglobin'] = random.uniform(7.0, 11.0)  # Anemia
            base_values['bicarbonate'] = random.uniform(15, 22)  # Acidosis
        elif base_values['egfr'] < 60:
            # Moderate CKD
            base_values['phosphorus'] = random.uniform(3.5, 5.5)
            base_values['pth'] = random.uniform(50, 200)
            base_values['hemoglobin'] = random.uniform(10.0, 13.0)
            base_values['bicarbonate'] = random.uniform(20, 26)
        
        # Generate visits with trends
        start_date = fake.date_time_between(start_date='-2y', end_date='-6m')
        visit_dates = [start_date + timedelta(days=x*30) for x in range(visit_count)]
        
        for visit_num, visit_date in enumerate(visit_dates, 1):
            visit_id = str(uuid.uuid4())
            
            # Generate lab values with trends
            lab_values = {}
            for feature_name, base_value in base_values.items():
                direction = 'stable'
                rate = 1.0
                
                if final_outcome == 'Improving':
                    if feature_name in ['serum_creatinine', 'bun', 'uacr', 'phosphorus', 'pth']:
                        direction, rate = 'improving', random.uniform(1.2, 1.8)  # Add randomness
                    elif feature_name in ['egfr', 'hemoglobin', 'bicarbonate']:
                        direction, rate = 'improving', random.uniform(1.2, 1.8)
                elif final_outcome == 'Rapidly Progressing':
                    if feature_name in ['serum_creatinine', 'bun', 'uacr', 'phosphorus', 'pth']:
                        direction, rate = 'worsening', random.uniform(1.5, 2.5)  # Add randomness
                    elif feature_name in ['egfr', 'hemoglobin', 'bicarbonate']:
                        direction, rate = 'worsening', random.uniform(1.5, 2.5)
                elif final_outcome == 'Slowly Progressing':
                    if feature_name in ['serum_creatinine', 'bun', 'uacr', 'phosphorus', 'pth']:
                        direction, rate = 'worsening', random.uniform(0.3, 0.7)  # Add randomness
                    elif feature_name in ['egfr', 'hemoglobin', 'bicarbonate']:
                        direction, rate = 'worsening', random.uniform(0.3, 0.7)
                elif final_outcome == 'Stable':
                    direction, rate = 'stable', random.uniform(0.2, 0.4)  # Add randomness
                elif final_outcome == 'Normal':
                    direction, rate = 'stable', random.uniform(0.05, 0.15)  # Normal should stay stable
                
                trend_values = generate_trend_v2(base_value, visit_count, feature_name, direction, rate)
                lab_values[feature_name] = trend_values[visit_num - 1]
            
            # Recalculate eGFR from creatinine (more realistic)
            lab_values['egfr'] = calculate_egfr_from_creatinine(
                lab_values['serum_creatinine'], 
                age, 
                patient['gender']
            )
            
            # Determine diagnosis
            diagnosis = determine_ckd_diagnosis(
                lab_values['egfr'], 
                lab_values['uacr'],
                lab_values['serum_creatinine']
            )
            has_ckd = "Normal" not in diagnosis
            
            # Generate doctor visit
            doctor = random.choice(doctors_data)
            doctor_note = random.choice(CKD_DOCTOR_NOTES)
            chief_complaint = random.choice(CKD_CHIEF_COMPLAINTS)
            
            visit = {
                "visit_id": visit_id,
                "patient_id": patient_id,
                "doctor_patient_id": doctor["doctor_id"],
                "visit_date": visit_date,
                "chief_complaint": chief_complaint,
                "visit_type": random.choice(["consultation", "follow_up", "routine_checkup", "lab_review"]),
                "doctor_notes": doctor_note,
                "created_at": visit_date,
                "updated_at": visit_date
            }
            doctor_visits_data.append(visit)
            
            # Generate symptoms (0-5 symptoms per visit)
            num_symptoms = random.randint(0, 5)
            if has_ckd and num_symptoms > 0:
                symptom_list = random.sample(CKD_SYMPTOMS, min(num_symptoms, len(CKD_SYMPTOMS)))
            else:
                general_symptoms = ["Headache", "Fatigue", "Mild weakness", "General malaise"]
                symptom_list = random.sample(general_symptoms, min(num_symptoms, len(general_symptoms))) if num_symptoms > 0 else []
            
            for symptom_name in symptom_list:
                symptom = {
                    "id": str(uuid.uuid4()),
                    "visit_id": visit_id,
                    "symptom_name": symptom_name,
                    "severity": random.randint(1, 10),
                    "duration_days": random.randint(1, 90),
                    "notes": fake.text(max_nb_chars=100)
                }
                symptoms_data.append(symptom)
            
            # Generate diagnosis
            diagnosis_record = {
                "diagnosis_id": str(uuid.uuid4()),
                "visit_id": visit_id,
                "disease_name": diagnosis,
                "diagnosis_date": visit_date,
                "confidence_score": random.uniform(0.8, 1.0) if has_ckd else random.uniform(0.6, 0.9),
                "ml_model_used": "xgb_ckd_v1" if has_ckd else None,
                "status": "confirmed" if has_ckd else "suspected",
                "notes": None,
                "created_at": visit_date
            }
            diagnoses_data.append(diagnosis_record)
            
            # Generate prescription (if CKD)
            if has_ckd and random.random() < 0.8:
                medication = random.choice(CKD_MEDICATIONS)
                fixed_instruction = CKD_PRESCRIPTION_INSTRUCTIONS.get(medication, "Take as directed by physician.")
                
                dosage_map = {
                    "Lisinopril": "10mg", "Losartan": "50mg", "Furosemide": "40mg",
                    "Spironolactone": "25mg", "Calcium Carbonate": "500mg",
                    "Sevelamer": "800mg", "Calcitriol": "0.25mcg", "Erythropoietin": "2000 units",
                    "Iron Sucrose IV": "200mg", "Sodium Bicarbonate": "650mg",
                    "Cholecalciferol": "1000 IU", "Amlodipine": "5mg"
                }
                
                prescription = {
                    "prescription_id": str(uuid.uuid4()),
                    "visit_id": visit_id,
                    "medication_name": medication,
                    "dosage": dosage_map.get(medication, "As directed"),
                    "frequency": "Once daily" if "IV" not in medication else "Weekly",
                    "duration_days": random.randint(30, 180),
                    "instructions": fixed_instruction,
                    "created_at": visit_date
                }
                prescriptions_data.append(prescription)
            
            # Generate lab report
            lab = random.choice(labs_data)
            lab_report = {
                "report_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "lab_id": lab["lab_id"],
                "visit_id": visit_id,
                "report_date": visit_date + timedelta(days=random.randint(1, 3)),
                "report_type": "comprehensive_metabolic_panel_ckd",
                "status": "completed",
                "pdf_url": f"https://lab-reports.com/report_{visit_id}.pdf",
                "created_at": visit_date,
                "updated_at": visit_date
            }
            lab_reports_data.append(lab_report)
            
            # Generate lab test results
            for test_name, test_value in lab_values.items():
                # Define reference ranges
                if patient['gender'] == 'male':
                    ref_ranges = {
                        "serum_creatinine": (0.6, 1.2), "egfr": (90, 120), "uacr": (0, 30),
                        "bun": (7, 20), "sodium": (136, 145), "potassium": (3.5, 5.0),
                        "calcium": (8.5, 10.5), "phosphorus": (2.5, 4.5),
                        "hemoglobin": (13.5, 17.5), "pth": (10, 65), "bicarbonate": (22, 28),
                        "albumin": (3.5, 5.0), "bmi": (18.5, 24.9),
                        "systolic_bp": (90, 120), "diastolic_bp": (60, 80)
                    }
                else:  # female
                    ref_ranges = {
                        "serum_creatinine": (0.5, 1.1), "egfr": (90, 120), "uacr": (0, 30),
                        "bun": (7, 20), "sodium": (136, 145), "potassium": (3.5, 5.0),
                        "calcium": (8.5, 10.5), "phosphorus": (2.5, 4.5),
                        "hemoglobin": (12.0, 15.5), "pth": (10, 65), "bicarbonate": (22, 28),
                        "albumin": (3.5, 5.0), "bmi": (18.5, 24.9),
                        "systolic_bp": (90, 120), "diastolic_bp": (60, 80)
                    }
                
                ref_min, ref_max = ref_ranges.get(test_name, (0, 100))
                is_abnormal = test_value < ref_min or test_value > ref_max
                
                # Determine units
                unit_map = {
                    "serum_creatinine": "mg/dL", "egfr": "mL/min/1.73 m²", "uacr": "mg/g",
                    "bun": "mg/dL", "sodium": "mEq/L", "potassium": "mEq/L",
                    "calcium": "mg/dL", "phosphorus": "mg/dL", "hemoglobin": "g/dL",
                    "pth": "pg/mL", "bicarbonate": "mEq/L", "albumin": "g/dL",
                    "bmi": "kg/m²", "systolic_bp": "mmHg", "diastolic_bp": "mmHg"
                }
                
                test_result = {
                    "result_id": str(uuid.uuid4()),
                    "report_id": lab_report["report_id"],
                    "test_name": test_name,
                    "test_value": test_value,
                    "unit": unit_map.get(test_name, ""),
                    "reference_range_min": ref_min,
                    "reference_range_max": ref_max,
                    "is_abnormal": is_abnormal,
                    "created_at": visit_date
                }
                lab_test_results_data.append(test_result)
            
            # Generate disease progression (once per patient, at the end)
            if visit_num == visit_count:
                progression = {
                    "progression_id": str(uuid.uuid4()),
                    "patient_id": patient_id,
                    "disease_name": "chronic_kidney_disease",
                    "progression_stage": final_outcome,
                    "assessed_date": visit_date,
                    "ml_model_used": "lstm_ckd_progression_v1",
                    "confidence_score": random.uniform(0.8, 1.0),
                    "notes": f"Final assessment: {final_outcome}",
                    "created_at": visit_date
                }
                disease_progressions_data.append(progression)
    
    print("Data generation complete!")
    
    return {
        "patients": patients_data,
        "doctors": doctors_data,
        "labs": labs_data,
        "family_relationships": family_relationships_data,
        "family_disease_history": family_disease_history_data,
        "doctor_visits": doctor_visits_data,
        "symptoms": symptoms_data,
        "diagnoses": diagnoses_data,
        "prescriptions": prescriptions_data,
        "lab_reports": lab_reports_data,
        "lab_test_results": lab_test_results_data,
        "disease_progressions": disease_progressions_data,
        # Store CKD-specific data for ML training
        "ckd_training_data": {
            "features": list(ckd_ranges.keys()),
            "ranges": ckd_ranges,
            "progression_outcomes": progression_outcomes
        }
    }

if __name__ == "__main__":
    # Load names from JSON
    names_list = load_names_from_json("data generation/names.json")
    
    # Generate CKD data - use ALL names from JSON file
    data = generate_ckd_data(names_list)
    
    # Save to JSON files with "ckd_" prefix
    print("\nSaving generated data to JSON files...")
    
    file_mappings = {
        "patients": "generated_ckd_patients.json",
        "doctors": "generated_ckd_doctors.json",
        "labs": "generated_ckd_labs.json",
        "family_relationships": "generated_ckd_family_relationships.json",
        "family_disease_history": "generated_ckd_family_disease_history.json",
        "doctor_visits": "generated_ckd_doctor_visits.json",
        "symptoms": "generated_ckd_symptoms.json",
        "diagnoses": "generated_ckd_diagnoses.json",
        "prescriptions": "generated_ckd_prescriptions.json",
        "lab_reports": "generated_ckd_lab_reports.json",
        "lab_test_results": "generated_ckd_lab_test_results.json",
        "disease_progressions": "generated_ckd_disease_progressions.json"
    }
    
    for key, filename in file_mappings.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data[key], f, indent=2, default=str, ensure_ascii=False)
        print(f"✅ Saved {len(data[key])} records to {filename}")
    
    print("\n✅ CKD data generation complete!")
    print(f"\nSummary:")
    print(f"  - Patients: {len(data['patients'])}")
    print(f"  - Doctors: {len(data['doctors'])}")
    print(f"  - Labs: {len(data['labs'])}")
    print(f"  - Visits: {len(data['doctor_visits'])}")
    print(f"  - Lab Test Results: {len(data['lab_test_results'])}")
    print(f"  - Progressions: {len(data['disease_progressions'])}")
    print(f"\nFeatures for ML training: {data['ckd_training_data']['features']}")
    print(f"Progression outcomes: {data['ckd_training_data']['progression_outcomes']}")

