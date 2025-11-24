"""
Iron Deficiency Anemia Data Generation for LifeChain AI Healthcare System
Uses fixed real prescriptions and doctor notes (no AI generation)
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

# Fixed real doctor notes templates
ANEMIA_DOCTOR_NOTES = [
    "Patient presents with fatigue and pallor. Physical examination reveals pale conjunctiva and koilonychia. Blood work ordered to assess iron levels and hemoglobin status.",
    "Follow-up visit for iron deficiency anemia. Patient reports improved energy levels. Compliance with iron supplementation noted. Continue current treatment regimen.",
    "Patient complains of persistent weakness and shortness of breath on exertion. Clinical examination shows pale mucous membranes. Recommend iron studies and complete blood count.",
    "Routine check-up for anemia management. Patient shows good response to oral iron therapy. Hemoglobin levels trending upward. Advised to continue iron supplements with vitamin C.",
    "Patient presents with brittle nails and hair loss. History of heavy menstrual bleeding noted. Iron deficiency anemia suspected. Prescribed iron supplementation and dietary modifications.",
    "Laboratory results confirm microcytic hypochromic anemia. Serum ferritin low. Recommend oral ferrous sulfate 325mg daily with orange juice to enhance absorption.",
    "Patient reports difficulty concentrating and cold extremities. Physical exam reveals tachycardia and pale nail beds. Iron deficiency anemia diagnosed. Treatment initiated.",
    "Follow-up after 3 months of iron therapy. Patient feeling better with increased energy. Lab results show improvement in hemoglobin and ferritin levels. Continue treatment for 3 more months.",
    "Patient with history of GI bleeding now presenting with anemia symptoms. Stool occult blood test positive. Iron replacement therapy started along with gastroenterology referral.",
    "Young female patient with heavy menstrual periods and fatigue. Iron studies reveal low serum iron and ferritin. Started on oral iron supplementation and gynecology consultation advised.",
    "Elderly patient with chronic kidney disease presenting with anemia. EPO levels assessed. Iron supplementation initiated along with erythropoietin therapy consideration.",
    "Patient shows signs of pica (ice craving) and restless leg syndrome. Blood work confirms severe iron deficiency. Initiated aggressive oral iron replacement therapy.",
    "Post-partum patient presenting with extreme fatigue and dizziness. Hemoglobin 8.5 g/dL. Iron deficiency anemia confirmed. Started on high-dose iron supplements.",
    "Patient with celiac disease presenting with refractory anemia. Malabsorption suspected. Considering IV iron therapy due to poor oral absorption.",
    "Vegetarian patient with inadequate iron intake. Dietary counseling provided emphasizing iron-rich plant foods and vitamin C sources. Oral iron supplements prescribed."
]

# Fixed real prescription instructions for anemia medications
ANEMIA_PRESCRIPTION_INSTRUCTIONS = {
    "Ferrous Sulfate": "Take one 325mg tablet daily on empty stomach with orange juice. Avoid dairy products, tea, or coffee within 2 hours of taking. May cause dark stools (normal). Continue for at least 3-6 months even after symptoms improve.",
    "Ferrous Gluconate": "Take one 300mg tablet twice daily with meals to reduce GI upset. Take with vitamin C for better absorption. Space apart from calcium supplements and antacids by 2 hours.",
    "Ferrous Fumarate": "Take one 200mg tablet daily in the morning on empty stomach. If stomach upset occurs, take with small amount of food. Store in cool, dry place away from children.",
    "Iron Polymaltose Complex": "Take one 100mg tablet daily with or without food. Well tolerated with minimal GI side effects. Complete the full course as prescribed even if feeling better.",
    "IV Iron Sucrose": "Administered intravenously at hospital/clinic. 200mg infusion over 2-3 hours. Monitor for allergic reactions. Schedule follow-up lab work in 4 weeks.",
    "Folic Acid": "Take 5mg tablet once daily. Essential for red blood cell production. Continue throughout pregnancy if applicable. May be taken with or without food.",
    "Vitamin B12": "Take 1000mcg tablet once daily or as directed. Sublingual form available for better absorption. Important for nerve function and RBC formation.",
    "Vitamin C": "Take 500mg tablet with iron supplement to enhance iron absorption. Do not exceed 2000mg daily. Helps convert iron to absorbable form.",
    "Feroglobin Capsules": "Take one capsule daily with water, preferably with main meal. Contains iron, folic acid, and vitamin B12. Complete multivitamin for anemia management.",
    "Iron Dextran": "Intramuscular or IV injection administered by healthcare provider. Test dose required before full dose. Monitor closely for anaphylactic reactions."
}

# Chief complaints for anemia patients
ANEMIA_CHIEF_COMPLAINTS = [
    "Persistent fatigue and weakness for the past 2 months",
    "Shortness of breath on minimal exertion",
    "Dizziness and lightheadedness when standing",
    "Unusual ice cravings (pagophagia)",
    "Brittle nails and hair loss",
    "Pale skin and cold hands and feet",
    "Heart palpitations and chest discomfort",
    "Difficulty concentrating and memory problems",
    "Heavy menstrual bleeding and fatigue",
    "Sore tongue and mouth ulcers",
    "Restless leg syndrome disturbing sleep",
    "Headaches and ringing in ears",
    "Loss of appetite and weight loss",
    "Routine check-up for anemia management",
    "Follow-up after starting iron supplements"
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

# Doctor specializations relevant to anemia
DOCTOR_SPECIALIZATIONS = [
    "Hematologist", "General Physician", "Internal Medicine", "Gastroenterologist",
    "Gynecologist", "Family Medicine", "Nephrologist", "Oncologist"
]

# Medication names for iron deficiency anemia
ANEMIA_MEDICATIONS = [
    "Ferrous Sulfate", "Ferrous Gluconate", "Ferrous Fumarate", "Iron Polymaltose Complex",
    "IV Iron Sucrose", "Folic Acid", "Vitamin B12", "Vitamin C", "Feroglobin Capsules", "Iron Dextran"
]

# Common symptoms of iron deficiency anemia
ANEMIA_SYMPTOMS = [
    "Extreme fatigue", "Weakness", "Pale skin", "Shortness of breath", "Dizziness",
    "Cold hands and feet", "Brittle nails", "Headaches", "Fast heartbeat", "Sore tongue",
    "Unusual cravings for ice or dirt (pica)", "Poor appetite", "Restless leg syndrome",
    "Hair loss", "Difficulty concentrating"
]

def generate_anemia_data(names_list: List[Dict[str, str]], num_patients: int = None) -> Dict[str, Any]:
    """
    Generate comprehensive healthcare data for Iron Deficiency Anemia
    Uses fixed real prescriptions and doctor notes
    """
    if num_patients is None:
        num_patients = len(names_list)  # Use ALL names from JSON file
    
    print(f"Generating Iron Deficiency Anemia data for {num_patients} patients...")
    
    # Lab value ranges for iron deficiency anemia
    anemia_ranges = {
        "hemoglobin": (6.0, 18.0),           # g/dL (normal: M 13.5-17.5, F 12-15.5)
        "hematocrit": (20, 55),               # % (normal: M 38-50, F 36-44)
        "mcv": (60, 100),                     # fL (normal: 80-100)
        "mch": (20, 35),                      # pg (normal: 27-33)
        "mchc": (28, 36),                     # g/dL (normal: 32-36)
        "rdw": (11, 20),                      # % (normal: 11.5-14.5)
        "serum_iron": (10, 180),              # μg/dL (normal: 60-170)
        "ferritin": (5, 300),                 # ng/mL (normal: M 24-336, F 11-307)
        "tibc": (250, 500),                   # μg/dL (normal: 250-450)
        "transferrin_saturation": (5, 50),    # % (normal: 20-50)
        "reticulocyte_count": (0.2, 2.5),     # % (normal: 0.5-2.5)
        "wbc": (4000, 12000),                 # cells/μL
        "platelet_count": (150000, 450000),   # cells/μL
        "esr": (0, 40),                       # mm/hr
        "bmi": (16, 45),
        "systolic_bp": (90, 160),
        "diastolic_bp": (50, 100)
    }

    def generate_trend_v2(base_value, visits, feature_name, direction, rate):
        """Generate trend for lab values over visits"""
        values = [base_value]
        for _ in range(1, visits):
            noise = random.gauss(0, 0.05 * base_value)
            change = (base_value * 0.05 * rate) + noise
            if direction == 'improving': 
                base_value += change  # For anemia, increasing hemoglobin/ferritin is improvement
            elif direction == 'worsening': 
                base_value -= change  # Decreasing is worsening
            else: 
                base_value += random.choice([-1, 1]) * noise
            min_val, max_val = anemia_ranges[feature_name]
            base_value = max(min_val, min(max_val, base_value))
            values.append(round(base_value, 2))
        return values
        
    def determine_anemia_diagnosis(hemoglobin, ferritin, mcv, gender):
        """Determine anemia diagnosis based on lab values"""
        # Normal hemoglobin: Male 13.5-17.5, Female 12-15.5
        normal_hb = 13.5 if gender == 'male' else 12.0
        mild_anemia = normal_hb - 2
        moderate_anemia = normal_hb - 4
        severe_anemia = normal_hb - 6
        
        if hemoglobin < severe_anemia or ferritin < 10:
            return "Severe Iron Deficiency Anemia"
        elif hemoglobin < moderate_anemia or ferritin < 20:
            return "Moderate Iron Deficiency Anemia"
        elif hemoglobin < mild_anemia or ferritin < 30:
            return "Mild Iron Deficiency Anemia"
        elif ferritin < 50 and mcv < 80:
            return "Iron Deficiency without Anemia"
        else:
            return "Normal"

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
    # Doctors must be created as patients first with all patient fields
    num_doctors = random.randint(5, 10)
    doctor_patient_ids = []
    for i in range(num_doctors):
        doctor_patient_id = str(uuid.uuid4())
        doctor_patient_ids.append(doctor_patient_id)
        
        # Calculate age and birth date for doctor
        age = random.randint(30, 65)  # Doctors are typically 30-65 years old
        birth_date = fake.date_of_birth(minimum_age=age, maximum_age=age)
        
        # Split fake name into first and last name
        full_name = fake.name().split()
        first_name = full_name[0] if len(full_name) > 0 else "Dr."
        last_name = " ".join(full_name[1:]) if len(full_name) > 1 else fake.last_name()
        
        # Create doctor as a patient with doctor-specific fields
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
            # Doctor-specific fields
            "is_doctor": True,
            "specialization": random.choice(DOCTOR_SPECIALIZATIONS),
            "license_number": f"PMDC-{random.randint(10000, 99999)}",
            "hospital_affiliation": fake.company()
        }
        # Add to patients_data (doctors are patients)
        patients_data.append(doctor_patient)
        
        # Also keep in doctors_data for backward compatibility in visit generation
        doctors_data.append({
            "doctor_id": doctor_patient_id,  # Use patient_id as doctor_id
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
    
    # Progression outcomes for anemia
    progression_outcomes = ['Resolved', 'Improving', 'Stable', 'Worsening', 'Severe']
    
    # Generate patients and all related data
    for i in range(num_patients):
        person = names_list[i % len(names_list)]
        
        print(f"Processing patient {i+1}/{num_patients} - {person['first_name']} {person['last_name']}")
        
        patient_id = str(uuid.uuid4())
        
        # Calculate age and birth date
        age = random.randint(18, 80)
        birth_date = fake.date_of_birth(minimum_age=age, maximum_age=age)
        
        # Generate patient data
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
        
        # Generate family relationships (20% chance per patient)
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
            diseases = ["Iron Deficiency Anemia", "Thalassemia", "Celiac Disease", "GI Bleeding", "Heavy Menstrual Bleeding"]
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
        
        # Determine final outcome for this patient
        final_outcome = random.choice(progression_outcomes)
        visit_count = random.randint(8, 20)
        base_values = {k: random.uniform(*v) for k, v in anemia_ranges.items()}
        
        # Set initial values based on outcome
        if final_outcome in ['Resolved', 'Improving']:
            # Start with low values that will improve
            base_values['hemoglobin'] = random.uniform(7.0, 10.0)
            base_values['ferritin'] = random.uniform(5, 20)
            base_values['mcv'] = random.uniform(65, 75)
            base_values['serum_iron'] = random.uniform(15, 40)
        elif final_outcome in ['Worsening', 'Severe']:
            # Start moderate and worsen
            base_values['hemoglobin'] = random.uniform(8.0, 11.0)
            base_values['ferritin'] = random.uniform(10, 30)
            base_values['mcv'] = random.uniform(70, 80)
            base_values['serum_iron'] = random.uniform(20, 50)
        else:  # Stable
            # Maintain borderline values
            base_values['hemoglobin'] = random.uniform(10.0, 12.0)
            base_values['ferritin'] = random.uniform(20, 40)
            base_values['mcv'] = random.uniform(75, 85)
            base_values['serum_iron'] = random.uniform(40, 70)
        
        # Generate visits and all related data
        for visit_num in range(1, visit_count + 1):
            visit_id = str(uuid.uuid4())
            visit_date = fake.date_time_between(start_date='-2y', end_date='now')
            
            # Generate anemia lab values with trends
            lab_values = {}
            for k, v in base_values.items():
                direction, rate = 'stable', 1.0
                
                if final_outcome == 'Resolved':
                    if k in ['hemoglobin', 'ferritin', 'serum_iron', 'mcv']: 
                        direction, rate = 'improving', 2.0
                    elif k in ['tibc', 'rdw']: 
                        direction, rate = 'worsening', 1.0  # These normalize
                elif final_outcome == 'Improving':
                    if k in ['hemoglobin', 'ferritin', 'serum_iron']: 
                        direction, rate = 'improving', 1.0
                elif final_outcome == 'Worsening':
                    if k in ['hemoglobin', 'ferritin', 'serum_iron', 'mcv']: 
                        direction, rate = 'worsening', 1.5
                    elif k in ['tibc', 'rdw']: 
                        direction, rate = 'improving', 1.5  # These worsen by increasing
                elif final_outcome == 'Severe':
                    if k in ['hemoglobin', 'ferritin']: 
                        direction, rate = 'worsening', 2.0
                
                trend_values = generate_trend_v2(v, visit_count, feature_name=k, direction=direction, rate=rate)
                lab_values[k] = trend_values[visit_num - 1]
            
            # Determine diagnosis
            diagnosis = determine_anemia_diagnosis(
                lab_values['hemoglobin'], 
                lab_values['ferritin'], 
                lab_values['mcv'],
                patient['gender']
            )
            has_anemia = "Anemia" in diagnosis
            
            # Generate doctor visit with fixed real notes
            doctor = random.choice(doctors_data)
            doctor_note = random.choice(ANEMIA_DOCTOR_NOTES)
            chief_complaint = random.choice(ANEMIA_CHIEF_COMPLAINTS)
            
            visit = {
                "visit_id": visit_id,
                "patient_id": patient_id,
                "doctor_patient_id": doctor["doctor_id"],  # Use doctor_patient_id instead of doctor_id
                "visit_date": visit_date,
                "chief_complaint": chief_complaint,
                "visit_type": random.choice(["consultation", "follow_up", "routine_checkup", "lab_review"]),
                "doctor_notes": doctor_note,
                "created_at": visit_date,
                "updated_at": visit_date
            }
            doctor_visits_data.append(visit)
            
            # Generate symptoms (0-4 symptoms per visit)
            num_symptoms = random.randint(0, 4)
            if has_anemia and num_symptoms > 0:
                symptom_list = random.sample(ANEMIA_SYMPTOMS, min(num_symptoms, len(ANEMIA_SYMPTOMS)))
            else:
                symptom_list = random.sample(["Headache", "Fatigue", "Mild weakness", "General malaise"], num_symptoms)
            
            for symptom_name in symptom_list:
                symptom = {
                    "id": str(uuid.uuid4()),
                    "visit_id": visit_id,
                    "symptom_name": symptom_name,
                    "severity": random.randint(1, 10),
                    "duration_days": random.randint(1, 60),
                    "notes": fake.text(max_nb_chars=100)
                }
                symptoms_data.append(symptom)
            
            # Generate diagnosis
            diagnosis_record = {
                "diagnosis_id": str(uuid.uuid4()),
                "visit_id": visit_id,
                "disease_name": diagnosis,
                "diagnosis_date": visit_date,
                "confidence_score": random.uniform(0.8, 1.0) if has_anemia else random.uniform(0.6, 0.9),
                "ml_model_used": "xgb_anemia_v1" if has_anemia else None,
                "status": "confirmed" if has_anemia else "suspected",
                "notes": None,
                "created_at": visit_date
            }
            diagnoses_data.append(diagnosis_record)
            
            # Generate prescription (if anemia) with fixed real instructions
            if has_anemia and random.random() < 0.8:
                medication = random.choice(ANEMIA_MEDICATIONS)
                
                # Get fixed instruction for this medication
                fixed_instruction = ANEMIA_PRESCRIPTION_INSTRUCTIONS.get(medication, "Take as directed by physician.")
                
                # Determine appropriate dosage based on medication
                dosage_map = {
                    "Ferrous Sulfate": "325mg",
                    "Ferrous Gluconate": "300mg",
                    "Ferrous Fumarate": "200mg",
                    "Iron Polymaltose Complex": "100mg",
                    "IV Iron Sucrose": "200mg",
                    "Folic Acid": "5mg",
                    "Vitamin B12": "1000mcg",
                    "Vitamin C": "500mg",
                    "Feroglobin Capsules": "1 capsule",
                    "Iron Dextran": "100mg"
                }
                
                prescription = {
                    "prescription_id": str(uuid.uuid4()),
                    "visit_id": visit_id,
                    "medication_name": medication,
                    "dosage": dosage_map.get(medication, "As directed"),
                    "frequency": "Once daily" if "IV" in medication else random.choice(["Once daily", "Twice daily", "Three times daily"]),
                    "duration_days": random.randint(90, 180),
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
                "report_type": "complete_blood_count_iron_panel",
                "status": "completed",
                "pdf_url": f"https://lab-reports.com/report_{visit_id}.pdf",
                "created_at": visit_date,
                "updated_at": visit_date
            }
            lab_reports_data.append(lab_report)
            
            # Generate lab test results
            for test_name, test_value in lab_values.items():
                # Define reference ranges (gender-specific for some)
                if patient['gender'] == 'male':
                    ref_ranges = {
                        "hemoglobin": (13.5, 17.5), "hematocrit": (38, 50), "mcv": (80, 100),
                        "mch": (27, 33), "mchc": (32, 36), "rdw": (11.5, 14.5),
                        "serum_iron": (60, 170), "ferritin": (24, 336), "tibc": (250, 450),
                        "transferrin_saturation": (20, 50), "reticulocyte_count": (0.5, 2.5),
                        "wbc": (4000, 11000), "platelet_count": (150000, 400000), "esr": (0, 15),
                        "bmi": (18.5, 24.9), "systolic_bp": (90, 120), "diastolic_bp": (60, 80)
                    }
                else:  # female
                    ref_ranges = {
                        "hemoglobin": (12.0, 15.5), "hematocrit": (36, 44), "mcv": (80, 100),
                        "mch": (27, 33), "mchc": (32, 36), "rdw": (11.5, 14.5),
                        "serum_iron": (60, 170), "ferritin": (11, 307), "tibc": (250, 450),
                        "transferrin_saturation": (20, 50), "reticulocyte_count": (0.5, 2.5),
                        "wbc": (4000, 11000), "platelet_count": (150000, 400000), "esr": (0, 20),
                        "bmi": (18.5, 24.9), "systolic_bp": (90, 120), "diastolic_bp": (60, 80)
                    }
                
                ref_min, ref_max = ref_ranges.get(test_name, (0, 100))
                is_abnormal = test_value < ref_min or test_value > ref_max
                
                # Determine units
                unit_map = {
                    "hemoglobin": "g/dL", "hematocrit": "%", "mcv": "fL",
                    "mch": "pg", "mchc": "g/dL", "rdw": "%",
                    "serum_iron": "μg/dL", "ferritin": "ng/mL", "tibc": "μg/dL",
                    "transferrin_saturation": "%", "reticulocyte_count": "%",
                    "wbc": "cells/μL", "platelet_count": "cells/μL", "esr": "mm/hr",
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
                    "disease_name": "iron_deficiency_anemia",
                    "progression_stage": final_outcome,
                    "assessed_date": visit_date,
                    "ml_model_used": "lstm_anemia_progression_v1",
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
        # Store anemia-specific data for ML training
        "anemia_training_data": {
            "features": list(anemia_ranges.keys()),
            "ranges": anemia_ranges,
            "progression_outcomes": progression_outcomes
        }
    }

if __name__ == "__main__":
    # Load names from JSON
    names_list = load_names_from_json("names.json")
    
    # Generate anemia data - use ALL names from JSON file
    data = generate_anemia_data(names_list)
    
    # Save to JSON files for inspection
    for table_name, table_data in data.items():
        if table_name != "anemia_training_data":
            with open(f"generated_anemia_{table_name}.json", "w") as f:
                json.dump(table_data, f, indent=2, default=str)
    
    print(f"\n=== Iron Deficiency Anemia Data Generation Summary ===")
    print(f"Generated data for {len(data['patients'])} patients")
    print(f"Generated {len(data['doctor_visits'])} doctor visits")
    print(f"Generated {len(data['lab_reports'])} lab reports")
    print(f"Generated {len(data['symptoms'])} symptoms")
    print(f"Generated {len(data['prescriptions'])} prescriptions")
    print(f"Generated {len(data['lab_test_results'])} lab test results")
    print(f"Generated {len(data['diagnoses'])} diagnoses")
    print(f"Generated {len(data['disease_progressions'])} disease progressions")
    print(f"\nAll data saved with 'anemia_' prefix")

