import pandas as pd
import numpy as np
import random

def generate_data(names_list):
    """
    Generates a high-quality synthetic dataset and prints a summary.
    """
    num_patients = len(names_list)
    print(f"Generating data for {num_patients} patients...")
    
    ranges = {
        "fasting_glucose": (60, 200), "random_glucose": (80, 300), "postprandial_glucose": (90, 350),
        "hba1c": (4.0, 14.0), "hdl": (20, 80), "ldl": (50, 220), "triglycerides": (50, 400),
        "total_cholesterol": (100, 320), "creatinine": (0.4, 2.0), "urea": (10, 70),
        "microalbumin": (0, 350), "alt": (7, 80), "ast": (7, 80), "insulin": (1, 30),
        "bmi": (16, 45), "systolic_bp": (90, 200), "diastolic_bp": (50, 120)
    }

    def generate_trend_v2(base_value, visits, feature_name, direction, rate):
        values = [base_value]
        for _ in range(1, visits):
            noise = random.gauss(0, 0.05 * base_value)
            change = (base_value * 0.05 * rate) + noise
            if direction == 'improving': base_value -= change
            elif direction == 'worsening': base_value += change
            else: base_value += random.choice([-1, 1]) * noise
            min_val, max_val = ranges[feature_name]
            base_value = max(min_val, min(max_val, base_value))
            values.append(round(base_value, 2))
        return values
        
    def determine_diagnosis_realistic(hba1c, f_glucose, bmi):
        hba1c_pre, hba1c_dia = random.gauss(5.7, 0.2), random.gauss(6.5, 0.3)
        fg_pre, fg_dia = random.gauss(100, 5), random.gauss(126, 10)
        score = 0
        if hba1c > hba1c_dia or f_glucose > fg_dia: score += 2
        elif hba1c > hba1c_pre or f_glucose > fg_pre: score += 1
        if bmi > 30: score += 0.5
        if bmi > 35: score += 0.5
        if score >= 2.0: return "Diabetes"
        elif score >= 1.0: return "Prediabetes"
        else: return "Normal"

    all_records = []
    progression_outcomes = ['Complicated', 'Controlled', 'Cured', 'Diabetes', 'Normal']

    for pid, person in enumerate(names_list, start=1):
        final_outcome = random.choice(progression_outcomes)
        visit_count = random.randint(10, 25)
        base_values = {k: random.uniform(*v) for k, v in ranges.items()}
        
        if final_outcome in ['Cured', 'Controlled', 'Diabetes', 'Complicated']:
            base_values['hba1c'] = random.uniform(7.5, 12.0)
            base_values['fasting_glucose'] = random.uniform(140, 190)
        else:
            base_values['hba1c'] = random.uniform(4.5, 5.5)
            base_values['fasting_glucose'] = random.uniform(70, 95)

        for visit in range(1, visit_count + 1):
            rec = { "patient_id": pid, "name": f"{person['first_name']} {person['last_name']}", "gender": person["gender"], "visit_no": visit }
            for k, v in base_values.items():
                direction, rate = 'stable', 1.0
                if final_outcome == 'Cured':
                    if k in ['hba1c', 'fasting_glucose']: direction, rate = 'improving', 2.0
                elif final_outcome == 'Controlled':
                    if k in ['hba1c', 'fasting_glucose']: direction, rate = 'improving', 0.5
                elif final_outcome == 'Complicated':
                    if k in ['creatinine', 'microalbumin']: direction, rate = 'worsening', 1.5
                elif final_outcome == 'Diabetes':
                    if k in ['hba1c', 'fasting_glucose']: direction, rate = 'worsening', 0.2
                trend_values = generate_trend_v2(v, visit_count, feature_name=k, direction=direction, rate=rate)
                rec[k] = trend_values[visit - 1]
            
            rec['diagnosis'] = determine_diagnosis_realistic(rec['hba1c'], rec['fasting_glucose'], rec['bmi'])
            rec['progression'] = final_outcome
            all_records.append(rec)

    df = pd.DataFrame(all_records)
    
    # ⭐️ NEW: Print a detailed summary of the generated data
    print("\n--- Dataset Generation Complete ---")
    print(f"Total records (visits) created: {len(df)}")
    print(f"Total unique patients: {df['patient_id'].nunique()}")
    
    print("\nProgression (Final Outcome) Distribution:")
    print(df.groupby('patient_id')['progression'].first().value_counts())
    
    print("\nDiagnosis (Per-Visit) Distribution:")
    print(df['diagnosis'].value_counts())
    
    print("\nData Preview:")
    print(df.head())
    
    return df

# --- HOW TO USE ---
# Assume 'my_names_list' is your list of patient names. For example:
# my_names_list = [{'first_name': 'Patient', 'last_name': str(i), 'gender': 'Male'} for i in range(5000)]
# 
# Then, generate the DataFrame like this:
df = generate_data(names_list=names)