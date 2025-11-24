"""
Model Testing Script for LifeChain AI Healthcare System
Loads trained models and tests them on randomly selected data from lab reports
"""

import json
import random
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_inference import ModelInference, ProgressionBiLSTM

def load_test_data(num_samples: int = 50):
    """Load random samples from JSON files for testing"""
    print(f"Loading {num_samples} random test samples...")
    
    try:
        # Load lab test results
        with open('generated_lab_test_results.json', 'r') as f:
            lab_results = json.load(f)
        
        # Load doctor visits
        with open('generated_doctor_visits.json', 'r') as f:
            visits = json.load(f)
        
        # Load diagnoses
        with open('generated_diagnoses.json', 'r') as f:
            diagnoses = json.load(f)
        
        # Load disease progressions
        with open('generated_disease_progressions.json', 'r') as f:
            progressions = json.load(f)
        
        # Load patients
        with open('generated_patients.json', 'r') as f:
            patients = json.load(f)
        
        # Load lab reports
        with open('generated_lab_reports.json', 'r') as f:
            lab_reports = json.load(f)
        
        print(f"Loaded {len(lab_results)} lab test results")
        print(f"Loaded {len(visits)} doctor visits")
        print(f"Loaded {len(patients)} patients")
        
        # Create lookup dictionaries
        visits_dict = {v['visit_id']: v for v in visits}
        diagnoses_dict = {d['visit_id']: d for d in diagnoses}
        progressions_dict = {p['patient_id']: p for p in progressions}
        patients_dict = {p['patient_id']: p for p in patients}
        report_to_visit = {r['report_id']: r['visit_id'] for r in lab_reports if r.get('visit_id')}
        
        # Filter lab results for our target features
        target_features = [
            'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
            'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
        ]
        
        filtered_results = [r for r in lab_results if r['test_name'] in target_features]
        
        # Group by patient and visit to create test samples
        patient_visit_data = {}
        
        for result in filtered_results:
            report_id = result['report_id']
            if report_id in report_to_visit:
                visit_id = report_to_visit[report_id]
                if visit_id in visits_dict:
                    visit = visits_dict[visit_id]
                    patient_id = visit['patient_id']
                    
                    if patient_id not in patient_visit_data:
                        patient_visit_data[patient_id] = {}
                    
                    if visit_id not in patient_visit_data[patient_id]:
                        patient_visit_data[patient_id][visit_id] = {
                            'patient_id': patient_id,
                            'visit_id': visit_id,
                            'visit_date': visit['visit_date'],
                            'tests': {},
                            'patient_info': patients_dict.get(patient_id, {}),
                            'diagnosis': diagnoses_dict.get(visit_id, {}).get('disease_name', 'Normal'),
                            'progression': progressions_dict.get(patient_id, {}).get('progression_stage', 'Normal')
                        }
                    
                    patient_visit_data[patient_id][visit_id]['tests'][result['test_name']] = result['test_value']
        
        # Convert to list and filter complete samples
        test_samples = []
        for patient_id, visits in patient_visit_data.items():
            for visit_id, visit_data in visits.items():
                # Check if we have all required features
                if all(feature in visit_data['tests'] for feature in target_features):
                    test_samples.append(visit_data)
        
        print(f"Found {len(test_samples)} complete test samples")
        
        # Randomly select samples
        if len(test_samples) > num_samples:
            test_samples = random.sample(test_samples, num_samples)
        
        return test_samples, target_features
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise

def test_diagnosis_model(inference: ModelInference, test_samples: List[Dict], target_features: List[str]):
    """Test the diagnosis model on random samples"""
    print("\n" + "="*60)
    print("🔬 TESTING DIAGNOSIS MODEL")
    print("="*60)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, sample in enumerate(test_samples[:10]):  # Test first 10 samples
        print(f"\n--- Test Sample {i+1} ---")
        print(f"Patient: {sample['patient_info'].get('first_name', 'Unknown')} {sample['patient_info'].get('last_name', 'Unknown')}")
        print(f"Visit Date: {sample['visit_date']}")
        print(f"True Diagnosis: {sample['diagnosis']}")
        
        # Prepare test data
        test_data = {}
        for feature in target_features:
            test_data[feature] = sample['tests'].get(feature, 0.0)
        
        # Make prediction
        result = inference.predict_diagnosis(test_data)
        
        if 'error' in result:
            print(f"❌ Prediction Error: {result['error']}")
            continue
        
        predicted_class = result['predicted_class']
        confidence = result['prediction_confidence']
        
        print(f"Predicted Diagnosis: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Confidence Scores: {result['confidence_scores']}")
        
        # Check if prediction is correct
        if predicted_class == sample['diagnosis']:
            print("✅ CORRECT PREDICTION")
            correct_predictions += 1
        else:
            print("❌ INCORRECT PREDICTION")
        
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n📊 Diagnosis Model Test Results:")
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def test_progression_model(inference: ModelInference, test_samples: List[Dict], target_features: List[str]):
    """Test the progression model on random samples"""
    print("\n" + "="*60)
    print("🔬 TESTING PROGRESSION MODEL")
    print("="*60)
    
    # Group samples by patient for sequence testing
    patient_sequences = {}
    for sample in test_samples:
        patient_id = sample['patient_id']
        if patient_id not in patient_sequences:
            patient_sequences[patient_id] = []
        patient_sequences[patient_id].append(sample)
    
    # Sort visits by date for each patient
    for patient_id in patient_sequences:
        patient_sequences[patient_id].sort(key=lambda x: x['visit_date'])
    
    correct_predictions = 0
    total_predictions = 0
    
    # Test on patients with multiple visits
    test_patients = [pid for pid, visits in patient_sequences.items() if len(visits) >= 2]
    
    if not test_patients:
        print("❌ No patients with multiple visits found for progression testing")
        return 0
    
    # Test first 5 patients with multiple visits
    for i, patient_id in enumerate(test_patients[:5]):
        visits = patient_sequences[patient_id]
        print(f"\n--- Patient {i+1} ({len(visits)} visits) ---")
        print(f"Patient: {visits[0]['patient_info'].get('first_name', 'Unknown')} {visits[0]['patient_info'].get('last_name', 'Unknown')}")
        print(f"True Progression: {visits[0]['progression']}")
        
        # Prepare sequence data
        sequence_data = []
        for visit in visits:
            visit_data = {}
            for feature in target_features:
                visit_data[feature] = visit['tests'].get(feature, 0.0)
            sequence_data.append(visit_data)
        
        # Make prediction
        result = inference.predict_progression(sequence_data)
        
        if 'error' in result:
            print(f"❌ Prediction Error: {result['error']}")
            continue
        
        predicted_class = result['predicted_class']
        confidence = result['prediction_confidence']
        
        print(f"Predicted Progression: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Confidence Scores: {result['confidence_scores']}")
        
        # Check if prediction is correct
        if predicted_class == visits[0]['progression']:
            print("✅ CORRECT PREDICTION")
            correct_predictions += 1
        else:
            print("❌ INCORRECT PREDICTION")
        
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n📊 Progression Model Test Results:")
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def run_comprehensive_test():
    """Run comprehensive tests on both models"""
    print("🚀 Starting Model Testing...")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load test data
    test_samples, target_features = load_test_data(num_samples=100)
    
    if not test_samples:
        print("❌ No test samples found!")
        return
    
    # Initialize inference
    inference = ModelInference()
    inference.load_models()
    
    # Check if models are loaded
    model_info = inference.get_model_info()
    print(f"\n📋 Model Information:")
    print(f"Diagnosis Model Loaded: {model_info['models_loaded']['diagnosis']}")
    print(f"Progression Model Loaded: {model_info['models_loaded']['progression']}")
    print(f"Device: {model_info['device']}")
    
    if not model_info['models_loaded']['diagnosis'] and not model_info['models_loaded']['progression']:
        print("❌ No models loaded! Please train models first.")
        return
    
    # Test diagnosis model
    diagnosis_accuracy = 0
    if model_info['models_loaded']['diagnosis']:
        diagnosis_accuracy = test_diagnosis_model(inference, test_samples, target_features)
    
    # Test progression model
    progression_accuracy = 0
    if model_info['models_loaded']['progression']:
        progression_accuracy = test_progression_model(inference, test_samples, target_features)
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL TEST RESULTS")
    print("="*60)
    print(f"Diagnosis Model Accuracy: {diagnosis_accuracy:.4f} ({diagnosis_accuracy*100:.2f}%)")
    print(f"Progression Model Accuracy: {progression_accuracy:.4f} ({progression_accuracy*100:.2f}%)")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save test results
    test_results = {
        'test_date': datetime.now().isoformat(),
        'num_test_samples': len(test_samples),
        'diagnosis_accuracy': diagnosis_accuracy,
        'progression_accuracy': progression_accuracy,
        'target_features': target_features
    }
    
    with open('models/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to models/test_results.json")

def test_single_prediction():
    """Test single prediction with sample data"""
    print("\n" + "="*60)
    print("🧪 SINGLE PREDICTION TEST")
    print("="*60)
    
    # Initialize inference
    inference = ModelInference()
    inference.load_models()
    
    # Sample data for diagnosis test
    sample_diagnosis_data = {
        "fasting_glucose": 140.0,
        "hba1c": 7.2,
        "hdl": 35.0,
        "ldl": 150.0,
        "triglycerides": 200.0,
        "total_cholesterol": 250.0,
        "creatinine": 1.2,
        "bmi": 32.0,
        "systolic_bp": 140.0,
        "diastolic_bp": 90.0
    }
    
    print("🔬 Testing Diagnosis Model:")
    print("Sample Data:", sample_diagnosis_data)
    
    result = inference.predict_diagnosis(sample_diagnosis_data)
    print("Prediction Result:", result)
    
    # Sample sequence for progression test
    sample_sequence = [
        {"fasting_glucose": 100.0, "hba1c": 5.5, "hdl": 50.0, "ldl": 100.0, "triglycerides": 120.0, "total_cholesterol": 180.0, "creatinine": 0.9, "bmi": 25.0, "systolic_bp": 120.0, "diastolic_bp": 80.0},
        {"fasting_glucose": 110.0, "hba1c": 5.8, "hdl": 48.0, "ldl": 110.0, "triglycerides": 130.0, "total_cholesterol": 190.0, "creatinine": 0.95, "bmi": 26.0, "systolic_bp": 125.0, "diastolic_bp": 82.0},
        {"fasting_glucose": 120.0, "hba1c": 6.2, "hdl": 45.0, "ldl": 120.0, "triglycerides": 150.0, "total_cholesterol": 200.0, "creatinine": 1.0, "bmi": 28.5, "systolic_bp": 130.0, "diastolic_bp": 85.0}
    ]
    
    print("\n🔬 Testing Progression Model:")
    print("Sample Sequence Length:", len(sample_sequence))
    
    result = inference.predict_progression(sample_sequence)
    print("Prediction Result:", result)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained ML models')
    parser.add_argument('--mode', choices=['comprehensive', 'single'], default='comprehensive',
                       help='Test mode: comprehensive (full test) or single (sample prediction)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of test samples for comprehensive test')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        run_comprehensive_test()
    else:
        test_single_prediction()
