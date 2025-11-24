"""
Multi-Disease Inference Service
Handles predictions for both Diabetes and Iron Deficiency Anemia
"""

import pickle
import json
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.services.model_loader import load_lstm_model

class MultiDiseaseInference:
    """Handles ML predictions for multiple diseases"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models = {
            'diabetes': {},
            'anemia': {},
            'ckd': {}
        }
        self._models_loaded = False
    
    def load_models(self):
        """Load all disease models"""
        print("Loading ML models...")
        
        try:
            # Load Diabetes Models
            self._load_diabetes_models()
            
            # Load Anemia Models
            self._load_anemia_models()
            
            # Load CKD Models
            self._load_ckd_models()
            
            self._models_loaded = True
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def _load_diabetes_models(self):
        """Load diabetes diagnosis and progression models"""
        print("\n📊 Loading Diabetes models...")
        
        # Load XGBoost diagnosis model
        diagnosis_path = self.models_dir / "diabetes_diagnosis_xgb.pkl"
        if diagnosis_path.exists():
            with open(diagnosis_path, 'rb') as f:
                self.models['diabetes']['diagnosis'] = pickle.load(f)
            print("  ✅ Diabetes diagnosis model loaded")
        
        # Load LSTM progression model
        progression_path = self.models_dir / "diabetes_progression_lstm.pth"
        if progression_path.exists():
            model, checkpoint = load_lstm_model(str(progression_path), device='cpu', model_type='diabetes')
            self.models['diabetes']['progression'] = {
                'model': model,
                'checkpoint': checkpoint
            }
            print("  ✅ Diabetes progression model loaded")
    
    def _load_anemia_models(self):
        """Load anemia diagnosis and progression models"""
        print("\n🩸 Loading Anemia models...")
        
        # Load XGBoost diagnosis model
        diagnosis_path = self.models_dir / "anemia_diagnosis_xgb.pkl"
        if diagnosis_path.exists():
            with open(diagnosis_path, 'rb') as f:
                self.models['anemia']['diagnosis'] = pickle.load(f)
            print("  ✅ Anemia diagnosis model loaded")
        
        # Load diagnosis features
        features_path = self.models_dir / "anemia_diagnosis_features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.models['anemia']['diagnosis_features'] = json.load(f)
            print("  ✅ Anemia diagnosis features loaded")
        
        # Load LSTM progression model
        progression_path = self.models_dir / "anemia_progression_lstm.pth"
        if progression_path.exists():
            model, checkpoint = load_lstm_model(str(progression_path), device='cpu', model_type='anemia')
            self.models['anemia']['progression'] = {
                'model': model,
                'checkpoint': checkpoint
            }
            print("  ✅ Anemia progression model loaded")
        
        # Load scaler and encoder for progression
        scaler_path = self.models_dir / "anemia_progression_scaler.pkl"
        encoder_path = self.models_dir / "anemia_progression_encoder.pkl"
        config_path = self.models_dir / "anemia_progression_config.json"
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.models['anemia']['progression_scaler'] = pickle.load(f)
            print("  ✅ Anemia progression scaler loaded")
        
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.models['anemia']['progression_encoder'] = pickle.load(f)
            print("  ✅ Anemia progression encoder loaded")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.models['anemia']['progression_config'] = json.load(f)
            print("  ✅ Anemia progression config loaded")
    
    def _load_ckd_models(self):
        """Load CKD diagnosis and progression models"""
        print("\n🫘 Loading CKD models...")
        
        # Load XGBoost diagnosis model (saved with joblib)
        diagnosis_path = self.models_dir / "ckd_diagnosis_xgb_model.pkl"
        if diagnosis_path.exists():
            self.models['ckd']['diagnosis'] = joblib.load(diagnosis_path)
            print("  ✅ CKD diagnosis model loaded")
        
        # Load diagnosis features
        features_path = self.models_dir / "ckd_diagnosis_features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.models['ckd']['diagnosis_features'] = json.load(f)
            print("  ✅ CKD diagnosis features loaded")
        
        # Load label encoder (saved with joblib)
        label_encoder_path = self.models_dir / "ckd_diagnosis_label_encoder.pkl"
        if label_encoder_path.exists():
            self.models['ckd']['diagnosis_encoder'] = joblib.load(label_encoder_path)
            print("  ✅ CKD diagnosis label encoder loaded")
        
        # Load LSTM progression model
        progression_path = self.models_dir / "ckd_progression_lstm_model.pth"
        if progression_path.exists():
            model, checkpoint = load_lstm_model(str(progression_path), device='cpu', model_type='ckd')
            self.models['ckd']['progression'] = {
                'model': model,
                'checkpoint': checkpoint
            }
            print("  ✅ CKD progression model loaded")
        
        # Load scaler and encoder for progression
        scaler_path = self.models_dir / "ckd_progression_scaler.pkl"
        encoder_path = self.models_dir / "ckd_progression_encoder.pkl"
        features_path = self.models_dir / "ckd_progression_features.json"
        
        if scaler_path.exists():
            self.models['ckd']['progression_scaler'] = joblib.load(scaler_path)
            print("  ✅ CKD progression scaler loaded")
        
        if encoder_path.exists():
            self.models['ckd']['progression_encoder'] = joblib.load(encoder_path)
            print("  ✅ CKD progression encoder loaded")
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                features = json.load(f)
                self.models['ckd']['progression_config'] = {
                    'features': features,
                    'max_length': 25  # From ckd_progression_lstm.py
                }
            print("  ✅ CKD progression config loaded")
    
    def predict_diabetes_diagnosis(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict diabetes diagnosis from single visit data"""
        if not self._models_loaded:
            self.load_models()
        
        model = self.models['diabetes']['diagnosis']
        
        # Expected features for diabetes
        features = [
            'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
            'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
        ]
        
        # Prepare input
        X = np.array([[patient_data.get(f, 0.0) for f in features]])
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        diagnosis_map = {0: "Normal", 1: "Prediabetes", 2: "Diabetes"}
        
        return {
            'diagnosis': diagnosis_map[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                'Normal': float(probabilities[0]),
                'Prediabetes': float(probabilities[1]),
                'Diabetes': float(probabilities[2])
            },
            'input_features': patient_data
        }
    
    def predict_anemia_diagnosis(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict anemia diagnosis from single visit data"""
        if not self._models_loaded:
            self.load_models()
        
        model = self.models['anemia']['diagnosis']
        features_config = self.models['anemia'].get('diagnosis_features', {})
        
        # Expected features for anemia
        features = features_config.get('features', [
            'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
            'serum_iron', 'ferritin', 'tibc', 'transferrin_saturation',
            'reticulocyte_count', 'wbc', 'platelet_count', 'esr',
            'bmi', 'systolic_bp', 'diastolic_bp'
        ])
        
        # Prepare input
        X = np.array([[patient_data.get(f, 0.0) for f in features]])
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        diagnosis_map = features_config.get('diagnosis_mapping', {
            0: "Normal",
            1: "Iron Deficiency without Anemia",
            2: "Mild Iron Deficiency Anemia",
            3: "Moderate Iron Deficiency Anemia",
            4: "Severe Iron Deficiency Anemia"
        })
        
        # Handle reversed mapping (diagnosis_name -> label) by inverting it
        if diagnosis_map and len(diagnosis_map) > 0:
            first_key = list(diagnosis_map.keys())[0]
            if isinstance(first_key, str):
                # Mapping is reversed (diagnosis -> label), invert it
                # Values should be integers (labels), keys are diagnosis names
                diagnosis_map = {int(v): k for k, v in diagnosis_map.items()}
        
        target_names = features_config.get('target_names', list(diagnosis_map.values()))
        
        prob_dict = {name: float(probabilities[i]) for i, name in enumerate(target_names)}
        
        return {
            'diagnosis': diagnosis_map[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'input_features': patient_data
        }
    
    def predict_anemia_progression(self, patient_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """Predict anemia progression from patient visit sequence"""
        if not self._models_loaded:
            self.load_models()
        
        model = self.models['anemia']['progression']['model']
        checkpoint = self.models['anemia']['progression']['checkpoint']
        scaler = self.models['anemia']['progression_scaler']
        encoder = self.models['anemia']['progression_encoder']
        config = self.models['anemia']['progression_config']
        
        # Get features from config
        features = config.get('features', checkpoint['features'])
        max_length = config.get('max_length', 20)
        
        # Prepare sequence
        sequence = []
        for visit in patient_sequence:
            visit_data = [visit.get(f, 0.0) for f in features]
            sequence.append(visit_data)
        
        # Pad sequence
        X = pad_sequences([sequence], maxlen=max_length, dtype='float32', padding='pre')
        
        # Scale
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            prediction = torch.argmax(outputs, dim=1).item()
        
        # Get class names
        progression_classes = encoder.classes_
        
        prob_dict = {str(cls): float(probabilities[i]) for i, cls in enumerate(progression_classes)}
        
        return {
            'progression': str(progression_classes[prediction]),
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'num_visits': len(patient_sequence)
        }
    
    def predict_ckd_diagnosis(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict CKD diagnosis from single visit data"""
        if not self._models_loaded:
            self.load_models()
        
        model = self.models['ckd']['diagnosis']
        encoder = self.models['ckd']['diagnosis_encoder']
        features_config = self.models['ckd'].get('diagnosis_features', [])
        
        # Expected features for CKD
        # features_config is a JSON array, so it's already a list
        if isinstance(features_config, list) and len(features_config) > 0:
            features = features_config
        else:
            # Fallback to default features
            features = [
                'serum_creatinine', 'egfr', 'uacr', 'bun', 'sodium', 'potassium',
                'calcium', 'phosphorus', 'hemoglobin', 'pth', 'bicarbonate',
                'albumin', 'bmi', 'systolic_bp', 'diastolic_bp'
            ]
        
        # Prepare input
        X = np.array([[patient_data.get(f, 0.0) for f in features]])
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Decode prediction using label encoder
        diagnosis = encoder.inverse_transform([prediction])[0]
        
        # Get all class names
        class_names = encoder.classes_
        prob_dict = {str(cls): float(probabilities[i]) for i, cls in enumerate(class_names)}
        
        return {
            'diagnosis': str(diagnosis),
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'input_features': patient_data
        }
    
    def predict_ckd_progression(self, patient_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """Predict CKD progression from patient visit sequence"""
        if not self._models_loaded:
            self.load_models()
        
        model = self.models['ckd']['progression']['model']
        checkpoint = self.models['ckd']['progression']['checkpoint']
        scaler = self.models['ckd']['progression_scaler']
        encoder = self.models['ckd']['progression_encoder']
        config = self.models['ckd']['progression_config']
        
        # Get features from config
        features = config.get('features', checkpoint.get('features', []))
        max_length = config.get('max_length', 25)
        
        # Prepare sequence
        sequence = []
        for visit in patient_sequence:
            visit_data = [visit.get(f, 0.0) for f in features]
            sequence.append(visit_data)
        
        # Pad sequence
        X = pad_sequences([sequence], maxlen=max_length, dtype='float32', padding='pre', truncating='pre')
        
        # Scale
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            prediction = torch.argmax(outputs, dim=1).item()
        
        # Get class names
        progression_classes = encoder.classes_
        
        prob_dict = {str(cls): float(probabilities[i]) for i, cls in enumerate(progression_classes)}
        
        return {
            'progression': str(progression_classes[prediction]),
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'num_visits': len(patient_sequence)
        }
    
    def predict_diagnosis(self, disease_name: str, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Generic method to predict diagnosis for any disease
        
        Args:
            disease_name: Name of the disease ('diabetes', 'anemia', 'iron_deficiency_anemia')
            patient_data: Dictionary of patient features
            
        Returns:
            Dictionary with diagnosis prediction results
        """
        # Normalize disease name
        disease_name = disease_name.lower().strip()
        
        # Map common variations
        disease_map = {
            'diabetes': 'diabetes',
            'diabetic': 'diabetes',
            'anemia': 'anemia',
            'iron_deficiency_anemia': 'anemia',
            'iron deficiency anemia': 'anemia',
            'ida': 'anemia',
            'ckd': 'ckd',
            'chronic_kidney_disease': 'ckd',
            'chronic kidney disease': 'ckd',
            'kidney_disease': 'ckd',
            'kidney disease': 'ckd'
        }
        
        disease_key = disease_map.get(disease_name, disease_name)
        
        if disease_key not in self.models:
            raise ValueError(f"Unsupported disease: {disease_name}. Supported diseases: {list(self.models.keys())}")
        
        # Route to appropriate method
        if disease_key == 'diabetes':
            return self.predict_diabetes_diagnosis(patient_data)
        elif disease_key == 'anemia':
            return self.predict_anemia_diagnosis(patient_data)
        elif disease_key == 'ckd':
            return self.predict_ckd_diagnosis(patient_data)
        else:
            raise ValueError(f"Prediction not implemented for disease: {disease_name}")
    
    def predict_progression(self, disease_name: str, patient_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Generic method to predict progression for any disease
        
        Args:
            disease_name: Name of the disease ('diabetes', 'anemia', 'iron_deficiency_anemia')
            patient_sequence: List of visit data dictionaries
            
        Returns:
            Dictionary with progression prediction results
        """
        # Normalize disease name
        disease_name = disease_name.lower().strip()
        
        # Map common variations
        disease_map = {
            'diabetes': 'diabetes',
            'diabetic': 'diabetes',
            'anemia': 'anemia',
            'iron_deficiency_anemia': 'anemia',
            'iron deficiency anemia': 'anemia',
            'ida': 'anemia',
            'ckd': 'ckd',
            'chronic_kidney_disease': 'ckd',
            'chronic kidney disease': 'ckd',
            'kidney_disease': 'ckd',
            'kidney disease': 'ckd'
        }
        
        disease_key = disease_map.get(disease_name, disease_name)
        
        if disease_key not in self.models:
            raise ValueError(f"Unsupported disease: {disease_name}. Supported diseases: {list(self.models.keys())}")
        
        # Route to appropriate method
        if disease_key == 'diabetes':
            # TODO: Implement diabetes progression if needed
            raise NotImplementedError("Diabetes progression prediction not yet implemented")
        elif disease_key == 'anemia':
            return self.predict_anemia_progression(patient_sequence)
        elif disease_key == 'ckd':
            return self.predict_ckd_progression(patient_sequence)
        else:
            raise ValueError(f"Progression prediction not implemented for disease: {disease_name}")
    
    def get_supported_diseases(self) -> List[str]:
        """Get list of supported disease names"""
        return list(self.models.keys())
    
    def get_disease_features(self, disease_name: str, prediction_type: str = 'diagnosis') -> List[str]:
        """
        Get required features for a disease
        
        Args:
            disease_name: Name of the disease
            prediction_type: 'diagnosis' or 'progression'
            
        Returns:
            List of required feature names
        """
        disease_name = disease_name.lower().strip()
        disease_map = {
            'diabetes': 'diabetes',
            'diabetic': 'diabetes',
            'anemia': 'anemia',
            'iron_deficiency_anemia': 'anemia',
            'iron deficiency anemia': 'anemia',
            'ida': 'anemia',
            'ckd': 'ckd',
            'chronic_kidney_disease': 'ckd',
            'chronic kidney disease': 'ckd',
            'kidney_disease': 'ckd',
            'kidney disease': 'ckd'
        }
        
        disease_key = disease_map.get(disease_name, disease_name)
        
        if disease_key == 'diabetes' and prediction_type == 'diagnosis':
            return [
                'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
                'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
            ]
        elif disease_key == 'anemia' and prediction_type == 'diagnosis':
            return [
                'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
                'serum_iron', 'ferritin', 'tibc', 'transferrin_saturation',
                'reticulocyte_count', 'wbc', 'platelet_count', 'esr',
                'bmi', 'systolic_bp', 'diastolic_bp'
            ]
        elif disease_key == 'anemia' and prediction_type == 'progression':
            return [
                'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
                'serum_iron', 'ferritin', 'tibc', 'transferrin_saturation',
                'reticulocyte_count', 'bmi', 'systolic_bp', 'diastolic_bp'
            ]
        elif disease_key == 'ckd' and prediction_type == 'diagnosis':
            return [
                'serum_creatinine', 'egfr', 'uacr', 'bun', 'sodium', 'potassium',
                'calcium', 'phosphorus', 'hemoglobin', 'pth', 'bicarbonate',
                'albumin', 'bmi', 'systolic_bp', 'diastolic_bp'
            ]
        elif disease_key == 'ckd' and prediction_type == 'progression':
            return [
                'serum_creatinine', 'egfr', 'uacr', 'bun', 'sodium', 'potassium',
                'calcium', 'phosphorus', 'hemoglobin', 'pth', 'bicarbonate',
                'albumin', 'bmi', 'systolic_bp', 'diastolic_bp'
            ]
        else:
            raise ValueError(f"Features not available for {disease_name} {prediction_type}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'diabetes': {
                'diagnosis_loaded': 'diagnosis' in self.models['diabetes'],
                'progression_loaded': 'progression' in self.models['diabetes'],
            },
            'anemia': {
                'diagnosis_loaded': 'diagnosis' in self.models['anemia'],
                'progression_loaded': 'progression' in self.models['anemia'],
            },
            'ckd': {
                'diagnosis_loaded': 'diagnosis' in self.models['ckd'],
                'progression_loaded': 'progression' in self.models['ckd'],
            },
            'models_loaded': self._models_loaded,
            'supported_diseases': self.get_supported_diseases()
        }

# Global instance
multi_disease_inference = MultiDiseaseInference()

