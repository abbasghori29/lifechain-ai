"""
Model Inference Script for LifeChain AI Healthcare System
Loads trained models and provides prediction functions
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ProgressionBiLSTM(nn.Module):
    """LSTM model class for diabetes progression prediction"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ProgressionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_size * 2, 64) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out); out = self.fc1(out)
        out = self.relu(out); out = self.dropout(out)
        out = self.fc2(out)
        return out

class ModelInference:
    """Model inference class for diabetes diagnosis and progression prediction"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.diagnosis_model = None
        self.progression_model = None
        self.diagnosis_scaler = None
        self.progression_scaler = None
        self.progression_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load diagnosis model
            diagnosis_path = self.models_dir / "diabetes_diagnosis_xgb.pkl"
            if diagnosis_path.exists():
                with open(diagnosis_path, 'rb') as f:
                    diagnosis_data = pickle.load(f)
                    self.diagnosis_model = diagnosis_data['model']
                    self.diagnosis_feature_columns = diagnosis_data['feature_columns']
                    print(f"✅ Loaded diagnosis model (accuracy: {diagnosis_data['accuracy']:.4f})")
            else:
                print("❌ Diagnosis model not found")
            
            # Load progression model using the model loader
            progression_path = self.models_dir / "diabetes_progression_lstm.pth"
            if progression_path.exists():
                # Import the model loader
                from app.services.model_loader import load_lstm_model
                
                self.progression_model, checkpoint = load_lstm_model(str(progression_path), str(self.device))
                
                if self.progression_model is not None:
                    # Load scaler and encoder
                    self.progression_scaler = checkpoint['scaler']
                    self.progression_encoder = checkpoint['encoder']
                    self.progression_feature_columns = checkpoint['feature_columns']
                    print(f"✅ Loaded progression model (accuracy: {checkpoint['accuracy']:.4f})")
                else:
                    print("❌ Failed to load progression model")
            else:
                print("❌ Progression model not found")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict_diagnosis(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict diabetes diagnosis from single visit data
        
        Args:
            patient_data: Dictionary with feature values
                Required keys: fasting_glucose, hba1c, hdl, ldl, triglycerides,
                total_cholesterol, creatinine, bmi, systolic_bp, diastolic_bp
        
        Returns:
            Dictionary with prediction results
        """
        if self.diagnosis_model is None:
            return {"error": "Diagnosis model not loaded"}
        
        try:
            # Prepare data
            df = pd.DataFrame([patient_data])
            
            # Ensure all required features are present
            missing_features = set(self.diagnosis_feature_columns) - set(df.columns)
            if missing_features:
                return {"error": f"Missing features: {missing_features}"}
            
            # Select features in correct order
            X = df[self.diagnosis_feature_columns]
            
            # Make prediction
            prediction = self.diagnosis_model.predict(X)[0]
            probabilities = self.diagnosis_model.predict_proba(X)[0]
            
            # Map prediction to class name
            class_names = ["Normal", "Prediabetes", "Diabetes"]
            predicted_class = class_names[prediction]
            
            # Get confidence scores
            confidence_scores = {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            }
            
            return {
                "predicted_class": predicted_class,
                "confidence_scores": confidence_scores,
                "prediction_confidence": float(max(probabilities)),
                "model_used": "XGBoost",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_progression(self, patient_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Predict diabetes progression from patient visit sequence
        
        Args:
            patient_sequence: List of dictionaries with feature values for each visit
                Each dict should have: fasting_glucose, hba1c, hdl, ldl, triglycerides,
                total_cholesterol, creatinine, bmi, systolic_bp, diastolic_bp
        
        Returns:
            Dictionary with prediction results
        """
        if self.progression_model is None:
            return {"error": "Progression model not loaded"}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(patient_sequence)
            
            # Ensure all required features are present
            missing_features = set(self.progression_feature_columns) - set(df.columns)
            if missing_features:
                return {"error": f"Missing features: {missing_features}"}
            
            # Select features in correct order
            X = df[self.progression_feature_columns].values
            
            # Pad sequence to max length (25)
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            X_padded = pad_sequences([X], maxlen=25, dtype='float32', padding='pre', truncating='pre')
            
            # Scale features
            X_scaled = self.progression_scaler.transform(X_padded.reshape(-1, X_padded.shape[-1])).reshape(X_padded.shape)
            
            # Convert to tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.progression_model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
            # Map prediction to class name
            predicted_class = self.progression_encoder.inverse_transform([prediction])[0]
            
            # Get confidence scores
            confidence_scores = {
                self.progression_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(probabilities[0].cpu().numpy())
            }
            
            return {
                "predicted_class": predicted_class,
                "confidence_scores": confidence_scores,
                "prediction_confidence": float(max(probabilities[0].cpu().numpy())),
                "model_used": "BiLSTM",
                "sequence_length": len(patient_sequence),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "models_loaded": {
                "diagnosis": self.diagnosis_model is not None,
                "progression": self.progression_model is not None
            },
            "device": str(self.device),
            "models_directory": str(self.models_dir)
        }
        
        if self.diagnosis_model is not None:
            info["diagnosis_features"] = self.diagnosis_feature_columns
        
        if self.progression_model is not None:
            info["progression_features"] = self.progression_feature_columns
            info["progression_classes"] = self.progression_encoder.classes_.tolist()
        
        return info

# Example usage
if __name__ == "__main__":
    # Initialize inference
    inference = ModelInference()
    inference.load_models()
    
    # Print model info
    print("Model Information:")
    print(inference.get_model_info())
    
    # Example diagnosis prediction
    sample_patient = {
        "fasting_glucose": 120.0,
        "hba1c": 6.2,
        "hdl": 45.0,
        "ldl": 120.0,
        "triglycerides": 150.0,
        "total_cholesterol": 200.0,
        "creatinine": 1.0,
        "bmi": 28.5,
        "systolic_bp": 130.0,
        "diastolic_bp": 85.0
    }
    
    print("\nSample Diagnosis Prediction:")
    result = inference.predict_diagnosis(sample_patient)
    print(result)
    
    # Example progression prediction
    sample_sequence = [
        {"fasting_glucose": 100.0, "hba1c": 5.5, "hdl": 50.0, "ldl": 100.0, "triglycerides": 120.0, "total_cholesterol": 180.0, "creatinine": 0.9, "bmi": 25.0, "systolic_bp": 120.0, "diastolic_bp": 80.0},
        {"fasting_glucose": 110.0, "hba1c": 5.8, "hdl": 48.0, "ldl": 110.0, "triglycerides": 130.0, "total_cholesterol": 190.0, "creatinine": 0.95, "bmi": 26.0, "systolic_bp": 125.0, "diastolic_bp": 82.0},
        {"fasting_glucose": 120.0, "hba1c": 6.2, "hdl": 45.0, "ldl": 120.0, "triglycerides": 150.0, "total_cholesterol": 200.0, "creatinine": 1.0, "bmi": 28.5, "systolic_bp": 130.0, "diastolic_bp": 85.0}
    ]
    
    print("\nSample Progression Prediction:")
    result = inference.predict_progression(sample_sequence)
    print(result)
