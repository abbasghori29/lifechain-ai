"""
Model Training Script for LifeChain AI Healthcare System
Loads data from database and trains ML models for diabetes diagnosis and progression
"""

import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import json
from pathlib import Path
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select
from app.core.config import get_settings
from app.models import Patient, DoctorVisit, LabTestResult, Diagnosis, DiseaseProgression

# Create models directory
os.makedirs("models", exist_ok=True)

# Define the Bidirectional LSTM Model (moved outside function for pickling)
class ProgressionBiLSTM(nn.Module):
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

def load_training_data():
    """Load training data from JSON files"""
    print("Loading training data from JSON files...")
    
    # Load the generated data files
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
        
        print(f"Loaded {len(lab_results)} lab test results")
        print(f"Loaded {len(visits)} doctor visits")
        print(f"Loaded {len(diagnoses)} diagnoses")
        print(f"Loaded {len(progressions)} disease progressions")
        print(f"Loaded {len(patients)} patients")
        
        # Create lookup dictionaries
        visits_dict = {v['visit_id']: v for v in visits}
        diagnoses_dict = {d['visit_id']: d for d in diagnoses}
        progressions_dict = {p['patient_id']: p for p in progressions}
        patients_dict = {p['patient_id']: p for p in patients}
        
        # Filter lab results for our target features
        target_features = [
            'fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides',
            'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp'
        ]
        
        filtered_results = [r for r in lab_results if r['test_name'] in target_features]
        print(f"Filtered to {len(filtered_results)} relevant lab test results")
        
        # Build training data
        training_data = []
        for result in filtered_results:
            visit_id = result['report_id']  # This should be visit_id from the lab report
            # We need to find the visit_id from the lab report
            # Let's get it from the lab reports file
            continue  # Skip for now, we'll use a different approach
        
        # Alternative approach: Use the existing data structure
        # Load lab reports to get visit_id mapping
        with open('generated_lab_reports.json', 'r') as f:
            lab_reports = json.load(f)
        
        # Create report_id to visit_id mapping
        report_to_visit = {r['report_id']: r['visit_id'] for r in lab_reports if r.get('visit_id')}
        
        # Build training data
        training_data = []
        for result in filtered_results:
            report_id = result['report_id']
            if report_id in report_to_visit:
                visit_id = report_to_visit[report_id]
                if visit_id in visits_dict:
                    visit = visits_dict[visit_id]
                    patient_id = visit['patient_id']
                    
                    if patient_id in patients_dict:
                        patient = patients_dict[patient_id]
                        
                        # Get diagnosis and progression
                        diagnosis = diagnoses_dict.get(visit_id, {}).get('disease_name', 'Normal')
                        progression = progressions_dict.get(patient_id, {}).get('progression_stage', 'Normal')
                        
                        training_data.append({
                            'patient_id': patient_id,
                            'first_name': patient['first_name'],
                            'last_name': patient['last_name'],
                            'gender': patient['gender'],
                            'visit_date': visit['visit_date'],
                            'test_name': result['test_name'],
                            'test_value': result['test_value'],
                            'diagnosis': diagnosis,
                            'progression': progression
                        })
        
        print(f"Built {len(training_data)} training records")
        
        if not training_data:
            raise ValueError("No training data found in JSON files")
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Pivot the data to get features as columns
        df_pivot = df.pivot_table(
            index=['patient_id', 'first_name', 'last_name', 'gender', 'visit_date', 'diagnosis', 'progression'],
            columns='test_name',
            values='test_value',
            aggfunc='first'
        ).reset_index()
        
        # Fill missing values with median
        feature_cols = ['fasting_glucose', 'hba1c', 'hdl', 'ldl', 'triglycerides', 
                       'total_cholesterol', 'creatinine', 'bmi', 'systolic_bp', 'diastolic_bp']
        
        for col in feature_cols:
            if col in df_pivot.columns:
                df_pivot[col] = df_pivot[col].fillna(df_pivot[col].median())
            else:
                df_pivot[col] = 0  # If column doesn't exist, fill with 0
        
        # Add visit number for each patient
        df_pivot['visit_no'] = df_pivot.groupby('patient_id').cumcount() + 1
        
        print(f"Prepared dataset with {len(df_pivot)} records for {df_pivot['patient_id'].nunique()} patients")
        return df_pivot
        
    except Exception as e:
        print(f"Error loading data from JSON: {e}")
        raise

def train_diagnosis_model(df):
    """Train XGBoost model for diabetes diagnosis"""
    print("\n--- Training XGBoost Diagnostic Model ---")
    
    # Prepare data
    df['diagnosis_label'] = df['diagnosis'].map({"Normal": 0, "Prediabetes": 1, "Diabetes": 2})
    exclude_cols = ['patient_id', 'first_name', 'last_name', 'gender', 'visit_no', 'diagnosis', 'progression', 'diagnosis_label', 'visit_date']
    X = df.drop(columns=exclude_cols)
    y = df['diagnosis_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        learning_rate=0.05, n_estimators=100, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=10
    )

    eval_set = [(X_test, y_test)]
    model_xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_pred = model_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🔹 XGBoost Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Prediabetes", "Diabetes"]))

    # Save model
    model_path = "models/diabetes_diagnosis_xgb.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model_xgb,
            'scaler': None,  # XGBoost doesn't need scaling
            'feature_columns': X.columns.tolist(),
            'accuracy': accuracy,
            'trained_at': datetime.now().isoformat()
        }, f)
    
    print(f"✅ Diagnosis model saved to {model_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', 
                xticklabels=["Normal", "Prediabetes", "Diabetes"], yticklabels=["Normal", "Prediabetes", "Diabetes"])
    plt.title("XGBoost - Diagnostic Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("models/diagnosis_confusion_matrix.png")
    plt.close()
    
    return model_xgb

def train_progression_model(df):
    """Train LSTM model for diabetes progression"""
    print("\n--- Training PyTorch Progression Model ---")
    
    # Prepare sequences
    features = [
        "fasting_glucose", "hba1c", "hdl", "ldl", "triglycerides",
        "total_cholesterol", "creatinine", "bmi", "systolic_bp", "diastolic_bp"
    ]
    
    grouped = df.groupby('patient_id')
    sequences, targets = [], []
    for _, group in grouped:
        sequences.append(group[features].values)
        targets.append(group['progression'].iloc[-1])

    X_padded = pad_sequences(sequences, maxlen=25, dtype='float32', padding='pre', truncating='pre')
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(targets)
    num_classes = len(encoder.classes_)
    print(f"Progression classes found: {encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    scaler = StandardScaler()

    # Reshape using -1 to automatically calculate the dimension
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Create PyTorch Dataset and DataLoader
    class PatientDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.sequences)
        def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

    train_dataset, test_dataset = PatientDataset(X_train, y_train), PatientDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Use the globally defined ProgressionBiLSTM class

    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size, hidden_size, num_layers, epochs = len(features), 128, 2, 25

    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Using class weights: {weights_tensor}")

    model = ProgressionBiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels_cpu in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels_cpu.numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\n🔹 PyTorch LSTM Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=encoder.classes_))

    # Save model
    model_path = "models/diabetes_progression_lstm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': ProgressionBiLSTM,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'scaler': scaler,
        'encoder': encoder,
        'feature_columns': features,
        'accuracy': accuracy,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    
    print(f"✅ Progression model saved to {model_path}")
    
    return model

def main():
    """Main training function"""
    print("🚀 Starting model training...")
    
    # Load data from JSON files
    df = load_training_data()
    
    # Train diagnosis model
    diagnosis_model = train_diagnosis_model(df)
    
    # Train progression model
    progression_model = train_progression_model(df)
    
    print("\n🎉 All models trained and saved successfully!")
    print("📁 Models saved in 'models/' directory")

if __name__ == "__main__":
    main()
