"""
CKD Progression Model Training - LSTM Sequence Model
Trains a BiLSTM model to predict CKD progression from patient visit sequences
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
import os
import random

# Define features for CKD progression
features = [
    'serum_creatinine', 'egfr', 'uacr', 'bun',
    'sodium', 'potassium', 'calcium', 'phosphorus',
    'hemoglobin', 'pth', 'bicarbonate', 'albumin',
    'bmi', 'systolic_bp', 'diastolic_bp'
]

class PatientDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class ProgressionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=40, num_layers=2, num_classes=5, dropout=0.6):
        super(ProgressionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 40)  # Further reduced
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(40, 28)  # Further reduced
        self.dropout2 = nn.Dropout(dropout * 0.9)  # Slightly less dropout in second layer
        self.fc3 = nn.Linear(28, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

def load_and_prepare_progression_data(json_prefix='generated_ckd_'):
    """
    Load data from generated JSON files and prepare for progression training.
    """
    print("\n📂 Loading data from JSON files...")
    
    try:
        # Load data
        patients = pd.read_json(f'{json_prefix}patients.json')
        visits = pd.read_json(f'{json_prefix}doctor_visits.json')
        lab_reports = pd.read_json(f'{json_prefix}lab_reports.json')
        lab_results = pd.read_json(f'{json_prefix}lab_test_results.json')
        progressions = pd.read_json(f'{json_prefix}disease_progressions.json')
        
        print(f"✅ Loaded {len(patients)} patients")
        print(f"✅ Loaded {len(visits)} visits")
        print(f"✅ Loaded {len(lab_results)} lab test results")
        print(f"✅ Loaded {len(progressions)} progression records")
        
        # Pivot lab results
        print("\n🔄 Processing data...")
        lab_pivot = lab_results.pivot_table(
            index='report_id',
            columns='test_name',
            values='test_value',
            aggfunc='first'
        ).reset_index()
        
        # Merge data step by step
        df = lab_reports.merge(lab_pivot, on='report_id', how='left')
        print(f"   After lab_pivot merge: {len(df)} rows")
        
        # Check if patient_id already exists in lab_reports
        if 'patient_id' not in df.columns:
            df = df.merge(visits[['visit_id', 'patient_id', 'visit_date']], on='visit_id', how='left')
            print(f"   After visits merge: {len(df)} rows, has patient_id: {'patient_id' in df.columns}")
        else:
            df = df.merge(visits[['visit_id', 'visit_date']], on='visit_id', how='left')
            print(f"   After visits merge: {len(df)} rows")
        
        # Add progression information
        print(f"   Processing progression data...")
        progression_map = progressions.set_index('patient_id')['progression_stage'].to_dict()
        df['progression'] = df['patient_id'].map(progression_map)
        print(f"   Progression data added: {df['progression'].notna().sum()} records with progression info")
        
        # Sort by patient and date
        df = df.sort_values(['patient_id', 'visit_date'])
        
        print(f"\n✅ Prepared {len(df)} records from {df['patient_id'].nunique()} patients")
        
        return df
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find data files. Please run the data generation script first.")
        print(f"   Expected files: {json_prefix}*.json")
        raise
    except Exception as e:
        print(f"\n❌ Error during data processing: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise

# Load data
try:
    df = load_and_prepare_progression_data('generated_ckd_')
except Exception as e:
    print(f"\n⚠️  Error loading data: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("\nPlease ensure you have run: python ckd_data_generation.py")
    exit(1)

# Check which features are available
available_features = [f for f in features if f in df.columns]
print(f"\nUsing {len(available_features)} features:")
print(", ".join(available_features))

# Fill missing features with median
for feat in features:
    if feat not in df.columns:
        df[feat] = 0
    df[feat] = df[feat].fillna(df[feat].median())

# Add noise to simulate measurement errors and real-world variability
print("\n🔧 Adding noise to simulate real-world measurement errors...")
def add_measurement_noise(value, feature_name):
    """Add realistic measurement noise based on feature type"""
    # Different noise levels for different types of measurements (increased for realism)
    noise_levels = {
        'serum_creatinine': 0.08,  # 8% noise (increased from 5%)
        'egfr': 0.12,  # 12% noise (increased from 8% - calculated value, more variability)
        'uacr': 0.15,  # 15% noise (increased from 10% - urine test, more variability)
        'bun': 0.10,  # 10% noise (increased from 6%)
        'sodium': 0.03,  # 3% noise (increased from 2%)
        'potassium': 0.05,  # 5% noise (increased from 3%)
        'calcium': 0.06,  # 6% noise (increased from 4%)
        'phosphorus': 0.08,  # 8% noise (increased from 5%)
        'hemoglobin': 0.06,  # 6% noise (increased from 4%)
        'pth': 0.18,  # 18% noise (increased from 12% - hormone, high variability)
        'bicarbonate': 0.08,  # 8% noise (increased from 5%)
        'albumin': 0.06,  # 6% noise (increased from 4%)
        'bmi': 0.03,  # 3% noise (increased from 2%)
        'systolic_bp': 0.08,  # 8% noise (increased from 5% - BP can vary significantly)
        'diastolic_bp': 0.08  # 8% noise (increased from 5%)
    }
    
    noise_factor = noise_levels.get(feature_name, 0.05)  # Default 5% noise
    noise = np.random.normal(0, abs(value) * noise_factor)
    return value + noise

# Apply noise to all lab values
np.random.seed(42)  # For reproducibility
for feat in features:
    if feat in df.columns:
        df[feat] = df[feat].apply(lambda x: add_measurement_noise(x, feat) if pd.notna(x) else x)

print("   ✅ Noise added to simulate real-world variability")

# Add label noise to progression labels before creating sequences (simulate real-world misclassification)
print("\n🔧 Adding label noise to simulate real-world misclassification...")
progression_series = df.groupby('patient_id')['progression'].last()
valid_progressions = progression_series.dropna()
unique_progressions = valid_progressions.unique()
num_patients = len(valid_progressions)
noise_percentage = 0.08  # 8% of labels will be randomly changed
num_noisy = int(num_patients * noise_percentage)

# Create a mapping to add noise
if num_noisy > 0 and len(unique_progressions) > 1:
    np.random.seed(42)  # For reproducibility
    noisy_patient_ids = np.random.choice(
        valid_progressions.index, 
        size=num_noisy, 
        replace=False
    )
    
    for patient_id in noisy_patient_ids:
        current_label = progression_series[patient_id]
        # Randomly change to a different class
        possible_labels = [p for p in unique_progressions if p != current_label]
        if possible_labels:
            new_label = np.random.choice(possible_labels)
            # Update all rows for this patient
            df.loc[df['patient_id'] == patient_id, 'progression'] = new_label
    
    print(f"   ✅ Added noise to {len(noisy_patient_ids)} patient labels ({noise_percentage*100:.1f}%)")
else:
    print(f"   ⚠️  Skipping label noise (not enough diversity)")

# Group by patient and create sequences
grouped = df.groupby('patient_id')
sequences, targets, patient_ids = [], [], []

for patient_id, group in grouped:
    # Get feature values for this patient across all visits
    patient_sequence = group[features].values
    
    # Only include if we have progression information
    if 'progression' in group.columns and not pd.isna(group['progression'].iloc[-1]):
        sequences.append(patient_sequence)
        targets.append(group['progression'].iloc[-1])  # Final progression status
        patient_ids.append(patient_id)

print(f"\n✅ Created {len(sequences)} patient sequences")
print(f"   Average sequence length: {np.mean([len(s) for s in sequences]):.1f} visits")
print(f"   Min sequence length: {min([len(s) for s in sequences])} visits")
print(f"   Max sequence length: {max([len(s) for s in sequences])} visits")

# Pad sequences to same length
max_length = 25  # Maximum number of visits to consider
X_padded = pad_sequences(
    sequences, 
    maxlen=max_length, 
    dtype='float32', 
    padding='pre', 
    truncating='pre'
)

print(f"\n✅ Padded sequences to length {max_length}")

# Encode progression labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(targets)
num_classes = len(encoder.classes_)

print(f"\n📋 Progression classes found: {list(encoder.classes_)}")
print(f"Number of classes: {num_classes}")

# Class distribution
print("\nProgression class distribution:")
for i, class_name in enumerate(encoder.classes_):
    count = (y_encoded == i).sum()
    percentage = (count / len(y_encoded)) * 100
    print(f"  {class_name}: {count} ({percentage:.1f}%)")

# Run 5-Fold Cross-Validation first
print("\n📊 Running 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Simple function to train and evaluate a model for CV
def train_evaluate_model_cv(X_train_cv, X_val_cv, y_train_cv, y_val_cv, device, num_classes, features):
    """Train and evaluate a single model for CV"""
    # Normalize features
    scaler_cv = StandardScaler()
    X_train_flat = X_train_cv.reshape(-1, X_train_cv.shape[-1])
    X_val_flat = X_val_cv.reshape(-1, X_val_cv.shape[-1])
    
    X_train_scaled_flat = scaler_cv.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler_cv.transform(X_val_flat)
    
    X_train_scaled = X_train_scaled_flat.reshape(X_train_cv.shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val_cv.shape)
    
    # Create datasets
    train_dataset = PatientDataset(X_train_scaled, y_train_cv)
    val_dataset = PatientDataset(X_val_scaled, y_val_cv)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = ProgressionBiLSTM(
        input_size=len(features),
        hidden_size=40,  # Further reduced
        num_layers=2,
        num_classes=num_classes,
        dropout=0.6  # Increased dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.015)  # Increased weight decay, slightly lower LR
    
    # Train for fewer epochs for CV
    model.train()
    for epoch in range(20):  # Reduced epochs for CV
        for sequences_batch, labels_batch in train_loader:
            sequences_batch = sequences_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences_batch, labels_batch in val_loader:
            sequences_batch = sequences_batch.to(device)
            outputs = model(sequences_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted.cpu() == labels_batch).sum().item()
    
    return correct / total

# Run CV
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_padded, y_encoded), 1):
    X_train_cv, X_val_cv = X_padded[train_idx], X_padded[val_idx]
    y_train_cv, y_val_cv = y_encoded[train_idx], y_encoded[val_idx]
    
    score = train_evaluate_model_cv(X_train_cv, X_val_cv, y_train_cv, y_val_cv, 
                                    torch.device('cpu'), num_classes, features)
    cv_scores.append(score)
    print(f"   Fold {fold}: {score:.4f} ({score*100:.2f}%)")

print(f"\n✅ CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print(f"   Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")

# Now do the main train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✅ Train set: {len(X_train)} patients")
print(f"✅ Test set: {len(X_test)} patients")

# Normalize features
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

print("\n✅ Features normalized")

# Create PyTorch Dataset and DataLoader
train_dataset = PatientDataset(X_train_scaled, y_train)
test_dataset = PatientDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("\n✅ DataLoaders created")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

model = ProgressionBiLSTM(
    input_size=len(features),
    hidden_size=40,  # Further reduced
    num_layers=2,
    num_classes=num_classes,
    dropout=0.6  # Increased dropout further
).to(device)

print(f"\n📊 Model architecture:")
print(model)

# Training setup with stronger regularization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.015)  # Increased weight decay, slightly lower LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
num_epochs = 50
best_accuracy = 0.0
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print(f"\n🚀 Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for sequences_batch, labels_batch in train_loader:
        sequences_batch = sequences_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels_batch.size(0)
        train_correct += (predicted == labels_batch).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for sequences_batch, labels_batch in test_loader:
            sequences_batch = sequences_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels_batch.size(0)
            val_correct += (predicted == labels_batch).sum().item()
    
    val_loss /= len(test_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/ckd_progression_lstm_model.pth')
        print(f"   💾 Saved best model (accuracy: {best_accuracy:.2f}%)")

# Load best model for final evaluation
model.load_state_dict(torch.load('models/ckd_progression_lstm_model.pth'))
model.eval()

# Final evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences_batch, labels_batch in test_loader:
        sequences_batch = sequences_batch.to(device)
        outputs = model(sequences_batch)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

# Convert back to class names
y_pred = encoder.inverse_transform(all_preds)
y_true = encoder.inverse_transform(all_labels)

# Final metrics
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save scaler and encoder
joblib.dump(scaler, 'models/ckd_progression_scaler.pkl')
joblib.dump(encoder, 'models/ckd_progression_encoder.pkl')

# Save feature list
with open('models/ckd_progression_features.json', 'w') as f:
    json.dump(features, f, indent=2)

print("\n💾 Model and preprocessing objects saved to 'models/' directory")
print("\n🎉 Training complete!")

