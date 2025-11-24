"""
Iron Deficiency Anemia Progression Model using PyTorch LSTM
Predicts disease progression over time based on sequential lab values
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pickle

# This script assumes the 'df' DataFrame from the anemia data generation script exists.

print("\n" + "="*60)
print("TRAINING PYTORCH LSTM ANEMIA PROGRESSION MODEL")
print("="*60)

# --- 1. Data Preparation ---
print("\n📊 Preparing sequential data...")

# Define features for anemia progression prediction
features = [
    "hemoglobin", "hematocrit", "mcv", "mch", "mchc", "rdw",
    "serum_iron", "ferritin", "tibc", "transferrin_saturation",
    "reticulocyte_count", "bmi", "systolic_bp", "diastolic_bp"
]

def load_and_prepare_progression_data(json_prefix='generated_anemia_'):
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
    df = load_and_prepare_progression_data('generated_anemia_')
except Exception as e:
    print(f"\n⚠️  Error loading data: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("\nPlease ensure you have run: python iron_deficiency_anemia_data_generation.py")
    exit(1)

# Check which features are available
available_features = [f for f in features if f in df.columns]
print(f"\nUsing {len(available_features)} features:")
print(", ".join(available_features))

# Group by patient and create sequences
grouped = df.groupby('patient_id')
sequences, targets, patient_ids = [], [], []

for patient_id, group in grouped:
    # Get feature values for this patient across all visits
    patient_sequence = group[available_features].values
    
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
max_length = 20  # Maximum number of visits to consider
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

# Split data
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

# --- 2. Create PyTorch Dataset and DataLoader ---
class PatientDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = PatientDataset(X_train_scaled, y_train)
test_dataset = PatientDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✅ PyTorch dataloaders created")

# --- 3. Define the Bidirectional LSTM Model ---
class AnemiaProgressionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AnemiaProgressionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.4 if num_layers > 1 else 0  # Increased from 0.3 to 0.4
        )
        self.dropout = nn.Dropout(0.5)  # Increased from 0.4 to 0.5
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # LSTM layer
        _, (h_n, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # Fully connected layers with dropout
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

# --- 4. Training Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

# Hyperparameters
input_size = len(available_features)
hidden_size = 128
num_layers = 2
epochs = 25  # Reduced from 30 to prevent overfitting (best performance at epoch 25)
learning_rate = 0.0005

print(f"\n🔧 Model Configuration:")
print(f"   Input size: {input_size}")
print(f"   Hidden size: {hidden_size}")
print(f"   Number of LSTM layers: {num_layers}")
print(f"   Output classes: {num_classes}")
print(f"   Epochs: {epochs}")
print(f"   Learning rate: {learning_rate}")

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\n⚖️  Class weights (for imbalanced data): {class_weights}")

# Initialize model
model = AnemiaProgressionBiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added L2 regularization

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print("\n🚀 Starting training...")
print("-" * 60)

# --- 5. Training Loop ---
train_losses = []
test_losses = []
best_test_loss = float('inf')
patience_counter = 0
early_stop_patience = 7  # Stop if no improvement for 7 epochs

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Update learning rate
    scheduler.step(avg_test_loss)
    
    # Save best model and early stopping
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0  # Reset patience
        import os
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/anemia_progression_lstm_best.pth')
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f'\n⚠️  Early stopping triggered at epoch {epoch+1}')
        print(f'   No improvement in test loss for {early_stop_patience} epochs')
        print(f'   Best test loss: {best_test_loss:.4f}')
        break

print("\n✅ Training complete!")

# Load best model
model.load_state_dict(torch.load('models/anemia_progression_lstm_best.pth'))

# --- 6. Evaluation ---
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for sequences, labels_cpu in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_cpu.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print(f"\n🔹 Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🔹 Weighted F1-Score: {f1_weighted:.4f}")

print("\n📊 Classification Report:")
print("-" * 60)
print(classification_report(all_labels, all_preds, target_names=encoder.classes_, digits=4))

# --- 7. Visualizations ---

# Create models directory if it doesn't exist
import os
os.makedirs('models', exist_ok=True)

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, cmap='Reds', fmt='d', 
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title("LSTM - Anemia Progression Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Progression", fontsize=12)
plt.ylabel("True Progression", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('models/anemia_progression_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n💾 Confusion matrix saved as 'models/anemia_progression_confusion_matrix.png'")
plt.show()

# Training Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o', linewidth=2)
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='s', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Test Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/anemia_progression_loss_curve.png', dpi=300, bbox_inches='tight')
print("💾 Loss curve saved as 'models/anemia_progression_loss_curve.png'")
plt.show()

# --- 8. Save Model and Configuration ---
print("\n💾 Saving model and configuration...")

# Save model (consistent with diabetes model naming)
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'num_classes': num_classes,
    'features': available_features,
    'encoder_classes': list(encoder.classes_)
}, 'models/anemia_progression_lstm.pth')

# Save scaler
with open('models/anemia_progression_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoder
with open('models/anemia_progression_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Save configuration
config = {
    'features': available_features,
    'max_length': max_length,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'num_classes': num_classes,
    'encoder_classes': list(encoder.classes_),
    'test_accuracy': float(accuracy),
    'test_f1_score': float(f1_weighted)
}

with open('models/anemia_progression_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Model saved as 'models/anemia_progression_lstm.pth'")
print("✅ Scaler saved as 'models/anemia_progression_scaler.pkl'")
print("✅ Encoder saved as 'models/anemia_progression_encoder.pkl'")
print("✅ Configuration saved as 'models/anemia_progression_config.json'")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
print("\n🎉 Anemia progression model is ready for deployment!")

