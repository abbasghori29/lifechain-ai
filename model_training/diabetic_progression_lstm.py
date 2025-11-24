import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt

# This script assumes the 'df' DataFrame from the data generation script exists.

print("\n--- Training PyTorch Progression Model ---")

# --- 1. Data Preparation ---
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

# ⭐️ CORRECTED LINES: Reshape using -1 to automatically calculate the dimension
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# --- 2. Create PyTorch Dataset and DataLoader ---
class PatientDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

train_dataset, test_dataset = PatientDataset(X_train, y_train), PatientDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- 3. Define the Bidirectional LSTM Model ---
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

# --- 4. Training Loop ---
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

# --- 5. Evaluation ---
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