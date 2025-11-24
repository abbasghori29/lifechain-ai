import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# This script assumes the 'df' DataFrame from the data generation script exists.

def train_xgb_diagnostic_model(diagnostic_df):
    """
    Trains an XGBoost model for single-visit diagnosis.
    """
    print("\n--- Training XGBoost Diagnostic Model ---")
    
    diagnostic_df['diagnosis_label'] = diagnostic_df['diagnosis'].map({"Normal": 0, "Prediabetes": 1, "Diabetes": 2})
    exclude_cols = ['patient_id', 'name', 'gender', 'visit_no', 'diagnosis', 'progression', 'diagnosis_label']
    X = diagnostic_df.drop(columns=exclude_cols)
    y = diagnostic_df['diagnosis_label']

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
    print(f"\n🔹 XGBoost Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Prediabetes", "Diabetes"]))

    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', 
                xticklabels=["Normal", "Prediabetes", "Diabetes"], yticklabels=["Normal", "Prediabetes", "Diabetes"])
    plt.title("XGBoost - Diagnostic Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Prepare and run the diagnostic model
diagnostic_df = df[df['diagnosis'].isin(["Normal", "Prediabetes", "Diabetes"])].copy()
train_xgb_diagnostic_model(diagnostic_df)