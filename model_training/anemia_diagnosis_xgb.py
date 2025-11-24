"""
Iron Deficiency Anemia Diagnosis Model using XGBoost
Trains a classifier to diagnose anemia severity based on lab values
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import json

# This script assumes the 'df' DataFrame from the anemia data generation script exists.

def train_xgb_anemia_diagnostic_model(diagnostic_df):
    """
    Trains an XGBoost model for single-visit anemia diagnosis.
    Classifies into: Normal, Iron Deficiency without Anemia, Mild, Moderate, Severe Anemia
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST ANEMIA DIAGNOSTIC MODEL")
    print("="*60)
    
    # Map diagnosis to numeric labels
    diagnosis_mapping = {
        "Normal": 0,
        "Iron Deficiency without Anemia": 1,
        "Mild Iron Deficiency Anemia": 2,
        "Moderate Iron Deficiency Anemia": 3,
        "Severe Iron Deficiency Anemia": 4
    }
    
    diagnostic_df['diagnosis_label'] = diagnostic_df['diagnosis'].map(diagnosis_mapping)
    
    # Define features specific to iron deficiency anemia
    anemia_features = [
        'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
        'serum_iron', 'ferritin', 'tibc', 'transferrin_saturation',
        'reticulocyte_count', 'wbc', 'platelet_count', 'esr',
        'bmi', 'systolic_bp', 'diastolic_bp'
    ]
    
    # Select only features that exist in the dataframe
    available_features = [f for f in anemia_features if f in diagnostic_df.columns]
    print(f"\nUsing {len(available_features)} features for training:")
    print(", ".join(available_features))
    
    X = diagnostic_df[available_features]
    y = diagnostic_df['diagnosis_label']
    
    # Check for missing values
    if X.isnull().any().any():
        print("\n⚠️  Warning: Missing values detected. Filling with median values...")
        X = X.fillna(X.median())
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Class distribution:")
    for diagnosis, label in diagnosis_mapping.items():
        count = (y == label).sum()
        percentage = (count / len(y)) * 100
        print(f"  {diagnosis}: {count} ({percentage:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost model
    print("\n🚀 Training XGBoost classifier...")
    model_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(diagnosis_mapping),
        eval_metric='mlogloss',
        learning_rate=0.05,
        n_estimators=150,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=15,
        tree_method='hist'
    )
    
    eval_set = [(X_test, y_test)]
    model_xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Predictions
    y_pred = model_xgb.predict(X_test)
    y_pred_proba = model_xgb.predict_proba(X_test)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"🔹 Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🔹 Weighted F1-Score: {f1_weighted:.4f}")
    
    print("\n📊 Classification Report:")
    print("-" * 60)
    target_names = list(diagnosis_mapping.keys())
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    # Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    print("-" * 60)
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    plt.title("XGBoost - Anemia Diagnosis Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Diagnosis", fontsize=12)
    plt.ylabel("True Diagnosis", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/anemia_diagnosis_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n💾 Confusion matrix saved as 'models/anemia_diagnosis_confusion_matrix.png'")
    plt.show()
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 15 Feature Importances - Anemia Diagnosis', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/anemia_diagnosis_feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Feature importance plot saved as 'models/anemia_diagnosis_feature_importance.png'")
    plt.show()
    
    # Save model in pickle format (consistent with diabetes model)
    import pickle
    with open('models/anemia_diagnosis_xgb.pkl', 'wb') as f:
        pickle.dump(model_xgb, f)
    print("\n💾 Model saved as 'models/anemia_diagnosis_xgb.pkl'")
    
    # Save feature names for later use
    with open('models/anemia_diagnosis_features.json', 'w') as f:
        json.dump({
            'features': available_features,
            'diagnosis_mapping': diagnosis_mapping,
            'target_names': target_names
        }, f, indent=2)
    print("💾 Feature configuration saved as 'models/anemia_diagnosis_features.json'")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    
    return model_xgb, feature_importance

def load_and_prepare_data(json_prefix='generated_anemia_'):
    """
    Load data from generated JSON files and prepare for training.
    """
    print("\n📂 Loading data from JSON files...")
    
    try:
        # Load data
        patients = pd.read_json(f'{json_prefix}patients.json')
        visits = pd.read_json(f'{json_prefix}doctor_visits.json')
        diagnoses = pd.read_json(f'{json_prefix}diagnoses.json')
        lab_reports = pd.read_json(f'{json_prefix}lab_reports.json')
        lab_results = pd.read_json(f'{json_prefix}lab_test_results.json')
        progressions = pd.read_json(f'{json_prefix}disease_progressions.json')
        
        print(f"✅ Loaded {len(patients)} patients")
        print(f"✅ Loaded {len(visits)} visits")
        print(f"✅ Loaded {len(diagnoses)} diagnoses")
        print(f"✅ Loaded {len(lab_results)} lab test results")
        
        # Pivot lab results to get one row per report with all test values
        lab_pivot = lab_results.pivot_table(
            index='report_id',
            columns='test_name',
            values='test_value',
            aggfunc='first'
        ).reset_index()
        
        # Merge data step by step
        print("\n🔄 Merging data...")
        df = lab_reports.merge(lab_pivot, on='report_id', how='left')
        print(f"   After lab_pivot merge: {len(df)} rows, columns: {list(df.columns)[:5]}...")
        
        # Check if patient_id already exists in lab_reports
        if 'patient_id' not in df.columns:
            df = df.merge(visits[['visit_id', 'patient_id']], on='visit_id', how='left')
            print(f"   After visits merge: {len(df)} rows, has patient_id: {'patient_id' in df.columns}")
        
        df = df.merge(diagnoses[['visit_id', 'disease_name']], on='visit_id', how='left', suffixes=('', '_diag'))
        print(f"   After diagnoses merge: {len(df)} rows")
        
        df = df.merge(patients[['patient_id', 'gender']], on='patient_id', how='left', suffixes=('', '_pat'))
        print(f"   After patients merge: {len(df)} rows")
        
        # Rename disease_name to diagnosis for consistency
        df = df.rename(columns={'disease_name': 'diagnosis'})
        
        # Filter only anemia-related diagnoses
        anemia_diagnoses = [
            "Normal",
            "Iron Deficiency without Anemia",
            "Mild Iron Deficiency Anemia",
            "Moderate Iron Deficiency Anemia",
            "Severe Iron Deficiency Anemia"
        ]
        
        df = df[df['diagnosis'].isin(anemia_diagnoses)].copy()
        
        print(f"\n✅ Prepared {len(df)} records for training")
        
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

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IRON DEFICIENCY ANEMIA DIAGNOSIS MODEL")
    print("XGBoost Multi-Class Classification")
    print("="*60)
    
    # Option 1: Load from generated JSON files
    try:
        df = load_and_prepare_data('generated_anemia_')
    except Exception as e:
        print(f"\n⚠️  Error loading data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure you have run: python iron_deficiency_anemia_data_generation.py")
        exit(1)
    
    # Option 2: If df already exists in memory (uncomment if using notebook)
    # df = your_existing_dataframe
    
    # Train the model
    model, feature_importance = train_xgb_anemia_diagnostic_model(df)
    
    print("\n🎉 All done! Model is ready for deployment.")

