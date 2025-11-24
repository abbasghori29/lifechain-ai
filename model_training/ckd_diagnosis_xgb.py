"""
CKD Diagnosis Model Training - XGBoost Multi-Class Classification
Trains a model to diagnose CKD stages from lab test results
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

def train_xgb_ckd_diagnostic_model(diagnostic_df):
    """
    Train XGBoost model for CKD diagnosis classification
    
    Args:
        diagnostic_df: DataFrame with features and diagnosis labels
    
    Returns:
        Trained model and feature importance
    """
    print("\n" + "="*60)
    print("TRAINING CKD DIAGNOSIS MODEL (XGBoost)")
    print("="*60)
    
    # Define features for CKD diagnosis
    features = [
        'serum_creatinine', 'egfr', 'uacr', 'bun',
        'sodium', 'potassium', 'calcium', 'phosphorus',
        'hemoglobin', 'pth', 'bicarbonate', 'albumin',
        'bmi', 'systolic_bp', 'diastolic_bp'
    ]
    
    # Check which features are available
    available_features = [f for f in features if f in diagnostic_df.columns]
    missing_features = [f for f in features if f not in diagnostic_df.columns]
    
    if missing_features:
        print(f"\n⚠️  Missing features: {missing_features}")
        print(f"   Filling with median values...")
        for feat in missing_features:
            diagnostic_df[feat] = diagnostic_df[feat] if feat in diagnostic_df.columns else 0
    
    print(f"\n✅ Using {len(available_features)} features:")
    print(f"   {', '.join(available_features)}")
    
    # Prepare features and target
    X = diagnostic_df[features].fillna(diagnostic_df[features].median())
    y = diagnostic_df['diagnosis']
    
    # Filter to CKD-related diagnoses
    ckd_diagnoses = [
        "Normal Kidney Function",
        "Early CKD Stage 1",
        "Early CKD Stage 2",
        "Moderate CKD Stage 3a",
        "Moderate CKD Stage 3b",
        "Advanced CKD Stage 4",
        "End Stage Renal Disease (ESRD)"
    ]
    
    # Map similar diagnoses
    diagnosis_mapping = {
        "Normal": "Normal Kidney Function",
        "Normal Kidney Function": "Normal Kidney Function",
        "CKD Stage 1": "Early CKD Stage 1",
        "Early CKD Stage 1": "Early CKD Stage 1",
        "CKD Stage 2": "Early CKD Stage 2",
        "Early CKD Stage 2": "Early CKD Stage 2",
        "CKD Stage 3a": "Moderate CKD Stage 3a",
        "Moderate CKD Stage 3a": "Moderate CKD Stage 3a",
        "CKD Stage 3b": "Moderate CKD Stage 3b",
        "Moderate CKD Stage 3b": "Moderate CKD Stage 3b",
        "CKD Stage 4": "Advanced CKD Stage 4",
        "Advanced CKD Stage 4": "Advanced CKD Stage 4",
        "CKD Stage 5 (ESRD)": "End Stage Renal Disease (ESRD)",
        "End Stage Renal Disease (ESRD)": "End Stage Renal Disease (ESRD)"
    }
    
    y_mapped = y.map(diagnosis_mapping).fillna("Normal Kidney Function")
    diagnostic_df['diagnosis'] = y_mapped
    
    # Filter to valid diagnoses
    mask = diagnostic_df['diagnosis'].isin(ckd_diagnoses)
    X = X[mask]
    y = diagnostic_df.loc[mask, 'diagnosis']
    
    # Encode labels to numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n📊 Dataset size: {len(X)} records")
    print(f"📊 Number of classes: {len(label_encoder.classes_)}")
    print(f"\nClass distribution:")
    for i, diagnosis in enumerate(label_encoder.classes_):
        count = (y_encoded == i).sum()
        percentage = (count / len(y_encoded)) * 100
        print(f"  {diagnosis}: {count} ({percentage:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✅ Train set: {len(X_train)} records")
    print(f"✅ Test set: {len(X_test)} records")
    
    # Calculate class weights for imbalanced classes
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    
    print("\n📊 Running 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            objective='multi:softprob'
        ),
        X_train, y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"   Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Apply SMOTE to balance classes (optional - can be toggled)
    print("\n🔄 Applying SMOTE to balance classes...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(X_train[y_train == y_train.min()]) - 1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {len(X_train_balanced)} records (was {len(X_train)})")
        
        # Show new class distribution
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        print("   Balanced class distribution:")
        for label, count in zip(label_encoder.inverse_transform(unique), counts):
            print(f"     {label}: {count}")
        
        X_train_final = X_train_balanced
        y_train_final = y_train_balanced
        use_sample_weights = False
    except Exception as e:
        print(f"   ⚠️  SMOTE failed: {str(e)}")
        print(f"   Using class weights instead...")
        X_train_final = X_train
        y_train_final = y_train
        use_sample_weights = True
    
    # Train XGBoost model
    print("\n🚀 Training XGBoost model...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        objective='multi:softprob',
        early_stopping_rounds=20,
        scale_pos_weight=1  # Will use sample_weight if SMOTE not used
    )
    
    if use_sample_weights:
        model_xgb.fit(
            X_train_final, y_train_final,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    else:
        model_xgb.fit(
            X_train_final, y_train_final,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    
    # Make predictions
    y_pred = model_xgb.predict(X_test)
    y_pred_proba = model_xgb.predict_proba(X_test)
    
    # Convert back to class names for reporting
    y_test_names = label_encoder.inverse_transform(y_test)
    y_pred_names = label_encoder.inverse_transform(y_pred)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ F1-Score (Macro): {f1_macro:.4f}")
    print(f"✅ F1-Score (Weighted): {f1_weighted:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test_names, y_pred_names))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test_names, y_pred_names)
    print(cm)
    
    # Calculate per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_idx = i
        if class_idx < len(cm):
            correct = cm[class_idx, class_idx]
            total = cm[class_idx, :].sum()
            class_acc = correct / total if total > 0 else 0
            print(f"   {class_name}: {class_acc:.4f} ({correct}/{total})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ckd_diagnosis_xgb_model.pkl'
    joblib.dump(model_xgb, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save feature importance
    importance_path = 'models/ckd_diagnosis_feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(feature_importance.to_dict('records'), f, indent=2)
    print(f"💾 Feature importance saved to: {importance_path}")
    
    # Save feature list
    feature_list_path = 'models/ckd_diagnosis_features.json'
    with open(feature_list_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"💾 Feature list saved to: {feature_list_path}")
    
    # Save label encoder
    encoder_path = 'models/ckd_diagnosis_label_encoder.pkl'
    joblib.dump(label_encoder, encoder_path)
    print(f"💾 Label encoder saved to: {encoder_path}")
    
    return model_xgb, feature_importance

def load_and_prepare_data(json_prefix='generated_ckd_'):
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
        
        # Filter only CKD-related diagnoses
        ckd_diagnoses = [
            "Normal Kidney Function",
            "Normal",
            "Early CKD Stage 1",
            "CKD Stage 1",
            "Early CKD Stage 2",
            "CKD Stage 2",
            "Moderate CKD Stage 3a",
            "CKD Stage 3a",
            "Moderate CKD Stage 3b",
            "CKD Stage 3b",
            "Advanced CKD Stage 4",
            "CKD Stage 4",
            "End Stage Renal Disease (ESRD)",
            "CKD Stage 5 (ESRD)"
        ]
        
        df = df[df['diagnosis'].isin(ckd_diagnoses)].copy()
        
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
    print("CHRONIC KIDNEY DISEASE (CKD) DIAGNOSIS MODEL")
    print("XGBoost Multi-Class Classification")
    print("="*60)
    
    # Option 1: Load from generated JSON files
    try:
        df = load_and_prepare_data('generated_ckd_')
    except Exception as e:
        print(f"\n⚠️  Error loading data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure you have run: python ckd_data_generation.py")
        exit(1)
    
    # Train the model
    model, feature_importance = train_xgb_ckd_diagnostic_model(df)
    
    print("\n🎉 All done! Model is ready for deployment.")

