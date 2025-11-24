# LifeChain AI - Healthcare Management System

A comprehensive FastAPI-based healthcare management system with ML-powered disease diagnosis and progression monitoring. Features unified patient records, family tree tracking, multi-lab integration, and multi-language support for healthcare providers in non-English speaking countries.

## 🎯 Project Overview

LifeChain AI is designed to manage healthcare records for every individual from birth, similar to how NADRA maintains national identity records. The system provides:

- **Unified Patient Records**: Single CNIC-based identity across all healthcare facilities
- **ML-Powered Diagnosis**: Automated disease diagnosis using XGBoost and LSTM models
- **Progression Monitoring**: Time-series disease progression tracking
- **Family Tree Integration**: Automatic disease risk assessment from family history
- **Multi-Lab Support**: Consolidate lab reports from different laboratories
- **Multi-Language Support**: Translation service for non-English speaking healthcare providers
- **Auto-Diagnosis**: Automatic creation of suspected diagnoses from ML predictions

## ✨ Key Features

### 1. **Unified Patient Management**
- CNIC-based patient identification
- Complete lifetime medical records
- Doctor-as-Patient architecture (doctors are patients with additional fields)
- Family relationship tracking with auto-inference

### 2. **ML-Powered Disease Diagnosis & Progression**
- **Supported Diseases:**
  - **Diabetes**: Diagnosis and progression monitoring
  - **Iron Deficiency Anemia**: Diagnosis and progression monitoring
  - **Chronic Kidney Disease (CKD)**: Diagnosis and progression monitoring

- **ML Models:**
  - XGBoost for multi-class disease diagnosis
  - BiLSTM for sequence-based progression prediction
  - Cross-validation and regularization for generalization

### 3. **Intelligent Features**
- **Auto-Diagnosis**: Automatically creates "suspected" diagnoses from ML predictions
- **Lab Review Visits**: Auto-creates visits when lab results need review
- **Family Tree Auto-Inference**: Automatically infers related family relationships
- **Reference Range Validation**: Automatic abnormal value detection for lab tests

### 4. **Multi-Language Support**
- Translation service using Groq LLM (Llama 3.1)
- Supports: English, Urdu, Arabic, Hindi, Bengali
- Translates medical terminology accurately
- Applied to all text fields in API responses

### 5. **Comprehensive Lab Management**
- Multi-lab support
- Lab report upload and management
- Test result tracking with reference ranges
- Abnormal value detection

## 🏗️ Architecture

```
lifechain-ai-fyp/
├── app/                          # Main application
│   ├── api/v1/                   # API routes
│   │   ├── endpoints/            # Individual endpoint modules
│   │   │   ├── patients.py      # Patient CRUD & family relationships
│   │   │   ├── doctors.py       # Doctor management
│   │   │   ├── visits.py        # Doctor visits, symptoms, diagnoses
│   │   │   ├── labs.py          # Lab & lab report management
│   │   │   ├── unified_inference.py  # ML prediction endpoints
│   │   │   ├── progression_report.py # Progression reports & analytics
│   │   │   └── health.py        # Health check
│   │   └── dependencies.py      # Translation dependencies
│   ├── core/                     # Core configuration
│   │   ├── config.py            # Settings & environment variables
│   │   └── test_reference_ranges.py  # Lab test reference ranges
│   ├── db/                       # Database session management
│   ├── models/                   # SQLAlchemy ORM models
│   │   ├── patient.py           # Patient model (includes doctor fields)
│   │   ├── visit.py             # DoctorVisit, Symptom, Diagnosis, Prescription
│   │   ├── lab.py               # Lab, LabReport, LabTestResult
│   │   ├── family.py            # FamilyRelationship, FamilyDiseaseHistory
│   │   └── ml.py                # MLPrediction, DiseaseProgression
│   ├── schemas/                  # Pydantic validation schemas
│   └── services/                 # Business logic layer
│       ├── multi_disease_inference.py  # ML model loading & inference
│       ├── model_loader.py       # PyTorch model loading utilities
│       ├── translation.py       # Multi-language translation service
│       └── progression_report_service.py  # Progression report generation
├── model_training/               # ML model training scripts
│   ├── diabetic_diagnosis_xgb.py
│   ├── diabetic_progression_lstm.py
│   ├── anemia_diagnosis_xgb.py
│   ├── anemia_progression_lstm.py
│   ├── ckd_diagnosis_xgb.py
│   ├── ckd_progression_lstm.py
│   ├── train_models.py
│   └── test_models.py
├── data generation/              # Data generation & seeding scripts
│   ├── enhanced_data_generation.py
│   ├── iron_deficiency_anemia_data_generation.py
│   ├── ckd_data_generation.py
│   ├── seed_database.py
│   ├── seed_anemia_database.py
│   └── seed_ckd_database.py
├── models/                       # Trained ML models (saved models)
├── migrations/                   # Alembic database migrations
└── tests/                        # Test suite
```

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI (async web framework)
- **Database**: PostgreSQL (local instance)
- **ORM**: SQLAlchemy 2.0 (async support)
- **Migrations**: Alembic
- **Validation**: Pydantic v2
- **ML Framework**: 
  - XGBoost (diagnosis models)
  - PyTorch (LSTM progression models)
  - scikit-learn (preprocessing, evaluation)
- **AI/Translation**: LangChain Groq (Llama 3.1)
- **Other**: 
  - joblib (model serialization)
  - imbalanced-learn (SMOTE for class balancing)

## 📊 Database Schema

### Core Entities

#### **Patient** (Unified Entity)
- `patient_id` (UUID, Primary Key)
- `cnic` (String, Unique, Indexed) - National ID
- `first_name`, `last_name`
- `date_of_birth`, `gender` (Enum: male, female, other)
- `blood_group`, `phone`, `email`, `address`
- **Doctor-specific fields** (nullable):
  - `is_doctor` (Boolean)
  - `specialization` (String)
  - `license_number` (String, Unique)
  - `hospital_affiliation` (String)

#### **FamilyRelationship**
- Links patients via `patient_id` and `relative_patient_id`
- `relationship_type` (Enum: parent, child, sibling, spouse, grandparent, etc.)
- `is_blood_relative` (Boolean)

#### **FamilyDiseaseHistory**
- Tracks diseases in family members
- `disease_name`, `severity` (Enum: mild, moderate, severe)
- Links to `FamilyRelationship`

#### **DoctorVisit**
- `visit_id` (UUID, Primary Key)
- `patient_id` → Patient
- `doctor_patient_id` → Patient (where `is_doctor=True`)
- `visit_type` (Enum: consultation, follow_up, routine_checkup, lab_review, emergency)
- `visit_date`, `chief_complaint`, `doctor_notes`

#### **Symptom**
- Linked to `visit_id`
- `symptom_name`, `severity`, `notes`

#### **Diagnosis**
- Linked to `visit_id`
- `disease_name`
- `diagnosis_date`
- `status` (Enum: suspected, confirmed)
- `confidence_score`, `ml_model_used`, `notes`

#### **Prescription**
- Linked to `visit_id`
- `medication_name`, `dosage`, `frequency`, `duration`, `instructions`

#### **Lab**
- `lab_id` (UUID, Primary Key)
- `lab_name`, `lab_location`
- `accreditation_number`, `phone`, `email`

#### **LabReport**
- `report_id` (UUID, Primary Key)
- `patient_id` → Patient
- `lab_id` → Lab
- `visit_id` → DoctorVisit (nullable)
- `report_date`, `report_type`
- `status` (Enum: pending, completed)

#### **LabTestResult**
- Linked to `report_id`
- `test_name`, `test_value`, `unit`
- `reference_range`, `is_abnormal`

#### **DiseaseProgression**
- Time-series progression tracking
- `patient_id` → Patient
- `disease_name`
- `stage`, `progression_date`
- `ml_features` (JSONB) - ML model features

#### **MLPrediction**
- Audit trail for ML predictions
- `patient_id` → Patient
- `disease_name`, `prediction_type` (diagnosis/progression)
- `prediction_result` (JSONB), `model_version`, `timestamp`

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- Virtual environment (recommended)

### 1. Clone Repository

```bash
git clone <repository-url>
cd lifechain-ai-fyp
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
# Database Configuration (Local PostgreSQL)
DATABASE_URL=postgresql+asyncpg://postgres:abbas@localhost:5432/lifechain-ai
DIRECT_DATABASE_URL=postgresql+psycopg2://postgres:abbas@localhost:5432/lifechain-ai

# AI/ML APIs
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Application Settings
APP_NAME=LifeChain API
ENV=development
DEBUG=True
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS (comma-separated list)
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

**Important**: 
- Replace `postgres` and `abbas` with your PostgreSQL username and password
- Replace `lifechain-ai` with your database name
- Add your actual API keys

### 5. Database Setup

#### Create Database

```sql
CREATE DATABASE lifechain_ai;
```

#### Run Migrations

```bash
# Generate initial migration (if needed)
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### 6. Seed Database (Optional)

```bash
# Seed with diabetes data
python "data generation/seed_database.py" --seed

# Seed with anemia data
python "data generation/seed_anemia_database.py" --seed

# Seed with CKD data
python "data generation/seed_ckd_database.py" --seed
```

### 7. Run Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

API will be available at:
- **Base URL**: `http://localhost:8001`
- **Interactive Docs**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## 📡 API Documentation

### Base URL

All endpoints are prefixed with `/api/v1`

### Authentication

Currently, the API does not require authentication (development mode). Authentication will be added in future releases.

### Translation Support

Most endpoints support a `lang` query parameter for translation:
- `lang=en` (default) - English
- `lang=ur` - Urdu
- `lang=ar` - Arabic
- `lang=hi` - Hindi
- `lang=bn` - Bengali

---

## 🏥 Patient Management APIs

### Create Patient
```http
POST /api/v1/patients
Content-Type: application/json

{
  "cnic": "42101-1234567-1",
  "first_name": "John",
  "last_name": "Doe",
  "date_of_birth": "1990-01-15",
  "gender": "male",
  "blood_group": "O+",
  "phone": "03001234567",
  "email": "john.doe@example.com",
  "address": "123 Main St, City"
}
```

### Get Patients (List)
```http
GET /api/v1/patients?skip=0&limit=100&search=john
```

### Get Patient by ID
```http
GET /api/v1/patients/{patient_id}
```

### Update Patient
```http
PUT /api/v1/patients/{patient_id}
Content-Type: application/json

{
  "phone": "03001234568",
  "email": "newemail@example.com"
}
```

### Delete Patient
```http
DELETE /api/v1/patients/{patient_id}
```

### Create Family Relationship
```http
POST /api/v1/patients/{patient_id}/family-relationships
Content-Type: application/json

{
  "relative_patient_id": "uuid-of-relative",
  "relationship_type": "parent",
  "is_blood_relative": true
}
```

### Create Family Relationship (Auto-Inference)
```http
POST /api/v1/patients/{patient_id}/family-relationships/auto
Content-Type: application/json

{
  "relative_patient_id": "uuid-of-relative",
  "relationship_type": "parent",
  "is_blood_relative": true,
  "auto_infer": true,
  "max_depth": 10
}
```

**Auto-Inference**: Automatically creates related relationships (e.g., if you add a parent, it will also create relationships with siblings, grandparents, etc.)

### Get Family Disease History
```http
GET /api/v1/patients/{patient_id}/family-disease-history?lang=ur
```

### Get Patients with Family Tree
```http
GET /api/v1/patients/with-family-tree/list?skip=0&limit=100
```

---

## 👨‍⚕️ Doctor Management APIs

### Create Doctor
**Note**: Doctors must first exist as patients. This endpoint adds doctor-specific fields to an existing patient.

```http
POST /api/v1/doctors
Content-Type: application/json

{
  "patient_id": "uuid-of-existing-patient",
  "specialization": "Cardiologist",
  "license_number": "PMDC-12345",
  "hospital_affiliation": "City Hospital"
}
```

### Get Doctors (List)
```http
GET /api/v1/doctors?skip=0&limit=100
```

### Get Doctor by ID
```http
GET /api/v1/doctors/{doctor_id}
```

### Update Doctor
```http
PUT /api/v1/doctors/{doctor_id}
Content-Type: application/json

{
  "specialization": "Neurologist",
  "hospital_affiliation": "New Hospital"
}
```

### Delete Doctor
**Note**: This removes doctor status but keeps the patient record.

```http
DELETE /api/v1/doctors/{doctor_id}
```

### Get Specializations List
```http
GET /api/v1/doctors/specializations/list
```

---

## 🏥 Visit Management APIs

### Create Visit
```http
POST /api/v1/visits
Content-Type: application/json

{
  "patient_id": "uuid-of-patient",
  "doctor_patient_id": "uuid-of-doctor-patient",
  "visit_date": "2024-01-15T10:00:00",
  "visit_type": "consultation",
  "chief_complaint": "Headache and fever",
  "doctor_notes": "Patient reports persistent headache"
}
```

### Get Visits (List)
```http
GET /api/v1/visits?patient_id={uuid}&doctor_patient_id={uuid}&visit_type=consultation&skip=0&limit=100&lang=ur
```

### Get Visit by ID
```http
GET /api/v1/visits/{visit_id}?lang=ur
```

### Update Visit
```http
PUT /api/v1/visits/{visit_id}
Content-Type: application/json

{
  "doctor_notes": "Updated notes"
}
```

### Delete Visit
```http
DELETE /api/v1/visits/{visit_id}
```

### Add Symptom to Visit
```http
POST /api/v1/visits/{visit_id}/symptoms
Content-Type: application/json

{
  "symptom_name": "Headache",
  "severity": "moderate",
  "notes": "Persistent for 3 days"
}
```

### Get Symptoms for Visit
```http
GET /api/v1/visits/{visit_id}/symptoms?lang=ur
```

### Add Diagnosis to Visit
```http
POST /api/v1/visits/{visit_id}/diagnoses
Content-Type: application/json

{
  "disease_name": "Diabetes Type 2",
  "diagnosis_date": "2024-01-15",
  "status": "confirmed",
  "confidence_score": 0.95,
  "ml_model_used": "xgb_diabetes_v1",
  "notes": "Confirmed based on lab results"
}
```

### Get Diagnoses for Visit
```http
GET /api/v1/visits/{visit_id}/diagnoses?lang=ur
```

### Add Prescription to Visit
```http
POST /api/v1/visits/{visit_id}/prescriptions
Content-Type: application/json

{
  "medication_name": "Metformin",
  "dosage": "500mg",
  "frequency": "Twice daily",
  "duration": "30 days",
  "instructions": "Take with meals"
}
```

### Get Prescriptions for Visit
```http
GET /api/v1/visits/{visit_id}/prescriptions?lang=ur
```

---

## 🧪 Lab Management APIs

### Create Lab
```http
POST /api/v1/labs
Content-Type: application/json

{
  "lab_name": "City Diagnostic Lab",
  "lab_location": "123 Medical Street",
  "accreditation_number": "PAL-12345",
  "phone": "03001234567",
  "email": "info@lab.com"
}
```

### Get Labs (List)
```http
GET /api/v1/labs?skip=0&limit=100
```

### Get Lab by ID
```http
GET /api/v1/labs/{lab_id}
```

### Update Lab
```http
PUT /api/v1/labs/{lab_id}
Content-Type: application/json

{
  "phone": "03001234568"
}
```

### Delete Lab
```http
DELETE /api/v1/labs/{lab_id}
```

### Create Lab Report
```http
POST /api/v1/labs/reports
Content-Type: application/json

{
  "patient_id": "uuid-of-patient",
  "lab_id": "uuid-of-lab",
  "visit_id": "uuid-of-visit",  // Optional
  "report_date": "2024-01-15T10:00:00",
  "report_type": "blood_test",
  "status": "completed"
}
```

### Get Lab Reports (List)
```http
GET /api/v1/labs/reports?patient_id={uuid}&lab_id={uuid}&status=completed&skip=0&limit=100&lang=ur
```

### Get Lab Report by ID
```http
GET /api/v1/labs/reports/{report_id}?lang=ur
```

### Update Lab Report
```http
PUT /api/v1/labs/reports/{report_id}
Content-Type: application/json

{
  "status": "completed"
}
```

### Delete Lab Report
```http
DELETE /api/v1/labs/reports/{report_id}
```

### Add Test Result to Report
```http
POST /api/v1/labs/reports/{report_id}/test-results
Content-Type: application/json

{
  "test_name": "Hemoglobin",
  "test_value": 12.5,
  "unit": "g/dL",
  "reference_range": "12.0-15.5",
  "is_abnormal": false
}
```

### Get Test Results for Report
```http
GET /api/v1/labs/reports/{report_id}/test-results?lang=ur
```

### Get Abnormal Test Results
```http
GET /api/v1/labs/test-results/abnormal?patient_id={uuid}&skip=0&limit=100&lang=ur
```

### Get Supported Tests
```http
GET /api/v1/labs/tests/supported?lang=ur
```

---

## 🤖 ML Inference APIs

### Get Supported Diseases
```http
GET /api/v1/ml/diseases
```

**Response:**
```json
{
  "supported_diseases": ["diabetes", "anemia", "ckd"],
  "disease_info": {
    "diabetes": {
      "diagnosis_features": ["fasting_glucose", "hba1c", ...],
      "progression_features": null
    },
    "anemia": {
      "diagnosis_features": ["hemoglobin", "hematocrit", ...],
      "progression_features": ["hemoglobin", "hematocrit", ...]
    },
    "ckd": {
      "diagnosis_features": ["serum_creatinine", "egfr", ...],
      "progression_features": ["serum_creatinine", "egfr", ...]
    }
  }
}
```

### Get Model Info
```http
GET /api/v1/ml/models/info
```

### Predict Diagnosis (Direct)
```http
POST /api/v1/ml/diagnosis/predict
Content-Type: application/json

{
  "disease_name": "diabetes",
  "features": {
    "fasting_glucose": 120,
    "hba1c": 7.5,
    "hdl": 45,
    "ldl": 120,
    "triglycerides": 150,
    "total_cholesterol": 200,
    "creatinine": 1.0,
    "bmi": 28,
    "systolic_bp": 130,
    "diastolic_bp": 85
  }
}
```

**Response:**
```json
{
  "disease": "diabetes",
  "diagnosis": "Diabetes Type 2",
  "confidence": 0.92,
  "probabilities": {
    "Normal": 0.05,
    "Prediabetes": 0.03,
    "Diabetes Type 2": 0.92
  },
  "input_features": {...}
}
```

### Predict Diagnosis (Patient-Based with Auto-Save)
```http
POST /api/v1/ml/diagnosis/patient/{patient_id}?disease_name=diabetes&auto_save=true&lang=ur
```

**Features:**
- Automatically fetches latest lab results for the patient
- Makes prediction using appropriate ML model
- If `auto_save=true` (default):
  - Finds or creates a "lab_review" visit (within 7 days)
  - Creates a diagnosis with status "suspected"
  - Returns diagnosis ID and visit ID

**Response:**
```json
{
  "patient_id": "uuid",
  "disease": "diabetes",
  "prediction_type": "diagnosis",
  "diagnosis": "Diabetes Type 2",
  "confidence": 0.92,
  "probabilities": {...},
  "diagnosis_saved": true,
  "diagnosis_id": "uuid",
  "visit_id": "uuid",
  "diagnosis_status": "suspected",
  "note": "Diagnosis automatically created with 'suspected' status. Doctor review recommended."
}
```

### Predict Progression (Direct)
```http
POST /api/v1/ml/progression/predict
Content-Type: application/json

{
  "disease_name": "ckd",
  "sequence": [
    {
      "serum_creatinine": 1.2,
      "egfr": 65,
      "uacr": 45,
      "bun": 22,
      "sodium": 140,
      "potassium": 4.2,
      "calcium": 9.8,
      "phosphorus": 3.8,
      "hemoglobin": 13.5,
      "pth": 45,
      "bicarbonate": 24,
      "albumin": 4.2,
      "bmi": 26,
      "systolic_bp": 128,
      "diastolic_bp": 82
    },
    {
      "serum_creatinine": 1.4,
      "egfr": 58,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "disease": "ckd",
  "progression": "Slowly Progressing",
  "confidence": 0.85,
  "probabilities": {
    "Normal": 0.05,
    "Stable": 0.10,
    "Slowly Progressing": 0.85
  },
  "num_visits": 2
}
```

### Predict Progression (Patient-Based)
```http
POST /api/v1/ml/progression/patient/{patient_id}?disease_name=ckd&lang=ur
```

Automatically fetches patient's historical lab data and creates sequences for progression prediction.

### ML Health Check
```http
GET /api/v1/ml/health
```

---

## 📊 Progression Reports APIs

### Get Progression Report
```http
GET /api/v1/reports/patient/{patient_id}/progression-report?disease_name=diabetes&lang=ur
```

**Response includes:**
- Current disease stage
- Progression trend
- Risk factors
- Recommendations

### Get Progression Timeline
```http
GET /api/v1/reports/patient/{patient_id}/progression-timeline?disease_name=diabetes
```

### Get Risk Assessment
```http
GET /api/v1/reports/patient/{patient_id}/risk-assessment?disease_name=diabetes
```

### Get Recommendations
```http
GET /api/v1/reports/patient/{patient_id}/recommendations?disease_name=diabetes&lang=ur
```

### Get Family History
```http
GET /api/v1/reports/patient/{patient_id}/family-history?disease_name=diabetes&lang=ur
```

### Predict Future Progression
```http
POST /api/v1/reports/patient/{patient_id}/predict-progression
Content-Type: application/json

{
  "disease_name": "diabetes",
  "future_months": 6
}
```

### Get Lab Measurements Timeline
```http
GET /api/v1/reports/patient/{patient_id}/lab-measurements-timeline?disease_name=diabetes&test_name=hemoglobin
```

---

## 🧬 ML Models Details

### Diabetes

**Diagnosis Model (XGBoost)**
- **Features**: `fasting_glucose`, `hba1c`, `hdl`, `ldl`, `triglycerides`, `total_cholesterol`, `creatinine`, `bmi`, `systolic_bp`, `diastolic_bp`
- **Classes**: Normal, Prediabetes, Diabetes Type 1, Diabetes Type 2
- **Model File**: `models/diabetes_diagnosis_xgb.pkl`

**Progression Model (LSTM)**
- **Features**: Same as diagnosis
- **Classes**: Normal, Stable, Improving, Slowly Progressing, Rapidly Progressing
- **Model File**: `models/diabetes_progression_lstm.pth`

### Iron Deficiency Anemia

**Diagnosis Model (XGBoost)**
- **Features**: `hemoglobin`, `hematocrit`, `mcv`, `mch`, `mchc`, `rdw`, `serum_iron`, `ferritin`, `tibc`, `transferrin_saturation`, `reticulocyte_count`, `wbc`, `platelet_count`, `esr`, `bmi`, `systolic_bp`, `diastolic_bp`
- **Classes**: Normal, Mild Anemia, Moderate Anemia, Severe Anemia
- **Model File**: `models/anemia_diagnosis_xgb.pkl`

**Progression Model (BiLSTM)**
- **Features**: Same as diagnosis
- **Classes**: Normal, Improving, Stable, Slowly Progressing, Rapidly Progressing, Critical
- **Model File**: `models/anemia_progression_lstm.pth`

### Chronic Kidney Disease (CKD)

**Diagnosis Model (XGBoost)**
- **Features**: `serum_creatinine`, `egfr`, `uacr`, `bun`, `sodium`, `potassium`, `calcium`, `phosphorus`, `hemoglobin`, `pth`, `bicarbonate`, `albumin`, `bmi`, `systolic_bp`, `diastolic_bp`
- **Classes**: Normal Kidney Function, CKD Stage 1, CKD Stage 2, CKD Stage 3a, CKD Stage 3b, CKD Stage 4, CKD Stage 5/ESRD
- **Model File**: `models/ckd_diagnosis_xgb_model.pkl`

**Progression Model (BiLSTM)**
- **Features**: Same as diagnosis
- **Classes**: Normal, Stable, Slowly Progressing, Rapidly Progressing, Improving, ESRD
- **Model File**: `models/ckd_progression_lstm_model.pth`

### Model Training

Models are trained using scripts in `model_training/`:

```bash
# Train diabetes models
python model_training/diabetic_diagnosis_xgb.py
python model_training/diabetic_progression_lstm.py

# Train anemia models
python model_training/anemia_diagnosis_xgb.py
python model_training/anemia_progression_lstm.py

# Train CKD models
python model_training/ckd_diagnosis_xgb.py
python model_training/ckd_progression_lstm.py

# Train all models
python model_training/train_models.py
```

**Training Features:**
- 5-fold cross-validation
- SMOTE for class imbalance (diagnosis models)
- Regularization (dropout, weight decay)
- Data augmentation (noise injection)
- Early stopping

---

## 🌐 Translation Service

The translation service uses Groq's Llama 3.1 model to translate medical terminology accurately.

### Supported Languages
- `en` - English (default)
- `ur` - Urdu
- `ar` - Arabic
- `hi` - Hindi
- `bn` - Bengali

### Usage

Add `?lang=ur` to any endpoint that returns text data:

```http
GET /api/v1/visits?lang=ur
GET /api/v1/patients/{patient_id}/family-disease-history?lang=ar
GET /api/v1/ml/diagnosis/patient/{patient_id}?disease_name=diabetes&lang=hi
```

### Translated Fields

- **Visits**: `chief_complaint`, `doctor_notes`
- **Symptoms**: `symptom_name`, `notes`
- **Diagnoses**: `disease_name`, `notes`
- **Prescriptions**: `medication_name`, `instructions`
- **Lab Reports**: `test_name`, `description`
- **Family History**: `disease_name`, `notes`

**Note**: Enum fields (like `visit_type`, `status`) are NOT translated to maintain schema validation.

---

## 🔄 Auto-Diagnosis Feature

When calling `/api/v1/ml/diagnosis/patient/{patient_id}` with `auto_save=true`:

1. **Fetches Latest Lab Results**: Retrieves the most recent lab test results for required features
2. **Makes ML Prediction**: Uses the appropriate model to predict diagnosis
3. **Finds or Creates Visit**:
   - Looks for a recent visit (within 7 days)
   - If none found, creates a new "lab_review" visit
   - Assigns any available doctor in the system
4. **Creates Diagnosis**:
   - Status: "suspected"
   - Links to the visit
   - Includes confidence score and model info
   - Adds note recommending doctor review

**Benefits:**
- No manual entry required
- Automatic visit creation when needed
- Doctor can review and change status to "confirmed"
- Complete audit trail

---

## 📦 Data Generation

Synthetic data generation scripts are available in `data generation/`:

### Generate Data

```bash
# Generate diabetes data
python "data generation/enhanced_data_generation.py"

# Generate anemia data
python "data generation/iron_deficiency_anemia_data_generation.py"

# Generate CKD data
python "data generation/ckd_data_generation.py"
```

### Seed Database

```bash
# Seed diabetes data
python "data generation/seed_database.py" --seed

# Seed anemia data
python "data generation/seed_anemia_database.py" --seed

# Seed CKD data
python "data generation/seed_ckd_database.py" --seed

# Clear and seed (development only)
python "data generation/seed_database.py" --clear --seed
```

---

## 🧪 Development

### Run Tests

```bash
pytest
```

### Create Migration

```bash
alembic revision --autogenerate -m "Description of changes"
alembic upgrade head
```

### Database Reset (Development Only)

```bash
# Downgrade all migrations
alembic downgrade base

# Upgrade to latest
alembic upgrade head
```

### Code Structure Guidelines

- **Models**: SQLAlchemy ORM models in `app/models/`
- **Schemas**: Pydantic validation schemas in `app/schemas/`
- **Endpoints**: FastAPI route handlers in `app/api/v1/endpoints/`
- **Services**: Business logic in `app/services/`
- **Dependencies**: Reusable dependencies in `app/api/v1/dependencies.py`

---

## 🔐 Security Notes

**Current Status**: Authentication is not implemented (development mode)

**Planned Features:**
- JWT authentication
- Role-based access control (Doctor, Patient, Admin, Lab)
- HIPAA-compliant data handling
- Audit logging
- Rate limiting

---

## 📝 API Response Format

### Success Response
```json
{
  "field1": "value1",
  "field2": "value2"
}
```

### Error Response
```json
{
  "detail": "Error message here"
}
```

### Pagination
List endpoints support `skip` and `limit` query parameters:
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum number of records to return (default: 100, max: 1000)

---

## 🐛 Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check `.env` file has correct database credentials
- Ensure database exists: `CREATE DATABASE lifechain_ai;`

### Migration Issues
- If migrations fail, check `migrations/versions/` for conflicts
- Reset migrations: `alembic downgrade base && alembic upgrade head`

### Model Loading Issues
- Ensure model files exist in `models/` directory
- Check model file paths in `app/services/multi_disease_inference.py`

### Translation Issues
- Verify `GROQ_API_KEY` is set in `.env`
- Check API quota/limits

---

## 📄 License

MIT License

---

## 👥 Contributors

LifeChain AI Team - Final Year Project

---

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Last Updated**: November 2024
