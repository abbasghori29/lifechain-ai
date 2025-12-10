"""
Comprehensive Test Script for LifeChain AI API
Tests all ML Inference and Progression Report endpoints for demo patient
"""
import requests
import json
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8080"
PATIENT_ID = "98e81961-2cf3-4ee2-956d-5bdd9026532a"

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def test_endpoint(method: str, endpoint: str, description: str, data: Dict = None, params: Dict = None) -> Dict[str, Any]:
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"{Colors.BOLD}Testing: {description}{Colors.RESET}")
    print(f"  {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, params=params, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Status: {response.status_code}")
            print_info(f"Response keys: {list(result.keys()) if isinstance(result, dict) else 'List/Array'}")
            return {"success": True, "status": response.status_code, "data": result}
        else:
            error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
            print_error(f"Status: {response.status_code}")
            print_error(f"Error: {error_detail}")
            return {"success": False, "status": response.status_code, "error": error_detail}
    
    except requests.exceptions.ConnectionError:
        print_error("Connection failed - Is the server running?")
        return {"success": False, "error": "Connection failed"}
    except requests.exceptions.Timeout:
        print_error("Request timeout")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    """Run all endpoint tests"""
    print_header("LIFECHAIN AI - COMPREHENSIVE API TEST SUITE")
    print(f"Patient ID: {PATIENT_ID}")
    print(f"Base URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "ml_inference": {},
        "progression_reports": {}
    }
    
    # ============================================================
    # ML INFERENCE ENDPOINTS
    # ============================================================
    print_header("ML INFERENCE ENDPOINTS")
    
    # 1. Health Check
    results["ml_inference"]["health"] = test_endpoint(
        "GET",
        "/api/v1/ml/health",
        "ML Service Health Check"
    )
    
    # 2. Get Model Info
    results["ml_inference"]["model_info"] = test_endpoint(
        "GET",
        "/api/v1/ml/models/info",
        "Get Model Information"
    )
    
    # 3. Get Supported Diseases
    results["ml_inference"]["supported_diseases"] = test_endpoint(
        "GET",
        "/api/v1/ml/diseases",
        "Get Supported Diseases"
    )
    
    # 4. Predict Diagnosis (Direct)
    results["ml_inference"]["diagnosis_predict"] = test_endpoint(
        "POST",
        "/api/v1/ml/diagnosis/predict",
        "Predict Diagnosis (Direct - Diabetes)",
        data={
            "disease_name": "diabetes",
            "features": {
                "fasting_glucose": 120,
                "hba1c": 6.5,
                "hdl": 45,
                "ldl": 130,
                "triglycerides": 180,
                "total_cholesterol": 210,
                "creatinine": 1.1,
                "bmi": 28,
                "systolic_bp": 135,
                "diastolic_bp": 85
            }
        }
    )
    
    # 5. Predict Progression (Direct)
    results["ml_inference"]["progression_predict"] = test_endpoint(
        "POST",
        "/api/v1/ml/progression/predict",
        "Predict Progression (Direct - Diabetes)",
        data={
            "disease_name": "diabetes",
            "sequence": [
                {
                    "fasting_glucose": 118,
                    "hba1c": 6.2,
                    "hdl": 42,
                    "ldl": 135,
                    "triglycerides": 165,
                    "total_cholesterol": 210,
                    "creatinine": 1.1,
                    "bmi": 28,
                    "systolic_bp": 130,
                    "diastolic_bp": 82
                },
                {
                    "fasting_glucose": 105,
                    "hba1c": 5.9,
                    "hdl": 45,
                    "ldl": 125,
                    "triglycerides": 150,
                    "total_cholesterol": 195,
                    "creatinine": 1.05,
                    "bmi": 27,
                    "systolic_bp": 128,
                    "diastolic_bp": 80
                },
                {
                    "fasting_glucose": 98,
                    "hba1c": 5.7,
                    "hdl": 48,
                    "ldl": 115,
                    "triglycerides": 140,
                    "total_cholesterol": 188,
                    "creatinine": 1.0,
                    "bmi": 26,
                    "systolic_bp": 125,
                    "diastolic_bp": 78
                }
            ]
        }
    )
    
    # 6. Predict Patient Diagnosis (Diabetes)
    results["ml_inference"]["patient_diagnosis_diabetes"] = test_endpoint(
        "POST",
        f"/api/v1/ml/diagnosis/patient/{PATIENT_ID}",
        "Predict Patient Diagnosis (Diabetes)",
        params={"disease_name": "diabetes", "auto_save": "false", "lang": "en"}
    )
    
    # 7. Predict Patient Diagnosis (CKD)
    results["ml_inference"]["patient_diagnosis_ckd"] = test_endpoint(
        "POST",
        f"/api/v1/ml/diagnosis/patient/{PATIENT_ID}",
        "Predict Patient Diagnosis (CKD)",
        params={"disease_name": "ckd", "auto_save": "false", "lang": "en"}
    )
    
    # 8. Predict Patient Diagnosis (Anemia)
    results["ml_inference"]["patient_diagnosis_anemia"] = test_endpoint(
        "POST",
        f"/api/v1/ml/diagnosis/patient/{PATIENT_ID}",
        "Predict Patient Diagnosis (Anemia)",
        params={"disease_name": "anemia", "auto_save": "false", "lang": "en"}
    )
    
    # 9. Predict Patient Progression (Diabetes)
    results["ml_inference"]["patient_progression_diabetes"] = test_endpoint(
        "POST",
        f"/api/v1/ml/progression/patient/{PATIENT_ID}",
        "Predict Patient Progression (Diabetes)",
        params={"disease_name": "diabetes", "lang": "en"}
    )
    
    # 10. Predict Patient Progression (CKD)
    results["ml_inference"]["patient_progression_ckd"] = test_endpoint(
        "POST",
        f"/api/v1/ml/progression/patient/{PATIENT_ID}",
        "Predict Patient Progression (CKD)",
        params={"disease_name": "ckd", "lang": "en"}
    )
    
    # ============================================================
    # PROGRESSION REPORTS ENDPOINTS
    # ============================================================
    print_header("PROGRESSION REPORTS ENDPOINTS")
    
    # 11. Get Progression Report (Diabetes)
    results["progression_reports"]["progression_report_diabetes"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/progression-report",
        "Get Progression Report (Diabetes)",
        params={"disease_name": "diabetes", "months_back": "36"}
    )
    
    # 12. Get Progression Report (CKD)
    results["progression_reports"]["progression_report_ckd"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/progression-report",
        "Get Progression Report (CKD)",
        params={"disease_name": "ckd", "months_back": "36"}
    )
    
    # 13. Get Progression Timeline (Diabetes)
    results["progression_reports"]["progression_timeline_diabetes"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/progression-timeline",
        "Get Progression Timeline (Diabetes)",
        params={"disease_name": "diabetes", "months_back": "36"}
    )
    
    # 14. Get Progression Timeline (CKD)
    results["progression_reports"]["progression_timeline_ckd"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/progression-timeline",
        "Get Progression Timeline (CKD)",
        params={"disease_name": "ckd", "months_back": "36"}
    )
    
    # 15. Get Risk Assessment
    results["progression_reports"]["risk_assessment"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/risk-assessment",
        "Get Risk Assessment (Genetic Risk)"
    )
    
    # 16. Get Recommendations
    results["progression_reports"]["recommendations"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/recommendations",
        "Get AI Recommendations"
    )
    
    # 17. Get Family Disease History (Diabetes)
    results["progression_reports"]["family_history_diabetes"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/family-history",
        "Get Family Disease History (Diabetes)",
        params={"disease_name": "diabetes"}
    )
    
    # 18. Get Family Disease History (CKD)
    results["progression_reports"]["family_history_ckd"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/family-history",
        "Get Family Disease History (CKD)",
        params={"disease_name": "ckd"}
    )
    
    # 19. Predict Future Progression
    results["progression_reports"]["predict_future"] = test_endpoint(
        "POST",
        f"/api/v1/reports/patient/{PATIENT_ID}/predict-progression",
        "Predict Future Progression (All Conditions)",
        params={"months_ahead": "6"}
    )
    
    # 20. Get Lab Measurements Timeline (All)
    results["progression_reports"]["lab_timeline_all"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/lab-measurements-timeline",
        "Get Lab Measurements Timeline (All Tests)",
        params={"months_back": "36"}
    )
    
    # 21. Get Lab Measurements Timeline (HbA1c)
    results["progression_reports"]["lab_timeline_hba1c"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/lab-measurements-timeline",
        "Get Lab Measurements Timeline (HbA1c)",
        params={"test_name": "hba1c", "months_back": "36"}
    )
    
    # 22. Get Lab Measurements Timeline (Creatinine)
    results["progression_reports"]["lab_timeline_creatinine"] = test_endpoint(
        "GET",
        f"/api/v1/reports/patient/{PATIENT_ID}/lab-measurements-timeline",
        "Get Lab Measurements Timeline (Creatinine)",
        params={"test_name": "creatinine", "months_back": "36"}
    )
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print_header("TEST SUMMARY")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for category, tests in results.items():
        print(f"\n{Colors.BOLD}{category.upper().replace('_', ' ')}{Colors.RESET}")
        for test_name, result in tests.items():
            total_tests += 1
            if result.get("success"):
                passed_tests += 1
                print_success(f"{test_name}: PASSED")
            else:
                failed_tests += 1
                print_error(f"{test_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}Total Tests: {total_tests}{Colors.RESET}")
    print_success(f"Passed: {passed_tests}")
    if failed_tests > 0:
        print_error(f"Failed: {failed_tests}")
    else:
        print_success(f"Failed: {failed_tests}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")
    
    # Save results to file
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "patient_id": PATIENT_ID,
            "test_time": datetime.now().isoformat(),
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests
            },
            "results": results
        }, f, indent=2)
    
    print_info(f"Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
    except Exception as e:
        print_error(f"Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

