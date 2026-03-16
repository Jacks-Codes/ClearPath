"""
Mock EMR data for 4 departments across 12 months.
Metrics per month:
  - patient_fall_rate: falls per 1000 patient days
  - medication_error_rate: errors per 1000 medication administrations
  - readmission_rate_30d: % readmitted within 30 days
  - handoff_documentation_score: 0–100 completeness score
  - nurse_patient_ratio: patients per nurse
  - sepsis_response_time_minutes: avg minutes from alert to intervention
"""

DEPARTMENTS = {
    "icu": "ICU",
    "med_surg": "Medical-Surgical",
    "ed": "Emergency Department",
    "oncology": "Oncology",
}

# Month indices 0–11 correspond to Jan–Dec
EMR_DATA: dict[str, list[dict]] = {
    "icu": [
        # High acuity — elevated fall rates due to sedation/delirium, tight ratios, sepsis vigilance
        {"month": 1,  "patient_fall_rate": 3.2, "medication_error_rate": 4.1, "readmission_rate_30d": 18.5, "handoff_documentation_score": 84, "nurse_patient_ratio": 2.1, "sepsis_response_time_minutes": 28},
        {"month": 2,  "patient_fall_rate": 3.5, "medication_error_rate": 4.4, "readmission_rate_30d": 19.2, "handoff_documentation_score": 82, "nurse_patient_ratio": 2.2, "sepsis_response_time_minutes": 31},
        {"month": 3,  "patient_fall_rate": 3.0, "medication_error_rate": 3.9, "readmission_rate_30d": 17.8, "handoff_documentation_score": 86, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 26},
        {"month": 4,  "patient_fall_rate": 2.8, "medication_error_rate": 3.7, "readmission_rate_30d": 17.1, "handoff_documentation_score": 87, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 25},
        {"month": 5,  "patient_fall_rate": 2.9, "medication_error_rate": 3.8, "readmission_rate_30d": 17.3, "handoff_documentation_score": 88, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 24},
        {"month": 6,  "patient_fall_rate": 3.1, "medication_error_rate": 4.0, "readmission_rate_30d": 18.0, "handoff_documentation_score": 85, "nurse_patient_ratio": 2.1, "sepsis_response_time_minutes": 27},
        {"month": 7,  "patient_fall_rate": 3.4, "medication_error_rate": 4.3, "readmission_rate_30d": 19.0, "handoff_documentation_score": 83, "nurse_patient_ratio": 2.3, "sepsis_response_time_minutes": 30},
        {"month": 8,  "patient_fall_rate": 3.3, "medication_error_rate": 4.2, "readmission_rate_30d": 18.7, "handoff_documentation_score": 84, "nurse_patient_ratio": 2.2, "sepsis_response_time_minutes": 29},
        {"month": 9,  "patient_fall_rate": 3.0, "medication_error_rate": 3.9, "readmission_rate_30d": 18.1, "handoff_documentation_score": 86, "nurse_patient_ratio": 2.1, "sepsis_response_time_minutes": 27},
        {"month": 10, "patient_fall_rate": 2.9, "medication_error_rate": 3.8, "readmission_rate_30d": 17.6, "handoff_documentation_score": 87, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 25},
        {"month": 11, "patient_fall_rate": 3.3, "medication_error_rate": 4.2, "readmission_rate_30d": 19.0, "handoff_documentation_score": 83, "nurse_patient_ratio": 2.2, "sepsis_response_time_minutes": 32},
        {"month": 12, "patient_fall_rate": 3.6, "medication_error_rate": 4.5, "readmission_rate_30d": 20.1, "handoff_documentation_score": 80, "nurse_patient_ratio": 2.4, "sepsis_response_time_minutes": 35},
    ],
    "med_surg": [
        # Staffing strain visible in ratio and handoff scores; mid-year deterioration
        {"month": 1,  "patient_fall_rate": 2.1, "medication_error_rate": 2.8, "readmission_rate_30d": 14.2, "handoff_documentation_score": 76, "nurse_patient_ratio": 5.2, "sepsis_response_time_minutes": 42},
        {"month": 2,  "patient_fall_rate": 2.3, "medication_error_rate": 3.0, "readmission_rate_30d": 14.8, "handoff_documentation_score": 74, "nurse_patient_ratio": 5.4, "sepsis_response_time_minutes": 45},
        {"month": 3,  "patient_fall_rate": 2.0, "medication_error_rate": 2.7, "readmission_rate_30d": 13.9, "handoff_documentation_score": 78, "nurse_patient_ratio": 5.1, "sepsis_response_time_minutes": 40},
        {"month": 4,  "patient_fall_rate": 1.9, "medication_error_rate": 2.5, "readmission_rate_30d": 13.5, "handoff_documentation_score": 79, "nurse_patient_ratio": 5.0, "sepsis_response_time_minutes": 38},
        {"month": 5,  "patient_fall_rate": 2.0, "medication_error_rate": 2.6, "readmission_rate_30d": 13.7, "handoff_documentation_score": 78, "nurse_patient_ratio": 5.1, "sepsis_response_time_minutes": 39},
        {"month": 6,  "patient_fall_rate": 2.4, "medication_error_rate": 3.1, "readmission_rate_30d": 15.1, "handoff_documentation_score": 72, "nurse_patient_ratio": 5.7, "sepsis_response_time_minutes": 48},
        {"month": 7,  "patient_fall_rate": 2.7, "medication_error_rate": 3.4, "readmission_rate_30d": 15.9, "handoff_documentation_score": 69, "nurse_patient_ratio": 6.1, "sepsis_response_time_minutes": 53},
        {"month": 8,  "patient_fall_rate": 2.8, "medication_error_rate": 3.5, "readmission_rate_30d": 16.2, "handoff_documentation_score": 67, "nurse_patient_ratio": 6.3, "sepsis_response_time_minutes": 55},
        {"month": 9,  "patient_fall_rate": 2.5, "medication_error_rate": 3.2, "readmission_rate_30d": 15.0, "handoff_documentation_score": 71, "nurse_patient_ratio": 5.8, "sepsis_response_time_minutes": 47},
        {"month": 10, "patient_fall_rate": 2.2, "medication_error_rate": 2.9, "readmission_rate_30d": 14.1, "handoff_documentation_score": 75, "nurse_patient_ratio": 5.3, "sepsis_response_time_minutes": 43},
        {"month": 11, "patient_fall_rate": 2.4, "medication_error_rate": 3.1, "readmission_rate_30d": 14.7, "handoff_documentation_score": 73, "nurse_patient_ratio": 5.5, "sepsis_response_time_minutes": 46},
        {"month": 12, "patient_fall_rate": 2.6, "medication_error_rate": 3.3, "readmission_rate_30d": 15.5, "handoff_documentation_score": 70, "nurse_patient_ratio": 5.9, "sepsis_response_time_minutes": 50},
    ],
    "ed": [
        # Chronically low handoff documentation scores; high volume drives errors
        {"month": 1,  "patient_fall_rate": 1.5, "medication_error_rate": 3.5, "readmission_rate_30d": 12.0, "handoff_documentation_score": 61, "nurse_patient_ratio": 4.5, "sepsis_response_time_minutes": 22},
        {"month": 2,  "patient_fall_rate": 1.6, "medication_error_rate": 3.7, "readmission_rate_30d": 12.4, "handoff_documentation_score": 59, "nurse_patient_ratio": 4.6, "sepsis_response_time_minutes": 23},
        {"month": 3,  "patient_fall_rate": 1.4, "medication_error_rate": 3.3, "readmission_rate_30d": 11.8, "handoff_documentation_score": 63, "nurse_patient_ratio": 4.4, "sepsis_response_time_minutes": 21},
        {"month": 4,  "patient_fall_rate": 1.3, "medication_error_rate": 3.2, "readmission_rate_30d": 11.5, "handoff_documentation_score": 65, "nurse_patient_ratio": 4.3, "sepsis_response_time_minutes": 20},
        {"month": 5,  "patient_fall_rate": 1.4, "medication_error_rate": 3.3, "readmission_rate_30d": 11.7, "handoff_documentation_score": 64, "nurse_patient_ratio": 4.4, "sepsis_response_time_minutes": 21},
        {"month": 6,  "patient_fall_rate": 1.7, "medication_error_rate": 3.9, "readmission_rate_30d": 12.8, "handoff_documentation_score": 57, "nurse_patient_ratio": 4.9, "sepsis_response_time_minutes": 25},
        {"month": 7,  "patient_fall_rate": 1.9, "medication_error_rate": 4.2, "readmission_rate_30d": 13.3, "handoff_documentation_score": 54, "nurse_patient_ratio": 5.2, "sepsis_response_time_minutes": 28},
        {"month": 8,  "patient_fall_rate": 1.8, "medication_error_rate": 4.1, "readmission_rate_30d": 13.0, "handoff_documentation_score": 55, "nurse_patient_ratio": 5.1, "sepsis_response_time_minutes": 27},
        {"month": 9,  "patient_fall_rate": 1.6, "medication_error_rate": 3.7, "readmission_rate_30d": 12.3, "handoff_documentation_score": 60, "nurse_patient_ratio": 4.7, "sepsis_response_time_minutes": 24},
        {"month": 10, "patient_fall_rate": 1.5, "medication_error_rate": 3.5, "readmission_rate_30d": 12.0, "handoff_documentation_score": 62, "nurse_patient_ratio": 4.5, "sepsis_response_time_minutes": 22},
        {"month": 11, "patient_fall_rate": 1.7, "medication_error_rate": 3.8, "readmission_rate_30d": 12.6, "handoff_documentation_score": 58, "nurse_patient_ratio": 4.8, "sepsis_response_time_minutes": 24},
        {"month": 12, "patient_fall_rate": 1.8, "medication_error_rate": 4.0, "readmission_rate_30d": 12.9, "handoff_documentation_score": 56, "nurse_patient_ratio": 5.0, "sepsis_response_time_minutes": 26},
    ],
    "oncology": [
        # Generally cleaner metrics; notable readmission spike in summer months (June–August)
        {"month": 1,  "patient_fall_rate": 1.2, "medication_error_rate": 1.8, "readmission_rate_30d": 11.0, "handoff_documentation_score": 91, "nurse_patient_ratio": 3.8, "sepsis_response_time_minutes": 20},
        {"month": 2,  "patient_fall_rate": 1.3, "medication_error_rate": 1.9, "readmission_rate_30d": 11.2, "handoff_documentation_score": 90, "nurse_patient_ratio": 3.9, "sepsis_response_time_minutes": 21},
        {"month": 3,  "patient_fall_rate": 1.1, "medication_error_rate": 1.7, "readmission_rate_30d": 10.8, "handoff_documentation_score": 92, "nurse_patient_ratio": 3.7, "sepsis_response_time_minutes": 19},
        {"month": 4,  "patient_fall_rate": 1.1, "medication_error_rate": 1.7, "readmission_rate_30d": 10.6, "handoff_documentation_score": 93, "nurse_patient_ratio": 3.7, "sepsis_response_time_minutes": 18},
        {"month": 5,  "patient_fall_rate": 1.2, "medication_error_rate": 1.8, "readmission_rate_30d": 10.9, "handoff_documentation_score": 92, "nurse_patient_ratio": 3.8, "sepsis_response_time_minutes": 19},
        {"month": 6,  "patient_fall_rate": 1.4, "medication_error_rate": 2.0, "readmission_rate_30d": 14.7, "handoff_documentation_score": 88, "nurse_patient_ratio": 4.0, "sepsis_response_time_minutes": 22},
        {"month": 7,  "patient_fall_rate": 1.5, "medication_error_rate": 2.1, "readmission_rate_30d": 16.3, "handoff_documentation_score": 87, "nurse_patient_ratio": 4.1, "sepsis_response_time_minutes": 23},
        {"month": 8,  "patient_fall_rate": 1.4, "medication_error_rate": 2.0, "readmission_rate_30d": 15.8, "handoff_documentation_score": 88, "nurse_patient_ratio": 4.0, "sepsis_response_time_minutes": 22},
        {"month": 9,  "patient_fall_rate": 1.2, "medication_error_rate": 1.8, "readmission_rate_30d": 11.5, "handoff_documentation_score": 91, "nurse_patient_ratio": 3.8, "sepsis_response_time_minutes": 20},
        {"month": 10, "patient_fall_rate": 1.1, "medication_error_rate": 1.7, "readmission_rate_30d": 10.7, "handoff_documentation_score": 93, "nurse_patient_ratio": 3.7, "sepsis_response_time_minutes": 18},
        {"month": 11, "patient_fall_rate": 1.2, "medication_error_rate": 1.8, "readmission_rate_30d": 11.0, "handoff_documentation_score": 91, "nurse_patient_ratio": 3.8, "sepsis_response_time_minutes": 20},
        {"month": 12, "patient_fall_rate": 1.3, "medication_error_rate": 1.9, "readmission_rate_30d": 11.3, "handoff_documentation_score": 90, "nurse_patient_ratio": 3.9, "sepsis_response_time_minutes": 21},
    ],
}


def get_department_metrics(department_id: str, month: int) -> dict | None:
    data = EMR_DATA.get(department_id)
    if not data:
        return None
    for record in data:
        if record["month"] == month:
            return {**record, "department_id": department_id, "department_name": DEPARTMENTS[department_id], "year": 2025}
    return None


def get_all_months_for_department(department_id: str) -> list[dict] | None:
    data = EMR_DATA.get(department_id)
    if not data:
        return None
    return [
        {**record, "department_id": department_id, "department_name": DEPARTMENTS[department_id], "year": 2025}
        for record in data
    ]
