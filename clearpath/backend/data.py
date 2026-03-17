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
        # High acuity — clear deterioration trend Oct→Mar (staffing strain, winter surge)
        {"month": 1,  "patient_fall_rate": 2.9, "medication_error_rate": 3.5, "readmission_rate_30d": 16.9, "handoff_documentation_score": 84, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 26},
        {"month": 2,  "patient_fall_rate": 3.2, "medication_error_rate": 3.8, "readmission_rate_30d": 18.1, "handoff_documentation_score": 81, "nurse_patient_ratio": 2.1, "sepsis_response_time_minutes": 28},
        {"month": 3,  "patient_fall_rate": 3.5, "medication_error_rate": 4.0, "readmission_rate_30d": 18.5, "handoff_documentation_score": 80, "nurse_patient_ratio": 2.1, "sepsis_response_time_minutes": 28},
        {"month": 4,  "patient_fall_rate": 2.8, "medication_error_rate": 3.7, "readmission_rate_30d": 17.1, "handoff_documentation_score": 87, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 25},
        {"month": 5,  "patient_fall_rate": 2.9, "medication_error_rate": 3.8, "readmission_rate_30d": 17.3, "handoff_documentation_score": 88, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 24},
        {"month": 6,  "patient_fall_rate": 3.1, "medication_error_rate": 4.0, "readmission_rate_30d": 18.0, "handoff_documentation_score": 85, "nurse_patient_ratio": 2.1, "sepsis_response_time_minutes": 27},
        {"month": 7,  "patient_fall_rate": 3.4, "medication_error_rate": 4.3, "readmission_rate_30d": 19.0, "handoff_documentation_score": 83, "nurse_patient_ratio": 2.3, "sepsis_response_time_minutes": 30},
        {"month": 8,  "patient_fall_rate": 3.3, "medication_error_rate": 4.2, "readmission_rate_30d": 18.7, "handoff_documentation_score": 84, "nurse_patient_ratio": 2.2, "sepsis_response_time_minutes": 29},
        {"month": 9,  "patient_fall_rate": 2.5, "medication_error_rate": 2.9, "readmission_rate_30d": 14.2, "handoff_documentation_score": 90, "nurse_patient_ratio": 1.9, "sepsis_response_time_minutes": 21},
        {"month": 10, "patient_fall_rate": 2.1, "medication_error_rate": 2.6, "readmission_rate_30d": 13.2, "handoff_documentation_score": 91, "nurse_patient_ratio": 1.9, "sepsis_response_time_minutes": 21},
        {"month": 11, "patient_fall_rate": 2.4, "medication_error_rate": 3.0, "readmission_rate_30d": 14.5, "handoff_documentation_score": 88, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 23},
        {"month": 12, "patient_fall_rate": 2.7, "medication_error_rate": 3.3, "readmission_rate_30d": 15.8, "handoff_documentation_score": 86, "nurse_patient_ratio": 2.0, "sepsis_response_time_minutes": 24},
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
        # Clear deterioration Oct→Mar — handoff collapsing, sepsis response worsening
        {"month": 1,  "patient_fall_rate": 1.3, "medication_error_rate": 3.0, "readmission_rate_30d": 11.5, "handoff_documentation_score": 65, "nurse_patient_ratio": 4.3, "sepsis_response_time_minutes": 20},
        {"month": 2,  "patient_fall_rate": 1.5, "medication_error_rate": 3.2, "readmission_rate_30d": 12.1, "handoff_documentation_score": 62, "nurse_patient_ratio": 4.5, "sepsis_response_time_minutes": 22},
        {"month": 3,  "patient_fall_rate": 1.4, "medication_error_rate": 3.3, "readmission_rate_30d": 11.8, "handoff_documentation_score": 63, "nurse_patient_ratio": 4.4, "sepsis_response_time_minutes": 21},
        {"month": 4,  "patient_fall_rate": 1.3, "medication_error_rate": 3.2, "readmission_rate_30d": 11.5, "handoff_documentation_score": 65, "nurse_patient_ratio": 4.3, "sepsis_response_time_minutes": 20},
        {"month": 5,  "patient_fall_rate": 1.4, "medication_error_rate": 3.3, "readmission_rate_30d": 11.7, "handoff_documentation_score": 64, "nurse_patient_ratio": 4.4, "sepsis_response_time_minutes": 21},
        {"month": 6,  "patient_fall_rate": 1.7, "medication_error_rate": 3.9, "readmission_rate_30d": 12.8, "handoff_documentation_score": 57, "nurse_patient_ratio": 4.9, "sepsis_response_time_minutes": 25},
        {"month": 7,  "patient_fall_rate": 1.9, "medication_error_rate": 4.2, "readmission_rate_30d": 13.3, "handoff_documentation_score": 54, "nurse_patient_ratio": 5.2, "sepsis_response_time_minutes": 28},
        {"month": 8,  "patient_fall_rate": 1.8, "medication_error_rate": 4.1, "readmission_rate_30d": 13.0, "handoff_documentation_score": 55, "nurse_patient_ratio": 5.1, "sepsis_response_time_minutes": 27},
        {"month": 9,  "patient_fall_rate": 1.0, "medication_error_rate": 2.3, "readmission_rate_30d": 9.8,  "handoff_documentation_score": 78, "nurse_patient_ratio": 4.0, "sepsis_response_time_minutes": 16},
        {"month": 10, "patient_fall_rate": 1.0, "medication_error_rate": 2.5, "readmission_rate_30d": 10.2, "handoff_documentation_score": 76, "nurse_patient_ratio": 4.1, "sepsis_response_time_minutes": 17},
        {"month": 11, "patient_fall_rate": 1.1, "medication_error_rate": 2.7, "readmission_rate_30d": 10.8, "handoff_documentation_score": 72, "nurse_patient_ratio": 4.2, "sepsis_response_time_minutes": 18},
        {"month": 12, "patient_fall_rate": 1.2, "medication_error_rate": 2.9, "readmission_rate_30d": 11.2, "handoff_documentation_score": 68, "nurse_patient_ratio": 4.3, "sepsis_response_time_minutes": 19},
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


BENCHMARKS = {
    "icu": {
        "readmission_rate_30d": {"target": 12.0, "source": "CMS HRRP", "label": "30d Readmission"},
        "handoff_documentation_score": {"target": 90, "source": "Joint Commission NPSG", "label": "Handoff Score"},
        "nurse_patient_ratio": {"target": 2.0, "source": "ANA Staffing Standards", "label": "Nurse:Patient Ratio"},
        "sepsis_response_time_minutes": {"target": 20, "source": "CMS SEP-1", "label": "Sepsis Response"},
        "medication_error_rate": {"target": 3.0, "source": "NQF Safe Practice #18", "label": "Med Errors /1k"},
    },
    "med_surg": {
        "readmission_rate_30d": {"target": 10.0, "source": "CMS HRRP", "label": "30d Readmission"},
        "handoff_documentation_score": {"target": 85, "source": "Joint Commission NPSG", "label": "Handoff Score"},
        "nurse_patient_ratio": {"target": 5.0, "source": "ANA Staffing Standards", "label": "Nurse:Patient Ratio"},
        "medication_error_rate": {"target": 2.5, "source": "NQF Safe Practice #18", "label": "Med Errors /1k"},
    },
    "ed": {
        "handoff_documentation_score": {"target": 80, "source": "Joint Commission NPSG", "label": "Handoff Score"},
        "sepsis_response_time_minutes": {"target": 15, "source": "CMS SEP-1", "label": "Sepsis Response"},
        "medication_error_rate": {"target": 2.0, "source": "Leapfrog Safe Practices", "label": "Med Errors /1k"},
        "nurse_patient_ratio": {"target": 4.0, "source": "ANA Staffing Standards", "label": "Nurse:Patient Ratio"},
    },
    "oncology": {
        "readmission_rate_30d": {"target": 9.0, "source": "CMS HRRP", "label": "30d Readmission"},
        "handoff_documentation_score": {"target": 92, "source": "Joint Commission NPSG", "label": "Handoff Score"},
        "medication_error_rate": {"target": 1.5, "source": "NQF Safe Practice #18", "label": "Med Errors /1k"},
    },
}

SOAP_NOTES = {
    "icu": [
        {
            "note_id": "ICU-2026-0312",
            "date": "2026-03-04",
            "patient_id": "PT-88421",
            "provider": "RN C. Martinez",
            "note_type": "Incident Report",
            "subjective": "Night shift RN reported feeling rushed during 0700 handoff. Stated 'I didn't have time to finish SBAR on bed 4 because we were short-staffed and I had a rapid response at 0630.'",
            "objective": "Handoff documentation for bed 4 (post-cardiac surgery, day 2) was incomplete — medication reconciliation section blank, hemodynamic trend not documented. Oncoming RN administered scheduled vasopressor dose without awareness of titration change at 0500. Near-miss medication error caught by charge nurse during bedside verification.",
            "assessment": "Incomplete SBAR handoff due to staffing strain (1:3 ratio overnight vs. target 1:2) directly contributed to near-miss medication error. Pattern consistent with prior incidents on understaffed shifts.",
            "plan": "1) Mandatory bedside handoff verification for all vasoactive drips. 2) Escalate staffing concern to unit manager. 3) Schedule SBAR refresher training for night shift. 4) Implement 'critical med' flag in handoff template.",
        },
        {
            "note_id": "ICU-2026-0287",
            "date": "2026-02-18",
            "patient_id": "PT-77103",
            "provider": "RN A. Thompson",
            "note_type": "Progress Note",
            "subjective": "Patient (72M, septic shock day 3) family expressed concern that 'different nurses keep telling us different things about his progress.' Wife noted conflicting information about ventilator weaning plan.",
            "objective": "Review of nursing documentation reveals inconsistent care plan communication across 3 shift changes. Ventilator weaning protocol initiated by day shift was not reflected in evening handoff notes. FiO2 was increased by night shift without documented rationale. Sepsis response initiated at 42 min from alert (target: 20 min).",
            "assessment": "Communication breakdown across shifts contributing to care plan inconsistency. Delayed sepsis bundle completion linked to documentation gaps at handoff. Family trust eroding due to conflicting information.",
            "plan": "1) Implement daily interdisciplinary rounding summary in handoff template. 2) Assign primary nurse for continuity. 3) Family conference to align on care goals. 4) Sepsis response time audit for Q1.",
        },
    ],
    "med_surg": [
        {
            "note_id": "MS-2026-0445",
            "date": "2026-03-08",
            "patient_id": "PT-91205",
            "provider": "RN J. Davis",
            "note_type": "Incident Report",
            "subjective": "Patient (68F, post-hip replacement day 1) found on floor beside bed at 0215. Patient states 'I tried to get to the bathroom but the call light was out of reach.' No loss of consciousness reported.",
            "objective": "Fall risk assessment (Morse scale) was overdue by 8 hours — last documented on admission. Bed alarm was not activated despite high fall risk designation. Nurse assigned 6 patients at time of fall (unit target 1:5). Patient sustained left wrist contusion, no fracture on x-ray. Call light placed on non-operative side, out of patient reach.",
            "assessment": "Preventable patient fall linked to missed reassessment and inadequate safety precautions. High nurse-patient ratio (1:6) contributed to delayed fall risk screening and missed bed alarm activation. Third fall on this unit in 30 days.",
            "plan": "1) Immediate fall risk reassessment for all patients on unit. 2) Bed alarm compliance audit. 3) Call light placement standardization protocol. 4) Request staffing review — 1:6 ratio unsustainable for fall prevention. 5) Schedule AHRQ fall prevention bundle refresher.",
        },
        {
            "note_id": "MS-2026-0398",
            "date": "2026-02-22",
            "patient_id": "PT-84367",
            "provider": "RN K. Patel",
            "note_type": "Progress Note",
            "subjective": "Patient (74M, CHF exacerbation) reports increasing shortness of breath overnight. States 'Nobody checked my weight this morning.' Daily weights ordered for fluid management.",
            "objective": "Daily weight not recorded for 2 consecutive days. I/O documentation incomplete — output not recorded for evening shift. BNP trending upward (890 → 1240 pg/mL). Patient gained 2.3 kg when finally weighed. Diuretic dose adjustment delayed by 14 hours due to missing weight data.",
            "assessment": "Incomplete monitoring documentation (daily weights, I/O) delayed recognition of fluid overload progression. Staffing ratio of 1:6 on evening shift likely contributed to missed assessments. Risk of preventable readmission if fluid management not optimized.",
            "plan": "1) Immediate diuretic dose adjustment per CHF protocol. 2) Mandatory weight documentation at 0600 daily — add to nursing workflow checklist. 3) I/O documentation compliance audit for unit. 4) Educate patient on daily weight self-monitoring for post-discharge.",
        },
    ],
    "ed": [
        {
            "note_id": "ED-2026-0621",
            "date": "2026-03-10",
            "patient_id": "PT-95533",
            "provider": "RN M. Chen",
            "note_type": "Discharge Summary",
            "subjective": "Patient (58M) presented with fever 101.8F, tachycardia (HR 112), altered mental status per family. Family states symptoms started 6 hours ago. 'We waited in the lobby for over an hour before being seen.'",
            "objective": "Triage to physician assessment: 47 minutes (target: 15 min for ESI-2). Sepsis screening triggered at triage but workup initiation delayed — blood cultures drawn at 52 min, lactate at 58 min, antibiotics at 71 min (SEP-1 target: 60 min from recognition). ED census: 42 patients with 8 boarders occupying treatment spaces. Nurse-patient ratio at time of care: 1:5.2.",
            "assessment": "Sepsis bundle compliance failure — antibiotics administered 11 minutes past SEP-1 window. Contributing factors: ED overcrowding with 8 boarders reducing effective treatment spaces, high nurse-patient ratio, and delayed triage-to-assessment time. Pattern consistent with boarding-related delays documented in prior months.",
            "plan": "1) Initiate standing sepsis order set at triage for ESI-2 patients with SIRS criteria. 2) Escalate boarding situation to hospital administration — 8 boarders critically impacting ED throughput. 3) Request dedicated sepsis response nurse during high-census periods. 4) Schedule SEP-1 bundle compliance refresher for all ED nursing staff.",
        },
        {
            "note_id": "ED-2026-0589",
            "date": "2026-02-28",
            "patient_id": "PT-93201",
            "provider": "RN S. Williams",
            "note_type": "Incident Report",
            "subjective": "Charge nurse flagged incomplete handoff for patient transferred from ED to Med-Surg. Receiving RN states 'I didn't know the patient was on a heparin drip — it wasn't in the handoff and wasn't scanned in the transfer meds.'",
            "objective": "ED-to-inpatient handoff documentation missing: active IV medications (heparin drip 800 units/hr), pending lab results (troponin T2), and code status clarification. Transfer occurred during shift change with 4 simultaneous admissions in progress. Handoff documentation score for this transfer: 35/100.",
            "assessment": "Critical medication information lost during ED-to-inpatient transfer. High-risk handoff failure during peak volume period. Heparin drip running without receiving unit awareness — potential for dosing error or unmonitored anticoagulation.",
            "plan": "1) Immediate medication reconciliation for transferred patient. 2) Implement mandatory 'critical medication' checkbox in ED transfer template. 3) No-interruption policy during handoff communication. 4) I-PASS handoff training for all ED-to-inpatient transitions.",
        },
    ],
    "oncology": [
        {
            "note_id": "ONC-2026-0156",
            "date": "2026-03-01",
            "patient_id": "PT-82910",
            "provider": "RN L. Nguyen",
            "note_type": "Readmission Note",
            "subjective": "Patient (61F, AML, post-chemo cycle 3) presents with fever 102.4F, rigors, and productive cough. Discharged 5 days ago. Patient states 'They told me to come back if I got a fever but I wasn't sure how high was too high. The discharge papers were confusing.'",
            "objective": "Neutropenic (ANC 280). Blood cultures drawn — preliminary gram-positive cocci. Prior discharge education documentation: 3 minutes total education time recorded, standardized print handout provided, no teach-back documented. Discharge summary lacks specific fever threshold (100.4F) — only states 'return for fever.' Previous admission LOS: 4 days.",
            "assessment": "Preventable readmission in immunocompromised patient. Abbreviated discharge education (3 min, no teach-back) failed to convey actionable fever parameters. Patient delayed seeking care by ~18 hours due to unclear guidance. Consistent with department trend of rising readmission rates.",
            "plan": "1) Admit to isolation, start empiric broad-spectrum antibiotics per neutropenic fever protocol. 2) Implement mandatory teach-back for all oncology discharge education. 3) Revise discharge materials — specific numeric thresholds, visual aids, multilingual options. 4) Track discharge education time as quality metric (target: minimum 15 min for chemo patients).",
        },
        {
            "note_id": "ONC-2026-0142",
            "date": "2026-02-14",
            "patient_id": "PT-80654",
            "provider": "RN R. Jackson",
            "note_type": "Progress Note",
            "subjective": "Patient (55M, stage III colorectal CA, day 2 post-colectomy) reports pain 8/10 despite PCA. States 'The night nurse didn't believe me about my pain. She said I was hitting the button too much.'",
            "objective": "PCA utilization log shows 47 attempts, 12 deliveries in 8-hour period. Pain reassessment not documented for 6 hours (standard: q2h post-op). Multimodal pain protocol ordered but scheduled ketorolac doses missed x2 — documented as 'held, patient sleeping' but PCA log contradicts. Patient has history of opioid tolerance documented in anesthesia pre-op note but not reflected in nursing care plan.",
            "assessment": "Inadequate pain management due to missed multimodal medication doses and failure to incorporate opioid tolerance history into care plan. Pain reassessment documentation gap (6 hours) violates unit standard. Risk of prolonged recovery, patient dissatisfaction, and potential for uncontrolled pain-related complications.",
            "plan": "1) Acute pain service consult for PCA adjustment accounting for opioid tolerance. 2) Resume multimodal protocol — administer overdue ketorolac. 3) Pain reassessment q2h with documentation compliance. 4) Schedule oncology pain management CE focusing on opioid-tolerant patients. 5) Update nursing care plan to reflect anesthesia pre-op opioid history.",
        },
    ],
}

BEST_PRACTICE_REFERENCES = [
    {"id": "AHRQ-IPASS", "name": "I-PASS Handoff", "source": "AHRQ", "description": "Standardized handoff communication framework (Illness severity, Patient summary, Action list, Situation awareness, Synthesis by receiver)"},
    {"id": "AHRQ-FALL", "name": "Fall Prevention Bundle", "source": "AHRQ", "description": "Evidence-based fall prevention program including universal fall precautions, individualized interventions, and post-fall huddle protocol"},
    {"id": "NQF-SP18", "name": "Safe Practice #18", "source": "NQF", "description": "Medication reconciliation at all care transitions — standardized processes to prevent medication errors during handoffs"},
    {"id": "CMS-SEP1", "name": "SEP-1 Measure", "source": "CMS", "description": "Early management bundle for severe sepsis/septic shock — 3-hour and 6-hour bundle elements with specific timing requirements"},
    {"id": "CMS-HRRP", "name": "Hospital Readmissions Reduction Program", "source": "CMS", "description": "Penalizes hospitals with excess 30-day readmissions for targeted conditions — drives focus on discharge planning and transitions of care"},
    {"id": "ANA-STAFF", "name": "Staffing Standards", "source": "ANA", "description": "ANA Principles for Nurse Staffing — evidence-based nurse-to-patient ratios by unit type to ensure safe patient care"},
    {"id": "JC-NPSG", "name": "National Patient Safety Goals", "source": "Joint Commission", "description": "Annual safety priorities including patient identification, communication improvement, medication safety, and fall reduction"},
    {"id": "JC-HANDOFF", "name": "Handoff Communication Standard", "source": "Joint Commission", "description": "PC.02.02.01 — standardized approach to handoff communication including opportunity to ask and respond to questions"},
    {"id": "LEAPFROG-SP", "name": "Safe Practices Score", "source": "Leapfrog", "description": "National Quality Forum-endorsed safe practices survey — measures hospital adherence to evidence-based safety processes"},
    {"id": "AHRQ-CUSP", "name": "CUSP Toolkit", "source": "AHRQ", "description": "Comprehensive Unit-based Safety Program — structured approach to identify and mitigate unit-level safety hazards"},
    {"id": "ANA-SCOPE", "name": "Scope and Standards of Practice", "source": "ANA", "description": "Defines competent-level nursing practice including assessment, diagnosis, outcomes identification, planning, implementation, and evaluation"},
    {"id": "AHRQ-TEAMSTEPPS", "name": "TeamSTEPPS", "source": "AHRQ", "description": "Team Strategies and Tools to Enhance Performance and Patient Safety — evidence-based teamwork system for healthcare"},
]


def get_department_metrics(department_id: str, month: int) -> dict | None:
    data = EMR_DATA.get(department_id)
    if not data:
        return None
    for record in data:
        if record["month"] == month:
            return {**record, "department_id": department_id, "department_name": DEPARTMENTS[department_id], "year": 2026}
    return None


def get_all_months_for_department(department_id: str) -> list[dict] | None:
    data = EMR_DATA.get(department_id)
    if not data:
        return None
    return [
        {**record, "department_id": department_id, "department_name": DEPARTMENTS[department_id], "year": 2026}
        for record in data
    ]
