"""
Structured seasonal knowledge base mapping months to known healthcare risk patterns.
Each entry covers affected departments, risk level, and recommended CE topic areas.
"""

SEASONAL_RISKS = [
    {
        "name": "Respiratory Illness Season",
        "months": [11, 12, 1, 2],
        "affected_departments": ["ICU", "Medical-Surgical", "Emergency Department"],
        "risk_level": "high",
        "ce_topic_areas": [
            "Ventilator management and weaning protocols",
            "Respiratory isolation and infection control",
            "Early deterioration recognition (NEWS2/MEWS scoring)",
            "Oxygen therapy and non-invasive ventilation",
            "Influenza and RSV triage protocols",
        ],
        "description": (
            "November through February brings peak respiratory illness burden including "
            "influenza, RSV, and community-acquired pneumonia. ICU and Med-Surg face "
            "increased intubation needs and longer length of stay; EDs see surge volumes "
            "that strain triage capacity and handoff documentation."
        ),
    },
    {
        "name": "Trauma and Heat Illness Spike",
        "months": [6, 7, 8],
        "affected_departments": ["Emergency Department", "ICU"],
        "risk_level": "high",
        "ce_topic_areas": [
            "Trauma assessment and damage control resuscitation",
            "Heat stroke recognition and rapid cooling protocols",
            "Rhabdomyolysis management",
            "Massive transfusion protocol activation",
            "Pediatric and geriatric trauma considerations",
        ],
        "description": (
            "June through August brings elevated trauma admissions from MVAs, outdoor "
            "activities, and violence, alongside heat-related emergencies during heat "
            "waves. EDs face peak throughput pressure compounding pre-existing handoff "
            "documentation weaknesses."
        ),
    },
    {
        "name": "Sepsis Complication Rise",
        "months": [12, 1, 2],
        "affected_departments": ["ICU", "Medical-Surgical", "Emergency Department"],
        "risk_level": "critical",
        "ce_topic_areas": [
            "Sepsis-3 criteria recognition and SOFA scoring",
            "Hour-1 bundle compliance (lactate, cultures, antibiotics)",
            "Fluid resuscitation and vasopressor initiation",
            "Central line insertion and maintenance (CLABSI prevention)",
            "Sepsis documentation and coding accuracy",
        ],
        "description": (
            "December through February correlates with peak sepsis incidence driven by "
            "respiratory infections progressing to sepsis. ICU sepsis response times "
            "historically worsen during holiday staffing reductions. Early identification "
            "and bundle compliance are the highest-leverage intervention points."
        ),
    },
    {
        "name": "Pediatric Admission Surge",
        "months": [8, 9],
        "affected_departments": ["Emergency Department", "Medical-Surgical"],
        "risk_level": "moderate",
        "ce_topic_areas": [
            "Pediatric assessment triangle and early warning signs",
            "Weight-based medication dosing and error prevention",
            "Bronchiolitis and croup management",
            "Back-to-school respiratory illness triage",
            "Family-centered care and communication",
        ],
        "description": (
            "August and September mark return-to-school season with surges in pediatric "
            "respiratory illness, asthma exacerbations, and injury. EDs and Med-Surg units "
            "that are not primarily pediatric may see an unfamiliar patient population "
            "requiring refreshed competencies."
        ),
    },
    {
        "name": "Allergy and Asthma Season",
        "months": [3, 4, 5],
        "affected_departments": ["Emergency Department", "Medical-Surgical"],
        "risk_level": "moderate",
        "ce_topic_areas": [
            "Acute asthma exacerbation management (GINA guidelines)",
            "Anaphylaxis recognition and epinephrine administration",
            "Peak flow monitoring and inhaler technique education",
            "Chronic obstructive pulmonary disease (COPD) exacerbation care",
            "Patient education for self-management and inhaler adherence",
        ],
        "description": (
            "March through May brings elevated pollen counts triggering asthma "
            "exacerbations, allergic reactions, and anaphylaxis presentations. EDs "
            "see increased acute respiratory visits; Med-Surg units admit patients "
            "with status asthmaticus requiring close monitoring and education."
        ),
    },
    {
        "name": "Norovirus Outbreak Season",
        "months": [1, 2, 3],
        "affected_departments": ["Medical-Surgical", "Oncology", "ICU"],
        "risk_level": "moderate",
        "ce_topic_areas": [
            "Contact precautions and enhanced environmental cleaning",
            "Outbreak containment and cohorting protocols",
            "Hand hygiene auditing and compliance improvement",
            "Dehydration and electrolyte management in vulnerable populations",
            "Staff illness reporting and return-to-work criteria",
        ],
        "description": (
            "January through March is peak norovirus season with high risk of healthcare-"
            "associated outbreaks. Immunocompromised patients in Oncology and critically "
            "ill patients in the ICU face severe dehydration risk. Rapid containment "
            "through cohorting and contact precautions is essential to prevent ward-wide spread."
        ),
    },
]


def get_seasonal_risks_for_month(month: int) -> list[dict]:
    """Return all seasonal risk entries that include the given month."""
    return [risk for risk in SEASONAL_RISKS if month in risk["months"]]


def get_upcoming_risks(current_month: int, lookahead_months: int = 2) -> list[dict]:
    """Return risks active in the next N months (excluding current month)."""
    upcoming = []
    for offset in range(1, lookahead_months + 1):
        check_month = ((current_month - 1 + offset) % 12) + 1
        for risk in SEASONAL_RISKS:
            if check_month in risk["months"] and risk not in upcoming:
                upcoming.append(risk)
    return upcoming
