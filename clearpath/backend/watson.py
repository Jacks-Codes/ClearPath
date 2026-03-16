"""
IBM watsonx.ai Runtime client for ClearPath AI analysis.
Uses meta-llama/llama-3-3-70b-instruct via the ibm-watsonx-ai SDK.
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "meta-llama/llama-3-3-70b-instruct"

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def _build_prompt(
    department_name: str,
    department_data: list[dict],
    seasonal_data: list[dict],
    current_month: int,
) -> str:
    month_name = MONTH_NAMES[current_month]

    # Summarise last 3 months of metrics as trend context
    recent = sorted(department_data, key=lambda x: x["month"])
    recent_3 = recent[-3:] if len(recent) >= 3 else recent
    current_record = next((r for r in department_data if r["month"] == current_month), recent[-1])

    metrics_text = "\n".join(
        f"  Month {r['month']:>2} ({MONTH_NAMES[r['month']][:3]}): "
        f"falls={r['patient_fall_rate']}/1000pd, "
        f"med_errors={r['medication_error_rate']}/1000admin, "
        f"readmit_30d={r['readmission_rate_30d']}%, "
        f"handoff_score={r['handoff_documentation_score']}/100, "
        f"nurse:pt_ratio=1:{r['nurse_patient_ratio']}, "
        f"sepsis_response={r['sepsis_response_time_minutes']}min"
        for r in recent_3
    )

    current_metrics_text = (
        f"  falls={current_record['patient_fall_rate']}/1000pd, "
        f"med_errors={current_record['medication_error_rate']}/1000admin, "
        f"readmit_30d={current_record['readmission_rate_30d']}%, "
        f"handoff_score={current_record['handoff_documentation_score']}/100, "
        f"nurse:pt_ratio=1:{current_record['nurse_patient_ratio']}, "
        f"sepsis_response={current_record['sepsis_response_time_minutes']}min"
    )

    seasonal_text = ""
    if seasonal_data:
        seasonal_text = "\n".join(
            f"  - {s['name']} (risk: {s['risk_level']}): {s['description']}\n"
            f"    Recommended CE areas: {', '.join(s['ce_topic_areas'][:3])}"
            for s in seasonal_data
        )
    else:
        seasonal_text = "  No major seasonal risk patterns active this month."

    return f"""You are a clinical education analytics AI for a hospital nursing education director.
Your task is to analyze EMR quality metrics for the {department_name} department and generate
prioritized continuing education (CE) recommendations for nursing staff.

## Current Month
{month_name} (month {current_month})

## Department: {department_name}
### Current Month Metrics ({month_name})
{current_metrics_text}

### Recent Trend (last 3 months)
{metrics_text}

## Active Seasonal Risk Patterns for {month_name}
{seasonal_text}

## Analysis Instructions
Reason across EXACTLY these three signal types and identify anomalies or risks in each:

1. STAFFING ANOMALIES — Evaluate nurse-to-patient ratio. Ratios above 1:5 for Med-Surg,
   1:2.5 for ICU, or 1:5 for ED indicate strain. Rising ratios across months signal
   escalating risk.

2. HANDOFF QUALITY DETERIORATION — Evaluate handoff documentation completeness score.
   Scores below 75 are concerning; below 65 are critical. Declining trends amplify
   medication error and readmission risk.

3. READMISSION PATTERNS — Evaluate 30-day readmission rate. Rates above 15% are a
   quality flag. Cross-reference with handoff scores and ratios to identify causal chains.

Also factor in the active seasonal risks above — if seasonal patterns amplify existing
department weaknesses, elevate urgency scores accordingly.

## Required Output Format
Respond ONLY with valid JSON. Do not include any explanation or text outside the JSON.
The JSON must match this exact schema:

{{
  "recommendations": [
    {{
      "topic": "string — specific CE topic title",
      "hours": number — recommended CE credit hours (0.5 to 8.0),
      "timing": "immediate" | "this_month" | "next_quarter",
      "urgency_score": integer 1-10,
      "reasoning": "string — 2-3 sentences citing specific metric values and trends"
    }}
  ],
  "risk_summary": "string — 3-5 sentence plain language summary of department risk profile for a nursing director",
  "causal_chain": "string or null — if a causal chain exists, describe it (e.g. 'staffing strain → handoff quality → readmissions'), otherwise null"
}}

Generate 3-5 CE recommendations ordered by urgency_score descending. Be specific —
name actual clinical protocols, tools, or frameworks in the topic field.
"""


def analyze_department(
    department_data: list[dict],
    seasonal_data: list[dict],
    department_name: str,
    current_month: int,
) -> dict:
    """
    Call watsonx.ai to analyze department metrics and return structured CE recommendations.
    Returns a dict with keys: recommendations, risk_summary, causal_chain.
    Raises RuntimeError with a descriptive message on failure.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        raise RuntimeError(
            "Missing IBM watsonx credentials. Set WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file."
        )

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    except ImportError as exc:
        raise RuntimeError(
            "ibm-watsonx-ai package is not installed. Run: pip install ibm-watsonx-ai"
        ) from exc

    try:
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY,
        )
        client = APIClient(credentials)

        model = ModelInference(
            model_id=MODEL_ID,
            api_client=client,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 2048,
                GenParams.TEMPERATURE: 0.2,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.05,
            },
        )

        prompt = _build_prompt(department_name, department_data, seasonal_data, current_month)
        response = model.generate_text(prompt=prompt)

    except Exception as exc:
        raise RuntimeError(f"watsonx.ai API call failed: {exc}") from exc

    # Parse the model's JSON response
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Model returned non-JSON response. Raw output:\n{response[:500]}"
        ) from exc

    # Validate required keys
    required_keys = {"recommendations", "risk_summary", "causal_chain"}
    missing = required_keys - set(parsed.keys())
    if missing:
        raise RuntimeError(f"Model response missing required fields: {missing}")

    return parsed
