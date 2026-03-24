"""
Anthropic Claude API client for ClearPath AI analysis.
Uses claude-sonnet-4-20250514 via the anthropic Python SDK.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "claude-sonnet-4-20250514"

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def clean_json_response(raw: str) -> str:
    """
    Extract the first complete JSON object from a model response.

    The model sometimes repeats its output several times, each prefixed with a
    ```json fence, so regex-based fence stripping leaves multiple concatenated
    objects that fail to parse. Instead we locate the first '{' and walk forward
    counting braces (respecting quoted strings and escape sequences) until the
    matching '}' is found, then return only that slice.
    """
    start = raw.find('{')
    if start == -1:
        return raw.strip()

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]

    # No balanced closing brace found — return from first '{' and let json.loads report the error
    return raw[start:].strip()


def _build_prompt(
    department_name: str,
    department_data: list[dict],
    seasonal_data: list[dict],
    current_month: int,
    soap_notes: list[dict] | None = None,
    benchmarks: dict | None = None,
    references: list[dict] | None = None,
) -> tuple[str, str]:
    """Build system prompt and user message for Claude. Returns (system, user)."""
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

    # Build SOAP notes context
    soap_text = ""
    if soap_notes:
        soap_entries = []
        for note in soap_notes:
            soap_entries.append(
                f"  [{note['note_type']}] {note['date']} — {note['provider']}\n"
                f"    S: {note['subjective'][:200]}\n"
                f"    O: {note['objective'][:200]}\n"
                f"    A: {note['assessment'][:200]}\n"
                f"    P: {note['plan'][:200]}"
            )
        soap_text = "\n\n".join(soap_entries)

    # Build benchmarks context
    benchmark_text = ""
    if benchmarks:
        benchmark_lines = []
        for metric_key, bench in benchmarks.items():
            benchmark_lines.append(f"  - {bench['label']}: target {bench['target']} (source: {bench['source']})")
        benchmark_text = "\n".join(benchmark_lines)

    # Build references context
    reference_text = ""
    if references:
        reference_text = ", ".join(f"{r['name']} ({r['source']})" for r in references[:8])

    system = (
        "You are a clinical education analytics AI for a hospital nursing education director. "
        "Your task is to analyze EMR quality metrics and generate prioritized continuing education (CE) "
        "recommendations for nursing staff. Respond ONLY with valid JSON — no explanation or text outside the JSON."
    )

    user = f"""Analyze EMR quality metrics for the {department_name} department and generate
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

## Objective Benchmark Targets
{benchmark_text if benchmark_text else "  No benchmarks available."}

## Clinical Documentation Samples (SOAP Notes)
{soap_text if soap_text else "  No clinical documentation available."}

## Applicable Best Practice Standards
{reference_text if reference_text else "  No references available."}

## Analysis Instructions
Identify quality/safety signals in the clinical narratives above. Reference applicable standards
from AHRQ, NQF, CMS, ANA, Leapfrog, or Joint Commission in your reasoning where relevant.
Compare current metrics against the benchmark targets above and flag any that exceed thresholds.

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
name actual clinical protocols, tools, or frameworks in the topic field."""

    return system, user


def analyze_department(
    department_data: list[dict],
    seasonal_data: list[dict],
    department_name: str,
    current_month: int,
    soap_notes: list[dict] | None = None,
    benchmarks: dict | None = None,
    references: list[dict] | None = None,
) -> dict:
    """
    Call Anthropic Claude API to analyze department metrics and return structured CE recommendations.
    Returns a dict with keys: recommendations, risk_summary, causal_chain.
    Raises RuntimeError with a descriptive message on failure.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Anthropic API key. Set ANTHROPIC_API_KEY in your environment variables."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package is not installed. Run: pip install anthropic"
        ) from exc

    try:
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt, user_message = _build_prompt(
            department_name, department_data, seasonal_data, current_month,
            soap_notes, benchmarks, references,
        )

        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        response = message.content[0].text

    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

    # Parse the model's JSON response
    cleaned = clean_json_response(response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON parse failed after fence stripping.\n"
            "--- RAW RESPONSE ---\n%s\n--- CLEANED ---\n%s",
            response,
            cleaned,
        )
        raise RuntimeError(
            f"Model returned non-JSON response. Raw output:\n{response[:800]}"
        ) from exc

    # Validate required keys
    required_keys = {"recommendations", "risk_summary", "causal_chain"}
    missing = required_keys - set(parsed.keys())
    if missing:
        raise RuntimeError(f"Model response missing required fields: {missing}")

    return parsed
