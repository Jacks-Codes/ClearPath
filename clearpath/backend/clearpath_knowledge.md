# ClearPath — Nursing CE Intelligence Platform

## What is ClearPath?

ClearPath is an AI-powered platform that helps hospital nursing education directors make data-driven continuing education (CE) mandate decisions. Today, most education directors plan annual CE requirements based on gut instinct, outdated compliance checklists, and whatever topics vendors happen to be promoting. They have no real-time visibility into which units are struggling, what the actual performance gaps are, or how seasonal risk patterns should influence training priorities.

ClearPath solves this by analyzing Electronic Medical Record (EMR) data across hospital departments, identifying performance gaps against national benchmarks, overlaying seasonal risk intelligence, and generating specific CE recommendations powered by IBM watsonx.ai (Llama 3.3 70B Instruct). The result is a prioritized, evidence-based CE plan that ties directly to patient safety outcomes.

## Departments

ClearPath monitors four hospital departments. Each department has a unique risk profile based on patient acuity, staffing patterns, and operational characteristics.

### ICU (department ID: `icu`)

The Intensive Care Unit handles the highest-acuity patients. Current risk profile shows elevated concern across multiple signals:

- **Sepsis response time**: 28 minutes (target: 20 minutes per CMS SEP-1). This is 40% above the national benchmark and represents the most urgent intervention point.
- **Handoff documentation score**: 80/100 (target: 90 per Joint Commission NPSG). Declining trend over the past 6 months, down from 91 in October.
- **30-day readmission rate**: 18.5% (target: 12% per CMS HRRP). Steadily climbing from 13.2% six months ago.
- **Medication error rate**: 4.0 per 1,000 administrations (target: 3.0 per NQF Safe Practice #18).
- **Nurse-to-patient ratio**: 1:2.1 (target: 1:2 per ANA Staffing Standards). Slightly over threshold.

The ICU shows a clear deterioration trend from October through March, consistent with winter surge staffing strain. The identified causal chain is: staffing strain leads to handoff quality deterioration, which leads to increased readmissions and medication errors.

### Medical-Surgical (department ID: `med_surg`)

Medical-Surgical is the hospital's largest unit by bed count, handling a broad mix of post-operative and general medical patients. Current risk profile:

- **Handoff documentation score**: 78/100 (target: 85 per Joint Commission NPSG). Consistently below benchmark.
- **Nurse-to-patient ratio**: 1:5.1 (target: 1:5 per ANA Staffing Standards). Operating at the edge of safe staffing.
- **30-day readmission rate**: 13.9% (target: 10% per CMS HRRP). Trending upward, indicating discharge planning gaps.
- **Medication error rate**: 2.7 per 1,000 (target: 2.5 per NQF Safe Practice #18). Slightly above benchmark.

Staffing strain is the primary driver of quality gaps in Med-Surg. When the ratio exceeds 1:5, handoff quality and fall rates both worsen measurably.

### Emergency Department (department ID: `ed`)

The Emergency Department operates under constant volume pressure with the lowest handoff documentation scores in the hospital. Current risk profile:

- **Handoff documentation score**: 63/100 (target: 80 per Joint Commission NPSG). This is the lowest score of any department and represents a 21% gap from target. The score has dropped from 76 six months ago.
- **Sepsis response time**: 21 minutes (target: 15 minutes per CMS SEP-1). 40% above the tighter ED-specific benchmark.
- **Medication error rate**: 3.3 per 1,000 (target: 2.0 per Leapfrog Safe Practices). The ED has a stricter benchmark due to higher error consequence in emergency settings.
- **Nurse-to-patient ratio**: 1:4.4 (target: 1:4 per ANA Staffing Standards). Volume-driven strain.

The ED's primary risk driver is volume-induced documentation shortcuts. When patient volume spikes, handoff quality is the first thing to degrade, creating downstream risks for medication errors and missed sepsis identification.

### Oncology (department ID: `oncology`)

Oncology generally maintains the cleanest metrics of all departments due to specialized staff and lower patient turnover. Current risk profile:

- **30-day readmission rate**: 10.8% (target: 9% per CMS HRRP). A recent uptick that warrants monitoring, potentially linked to discharge education gaps for immunocompromised patients.
- **Handoff documentation score**: 92/100 (target: 92 per Joint Commission NPSG). Meeting benchmark.
- **Medication error rate**: 1.7 per 1,000 (target: 1.5 per NQF Safe Practice #18). Slightly above the tight oncology-specific benchmark.

Oncology's risk is concentrated in readmission patterns, particularly during summer months (June-August) when readmission rates spike to 15-16% due to immunocompromised patients and seasonal infection exposure.

## Risk Scoring

ClearPath assigns each department a risk score from 1 to 10 based on how far its current metrics deviate from national benchmarks. The score uses a blended formula that weighs both the average gap across all metrics and the single worst metric gap, ensuring that one severely failing metric raises the overall score appropriately.

- **1-3 (Low risk)**: Metrics are at or near benchmark targets. Routine CE maintenance is appropriate.
- **4-6 (Moderate risk)**: Some metrics exceed benchmarks. Targeted CE should address specific gaps within the next quarter.
- **7-8 (High risk)**: Multiple metrics significantly exceed benchmarks. CE interventions should be prioritized this month.
- **9-10 (Critical risk)**: Severe and worsening performance gaps. Immediate CE intervention and operational review required.

Current risk scores (March 2026): ICU is at high risk, ED is at high risk, Med-Surg is at moderate risk, and Oncology is at low risk.

## CE Recommendations

When a department analysis is run, ClearPath generates specific CE recommendations. Each recommendation includes:

- **Topic**: The specific training subject (e.g., "Sepsis Recognition and Management using SEP-1 Guidelines")
- **Urgency score**: 1-10 rating of how urgently the training is needed
- **Timing**: When the training should be delivered — "immediate" for critical gaps, "this_month" for high-priority items, or "next_quarter" for moderate concerns
- **Hours**: Recommended CE contact hours for the training
- **Reasoning**: Data-backed explanation citing specific metrics, benchmarks, and trends that justify the recommendation

Recommendations reference national standards from AHRQ, NQF, CMS, ANA, Joint Commission, and Leapfrog Group. They also incorporate seasonal risk factors and clinical documentation (SOAP notes) when available.

## Causal Chain Analysis

ClearPath identifies causal chains — sequences of connected problems where one issue drives another. For example:

**Staffing strain → Handoff quality deterioration → Increased readmissions and medication errors**

This chain, commonly seen in ICU and Med-Surg, shows how understaffing creates rushed shift changes, which leads to incomplete SBAR documentation, which results in oncoming nurses missing critical patient information, ultimately causing preventable readmissions and medication errors. Understanding the causal chain helps education directors target the root cause rather than just the symptoms.

## How to Query ClearPath

Valid department IDs for API queries are: `icu`, `med_surg`, `ed`, `oncology`. Analysis can be run for any month (1-12) to see how risk profiles change across seasons. The platform tracks 12 months of historical EMR data and overlays seasonal risk intelligence for the selected month.
