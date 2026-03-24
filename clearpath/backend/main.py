import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import jwt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data import DEPARTMENTS, EMR_DATA, BENCHMARKS, SOAP_NOTES, BEST_PRACTICE_REFERENCES, get_all_months_for_department, get_department_metrics
from models import AnalysisRequest, AnalysisResponse, CERecommendation, DepartmentMetrics, SOAPNote, Benchmark
from seasonal import get_seasonal_risks_for_month
from claude_client import analyze_department

app = FastAPI(
    title="ClearPath API",
    description="AI-powered nursing CE recommendation engine backed by Anthropic Claude",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load RSA private key for JWT signing (Orchestrate chat auth)
_KEY_PATH = Path(__file__).parent / "chat_jwt_private.pem"
_JWT_PRIVATE_KEY = _KEY_PATH.read_text() if _KEY_PATH.exists() else None


@app.get("/chat-token", tags=["chat"])
def chat_token():
    """Generate a signed JWT for chat authentication."""
    if not _JWT_PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="Chat JWT private key not configured.")
    payload = {
        "sub": f"clearpath-user-{uuid.uuid4().hex[:8]}",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "context": {
            "app": "ClearPath",
            "role": "nursing_director",
        },
    }
    token = jwt.encode(payload, _JWT_PRIVATE_KEY, algorithm="RS256")
    return {"token": token}


@app.get("/", tags=["health"])
def health_check():
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    return {"status": "ok", "service": "ClearPath API", "has_anthropic_key": has_key, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/departments", tags=["departments"])
def list_departments():
    """Return all department IDs and display names."""
    return [
        {"department_id": dept_id, "department_name": name}
        for dept_id, name in DEPARTMENTS.items()
    ]


@app.get("/departments/{department_id}/metrics", tags=["departments"])
def get_metrics(department_id: str):
    """Return all 12 months of raw EMR metrics for a department."""
    if department_id not in DEPARTMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Department '{department_id}' not found. Valid IDs: {list(DEPARTMENTS.keys())}",
        )
    records = get_all_months_for_department(department_id)
    return {
        "department_id": department_id,
        "department_name": DEPARTMENTS[department_id],
        "metrics": [DepartmentMetrics(**r) for r in records],
    }


@app.get("/seasonal/{month}", tags=["seasonal"])
def seasonal_risks(month: int):
    """Return active seasonal risk patterns for a given month (1–12)."""
    if not 1 <= month <= 12:
        raise HTTPException(status_code=422, detail="Month must be between 1 and 12.")
    risks = get_seasonal_risks_for_month(month)
    return {"month": month, "seasonal_risks": risks}


@app.get("/departments/{department_id}/soap-notes", tags=["departments"])
def get_soap_notes(department_id: str):
    """Return sample SOAP notes / clinical documentation for a department."""
    if department_id not in DEPARTMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Department '{department_id}' not found. Valid IDs: {list(DEPARTMENTS.keys())}",
        )
    notes = SOAP_NOTES.get(department_id, [])
    return {
        "department_id": department_id,
        "department_name": DEPARTMENTS[department_id],
        "soap_notes": [SOAPNote(**n) for n in notes],
    }


@app.get("/departments/{department_id}/benchmarks", tags=["departments"])
def get_benchmarks(department_id: str):
    """Return objective benchmark targets for a department's metrics."""
    if department_id not in DEPARTMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Department '{department_id}' not found. Valid IDs: {list(DEPARTMENTS.keys())}",
        )
    benchmarks = BENCHMARKS.get(department_id, {})
    return {
        "department_id": department_id,
        "department_name": DEPARTMENTS[department_id],
        "benchmarks": {k: Benchmark(**v) for k, v in benchmarks.items()},
    }


@app.get("/references", tags=["references"])
def get_references():
    """Return best practice reference standards used in analysis."""
    return {"references": BEST_PRACTICE_REFERENCES}


@app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
def analyze(request: AnalysisRequest):
    """
    Analyze a department's EMR data for the given month and return AI-generated
    CE recommendations powered by Anthropic Claude.
    """
    department_id = request.department_id
    current_month = request.current_month

    if department_id not in DEPARTMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Department '{department_id}' not found. Valid IDs: {list(DEPARTMENTS.keys())}",
        )

    department_data = get_all_months_for_department(department_id)
    seasonal_data = get_seasonal_risks_for_month(current_month)
    department_name = DEPARTMENTS[department_id]
    soap_notes = SOAP_NOTES.get(department_id, [])
    benchmarks = BENCHMARKS.get(department_id, {})

    try:
        result = analyze_department(
            department_data=department_data,
            seasonal_data=seasonal_data,
            department_name=department_name,
            current_month=current_month,
            soap_notes=soap_notes,
            benchmarks=benchmarks,
            references=BEST_PRACTICE_REFERENCES,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    recommendations = [CERecommendation(**rec) for rec in result.get("recommendations", [])]

    return AnalysisResponse(
        department_id=department_id,
        department_name=department_name,
        current_month=current_month,
        recommendations=recommendations,
        risk_summary=result["risk_summary"],
        causal_chain=result.get("causal_chain"),
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
    )
