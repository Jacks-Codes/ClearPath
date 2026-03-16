from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data import DEPARTMENTS, EMR_DATA, get_all_months_for_department, get_department_metrics
from models import AnalysisRequest, AnalysisResponse, CERecommendation, DepartmentMetrics
from seasonal import get_seasonal_risks_for_month
from watson import analyze_department

app = FastAPI(
    title="ClearPath API",
    description="AI-powered nursing CE recommendation engine backed by IBM watsonx.ai",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok", "service": "ClearPath API", "timestamp": datetime.now(timezone.utc).isoformat()}


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


@app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
def analyze(request: AnalysisRequest):
    """
    Analyze a department's EMR data for the given month and return AI-generated
    CE recommendations powered by IBM watsonx.ai (Llama 3.3 70B).
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

    try:
        result = analyze_department(
            department_data=department_data,
            seasonal_data=seasonal_data,
            department_name=department_name,
            current_month=current_month,
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
