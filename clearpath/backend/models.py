from pydantic import BaseModel, Field
from typing import Optional


class DepartmentMetrics(BaseModel):
    department_id: str
    department_name: str
    month: int
    year: int
    patient_fall_rate: float = Field(..., description="Falls per 1000 patient days")
    medication_error_rate: float = Field(..., description="Errors per 1000 medication administrations")
    readmission_rate_30d: float = Field(..., description="Percentage of patients readmitted within 30 days")
    handoff_documentation_score: float = Field(..., description="Completeness score 0-100")
    nurse_patient_ratio: float = Field(..., description="Patients per nurse")
    sepsis_response_time_minutes: float = Field(..., description="Average minutes from alert to intervention")


class SeasonalRisk(BaseModel):
    name: str
    months: list[int]
    affected_departments: list[str]
    risk_level: str = Field(..., description="low, moderate, high, critical")
    ce_topic_areas: list[str]
    description: str


class CERecommendation(BaseModel):
    topic: str
    hours: float = Field(..., description="Recommended CE credit hours")
    timing: str = Field(..., description="When to deliver: immediate, this_month, next_quarter")
    urgency_score: int = Field(..., ge=1, le=10, description="Urgency from 1 (low) to 10 (critical)")
    reasoning: str = Field(..., description="Data-backed rationale for this recommendation")


class AnalysisRequest(BaseModel):
    department_id: str
    current_month: int = Field(..., ge=1, le=12, description="Month number 1-12")


class AnalysisResponse(BaseModel):
    department_id: str
    department_name: str
    current_month: int
    recommendations: list[CERecommendation]
    risk_summary: str = Field(..., description="Plain language summary of the department risk profile")
    causal_chain: Optional[str] = Field(None, description="Identified causal chain e.g. staffing strain → handoff quality → readmissions")
    analysis_timestamp: str
