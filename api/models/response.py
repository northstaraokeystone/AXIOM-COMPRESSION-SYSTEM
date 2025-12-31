"""response.py - Pydantic response models for API endpoints."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class EntropyAnalysisResponse(BaseModel):
    """Entropy analysis details in verification response."""

    measured_entropy: float = Field(..., description="Computed entropy value")
    baseline_min: float = Field(..., description="Baseline minimum entropy")
    baseline_max: float = Field(..., description="Baseline maximum entropy")
    deviation_pct: float = Field(..., description="Deviation from baseline (%)")
    flags: List[str] = Field(default=[], description="Detection flags")
    comparison: str = Field(..., description="Human-readable comparison explanation")


class VerificationResponse(BaseModel):
    """Standard verification response."""

    verdict: Literal["AUTHENTIC", "COUNTERFEIT", "SUSPICIOUS"] = Field(
        ..., description="Verification verdict"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Risk score (1 - confidence)"
    )
    entropy_analysis: EntropyAnalysisResponse = Field(
        ..., description="Entropy analysis details"
    )
    receipt: Dict[str, Any] = Field(..., description="Complete verification receipt")
    receipt_id: str = Field(..., description="Receipt identifier (payload_hash)")
    domain: str = Field(..., description="Verification domain")
    item_id: str = Field(..., description="Verified item identifier")
    risk_level: Optional[str] = Field(
        default=None,
        description="Risk level for medical domain (CRITICAL, HIGH, MEDIUM, LOW)",
    )
    compliance_report_url: Optional[str] = Field(
        default=None, description="URL to pre-generated compliance report"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    receipt: Optional[Dict[str, Any]] = Field(
        default=None, description="Error receipt for audit trail"
    )
    timestamp: str = Field(..., description="Error timestamp (ISO8601)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Service health status"
    )
    domains: List[str] = Field(
        default=["aerospace", "food", "medical"],
        description="Available verification domains",
    )
    version: str = Field(..., description="API version")
    ledger_writable: bool = Field(..., description="Receipt ledger is writable")
    config_readable: bool = Field(..., description="Configuration files are readable")
