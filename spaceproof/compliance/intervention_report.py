"""intervention_report.py - Human intervention reports.

Generate reports on human interventions and override patterns.
Aggregates by reason code and severity for pattern analysis.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

COMPLIANCE_TENANT = "spaceproof-compliance"


@dataclass
class InterventionReport:
    """Human intervention report."""

    report_id: str
    generated_at: str
    time_range_start: str
    time_range_end: str
    total_interventions: int
    intervention_type_distribution: Dict[str, int]
    reason_code_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    intervener_distribution: Dict[str, int]
    retraining_required_count: int
    average_interventions_per_day: float
    top_reason_codes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "total_interventions": self.total_interventions,
            "intervention_type_distribution": self.intervention_type_distribution,
            "reason_code_distribution": self.reason_code_distribution,
            "severity_distribution": self.severity_distribution,
            "intervener_distribution": self.intervener_distribution,
            "retraining_required_count": self.retraining_required_count,
            "average_interventions_per_day": self.average_interventions_per_day,
            "top_reason_codes": self.top_reason_codes,
        }


# Storage for interventions (in production, would query ledger)
_interventions: List[Dict[str, Any]] = []


def record_intervention(
    intervention_id: str,
    intervention_type: str,
    reason_code: str,
    severity: str,
    intervener_id: str,
    requires_retraining: bool = False,
) -> None:
    """Record an intervention for reporting.

    Args:
        intervention_id: Intervention identifier
        intervention_type: Type (OVERRIDE, CORRECTION, etc.)
        reason_code: Reason code
        severity: Severity level
        intervener_id: Intervener identifier
        requires_retraining: Whether retraining is required
    """
    _interventions.append(
        {
            "intervention_id": intervention_id,
            "intervention_type": intervention_type,
            "reason_code": reason_code,
            "severity": severity,
            "intervener_id": intervener_id,
            "requires_retraining": requires_retraining,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


def get_intervention_metrics() -> Dict[str, Any]:
    """Get intervention metrics summary.

    Returns:
        Metrics summary
    """
    if not _interventions:
        return {
            "total_interventions": 0,
            "retraining_required_rate": 0.0,
            "most_common_reason": None,
        }

    retraining_count = sum(1 for i in _interventions if i["requires_retraining"])
    reason_counts = Counter(i["reason_code"] for i in _interventions)

    return {
        "total_interventions": len(_interventions),
        "retraining_required_rate": retraining_count / len(_interventions) * 100,
        "most_common_reason": reason_counts.most_common(1)[0] if reason_counts else None,
        "severity_breakdown": dict(Counter(i["severity"] for i in _interventions)),
    }


def generate_intervention_report(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> InterventionReport:
    """Generate intervention report.

    Args:
        start_time: Filter interventions after this time
        end_time: Filter interventions before this time

    Returns:
        InterventionReport
    """
    import uuid

    # Filter interventions by time
    interventions = _interventions
    if start_time:
        interventions = [i for i in interventions if i["timestamp"] >= start_time]
    if end_time:
        interventions = [i for i in interventions if i["timestamp"] <= end_time]

    # Compute distributions
    type_dist = Counter(i["intervention_type"] for i in interventions)
    reason_dist = Counter(i["reason_code"] for i in interventions)
    severity_dist = Counter(i["severity"] for i in interventions)
    intervener_dist = Counter(i["intervener_id"] for i in interventions)

    # Retraining required
    retraining_count = sum(1 for i in interventions if i["requires_retraining"])

    # Average per day (approximate)
    if interventions:
        timestamps = [i["timestamp"] for i in interventions]
        first = min(timestamps)
        last = max(timestamps)
        # Simple day count
        days = max(1, (datetime.fromisoformat(last.rstrip("Z")) - datetime.fromisoformat(first.rstrip("Z"))).days + 1)
        avg_per_day = len(interventions) / days
    else:
        avg_per_day = 0.0

    # Top reason codes
    top_reasons = [
        {"code": code, "count": count, "percentage": count / len(interventions) * 100 if interventions else 0}
        for code, count in reason_dist.most_common(5)
    ]

    report = InterventionReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.utcnow().isoformat() + "Z",
        time_range_start=start_time or "epoch",
        time_range_end=end_time or datetime.utcnow().isoformat() + "Z",
        total_interventions=len(interventions),
        intervention_type_distribution=dict(type_dist),
        reason_code_distribution=dict(reason_dist),
        severity_distribution=dict(severity_dist),
        intervener_distribution=dict(intervener_dist),
        retraining_required_count=retraining_count,
        average_interventions_per_day=avg_per_day,
        top_reason_codes=top_reasons,
    )

    # Emit receipt
    emit_receipt(
        "intervention_report",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "report_id": report.report_id,
            "total_interventions": report.total_interventions,
            "retraining_required_count": report.retraining_required_count,
        },
    )

    return report


def clear_interventions() -> None:
    """Clear interventions (for testing)."""
    global _interventions
    _interventions = []


def emit_intervention_report_receipt(report: InterventionReport) -> Dict[str, Any]:
    """Emit a receipt for intervention report generation.

    Args:
        report: The generated intervention report

    Returns:
        Receipt dictionary
    """
    return emit_receipt(
        "intervention_report",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "report_id": report.report_id,
            "total_interventions": report.total_interventions,
            "retraining_required_count": report.retraining_required_count,
        },
    )
