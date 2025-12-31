"""raci_report.py - RACI accountability reports.

Generate reports on RACI assignments and accountability.
Tracks coverage and identifies decisions without proper assignment.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

COMPLIANCE_TENANT = "spaceproof-compliance"


@dataclass
class RACIReport:
    """RACI accountability report."""

    report_id: str
    generated_at: str
    time_range_start: str
    time_range_end: str
    total_decisions: int
    decisions_with_raci: int
    raci_coverage: float
    role_distribution: Dict[str, Dict[str, int]]
    event_type_distribution: Dict[str, int]
    escalation_summary: Dict[str, int]
    accountability_gaps: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "total_decisions": self.total_decisions,
            "decisions_with_raci": self.decisions_with_raci,
            "raci_coverage": self.raci_coverage,
            "role_distribution": self.role_distribution,
            "event_type_distribution": self.event_type_distribution,
            "escalation_summary": self.escalation_summary,
            "accountability_gaps": self.accountability_gaps,
        }


# Storage for RACI decisions (in production, would query ledger)
_raci_decisions: List[Dict[str, Any]] = []


def record_raci_decision(
    decision_id: str,
    event_type: str,
    responsible: str,
    accountable: str,
    consulted: List[str],
    informed: List[str],
    escalated: bool = False,
    escalation_level: Optional[str] = None,
) -> None:
    """Record a RACI decision for reporting.

    Args:
        decision_id: Decision identifier
        event_type: Type of event
        responsible: Responsible party
        accountable: Accountable party
        consulted: Consulted parties
        informed: Informed parties
        escalated: Whether decision was escalated
        escalation_level: Level of escalation
    """
    _raci_decisions.append(
        {
            "decision_id": decision_id,
            "event_type": event_type,
            "responsible": responsible,
            "accountable": accountable,
            "consulted": consulted,
            "informed": informed,
            "escalated": escalated,
            "escalation_level": escalation_level,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


def get_accountability_summary() -> Dict[str, Any]:
    """Get summary of accountability assignments.

    Returns:
        Summary statistics
    """
    if not _raci_decisions:
        return {
            "total_decisions": 0,
            "unique_accountable_parties": 0,
            "escalation_rate": 0.0,
        }

    accountable_parties = set(d["accountable"] for d in _raci_decisions)
    escalated_count = sum(1 for d in _raci_decisions if d["escalated"])

    return {
        "total_decisions": len(_raci_decisions),
        "unique_accountable_parties": len(accountable_parties),
        "escalation_rate": escalated_count / len(_raci_decisions) * 100,
        "most_common_accountable": Counter(d["accountable"] for d in _raci_decisions).most_common(5),
    }


def find_accountability_gaps(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find gaps in accountability assignments.

    Args:
        decisions: List of RACI decisions

    Returns:
        List of identified gaps
    """
    gaps = []

    for d in decisions:
        # Check for missing accountable
        if not d.get("accountable"):
            gaps.append(
                {
                    "decision_id": d["decision_id"],
                    "gap_type": "missing_accountable",
                    "description": "No accountable party assigned",
                }
            )

        # Check for missing responsible
        if not d.get("responsible"):
            gaps.append(
                {
                    "decision_id": d["decision_id"],
                    "gap_type": "missing_responsible",
                    "description": "No responsible party assigned",
                }
            )

        # Check for same R and A (potential conflict of interest)
        if d.get("responsible") == d.get("accountable") and d.get("responsible"):
            gaps.append(
                {
                    "decision_id": d["decision_id"],
                    "gap_type": "r_a_same",
                    "description": "Responsible and Accountable are the same",
                    "severity": "low",
                }
            )

    return gaps


def generate_raci_report(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> RACIReport:
    """Generate RACI accountability report.

    Args:
        start_time: Filter decisions after this time
        end_time: Filter decisions before this time

    Returns:
        RACIReport
    """
    import uuid

    # Filter decisions by time
    decisions = _raci_decisions
    if start_time:
        decisions = [d for d in decisions if d["timestamp"] >= start_time]
    if end_time:
        decisions = [d for d in decisions if d["timestamp"] <= end_time]

    # Compute role distribution
    role_distribution = {
        "responsible": Counter(d["responsible"] for d in decisions if d.get("responsible")),
        "accountable": Counter(d["accountable"] for d in decisions if d.get("accountable")),
    }

    # Event type distribution
    event_type_distribution = Counter(d["event_type"] for d in decisions)

    # Escalation summary
    escalation_summary = Counter(d.get("escalation_level", "none") for d in decisions if d.get("escalated"))

    # Find accountability gaps
    gaps = find_accountability_gaps(decisions)

    # Compute coverage
    decisions_with_raci = sum(1 for d in decisions if d.get("responsible") and d.get("accountable"))
    coverage = decisions_with_raci / len(decisions) * 100 if decisions else 0

    report = RACIReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.utcnow().isoformat() + "Z",
        time_range_start=start_time or "epoch",
        time_range_end=end_time or datetime.utcnow().isoformat() + "Z",
        total_decisions=len(decisions),
        decisions_with_raci=decisions_with_raci,
        raci_coverage=coverage,
        role_distribution={k: dict(v) for k, v in role_distribution.items()},
        event_type_distribution=dict(event_type_distribution),
        escalation_summary=dict(escalation_summary),
        accountability_gaps=gaps,
    )

    # Emit receipt
    emit_receipt(
        "raci_report",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "report_id": report.report_id,
            "total_decisions": report.total_decisions,
            "raci_coverage": report.raci_coverage,
            "gaps_found": len(gaps),
        },
    )

    return report


def clear_raci_decisions() -> None:
    """Clear RACI decisions (for testing)."""
    global _raci_decisions
    _raci_decisions = []
