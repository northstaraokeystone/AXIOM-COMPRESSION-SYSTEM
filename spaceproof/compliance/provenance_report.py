"""provenance_report.py - Model/policy provenance reports.

Track model versions and policy changes over time.
Provides version history for reproducibility and compliance.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

COMPLIANCE_TENANT = "spaceproof-compliance"


@dataclass
class ProvenanceReport:
    """Model/policy provenance report."""

    report_id: str
    generated_at: str
    time_range_start: str
    time_range_end: str
    model_versions: List[Dict[str, Any]]
    policy_versions: List[Dict[str, Any]]
    model_changes: int
    policy_changes: int
    decisions_per_model: Dict[str, int]
    decisions_per_policy: Dict[str, int]
    rollbacks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "model_versions": self.model_versions,
            "policy_versions": self.policy_versions,
            "model_changes": self.model_changes,
            "policy_changes": self.policy_changes,
            "decisions_per_model": self.decisions_per_model,
            "decisions_per_policy": self.decisions_per_policy,
            "rollbacks": self.rollbacks,
        }


# Storage for provenance records
_model_versions: List[Dict[str, Any]] = []
_policy_versions: List[Dict[str, Any]] = []
_decisions: List[Dict[str, Any]] = []
_rollbacks: List[Dict[str, Any]] = []


def record_model_version(
    model_id: str,
    version: str,
    model_hash: str,
    deployed_at: Optional[str] = None,
) -> None:
    """Record a model version deployment.

    Args:
        model_id: Model identifier
        version: Version string
        model_hash: Model hash
        deployed_at: Deployment timestamp
    """
    _model_versions.append(
        {
            "model_id": model_id,
            "version": version,
            "model_hash": model_hash,
            "deployed_at": deployed_at or datetime.utcnow().isoformat() + "Z",
        }
    )


def record_policy_version(
    policy_id: str,
    version: str,
    policy_hash: str,
    activated_at: Optional[str] = None,
) -> None:
    """Record a policy version activation.

    Args:
        policy_id: Policy identifier
        version: Version string
        policy_hash: Policy hash
        activated_at: Activation timestamp
    """
    _policy_versions.append(
        {
            "policy_id": policy_id,
            "version": version,
            "policy_hash": policy_hash,
            "activated_at": activated_at or datetime.utcnow().isoformat() + "Z",
        }
    )


def record_decision_provenance(
    decision_id: str,
    model_id: str,
    model_version: str,
    policy_id: str,
    policy_version: str,
) -> None:
    """Record decision provenance.

    Args:
        decision_id: Decision identifier
        model_id: Model used
        model_version: Model version
        policy_id: Policy applied
        policy_version: Policy version
    """
    _decisions.append(
        {
            "decision_id": decision_id,
            "model_id": model_id,
            "model_version": model_version,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


def record_rollback(
    rollback_id: str,
    rollback_type: str,  # "model" or "policy"
    from_version: str,
    to_version: str,
    reason: str,
) -> None:
    """Record a model or policy rollback.

    Args:
        rollback_id: Rollback identifier
        rollback_type: Type of rollback
        from_version: Version rolled back from
        to_version: Version rolled back to
        reason: Rollback reason
    """
    _rollbacks.append(
        {
            "rollback_id": rollback_id,
            "rollback_type": rollback_type,
            "from_version": from_version,
            "to_version": to_version,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )

    # Emit rollback receipt
    emit_receipt(
        "rollback",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "rollback_id": rollback_id,
            "rollback_type": rollback_type,
            "from_version": from_version,
            "to_version": to_version,
            "reason": reason,
        },
    )


def get_model_history(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get model version history.

    Args:
        model_id: Filter by model ID

    Returns:
        List of model versions
    """
    if model_id:
        return [m for m in _model_versions if m["model_id"] == model_id]
    return _model_versions.copy()


def get_policy_history(policy_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get policy version history.

    Args:
        policy_id: Filter by policy ID

    Returns:
        List of policy versions
    """
    if policy_id:
        return [p for p in _policy_versions if p["policy_id"] == policy_id]
    return _policy_versions.copy()


def generate_provenance_report(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> ProvenanceReport:
    """Generate provenance report.

    Args:
        start_time: Filter after this time
        end_time: Filter before this time

    Returns:
        ProvenanceReport
    """
    import uuid

    # Filter by time
    models = _model_versions
    policies = _policy_versions
    decisions = _decisions
    rollbacks = _rollbacks

    if start_time:
        models = [m for m in models if m["deployed_at"] >= start_time]
        policies = [p for p in policies if p["activated_at"] >= start_time]
        decisions = [d for d in decisions if d["timestamp"] >= start_time]
        rollbacks = [r for r in rollbacks if r["timestamp"] >= start_time]

    if end_time:
        models = [m for m in models if m["deployed_at"] <= end_time]
        policies = [p for p in policies if p["activated_at"] <= end_time]
        decisions = [d for d in decisions if d["timestamp"] <= end_time]
        rollbacks = [r for r in rollbacks if r["timestamp"] <= end_time]

    # Decisions per model version
    decisions_per_model: Dict[str, int] = defaultdict(int)
    for d in decisions:
        key = f"{d['model_id']}:{d['model_version']}"
        decisions_per_model[key] += 1

    # Decisions per policy version
    decisions_per_policy: Dict[str, int] = defaultdict(int)
    for d in decisions:
        key = f"{d['policy_id']}:{d['policy_version']}"
        decisions_per_policy[key] += 1

    report = ProvenanceReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.utcnow().isoformat() + "Z",
        time_range_start=start_time or "epoch",
        time_range_end=end_time or datetime.utcnow().isoformat() + "Z",
        model_versions=models,
        policy_versions=policies,
        model_changes=len(models),
        policy_changes=len(policies),
        decisions_per_model=dict(decisions_per_model),
        decisions_per_policy=dict(decisions_per_policy),
        rollbacks=rollbacks,
    )

    # Emit receipt
    emit_receipt(
        "provenance_report",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "report_id": report.report_id,
            "model_changes": report.model_changes,
            "policy_changes": report.policy_changes,
            "rollbacks": len(rollbacks),
        },
    )

    return report


def clear_provenance_records() -> None:
    """Clear all provenance records (for testing)."""
    global _model_versions, _policy_versions, _decisions, _rollbacks
    _model_versions = []
    _policy_versions = []
    _decisions = []
    _rollbacks = []


def get_provenance_history(
    entity_id: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get combined provenance history for models and policies.

    Args:
        entity_id: Filter by entity ID
        entity_type: Filter by entity type ("model" or "policy")

    Returns:
        Combined list of provenance records
    """
    history = []

    # Include model versions
    if entity_type is None or entity_type == "model":
        for m in _model_versions:
            if entity_id is None or m["model_id"] == entity_id:
                history.append({**m, "entity_type": "model"})

    # Include policy versions
    if entity_type is None or entity_type == "policy":
        for p in _policy_versions:
            if entity_id is None or p["policy_id"] == entity_id:
                history.append({**p, "entity_type": "policy"})

    # Sort by timestamp
    history.sort(
        key=lambda x: x.get("deployed_at") or x.get("activated_at") or "",
        reverse=True,
    )

    return history


def emit_provenance_report_receipt(report: ProvenanceReport) -> Dict[str, Any]:
    """Emit a receipt for provenance report generation.

    Args:
        report: The generated provenance report

    Returns:
        Receipt dictionary
    """
    return emit_receipt(
        "provenance_report",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "report_id": report.report_id,
            "model_changes": report.model_changes,
            "policy_changes": report.policy_changes,
            "rollbacks": len(report.rollbacks),
        },
    )
