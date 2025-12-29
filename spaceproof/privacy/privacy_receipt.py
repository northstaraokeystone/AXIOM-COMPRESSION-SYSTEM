"""privacy_receipt.py - Privacy operation proofs.

Track and audit all privacy-related operations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

PRIVACY_TENANT = "spaceproof-privacy"


@dataclass
class PrivacyOperation:
    """Record of a privacy operation."""

    operation_id: str
    operation_type: str  # redaction, dp_query, budget_check, etc.
    actor_id: str
    target_type: str
    target_id: str
    details: Dict[str, Any]
    privacy_cost: float  # epsilon spent
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "actor_id": self.actor_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "details": self.details,
            "privacy_cost": self.privacy_cost,
            "success": self.success,
            "timestamp": self.timestamp,
        }


# Privacy audit log
_privacy_audit_log: List[PrivacyOperation] = []


def track_privacy_operation(
    operation_type: str,
    actor_id: str,
    target_type: str,
    target_id: str,
    details: Optional[Dict[str, Any]] = None,
    privacy_cost: float = 0.0,
    success: bool = True,
) -> PrivacyOperation:
    """Track a privacy operation.

    Args:
        operation_type: Type of operation
        actor_id: Who performed the operation
        target_type: Type of target (receipt, query, etc.)
        target_id: Target identifier
        details: Operation details
        privacy_cost: Epsilon spent
        success: Whether operation succeeded

    Returns:
        PrivacyOperation record
    """
    operation = PrivacyOperation(
        operation_id=str(uuid.uuid4()),
        operation_type=operation_type,
        actor_id=actor_id,
        target_type=target_type,
        target_id=target_id,
        details=details or {},
        privacy_cost=privacy_cost,
        success=success,
    )

    _privacy_audit_log.append(operation)

    return operation


def emit_privacy_receipt(
    operation: PrivacyOperation,
    additional_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit privacy operation receipt.

    Args:
        operation: PrivacyOperation to emit
        additional_context: Optional additional context

    Returns:
        Receipt dict
    """
    receipt_data = {
        "tenant_id": PRIVACY_TENANT,
        **operation.to_dict(),
    }

    if additional_context:
        receipt_data["context"] = additional_context

    return emit_receipt("privacy_operation", receipt_data)


def get_privacy_audit_log(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    operation_types: Optional[List[str]] = None,
    actor_id: Optional[str] = None,
) -> List[PrivacyOperation]:
    """Get privacy audit log with filters.

    Args:
        start_time: Filter operations after this time
        end_time: Filter operations before this time
        operation_types: Filter by operation types
        actor_id: Filter by actor

    Returns:
        List of PrivacyOperation objects
    """
    results = []

    for op in _privacy_audit_log:
        if start_time and op.timestamp < start_time:
            continue
        if end_time and op.timestamp > end_time:
            continue
        if operation_types and op.operation_type not in operation_types:
            continue
        if actor_id and op.actor_id != actor_id:
            continue

        results.append(op)

    return results


def get_privacy_summary() -> Dict[str, Any]:
    """Get privacy operations summary.

    Returns:
        Summary statistics
    """
    if not _privacy_audit_log:
        return {
            "total_operations": 0,
            "total_privacy_cost": 0.0,
            "operations_by_type": {},
            "success_rate": 0.0,
        }

    total_cost = sum(op.privacy_cost for op in _privacy_audit_log)
    success_count = sum(1 for op in _privacy_audit_log if op.success)

    from collections import Counter

    type_counts = Counter(op.operation_type for op in _privacy_audit_log)

    return {
        "total_operations": len(_privacy_audit_log),
        "total_privacy_cost": total_cost,
        "operations_by_type": dict(type_counts),
        "success_rate": success_count / len(_privacy_audit_log) * 100,
    }


def clear_privacy_audit_log() -> None:
    """Clear privacy audit log (for testing)."""
    global _privacy_audit_log
    _privacy_audit_log = []


# Additional functions for API compatibility
def emit_privacy_operation_receipt(
    operation_type: str,
    actor_id: str,
    target_type: str,
    target_id: str,
    details: Optional[Dict[str, Any]] = None,
    privacy_cost: float = 0.0,
    success: bool = True,
) -> Dict[str, Any]:
    """Track and emit privacy operation receipt in one call.

    Args:
        operation_type: Type of operation
        actor_id: Who performed the operation
        target_type: Type of target
        target_id: Target identifier
        details: Operation details
        privacy_cost: Epsilon spent
        success: Whether operation succeeded

    Returns:
        Receipt dict
    """
    operation = track_privacy_operation(
        operation_type, actor_id, target_type, target_id, details, privacy_cost, success
    )
    return emit_privacy_receipt(operation)


def get_privacy_operations(
    operation_types: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get privacy operations as dict list.

    Args:
        operation_types: Filter by operation types
        limit: Maximum results

    Returns:
        List of operation dicts
    """
    operations = get_privacy_audit_log(operation_types=operation_types)
    return [op.to_dict() for op in operations[:limit]]


def validate_privacy_compliance(
    actor_id: str,
    budget_threshold: float = 10.0,
    max_operations_per_day: int = 1000,
) -> Dict[str, Any]:
    """Validate privacy compliance for an actor.

    Args:
        actor_id: Actor to validate
        budget_threshold: Maximum budget allowed
        max_operations_per_day: Maximum operations per day

    Returns:
        Compliance result dict
    """
    operations = get_privacy_audit_log(actor_id=actor_id)
    total_cost = sum(op.privacy_cost for op in operations)

    # Check for today's operations
    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_operations = [op for op in operations if op.timestamp.startswith(today)]

    violations = []

    if total_cost > budget_threshold:
        violations.append(f"Budget exceeded: {total_cost:.2f} > {budget_threshold}")

    if len(today_operations) > max_operations_per_day:
        violations.append(f"Daily limit exceeded: {len(today_operations)} > {max_operations_per_day}")

    return {
        "actor_id": actor_id,
        "compliant": len(violations) == 0,
        "total_privacy_cost": total_cost,
        "operations_today": len(today_operations),
        "violations": violations,
    }
