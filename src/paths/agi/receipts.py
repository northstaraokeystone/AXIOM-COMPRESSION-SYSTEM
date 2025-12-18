"""AGI ethics path receipt helpers.

Receipt types:
- agi_status_receipt: Path status
- agi_policy_receipt: Policy generation results
- agi_ethics_receipt: Ethics evaluation results
- agi_alignment_receipt: Alignment metrics
- agi_audit_receipt: Decision audit trail

KEY INSIGHT: "Audit trail IS alignment"

Source: AXIOM scalable paths architecture - AGI ethics modeling
"""

from typing import Dict, Any

from ..base import emit_path_receipt


# === RECEIPT EMISSION HELPERS ===


def emit_agi_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """Emit AGI status receipt.

    Args:
        status: Status data

    Returns:
        Complete receipt
    """
    return emit_path_receipt("agi", "status", {**status, "receipt_subtype": "status"})


def emit_agi_policy(policy_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit AGI policy generation receipt.

    Args:
        policy_result: Policy generation results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "agi", "policy", {**policy_result, "receipt_subtype": "fractal_policy"}
    )


def emit_agi_ethics(ethics_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit AGI ethics evaluation receipt.

    Args:
        ethics_result: Ethics evaluation results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "agi", "ethics", {**ethics_result, "receipt_subtype": "ethics_evaluation"}
    )


def emit_agi_alignment(alignment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit AGI alignment receipt.

    Args:
        alignment_result: Alignment calculation results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "agi", "alignment", {**alignment_result, "receipt_subtype": "alignment_metric"}
    )


def emit_agi_audit(audit_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit AGI audit receipt.

    Args:
        audit_result: Audit trail results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "agi", "audit", {**audit_result, "receipt_subtype": "decision_audit"}
    )
