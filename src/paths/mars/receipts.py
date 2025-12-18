"""Mars path receipt helpers.

Receipt types:
- mars_status_receipt: Path status
- mars_dome_receipt: Dome simulation results
- mars_isru_receipt: ISRU closure metrics
- mars_sovereignty_receipt: Sovereignty threshold results
- mars_optimize_receipt: Optimization suggestions

Source: AXIOM scalable paths architecture - Mars autonomous habitat
"""

from typing import Dict, Any

from ..base import emit_path_receipt


# === RECEIPT EMISSION HELPERS ===


def emit_mars_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """Emit Mars status receipt.

    Args:
        status: Status data

    Returns:
        Complete receipt
    """
    return emit_path_receipt("mars", "status", {**status, "receipt_subtype": "status"})


def emit_mars_dome(dome_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit Mars dome simulation receipt.

    Args:
        dome_result: Dome simulation results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "mars", "dome", {**dome_result, "receipt_subtype": "dome_simulation"}
    )


def emit_mars_isru(isru_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit Mars ISRU closure receipt.

    Args:
        isru_result: ISRU calculation results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "mars", "isru", {**isru_result, "receipt_subtype": "isru_closure"}
    )


def emit_mars_sovereignty(sovereignty_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit Mars sovereignty receipt.

    Args:
        sovereignty_result: Sovereignty calculation results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "mars",
        "sovereignty",
        {**sovereignty_result, "receipt_subtype": "sovereignty_check"},
    )


def emit_mars_optimize(optimize_result: Dict[str, Any]) -> Dict[str, Any]:
    """Emit Mars optimization receipt.

    Args:
        optimize_result: Optimization results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "mars",
        "optimize",
        {**optimize_result, "receipt_subtype": "resource_optimization"},
    )
