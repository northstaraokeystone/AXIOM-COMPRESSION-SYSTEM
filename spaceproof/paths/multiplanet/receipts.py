"""Multi-planet path receipt helpers.

Receipt types:
- mp_status_receipt: Path status
- mp_sequence_receipt: Expansion sequence
- mp_body_receipt: Per-body configuration
- mp_telemetry_receipt: Compression metrics per body
- mp_latency_receipt: Latency budget per body
- mp_autonomy_receipt: Autonomy requirements

Source: SpaceProof scalable paths architecture - Multi-planet expansion
"""

from typing import Dict, Any

from ..base import emit_path_receipt


# === RECEIPT EMISSION HELPERS ===


def emit_mp_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """Emit multi-planet status receipt.

    Args:
        status: Status data

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "multiplanet", "status", {**status, "receipt_subtype": "status"}
    )


def emit_mp_sequence(sequence_data: Dict[str, Any]) -> Dict[str, Any]:
    """Emit expansion sequence receipt.

    Args:
        sequence_data: Sequence information

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "multiplanet",
        "sequence",
        {**sequence_data, "receipt_subtype": "expansion_sequence"},
    )


def emit_mp_body(body_config: Dict[str, Any]) -> Dict[str, Any]:
    """Emit body configuration receipt.

    Args:
        body_config: Body-specific configuration

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "multiplanet", "body", {**body_config, "receipt_subtype": "body_config"}
    )


def emit_mp_telemetry(telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
    """Emit telemetry compression receipt.

    Args:
        telemetry_data: Telemetry compression results

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "multiplanet",
        "telemetry",
        {**telemetry_data, "receipt_subtype": "telemetry_compression"},
    )


def emit_mp_latency(latency_data: Dict[str, Any]) -> Dict[str, Any]:
    """Emit latency budget receipt.

    Args:
        latency_data: Latency budget information

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "multiplanet", "latency", {**latency_data, "receipt_subtype": "latency_budget"}
    )


def emit_mp_autonomy(autonomy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Emit autonomy requirement receipt.

    Args:
        autonomy_data: Autonomy requirement information

    Returns:
        Complete receipt
    """
    return emit_path_receipt(
        "multiplanet",
        "autonomy",
        {**autonomy_data, "receipt_subtype": "autonomy_requirement"},
    )
