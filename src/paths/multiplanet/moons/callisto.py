"""Callisto moon integration module.

This module handles Callisto ice operations within the multi-planet path.
Callisto is Jupiter's outermost large moon with low radiation and optimal hub location.

Functions:
- integrate_callisto: Wire Callisto ice operations to multi-planet path
- compute_callisto_autonomy: Compute Callisto-specific autonomy metrics
"""

from typing import Dict, Any, Optional

from ...base import emit_path_receipt
from ..core import (
    EXPANSION_SEQUENCE,
    LATENCY_BOUNDS_MIN,
    LATENCY_BOUNDS_MAX,
    AUTONOMY_REQUIREMENT,
    BANDWIDTH_BUDGET_MBPS,
    MULTIPLANET_TENANT_ID,
)


# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA = {
    "callisto_integrate": "mp_callisto_integrate",
    "callisto_autonomy": "mp_callisto_autonomy",
}
"""Receipt types emitted by Callisto module."""


# === EXPORTS ===

__all__ = [
    "integrate_callisto",
    "compute_callisto_autonomy",
    "RECEIPT_SCHEMA",
]


# === CALLISTO INTEGRATION ===


def integrate_callisto(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Callisto ice operations to multi-planet path.

    Args:
        config: Optional Callisto config override

    Returns:
        Dict with Callisto integration results

    Receipt: mp_callisto_integrate
    """
    # Import Callisto module
    from ...callisto_ice import (
        load_callisto_config,
        simulate_extraction,
        compute_autonomy,
        CALLISTO_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_callisto_config()

    # Get Callisto body config (add to sequence if not present)
    if "callisto" not in EXPANSION_SEQUENCE:
        EXPANSION_SEQUENCE.append("callisto")
        LATENCY_BOUNDS_MIN["callisto"] = 33
        LATENCY_BOUNDS_MAX["callisto"] = 53
        AUTONOMY_REQUIREMENT["callisto"] = 0.98
        BANDWIDTH_BUDGET_MBPS["callisto"] = 15

    # Run extraction simulation
    extraction = simulate_extraction(rate_kg_hr=100, duration_days=30)
    autonomy = compute_autonomy(extraction)

    result = {
        "integrated": True,
        "body": "callisto",
        "callisto_config": config,
        "extraction_simulation": {
            "duration_days": extraction["duration_days"],
            "total_extracted_kg": extraction["total_extracted_kg"],
            "energy_kwh": extraction["energy_kwh"],
            "autonomy_achieved": extraction["autonomy_achieved"],
        },
        "autonomy_requirement": CALLISTO_AUTONOMY_REQUIREMENT,
        "autonomy_met": autonomy >= CALLISTO_AUTONOMY_REQUIREMENT,
        "hub_suitability": "optimal",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "callisto_integrate", result)
    return result


def compute_callisto_autonomy() -> float:
    """Compute Callisto-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_callisto_autonomy
    """
    # Import Callisto module
    from ...callisto_ice import (
        load_callisto_config,
        simulate_extraction,
        compute_autonomy,
    )

    config = load_callisto_config()
    extraction = simulate_extraction(rate_kg_hr=100, duration_days=30)
    autonomy = compute_autonomy(extraction)

    result = {
        "body": "callisto",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "radiation_level": config["radiation_level"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "callisto_autonomy", result)
    return autonomy
