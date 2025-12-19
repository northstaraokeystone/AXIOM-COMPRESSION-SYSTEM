"""Titan moon integration module.

This module handles Titan methane harvesting operations within the multi-planet path.
Titan is Saturn's largest moon with abundant methane resources for fuel production.

Functions:
- integrate_titan: Wire Titan methane harvesting to multi-planet path
- compute_titan_autonomy: Compute Titan-specific autonomy metrics
- simulate_titan_methane: Run Titan methane simulation
"""

from typing import Dict, Any, Optional

from ...base import emit_path_receipt
from ..core import (
    get_body_config,
    EXPANSION_SEQUENCE,
    MULTIPLANET_TENANT_ID,
)


# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA = {
    "titan_integrate": "mp_titan_integrate",
    "titan_autonomy": "mp_titan_autonomy",
    "titan_simulate": "mp_titan_simulate",
}
"""Receipt types emitted by Titan module."""


# === EXPORTS ===

__all__ = [
    "integrate_titan",
    "compute_titan_autonomy",
    "simulate_titan_methane",
    "RECEIPT_SCHEMA",
]


# === TITAN INTEGRATION ===


def integrate_titan(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Titan methane harvesting to multi-planet path.

    Args:
        config: Optional Titan config override

    Returns:
        Dict with Titan integration results

    Receipt: mp_titan_integrate
    """
    # Import Titan module
    from ...titan_methane_hybrid import (
        load_titan_config,
        simulate_harvest,
        TITAN_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_titan_config()

    # Get Titan body config
    titan_body = get_body_config("titan")

    # Run harvest simulation
    harvest = simulate_harvest(duration_days=30)

    result = {
        "integrated": True,
        "body": "titan",
        "body_config": titan_body,
        "titan_config": config,
        "harvest_simulation": {
            "duration_days": harvest["duration_days"],
            "processed_kg": harvest["processed_kg"],
            "energy_kwh": harvest["energy_kwh"],
            "autonomy_achieved": harvest["autonomy_achieved"],
        },
        "autonomy_requirement": TITAN_AUTONOMY_REQUIREMENT,
        "autonomy_met": harvest["autonomy_achieved"] >= TITAN_AUTONOMY_REQUIREMENT,
        "sequence_position": EXPANSION_SEQUENCE.index("titan") + 1,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_integrate", result)
    return result


def compute_titan_autonomy() -> float:
    """Compute Titan-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_titan_autonomy
    """
    # Import Titan module
    from ...titan_methane_hybrid import (
        load_titan_config,
        simulate_harvest,
    )

    config = load_titan_config()
    harvest = simulate_harvest(duration_days=30)

    autonomy = harvest["autonomy_achieved"]

    result = {
        "body": "titan",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_autonomy", result)
    return autonomy


def simulate_titan_methane(
    duration_days: int = 30, extraction_rate_kg_hr: float = 10.0
) -> Dict[str, Any]:
    """Run Titan methane simulation within multiplanet context.

    Args:
        duration_days: Simulation duration
        extraction_rate_kg_hr: Extraction rate

    Returns:
        Dict with simulation results

    Receipt: mp_titan_simulate
    """
    # Import Titan module
    from ...titan_methane_hybrid import (
        simulate_harvest,
        methane_to_fuel,
        TITAN_AUTONOMY_REQUIREMENT,
    )

    # Run harvest simulation
    harvest = simulate_harvest(duration_days, extraction_rate_kg_hr)

    # Get fuel conversion metrics
    fuel = methane_to_fuel(harvest["processed_kg"])

    result = {
        "body": "titan",
        "simulation_type": "methane_harvest",
        "duration_days": duration_days,
        "extraction_rate_kg_hr": extraction_rate_kg_hr,
        "harvest": {
            "processed_kg": harvest["processed_kg"],
            "energy_kwh": harvest["energy_kwh"],
            "autonomy_achieved": harvest["autonomy_achieved"],
        },
        "fuel_conversion": fuel,
        "autonomy_met": harvest["autonomy_achieved"] >= TITAN_AUTONOMY_REQUIREMENT,
        "sequence": EXPANSION_SEQUENCE,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_simulate", result)
    return result
