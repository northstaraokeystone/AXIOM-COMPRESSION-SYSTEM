"""Ganymede moon integration module.

This module handles Ganymede magnetic field navigation operations within the multi-planet path.
Ganymede is Jupiter's largest moon with its own magnetic field for navigation and radiation shielding.

Functions:
- integrate_ganymede: Wire Ganymede magnetic field navigation to multi-planet path
- compute_ganymede_autonomy: Compute Ganymede-specific autonomy metrics
- simulate_ganymede_navigation: Run Ganymede navigation simulation
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
    "ganymede_integrate": "mp_ganymede_integrate",
    "ganymede_autonomy": "mp_ganymede_autonomy",
    "ganymede_simulate": "mp_ganymede_simulate",
}
"""Receipt types emitted by Ganymede module."""


# === EXPORTS ===

__all__ = [
    "integrate_ganymede",
    "compute_ganymede_autonomy",
    "simulate_ganymede_navigation",
    "RECEIPT_SCHEMA",
]


# === GANYMEDE INTEGRATION ===


def integrate_ganymede(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Ganymede magnetic field navigation to multi-planet path.

    Args:
        config: Optional Ganymede config override

    Returns:
        Dict with Ganymede integration results

    Receipt: mp_ganymede_integrate
    """
    # Import Ganymede module
    from ...ganymede_mag_hybrid import (
        load_ganymede_config,
        simulate_navigation,
        GANYMEDE_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_ganymede_config()

    # Get Ganymede body config
    ganymede_body = get_body_config("ganymede")

    # Run navigation simulation
    navigation = simulate_navigation(mode="field_following", duration_hrs=24)

    result = {
        "integrated": True,
        "body": "ganymede",
        "body_config": ganymede_body,
        "ganymede_config": config,
        "navigation_simulation": {
            "mode": navigation["mode"],
            "duration_hrs": navigation["duration_hrs"],
            "autonomy_achieved": navigation["autonomy"],
        },
        "autonomy_requirement": GANYMEDE_AUTONOMY_REQUIREMENT,
        "autonomy_met": navigation["autonomy"] >= GANYMEDE_AUTONOMY_REQUIREMENT,
        "sequence_position": EXPANSION_SEQUENCE.index("ganymede") + 1,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "ganymede_integrate", result)
    return result


def compute_ganymede_autonomy() -> float:
    """Compute Ganymede-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_ganymede_autonomy
    """
    # Import Ganymede module
    from ...ganymede_mag_hybrid import (
        load_ganymede_config,
        simulate_navigation,
    )

    config = load_ganymede_config()
    navigation = simulate_navigation(mode="field_following", duration_hrs=24)

    autonomy = navigation["autonomy"]

    result = {
        "body": "ganymede",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "ganymede_autonomy", result)
    return autonomy


def simulate_ganymede_navigation(
    mode: str = "field_following", duration_hrs: int = 24
) -> Dict[str, Any]:
    """Run Ganymede navigation simulation within multiplanet context.

    Args:
        mode: Navigation mode
        duration_hrs: Simulation duration

    Returns:
        Dict with simulation results

    Receipt: mp_ganymede_simulate
    """
    # Import Ganymede module
    from ...ganymede_mag_hybrid import (
        simulate_navigation,
        compute_radiation_shielding,
        GANYMEDE_AUTONOMY_REQUIREMENT,
        GANYMEDE_RADIUS_KM,
    )

    # Run navigation simulation
    navigation = simulate_navigation(mode, duration_hrs)

    # Get radiation shielding at typical position
    shielding = compute_radiation_shielding((GANYMEDE_RADIUS_KM + 500, 0, 0))

    result = {
        "body": "ganymede",
        "simulation_type": "magnetic_navigation",
        "mode": mode,
        "duration_hrs": duration_hrs,
        "navigation": {
            "autonomy_achieved": navigation["autonomy"],
            "autonomy_met": navigation["autonomy_met"],
        },
        "radiation_shielding": shielding,
        "autonomy_met": navigation["autonomy"] >= GANYMEDE_AUTONOMY_REQUIREMENT,
        "sequence": EXPANSION_SEQUENCE,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "ganymede_simulate", result)
    return result
