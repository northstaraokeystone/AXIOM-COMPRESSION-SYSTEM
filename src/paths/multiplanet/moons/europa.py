"""Europa moon integration module.

This module handles Europa ice drilling operations within the multi-planet path.
Europa is Jupiter's moon with subsurface ocean and water ice resources.

Functions:
- integrate_europa: Wire Europa ice drilling to multi-planet path
- compute_europa_autonomy: Compute Europa-specific autonomy metrics
- simulate_europa_drilling: Run Europa drilling simulation
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
    "europa_integrate": "mp_europa_integrate",
    "europa_autonomy": "mp_europa_autonomy",
    "europa_simulate": "mp_europa_simulate",
}
"""Receipt types emitted by Europa module."""


# === EXPORTS ===

__all__ = [
    "integrate_europa",
    "compute_europa_autonomy",
    "simulate_europa_drilling",
    "RECEIPT_SCHEMA",
]


# === EUROPA INTEGRATION ===


def integrate_europa(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Europa ice drilling to multi-planet path.

    Args:
        config: Optional Europa config override

    Returns:
        Dict with Europa integration results

    Receipt: mp_europa_integrate
    """
    # Import Europa module
    from ...europa_ice_hybrid import (
        load_europa_config,
        simulate_drilling,
        EUROPA_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_europa_config()

    # Get Europa body config
    europa_body = get_body_config("europa")

    # Run drilling simulation
    drilling = simulate_drilling(depth_m=1000, duration_days=30)

    result = {
        "integrated": True,
        "body": "europa",
        "body_config": europa_body,
        "europa_config": config,
        "drilling_simulation": {
            "depth_m": drilling["actual_depth_m"],
            "water_kg": drilling["water_extracted_kg"],
            "energy_kwh": drilling["melting_energy_kwh"],
            "autonomy_achieved": drilling["autonomy_achieved"],
        },
        "autonomy_requirement": EUROPA_AUTONOMY_REQUIREMENT,
        "autonomy_met": drilling["autonomy_achieved"] >= EUROPA_AUTONOMY_REQUIREMENT,
        "sequence_position": EXPANSION_SEQUENCE.index("europa") + 1,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "europa_integrate", result)
    return result


def compute_europa_autonomy() -> float:
    """Compute Europa-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_europa_autonomy
    """
    # Import Europa module
    from ...europa_ice_hybrid import (
        load_europa_config,
        simulate_drilling,
    )

    config = load_europa_config()
    drilling = simulate_drilling(depth_m=1000, duration_days=30)

    autonomy = drilling["autonomy_achieved"]

    result = {
        "body": "europa",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "europa_autonomy", result)
    return autonomy


def simulate_europa_drilling(
    depth_m: int = 1000, duration_days: int = 30, drill_rate_m_hr: float = 2.0
) -> Dict[str, Any]:
    """Run Europa drilling simulation within multiplanet context.

    Args:
        depth_m: Target drill depth in meters
        duration_days: Simulation duration
        drill_rate_m_hr: Drilling rate

    Returns:
        Dict with simulation results

    Receipt: mp_europa_simulate
    """
    # Import Europa module
    from ...europa_ice_hybrid import (
        simulate_drilling,
        ice_to_water,
        EUROPA_AUTONOMY_REQUIREMENT,
    )

    # Run drilling simulation
    drilling = simulate_drilling(depth_m, duration_days, drill_rate_m_hr)

    # Get water conversion metrics
    water = ice_to_water(drilling["ice_mass_kg"])

    result = {
        "body": "europa",
        "simulation_type": "ice_drilling",
        "depth_m": depth_m,
        "duration_days": duration_days,
        "drill_rate_m_hr": drill_rate_m_hr,
        "drilling": {
            "actual_depth_m": drilling["actual_depth_m"],
            "ice_mass_kg": drilling["ice_mass_kg"],
            "water_kg": drilling["water_extracted_kg"],
            "autonomy_achieved": drilling["autonomy_achieved"],
        },
        "water_conversion": water,
        "autonomy_met": drilling["autonomy_achieved"] >= EUROPA_AUTONOMY_REQUIREMENT,
        "sequence": EXPANSION_SEQUENCE,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "europa_simulate", result)
    return result
