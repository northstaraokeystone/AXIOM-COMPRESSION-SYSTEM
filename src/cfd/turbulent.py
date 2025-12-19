"""Turbulent regime CFD functions.

PARADIGM:
    Turbulent flow regime dust dynamics for Mars dust storms.

Functions:
    - simulate_dust_storm: Simulate Mars dust storm effects
    - compute_deposition_rate: Compute dust deposition rate
"""

import json
from datetime import datetime
from typing import Any, Dict

from src.core import dual_hash, emit_receipt

from .constants import CFD_TENANT_ID

# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA: Dict[str, Any] = {
    "cfd_storm": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "cfd_storm",
        "description": "Dust storm simulation",
    },
}


# === DUST STORM SIMULATION ===


def simulate_dust_storm(
    intensity: float = 0.5,
    duration_hrs: float = 24.0,
    area_km2: float = 100.0,
) -> Dict[str, Any]:
    """Simulate Mars dust storm effects.

    Args:
        intensity: Storm intensity (0-1)
        duration_hrs: Storm duration in hours
        area_km2: Affected area in km^2

    Returns:
        Dict with dust storm simulation results

    Receipt: cfd_storm_receipt
    """
    # Storm parameters
    wind_speed_m_s = 10 + intensity * 90  # 10-100 m/s range
    particle_flux_kg_m2_s = intensity * 1e-6  # kg/m^2/s

    # Total dust mobilized
    total_dust_kg = particle_flux_kg_m2_s * (area_km2 * 1e6) * (duration_hrs * 3600)

    # Visibility reduction
    visibility_km = 10 * (1 - intensity)  # 0-10 km

    # Solar panel efficiency impact
    solar_impact = intensity * 0.5  # Up to 50% reduction

    result = {
        "intensity": intensity,
        "duration_hrs": duration_hrs,
        "area_km2": area_km2,
        "wind_speed_m_s": round(wind_speed_m_s, 1),
        "particle_flux_kg_m2_s": particle_flux_kg_m2_s,
        "total_dust_kg": round(total_dust_kg, 0),
        "visibility_km": round(visibility_km, 1),
        "solar_impact_fraction": round(solar_impact, 2),
        "storm_category": "regional" if intensity < 0.7 else "global",
    }

    emit_receipt(
        "cfd_storm",
        {
            "receipt_type": "cfd_storm",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "intensity": intensity,
            "duration_hrs": duration_hrs,
            "wind_speed_m_s": result["wind_speed_m_s"],
            "solar_impact": result["solar_impact_fraction"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === DEPOSITION RATE ===


def compute_deposition_rate(
    settling_velocity: float,
    concentration: float = 1e-6,
) -> float:
    """Compute dust deposition rate.

    Args:
        settling_velocity: Particle settling velocity in m/s
        concentration: Particle concentration in kg/m^3

    Returns:
        Deposition rate in kg/m^2/s
    """
    # Deposition = settling velocity * concentration
    deposition = settling_velocity * concentration
    return deposition


# === EXPORTS ===

__all__ = [
    "RECEIPT_SCHEMA",
    "simulate_dust_storm",
    "compute_deposition_rate",
]
