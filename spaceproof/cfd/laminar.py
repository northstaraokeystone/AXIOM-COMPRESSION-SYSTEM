"""Laminar regime CFD functions (Re < 2300).

PARADIGM:
    Laminar flow regime particle dynamics using Stokes settling.

THE PHYSICS:
    - Reynolds number (Mars): ~50 (laminar regime)
    - Settling model: Stokes (low-Re particles)

STOKES SETTLING:
    v_s = (2 * r^2 * g * (rho_p - rho_f)) / (9 * mu)

    Where:
    - v_s: Settling velocity
    - r: Particle radius
    - g: Gravitational acceleration
    - rho_p: Particle density
    - rho_f: Fluid density
    - mu: Dynamic viscosity

Functions:
    - stokes_settling: Compute Stokes settling velocity
    - simulate_particle_trajectory: Simulate particle trajectory under wind and gravity
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, Tuple

from spaceproof.core import dual_hash, emit_receipt

from .constants import (
    CFD_DENSITY_MARS_KG_M3,
    CFD_GRAVITY_MARS_M_S2,
    CFD_PARTICLE_DENSITY_KG_M3,
    CFD_SETTLING_MODEL,
    CFD_TENANT_ID,
    CFD_VISCOSITY_MARS_PA_S,
)

# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA: Dict[str, Any] = {
    "cfd_settling": {
        "tenant_id": "spaceproof-cfd",
        "receipt_type": "cfd_settling",
        "description": "Stokes settling velocity computation",
    },
    "cfd_trajectory": {
        "tenant_id": "spaceproof-cfd",
        "receipt_type": "cfd_trajectory",
        "description": "Particle trajectory simulation",
    },
}


# === STOKES SETTLING ===


def stokes_settling(
    particle_size_um: float,
    particle_density: float = CFD_PARTICLE_DENSITY_KG_M3,
    fluid_density: float = CFD_DENSITY_MARS_KG_M3,
    viscosity: float = CFD_VISCOSITY_MARS_PA_S,
    gravity: float = CFD_GRAVITY_MARS_M_S2,
) -> float:
    """Compute Stokes settling velocity for particle.

    v_s = (2 * r^2 * g * (rho_p - rho_f)) / (9 * mu)

    Args:
        particle_size_um: Particle diameter in micrometers
        particle_density: Particle density in kg/m^3
        fluid_density: Fluid density in kg/m^3
        viscosity: Dynamic viscosity in Pa*s
        gravity: Gravitational acceleration in m/s^2

    Returns:
        Settling velocity in m/s

    Receipt: cfd_settling_receipt
    """
    # Convert diameter to radius in meters
    radius_m = (particle_size_um * 1e-6) / 2

    # Stokes settling velocity
    if viscosity <= 0:
        return 0.0

    v_s = (2 * radius_m**2 * gravity * (particle_density - fluid_density)) / (
        9 * viscosity
    )

    # Ensure non-negative (particle should be denser than fluid)
    v_s = max(0.0, v_s)

    emit_receipt(
        "cfd_settling",
        {
            "receipt_type": "cfd_settling",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "particle_size_um": particle_size_um,
            "settling_velocity_m_s": round(v_s, 8),
            "model": CFD_SETTLING_MODEL,
            "payload_hash": dual_hash(
                json.dumps({"v_s": round(v_s, 8)}, sort_keys=True)
            ),
        },
    )

    return v_s


# === PARTICLE TRAJECTORY ===


def simulate_particle_trajectory(
    initial_pos: Tuple[float, float, float],
    wind: Tuple[float, float, float],
    duration_s: float,
    particle_size_um: float = 10.0,
    dt: float = 0.1,
) -> Dict[str, Any]:
    """Simulate dust particle trajectory under wind and gravity.

    Args:
        initial_pos: Initial position (x, y, z) in meters
        wind: Wind velocity (vx, vy, vz) in m/s
        duration_s: Simulation duration in seconds
        particle_size_um: Particle size in micrometers
        dt: Time step in seconds

    Returns:
        Dict with trajectory results

    Receipt: cfd_trajectory_receipt
    """
    # Get settling velocity
    v_settle = stokes_settling(particle_size_um)

    # Initialize position
    x, y, z = initial_pos
    wx, wy, wz = wind

    # Track trajectory
    trajectory = [(x, y, z)]
    t = 0.0

    while t < duration_s and z > 0:
        # Update position (wind + settling)
        x += wx * dt
        y += wy * dt
        z -= v_settle * dt  # Settling is downward

        # Apply some wind effect on horizontal position
        x += wz * 0.1 * dt  # Small turbulent dispersion

        trajectory.append((round(x, 4), round(y, 4), max(0, round(z, 4))))
        t += dt

        # Check if particle has settled
        if z <= 0:
            break

    # Final position
    final_pos = trajectory[-1]

    # Compute horizontal displacement
    dx = final_pos[0] - initial_pos[0]
    dy = final_pos[1] - initial_pos[1]
    horizontal_distance = math.sqrt(dx**2 + dy**2)

    result = {
        "initial_pos": initial_pos,
        "final_pos": final_pos,
        "wind": wind,
        "duration_s": round(t, 2),
        "particle_size_um": particle_size_um,
        "settling_velocity_m_s": round(v_settle, 8),
        "horizontal_distance_m": round(horizontal_distance, 2),
        "settled": final_pos[2] <= 0,
        "trajectory_points": len(trajectory),
    }

    emit_receipt(
        "cfd_trajectory",
        {
            "receipt_type": "cfd_trajectory",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "particle_size_um": particle_size_um,
            "duration_s": result["duration_s"],
            "horizontal_distance_m": result["horizontal_distance_m"],
            "settled": result["settled"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === EXPORTS ===

__all__ = [
    "RECEIPT_SCHEMA",
    "stokes_settling",
    "simulate_particle_trajectory",
]
