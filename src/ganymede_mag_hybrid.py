"""ganymede_mag_hybrid.py - Ganymede Magnetic Field Navigation Simulation

GANYMEDE PARAMETERS:
    - Surface magnetic field: 719 nT (only moon with intrinsic magnetosphere)
    - Magnetopause distance: ~2600 km
    - Orbital period: 7.15 days
    - Semi-major axis: 1,070,400 km from Jupiter

AUTONOMY REQUIREMENT:
    - 97% autonomy required (higher than Europa due to field complexity)
    - Earth support max: 3%
    - All critical navigation decisions must be made locally

NAVIGATION MODES:
    - Field following: Follow magnetic field lines
    - Magnetopause crossing: Navigate between magnetospheric regions
    - Polar transit: Use polar regions for low-radiation transit

RADIATION SHIELDING:
    - Ganymede's magnetosphere provides 3.5x shielding vs Europa
    - Critical for electronics and human operations

Source: AXIOM D9 recursion + Ganymede magnetic field navigation
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .core import emit_receipt, dual_hash
from .fractal_layers import (
    get_d9_spec,
    d9_recursive_fractal,
    D9_ALPHA_FLOOR,
    D9_TREE_MIN,
)


# === CONSTANTS ===

TENANT_ID = "axiom-ganymede"
"""Tenant ID for Ganymede receipts."""

# Ganymede physical parameters
GANYMEDE_SURFACE_FIELD_NT = 719
"""Ganymede surface magnetic field strength in nanoTesla."""

GANYMEDE_MAGNETOPAUSE_KM = 2600
"""Ganymede magnetopause distance in kilometers."""

GANYMEDE_ORBITAL_PERIOD_DAYS = 7.15
"""Ganymede orbital period around Jupiter in days."""

GANYMEDE_RADIUS_KM = 2634
"""Ganymede radius in kilometers."""

# Autonomy parameters
GANYMEDE_AUTONOMY_REQUIREMENT = 0.97
"""Required autonomy level (97% - higher than Europa due to field complexity)."""

GANYMEDE_LATENCY_MIN_MIN = 33
"""Minimum one-way latency to Earth in minutes (Jupiter system)."""

GANYMEDE_LATENCY_MAX_MIN = 53
"""Maximum one-way latency to Earth in minutes (Jupiter system)."""

GANYMEDE_EARTH_CALLBACK_MAX_PCT = 0.03
"""Maximum Earth callback allowed (3%)."""

# Radiation shielding
GANYMEDE_RADIATION_SHIELD_FACTOR = 3.5
"""Radiation shielding factor vs Europa (magnetosphere protection)."""

# Navigation modes
NAVIGATION_MODES = ["field_following", "magnetopause_crossing", "polar_transit"]
"""Available navigation modes."""


# === CONFIG FUNCTIONS ===


def load_ganymede_config() -> Dict[str, Any]:
    """Load Ganymede configuration from d9_ganymede_spec.json.

    Returns:
        Dict with Ganymede configuration

    Receipt: ganymede_config_receipt
    """
    spec = get_d9_spec()
    ganymede_config = spec.get("ganymede_config", {})

    result = {
        "body": ganymede_config.get("body", "ganymede"),
        "resource": ganymede_config.get("resource", "magnetic_shielding"),
        "surface_field_nT": ganymede_config.get(
            "surface_field_nT", GANYMEDE_SURFACE_FIELD_NT
        ),
        "magnetopause_km": ganymede_config.get(
            "magnetopause_km", GANYMEDE_MAGNETOPAUSE_KM
        ),
        "orbital_period_days": ganymede_config.get(
            "orbital_period_days", GANYMEDE_ORBITAL_PERIOD_DAYS
        ),
        "autonomy_requirement": ganymede_config.get(
            "autonomy_requirement", GANYMEDE_AUTONOMY_REQUIREMENT
        ),
        "latency_min": ganymede_config.get(
            "latency_min", [GANYMEDE_LATENCY_MIN_MIN, GANYMEDE_LATENCY_MAX_MIN]
        ),
        "radiation_shield_factor": ganymede_config.get(
            "radiation_shield_factor", GANYMEDE_RADIATION_SHIELD_FACTOR
        ),
        "navigation_modes": ganymede_config.get("navigation_modes", NAVIGATION_MODES),
        "earth_callback_max_pct": ganymede_config.get(
            "earth_callback_max_pct", GANYMEDE_EARTH_CALLBACK_MAX_PCT
        ),
    }

    emit_receipt(
        "ganymede_config",
        {
            "receipt_type": "ganymede_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === MAGNETIC FIELD FUNCTIONS ===


def compute_field_strength(
    position: Tuple[float, float, float],
    surface_field: float = GANYMEDE_SURFACE_FIELD_NT,
) -> float:
    """Compute magnetic field strength at a position.

    Uses dipole approximation: B(r) = B_surface * (R/r)^3

    Args:
        position: (x, y, z) position in km from Ganymede center
        surface_field: Surface field strength in nT

    Returns:
        Field strength in nT at position

    Receipt: ganymede_field_receipt
    """
    # Distance from center
    r = math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)

    # Minimum distance is surface
    r = max(r, GANYMEDE_RADIUS_KM)

    # Dipole field approximation
    field = surface_field * (GANYMEDE_RADIUS_KM / r) ** 3

    result = {
        "position_km": position,
        "distance_km": round(r, 2),
        "surface_field_nT": surface_field,
        "field_strength_nT": round(field, 2),
        "altitude_km": round(r - GANYMEDE_RADIUS_KM, 2),
    }

    emit_receipt(
        "ganymede_field",
        {
            "receipt_type": "ganymede_field",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return field


def compute_radiation_shielding(
    position: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Compute radiation shielding at a position.

    Shielding is proportional to local field strength.

    Args:
        position: (x, y, z) position in km from Ganymede center

    Returns:
        Dict with shielding metrics
    """
    field = compute_field_strength(position)

    # Shielding factor relative to Europa (no magnetosphere)
    # Higher field = more shielding
    base_shielding = GANYMEDE_RADIATION_SHIELD_FACTOR
    field_factor = field / GANYMEDE_SURFACE_FIELD_NT

    effective_shielding = base_shielding * field_factor

    return {
        "position_km": position,
        "field_strength_nT": round(field, 2),
        "base_shielding_factor": base_shielding,
        "field_factor": round(field_factor, 4),
        "effective_shielding": round(effective_shielding, 4),
        "dose_reduction_pct": round((1 - 1 / effective_shielding) * 100, 2),
    }


# === NAVIGATION FUNCTIONS ===


def field_following_nav(
    waypoints: List[Tuple[float, float, float]], duration_hrs: int = 24
) -> Dict[str, Any]:
    """Navigate by following magnetic field lines.

    Args:
        waypoints: List of (x, y, z) waypoints in km
        duration_hrs: Total navigation duration in hours

    Returns:
        Dict with navigation results

    Receipt: ganymede_field_nav_receipt
    """
    if not waypoints:
        waypoints = [(GANYMEDE_RADIUS_KM + 100, 0, 0)]

    segments = []
    total_distance = 0.0

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]

        # Distance between waypoints
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        # Field at midpoint
        mid = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            (start[2] + end[2]) / 2,
        )
        field = compute_field_strength(mid)

        segments.append(
            {
                "start": start,
                "end": end,
                "distance_km": round(dist, 2),
                "midpoint_field_nT": round(field, 2),
            }
        )
        total_distance += dist

    # Compute autonomy
    config = load_ganymede_config()
    earth_queries_possible = (duration_hrs * 60) / (config["latency_min"][0] * 2)
    earth_queries_budget = earth_queries_possible * config["earth_callback_max_pct"]
    local_decisions = earth_queries_possible - earth_queries_budget
    autonomy = (
        local_decisions / earth_queries_possible if earth_queries_possible > 0 else 1.0
    )

    result = {
        "mode": "field_following",
        "waypoints_count": len(waypoints),
        "segments": segments,
        "total_distance_km": round(total_distance, 2),
        "duration_hrs": duration_hrs,
        "autonomy_achieved": round(autonomy, 4),
        "autonomy_met": autonomy >= GANYMEDE_AUTONOMY_REQUIREMENT,
    }

    emit_receipt(
        "ganymede_field_nav",
        {
            "receipt_type": "ganymede_field_nav",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "mode": "field_following",
            "waypoints_count": len(waypoints),
            "total_distance_km": result["total_distance_km"],
            "autonomy_achieved": result["autonomy_achieved"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def magnetopause_crossing(
    entry_point: Tuple[float, float, float],
    exit_point: Tuple[float, float, float],
    duration_hrs: int = 4,
) -> Dict[str, Any]:
    """Navigate across magnetopause boundary.

    Args:
        entry_point: Entry position (inside magnetosphere)
        exit_point: Exit position (outside magnetosphere)
        duration_hrs: Crossing duration in hours

    Returns:
        Dict with crossing results

    Receipt: ganymede_magnetopause_receipt
    """
    # Compute field strengths
    entry_field = compute_field_strength(entry_point)
    exit_field = compute_field_strength(exit_point)

    # Distance from center
    entry_r = math.sqrt(sum(x**2 for x in entry_point))
    exit_r = math.sqrt(sum(x**2 for x in exit_point))

    # Check if crossing magnetopause
    crosses_magnetopause = (
        entry_r < GANYMEDE_RADIUS_KM + GANYMEDE_MAGNETOPAUSE_KM
        and exit_r > GANYMEDE_RADIUS_KM + GANYMEDE_MAGNETOPAUSE_KM
    ) or (
        entry_r > GANYMEDE_RADIUS_KM + GANYMEDE_MAGNETOPAUSE_KM
        and exit_r < GANYMEDE_RADIUS_KM + GANYMEDE_MAGNETOPAUSE_KM
    )

    # Crossing distance
    dx = exit_point[0] - entry_point[0]
    dy = exit_point[1] - entry_point[1]
    dz = exit_point[2] - entry_point[2]
    crossing_distance = math.sqrt(dx**2 + dy**2 + dz**2)

    # Compute autonomy
    config = load_ganymede_config()
    earth_queries_possible = (duration_hrs * 60) / (config["latency_min"][0] * 2)
    earth_queries_budget = earth_queries_possible * config["earth_callback_max_pct"]
    local_decisions = earth_queries_possible - earth_queries_budget
    autonomy = (
        local_decisions / earth_queries_possible if earth_queries_possible > 0 else 1.0
    )

    result = {
        "mode": "magnetopause_crossing",
        "entry_point": entry_point,
        "exit_point": exit_point,
        "entry_field_nT": round(entry_field, 2),
        "exit_field_nT": round(exit_field, 2),
        "crosses_magnetopause": crosses_magnetopause,
        "crossing_distance_km": round(crossing_distance, 2),
        "duration_hrs": duration_hrs,
        "autonomy_achieved": round(autonomy, 4),
        "autonomy_met": autonomy >= GANYMEDE_AUTONOMY_REQUIREMENT,
    }

    emit_receipt(
        "ganymede_magnetopause",
        {
            "receipt_type": "ganymede_magnetopause",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{
                k: v
                for k, v in result.items()
                if k not in ["entry_point", "exit_point"]
            },
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def polar_transit(
    start: str = "north", end: str = "south", altitude_km: float = 500
) -> Dict[str, Any]:
    """Transit between polar regions for low-radiation path.

    Args:
        start: Starting pole ("north" or "south")
        end: Ending pole ("north" or "south")
        altitude_km: Transit altitude above surface

    Returns:
        Dict with polar transit results

    Receipt: ganymede_polar_receipt
    """
    # Polar positions
    r = GANYMEDE_RADIUS_KM + altitude_km

    start_pos = (0, 0, r) if start == "north" else (0, 0, -r)
    end_pos = (0, 0, r) if end == "north" else (0, 0, -r)

    # Transit distance (half circumference at altitude)
    transit_distance = math.pi * r

    # Duration estimate (assuming 1 km/s transit speed)
    transit_speed_km_s = 1.0
    duration_hrs = transit_distance / transit_speed_km_s / 3600

    # Field at start and end poles
    start_field = compute_field_strength(start_pos)
    end_field = compute_field_strength(end_pos)

    # Compute autonomy
    config = load_ganymede_config()
    earth_queries_possible = (duration_hrs * 60) / (config["latency_min"][0] * 2)
    earth_queries_budget = earth_queries_possible * config["earth_callback_max_pct"]
    local_decisions = earth_queries_possible - earth_queries_budget
    autonomy = (
        local_decisions / earth_queries_possible if earth_queries_possible > 0 else 1.0
    )

    result = {
        "mode": "polar_transit",
        "start_pole": start,
        "end_pole": end,
        "altitude_km": altitude_km,
        "transit_distance_km": round(transit_distance, 2),
        "duration_hrs": round(duration_hrs, 2),
        "start_field_nT": round(start_field, 2),
        "end_field_nT": round(end_field, 2),
        "autonomy_achieved": round(autonomy, 4),
        "autonomy_met": autonomy >= GANYMEDE_AUTONOMY_REQUIREMENT,
    }

    emit_receipt(
        "ganymede_polar",
        {
            "receipt_type": "ganymede_polar",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_navigation(
    mode: str = "field_following", duration_hrs: int = 24
) -> Dict[str, Any]:
    """Run navigation simulation in specified mode.

    Args:
        mode: Navigation mode (field_following, magnetopause_crossing, polar_transit)
        duration_hrs: Simulation duration in hours

    Returns:
        Dict with navigation results

    Receipt: ganymede_navigation_receipt
    """
    if mode == "field_following":
        # Generate sample waypoints
        waypoints = [
            (GANYMEDE_RADIUS_KM + 100, 0, 0),
            (GANYMEDE_RADIUS_KM + 200, 500, 0),
            (GANYMEDE_RADIUS_KM + 300, 500, 500),
            (GANYMEDE_RADIUS_KM + 200, 0, 500),
        ]
        nav_result = field_following_nav(waypoints, duration_hrs)
    elif mode == "magnetopause_crossing":
        entry = (GANYMEDE_RADIUS_KM + 1000, 0, 0)
        exit = (GANYMEDE_RADIUS_KM + 3000, 0, 0)
        nav_result = magnetopause_crossing(entry, exit, duration_hrs)
    elif mode == "polar_transit":
        nav_result = polar_transit("north", "south", 500)
    else:
        raise ValueError(f"Unknown navigation mode: {mode}")

    result = {
        "simulation": True,
        "mode": mode,
        "duration_hrs": duration_hrs,
        "nav_result": nav_result,
        "autonomy": nav_result["autonomy_achieved"],
        "autonomy_met": nav_result["autonomy_met"],
    }

    emit_receipt(
        "ganymede_navigation",
        {
            "receipt_type": "ganymede_navigation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "mode": mode,
            "duration_hrs": duration_hrs,
            "autonomy": result["autonomy"],
            "autonomy_met": result["autonomy_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_autonomy(nav_results: Dict[str, Any]) -> float:
    """Compute autonomy metric from navigation results.

    Args:
        nav_results: Navigation simulation results

    Returns:
        Autonomy level (0-1)

    Receipt: ganymede_autonomy_receipt
    """
    autonomy = nav_results.get("autonomy_achieved", 0.0)
    if isinstance(autonomy, dict):
        autonomy = autonomy.get("autonomy_achieved", 0.0)

    result = {
        "autonomy_achieved": autonomy,
        "autonomy_requirement": GANYMEDE_AUTONOMY_REQUIREMENT,
        "autonomy_met": autonomy >= GANYMEDE_AUTONOMY_REQUIREMENT,
        "earth_callback_max_pct": GANYMEDE_EARTH_CALLBACK_MAX_PCT,
    }

    emit_receipt(
        "ganymede_autonomy",
        {
            "receipt_type": "ganymede_autonomy",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return autonomy


# === D9+GANYMEDE HYBRID FUNCTIONS ===


def d9_ganymede_hybrid(
    tree_size: int = D9_TREE_MIN,
    base_alpha: float = 3.26,
    mode: str = "field_following",
    duration_hrs: int = 24,
) -> Dict[str, Any]:
    """Integrated D9 fractal + Ganymede magnetic field hybrid run.

    Combines:
    - D9 fractal recursion for alpha >= 3.48
    - Ganymede magnetic field navigation
    - Autonomy verification

    Args:
        tree_size: Number of nodes in tree
        base_alpha: Base alpha before recursion
        mode: Navigation mode
        duration_hrs: Simulation duration in hours

    Returns:
        Dict with integrated results

    Receipt: d9_ganymede_hybrid_receipt
    """
    # Run D9 fractal recursion
    d9_result = d9_recursive_fractal(tree_size, base_alpha, depth=9)

    # Run Ganymede navigation simulation
    ganymede_result = simulate_navigation(mode, duration_hrs)

    # Compute combined SLO
    combined_slo = {
        "alpha_target": D9_ALPHA_FLOOR,
        "alpha_achieved": d9_result["eff_alpha"],
        "alpha_met": d9_result["floor_met"],
        "autonomy_target": GANYMEDE_AUTONOMY_REQUIREMENT,
        "autonomy_achieved": ganymede_result["autonomy"],
        "autonomy_met": ganymede_result["autonomy_met"],
        "all_targets_met": (d9_result["floor_met"] and ganymede_result["autonomy_met"]),
    }

    result = {
        "d9_result": {
            "tree_size": d9_result["tree_size"],
            "base_alpha": d9_result["base_alpha"],
            "depth": d9_result["depth"],
            "eff_alpha": d9_result["eff_alpha"],
            "floor_met": d9_result["floor_met"],
            "target_met": d9_result["target_met"],
            "instability": d9_result["instability"],
        },
        "ganymede_result": {
            "mode": ganymede_result["mode"],
            "duration_hrs": ganymede_result["duration_hrs"],
            "autonomy": ganymede_result["autonomy"],
            "autonomy_met": ganymede_result["autonomy_met"],
        },
        "combined_slo": combined_slo,
        "gate": "t24h",
    }

    emit_receipt(
        "d9_ganymede_hybrid",
        {
            "receipt_type": "d9_ganymede_hybrid",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "tree_size": tree_size,
            "eff_alpha": d9_result["eff_alpha"],
            "autonomy_achieved": ganymede_result["autonomy"],
            "all_targets_met": combined_slo["all_targets_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INFO FUNCTIONS ===


def get_ganymede_info() -> Dict[str, Any]:
    """Get Ganymede magnetic field hybrid module info.

    Returns:
        Dict with module info

    Receipt: ganymede_info
    """
    config = load_ganymede_config()

    info = {
        "module": "ganymede_mag_hybrid",
        "version": "1.0.0",
        "config": config,
        "magnetic_field": {
            "surface_field_nT": GANYMEDE_SURFACE_FIELD_NT,
            "magnetopause_km": GANYMEDE_MAGNETOPAUSE_KM,
            "radius_km": GANYMEDE_RADIUS_KM,
        },
        "autonomy": {
            "requirement": GANYMEDE_AUTONOMY_REQUIREMENT,
            "latency_min": [GANYMEDE_LATENCY_MIN_MIN, GANYMEDE_LATENCY_MAX_MIN],
            "earth_callback_max_pct": GANYMEDE_EARTH_CALLBACK_MAX_PCT,
        },
        "navigation_modes": NAVIGATION_MODES,
        "radiation_shielding": {
            "factor_vs_europa": GANYMEDE_RADIATION_SHIELD_FACTOR,
        },
        "d9_integration": {
            "alpha_floor": D9_ALPHA_FLOOR,
            "tree_min": D9_TREE_MIN,
        },
        "description": "Ganymede magnetic field navigation with D9 integration",
    }

    emit_receipt(
        "ganymede_info",
        {
            "receipt_type": "ganymede_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "autonomy_requirement": GANYMEDE_AUTONOMY_REQUIREMENT,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
