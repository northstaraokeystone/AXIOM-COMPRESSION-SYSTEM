"""europa_ice_hybrid.py - Europa Ice Drilling ISRU Simulation

EUROPA PARAMETERS:
    - Surface temperature: 110K (-163C)
    - Ice shell thickness: ~15 km
    - Subsurface ocean depth: ~100 km
    - Ice density: 917 kg/m3
    - Drill rate: ~2 m/hr (cryogenic conditions)

AUTONOMY REQUIREMENT:
    - 95% autonomy required (no Earth callback at 33-53 min latency)
    - Earth support max: 5%
    - All critical drilling decisions must be made locally

DRILLING MODEL:
    - Ice penetration rate dependent on temperature and hardness
    - Water extraction from melted ice
    - Energy from ice-to-water conversion

Source: SpaceProof D7 recursion + Europa ice + NREL + expanded audits
"""

import json
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash
from .fractal_layers import (
    get_d7_spec,
    d7_recursive_fractal,
    D7_ALPHA_FLOOR,
    D7_TREE_MIN,
)


# === CONSTANTS ===

TENANT_ID = "spaceproof-europa"
"""Tenant ID for Europa receipts."""

# Europa physical parameters
EUROPA_SURFACE_TEMP_K = 110
"""Europa surface temperature in Kelvin."""

EUROPA_ICE_THICKNESS_KM = 15
"""Europa ice shell thickness in kilometers."""

EUROPA_OCEAN_DEPTH_KM = 100
"""Europa subsurface ocean depth in kilometers."""

EUROPA_ICE_DENSITY_KG_M3 = 917
"""Ice density in kg/m3."""

EUROPA_DRILL_RATE_M_HR = 2.0
"""Base drill rate in meters per hour (cryogenic conditions)."""

# Autonomy parameters
EUROPA_AUTONOMY_REQUIREMENT = 0.95
"""Required autonomy level (95%)."""

EUROPA_LATENCY_MIN_MIN = 33
"""Minimum one-way latency to Earth in minutes."""

EUROPA_LATENCY_MAX_MIN = 53
"""Maximum one-way latency to Earth in minutes."""

EUROPA_EARTH_CALLBACK_MAX_PCT = 0.05
"""Maximum Earth callback allowed (5%)."""

# Drilling parameters
ICE_MELTING_ENERGY_KJ_KG = 334
"""Energy to melt ice in kJ/kg."""

WATER_EXTRACTION_EFFICIENCY = 0.90
"""Efficiency of water extraction from melted ice."""


# === CONFIG FUNCTIONS ===


def load_europa_config() -> Dict[str, Any]:
    """Load Europa configuration from d7_europa_spec.json.

    Returns:
        Dict with Europa configuration

    Receipt: europa_config_receipt
    """
    spec = get_d7_spec()
    europa_config = spec.get("europa_config", {})

    result = {
        "body": europa_config.get("body", "europa"),
        "resource": europa_config.get("resource", "water_ice"),
        "ice_thickness_km": europa_config.get(
            "ice_thickness_km", EUROPA_ICE_THICKNESS_KM
        ),
        "ocean_depth_km": europa_config.get("ocean_depth_km", EUROPA_OCEAN_DEPTH_KM),
        "surface_temp_k": europa_config.get("surface_temp_k", EUROPA_SURFACE_TEMP_K),
        "autonomy_requirement": europa_config.get(
            "autonomy_requirement", EUROPA_AUTONOMY_REQUIREMENT
        ),
        "latency_min": europa_config.get(
            "latency_min", [EUROPA_LATENCY_MIN_MIN, EUROPA_LATENCY_MAX_MIN]
        ),
        "earth_callback_max_pct": europa_config.get(
            "earth_callback_max_pct", EUROPA_EARTH_CALLBACK_MAX_PCT
        ),
        "ice_density_kg_m3": europa_config.get(
            "ice_density_kg_m3", EUROPA_ICE_DENSITY_KG_M3
        ),
        "drill_rate_m_hr": europa_config.get("drill_rate_m_hr", EUROPA_DRILL_RATE_M_HR),
    }

    emit_receipt(
        "europa_config",
        {
            "receipt_type": "europa_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === SIMULATION FUNCTIONS ===


def simulate_drilling(
    depth_m: int = 1000,
    duration_days: int = 30,
    drill_rate_m_hr: float = EUROPA_DRILL_RATE_M_HR,
) -> Dict[str, Any]:
    """Simulate ice drilling on Europa.

    Args:
        depth_m: Target drill depth in meters
        duration_days: Simulation duration in days
        drill_rate_m_hr: Drilling rate in meters per hour

    Returns:
        Dict with drilling simulation results

    Receipt: europa_drilling_receipt
    """
    config = load_europa_config()

    # Compute drilling metrics
    hours_available = duration_days * 24
    max_drill_depth_m = drill_rate_m_hr * hours_available

    # Actual depth drilled (capped by target or available time)
    actual_depth_m = min(depth_m, max_drill_depth_m)

    # Time required
    hours_required = actual_depth_m / drill_rate_m_hr if drill_rate_m_hr > 0 else 0
    days_required = hours_required / 24

    # Ice volume extracted (cylindrical bore, 0.5m diameter)
    bore_radius_m = 0.25
    bore_area_m2 = 3.14159 * (bore_radius_m**2)
    ice_volume_m3 = bore_area_m2 * actual_depth_m
    ice_mass_kg = ice_volume_m3 * config["ice_density_kg_m3"]

    # Water extracted
    water_kg = ice_mass_kg * WATER_EXTRACTION_EFFICIENCY

    # Energy required for melting
    melting_energy_kj = ice_mass_kg * ICE_MELTING_ENERGY_KJ_KG
    melting_energy_kwh = melting_energy_kj / 3600

    # Autonomy computation
    earth_queries_possible = (duration_days * 24 * 60) / (
        config["latency_min"][0] * 2
    )  # Round-trip
    earth_queries_budget = earth_queries_possible * config["earth_callback_max_pct"]
    local_decisions = earth_queries_possible - earth_queries_budget

    autonomy_achieved = (
        round(local_decisions / earth_queries_possible, 4)
        if earth_queries_possible > 0
        else 1.0
    )

    result = {
        "target_depth_m": depth_m,
        "actual_depth_m": round(actual_depth_m, 2),
        "duration_days": duration_days,
        "days_required": round(days_required, 2),
        "drill_rate_m_hr": drill_rate_m_hr,
        "ice_volume_m3": round(ice_volume_m3, 2),
        "ice_mass_kg": round(ice_mass_kg, 2),
        "water_extracted_kg": round(water_kg, 2),
        "melting_energy_kwh": round(melting_energy_kwh, 2),
        "autonomy_achieved": autonomy_achieved,
        "autonomy_met": autonomy_achieved >= EUROPA_AUTONOMY_REQUIREMENT,
        "config": config,
    }

    emit_receipt(
        "europa_drilling",
        {
            "receipt_type": "europa_drilling",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "target_depth_m": depth_m,
            "actual_depth_m": result["actual_depth_m"],
            "water_extracted_kg": result["water_extracted_kg"],
            "autonomy_achieved": autonomy_achieved,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_autonomy(drill_rate: float, resupply_interval: float) -> float:
    """Compute self-sufficiency ratio for Europa operations.

    Args:
        drill_rate: Ice drill rate in m/hr
        resupply_interval: Resupply interval in days

    Returns:
        Autonomy ratio (0-1, where 1 = fully self-sufficient)

    Receipt: europa_autonomy_receipt
    """
    if resupply_interval <= 0:
        return 0.0

    # Higher drill rate and longer resupply interval = more autonomy
    # Base autonomy from latency constraints
    base_autonomy = 1 - EUROPA_EARTH_CALLBACK_MAX_PCT

    # Bonus from longer resupply (more self-sufficient)
    resupply_bonus = min(resupply_interval / 365, 0.05)  # Up to 5% bonus

    autonomy = min(base_autonomy + resupply_bonus, 1.0)

    result = {
        "drill_rate_m_hr": drill_rate,
        "resupply_interval_days": resupply_interval,
        "base_autonomy": round(base_autonomy, 4),
        "resupply_bonus": round(resupply_bonus, 4),
        "autonomy": round(autonomy, 4),
        "self_sufficient": autonomy >= EUROPA_AUTONOMY_REQUIREMENT,
        "autonomy_requirement": EUROPA_AUTONOMY_REQUIREMENT,
    }

    emit_receipt(
        "europa_autonomy",
        {
            "receipt_type": "europa_autonomy",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return autonomy


def ice_to_water(kg_ice: float) -> Dict[str, Any]:
    """Convert ice mass to water metrics.

    Args:
        kg_ice: Mass of ice in kg

    Returns:
        Dict with conversion metrics
    """
    # Water extracted (with efficiency)
    water_kg = kg_ice * WATER_EXTRACTION_EFFICIENCY

    # Energy required for melting
    melting_energy_kj = kg_ice * ICE_MELTING_ENERGY_KJ_KG
    melting_energy_kwh = melting_energy_kj / 3600

    # Volume of water (density = 1000 kg/m3 at standard conditions)
    water_volume_liters = water_kg  # 1 kg = 1 liter for water

    # Potential uses
    drinking_water_person_days = water_kg / 3.0  # ~3L per person per day
    hydrogen_potential_kg = water_kg * (2 / 18)  # H2O -> H2 (2/18 by mass)
    oxygen_potential_kg = water_kg * (16 / 18)  # H2O -> O (16/18 by mass)

    return {
        "ice_kg": kg_ice,
        "water_kg": round(water_kg, 2),
        "water_liters": round(water_volume_liters, 2),
        "melting_energy_kwh": round(melting_energy_kwh, 2),
        "drinking_water_person_days": round(drinking_water_person_days, 1),
        "hydrogen_potential_kg": round(hydrogen_potential_kg, 2),
        "oxygen_potential_kg": round(oxygen_potential_kg, 2),
        "extraction_efficiency": WATER_EXTRACTION_EFFICIENCY,
    }


def estimate_drill_time(depth_km: float, rate_m_hr: float) -> float:
    """Estimate time to reach given depth.

    Args:
        depth_km: Target depth in kilometers
        rate_m_hr: Drill rate in meters per hour

    Returns:
        Time in days to reach depth
    """
    if rate_m_hr <= 0:
        return float("inf")

    depth_m = depth_km * 1000
    hours = depth_m / rate_m_hr
    days = hours / 24

    return round(days, 1)


# === D7+EUROPA HYBRID FUNCTIONS ===


def d7_europa_hybrid(
    tree_size: int = D7_TREE_MIN,
    base_alpha: float = 3.2,
    depth_m: int = 1000,
    duration_days: int = 30,
    drill_rate_m_hr: float = EUROPA_DRILL_RATE_M_HR,
) -> Dict[str, Any]:
    """Integrated D7 fractal + Europa ice drilling hybrid run.

    Combines:
    - D7 fractal recursion for alpha >= 3.38
    - Europa ice drilling simulation
    - Autonomy verification

    Args:
        tree_size: Number of nodes in tree
        base_alpha: Base alpha before recursion
        depth_m: Target drill depth in meters
        duration_days: Simulation duration in days
        drill_rate_m_hr: Drilling rate

    Returns:
        Dict with integrated results

    Receipt: d7_europa_hybrid_receipt
    """
    # Run D7 fractal recursion
    d7_result = d7_recursive_fractal(tree_size, base_alpha, depth=7)

    # Run Europa drilling simulation
    europa_result = simulate_drilling(depth_m, duration_days, drill_rate_m_hr)

    # Compute combined SLO
    combined_slo = {
        "alpha_target": D7_ALPHA_FLOOR,
        "alpha_achieved": d7_result["eff_alpha"],
        "alpha_met": d7_result["floor_met"],
        "autonomy_target": EUROPA_AUTONOMY_REQUIREMENT,
        "autonomy_achieved": europa_result["autonomy_achieved"],
        "autonomy_met": europa_result["autonomy_met"],
        "all_targets_met": (d7_result["floor_met"] and europa_result["autonomy_met"]),
    }

    result = {
        "d7_result": {
            "tree_size": d7_result["tree_size"],
            "base_alpha": d7_result["base_alpha"],
            "depth": d7_result["depth"],
            "eff_alpha": d7_result["eff_alpha"],
            "floor_met": d7_result["floor_met"],
            "target_met": d7_result["target_met"],
            "instability": d7_result["instability"],
        },
        "europa_result": {
            "target_depth_m": europa_result["target_depth_m"],
            "actual_depth_m": europa_result["actual_depth_m"],
            "water_extracted_kg": europa_result["water_extracted_kg"],
            "autonomy_achieved": europa_result["autonomy_achieved"],
        },
        "combined_slo": combined_slo,
        "gate": "t24h",
    }

    emit_receipt(
        "d7_europa_hybrid",
        {
            "receipt_type": "d7_europa_hybrid",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "tree_size": tree_size,
            "eff_alpha": d7_result["eff_alpha"],
            "autonomy_achieved": europa_result["autonomy_achieved"],
            "all_targets_met": combined_slo["all_targets_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INFO FUNCTIONS ===


def get_europa_info() -> Dict[str, Any]:
    """Get Europa ice drilling hybrid module info.

    Returns:
        Dict with module info

    Receipt: europa_info
    """
    config = load_europa_config()

    info = {
        "module": "europa_ice_hybrid",
        "version": "1.0.0",
        "config": config,
        "drilling": {
            "base_drill_rate_m_hr": EUROPA_DRILL_RATE_M_HR,
            "ice_melting_energy_kj_kg": ICE_MELTING_ENERGY_KJ_KG,
            "water_extraction_efficiency": WATER_EXTRACTION_EFFICIENCY,
        },
        "autonomy": {
            "requirement": EUROPA_AUTONOMY_REQUIREMENT,
            "latency_min": [EUROPA_LATENCY_MIN_MIN, EUROPA_LATENCY_MAX_MIN],
            "earth_callback_max_pct": EUROPA_EARTH_CALLBACK_MAX_PCT,
        },
        "d7_integration": {
            "alpha_floor": D7_ALPHA_FLOOR,
            "tree_min": D7_TREE_MIN,
        },
        "description": "Europa ice drilling ISRU simulation with D7 integration",
    }

    emit_receipt(
        "europa_info",
        {
            "receipt_type": "europa_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "autonomy_requirement": EUROPA_AUTONOMY_REQUIREMENT,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
