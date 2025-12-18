"""callisto_ice.py - Callisto Ice Operations and Low-Radiation Advantage

PARADIGM:
    Callisto is the outermost Galilean moon with lowest radiation (0.01 level).
    200 km ice/rock mix depth provides abundant water extraction.
    Ideal hub location for Jovian system coordination.

THE PHYSICS:
    Callisto advantages:
    - Outside Jupiter's main radiation belts
    - Radiation level: 0.01 (vs 5.4 for Europa, 0.08 for Ganymede)
    - 200 km ice/rock depth for water extraction
    - 16.69-day orbital period for predictable scheduling
    - 98% autonomy requirement (outermost = most autonomous)

    Hub suitability:
    - Low radiation = longer equipment lifespan
    - Stable orbit = predictable communications
    - Water ice = fuel production capability

Source: Grok - "Callisto focus" + "Hub suitability analysis"
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

CALLISTO_TENANT_ID = "axiom-callisto"
"""Tenant ID for Callisto receipts."""

CALLISTO_ICE_DEPTH_KM = 200
"""Ice/rock mix depth in km."""

CALLISTO_ORBITAL_PERIOD_DAYS = 16.69
"""Orbital period in days."""

CALLISTO_AUTONOMY_REQUIREMENT = 0.98
"""Required autonomy level (98%, outermost moon)."""

CALLISTO_LATENCY_MIN = [33, 53]
"""Latency bounds [min, max] in minutes."""

CALLISTO_RADIATION_LEVEL = 0.01
"""Radiation level (very low, outside magnetosphere)."""

CALLISTO_SURFACE_TEMP_K = 134
"""Surface temperature in Kelvin."""

CALLISTO_DIAMETER_KM = 4821
"""Diameter in km."""

# Extraction parameters
ICE_DENSITY_KG_M3 = 917.0
"""Ice density in kg/m3."""

EXTRACTION_ENERGY_KWH_PER_KG = 0.1
"""Energy required per kg of ice extraction."""


# === CONFIG LOADING ===


def load_callisto_config() -> Dict[str, Any]:
    """Load Callisto configuration from d10_jovian_spec.json.

    Returns:
        Dict with Callisto configuration

    Receipt: callisto_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d10_jovian_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    callisto_config = spec.get("callisto_config", {})

    result = {
        "body": callisto_config.get("body", "callisto"),
        "resource": callisto_config.get("resource", "water_ice"),
        "ice_depth_km": callisto_config.get("ice_depth_km", CALLISTO_ICE_DEPTH_KM),
        "orbital_period_days": callisto_config.get(
            "orbital_period_days", CALLISTO_ORBITAL_PERIOD_DAYS
        ),
        "autonomy_requirement": callisto_config.get(
            "autonomy_requirement", CALLISTO_AUTONOMY_REQUIREMENT
        ),
        "latency_min": callisto_config.get("latency_min", CALLISTO_LATENCY_MIN),
        "radiation_level": callisto_config.get(
            "radiation_level", CALLISTO_RADIATION_LEVEL
        ),
        "advantage": callisto_config.get("advantage", "outside_magnetosphere"),
        "surface_temp_k": callisto_config.get(
            "surface_temp_k", CALLISTO_SURFACE_TEMP_K
        ),
        "diameter_km": callisto_config.get("diameter_km", CALLISTO_DIAMETER_KM),
        "earth_callback_max_pct": callisto_config.get("earth_callback_max_pct", 0.02),
    }

    emit_receipt(
        "callisto_config",
        {
            "receipt_type": "callisto_config",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "ice_depth_km": result["ice_depth_km"],
            "radiation_level": result["radiation_level"],
            "autonomy_requirement": result["autonomy_requirement"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === ICE OPERATIONS ===


def compute_ice_availability(depth_km: float = CALLISTO_ICE_DEPTH_KM) -> Dict[str, Any]:
    """Compute ice availability at given depth.

    Args:
        depth_km: Depth in km (default: 200)

    Returns:
        Dict with ice availability estimates

    Receipt: callisto_ice_availability
    """
    # Assume 50% ice content in ice/rock mix
    ice_fraction = 0.50

    # Compute volume (simplified cylinder model)
    # Assume 1 km^2 extraction area
    extraction_area_km2 = 1.0
    volume_km3 = extraction_area_km2 * depth_km * ice_fraction

    # Convert to kg
    volume_m3 = volume_km3 * 1e9  # km3 to m3
    ice_mass_kg = volume_m3 * ICE_DENSITY_KG_M3

    result = {
        "depth_km": depth_km,
        "ice_fraction": ice_fraction,
        "extraction_area_km2": extraction_area_km2,
        "volume_km3": round(volume_km3, 4),
        "ice_mass_kg": ice_mass_kg,
        "ice_mass_tons": round(ice_mass_kg / 1000, 2),
        "sustainability_years": 1000,  # Effectively unlimited
    }

    emit_receipt(
        "callisto_ice_availability",
        {
            "receipt_type": "callisto_ice_availability",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "depth_km": depth_km,
            "ice_mass_tons": result["ice_mass_tons"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_extraction(
    rate_kg_hr: float = 100.0, duration_days: int = 30
) -> Dict[str, Any]:
    """Simulate ice extraction operation.

    Args:
        rate_kg_hr: Extraction rate in kg/hour
        duration_days: Simulation duration in days

    Returns:
        Dict with extraction simulation results

    Receipt: callisto_extraction
    """
    total_hours = duration_days * 24
    total_extracted_kg = rate_kg_hr * total_hours

    # Energy required
    energy_kwh = total_extracted_kg * EXTRACTION_ENERGY_KWH_PER_KG

    # Compute autonomy based on minimal Earth callback
    config = load_callisto_config()
    autonomy_achieved = 1.0 - config["earth_callback_max_pct"]

    result = {
        "rate_kg_hr": rate_kg_hr,
        "duration_days": duration_days,
        "total_hours": total_hours,
        "total_extracted_kg": round(total_extracted_kg, 2),
        "energy_kwh": round(energy_kwh, 2),
        "autonomy_achieved": autonomy_achieved,
        "autonomy_met": autonomy_achieved >= CALLISTO_AUTONOMY_REQUIREMENT,
        "efficiency": 0.95,  # Simulated extraction efficiency
    }

    emit_receipt(
        "callisto_extraction",
        {
            "receipt_type": "callisto_extraction",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "duration_days": duration_days,
            "total_extracted_kg": result["total_extracted_kg"],
            "autonomy_achieved": autonomy_achieved,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === RADIATION ADVANTAGE ===


def compute_radiation_advantage() -> Dict[str, Any]:
    """Compute Callisto's radiation advantage vs other Jovian moons.

    Returns:
        Dict with radiation comparison

    Receipt: callisto_radiation_receipt
    """
    # Radiation levels for Jovian moons (relative scale)
    radiation_levels = {
        "callisto": 0.01,  # Outside magnetosphere
        "ganymede": 0.08,  # Partial magnetosphere protection
        "europa": 5.4,  # High radiation (within belts)
        "io": 36.0,  # Extreme radiation
    }

    # Equipment lifespan multipliers (inverse of radiation)
    lifespan_multipliers = {
        moon: round(1.0 / rad if rad > 0 else 100, 2)
        for moon, rad in radiation_levels.items()
    }

    result = {
        "callisto_radiation": CALLISTO_RADIATION_LEVEL,
        "radiation_comparison": radiation_levels,
        "lifespan_multipliers": lifespan_multipliers,
        "callisto_advantage": "100x longer equipment lifespan vs Europa",
        "outside_magnetosphere": True,
        "hub_suitable": True,
    }

    emit_receipt(
        "callisto_radiation",
        {
            "receipt_type": "callisto_radiation",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "callisto_radiation": CALLISTO_RADIATION_LEVEL,
            "hub_suitable": True,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === AUTONOMY COMPUTATION ===


def compute_autonomy(extraction_results: Dict[str, Any]) -> float:
    """Compute Callisto autonomy based on extraction results.

    Args:
        extraction_results: Results from simulate_extraction

    Returns:
        Autonomy level (0-1)

    Receipt: callisto_autonomy_receipt
    """
    # Base autonomy from extraction efficiency
    efficiency = extraction_results.get("efficiency", 0.95)
    autonomy_achieved = extraction_results.get("autonomy_achieved", 0.98)

    # Combined autonomy (weighted)
    autonomy = (efficiency * 0.3) + (autonomy_achieved * 0.7)

    result = {
        "extraction_efficiency": efficiency,
        "base_autonomy": autonomy_achieved,
        "computed_autonomy": round(autonomy, 4),
        "requirement": CALLISTO_AUTONOMY_REQUIREMENT,
        "requirement_met": autonomy >= CALLISTO_AUTONOMY_REQUIREMENT,
    }

    emit_receipt(
        "callisto_autonomy",
        {
            "receipt_type": "callisto_autonomy",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "computed_autonomy": result["computed_autonomy"],
            "requirement_met": result["requirement_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return round(autonomy, 4)


# === HUB SUITABILITY ===


def evaluate_hub_suitability() -> Dict[str, Any]:
    """Evaluate Callisto's suitability as Jovian hub location.

    Returns:
        Dict with hub suitability analysis

    Receipt: callisto_hub_suitability
    """
    load_callisto_config()
    compute_radiation_advantage()
    compute_ice_availability()

    # Scoring factors (0-10 scale)
    scores = {
        "radiation": 10,  # Lowest in Jovian system
        "ice_availability": 9,  # Abundant
        "orbital_stability": 9,  # Regular 16.69-day orbit
        "communication": 8,  # Predictable windows
        "equipment_lifespan": 10,  # 100x vs Europa
    }

    overall_score = sum(scores.values()) / len(scores)

    result = {
        "scores": scores,
        "overall_score": round(overall_score, 2),
        "suitable": overall_score >= 8.0,
        "advantages": [
            "Lowest radiation (0.01 level)",
            "200 km ice depth",
            "Outside magnetosphere",
            "98% autonomy achievable",
            "100x equipment lifespan vs Europa",
        ],
        "disadvantages": [
            "Outermost moon (longest transfer times)",
            "Lower gravity than Ganymede",
        ],
        "recommendation": "OPTIMAL hub location for Jovian system",
    }

    emit_receipt(
        "callisto_hub_suitability",
        {
            "receipt_type": "callisto_hub_suitability",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "overall_score": result["overall_score"],
            "suitable": result["suitable"],
            "recommendation": result["recommendation"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INFO ===


def get_callisto_info() -> Dict[str, Any]:
    """Get Callisto configuration and status.

    Returns:
        Dict with Callisto info

    Receipt: callisto_info
    """
    config = load_callisto_config()

    info = {
        "body": "Callisto",
        "type": "Galilean moon (outermost)",
        "resource": "water_ice",
        "ice_depth_km": config["ice_depth_km"],
        "radiation_level": config["radiation_level"],
        "autonomy_requirement": config["autonomy_requirement"],
        "orbital_period_days": config["orbital_period_days"],
        "surface_temp_k": config["surface_temp_k"],
        "diameter_km": config["diameter_km"],
        "advantage": "Outside Jupiter's magnetosphere",
        "hub_role": "Optimal location for Jovian system hub",
        "description": "Fourth Galilean moon with lowest radiation and abundant ice",
    }

    emit_receipt(
        "callisto_info",
        {
            "receipt_type": "callisto_info",
            "tenant_id": CALLISTO_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "ice_depth_km": info["ice_depth_km"],
            "radiation_level": info["radiation_level"],
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
