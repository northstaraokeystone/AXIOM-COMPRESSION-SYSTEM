"""titan_methane_hybrid.py - Titan Methane Harvesting ISRU Simulation

TITAN PARAMETERS:
    - Surface temperature: 94K (-179C)
    - Surface pressure: 1.45 atm
    - Atmosphere: 95% N2, 5% CH4
    - Methane lakes: liquid hydrocarbon (Kraken Mare, Ligeia Mare)
    - Methane density: 1.5 kg/m3 (liquid)

AUTONOMY REQUIREMENT:
    - 99% autonomy required (no Earth callback at 70-90 min latency)
    - Earth support max: 1%
    - All critical decisions must be made locally

HARVESTING MODEL:
    - Methane extraction from lakes
    - Conversion to fuel (CH4 + 2O2 -> CO2 + 2H2O + energy)
    - Energy density: 55.5 MJ/kg

Source: AXIOM D6 recursion + Titan methane + adversarial audits
"""

import json
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash
from .fractal_layers import (
    get_d6_spec,
    d6_recursive_fractal,
    D6_ALPHA_FLOOR,
    D6_TREE_MIN,
)


# === CONSTANTS ===

TENANT_ID = "axiom-titan"
"""Tenant ID for Titan receipts."""

# Titan physical parameters
TITAN_SURFACE_TEMP_K = 94
"""Titan surface temperature in Kelvin."""

TITAN_SURFACE_PRESSURE_ATM = 1.45
"""Titan surface pressure in atmospheres."""

TITAN_METHANE_DENSITY_KG_M3 = 1.5
"""Liquid methane density in kg/m3."""

TITAN_METHANE_ENERGY_MJ_KG = 55.5
"""Methane energy density in MJ/kg."""

# Autonomy parameters
TITAN_AUTONOMY_REQUIREMENT = 0.99
"""Required autonomy level (99%)."""

TITAN_LATENCY_MIN_MIN = 70
"""Minimum one-way latency to Earth in minutes."""

TITAN_LATENCY_MAX_MIN = 90
"""Maximum one-way latency to Earth in minutes."""

TITAN_EARTH_CALLBACK_MAX_PCT = 0.01
"""Maximum Earth callback allowed (1%)."""

# Harvesting parameters
METHANE_EXTRACTION_RATE_KG_HR = 10.0
"""Base methane extraction rate in kg/hr."""

METHANE_PROCESSING_EFFICIENCY = 0.85
"""Processing efficiency for methane conversion."""


# === CONFIG FUNCTIONS ===


def load_titan_config() -> Dict[str, Any]:
    """Load Titan configuration from d6_titan_spec.json.

    Returns:
        Dict with Titan configuration

    Receipt: titan_config_receipt
    """
    spec = get_d6_spec()
    titan_config = spec.get("titan_config", {})

    result = {
        "body": titan_config.get("body", "titan"),
        "resource": titan_config.get("resource", "methane"),
        "methane_density_kg_m3": titan_config.get(
            "methane_density_kg_m3", TITAN_METHANE_DENSITY_KG_M3
        ),
        "surface_temp_k": titan_config.get("surface_temp_k", TITAN_SURFACE_TEMP_K),
        "surface_pressure_atm": titan_config.get(
            "surface_pressure_atm", TITAN_SURFACE_PRESSURE_ATM
        ),
        "autonomy_requirement": titan_config.get(
            "autonomy_requirement", TITAN_AUTONOMY_REQUIREMENT
        ),
        "latency_min": titan_config.get(
            "latency_min", [TITAN_LATENCY_MIN_MIN, TITAN_LATENCY_MAX_MIN]
        ),
        "earth_callback_max_pct": titan_config.get(
            "earth_callback_max_pct", TITAN_EARTH_CALLBACK_MAX_PCT
        ),
        "methane_energy_mj_kg": titan_config.get(
            "methane_energy_mj_kg", TITAN_METHANE_ENERGY_MJ_KG
        ),
    }

    emit_receipt("titan_config", {
        "receipt_type": "titan_config",
        "tenant_id": TENANT_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


# === SIMULATION FUNCTIONS ===


def simulate_harvest(
    duration_days: int = 30,
    extraction_rate_kg_hr: float = METHANE_EXTRACTION_RATE_KG_HR,
    efficiency: float = METHANE_PROCESSING_EFFICIENCY
) -> Dict[str, Any]:
    """Simulate methane harvesting on Titan.

    Args:
        duration_days: Simulation duration in days
        extraction_rate_kg_hr: Methane extraction rate in kg/hr
        efficiency: Processing efficiency (0-1)

    Returns:
        Dict with harvest simulation results

    Receipt: titan_harvest_receipt
    """
    config = load_titan_config()

    # Compute raw extraction
    hours = duration_days * 24
    raw_extraction_kg = extraction_rate_kg_hr * hours

    # Apply efficiency
    processed_kg = raw_extraction_kg * efficiency

    # Compute energy potential
    energy_mj = processed_kg * config["methane_energy_mj_kg"]
    energy_kwh = energy_mj / 3.6  # Convert MJ to kWh

    # Compute autonomy metrics
    earth_queries_possible = (duration_days * 24 * 60) / (
        config["latency_min"][0] * 2
    )  # Round-trip
    earth_queries_budget = earth_queries_possible * config["earth_callback_max_pct"]
    local_decisions = earth_queries_possible - earth_queries_budget

    result = {
        "duration_days": duration_days,
        "extraction_rate_kg_hr": extraction_rate_kg_hr,
        "efficiency": efficiency,
        "raw_extraction_kg": round(raw_extraction_kg, 2),
        "processed_kg": round(processed_kg, 2),
        "energy_mj": round(energy_mj, 2),
        "energy_kwh": round(energy_kwh, 2),
        "earth_queries_budget": int(earth_queries_budget),
        "local_decisions": int(local_decisions),
        "autonomy_achieved": round(
            local_decisions / earth_queries_possible, 4
        ) if earth_queries_possible > 0 else 1.0,
        "config": config,
    }

    emit_receipt("titan_harvest", {
        "receipt_type": "titan_harvest",
        "tenant_id": TENANT_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        "duration_days": duration_days,
        "processed_kg": result["processed_kg"],
        "energy_kwh": result["energy_kwh"],
        "autonomy_achieved": result["autonomy_achieved"],
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def compute_autonomy(
    harvest_rate: float,
    consumption_rate: float
) -> float:
    """Compute self-sufficiency ratio for Titan operations.

    Args:
        harvest_rate: Methane harvest rate in kg/hr
        consumption_rate: Fuel consumption rate in kg/hr

    Returns:
        Autonomy ratio (0-1, where 1 = fully self-sufficient)

    Receipt: titan_autonomy_receipt
    """
    if consumption_rate <= 0:
        return 0.0

    autonomy = min(harvest_rate / consumption_rate, 1.0)

    result = {
        "harvest_rate_kg_hr": harvest_rate,
        "consumption_rate_kg_hr": consumption_rate,
        "autonomy": round(autonomy, 4),
        "self_sufficient": autonomy >= TITAN_AUTONOMY_REQUIREMENT,
        "autonomy_requirement": TITAN_AUTONOMY_REQUIREMENT,
    }

    emit_receipt("titan_autonomy", {
        "receipt_type": "titan_autonomy",
        "tenant_id": TENANT_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return autonomy


def methane_to_fuel(kg_methane: float) -> Dict[str, Any]:
    """Convert methane mass to fuel metrics.

    CH4 + 2O2 -> CO2 + 2H2O + energy (890 kJ/mol)

    Args:
        kg_methane: Mass of methane in kg

    Returns:
        Dict with conversion metrics
    """
    # Methane molecular weight: 16 g/mol
    moles = (kg_methane * 1000) / 16

    # Energy: 890 kJ/mol
    energy_kj = moles * 890
    energy_mj = energy_kj / 1000
    energy_kwh = energy_mj / 3.6

    # O2 required: 2 mol O2 per mol CH4
    # O2 molecular weight: 32 g/mol
    o2_moles = moles * 2
    o2_kg = (o2_moles * 32) / 1000

    # CO2 produced: 1 mol CO2 per mol CH4
    # CO2 molecular weight: 44 g/mol
    co2_moles = moles
    co2_kg = (co2_moles * 44) / 1000

    # H2O produced: 2 mol H2O per mol CH4
    # H2O molecular weight: 18 g/mol
    h2o_moles = moles * 2
    h2o_kg = (h2o_moles * 18) / 1000

    return {
        "methane_kg": kg_methane,
        "energy_mj": round(energy_mj, 2),
        "energy_kwh": round(energy_kwh, 2),
        "o2_required_kg": round(o2_kg, 2),
        "co2_produced_kg": round(co2_kg, 2),
        "h2o_produced_kg": round(h2o_kg, 2),
    }


# === D6+TITAN HYBRID FUNCTIONS ===


def d6_titan_hybrid(
    tree_size: int = D6_TREE_MIN,
    base_alpha: float = 3.15,
    duration_days: int = 30,
    extraction_rate_kg_hr: float = METHANE_EXTRACTION_RATE_KG_HR
) -> Dict[str, Any]:
    """Integrated D6 fractal + Titan methane hybrid run.

    Combines:
    - D6 fractal recursion for alpha >= 3.31
    - Titan methane harvesting simulation
    - Autonomy verification

    Args:
        tree_size: Number of nodes in tree
        base_alpha: Base alpha before recursion
        duration_days: Simulation duration in days
        extraction_rate_kg_hr: Methane extraction rate

    Returns:
        Dict with integrated results

    Receipt: d6_titan_hybrid_receipt
    """
    # Run D6 fractal recursion
    d6_result = d6_recursive_fractal(tree_size, base_alpha, depth=6)

    # Run Titan harvest simulation
    titan_result = simulate_harvest(duration_days, extraction_rate_kg_hr)

    # Compute combined SLO
    combined_slo = {
        "alpha_target": D6_ALPHA_FLOOR,
        "alpha_achieved": d6_result["eff_alpha"],
        "alpha_met": d6_result["floor_met"],
        "autonomy_target": TITAN_AUTONOMY_REQUIREMENT,
        "autonomy_achieved": titan_result["autonomy_achieved"],
        "autonomy_met": titan_result["autonomy_achieved"] >= TITAN_AUTONOMY_REQUIREMENT,
        "all_targets_met": (
            d6_result["floor_met"] and
            titan_result["autonomy_achieved"] >= TITAN_AUTONOMY_REQUIREMENT
        ),
    }

    result = {
        "d6_result": {
            "tree_size": d6_result["tree_size"],
            "base_alpha": d6_result["base_alpha"],
            "depth": d6_result["depth"],
            "eff_alpha": d6_result["eff_alpha"],
            "floor_met": d6_result["floor_met"],
            "target_met": d6_result["target_met"],
            "instability": d6_result["instability"],
        },
        "titan_result": {
            "duration_days": titan_result["duration_days"],
            "processed_kg": titan_result["processed_kg"],
            "energy_kwh": titan_result["energy_kwh"],
            "autonomy_achieved": titan_result["autonomy_achieved"],
        },
        "combined_slo": combined_slo,
        "gate": "t24h",
    }

    emit_receipt("d6_titan_hybrid", {
        "receipt_type": "d6_titan_hybrid",
        "tenant_id": TENANT_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tree_size": tree_size,
        "eff_alpha": d6_result["eff_alpha"],
        "autonomy_achieved": titan_result["autonomy_achieved"],
        "all_targets_met": combined_slo["all_targets_met"],
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


# === INFO FUNCTIONS ===


def get_titan_info() -> Dict[str, Any]:
    """Get Titan methane hybrid module info.

    Returns:
        Dict with module info

    Receipt: titan_info
    """
    config = load_titan_config()

    info = {
        "module": "titan_methane_hybrid",
        "version": "1.0.0",
        "config": config,
        "harvesting": {
            "base_extraction_rate_kg_hr": METHANE_EXTRACTION_RATE_KG_HR,
            "processing_efficiency": METHANE_PROCESSING_EFFICIENCY,
            "energy_density_mj_kg": TITAN_METHANE_ENERGY_MJ_KG,
        },
        "autonomy": {
            "requirement": TITAN_AUTONOMY_REQUIREMENT,
            "latency_min": [TITAN_LATENCY_MIN_MIN, TITAN_LATENCY_MAX_MIN],
            "earth_callback_max_pct": TITAN_EARTH_CALLBACK_MAX_PCT,
        },
        "d6_integration": {
            "alpha_floor": D6_ALPHA_FLOOR,
            "tree_min": D6_TREE_MIN,
        },
        "description": "Titan methane harvesting ISRU simulation with D6 integration",
    }

    emit_receipt("titan_info", {
        "receipt_type": "titan_info",
        "tenant_id": TENANT_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        "version": info["version"],
        "autonomy_requirement": TITAN_AUTONOMY_REQUIREMENT,
        "payload_hash": dual_hash(json.dumps(info, sort_keys=True))
    })

    return info
