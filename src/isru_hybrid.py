"""isru_hybrid.py - MOXIE-calibrated ISRU simulation with D5 fractal integration.

PARADIGM:
    In-Situ Resource Utilization (ISRU) simulation calibrated to NASA MOXIE data.
    Integrated with D5 fractal recursion for alpha >= 3.25 target.

NASA MOXIE CALIBRATION (Dec 2025):
    - O2 total: 122g (16 runs)
    - O2 peak: 12 g/hr
    - O2 avg: 5.5 g/hr
    - Atmospheric CO2: 95.3%
    - Conversion efficiency: 6%

ISRU CLOSURE TARGET:
    - 85% self-sufficiency from local resources
    - 15% uplift from Earth (diminishing over time)
    - Resources: O2, H2O, CH4 (Sabatier process)

D5+ISRU HYBRID:
    - D5 recursion provides alpha uplift (+0.168)
    - ISRU calibration anchors to real-world data
    - Hybrid combines both for validated autonomy metrics

Source: NASA Perseverance MOXIE, Grok D5+ISRU hybrid spec
"""

import json
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash
from .fractal_layers import (
    get_d5_spec,
    d5_recursive_fractal,
    D5_ALPHA_TARGET,
    D5_UPLIFT,
    D5_TREE_MIN,
)


# === CONSTANTS ===

TENANT_ID = "axiom-isru"
"""Tenant ID for ISRU receipts."""

# MOXIE Calibration Constants (NASA Perseverance Dec 2025)
MOXIE_O2_TOTAL_G = 122
"""Total O2 produced by MOXIE (grams)."""

MOXIE_O2_PEAK_G_HR = 12
"""Peak O2 production rate (g/hr)."""

MOXIE_O2_AVG_G_HR = 5.5
"""Average O2 production rate (g/hr)."""

MOXIE_RUNS = 16
"""Number of MOXIE runs."""

MOXIE_CO2_PCT = 95.3
"""Mars atmospheric CO2 percentage."""

MOXIE_EFFICIENCY = 0.06
"""MOXIE conversion efficiency (6%)."""

# ISRU Configuration
ISRU_CLOSURE_TARGET = 0.85
"""Target closure ratio (85% self-sufficient)."""

ISRU_RESOURCES = ["o2", "h2o", "ch4"]
"""Resources tracked by ISRU."""

SABATIER_EFFICIENCY = 0.85
"""Sabatier reaction efficiency."""

ELECTROLYSIS_EFFICIENCY = 0.80
"""Water electrolysis efficiency."""

ICE_EXTRACTION_RATE_KG_HR = 0.5
"""Ice extraction rate (kg/hr)."""

# Crew consumption rates
O2_CONSUMPTION_KG_DAY = 0.84
"""O2 consumption per crew member per day (kg)."""

H2O_CONSUMPTION_KG_DAY = 3.0
"""Water consumption per crew member per day (kg)."""


# === MOXIE CALIBRATION FUNCTIONS ===


def load_moxie_calibration() -> Dict[str, Any]:
    """Load MOXIE calibration data from d5_isru_spec.json.

    Returns:
        Dict with MOXIE calibration parameters

    Receipt: moxie_calibration_load
    """
    spec = get_d5_spec()
    moxie = spec.get("moxie_calibration", {})

    calibration = {
        "source": moxie.get("source", "NASA Perseverance MOXIE"),
        "date": moxie.get("date", "2025-12"),
        "o2_total_g": moxie.get("o2_total_g", MOXIE_O2_TOTAL_G),
        "o2_peak_g_hr": moxie.get("o2_peak_g_hr", MOXIE_O2_PEAK_G_HR),
        "o2_avg_g_hr": moxie.get("o2_avg_g_hr", MOXIE_O2_AVG_G_HR),
        "runs": moxie.get("runs", MOXIE_RUNS),
        "atmospheric_co2_pct": moxie.get("atmospheric_co2_pct", MOXIE_CO2_PCT),
        "conversion_efficiency": moxie.get("conversion_efficiency", MOXIE_EFFICIENCY),
    }

    emit_receipt(
        "moxie_calibration_load",
        {
            "receipt_type": "moxie_calibration_load",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **calibration,
            "payload_hash": dual_hash(json.dumps(calibration, sort_keys=True)),
        },
    )

    return calibration


def moxie_calibration() -> Dict[str, Any]:
    """Return MOXIE calibration data with receipt.

    Returns:
        Dict with calibration data and validation

    Receipt: moxie_calibration_receipt
    """
    calibration = load_moxie_calibration()

    # Validate against expected values
    valid = (
        calibration["o2_total_g"] == MOXIE_O2_TOTAL_G
        and calibration["o2_peak_g_hr"] == MOXIE_O2_PEAK_G_HR
    )

    result = {
        **calibration,
        "validated": valid,
        "expected_total": MOXIE_O2_TOTAL_G,
        "expected_peak": MOXIE_O2_PEAK_G_HR,
    }

    emit_receipt(
        "moxie_calibration",
        {
            "receipt_type": "moxie_calibration",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "o2_total_g": result["o2_total_g"],
            "validated": valid,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === O2 PRODUCTION SIMULATION ===


def simulate_o2_production(
    hours: int = 24, crew: int = 4, moxie_units: int = 10
) -> Dict[str, Any]:
    """Simulate O2 production based on MOXIE calibration.

    Args:
        hours: Simulation duration in hours
        crew: Number of crew members
        moxie_units: Number of MOXIE units deployed

    Returns:
        Dict with production simulation results

    Receipt: isru_production_receipt
    """
    calibration = load_moxie_calibration()

    # Production calculation based on MOXIE average rate
    avg_rate = calibration["o2_avg_g_hr"]
    peak_rate = calibration["o2_peak_g_hr"]

    # Scale by number of units (linear scaling assumption)
    production_g = avg_rate * hours * moxie_units
    peak_production_g = peak_rate * hours * moxie_units

    # Convert to kg
    production_kg = production_g / 1000
    peak_production_kg = peak_production_g / 1000

    # Crew consumption
    consumption_kg = crew * O2_CONSUMPTION_KG_DAY * (hours / 24)

    # Net balance
    balance_kg = production_kg - consumption_kg
    self_sufficient = balance_kg >= 0

    # Production rate per crew member
    rate_per_crew = production_kg / crew if crew > 0 else 0

    result = {
        "hours": hours,
        "crew": crew,
        "moxie_units": moxie_units,
        "production_g": round(production_g, 2),
        "production_kg": round(production_kg, 4),
        "peak_production_kg": round(peak_production_kg, 4),
        "consumption_kg": round(consumption_kg, 4),
        "balance_kg": round(balance_kg, 4),
        "self_sufficient": self_sufficient,
        "rate_per_crew_kg_day": round(rate_per_crew * 24 / hours, 4),
        "efficiency": calibration["conversion_efficiency"],
    }

    emit_receipt(
        "isru_production",
        {
            "receipt_type": "isru_production",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "production_kg": result["production_kg"],
            "consumption_kg": result["consumption_kg"],
            "self_sufficient": self_sufficient,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === ISRU CLOSURE COMPUTATION ===


def compute_isru_closure(
    production: Dict[str, float], consumption: Dict[str, float]
) -> float:
    """Compute ISRU closure ratio.

    Closure = total_local_production / total_consumption

    Args:
        production: Dict with resource production values
        consumption: Dict with resource consumption values

    Returns:
        Closure ratio (0.0 to 1.0)

    Receipt: isru_closure_receipt
    """
    total_production = sum(production.values())
    total_consumption = sum(consumption.values())

    if total_consumption == 0:
        closure = 0.0
    else:
        closure = min(1.0, total_production / total_consumption)

    gap = ISRU_CLOSURE_TARGET - closure
    target_met = closure >= ISRU_CLOSURE_TARGET

    result = {
        "closure_ratio": round(closure, 4),
        "target": ISRU_CLOSURE_TARGET,
        "gap": round(gap, 4),
        "target_met": target_met,
        "uplift_required": round(1.0 - closure, 4),
        "production": production,
        "consumption": consumption,
    }

    emit_receipt(
        "isru_closure",
        {
            "receipt_type": "isru_closure",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "closure_ratio": result["closure_ratio"],
            "target_met": target_met,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return closure


def validate_closure(closure: float, target: float = ISRU_CLOSURE_TARGET) -> bool:
    """Validate closure ratio against target.

    Args:
        closure: Current closure ratio
        target: Target closure ratio (default: 0.85)

    Returns:
        True if closure >= target
    """
    return closure >= target


# === D5+ISRU HYBRID FUNCTIONS ===


def d5_isru_hybrid(
    tree_size: int = D5_TREE_MIN,
    base_alpha: float = 3.0,
    crew: int = 4,
    hours: int = 24,
    moxie_units: int = 10,
) -> Dict[str, Any]:
    """Integrated D5 fractal + ISRU simulation.

    Combines:
    - D5 recursion for alpha uplift (+0.168)
    - MOXIE-calibrated ISRU for autonomy metrics

    Args:
        tree_size: Tree size for D5 recursion
        base_alpha: Base alpha before D5 uplift
        crew: Number of crew members
        hours: Simulation duration in hours
        moxie_units: Number of MOXIE units

    Returns:
        Dict with integrated results

    Receipt: d5_isru_hybrid_receipt
    """
    # Run D5 recursion
    d5_result = d5_recursive_fractal(tree_size, base_alpha, depth=5)

    # Run ISRU simulation
    production_result = simulate_o2_production(hours, crew, moxie_units)

    # Compute ISRU closure for O2
    production = {"o2": production_result["production_kg"]}
    consumption = {"o2": production_result["consumption_kg"]}
    closure = compute_isru_closure(production, consumption)

    # Combined result
    result = {
        "mode": "d5_isru_hybrid",
        "tree_size": tree_size,
        "base_alpha": base_alpha,
        "d5_result": {
            "eff_alpha": d5_result["eff_alpha"],
            "uplift": d5_result["adjusted_uplift"],
            "target_met": d5_result["target_met"],
            "floor_met": d5_result["floor_met"],
        },
        "isru_result": {
            "production_kg": production_result["production_kg"],
            "consumption_kg": production_result["consumption_kg"],
            "balance_kg": production_result["balance_kg"],
            "self_sufficient": production_result["self_sufficient"],
        },
        "closure": {
            "ratio": round(closure, 4),
            "target": ISRU_CLOSURE_TARGET,
            "target_met": closure >= ISRU_CLOSURE_TARGET,
        },
        "combined_slo": {
            "alpha_target": D5_ALPHA_TARGET,
            "alpha_met": d5_result["eff_alpha"] >= D5_ALPHA_TARGET,
            "closure_target": ISRU_CLOSURE_TARGET,
            "closure_met": closure >= ISRU_CLOSURE_TARGET,
            "all_targets_met": (
                d5_result["eff_alpha"] >= D5_ALPHA_TARGET
                and closure >= ISRU_CLOSURE_TARGET
            ),
        },
        "crew": crew,
        "hours": hours,
        "moxie_units": moxie_units,
    }

    emit_receipt(
        "d5_isru_hybrid",
        {
            "receipt_type": "d5_isru_hybrid",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "eff_alpha": d5_result["eff_alpha"],
            "closure_ratio": round(closure, 4),
            "alpha_target_met": d5_result["target_met"],
            "closure_target_met": closure >= ISRU_CLOSURE_TARGET,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True, default=str)),
        },
    )

    return result


# === AUTONOMY METRICS ===


def compute_o2_autonomy(production_rate_kg_hr: float, crew: int) -> float:
    """Compute O2 self-sufficiency ratio.

    Args:
        production_rate_kg_hr: O2 production rate (kg/hr)
        crew: Number of crew members

    Returns:
        Autonomy ratio (production / consumption)

    Receipt: o2_autonomy_receipt
    """
    # Consumption rate per crew (convert daily to hourly)
    consumption_rate_kg_hr = crew * O2_CONSUMPTION_KG_DAY / 24

    if consumption_rate_kg_hr == 0:
        autonomy = 0.0
    else:
        autonomy = production_rate_kg_hr / consumption_rate_kg_hr

    # Clamp to reasonable range
    autonomy = min(10.0, max(0.0, autonomy))

    result = {
        "production_rate_kg_hr": round(production_rate_kg_hr, 6),
        "consumption_rate_kg_hr": round(consumption_rate_kg_hr, 6),
        "autonomy_ratio": round(autonomy, 4),
        "crew": crew,
        "self_sufficient": autonomy >= 1.0,
    }

    emit_receipt(
        "o2_autonomy",
        {
            "receipt_type": "o2_autonomy",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "autonomy_ratio": result["autonomy_ratio"],
            "self_sufficient": result["self_sufficient"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return autonomy


# === INFO FUNCTIONS ===


def get_isru_info() -> Dict[str, Any]:
    """Get ISRU hybrid module information.

    Returns:
        Dict with module configuration

    Receipt: isru_info
    """
    get_d5_spec()

    info = {
        "moxie_calibration": {
            "o2_total_g": MOXIE_O2_TOTAL_G,
            "o2_peak_g_hr": MOXIE_O2_PEAK_G_HR,
            "o2_avg_g_hr": MOXIE_O2_AVG_G_HR,
            "runs": MOXIE_RUNS,
            "efficiency": MOXIE_EFFICIENCY,
        },
        "isru_config": {
            "closure_target": ISRU_CLOSURE_TARGET,
            "resources": ISRU_RESOURCES,
            "sabatier_efficiency": SABATIER_EFFICIENCY,
            "electrolysis_efficiency": ELECTROLYSIS_EFFICIENCY,
        },
        "d5_integration": {
            "alpha_target": D5_ALPHA_TARGET,
            "uplift": D5_UPLIFT,
            "tree_min": D5_TREE_MIN,
        },
        "consumption_rates": {
            "o2_kg_day": O2_CONSUMPTION_KG_DAY,
            "h2o_kg_day": H2O_CONSUMPTION_KG_DAY,
        },
        "description": "MOXIE-calibrated ISRU simulation with D5 fractal integration",
    }

    emit_receipt(
        "isru_info",
        {
            "receipt_type": "isru_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "closure_target": ISRU_CLOSURE_TARGET,
            "alpha_target": D5_ALPHA_TARGET,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
