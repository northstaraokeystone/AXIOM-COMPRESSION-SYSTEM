"""nrel_validation.py - NREL Perovskite Efficiency Validation

NREL PARAMETERS (2024):
    - Lab efficiency record: 25.6%
    - Degradation rate: ~2% annually
    - Target stability: 25 years
    - Scaling factor vs MOXIE: 3.33x

VALIDATION MODEL:
    - Compare measured efficiency to NREL lab data
    - Project degradation over time
    - Validate Mars ISRU perovskite targets

Source: AXIOM D7 recursion + Europa ice + NREL + expanded audits
"""

import json
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash
from .fractal_layers import get_d7_spec


# === CONSTANTS ===

TENANT_ID = "axiom-nrel"
"""Tenant ID for NREL receipts."""

# NREL lab data (2024)
NREL_LAB_EFFICIENCY = 0.256
"""NREL 2024 lab record efficiency (25.6%)."""

NREL_DEGRADATION_RATE_ANNUAL = 0.02
"""Annual degradation rate (2%)."""

NREL_TARGET_STABILITY_YEARS = 25
"""Target stability in years."""

# MOXIE baseline
MOXIE_EFFICIENCY = 0.06
"""MOXIE baseline efficiency (6%)."""

# Scaling factor
NREL_SCALING_FACTOR = 3.33
"""Perovskite scaling factor vs MOXIE (3.33x)."""

# Validation thresholds
EFFICIENCY_VALIDATION_TOLERANCE = 0.05
"""Tolerance for efficiency validation (5%)."""


# === CONFIG FUNCTIONS ===


def load_nrel_config() -> Dict[str, Any]:
    """Load NREL configuration from d7_europa_spec.json.

    Returns:
        Dict with NREL configuration

    Receipt: nrel_config_receipt
    """
    spec = get_d7_spec()
    nrel_config = spec.get("nrel_config", {})

    result = {
        "source": nrel_config.get("source", "NREL Perovskite Reports 2024"),
        "lab_efficiency": nrel_config.get("lab_efficiency", NREL_LAB_EFFICIENCY),
        "degradation_rate_annual": nrel_config.get(
            "degradation_rate_annual", NREL_DEGRADATION_RATE_ANNUAL
        ),
        "target_stability_years": nrel_config.get(
            "target_stability_years", NREL_TARGET_STABILITY_YEARS
        ),
        "scaling_factor": nrel_config.get("scaling_factor", NREL_SCALING_FACTOR),
        "moxie_baseline": nrel_config.get("moxie_baseline", MOXIE_EFFICIENCY),
    }

    emit_receipt(
        "nrel_config",
        {
            "receipt_type": "nrel_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === VALIDATION FUNCTIONS ===


def validate_efficiency(
    measured: float, tolerance: float = EFFICIENCY_VALIDATION_TOLERANCE
) -> Dict[str, Any]:
    """Validate measured efficiency against NREL lab data.

    Args:
        measured: Measured efficiency (0-1)
        tolerance: Tolerance for validation

    Returns:
        Dict with validation results

    Receipt: nrel_validation_receipt
    """
    config = load_nrel_config()
    lab_efficiency = config["lab_efficiency"]

    # Check if within tolerance of lab efficiency
    deviation = abs(measured - lab_efficiency)
    relative_deviation = deviation / lab_efficiency if lab_efficiency > 0 else 0
    within_tolerance = relative_deviation <= tolerance

    # Determine validation status
    if measured >= lab_efficiency:
        status = "exceeds_lab"
    elif within_tolerance:
        status = "within_tolerance"
    else:
        status = "below_tolerance"

    result = {
        "measured_efficiency": measured,
        "lab_efficiency": lab_efficiency,
        "deviation": round(deviation, 4),
        "relative_deviation": round(relative_deviation, 4),
        "tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "validation_status": status,
        "validated": within_tolerance or measured >= lab_efficiency,
    }

    emit_receipt(
        "nrel_validation",
        {
            "receipt_type": "nrel_validation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def project_degradation(years: int, initial_eff: float = None) -> Dict[str, Any]:
    """Project efficiency degradation over time.

    Uses exponential decay: eff(t) = eff_0 * (1 - rate)^t

    Args:
        years: Number of years to project
        initial_eff: Initial efficiency (default: NREL lab)

    Returns:
        Dict with degradation projection

    Receipt: nrel_projection_receipt
    """
    config = load_nrel_config()

    if initial_eff is None:
        initial_eff = config["lab_efficiency"]

    degradation_rate = config["degradation_rate_annual"]
    projections = []

    for year in range(years + 1):
        # Exponential decay
        eff = initial_eff * ((1 - degradation_rate) ** year)
        projections.append(
            {
                "year": year,
                "efficiency": round(eff, 4),
                "efficiency_pct": round(eff * 100, 2),
                "retained_pct": round((eff / initial_eff) * 100, 2),
            }
        )

    # Find end-of-life year (when efficiency drops below 80% of initial)
    eol_threshold = initial_eff * 0.80
    eol_year = None
    for p in projections:
        if p["efficiency"] < eol_threshold:
            eol_year = p["year"]
            break

    result = {
        "initial_efficiency": initial_eff,
        "degradation_rate_annual": degradation_rate,
        "projection_years": years,
        "projections": projections,
        "final_efficiency": projections[-1]["efficiency"],
        "final_retained_pct": projections[-1]["retained_pct"],
        "eol_threshold": eol_threshold,
        "eol_year": eol_year,
        "meets_stability_target": (
            eol_year is None or eol_year >= config["target_stability_years"]
        ),
        "target_stability_years": config["target_stability_years"],
    }

    emit_receipt(
        "nrel_projection",
        {
            "receipt_type": "nrel_projection",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "projection_years": years,
            "initial_efficiency": initial_eff,
            "final_efficiency": result["final_efficiency"],
            "meets_stability_target": result["meets_stability_target"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_lifetime_output(
    eff: float,
    area_m2: float,
    years: int,
    solar_irradiance_w_m2: float = 590,  # Mars average
) -> Dict[str, Any]:
    """Compute total energy output over lifetime.

    Args:
        eff: Initial efficiency (0-1)
        area_m2: Panel area in square meters
        years: Operating lifetime in years
        solar_irradiance_w_m2: Solar irradiance (default: Mars average)

    Returns:
        Dict with lifetime output metrics
    """
    config = load_nrel_config()
    degradation_rate = config["degradation_rate_annual"]

    # Calculate energy output per year with degradation
    yearly_outputs_kwh = []
    total_kwh = 0

    for year in range(years):
        # Efficiency at start of year
        year_eff = eff * ((1 - degradation_rate) ** year)

        # Power output (W) = irradiance * area * efficiency
        power_w = solar_irradiance_w_m2 * area_m2 * year_eff

        # Energy per year (kWh) = power * hours per year
        # Assume 50% solar availability (day/night, dust, etc.)
        hours_per_year = 365 * 24 * 0.5
        yearly_kwh = (power_w / 1000) * hours_per_year

        yearly_outputs_kwh.append(
            {
                "year": year + 1,
                "efficiency": round(year_eff, 4),
                "power_w": round(power_w, 2),
                "output_kwh": round(yearly_kwh, 2),
            }
        )
        total_kwh += yearly_kwh

    return {
        "initial_efficiency": eff,
        "area_m2": area_m2,
        "years": years,
        "solar_irradiance_w_m2": solar_irradiance_w_m2,
        "degradation_rate": degradation_rate,
        "yearly_outputs": yearly_outputs_kwh,
        "total_output_kwh": round(total_kwh, 2),
        "total_output_mwh": round(total_kwh / 1000, 2),
        "avg_yearly_kwh": round(total_kwh / years, 2) if years > 0 else 0,
    }


def compare_to_moxie(nrel_eff: float, moxie_eff: float = None) -> Dict[str, Any]:
    """Compare perovskite efficiency to MOXIE baseline.

    Args:
        nrel_eff: NREL perovskite efficiency
        moxie_eff: MOXIE baseline efficiency (default from config)

    Returns:
        Dict with comparison metrics
    """
    config = load_nrel_config()

    if moxie_eff is None:
        moxie_eff = config["moxie_baseline"]

    # Compute scaling factor
    scaling_factor = nrel_eff / moxie_eff if moxie_eff > 0 else 0

    # Check against expected scaling
    expected_scaling = config["scaling_factor"]
    scaling_achieved = scaling_factor >= expected_scaling * 0.95  # 5% tolerance

    return {
        "nrel_efficiency": nrel_eff,
        "moxie_efficiency": moxie_eff,
        "scaling_factor": round(scaling_factor, 2),
        "expected_scaling": expected_scaling,
        "scaling_achieved": scaling_achieved,
        "improvement_pct": round((scaling_factor - 1) * 100, 1),
        "description": f"Perovskite provides {scaling_factor:.2f}x efficiency vs MOXIE",
    }


# === INFO FUNCTIONS ===


def get_nrel_info() -> Dict[str, Any]:
    """Get NREL validation module info.

    Returns:
        Dict with module info

    Receipt: nrel_info
    """
    config = load_nrel_config()

    info = {
        "module": "nrel_validation",
        "version": "1.0.0",
        "stub_mode": True,  # Lab data reference only
        "config": config,
        "constants": {
            "lab_efficiency": NREL_LAB_EFFICIENCY,
            "degradation_rate_annual": NREL_DEGRADATION_RATE_ANNUAL,
            "target_stability_years": NREL_TARGET_STABILITY_YEARS,
            "scaling_factor": NREL_SCALING_FACTOR,
            "moxie_baseline": MOXIE_EFFICIENCY,
        },
        "validation": {
            "tolerance": EFFICIENCY_VALIDATION_TOLERANCE,
            "source": "NREL Perovskite Reports 2024",
        },
        "description": "NREL perovskite efficiency validation (lab data reference)",
    }

    emit_receipt(
        "nrel_info",
        {
            "receipt_type": "nrel_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "lab_efficiency": NREL_LAB_EFFICIENCY,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
