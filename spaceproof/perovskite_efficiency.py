"""perovskite_efficiency.py - Efficiency Scaling from MOXIE 6% to Perovskite 20%

EFFICIENCY SCALING:
    - MOXIE baseline: 6% efficiency (validated Perseverance data)
    - Perovskite target: 20% efficiency (3.33x improvement)
    - Timeline: 10 years projected

PHYSICS NOTE:
    This is a STUB module for efficiency projection.
    Actual physics validation is separate.

Source: SpaceProof D6 recursion + Titan methane + adversarial audits
"""

import json
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash
from .fractal_layers import get_d6_spec


# === CONSTANTS ===

TENANT_ID = "spaceproof-perovskite"
"""Tenant ID for perovskite receipts."""

MOXIE_EFFICIENCY = 0.06
"""Current MOXIE efficiency (6%)."""

PEROVSKITE_SOLAR_EFF_TARGET = 0.20
"""Target perovskite solar efficiency (20%)."""

EFFICIENCY_SCALING_FACTOR = 3.33
"""Scaling factor: 20% / 6% = 3.33x."""

DEFAULT_TIMELINE_YEARS = 10
"""Default timeline for efficiency improvement in years."""


# === CONFIG FUNCTIONS ===


def load_efficiency_config() -> Dict[str, Any]:
    """Load efficiency configuration from d6_titan_spec.json.

    Returns:
        Dict with efficiency configuration

    Receipt: perovskite_config_receipt
    """
    spec = get_d6_spec()
    eff_config = spec.get("efficiency_config", {})

    result = {
        "moxie_baseline": eff_config.get("moxie_baseline", MOXIE_EFFICIENCY),
        "perovskite_target": eff_config.get(
            "perovskite_target", PEROVSKITE_SOLAR_EFF_TARGET
        ),
        "scaling_factor": eff_config.get("scaling_factor", EFFICIENCY_SCALING_FACTOR),
        "timeline_years": eff_config.get("timeline_years", DEFAULT_TIMELINE_YEARS),
    }

    emit_receipt(
        "perovskite_config",
        {
            "receipt_type": "perovskite_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === SCALING FUNCTIONS ===


def compute_scaling_factor(current: float, target: float) -> float:
    """Compute scaling factor from current to target efficiency.

    Args:
        current: Current efficiency (0-1)
        target: Target efficiency (0-1)

    Returns:
        Scaling factor (target / current)
    """
    if current <= 0:
        return 0.0
    return target / current


def project_efficiency(years: int, growth_rate: float = 0.10) -> Dict[str, Any]:
    """Project efficiency improvement over time.

    Uses compound growth model: efficiency = baseline * (1 + rate)^years

    Args:
        years: Number of years to project
        growth_rate: Annual efficiency growth rate (default: 10%)

    Returns:
        Dict with projected efficiency timeline

    Receipt: efficiency_scaling_receipt
    """
    config = load_efficiency_config()
    baseline = config["moxie_baseline"]
    target = config["perovskite_target"]

    projections = []

    for year in range(years + 1):
        efficiency = baseline * ((1 + growth_rate) ** year)
        efficiency = min(efficiency, target)  # Cap at target

        projections.append(
            {
                "year": year,
                "efficiency": round(efficiency, 4),
                "efficiency_pct": round(efficiency * 100, 2),
                "scaling_vs_baseline": round(efficiency / baseline, 2),
                "target_reached": efficiency >= target,
            }
        )

        if efficiency >= target:
            break

    # Find year when target is reached
    target_year = None
    for p in projections:
        if p["target_reached"]:
            target_year = p["year"]
            break

    result = {
        "baseline_efficiency": baseline,
        "target_efficiency": target,
        "growth_rate": growth_rate,
        "years_projected": years,
        "projections": projections,
        "target_year": target_year,
        "target_achievable": target_year is not None,
    }

    emit_receipt(
        "efficiency_scaling",
        {
            "receipt_type": "efficiency_scaling",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "baseline_efficiency": baseline,
            "target_efficiency": target,
            "growth_rate": growth_rate,
            "target_year": target_year,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def validate_perovskite_target() -> bool:
    """Validate that 20% efficiency target is achievable.

    STUB: Returns True based on research projections.
    Actual physics validation is separate.

    Returns:
        True if target is achievable, False otherwise
    """
    config = load_efficiency_config()

    # Check that scaling factor is reasonable (< 5x)
    scaling = config["scaling_factor"]
    achievable = scaling <= 5.0 and config["perovskite_target"] <= 0.30

    return achievable


# === INFO FUNCTIONS ===


def get_perovskite_info() -> Dict[str, Any]:
    """Get perovskite efficiency module info.

    Returns:
        Dict with module info

    Receipt: perovskite_info
    """
    config = load_efficiency_config()

    info = {
        "module": "perovskite_efficiency",
        "version": "1.0.0",
        "stub_mode": True,
        "config": config,
        "constants": {
            "moxie_efficiency": MOXIE_EFFICIENCY,
            "perovskite_target": PEROVSKITE_SOLAR_EFF_TARGET,
            "scaling_factor": EFFICIENCY_SCALING_FACTOR,
        },
        "validation": {
            "target_achievable": validate_perovskite_target(),
        },
        "description": "Efficiency scaling from MOXIE 6% to perovskite 20% (stub)",
    }

    emit_receipt(
        "perovskite_info",
        {
            "receipt_type": "perovskite_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "stub_mode": info["stub_mode"],
            "target_achievable": info["validation"]["target_achievable"],
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
