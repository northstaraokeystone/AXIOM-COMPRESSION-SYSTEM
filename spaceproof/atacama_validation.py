"""atacama_validation.py - Mars Dust/Perovskite Validation Using Atacama Analog Data

PARADIGM:
    The Atacama Desert provides the best Earth analog for Mars dust conditions.
    92% dust similarity enables perovskite efficiency calibration.

ATACAMA FACTS:
    - Solar flux: 1000 W/m^2 (ground level)
    - Mars solar flux: 590 W/m^2 (surface)
    - Dust similarity to Mars: 92%
    - Perovskite lab efficiency: 25.6% (NREL 2024)

CALIBRATION:
    Mars projection = Atacama efficiency * (Mars flux / Atacama flux) * dust_correction

Source: SpaceProof D8 Atacama validation - Mars dust calibration
"""

import json
from datetime import datetime
from typing import Any, Dict

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

ATACAMA_TENANT_ID = "spaceproof-atacama"
"""Tenant ID for Atacama receipts."""

# Atacama config defaults
ATACAMA_DUST_ANALOG_MATCH = 0.92
"""92% Mars dust similarity."""

ATACAMA_SOLAR_FLUX_W_M2 = 1000
"""Atacama ground-level solar flux in W/m^2."""

MARS_SOLAR_FLUX_W_M2 = 590
"""Mars surface solar flux in W/m^2."""

ATACAMA_PEROVSKITE_EFFICIENCY = 0.256
"""NREL 2024 lab perovskite efficiency (25.6%)."""


# === CONFIG LOADING ===


def load_atacama_config() -> Dict[str, Any]:
    """Load Atacama configuration from d8_multi_spec.json.

    Returns:
        Dict with Atacama configuration

    Receipt: atacama_config_receipt
    """
    import os

    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d8_multi_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("atacama_config", {})

    emit_receipt(
        "atacama_config",
        {
            "receipt_type": "atacama_config",
            "tenant_id": ATACAMA_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": config.get("source", "Atacama Desert Mars Analog Studies"),
            "dust_similarity": config.get("dust_similarity", ATACAMA_DUST_ANALOG_MATCH),
            "perovskite_efficiency": config.get(
                "perovskite_efficiency", ATACAMA_PEROVSKITE_EFFICIENCY
            ),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


# === DUST CORRECTION ===


def compute_dust_correction(atacama_flux: float, mars_flux: float) -> float:
    """Compute flux ratio correction factor.

    Args:
        atacama_flux: Atacama solar flux in W/m^2
        mars_flux: Mars solar flux in W/m^2

    Returns:
        Flux correction factor

    Receipt: dust_correction_receipt
    """
    if atacama_flux <= 0:
        return 0.0

    correction = mars_flux / atacama_flux

    emit_receipt(
        "dust_correction",
        {
            "receipt_type": "dust_correction",
            "tenant_id": ATACAMA_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "atacama_flux_w_m2": atacama_flux,
            "mars_flux_w_m2": mars_flux,
            "correction_factor": round(correction, 4),
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "atacama_flux": atacama_flux,
                        "mars_flux": mars_flux,
                        "correction": correction,
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return round(correction, 4)


def validate_perovskite_dust(efficiency: float, dust_load: float) -> Dict[str, Any]:
    """Validate perovskite efficiency under dust load.

    Args:
        efficiency: Base perovskite efficiency (0-1)
        dust_load: Dust load factor (0-1, where 0 = no dust, 1 = fully obscured)

    Returns:
        Dict with validation results
    """
    # Dust reduces efficiency linearly (simplified model)
    effective_efficiency = efficiency * (1 - dust_load * 0.5)  # 50% max reduction

    # Dust similarity threshold check
    valid_analog = dust_load <= (
        1 - ATACAMA_DUST_ANALOG_MATCH
    )  # 8% max additional dust

    return {
        "base_efficiency": efficiency,
        "dust_load": dust_load,
        "effective_efficiency": round(effective_efficiency, 4),
        "efficiency_reduction_pct": round(
            (1 - effective_efficiency / efficiency) * 100, 2
        ),
        "valid_analog": valid_analog,
        "dust_similarity_used": ATACAMA_DUST_ANALOG_MATCH,
    }


# === MARS PROJECTION ===


def project_mars_efficiency(atacama_eff: float, dust_correction: float) -> float:
    """Project Mars efficiency from Atacama data.

    Args:
        atacama_eff: Atacama-measured efficiency
        dust_correction: Flux correction factor

    Returns:
        Projected Mars efficiency
    """
    # Apply flux correction and dust analog adjustment
    mars_eff = atacama_eff * dust_correction * ATACAMA_DUST_ANALOG_MATCH

    return round(mars_eff, 4)


def compare_atacama_mars(
    atacama_data: Dict[str, Any], mars_estimate: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare Atacama measurements with Mars estimates.

    Args:
        atacama_data: Atacama measurement data
        mars_estimate: Mars estimate data

    Returns:
        Dict with comparison results
    """
    atacama_eff = atacama_data.get("efficiency", ATACAMA_PEROVSKITE_EFFICIENCY)
    mars_eff = mars_estimate.get("efficiency", 0.0)

    # Compute difference
    diff = abs(atacama_eff - mars_eff)
    ratio = mars_eff / atacama_eff if atacama_eff > 0 else 0.0

    return {
        "atacama_efficiency": atacama_eff,
        "mars_efficiency": mars_eff,
        "efficiency_diff": round(diff, 4),
        "mars_to_atacama_ratio": round(ratio, 4),
        "flux_ratio": round(MARS_SOLAR_FLUX_W_M2 / ATACAMA_SOLAR_FLUX_W_M2, 4),
        "dust_similarity": ATACAMA_DUST_ANALOG_MATCH,
        "comparison_valid": ratio >= 0.5,  # Mars should be at least 50% of Atacama
    }


# === VALIDATION RUNNER ===


def run_atacama_validation(simulate: bool = True) -> Dict[str, Any]:
    """Run Atacama validation workflow.

    Args:
        simulate: Whether to run in simulation mode

    Returns:
        Dict with validation results

    Receipt: atacama_validation_receipt
    """
    # Load config
    config = load_atacama_config()

    # Compute dust correction
    dust_correction = compute_dust_correction(
        config.get("solar_flux_w_m2", ATACAMA_SOLAR_FLUX_W_M2),
        config.get("mars_solar_flux_w_m2", MARS_SOLAR_FLUX_W_M2),
    )

    # Project Mars efficiency
    atacama_eff = config.get("perovskite_efficiency", ATACAMA_PEROVSKITE_EFFICIENCY)
    mars_eff = project_mars_efficiency(atacama_eff, dust_correction)

    # Validate dust load
    dust_validation = validate_perovskite_dust(atacama_eff, 0.05)  # 5% dust load

    # Compare
    comparison = compare_atacama_mars(
        {"efficiency": atacama_eff}, {"efficiency": mars_eff}
    )

    result = {
        "mode": "simulate" if simulate else "execute",
        "config": config,
        "dust_correction": dust_correction,
        "atacama_efficiency": atacama_eff,
        "mars_projected_efficiency": mars_eff,
        "dust_validation": dust_validation,
        "comparison": comparison,
        "dust_similarity": config.get("dust_similarity", ATACAMA_DUST_ANALOG_MATCH),
        "validation_passed": (
            config.get("dust_similarity", 0) >= ATACAMA_DUST_ANALOG_MATCH
            and comparison["comparison_valid"]
        ),
    }

    emit_receipt(
        "atacama_validation",
        {
            "receipt_type": "atacama_validation",
            "tenant_id": ATACAMA_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "dust_similarity": config.get("dust_similarity", ATACAMA_DUST_ANALOG_MATCH),
            "atacama_efficiency": atacama_eff,
            "mars_projected_efficiency": mars_eff,
            "validation_passed": result["validation_passed"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_atacama_info() -> Dict[str, Any]:
    """Get Atacama validation configuration info.

    Returns:
        Dict with Atacama info
    """
    config = load_atacama_config()

    info = {
        "source": config.get("source", "Atacama Desert Mars Analog Studies"),
        "dust_similarity": config.get("dust_similarity", ATACAMA_DUST_ANALOG_MATCH),
        "solar_flux_w_m2": config.get("solar_flux_w_m2", ATACAMA_SOLAR_FLUX_W_M2),
        "mars_solar_flux_w_m2": config.get(
            "mars_solar_flux_w_m2", MARS_SOLAR_FLUX_W_M2
        ),
        "perovskite_efficiency": config.get(
            "perovskite_efficiency", ATACAMA_PEROVSKITE_EFFICIENCY
        ),
        "flux_ratio": round(MARS_SOLAR_FLUX_W_M2 / ATACAMA_SOLAR_FLUX_W_M2, 4),
        "constants": {
            "atacama_dust_analog_match": ATACAMA_DUST_ANALOG_MATCH,
            "atacama_solar_flux_w_m2": ATACAMA_SOLAR_FLUX_W_M2,
            "mars_solar_flux_w_m2": MARS_SOLAR_FLUX_W_M2,
            "atacama_perovskite_efficiency": ATACAMA_PEROVSKITE_EFFICIENCY,
        },
    }

    return info
