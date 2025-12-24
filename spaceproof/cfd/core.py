"""Core CFD configuration and validation functions.

PARADIGM:
    Core CFD utilities for configuration management and Reynolds number computation.

Functions:
    - load_cfd_config: Load CFD configuration from d11_venus_spec.json
    - get_cfd_info: Get CFD configuration summary
    - compute_reynolds_number: Compute Reynolds number for flow conditions
    - run_cfd_validation: Run full CFD dust dynamics validation
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from spaceproof.core import dual_hash, emit_receipt

from .constants import (
    CFD_DENSITY_MARS_KG_M3,
    CFD_GRAVITY_MARS_M_S2,
    CFD_PARTICLE_DENSITY_KG_M3,
    CFD_PARTICLE_SIZE_UM,
    CFD_REYNOLDS_NUMBER_MARS,
    CFD_SETTLING_MODEL,
    CFD_TENANT_ID,
    CFD_TURBULENCE_MODEL,
    CFD_VISCOSITY_MARS_PA_S,
)

# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA: Dict[str, Any] = {
    "cfd_config": {
        "tenant_id": "spaceproof-cfd",
        "receipt_type": "cfd_config",
        "description": "CFD configuration loaded",
    },
    "cfd_info": {
        "tenant_id": "spaceproof-cfd",
        "receipt_type": "cfd_info",
        "description": "CFD configuration summary",
    },
    "cfd_reynolds": {
        "tenant_id": "spaceproof-cfd",
        "receipt_type": "cfd_reynolds",
        "description": "Reynolds number computation",
    },
    "cfd_dust": {
        "tenant_id": "spaceproof-cfd",
        "receipt_type": "cfd_dust",
        "description": "Full CFD validation",
    },
}


# === CONFIGURATION FUNCTIONS ===


def load_cfd_config() -> Dict[str, Any]:
    """Load CFD configuration from d11_venus_spec.json.

    Returns:
        Dict with CFD configuration

    Receipt: cfd_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "d11_venus_spec.json",
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("cfd_config", {})

    emit_receipt(
        "cfd_config",
        {
            "receipt_type": "cfd_config",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "model": config.get("model", "navier_stokes"),
            "reynolds_number_mars": config.get(
                "reynolds_number_mars", CFD_REYNOLDS_NUMBER_MARS
            ),
            "settling_model": config.get("settling_model", CFD_SETTLING_MODEL),
            "validated": config.get("validated", True),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


def get_cfd_info() -> Dict[str, Any]:
    """Get CFD configuration summary.

    Returns:
        Dict with CFD info

    Receipt: cfd_info_receipt
    """
    config = load_cfd_config()

    info = {
        "model": "navier_stokes",
        "reynolds_number_mars": CFD_REYNOLDS_NUMBER_MARS,
        "gravity_mars_m_s2": CFD_GRAVITY_MARS_M_S2,
        "viscosity_mars_pa_s": CFD_VISCOSITY_MARS_PA_S,
        "density_mars_kg_m3": CFD_DENSITY_MARS_KG_M3,
        "particle_density_kg_m3": CFD_PARTICLE_DENSITY_KG_M3,
        "particle_size_um": CFD_PARTICLE_SIZE_UM,
        "settling_model": CFD_SETTLING_MODEL,
        "turbulence_model": CFD_TURBULENCE_MODEL,
        "config": config,
    }

    emit_receipt(
        "cfd_info",
        {
            "receipt_type": "cfd_info",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "reynolds_number_mars": CFD_REYNOLDS_NUMBER_MARS,
            "settling_model": CFD_SETTLING_MODEL,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === REYNOLDS NUMBER COMPUTATION ===


def compute_reynolds_number(
    velocity: float,
    length: float,
    viscosity: float = CFD_VISCOSITY_MARS_PA_S,
    density: float = CFD_DENSITY_MARS_KG_M3,
) -> float:
    """Compute Reynolds number for given flow conditions.

    Re = (rho * v * L) / mu

    Args:
        velocity: Flow velocity in m/s
        length: Characteristic length in m
        viscosity: Dynamic viscosity in Pa*s (default: Mars)
        density: Fluid density in kg/m^3 (default: Mars)

    Returns:
        Reynolds number

    Receipt: cfd_reynolds_receipt
    """
    if viscosity <= 0:
        return 0.0

    re = (density * velocity * length) / viscosity

    emit_receipt(
        "cfd_reynolds",
        {
            "receipt_type": "cfd_reynolds",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "velocity_m_s": velocity,
            "length_m": length,
            "reynolds_number": round(re, 2),
            "regime": "laminar" if re < 2300 else "turbulent",
            "payload_hash": dual_hash(json.dumps({"re": round(re, 2)}, sort_keys=True)),
        },
    )

    return re


# === FULL CFD VALIDATION ===


def run_cfd_validation() -> Dict[str, Any]:
    """Run full CFD dust dynamics validation.

    Returns:
        Dict with complete validation results

    Receipt: cfd_dust_receipt
    """
    # Import here to avoid circular dependencies
    from .atacama import validate_against_atacama, project_mars_dynamics
    from .laminar import simulate_particle_trajectory, stokes_settling

    # Get configuration
    config = load_cfd_config()

    # Compute Reynolds number
    re = compute_reynolds_number(velocity=1.0, length=0.001)  # 1mm particle at 1 m/s

    # Compute settling for reference particle
    settling = stokes_settling(10.0)

    # Run trajectory simulation
    trajectory = simulate_particle_trajectory(
        initial_pos=(0, 0, 100),
        wind=(5, 0, 0),
        duration_s=1000,
        particle_size_um=10.0,
    )

    # Validate against Atacama
    validation = validate_against_atacama({"settling_velocity_m_s": settling})

    # Mars projection
    mars_proj = project_mars_dynamics()

    result = {
        "config": config,
        "reynolds_number": re,
        "settling_velocity_10um_m_s": settling,
        "trajectory_result": trajectory,
        "atacama_validation": validation,
        "mars_projection": mars_proj,
        "overall_validated": validation["validation_passed"],
        "cfd_model": "navier_stokes",
    }

    emit_receipt(
        "cfd_dust",
        {
            "receipt_type": "cfd_dust",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "reynolds_number": re,
            "settling_10um": settling,
            "atacama_correlation": validation["correlation"],
            "validated": result["overall_validated"],
            "payload_hash": dual_hash(
                json.dumps({"validated": result["overall_validated"]}, sort_keys=True)
            ),
        },
    )

    return result


# === EXPORTS ===

__all__ = [
    "RECEIPT_SCHEMA",
    "load_cfd_config",
    "get_cfd_info",
    "compute_reynolds_number",
    "run_cfd_validation",
]
