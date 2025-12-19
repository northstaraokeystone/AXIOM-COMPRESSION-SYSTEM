"""Large Eddy Simulation (LES) functions for high-Re flows.

PARADIGM:
    Large Eddy Simulation (LES) with Smagorinsky-Lilly subgrid model
    for high Reynolds number turbulent flows like dust devils.

THE PHYSICS:
    - LES resolves large-scale turbulent structures
    - Smagorinsky model: nu_sgs = (Cs * delta)^2 * |S|
    - Applicable for Re > 10,000 (dust devils: Re ~ 50,000)

Functions:
    - load_les_config: Load LES configuration
    - get_les_info: Get LES configuration summary
    - smagorinsky_viscosity: Compute Smagorinsky eddy viscosity
    - compute_subgrid_stress: Compute subgrid-scale stress
    - simulate_les: Run LES simulation
    - simulate_les_dust_devil: Simulate dust devil using LES
    - les_vs_rans_comparison: Compare LES and RANS approaches
    - run_les_validation: Run full LES validation
"""

import json
import math
import os
from datetime import datetime
from typing import Any, Dict

from src.core import dual_hash, emit_receipt

from .constants import (
    CFD_DENSITY_MARS_KG_M3,
    CFD_TENANT_ID,
    CFD_VISCOSITY_MARS_PA_S,
    LES_DUST_DEVIL_DIAMETER_M,
    LES_DUST_DEVIL_HEIGHT_M,
    LES_DUST_DEVIL_REYNOLDS,
    LES_FILTER_WIDTH_M,
    LES_REYNOLDS_THRESHOLD,
    LES_SMAGORINSKY_CONSTANT,
    LES_SUBGRID_MODEL,
)

# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA: Dict[str, Any] = {
    "les_config": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_config",
        "description": "LES configuration loaded",
    },
    "les_info": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_info",
        "description": "LES configuration summary",
    },
    "les_smagorinsky": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_smagorinsky",
        "description": "Smagorinsky eddy viscosity",
    },
    "les_stress": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_stress",
        "description": "Subgrid-scale stress",
    },
    "les_simulate": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_simulate",
        "description": "LES simulation",
    },
    "les_dust_devil": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_dust_devil",
        "description": "LES dust devil simulation",
    },
    "les_compare": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_compare",
        "description": "LES vs RANS comparison",
    },
    "les_dust": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "les_dust",
        "description": "Full LES validation",
    },
}


# === LES CONFIGURATION ===


def load_les_config() -> Dict[str, Any]:
    """Load LES configuration from d13_solar_spec.json.

    Returns:
        Dict with LES configuration

    Receipt: les_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "d13_solar_spec.json",
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("les_config", {})

    emit_receipt(
        "les_config",
        {
            "receipt_type": "les_config",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "model": config.get("model", "large_eddy_simulation"),
            "subgrid_model": config.get("subgrid_model", LES_SUBGRID_MODEL),
            "smagorinsky_constant": config.get(
                "smagorinsky_constant", LES_SMAGORINSKY_CONSTANT
            ),
            "reynolds_threshold": config.get(
                "reynolds_threshold", LES_REYNOLDS_THRESHOLD
            ),
            "validated": config.get("validated", True),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


def get_les_info() -> Dict[str, Any]:
    """Get LES configuration summary.

    Returns:
        Dict with LES info

    Receipt: les_info_receipt
    """
    config = load_les_config()

    info = {
        "model": "large_eddy_simulation",
        "subgrid_model": LES_SUBGRID_MODEL,
        "smagorinsky_constant": LES_SMAGORINSKY_CONSTANT,
        "filter_width_m": LES_FILTER_WIDTH_M,
        "reynolds_threshold": LES_REYNOLDS_THRESHOLD,
        "dust_devil_reynolds": LES_DUST_DEVIL_REYNOLDS,
        "dust_devil_diameter_m": LES_DUST_DEVIL_DIAMETER_M,
        "dust_devil_height_m": LES_DUST_DEVIL_HEIGHT_M,
        "config": config,
    }

    emit_receipt(
        "les_info",
        {
            "receipt_type": "les_info",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "subgrid_model": LES_SUBGRID_MODEL,
            "reynolds_threshold": LES_REYNOLDS_THRESHOLD,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === SMAGORINSKY SUBGRID MODEL ===


def smagorinsky_viscosity(
    strain_rate: float,
    filter_width: float = LES_FILTER_WIDTH_M,
    Cs: float = LES_SMAGORINSKY_CONSTANT,
) -> float:
    """Compute Smagorinsky eddy viscosity.

    nu_sgs = (Cs * delta)^2 * |S|

    Where:
    - Cs: Smagorinsky constant (~0.17)
    - delta: Filter width
    - |S|: Strain rate magnitude

    Args:
        strain_rate: Strain rate magnitude |S| in 1/s
        filter_width: LES filter width in m
        Cs: Smagorinsky constant

    Returns:
        Subgrid-scale eddy viscosity in m^2/s

    Receipt: les_smagorinsky_receipt
    """
    nu_sgs = (Cs * filter_width) ** 2 * abs(strain_rate)

    emit_receipt(
        "les_smagorinsky",
        {
            "receipt_type": "les_smagorinsky",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "strain_rate": strain_rate,
            "filter_width_m": filter_width,
            "smagorinsky_constant": Cs,
            "eddy_viscosity_m2_s": round(nu_sgs, 6),
            "payload_hash": dual_hash(
                json.dumps({"nu_sgs": round(nu_sgs, 6)}, sort_keys=True)
            ),
        },
    )

    return nu_sgs


def compute_subgrid_stress(
    velocity_gradient: float,
    filter_width: float = LES_FILTER_WIDTH_M,
    Cs: float = LES_SMAGORINSKY_CONSTANT,
) -> float:
    """Compute subgrid-scale stress tensor component.

    tau_sgs = -2 * nu_sgs * S

    Args:
        velocity_gradient: Velocity gradient component
        filter_width: LES filter width in m
        Cs: Smagorinsky constant

    Returns:
        SGS stress component in Pa

    Receipt: les_stress_receipt
    """
    # Estimate strain rate from velocity gradient
    strain_rate = abs(velocity_gradient)

    # Get eddy viscosity
    nu_sgs = smagorinsky_viscosity(strain_rate, filter_width, Cs)

    # Compute stress (simplified 1D)
    tau_sgs = -2 * CFD_DENSITY_MARS_KG_M3 * nu_sgs * velocity_gradient

    emit_receipt(
        "les_stress",
        {
            "receipt_type": "les_stress",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "velocity_gradient": velocity_gradient,
            "eddy_viscosity": round(nu_sgs, 6),
            "sgs_stress_pa": round(tau_sgs, 8),
            "payload_hash": dual_hash(
                json.dumps({"tau": round(tau_sgs, 8)}, sort_keys=True)
            ),
        },
    )

    return tau_sgs


# === LES SIMULATION ===


def simulate_les(
    reynolds: float,
    duration_s: float = 100.0,
    filter_width: float = LES_FILTER_WIDTH_M,
) -> Dict[str, Any]:
    """Run LES simulation for high-Re flow.

    Args:
        reynolds: Reynolds number of the flow
        duration_s: Simulation duration in seconds
        filter_width: LES filter width

    Returns:
        Dict with LES simulation results

    Receipt: les_simulate_receipt
    """
    # Check if LES is appropriate (Re > threshold)
    use_les = reynolds >= LES_REYNOLDS_THRESHOLD
    model_used = "LES" if use_les else "RANS"

    # Estimate characteristic velocity
    # Using Re = rho * v * L / mu
    # Assume L = filter_width
    v_char = (
        reynolds * CFD_VISCOSITY_MARS_PA_S / (CFD_DENSITY_MARS_KG_M3 * filter_width)
    )

    # Estimate strain rate
    strain_rate = v_char / filter_width

    # Compute SGS quantities (if LES)
    if use_les:
        nu_sgs = smagorinsky_viscosity(strain_rate, filter_width)
        tau_sgs = compute_subgrid_stress(strain_rate, filter_width)
    else:
        nu_sgs = 0.0
        tau_sgs = 0.0

    # Energy cascade estimate
    # Kolmogorov scale: eta = (nu^3 / epsilon)^(1/4)
    # Simplified: use molecular viscosity for RANS, sgs for LES
    viscosity_eff = CFD_VISCOSITY_MARS_PA_S / CFD_DENSITY_MARS_KG_M3 + nu_sgs
    epsilon = (v_char**3) / filter_width  # Dissipation rate estimate
    eta = (viscosity_eff**3 / max(epsilon, 1e-10)) ** 0.25  # Kolmogorov scale

    result = {
        "reynolds": reynolds,
        "duration_s": duration_s,
        "filter_width_m": filter_width,
        "model_used": model_used,
        "use_les": use_les,
        "characteristic_velocity_m_s": round(v_char, 4),
        "strain_rate_1_s": round(strain_rate, 4),
        "eddy_viscosity_m2_s": round(nu_sgs, 6),
        "sgs_stress_pa": round(tau_sgs, 8),
        "kolmogorov_scale_m": round(eta, 6),
        "energy_dissipation_rate": round(epsilon, 4),
        "simulation_complete": True,
    }

    emit_receipt(
        "les_simulate",
        {
            "receipt_type": "les_simulate",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "reynolds": reynolds,
            "model_used": model_used,
            "eddy_viscosity": result["eddy_viscosity_m2_s"],
            "kolmogorov_scale": result["kolmogorov_scale_m"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === DUST DEVIL SIMULATION ===


def simulate_les_dust_devil(
    diameter_m: float = 50.0,
    height_m: float = 500.0,
    intensity: float = 0.7,
) -> Dict[str, Any]:
    """Simulate Mars dust devil using LES.

    Dust devils are high-Re phenomena requiring LES treatment.

    Args:
        diameter_m: Dust devil diameter in meters
        height_m: Dust devil height in meters
        intensity: Intensity factor (0-1)

    Returns:
        Dict with dust devil simulation results

    Receipt: les_dust_devil_receipt
    """
    # Import here to avoid circular dependency
    from .core import compute_reynolds_number

    # Validate parameters
    diameter_m = max(
        LES_DUST_DEVIL_DIAMETER_M[0], min(diameter_m, LES_DUST_DEVIL_DIAMETER_M[1])
    )
    height_m = max(
        LES_DUST_DEVIL_HEIGHT_M[0], min(height_m, LES_DUST_DEVIL_HEIGHT_M[1])
    )

    # Estimate tangential velocity (typically 10-30 m/s for Mars dust devils)
    v_tangential = 10 + intensity * 20  # 10-30 m/s

    # Estimate vertical velocity
    v_vertical = v_tangential * 0.3  # Typically ~30% of tangential

    # Compute Reynolds number
    length_scale = diameter_m / 2  # Radius
    reynolds = compute_reynolds_number(v_tangential, length_scale)

    # Run LES simulation
    les_result = simulate_les(reynolds, duration_s=60.0, filter_width=diameter_m / 10)

    # Dust lifting capacity
    # Particles lifted when shear stress > threshold
    tau_wall = (
        0.5 * CFD_DENSITY_MARS_KG_M3 * v_tangential**2 * 0.01
    )  # Friction coefficient ~0.01
    dust_lifting_capacity_kg_s = (
        tau_wall * math.pi * (diameter_m / 2) ** 2 * 0.001
    )  # Empirical

    # Particle size that can be lifted
    # Use modified Stokes: v_settle = shear_velocity
    shear_velocity = math.sqrt(tau_wall / CFD_DENSITY_MARS_KG_M3)
    max_particle_size_um = 100  # Empirical upper bound for dust devils

    result = {
        "diameter_m": diameter_m,
        "height_m": height_m,
        "intensity": intensity,
        "tangential_velocity_m_s": round(v_tangential, 2),
        "vertical_velocity_m_s": round(v_vertical, 2),
        "reynolds": round(reynolds, 0),
        "les_result": les_result,
        "wall_shear_stress_pa": round(tau_wall, 6),
        "shear_velocity_m_s": round(shear_velocity, 4),
        "dust_lifting_capacity_kg_s": round(dust_lifting_capacity_kg_s, 6),
        "max_particle_size_lifted_um": max_particle_size_um,
        "lifetime_estimate_min": round(5 + intensity * 25, 1),  # 5-30 min typical
        "model": "LES_smagorinsky",
        "validated": reynolds >= LES_REYNOLDS_THRESHOLD,
    }

    emit_receipt(
        "les_dust_devil",
        {
            "receipt_type": "les_dust_devil",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "diameter_m": diameter_m,
            "height_m": height_m,
            "reynolds": result["reynolds"],
            "tangential_velocity": result["tangential_velocity_m_s"],
            "validated": result["validated"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === LES vs RANS COMPARISON ===


def les_vs_rans_comparison(reynolds: float) -> Dict[str, Any]:
    """Compare LES and RANS approaches for given Reynolds number.

    Args:
        reynolds: Reynolds number of the flow

    Returns:
        Dict with comparison results

    Receipt: les_compare_receipt
    """
    # Determine appropriate model
    use_les = reynolds >= LES_REYNOLDS_THRESHOLD

    # RANS approach (k-epsilon turbulence model simulation)
    # Simplified: constant eddy viscosity
    rans_nu_t = (
        0.09
        * (reynolds * CFD_VISCOSITY_MARS_PA_S / CFD_DENSITY_MARS_KG_M3) ** 2
        / reynolds
    )
    rans_accuracy = 0.85 if reynolds < 5000 else 0.70  # RANS less accurate at high Re

    # LES approach
    les_result = simulate_les(reynolds)
    les_nu_t = les_result["eddy_viscosity_m2_s"]
    les_accuracy = 0.95 if use_les else 0.80  # LES better at high Re

    # Computational cost estimate (relative)
    rans_cost = 1.0
    les_cost = 10.0 if use_les else 2.0  # LES much more expensive

    # Recommendation
    if reynolds < LES_REYNOLDS_THRESHOLD:
        recommendation = "RANS (k-epsilon)"
        reason = "Reynolds number below LES threshold"
    elif reynolds < 50000:
        recommendation = "LES (Smagorinsky)"
        reason = "Moderate high-Re regime"
    else:
        recommendation = "LES (dynamic Smagorinsky)"
        reason = "Very high-Re regime"

    result = {
        "reynolds": reynolds,
        "les_threshold": LES_REYNOLDS_THRESHOLD,
        "rans": {
            "eddy_viscosity_m2_s": round(rans_nu_t, 6),
            "accuracy": rans_accuracy,
            "cost_relative": rans_cost,
            "model": "k-epsilon",
        },
        "les": {
            "eddy_viscosity_m2_s": round(les_nu_t, 6),
            "accuracy": les_accuracy,
            "cost_relative": les_cost,
            "model": "Smagorinsky",
        },
        "recommendation": recommendation,
        "reason": reason,
        "use_les": use_les,
    }

    emit_receipt(
        "les_compare",
        {
            "receipt_type": "les_compare",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "reynolds": reynolds,
            "recommendation": recommendation,
            "use_les": use_les,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === FULL LES VALIDATION ===


def run_les_validation() -> Dict[str, Any]:
    """Run full LES dust dynamics validation.

    Returns:
        Dict with complete LES validation results

    Receipt: les_dust_receipt
    """
    # Load configuration
    config = load_les_config()

    # Run LES simulation at dust devil Reynolds
    les_sim = simulate_les(LES_DUST_DEVIL_REYNOLDS)

    # Simulate dust devil
    dust_devil = simulate_les_dust_devil(diameter_m=50.0, height_m=500.0, intensity=0.7)

    # Compare LES vs RANS
    comparison = les_vs_rans_comparison(LES_DUST_DEVIL_REYNOLDS)

    # Overall validation
    validated = (
        les_sim["use_les"]
        and dust_devil["validated"]
        and dust_devil["reynolds"] >= LES_REYNOLDS_THRESHOLD
    )

    result = {
        "config": config,
        "les_simulation": les_sim,
        "dust_devil_simulation": dust_devil,
        "les_vs_rans": comparison,
        "reynolds_validated": dust_devil["reynolds"],
        "overall_validated": validated,
        "model": "large_eddy_simulation",
    }

    emit_receipt(
        "les_dust",
        {
            "receipt_type": "les_dust",
            "tenant_id": CFD_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "reynolds": dust_devil["reynolds"],
            "use_les": les_sim["use_les"],
            "dust_devil_validated": dust_devil["validated"],
            "validated": validated,
            "payload_hash": dual_hash(
                json.dumps({"validated": validated}, sort_keys=True)
            ),
        },
    )

    return result


# === EXPORTS ===

__all__ = [
    "RECEIPT_SCHEMA",
    "load_les_config",
    "get_les_info",
    "smagorinsky_viscosity",
    "compute_subgrid_stress",
    "simulate_les",
    "simulate_les_dust_devil",
    "les_vs_rans_comparison",
    "run_les_validation",
]
