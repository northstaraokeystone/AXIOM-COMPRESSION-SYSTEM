"""CFD dust dynamics CLI commands."""

import json


def cmd_cfd_info():
    """Show CFD configuration."""
    from src.cfd_dust_dynamics import get_cfd_info

    info = get_cfd_info()
    print(json.dumps(info, indent=2))


def cmd_cfd_reynolds(velocity: float = 1.0, length: float = 0.001):
    """Compute Reynolds number."""
    from src.cfd_dust_dynamics import compute_reynolds_number

    re = compute_reynolds_number(velocity, length)
    print(json.dumps({"velocity_m_s": velocity, "length_m": length, "reynolds_number": re}, indent=2))


def cmd_cfd_settling(particle_size_um: float = 10.0):
    """Compute Stokes settling velocity."""
    from src.cfd_dust_dynamics import stokes_settling

    v_s = stokes_settling(particle_size_um)
    print(json.dumps({"particle_size_um": particle_size_um, "settling_velocity_m_s": v_s}, indent=2))


def cmd_cfd_storm(intensity: float = 0.5, duration_hrs: float = 24.0):
    """Run dust storm simulation."""
    from src.cfd_dust_dynamics import simulate_dust_storm

    result = simulate_dust_storm(intensity, duration_hrs)
    print(json.dumps(result, indent=2))


def cmd_cfd_validate():
    """Run full CFD validation against Atacama."""
    from src.cfd_dust_dynamics import run_cfd_validation

    result = run_cfd_validation()
    print(json.dumps(result, indent=2))


# Turbulent CFD commands

def cmd_turbulent_info():
    """Show turbulent CFD configuration."""
    from src.cfd_dust_dynamics import (
        CFD_REYNOLDS_TURBULENT_THRESHOLD,
        CFD_TURBULENCE_MODEL_KEPS,
        CFD_REYNOLDS_MARS_TURBULENT,
    )

    info = {
        "model": "navier_stokes_turbulent",
        "reynolds_threshold": CFD_REYNOLDS_TURBULENT_THRESHOLD,
        "turbulence_model": CFD_TURBULENCE_MODEL_KEPS,
        "mars_storm_reynolds": CFD_REYNOLDS_MARS_TURBULENT,
    }
    print(json.dumps(info, indent=2))


def cmd_turbulent_config():
    """Show turbulent CFD configuration from spec."""
    from src.cfd_dust_dynamics import load_turbulent_cfd_config

    config = load_turbulent_cfd_config()
    print(json.dumps(config, indent=2))


def cmd_turbulent_simulate(reynolds: float = 5000, duration_s: float = 100, simulate: bool = False):
    """Run turbulent flow simulation."""
    from src.cfd_dust_dynamics import simulate_turbulent

    result = simulate_turbulent(reynolds, duration_s)
    print(json.dumps(result, indent=2))


def cmd_turbulent_storm(intensity: float = 0.8, simulate: bool = False):
    """Run turbulent dust storm simulation."""
    from src.cfd_dust_dynamics import dust_storm_turbulent

    result = dust_storm_turbulent(intensity=intensity, duration_hrs=24)
    print(json.dumps(result, indent=2))


def cmd_turbulent_validate(simulate: bool = False):
    """Run turbulent CFD validation."""
    from src.cfd_dust_dynamics import run_turbulent_validation

    result = run_turbulent_validation()
    print(json.dumps(result, indent=2))
