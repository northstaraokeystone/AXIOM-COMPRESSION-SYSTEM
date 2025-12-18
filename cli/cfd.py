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
