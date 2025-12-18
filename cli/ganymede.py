"""Ganymede magnetic field navigation CLI commands.

Commands for:
- Ganymede configuration display
- Magnetic field navigation simulation
- D9+Ganymede hybrid runs
"""

import json


def cmd_ganymede_info():
    """Show Ganymede configuration."""
    from src.ganymede_mag_hybrid import get_ganymede_info

    info = get_ganymede_info()
    print(json.dumps(info, indent=2))


def cmd_ganymede_config():
    """Show Ganymede configuration from spec."""
    from src.ganymede_mag_hybrid import load_ganymede_config

    config = load_ganymede_config()
    print(json.dumps(config, indent=2))


def cmd_ganymede_navigate(
    mode: str = "field_following", duration_hrs: int = 24, simulate: bool = False
):
    """Run Ganymede navigation simulation.

    Args:
        mode: Navigation mode (field_following, magnetopause_crossing, polar_transit)
        duration_hrs: Simulation duration in hours
        simulate: Whether to run in simulation mode
    """
    from src.ganymede_mag_hybrid import simulate_navigation

    result = simulate_navigation(mode, duration_hrs)
    print(json.dumps(result, indent=2))


def cmd_ganymede_field(x: float, y: float, z: float):
    """Show magnetic field strength at position.

    Args:
        x: X coordinate in km from Ganymede center
        y: Y coordinate in km from Ganymede center
        z: Z coordinate in km from Ganymede center
    """
    from src.ganymede_mag_hybrid import compute_field_strength

    field = compute_field_strength((x, y, z))
    result = {
        "position_km": (x, y, z),
        "field_strength_nT": round(field, 2),
    }
    print(json.dumps(result, indent=2))


def cmd_ganymede_autonomy(simulate: bool = False):
    """Show Ganymede autonomy metrics."""
    from src.ganymede_mag_hybrid import simulate_navigation, compute_autonomy

    nav = simulate_navigation("field_following", 24)
    autonomy = compute_autonomy(nav)
    result = {
        "autonomy_achieved": autonomy,
        "navigation_mode": nav["mode"],
        "duration_hrs": nav["duration_hrs"],
    }
    print(json.dumps(result, indent=2))


def cmd_d9_ganymede_hybrid(
    tree_size: int = 10**12,
    base_alpha: float = 3.26,
    mode: str = "field_following",
    duration_hrs: int = 24,
    simulate: bool = False,
):
    """Run integrated D9+Ganymede hybrid.

    Args:
        tree_size: Tree size for D9 recursion
        base_alpha: Base alpha for D9
        mode: Navigation mode
        duration_hrs: Simulation duration
        simulate: Whether to run in simulation mode
    """
    from src.ganymede_mag_hybrid import d9_ganymede_hybrid

    result = d9_ganymede_hybrid(tree_size, base_alpha, mode, duration_hrs)
    print(json.dumps(result, indent=2))


def cmd_d9_push(
    tree_size: int = 10**12, base_alpha: float = 3.26, simulate: bool = False
):
    """Run D9 recursion push for alpha >= 3.50.

    Args:
        tree_size: Tree size
        base_alpha: Base alpha
        simulate: Whether to run in simulation mode
    """
    from src.fractal_layers import d9_push

    result = d9_push(tree_size, base_alpha, simulate)
    print(json.dumps(result, indent=2))


def cmd_d9_info():
    """Show D9 configuration."""
    from src.fractal_layers import get_d9_info

    info = get_d9_info()
    print(json.dumps(info, indent=2))


def cmd_drone_info():
    """Show Atacama drone configuration."""
    from src.atacama_drone import get_drone_info

    info = get_drone_info()
    print(json.dumps(info, indent=2))


def cmd_drone_config():
    """Show Atacama drone configuration from spec."""
    from src.atacama_drone import load_drone_config

    config = load_drone_config()
    print(json.dumps(config, indent=2))


def cmd_drone_coverage(
    n_drones: int = 10, area_km2: float = 1000.0, simulate: bool = False
):
    """Run drone swarm coverage simulation.

    Args:
        n_drones: Number of drones
        area_km2: Area to cover
        simulate: Whether to run in simulation mode
    """
    from src.atacama_drone import simulate_swarm_coverage

    result = simulate_swarm_coverage(n_drones, area_km2)
    print(json.dumps(result, indent=2))


def cmd_drone_sample(rate_hz: int = 10, duration_s: int = 60, simulate: bool = False):
    """Run drone dust sampling.

    Args:
        rate_hz: Sampling rate in Hz
        duration_s: Sampling duration in seconds
        simulate: Whether to run in simulation mode
    """
    from src.atacama_drone import sample_dust_metrics

    result = sample_dust_metrics(rate_hz, duration_s)
    print(json.dumps(result, indent=2))


def cmd_drone_validate(
    n_drones: int = 10,
    area_km2: float = 1000.0,
    duration_s: int = 60,
    simulate: bool = False,
):
    """Run full drone validation.

    Args:
        n_drones: Number of drones
        area_km2: Area to cover
        duration_s: Sampling duration
        simulate: Whether to run in simulation mode
    """
    from src.atacama_drone import run_drone_validation

    result = run_drone_validation(n_drones, area_km2, duration_s)
    print(json.dumps(result, indent=2))
