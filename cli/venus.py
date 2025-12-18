"""Venus acid-cloud autonomy CLI commands."""

import json


def cmd_venus_info():
    """Show Venus configuration."""
    from src.venus_acid_hybrid import get_venus_info

    info = get_venus_info()
    print(json.dumps(info, indent=2))


def cmd_venus_cloud(altitude_km: float = 55.0):
    """Show cloud zone analysis at altitude."""
    from src.venus_acid_hybrid import compute_cloud_zone

    result = compute_cloud_zone(altitude_km)
    print(json.dumps(result, indent=2))


def cmd_venus_acid(material: str = "ptfe"):
    """Test acid resistance for material."""
    from src.venus_acid_hybrid import simulate_acid_resistance

    result = simulate_acid_resistance(material)
    print(json.dumps(result, indent=2))


def cmd_venus_ops(duration_days: int = 30, altitude_km: float = 55.0):
    """Run Venus cloud operations simulation."""
    from src.venus_acid_hybrid import simulate_cloud_ops

    result = simulate_cloud_ops(duration_days, altitude_km)
    print(json.dumps(result, indent=2))


def cmd_venus_autonomy():
    """Show Venus autonomy metrics."""
    from src.venus_acid_hybrid import simulate_cloud_ops, compute_autonomy

    ops = simulate_cloud_ops(duration_days=30, altitude_km=55.0)
    autonomy = compute_autonomy(ops)
    print(json.dumps({"autonomy": autonomy, "ops": ops}, indent=2))


def cmd_venus_config():
    """Show Venus configuration from spec."""
    from src.venus_acid_hybrid import load_venus_config

    config = load_venus_config()
    print(json.dumps(config, indent=2))


def cmd_venus_simulate(altitude_km: float = 55.0, duration_hrs: int = 24, simulate: bool = False):
    """Run Venus cloud simulation."""
    from src.venus_acid_hybrid import simulate_cloud_ops

    result = simulate_cloud_ops(duration_days=duration_hrs / 24, altitude_km=altitude_km)
    print(json.dumps(result, indent=2))
