"""Mercury thermal autonomy CLI commands."""

import json


def cmd_mercury_info():
    """Show Mercury configuration."""
    from src.mercury_thermal_hybrid import get_mercury_info

    info = get_mercury_info()
    print(json.dumps(info, indent=2))


def cmd_mercury_thermal(time_of_day: float = 0.5, latitude: float = 0.0):
    """Show thermal zone at given time/location."""
    from src.mercury_thermal_hybrid import compute_thermal_zone

    result = compute_thermal_zone(time_of_day, latitude)
    print(json.dumps(result, indent=2))


def cmd_mercury_alloy(alloy: str = "inconel_718", temp_c: float = 400.0, duration_hrs: float = 100.0):
    """Test alloy performance under Mercury conditions."""
    from src.mercury_thermal_hybrid import simulate_alloy_performance

    result = simulate_alloy_performance(alloy, temp_c, duration_hrs)
    print(json.dumps(result, indent=2))


def cmd_mercury_ops(simulate: bool = False):
    """Run Mercury thermal operations simulation."""
    from src.mercury_thermal_hybrid import simulate_thermal_ops

    result = simulate_thermal_ops()
    print(json.dumps(result, indent=2))


def cmd_mercury_autonomy():
    """Show Mercury autonomy metrics."""
    from src.mercury_thermal_hybrid import simulate_thermal_ops, compute_autonomy

    ops = simulate_thermal_ops()
    autonomy = compute_autonomy(ops)
    print(json.dumps({"autonomy": autonomy, "requirement": 0.995, "met": autonomy >= 0.995}, indent=2))


def cmd_mercury_hazard():
    """Show Mercury hazard assessment."""
    from src.mercury_thermal_hybrid import hazard_assessment

    result = hazard_assessment()
    print(json.dumps(result, indent=2))


def cmd_mercury_shield(flux_w_m2: float = 9082.0):
    """Show solar shield requirements."""
    from src.mercury_thermal_hybrid import compute_solar_shield_requirement

    result = compute_solar_shield_requirement(flux_w_m2)
    print(json.dumps(result, indent=2))


def cmd_mercury_budget(power_w: float = 500.0, radiator_m2: float = 2.0):
    """Analyze thermal budget."""
    from src.mercury_thermal_hybrid import thermal_budget_analysis

    result = thermal_budget_analysis(power_w, radiator_m2)
    print(json.dumps(result, indent=2))


def cmd_mercury_config():
    """Show Mercury configuration from spec."""
    from src.mercury_thermal_hybrid import load_mercury_config

    config = load_mercury_config()
    print(json.dumps(config, indent=2))


def cmd_mercury_simulate(zone: str = "terminator", duration_hrs: int = 24, simulate: bool = False):
    """Run Mercury thermal simulation for specified zone."""
    from src.mercury_thermal_hybrid import simulate_thermal_ops

    result = simulate_thermal_ops(night_duration_hrs=duration_hrs / 2, day_duration_hrs=duration_hrs / 2)
    result["zone"] = zone
    print(json.dumps(result, indent=2))


def cmd_mercury_cycling(cycles: int = 100, alloy: str = "inconel_718", simulate: bool = False):
    """Run thermal cycling test."""
    from src.mercury_thermal_hybrid import simulate_thermal_cycling

    result = simulate_thermal_cycling(cycles, 200)  # 200°C swing
    result["alloy"] = alloy
    print(json.dumps(result, indent=2))
