"""Titan methane hybrid CLI commands.

Commands for D6 + Titan methane harvesting operations.
"""

import json

from src.fractal_layers import (
    get_d6_info,
    d6_push,
)
from src.titan_methane_hybrid import (
    load_titan_config,
    simulate_harvest,
    compute_autonomy,
    d6_titan_hybrid,
    get_titan_info,
    TITAN_AUTONOMY_REQUIREMENT,
)
from src.perovskite_efficiency import (
    project_efficiency,
    get_perovskite_info,
)


def cmd_d6_info():
    """Show D6 + Titan configuration."""
    info = get_d6_info()
    print(json.dumps(info, indent=2))


def cmd_d6_push(tree_size: int, base_alpha: float, simulate: bool):
    """Run D6 recursion for alpha >= 3.33."""
    result = d6_push(tree_size, base_alpha, simulate)
    print(json.dumps(result, indent=2))


def cmd_d6_titan_hybrid(
    tree_size: int, base_alpha: float, duration_days: int, simulate: bool
):
    """Run integrated D6 + Titan hybrid."""
    result = d6_titan_hybrid(tree_size, base_alpha, duration_days)
    print(json.dumps(result, indent=2))


def cmd_titan_info():
    """Show Titan methane hybrid configuration."""
    info = get_titan_info()
    print(json.dumps(info, indent=2))


def cmd_titan_config():
    """Show Titan configuration from spec."""
    config = load_titan_config()
    print(json.dumps(config, indent=2))


def cmd_titan_simulate(duration_days: int, extraction_rate: float, simulate: bool):
    """Run Titan methane harvest simulation."""
    result = simulate_harvest(duration_days, extraction_rate)
    print(json.dumps(result, indent=2))


def cmd_titan_autonomy(harvest_rate: float, consumption_rate: float):
    """Compute Titan autonomy metrics."""
    autonomy = compute_autonomy(harvest_rate, consumption_rate)
    result = {
        "harvest_rate_kg_hr": harvest_rate,
        "consumption_rate_kg_hr": consumption_rate,
        "autonomy": autonomy,
        "autonomy_requirement": TITAN_AUTONOMY_REQUIREMENT,
        "autonomy_met": autonomy >= TITAN_AUTONOMY_REQUIREMENT,
    }
    print(json.dumps(result, indent=2))


def cmd_perovskite_info():
    """Show perovskite efficiency configuration."""
    info = get_perovskite_info()
    print(json.dumps(info, indent=2))


def cmd_perovskite_project(years: int, growth_rate: float):
    """Project efficiency improvement timeline."""
    result = project_efficiency(years, growth_rate)
    print(json.dumps(result, indent=2))
