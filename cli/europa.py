"""Europa ice drilling hybrid CLI commands.

Commands for D7 + Europa ice drilling operations.
"""

import json

from src.fractal_layers import (
    get_d7_info,
    d7_push,
)
from src.europa_ice_hybrid import (
    load_europa_config,
    simulate_drilling,
    compute_autonomy,
    d7_europa_hybrid,
    get_europa_info,
    EUROPA_AUTONOMY_REQUIREMENT,
)


def cmd_d7_info():
    """Show D7 + Europa configuration."""
    info = get_d7_info()
    print(json.dumps(info, indent=2))


def cmd_d7_push(tree_size: int, base_alpha: float, simulate: bool):
    """Run D7 recursion for alpha >= 3.40."""
    result = d7_push(tree_size, base_alpha, simulate)
    print(json.dumps(result, indent=2))


def cmd_d7_europa_hybrid(
    tree_size: int, base_alpha: float, depth_m: int, duration_days: int, simulate: bool
):
    """Run integrated D7 + Europa hybrid."""
    result = d7_europa_hybrid(tree_size, base_alpha, depth_m, duration_days)
    print(json.dumps(result, indent=2))


def cmd_europa_info():
    """Show Europa ice drilling configuration."""
    info = get_europa_info()
    print(json.dumps(info, indent=2))


def cmd_europa_config():
    """Show Europa configuration from spec."""
    config = load_europa_config()
    print(json.dumps(config, indent=2))


def cmd_europa_simulate(
    depth_m: int, duration_days: int, drill_rate: float, simulate: bool
):
    """Run Europa ice drilling simulation."""
    result = simulate_drilling(depth_m, duration_days, drill_rate)
    print(json.dumps(result, indent=2))


def cmd_europa_autonomy(drill_rate: float, resupply_interval: float):
    """Compute Europa autonomy metrics."""
    autonomy = compute_autonomy(drill_rate, resupply_interval)
    result = {
        "drill_rate_m_hr": drill_rate,
        "resupply_interval_days": resupply_interval,
        "autonomy": autonomy,
        "autonomy_requirement": EUROPA_AUTONOMY_REQUIREMENT,
        "autonomy_met": autonomy >= EUROPA_AUTONOMY_REQUIREMENT,
    }
    print(json.dumps(result, indent=2))
