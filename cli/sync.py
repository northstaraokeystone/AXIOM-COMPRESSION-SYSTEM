"""Multi-planet sync CLI commands.

Commands for unified RL coordination between Titan and Europa.
"""

import json

from src.multi_planet_sync import (
    load_sync_config,
    run_sync,
    run_sync_cycle,
    d8_multi_sync,
    get_sync_info,
    RESOURCE_SHARE_EFFICIENCY,
)


def cmd_sync_info():
    """Show sync configuration."""
    info = get_sync_info()
    print(json.dumps(info, indent=2))


def cmd_sync_run(simulate: bool = True):
    """Run sync workflow."""
    result = run_sync(simulate)
    print(json.dumps(result, indent=2))


def cmd_sync_efficiency():
    """Show efficiency metrics."""
    config = load_sync_config()
    cycle = run_sync_cycle()
    result = {
        "target_efficiency": RESOURCE_SHARE_EFFICIENCY,
        "achieved_efficiency": cycle["efficiency"],
        "efficiency_met": cycle["efficiency"] >= RESOURCE_SHARE_EFFICIENCY,
        "moons": config.get("moons", []),
        "sync_interval_hrs": config.get("unified_rl", {}).get("sync_interval_hrs", 24),
    }
    print(json.dumps(result, indent=2))


def cmd_d8_multi_sync(tree_size: int, base_alpha: float, simulate: bool):
    """Run integrated D8 + multi-planet sync."""
    result = d8_multi_sync(tree_size, base_alpha)
    print(json.dumps(result, indent=2))
