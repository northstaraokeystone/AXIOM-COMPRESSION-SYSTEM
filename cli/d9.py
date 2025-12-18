"""D9 recursion CLI commands.

Commands for:
- D9 recursion configuration
- D9 push operations
- D9+Ganymede hybrid runs
"""

import json


def cmd_d9_info():
    """Show D9 configuration."""
    from src.fractal_layers import get_d9_info

    info = get_d9_info()
    print(json.dumps(info, indent=2))


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
