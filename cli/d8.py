"""D8 recursion CLI commands.

Commands for D8 fractal recursion with unified RL sync.
"""

import json

from src.fractal_layers import (
    get_d8_info,
    d8_push,
)


def cmd_d8_info():
    """Show D8 + unified RL configuration."""
    info = get_d8_info()
    print(json.dumps(info, indent=2))


def cmd_d8_push(tree_size: int, base_alpha: float, simulate: bool):
    """Run D8 recursion for alpha >= 3.45."""
    result = d8_push(tree_size, base_alpha, simulate)
    print(json.dumps(result, indent=2))
