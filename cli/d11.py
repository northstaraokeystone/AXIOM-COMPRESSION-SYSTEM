"""D11 recursion CLI commands."""

import json


def cmd_d11_info():
    """Show D11 configuration."""
    from src.fractal_layers import get_d11_info

    info = get_d11_info()
    print(json.dumps(info, indent=2))


def cmd_d11_push(tree_size: int, base_alpha: float, simulate: bool = False):
    """Run D11 recursion push for alpha >= 3.60."""
    from src.fractal_layers import d11_push

    result = d11_push(tree_size, base_alpha, simulate=simulate)
    print(json.dumps(result, indent=2))


def cmd_d11_venus_hybrid(tree_size: int, base_alpha: float, simulate: bool = False):
    """Run integrated D11 + Venus acid-cloud hybrid."""
    from src.venus_acid_hybrid import d11_venus_hybrid

    result = d11_venus_hybrid(tree_size, base_alpha, simulate=simulate)
    print(json.dumps(result, indent=2))
