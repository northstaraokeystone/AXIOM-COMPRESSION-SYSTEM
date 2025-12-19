"""D11-D14 depth recursion functions.

Placeholder - functions imported from parent fractal_layers.py
"""

from ...fractal_layers import (
    get_d11_spec, get_d11_uplift, d11_recursive_fractal, d11_push, get_d11_info,
    get_d13_spec, get_d13_uplift, d13_recursive_fractal, d13_push, get_d13_info,
    get_d14_spec, get_d14_uplift, d14_recursive_fractal, d14_push, get_d14_info,
)

# Also import adaptive_termination_check if it exists
try:
    from ...fractal_layers import adaptive_termination_check
except ImportError:
    from ..adaptive import adaptive_termination_check

__all__ = [
    "get_d11_spec", "get_d11_uplift", "d11_recursive_fractal", "d11_push", "get_d11_info",
    "get_d13_spec", "get_d13_uplift", "d13_recursive_fractal", "d13_push", "get_d13_info",
    "get_d14_spec", "get_d14_uplift", "d14_recursive_fractal", "d14_push", "get_d14_info",
]

RECEIPT_SCHEMA = {
    "module": "src.fractal.depths.d11_d14",
    "receipt_types": ["d11_fractal", "d13_fractal", "d14_fractal"],
    "version": "1.0.0",
}
