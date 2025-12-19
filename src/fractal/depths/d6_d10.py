"""D6-D10 depth recursion functions.

Placeholder - functions imported from parent fractal_layers.py
"""

from ...fractal_layers import (
    get_d6_spec, get_d6_uplift, d6_recursive_fractal, d6_push, get_d6_info,
    get_d7_spec, get_d7_uplift, d7_recursive_fractal, d7_push, get_d7_info,
    get_d8_spec, get_d8_uplift, d8_recursive_fractal, d8_push, get_d8_info,
    get_d9_spec, get_d9_uplift, d9_recursive_fractal, d9_push, get_d9_info,
    get_d10_spec, get_d10_uplift, d10_recursive_fractal, d10_push, get_d10_info,
)

__all__ = [
    "get_d6_spec", "get_d6_uplift", "d6_recursive_fractal", "d6_push", "get_d6_info",
    "get_d7_spec", "get_d7_uplift", "d7_recursive_fractal", "d7_push", "get_d7_info",
    "get_d8_spec", "get_d8_uplift", "d8_recursive_fractal", "d8_push", "get_d8_info",
    "get_d9_spec", "get_d9_uplift", "d9_recursive_fractal", "d9_push", "get_d9_info",
    "get_d10_spec", "get_d10_uplift", "d10_recursive_fractal", "d10_push", "get_d10_info",
]

RECEIPT_SCHEMA = {
    "module": "src.fractal.depths.d6_d10",
    "receipt_types": ["d6_fractal", "d7_fractal", "d8_fractal", "d9_fractal", "d10_fractal"],
    "version": "1.0.0",
}
