"""Depth-specific fractal recursion implementations.

D1-D5: Early depth specifications
D6-D10: Middle depth specifications
D11-D14: Deep recursion specifications
"""

from .d1_d5 import (
    get_d4_spec,
    get_d4_uplift,
    d4_recursive_fractal,
    d4_push,
    get_d4_info,
    get_d5_spec,
    get_d5_uplift,
    d5_recursive_fractal,
    d5_push,
    get_d5_info,
)
from .d6_d10 import (
    get_d6_spec,
    get_d6_uplift,
    d6_recursive_fractal,
    d6_push,
    get_d6_info,
    get_d7_spec,
    get_d7_uplift,
    d7_recursive_fractal,
    d7_push,
    get_d7_info,
    get_d8_spec,
    get_d8_uplift,
    d8_recursive_fractal,
    d8_push,
    get_d8_info,
    get_d9_spec,
    get_d9_uplift,
    d9_recursive_fractal,
    d9_push,
    get_d9_info,
    get_d10_spec,
    get_d10_uplift,
    d10_recursive_fractal,
    d10_push,
    get_d10_info,
)
from .d11_d14 import (
    get_d11_spec,
    get_d11_uplift,
    d11_recursive_fractal,
    d11_push,
    get_d11_info,
    get_d13_spec,
    get_d13_uplift,
    d13_recursive_fractal,
    d13_push,
    get_d13_info,
    get_d14_spec,
    get_d14_uplift,
    d14_recursive_fractal,
    d14_push,
    get_d14_info,
)

__all__ = [
    # D4-D5
    "get_d4_spec", "get_d4_uplift", "d4_recursive_fractal", "d4_push", "get_d4_info",
    "get_d5_spec", "get_d5_uplift", "d5_recursive_fractal", "d5_push", "get_d5_info",
    # D6-D10
    "get_d6_spec", "get_d6_uplift", "d6_recursive_fractal", "d6_push", "get_d6_info",
    "get_d7_spec", "get_d7_uplift", "d7_recursive_fractal", "d7_push", "get_d7_info",
    "get_d8_spec", "get_d8_uplift", "d8_recursive_fractal", "d8_push", "get_d8_info",
    "get_d9_spec", "get_d9_uplift", "d9_recursive_fractal", "d9_push", "get_d9_info",
    "get_d10_spec", "get_d10_uplift", "d10_recursive_fractal", "d10_push", "get_d10_info",
    # D11-D14
    "get_d11_spec", "get_d11_uplift", "d11_recursive_fractal", "d11_push", "get_d11_info",
    "get_d13_spec", "get_d13_uplift", "d13_recursive_fractal", "d13_push", "get_d13_info",
    "get_d14_spec", "get_d14_uplift", "d14_recursive_fractal", "d14_push", "get_d14_info",
]

RECEIPT_SCHEMA = {
    "module": "src.fractal.depths",
    "receipt_types": [
        "d4_recursion", "d5_recursion", "d6_recursion", "d7_recursion",
        "d8_recursion", "d9_recursion", "d10_recursion", "d11_recursion",
        "d13_recursion", "d14_recursion",
    ],
    "version": "1.0.0",
}
