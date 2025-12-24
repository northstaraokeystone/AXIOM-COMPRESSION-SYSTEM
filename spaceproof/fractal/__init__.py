"""Fractal entropy and recursion modules for SpaceProof compression.

This package provides multi-scale fractal entropy calculations,
recursive fractal boost functions, and depth-specific implementations.
"""

from .core import (
    recursive_fractal,
    recursive_fractal_sweep,
    multi_scale_fractal,
    get_recursive_fractal_info,
)
from .alpha import (
    scale_adjusted_correlation,
    get_scale_factor,
    compute_fractal_contribution,
    get_expected_alpha_at_scale,
)
from .adaptive import (
    adaptive_termination_check,
)

__all__ = [
    # Core fractal functions
    "recursive_fractal",
    "recursive_fractal_sweep",
    "multi_scale_fractal",
    "get_recursive_fractal_info",
    # Alpha computations
    "scale_adjusted_correlation",
    "get_scale_factor",
    "compute_fractal_contribution",
    "get_expected_alpha_at_scale",
    # Adaptive termination
    "adaptive_termination_check",
]

RECEIPT_SCHEMA = {
    "module": "src.fractal",
    "receipt_types": [
        "fractal_layer",
        "fractal_recursion",
        "fractal_contribution",
    ],
    "version": "1.0.0",
}
