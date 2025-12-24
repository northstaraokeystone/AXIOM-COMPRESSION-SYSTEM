"""fractal_layers.py - Re-export wrapper for backward compatibility.

All implementation moved to src/fractal/ package.
This file exists ONLY to maintain import compatibility.

CLAUDEME COMPLIANT: ≤50 lines
"""

# Core constants and functions from alpha module
from .fractal.alpha import (
    BASE_TREE_SIZE, CORRELATION_DECAY_FACTOR, FRACTAL_BASE_CORRELATION,
    FRACTAL_ALPHA_CONTRIBUTION, TENANT_ID, scale_adjusted_correlation,
    get_scale_factor, compute_fractal_contribution, get_expected_alpha_at_scale,
    validate_scale_physics, get_fractal_layers_info,
)

# Core fractal functions
from .fractal.core import (
    FRACTAL_SCALES, FRACTAL_DIM_MIN, FRACTAL_DIM_MAX, FRACTAL_UPLIFT,
    CROSS_SCALE_CORRELATION_MIN, CROSS_SCALE_CORRELATION_MAX,
    FRACTAL_RECURSION_MAX_DEPTH, FRACTAL_RECURSION_DEFAULT_DEPTH, FRACTAL_RECURSION_DECAY,
    fractal_entropy, compute_fractal_dimension, cross_scale_correlation,
    multi_scale_fractal, get_fractal_hybrid_spec, recursive_fractal,
    recursive_fractal_sweep, get_recursive_fractal_info,
)

# Adaptive termination
from .fractal.adaptive import adaptive_termination_check, D14_TERMINATION_THRESHOLD

# All depth implementations (D4-D18)
from .fractal.depths import *

__all__ = [
    # Alpha/scale
    "BASE_TREE_SIZE", "CORRELATION_DECAY_FACTOR", "FRACTAL_BASE_CORRELATION",
    "FRACTAL_ALPHA_CONTRIBUTION", "TENANT_ID", "scale_adjusted_correlation",
    "get_scale_factor", "compute_fractal_contribution", "get_expected_alpha_at_scale",
    # Core fractal
    "FRACTAL_SCALES", "FRACTAL_DIM_MIN", "FRACTAL_DIM_MAX", "FRACTAL_UPLIFT",
    "CROSS_SCALE_CORRELATION_MIN", "CROSS_SCALE_CORRELATION_MAX",
    "FRACTAL_RECURSION_MAX_DEPTH", "FRACTAL_RECURSION_DEFAULT_DEPTH", "FRACTAL_RECURSION_DECAY",
    "fractal_entropy", "compute_fractal_dimension", "cross_scale_correlation",
    "multi_scale_fractal", "get_fractal_hybrid_spec", "recursive_fractal",
    "recursive_fractal_sweep", "get_recursive_fractal_info",
    "validate_scale_physics", "get_fractal_layers_info",
    # Adaptive
    "adaptive_termination_check", "D14_TERMINATION_THRESHOLD",
]
