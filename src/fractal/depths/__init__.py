"""Depth-specific fractal recursion implementations.

D1-D5: Early depth specifications (in d1_d5.py)
D6-D10: Middle depth specifications (in d6_d10.py)
D11-D14: Deep recursion specifications (in d11_d14.py)
D15-D17: Advanced quantum/topological recursion (in d15_d17.py)
"""

from .d1_d5 import (
    # D4
    D4_ALPHA_FLOOR, D4_ALPHA_TARGET, D4_ALPHA_CEILING,
    D4_INSTABILITY_MAX, D4_TREE_MIN,
    get_d4_spec, get_d4_uplift, d4_recursive_fractal, d4_push, get_d4_info,
    # D5
    D5_ALPHA_FLOOR, D5_ALPHA_TARGET, D5_ALPHA_CEILING,
    D5_INSTABILITY_MAX, D5_TREE_MIN, D5_UPLIFT,
    get_d5_spec, get_d5_uplift, d5_recursive_fractal, d5_push, get_d5_info,
)
from .d6_d10 import (
    # D6
    D6_ALPHA_FLOOR, D6_ALPHA_TARGET, D6_ALPHA_CEILING,
    D6_INSTABILITY_MAX, D6_TREE_MIN, D6_UPLIFT,
    get_d6_spec, get_d6_uplift, d6_recursive_fractal, d6_push, get_d6_info,
    # D7
    D7_ALPHA_FLOOR, D7_ALPHA_TARGET, D7_ALPHA_CEILING,
    D7_INSTABILITY_MAX, D7_TREE_MIN, D7_UPLIFT,
    get_d7_spec, get_d7_uplift, d7_recursive_fractal, d7_push, get_d7_info,
    # D8
    D8_ALPHA_FLOOR, D8_ALPHA_TARGET, D8_ALPHA_CEILING,
    D8_INSTABILITY_MAX, D8_TREE_MIN, D8_UPLIFT,
    get_d8_spec, get_d8_uplift, d8_recursive_fractal, d8_push, get_d8_info,
    # D9
    D9_ALPHA_FLOOR, D9_ALPHA_TARGET, D9_ALPHA_CEILING,
    D9_INSTABILITY_MAX, D9_TREE_MIN, D9_UPLIFT,
    get_d9_spec, get_d9_uplift, d9_recursive_fractal, d9_push, get_d9_info,
    # D10
    D10_ALPHA_FLOOR, D10_ALPHA_TARGET, D10_ALPHA_CEILING,
    D10_INSTABILITY_MAX, D10_TREE_MIN, D10_UPLIFT,
    get_d10_spec, get_d10_uplift, d10_recursive_fractal, d10_push, get_d10_info,
)
from .d11_d14 import (
    # D11
    D11_ALPHA_FLOOR, D11_ALPHA_TARGET, D11_ALPHA_CEILING,
    D11_INSTABILITY_MAX, D11_TREE_MIN, D11_UPLIFT,
    get_d11_spec, get_d11_uplift, d11_recursive_fractal, d11_push, get_d11_info,
    # D12 (constants only)
    D12_ALPHA_FLOOR, D12_ALPHA_TARGET, D12_ALPHA_CEILING,
    D12_INSTABILITY_MAX, D12_TREE_MIN, D12_UPLIFT,
    # D13
    D13_ALPHA_FLOOR, D13_ALPHA_TARGET, D13_ALPHA_CEILING,
    D13_INSTABILITY_MAX, D13_TREE_MIN, D13_UPLIFT,
    get_d13_spec, get_d13_uplift, d13_recursive_fractal, d13_push, get_d13_info,
    # D14
    D14_ALPHA_FLOOR, D14_ALPHA_TARGET, D14_ALPHA_CEILING,
    D14_INSTABILITY_MAX, D14_TREE_MIN, D14_UPLIFT,
    D14_ADAPTIVE_TERMINATION, D14_TERMINATION_THRESHOLD,
    get_d14_spec, get_d14_uplift, d14_recursive_fractal, d14_push, get_d14_info,
)
from .d15_d17 import (
    # D15
    D15_ALPHA_FLOOR, D15_ALPHA_TARGET, D15_ALPHA_CEILING,
    D15_INSTABILITY_MAX, D15_TREE_MIN, D15_UPLIFT,
    D15_QUANTUM_ENTANGLEMENT, D15_ENTANGLEMENT_CORRELATION, D15_TERMINATION_THRESHOLD,
    get_d15_spec, get_d15_uplift, d15_recursive_fractal, d15_push, get_d15_info,
    d15_quantum_push, compute_entanglement_correlation, entangled_termination_check,
    # D16
    D16_ALPHA_FLOOR, D16_ALPHA_TARGET, D16_ALPHA_CEILING,
    D16_INSTABILITY_MAX, D16_TREE_MIN, D16_UPLIFT,
    D16_TOPOLOGICAL, D16_HOMOLOGY_DIMENSION, D16_PERSISTENCE_THRESHOLD,
    get_d16_spec, get_d16_uplift, d16_push, get_d16_info,
    d16_topological_push, d16_kuiper_hybrid,
    compute_persistent_homology, compute_betti_numbers, topological_compression_ratio,
    multidimensional_scaling,
    # D17
    D17_ALPHA_FLOOR, D17_ALPHA_TARGET, D17_ALPHA_CEILING,
    D17_INSTABILITY_MAX, D17_TREE_MIN, D17_UPLIFT,
    D17_DEPTH_FIRST, D17_NON_ASYMPTOTIC, D17_TERMINATION_THRESHOLD,
    get_d17_spec, get_d17_uplift, d17_push, get_d17_info,
    d17_depth_first_push, d17_heliosphere_hybrid,
    depth_first_traversal, check_asymptotic_ceiling, compute_uplift_sustainability,
)

__all__ = [
    # D4
    "D4_ALPHA_FLOOR", "D4_ALPHA_TARGET", "D4_ALPHA_CEILING",
    "D4_INSTABILITY_MAX", "D4_TREE_MIN",
    "get_d4_spec", "get_d4_uplift", "d4_recursive_fractal", "d4_push", "get_d4_info",
    # D5
    "D5_ALPHA_FLOOR", "D5_ALPHA_TARGET", "D5_ALPHA_CEILING",
    "D5_INSTABILITY_MAX", "D5_TREE_MIN", "D5_UPLIFT",
    "get_d5_spec", "get_d5_uplift", "d5_recursive_fractal", "d5_push", "get_d5_info",
    # D6
    "D6_ALPHA_FLOOR", "D6_ALPHA_TARGET", "D6_ALPHA_CEILING",
    "D6_INSTABILITY_MAX", "D6_TREE_MIN", "D6_UPLIFT",
    "get_d6_spec", "get_d6_uplift", "d6_recursive_fractal", "d6_push", "get_d6_info",
    # D7
    "D7_ALPHA_FLOOR", "D7_ALPHA_TARGET", "D7_ALPHA_CEILING",
    "D7_INSTABILITY_MAX", "D7_TREE_MIN", "D7_UPLIFT",
    "get_d7_spec", "get_d7_uplift", "d7_recursive_fractal", "d7_push", "get_d7_info",
    # D8
    "D8_ALPHA_FLOOR", "D8_ALPHA_TARGET", "D8_ALPHA_CEILING",
    "D8_INSTABILITY_MAX", "D8_TREE_MIN", "D8_UPLIFT",
    "get_d8_spec", "get_d8_uplift", "d8_recursive_fractal", "d8_push", "get_d8_info",
    # D9
    "D9_ALPHA_FLOOR", "D9_ALPHA_TARGET", "D9_ALPHA_CEILING",
    "D9_INSTABILITY_MAX", "D9_TREE_MIN", "D9_UPLIFT",
    "get_d9_spec", "get_d9_uplift", "d9_recursive_fractal", "d9_push", "get_d9_info",
    # D10
    "D10_ALPHA_FLOOR", "D10_ALPHA_TARGET", "D10_ALPHA_CEILING",
    "D10_INSTABILITY_MAX", "D10_TREE_MIN", "D10_UPLIFT",
    "get_d10_spec", "get_d10_uplift", "d10_recursive_fractal", "d10_push", "get_d10_info",
    # D11
    "D11_ALPHA_FLOOR", "D11_ALPHA_TARGET", "D11_ALPHA_CEILING",
    "D11_INSTABILITY_MAX", "D11_TREE_MIN", "D11_UPLIFT",
    "get_d11_spec", "get_d11_uplift", "d11_recursive_fractal", "d11_push", "get_d11_info",
    # D12
    "D12_ALPHA_FLOOR", "D12_ALPHA_TARGET", "D12_ALPHA_CEILING",
    "D12_INSTABILITY_MAX", "D12_TREE_MIN", "D12_UPLIFT",
    # D13
    "D13_ALPHA_FLOOR", "D13_ALPHA_TARGET", "D13_ALPHA_CEILING",
    "D13_INSTABILITY_MAX", "D13_TREE_MIN", "D13_UPLIFT",
    "get_d13_spec", "get_d13_uplift", "d13_recursive_fractal", "d13_push", "get_d13_info",
    # D14
    "D14_ALPHA_FLOOR", "D14_ALPHA_TARGET", "D14_ALPHA_CEILING",
    "D14_INSTABILITY_MAX", "D14_TREE_MIN", "D14_UPLIFT",
    "D14_ADAPTIVE_TERMINATION", "D14_TERMINATION_THRESHOLD",
    "get_d14_spec", "get_d14_uplift", "d14_recursive_fractal", "d14_push", "get_d14_info",
    # D15
    "D15_ALPHA_FLOOR", "D15_ALPHA_TARGET", "D15_ALPHA_CEILING",
    "D15_INSTABILITY_MAX", "D15_TREE_MIN", "D15_UPLIFT",
    "D15_QUANTUM_ENTANGLEMENT", "D15_ENTANGLEMENT_CORRELATION", "D15_TERMINATION_THRESHOLD",
    "get_d15_spec", "get_d15_uplift", "d15_recursive_fractal", "d15_push", "get_d15_info",
    "d15_quantum_push", "compute_entanglement_correlation", "entangled_termination_check",
    # D16
    "D16_ALPHA_FLOOR", "D16_ALPHA_TARGET", "D16_ALPHA_CEILING",
    "D16_INSTABILITY_MAX", "D16_TREE_MIN", "D16_UPLIFT",
    "D16_TOPOLOGICAL", "D16_HOMOLOGY_DIMENSION", "D16_PERSISTENCE_THRESHOLD",
    "get_d16_spec", "get_d16_uplift", "d16_push", "get_d16_info",
    "d16_topological_push", "d16_kuiper_hybrid",
    "compute_persistent_homology", "compute_betti_numbers", "topological_compression_ratio",
    "multidimensional_scaling",
    # D17
    "D17_ALPHA_FLOOR", "D17_ALPHA_TARGET", "D17_ALPHA_CEILING",
    "D17_INSTABILITY_MAX", "D17_TREE_MIN", "D17_UPLIFT",
    "D17_DEPTH_FIRST", "D17_NON_ASYMPTOTIC", "D17_TERMINATION_THRESHOLD",
    "get_d17_spec", "get_d17_uplift", "d17_push", "get_d17_info",
    "d17_depth_first_push", "d17_heliosphere_hybrid",
    "depth_first_traversal", "check_asymptotic_ceiling", "compute_uplift_sustainability",
]

RECEIPT_SCHEMA = {
    "module": "src.fractal.depths",
    "receipt_types": [
        "d4_recursion", "d5_recursion", "d6_recursion", "d7_recursion",
        "d8_recursion", "d9_recursion", "d10_recursion", "d11_recursion",
        "d13_recursion", "d14_recursion", "d15_recursion", "d16_recursion",
        "d17_recursion",
    ],
    "version": "1.0.0",
}
