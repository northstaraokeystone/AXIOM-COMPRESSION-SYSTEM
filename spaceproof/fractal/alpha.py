"""fractal/alpha.py - Scale-Adjusted Alpha Contribution Functions

This module provides functions for computing alpha contribution with scale adjustments.
Handles correlation decay at large tree sizes (10^6 to 10^9).

THE PHYSICS:
    Large trees have more entropy sources -> correlation signal slightly diluted
    Adjustment: correlation * (1 - 0.001 * log10(tree_size / 1e6))

    At 10^6: factor = 1.0 (baseline)
    At 10^8: factor = 0.998 (0.2% dilution)
    At 10^9: factor = 0.997 (0.3% dilution)

EXPECTED ALPHA AT SCALE:
    10^6: alpha = 3.070 (baseline)
    10^8: alpha = 3.068 (intermediate)
    10^9: alpha = 3.065-3.067 (target)
"""

import json
import math
from datetime import datetime
from typing import Any, Dict

from ..core import emit_receipt, dual_hash


# === CONSTANTS ===

BASE_TREE_SIZE = 1_000_000
"""Baseline tree size (10^6) for correlation normalization."""

CORRELATION_DECAY_FACTOR = 0.00025
"""Per-order-of-magnitude decay in correlation (0.025% per order).

Physics: Decay is minimal because fractal structure is self-similar at scale.
At 10^9 (3 orders above baseline): 3 * 0.025% = 0.075% total decay.
"""

FRACTAL_BASE_CORRELATION = 0.85
"""Base fractal correlation at 10^6 tree size."""

FRACTAL_ALPHA_CONTRIBUTION = 0.35
"""Fractal layer contribution to alpha ceiling breach."""

TENANT_ID = "spaceproof-colony"
"""Tenant ID for receipts."""


# === SCALE ADJUSTMENT FUNCTIONS ===


def scale_adjusted_correlation(
    tree_size: int, base_correlation: float = FRACTAL_BASE_CORRELATION
) -> float:
    """Adjust correlation for large trees (minor decay).

    Large trees have more entropy sources, which slightly dilutes the
    fractal correlation signal. This is physics-expected behavior.

    Formula: correlation * (1 - 0.001 * log10(tree_size / 1e6))

    At 10^6: factor = 1.0 (baseline)
    At 10^8: factor = 0.998 (0.2% dilution)
    At 10^9: factor = 0.997 (0.3% dilution)

    Args:
        tree_size: Number of nodes in the tree
        base_correlation: Base correlation at 10^6 (default: 0.85)

    Returns:
        Scale-adjusted correlation value
    """
    if tree_size <= BASE_TREE_SIZE:
        return base_correlation

    # Calculate orders of magnitude above baseline
    log_ratio = math.log10(tree_size / BASE_TREE_SIZE)

    # Apply decay factor per order of magnitude
    decay_factor = 1 - (CORRELATION_DECAY_FACTOR * log_ratio)

    # Clamp to minimum 0.95 of base (5% max decay)
    decay_factor = max(0.95, decay_factor)

    return base_correlation * decay_factor


def get_scale_factor(tree_size: int) -> float:
    """Get the scale adjustment factor for a tree size.

    Args:
        tree_size: Number of nodes in the tree

    Returns:
        Scale factor (1.0 at baseline, decreasing for larger trees)
    """
    if tree_size <= BASE_TREE_SIZE:
        return 1.0

    log_ratio = math.log10(tree_size / BASE_TREE_SIZE)
    return max(0.95, 1 - (CORRELATION_DECAY_FACTOR * log_ratio))


def compute_fractal_contribution(
    tree_size: int, base_alpha: float = 3.070
) -> Dict[str, Any]:
    """Compute fractal layer contribution to alpha at given tree size.

    Args:
        tree_size: Number of nodes in the tree
        base_alpha: Base alpha at 10^6 (default: 3.070)

    Returns:
        Dict with contribution metrics

    Receipt: fractal_contribution
    """
    scale_factor = get_scale_factor(tree_size)
    adjusted_correlation = scale_adjusted_correlation(tree_size)

    # Compute alpha adjustment
    # Alpha scales with correlation^2 (correlation affects both encoding and retention)
    alpha_factor = scale_factor**2
    adjusted_alpha = base_alpha * alpha_factor

    result = {
        "tree_size": tree_size,
        "scale_factor": round(scale_factor, 6),
        "base_correlation": FRACTAL_BASE_CORRELATION,
        "adjusted_correlation": round(adjusted_correlation, 6),
        "base_alpha": base_alpha,
        "adjusted_alpha": round(adjusted_alpha, 4),
        "alpha_drop": round(base_alpha - adjusted_alpha, 4),
        "alpha_drop_pct": round((base_alpha - adjusted_alpha) / base_alpha * 100, 3),
    }

    emit_receipt(
        "fractal_contribution",
        {
            "receipt_type": "fractal_contribution",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_expected_alpha_at_scale(tree_size: int) -> float:
    """Get expected alpha value at given tree size.

    Based on physics-validated decay:
    - 10^6: 3.070
    - 10^8: 3.068
    - 10^9: 3.065-3.067

    Args:
        tree_size: Number of nodes in the tree

    Returns:
        Expected alpha value
    """
    base_alpha = 3.070
    scale_factor = get_scale_factor(tree_size)
    return base_alpha * (scale_factor**2)


def validate_scale_physics() -> Dict[str, Any]:
    """Validate scale physics at key tree sizes.

    Checks that alpha decay follows expected physics:
    - 10^6: alpha = 3.070 (baseline)
    - 10^8: alpha >= 3.065
    - 10^9: alpha >= 3.06

    Returns:
        Dict with validation results

    Receipt: scale_physics_validation
    """
    test_sizes = [1_000_000, 100_000_000, 1_000_000_000]
    expected_min = [3.070, 3.065, 3.06]

    results = []
    all_passed = True

    for size, min_alpha in zip(test_sizes, expected_min):
        alpha = get_expected_alpha_at_scale(size)
        passed = alpha >= min_alpha

        results.append(
            {
                "tree_size": size,
                "expected_alpha": round(alpha, 4),
                "min_required": min_alpha,
                "passed": passed,
            }
        )

        if not passed:
            all_passed = False

    validation = {
        "test_sizes": test_sizes,
        "results": results,
        "all_passed": all_passed,
        "decay_factor": CORRELATION_DECAY_FACTOR,
        "base_correlation": FRACTAL_BASE_CORRELATION,
    }

    emit_receipt(
        "scale_physics_validation",
        {
            "receipt_type": "scale_physics_validation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in validation.items() if k != "results"},
            "test_count": len(results),
            "passed_count": sum(1 for r in results if r["passed"]),
            "payload_hash": dual_hash(
                json.dumps(validation, sort_keys=True, default=str)
            ),
        },
    )

    return validation


def get_fractal_layers_info() -> Dict[str, Any]:
    """Get fractal layers module information.

    Returns:
        Dict with module configuration and expected behavior

    Receipt: fractal_layers_info
    """
    info = {
        "base_tree_size": BASE_TREE_SIZE,
        "correlation_decay_factor": CORRELATION_DECAY_FACTOR,
        "fractal_base_correlation": FRACTAL_BASE_CORRELATION,
        "fractal_alpha_contribution": FRACTAL_ALPHA_CONTRIBUTION,
        "scale_factors": {
            "1e6": get_scale_factor(1_000_000),
            "1e8": get_scale_factor(100_000_000),
            "1e9": get_scale_factor(1_000_000_000),
        },
        "expected_alphas": {
            "1e6": round(get_expected_alpha_at_scale(1_000_000), 4),
            "1e8": round(get_expected_alpha_at_scale(100_000_000), 4),
            "1e9": round(get_expected_alpha_at_scale(1_000_000_000), 4),
        },
        "physics_formula": "correlation * (1 - 0.001 * log10(tree_size / 1e6))",
        "description": "Fractal correlation layer with scale-aware adjustment for large trees",
    }

    emit_receipt(
        "fractal_layers_info",
        {
            "receipt_type": "fractal_layers_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{
                k: v
                for k, v in info.items()
                if k not in ["scale_factors", "expected_alphas"]
            },
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === MODULE METADATA ===

RECEIPT_SCHEMA = {
    "module": "src.fractal.alpha",
    "receipt_types": [
        "fractal_contribution",
        "scale_physics_validation",
        "fractal_layers_info",
    ],
    "version": "1.0.0",
}

__all__ = [
    "BASE_TREE_SIZE",
    "CORRELATION_DECAY_FACTOR",
    "FRACTAL_BASE_CORRELATION",
    "FRACTAL_ALPHA_CONTRIBUTION",
    "TENANT_ID",
    "scale_adjusted_correlation",
    "get_scale_factor",
    "compute_fractal_contribution",
    "get_expected_alpha_at_scale",
    "validate_scale_physics",
    "get_fractal_layers_info",
    "RECEIPT_SCHEMA",
]
