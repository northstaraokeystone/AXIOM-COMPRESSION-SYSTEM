"""fractal/adaptive.py - Adaptive Termination for Fractal Recursion

This module provides adaptive termination functionality for fractal recursion.
Used in D14+ to stop recursion when diminishing returns are detected.

ADAPTIVE TERMINATION:
    Stops recursion when delta between iterations falls below threshold.
    Prevents unnecessary computation when uplift gains become negligible.
"""


# === ADAPTIVE TERMINATION CONSTANTS ===

D14_TERMINATION_THRESHOLD = 0.001
"""D14 adaptive termination threshold."""


# === ADAPTIVE TERMINATION FUNCTIONS ===


def adaptive_termination_check(
    current: float, previous: float, threshold: float = D14_TERMINATION_THRESHOLD
) -> bool:
    """Check if adaptive termination condition is met.

    Adaptive termination stops recursion when delta between iterations
    falls below threshold, indicating diminishing returns.

    Args:
        current: Current alpha value
        previous: Previous alpha value
        threshold: Termination threshold (default: 0.001)

    Returns:
        True if termination condition met (delta < threshold)
    """
    delta = abs(current - previous)
    return delta < threshold


# === MODULE METADATA ===

RECEIPT_SCHEMA = {
    "module": "src.fractal.adaptive",
    "receipt_types": [],
    "version": "1.0.0",
}

__all__ = [
    "D14_TERMINATION_THRESHOLD",
    "adaptive_termination_check",
    "RECEIPT_SCHEMA",
]
