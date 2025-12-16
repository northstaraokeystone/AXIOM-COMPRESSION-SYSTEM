"""sovereignty.py - The Core Equation

THE PEARL:
    sovereignty = internal_rate > external_rate

One equation. One curve. One number.

This module implements the sovereignty equation and threshold finding.
No speculative enhancements. No multi-body scaling. Just the math.

Source: Critical Review Dec 16, 2025 - "The equation is the pearl."
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from .core import emit_receipt
from .entropy_shannon import (
    internal_rate,
    external_rate,
    sovereignty_advantage,
    is_sovereign,
    STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
    MARS_LIGHT_DELAY_AVG_S,
)


@dataclass
class SovereigntyConfig:
    """Configuration for sovereignty calculation.

    Attributes:
        crew: Number of crew members
        compute_flops: Compute capacity in FLOPS (default 0)
        bandwidth_mbps: Communication bandwidth (default 2.0 Mbps minimum)
        delay_s: One-way light delay (default 480s = 8 min average)
    """
    crew: int
    compute_flops: float = 0.0
    bandwidth_mbps: float = 2.0
    delay_s: float = 480.0


@dataclass
class SovereigntyResult:
    """Result of sovereignty calculation.

    Attributes:
        internal_rate: Internal decision rate (bits/sec)
        external_rate: External decision rate (bits/sec)
        advantage: internal - external
        sovereign: True if advantage > 0
        threshold_crew: Crew where advantage crosses zero (if computed)
    """
    internal_rate: float
    external_rate: float
    advantage: float
    sovereign: bool
    threshold_crew: Optional[int] = None


def compute_sovereignty(config: SovereigntyConfig) -> SovereigntyResult:
    """THE core equation. Compute sovereignty for given configuration.

    sovereignty = internal_rate > external_rate

    Args:
        config: SovereigntyConfig with crew, compute, bandwidth, delay

    Returns:
        SovereigntyResult with rates, advantage, and sovereignty status

    The Equation:
        internal = log2(1 + crew * 10 + compute_flops * 1e-15)
        external = (bandwidth_mbps * 1e6) / (2 * delay_s)
        advantage = internal - external
        sovereign = advantage > 0
    """
    ir = internal_rate(config.crew, config.compute_flops)
    er = external_rate(config.bandwidth_mbps, config.delay_s)
    adv = sovereignty_advantage(ir, er)
    sov = is_sovereign(adv)

    return SovereigntyResult(
        internal_rate=ir,
        external_rate=er,
        advantage=adv,
        sovereign=sov
    )


def find_threshold(
    bandwidth_mbps: float = 2.0,
    delay_s: float = 480.0,
    compute_flops: float = 0.0,
    max_crew: int = 500
) -> int:
    """Binary search for crew where sovereign=True.

    Args:
        bandwidth_mbps: Communication bandwidth in Mbps
        delay_s: One-way light delay in seconds
        compute_flops: Compute capacity in FLOPS (default 0)
        max_crew: Maximum crew to search (default 500)

    Returns:
        Minimum crew size for sovereignty (advantage > 0)

    Algorithm:
        Binary search in [1, max_crew] for smallest crew where
        compute_sovereignty(config).sovereign == True
    """
    # First check if max_crew is sufficient
    config = SovereigntyConfig(
        crew=max_crew,
        compute_flops=compute_flops,
        bandwidth_mbps=bandwidth_mbps,
        delay_s=delay_s
    )
    result = compute_sovereignty(config)

    if not result.sovereign:
        # Even max_crew isn't enough
        return max_crew + 1

    # Binary search
    low, high = 1, max_crew

    while low < high:
        mid = (low + high) // 2
        config = SovereigntyConfig(
            crew=mid,
            compute_flops=compute_flops,
            bandwidth_mbps=bandwidth_mbps,
            delay_s=delay_s
        )
        result = compute_sovereignty(config)

        if result.sovereign:
            high = mid
        else:
            low = mid + 1

    return low


def sensitivity_analysis(
    param: str,
    range_values: Tuple[float, float],
    steps: int = 20,
    base_bandwidth: float = 2.0,
    base_delay: float = 480.0,
    base_compute: float = 0.0
) -> List[Tuple[float, int]]:
    """Vary one parameter, return (param_value, threshold) pairs.

    Args:
        param: Parameter to vary ("bandwidth", "delay", "compute")
        range_values: (min, max) for the parameter
        steps: Number of steps to evaluate
        base_bandwidth: Default bandwidth for non-varied params
        base_delay: Default delay for non-varied params
        base_compute: Default compute for non-varied params

    Returns:
        List of (param_value, threshold_crew) tuples

    Example:
        sensitivity_analysis("bandwidth", (1.0, 20.0), steps=10)
        -> [(1.0, 65), (3.1, 52), (5.2, 48), ...]
    """
    results = []

    min_val, max_val = range_values
    step_size = (max_val - min_val) / (steps - 1) if steps > 1 else 0

    for i in range(steps):
        val = min_val + i * step_size

        if param == "bandwidth":
            threshold = find_threshold(
                bandwidth_mbps=val,
                delay_s=base_delay,
                compute_flops=base_compute
            )
        elif param == "delay":
            threshold = find_threshold(
                bandwidth_mbps=base_bandwidth,
                delay_s=val,
                compute_flops=base_compute
            )
        elif param == "compute":
            threshold = find_threshold(
                bandwidth_mbps=base_bandwidth,
                delay_s=base_delay,
                compute_flops=val
            )
        else:
            raise ValueError(f"Unknown parameter: {param}")

        results.append((val, threshold))

    return results


def emit_sovereignty_receipt(config: SovereigntyConfig) -> dict:
    """Emit receipt for sovereignty calculation.

    MUST emit receipt per CLAUDEME.
    """
    result = compute_sovereignty(config)

    return emit_receipt("sovereignty_calculation", {
        "tenant_id": "axiom-core",
        "crew": config.crew,
        "compute_flops": config.compute_flops,
        "bandwidth_mbps": config.bandwidth_mbps,
        "delay_s": config.delay_s,
        "internal_rate": result.internal_rate,
        "external_rate": result.external_rate,
        "advantage": result.advantage,
        "sovereign": result.sovereign
    })
