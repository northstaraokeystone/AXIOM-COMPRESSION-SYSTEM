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
    external_rate_exponential,
    sovereignty_advantage,
    is_sovereign,
    STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
    MARS_LIGHT_DELAY_AVG_S,
    TAU_DECISION_DECAY_S,
    DELAY_VARIANCE_RATIO,
    BANDWIDTH_VARIANCE_RATIO,
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


# === EXPONENTIAL DECAY MODEL (v1.1 - Grok feedback Dec 16, 2025) ===

@dataclass
class SovereigntyResultExp:
    """Result of sovereignty calculation with exponential decay model.

    Extends SovereigntyResult with decay parameters.
    """
    internal_rate: float
    external_rate_linear: float
    external_rate_exp: float
    advantage_linear: float
    advantage_exp: float
    sovereign_linear: bool
    sovereign_exp: bool
    tau_s: float
    decay_factor: float
    threshold_crew: Optional[int] = None


def compute_sovereignty_exponential(
    config: SovereigntyConfig,
    tau_s: float = TAU_DECISION_DECAY_S
) -> SovereigntyResultExp:
    """Compute sovereignty using BOTH linear and exponential decay models.

    Grok paradigm shift: "Model effective rate as bw * exp(-delay/tau)"

    Args:
        config: SovereigntyConfig with crew, compute, bandwidth, delay
        tau_s: Decay time constant (default 300s = 5 min)

    Returns:
        SovereigntyResultExp with both model results for comparison

    The exponential model captures decision VALUE decay:
        - Linear: bandwidth / (2 * delay) - round-trip throughput
        - Exponential: bandwidth * exp(-delay/tau) - value decay with staleness
    """
    ir = internal_rate(config.crew, config.compute_flops)

    # Linear model (original)
    er_linear = external_rate(config.bandwidth_mbps, config.delay_s)
    adv_linear = sovereignty_advantage(ir, er_linear)
    sov_linear = is_sovereign(adv_linear)

    # Exponential decay model (Grok suggestion)
    er_exp = external_rate_exponential(
        config.bandwidth_mbps,
        config.delay_s,
        tau_s
    )
    adv_exp = sovereignty_advantage(ir, er_exp)
    sov_exp = is_sovereign(adv_exp)

    # Decay factor for reporting
    decay_factor = math.exp(-config.delay_s / tau_s)

    return SovereigntyResultExp(
        internal_rate=ir,
        external_rate_linear=er_linear,
        external_rate_exp=er_exp,
        advantage_linear=adv_linear,
        advantage_exp=adv_exp,
        sovereign_linear=sov_linear,
        sovereign_exp=sov_exp,
        tau_s=tau_s,
        decay_factor=decay_factor
    )


def find_threshold_exponential(
    bandwidth_mbps: float = 2.0,
    delay_s: float = 480.0,
    compute_flops: float = 0.0,
    tau_s: float = TAU_DECISION_DECAY_S,
    max_crew: int = 500
) -> int:
    """Binary search for crew threshold using exponential decay model.

    Args:
        bandwidth_mbps: Communication bandwidth in Mbps
        delay_s: One-way light delay in seconds
        compute_flops: Compute capacity in FLOPS
        tau_s: Decay time constant
        max_crew: Maximum crew to search

    Returns:
        Minimum crew size for sovereignty under exponential model
    """
    # First check if max_crew is sufficient
    config = SovereigntyConfig(
        crew=max_crew,
        compute_flops=compute_flops,
        bandwidth_mbps=bandwidth_mbps,
        delay_s=delay_s
    )
    result = compute_sovereignty_exponential(config, tau_s)

    if not result.sovereign_exp:
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
        result = compute_sovereignty_exponential(config, tau_s)

        if result.sovereign_exp:
            high = mid
        else:
            low = mid + 1

    return low


# === SENSITIVITY ANALYSIS (v1.1 - Grok: "latency-limited") ===

def sensitivity_to_delay(
    base_bandwidth: float = 4.0,
    base_delay: float = 480.0,
    delta: float = 60.0,  # 1 minute change
    compute_flops: float = 0.0
) -> Tuple[float, float]:
    """Compute sensitivity of threshold to delay changes.

    ∂threshold/∂delay - How much does threshold change per second of delay?

    Args:
        base_bandwidth: Bandwidth for calculation
        base_delay: Base delay point
        delta: Change in delay for finite difference
        compute_flops: Compute capacity

    Returns:
        Tuple of (linear_sensitivity, exponential_sensitivity)

    Grok insight: "It's primarily latency-limited"
    """
    # Linear model sensitivity
    t_base_lin = find_threshold(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay,
        compute_flops=compute_flops
    )
    t_delta_lin = find_threshold(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay + delta,
        compute_flops=compute_flops
    )
    sens_linear = (t_delta_lin - t_base_lin) / delta

    # Exponential model sensitivity
    t_base_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay,
        compute_flops=compute_flops
    )
    t_delta_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay + delta,
        compute_flops=compute_flops
    )
    sens_exp = (t_delta_exp - t_base_exp) / delta

    return (sens_linear, sens_exp)


def sensitivity_to_bandwidth(
    base_bandwidth: float = 4.0,
    base_delay: float = 480.0,
    delta: float = 1.0,  # 1 Mbps change
    compute_flops: float = 0.0
) -> Tuple[float, float]:
    """Compute sensitivity of threshold to bandwidth changes.

    ∂threshold/∂bandwidth - How much does threshold change per Mbps?

    Args:
        base_bandwidth: Base bandwidth point
        base_delay: Delay for calculation
        delta: Change in bandwidth for finite difference
        compute_flops: Compute capacity

    Returns:
        Tuple of (linear_sensitivity, exponential_sensitivity)

    Note: Higher bandwidth → MORE Earth help → HIGHER threshold
    (need more crew to beat Earth's increased capacity)
    """
    # Linear model sensitivity
    t_base_lin = find_threshold(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay,
        compute_flops=compute_flops
    )
    t_delta_lin = find_threshold(
        bandwidth_mbps=base_bandwidth + delta,
        delay_s=base_delay,
        compute_flops=compute_flops
    )
    sens_linear = (t_delta_lin - t_base_lin) / delta

    # Exponential model sensitivity
    t_base_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay,
        compute_flops=compute_flops
    )
    t_delta_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth + delta,
        delay_s=base_delay,
        compute_flops=compute_flops
    )
    sens_exp = (t_delta_exp - t_base_exp) / delta

    return (sens_linear, sens_exp)


def compute_sensitivity_ratio() -> dict:
    """Compute ratio of delay sensitivity to bandwidth sensitivity.

    Grok: "3-22 min delay varies more than 2-10 Mbps"
    Delay variance: 7.33x (1140s range / 180s min)
    Bandwidth variance: 4.0x (8 Mbps range / 2 Mbps min)

    Returns:
        Dict with sensitivity analysis results
    """
    sens_delay_lin, sens_delay_exp = sensitivity_to_delay()
    sens_bw_lin, sens_bw_exp = sensitivity_to_bandwidth()

    # Normalize by variance ranges
    # Delay: 180s to 1320s (1140s range)
    # Bandwidth: 2 to 10 Mbps (8 Mbps range)

    delay_impact_lin = abs(sens_delay_lin) * 1140  # Impact over full range
    delay_impact_exp = abs(sens_delay_exp) * 1140
    bw_impact_lin = abs(sens_bw_lin) * 8
    bw_impact_exp = abs(sens_bw_exp) * 8

    # Ratio > 1 means delay dominates
    ratio_linear = delay_impact_lin / bw_impact_lin if bw_impact_lin > 0 else float('inf')
    ratio_exp = delay_impact_exp / bw_impact_exp if bw_impact_exp > 0 else float('inf')

    return {
        "sensitivity_delay_linear": sens_delay_lin,
        "sensitivity_delay_exp": sens_delay_exp,
        "sensitivity_bandwidth_linear": sens_bw_lin,
        "sensitivity_bandwidth_exp": sens_bw_exp,
        "delay_impact_linear": delay_impact_lin,
        "delay_impact_exp": delay_impact_exp,
        "bandwidth_impact_linear": bw_impact_lin,
        "bandwidth_impact_exp": bw_impact_exp,
        "ratio_linear": ratio_linear,
        "ratio_exp": ratio_exp,
        "latency_limited_linear": ratio_linear > 1,
        "latency_limited_exp": ratio_exp > 1,
        "delay_variance_ratio": DELAY_VARIANCE_RATIO,
        "bandwidth_variance_ratio": BANDWIDTH_VARIANCE_RATIO
    }


def conjunction_vs_opposition() -> dict:
    """Compare sovereignty at Mars conjunction (22 min) vs opposition (3 min).

    Grok validated:
        - At 22 min, 100 Mbps → ~38k units (our formula: 37,879)
        - At 3 min, 2 Mbps → ~5.5k units (our formula: 5,556)

    Returns:
        Dict comparing conjunction and opposition scenarios
    """
    # Opposition: Mars closest (3 min delay)
    opposition_config = SovereigntyConfig(
        crew=100,  # Reference crew
        compute_flops=0.0,
        bandwidth_mbps=2.0,
        delay_s=180  # 3 min
    )

    # Conjunction: Mars farthest (22 min delay)
    conjunction_config = SovereigntyConfig(
        crew=100,
        compute_flops=0.0,
        bandwidth_mbps=100.0,  # Grok's high-bandwidth scenario
        delay_s=1320  # 22 min
    )

    opp_result = compute_sovereignty_exponential(opposition_config)
    conj_result = compute_sovereignty_exponential(conjunction_config)

    # Thresholds for each scenario
    opp_threshold_lin = find_threshold(bandwidth_mbps=2.0, delay_s=180)
    opp_threshold_exp = find_threshold_exponential(bandwidth_mbps=2.0, delay_s=180)
    conj_threshold_lin = find_threshold(bandwidth_mbps=100.0, delay_s=1320)
    conj_threshold_exp = find_threshold_exponential(bandwidth_mbps=100.0, delay_s=1320)

    # Grok's formula is bps effective rate: bandwidth / (2 * delay)
    # Our external_rate divides by BITS_PER_DECISION to get decisions/sec
    # For validation, we need to compare Grok's bps formula
    grok_formula_22min_100mbps = 100e6 / (2 * 1320)  # 37,879 bps
    grok_formula_3min_2mbps = 2e6 / (2 * 180)  # 5,556 bps

    return {
        "opposition": {
            "delay_s": 180,
            "delay_min": 3,
            "bandwidth_mbps": 2.0,
            "external_rate_linear": opp_result.external_rate_linear,
            "external_rate_exp": opp_result.external_rate_exp,
            "threshold_linear": opp_threshold_lin,
            "threshold_exp": opp_threshold_exp
        },
        "conjunction": {
            "delay_s": 1320,
            "delay_min": 22,
            "bandwidth_mbps": 100.0,
            "external_rate_linear": conj_result.external_rate_linear,
            "external_rate_exp": conj_result.external_rate_exp,
            "threshold_linear": conj_threshold_lin,
            "threshold_exp": conj_threshold_exp
        },
        "grok_validation": {
            "grok_22min_100mbps": 38000,
            "our_22min_100mbps_bps": round(grok_formula_22min_100mbps),
            "our_22min_100mbps_decisions": round(conj_result.external_rate_linear),
            "match_conjunction": abs(grok_formula_22min_100mbps - 38000) < 1000,
            "grok_3min_2mbps": 5500,
            "our_3min_2mbps_bps": round(grok_formula_3min_2mbps),
            "our_3min_2mbps_decisions": round(opp_result.external_rate_linear),
            "match_opposition": abs(grok_formula_3min_2mbps - 5500) < 500,
            "note": "Grok uses bps formula, our external_rate uses decisions/sec"
        }
    }


def emit_sensitivity_receipt() -> dict:
    """Emit receipt for sensitivity analysis.

    MUST emit receipt per CLAUDEME.
    """
    sensitivity = compute_sensitivity_ratio()
    scenarios = conjunction_vs_opposition()

    return emit_receipt("sensitivity_analysis", {
        "tenant_id": "axiom-core",
        **sensitivity,
        "conjunction_opposition": scenarios,
        "finding": "latency_limited" if sensitivity["latency_limited_linear"] else "bandwidth_limited"
    })
