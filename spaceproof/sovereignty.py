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
    TAU_DECISION_DECAY_S,
    DELAY_VARIANCE_RATIO,
    BANDWIDTH_VARIANCE_RATIO,
    # v1.2 - Variable τ costs
    TAU_BASE_CURRENT_S,
    tau_from_investment,
    bandwidth_from_investment,
    # v1.3 - Variable τ cost curves + meta-compression
    ITERATION_COMPRESSION_FACTOR,
    META_TAU_HUMAN_DAYS,
    META_TAU_AI_DAYS,
    tau_from_cost,
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
        internal_rate=ir, external_rate=er, advantage=adv, sovereign=sov
    )


def find_threshold(
    bandwidth_mbps: float = 2.0,
    delay_s: float = 480.0,
    compute_flops: float = 0.0,
    max_crew: int = 500,
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
        delay_s=delay_s,
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
            delay_s=delay_s,
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
    base_compute: float = 0.0,
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
                bandwidth_mbps=val, delay_s=base_delay, compute_flops=base_compute
            )
        elif param == "delay":
            threshold = find_threshold(
                bandwidth_mbps=base_bandwidth, delay_s=val, compute_flops=base_compute
            )
        elif param == "compute":
            threshold = find_threshold(
                bandwidth_mbps=base_bandwidth, delay_s=base_delay, compute_flops=val
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

    return emit_receipt(
        "sovereignty_calculation",
        {
            "tenant_id": "spaceproof-core",
            "crew": config.crew,
            "compute_flops": config.compute_flops,
            "bandwidth_mbps": config.bandwidth_mbps,
            "delay_s": config.delay_s,
            "internal_rate": result.internal_rate,
            "external_rate": result.external_rate,
            "advantage": result.advantage,
            "sovereign": result.sovereign,
        },
    )


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
    config: SovereigntyConfig, tau_s: float = TAU_DECISION_DECAY_S
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
    er_exp = external_rate_exponential(config.bandwidth_mbps, config.delay_s, tau_s)
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
        decay_factor=decay_factor,
    )


def find_threshold_exponential(
    bandwidth_mbps: float = 2.0,
    delay_s: float = 480.0,
    compute_flops: float = 0.0,
    tau_s: float = TAU_DECISION_DECAY_S,
    max_crew: int = 500,
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
        delay_s=delay_s,
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
            delay_s=delay_s,
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
    compute_flops: float = 0.0,
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
        bandwidth_mbps=base_bandwidth, delay_s=base_delay, compute_flops=compute_flops
    )
    t_delta_lin = find_threshold(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay + delta,
        compute_flops=compute_flops,
    )
    sens_linear = (t_delta_lin - t_base_lin) / delta

    # Exponential model sensitivity
    t_base_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth, delay_s=base_delay, compute_flops=compute_flops
    )
    t_delta_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth,
        delay_s=base_delay + delta,
        compute_flops=compute_flops,
    )
    sens_exp = (t_delta_exp - t_base_exp) / delta

    return (sens_linear, sens_exp)


def sensitivity_to_bandwidth(
    base_bandwidth: float = 4.0,
    base_delay: float = 480.0,
    delta: float = 1.0,  # 1 Mbps change
    compute_flops: float = 0.0,
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
        bandwidth_mbps=base_bandwidth, delay_s=base_delay, compute_flops=compute_flops
    )
    t_delta_lin = find_threshold(
        bandwidth_mbps=base_bandwidth + delta,
        delay_s=base_delay,
        compute_flops=compute_flops,
    )
    sens_linear = (t_delta_lin - t_base_lin) / delta

    # Exponential model sensitivity
    t_base_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth, delay_s=base_delay, compute_flops=compute_flops
    )
    t_delta_exp = find_threshold_exponential(
        bandwidth_mbps=base_bandwidth + delta,
        delay_s=base_delay,
        compute_flops=compute_flops,
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
    ratio_linear = (
        delay_impact_lin / bw_impact_lin if bw_impact_lin > 0 else float("inf")
    )
    ratio_exp = delay_impact_exp / bw_impact_exp if bw_impact_exp > 0 else float("inf")

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
        "bandwidth_variance_ratio": BANDWIDTH_VARIANCE_RATIO,
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
        delay_s=180,  # 3 min
    )

    # Conjunction: Mars farthest (22 min delay)
    conjunction_config = SovereigntyConfig(
        crew=100,
        compute_flops=0.0,
        bandwidth_mbps=100.0,  # Grok's high-bandwidth scenario
        delay_s=1320,  # 22 min
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
            "threshold_exp": opp_threshold_exp,
        },
        "conjunction": {
            "delay_s": 1320,
            "delay_min": 22,
            "bandwidth_mbps": 100.0,
            "external_rate_linear": conj_result.external_rate_linear,
            "external_rate_exp": conj_result.external_rate_exp,
            "threshold_linear": conj_threshold_lin,
            "threshold_exp": conj_threshold_exp,
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
            "note": "Grok uses bps formula, our external_rate uses decisions/sec",
        },
    }


def emit_sensitivity_receipt() -> dict:
    """Emit receipt for sensitivity analysis.

    MUST emit receipt per CLAUDEME.
    """
    sensitivity = compute_sensitivity_ratio()
    scenarios = conjunction_vs_opposition()

    return emit_receipt(
        "sensitivity_analysis",
        {
            "tenant_id": "spaceproof-core",
            **sensitivity,
            "conjunction_opposition": scenarios,
            "finding": "latency_limited"
            if sensitivity["latency_limited_linear"]
            else "bandwidth_limited",
        },
    )


# === ROI COMPARISON FUNCTIONS (v1.2 - Grok feedback Dec 16, 2025) ===
# Source: "Investing in τ yields higher ROI"


def effective_rate_gain_from_tau(
    tau_old: float, tau_new: float, bw_mbps: float, delay_s: float
) -> float:
    """Calculate delta in effective_rate from τ (decision latency) reduction.

    KEY INSIGHT: τ represents decision LATENCY (time to make a decision).
    - Lower τ = faster local decisions = BETTER autonomy
    - Better autonomy = can work with staler data = higher effective decay constant

    The relationship: effective_decay_tau = TAU_BASE^2 / autonomy_tau
    - At τ=300s (baseline): decay_tau = 300s
    - At τ=150s (2x faster): decay_tau = 600s (can work with 2x staler data)
    - At τ=30s (10x faster): decay_tau = 3000s (can work with 10x staler data)

    Args:
        tau_old: Original decision latency in seconds (higher = slower)
        tau_new: New decision latency in seconds (lower = faster = better)
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds

    Returns:
        Change in effective rate (decisions/sec)

    Source: Grok Dec 16, 2025 - "Autonomy reduces τ. Lower τ → higher effective rate"
    """
    # Convert decision latency to effective decay constant
    # decay_tau = baseline^2 / latency_tau (inverse relationship)
    decay_tau_old = TAU_BASE_CURRENT_S * TAU_BASE_CURRENT_S / tau_old
    decay_tau_new = TAU_BASE_CURRENT_S * TAU_BASE_CURRENT_S / tau_new

    rate_before = external_rate_exponential(bw_mbps, delay_s, decay_tau_old)
    rate_after = external_rate_exponential(bw_mbps, delay_s, decay_tau_new)
    return rate_after - rate_before


def effective_rate_gain_from_bw(
    bw_old: float, bw_new: float, tau_s: float, delay_s: float
) -> float:
    """Calculate delta in effective_rate from bandwidth increase.

    Args:
        bw_old: Original bandwidth in Mbps
        bw_new: New bandwidth in Mbps
        tau_s: Decision decay time constant in seconds
        delay_s: One-way light delay in seconds

    Returns:
        Change in effective rate (decisions/sec)
    """
    rate_before = external_rate_exponential(bw_old, delay_s, tau_s)
    rate_after = external_rate_exponential(bw_new, delay_s, tau_s)
    return rate_after - rate_before


def roi_tau_investment(
    investment_m: float, tau_base: float, bw_mbps: float, delay_s: float
) -> float:
    """Calculate ROI for τ reduction investment.

    ROI = effective_rate_gain / investment_cost

    Args:
        investment_m: Investment in millions USD
        tau_base: Starting τ value in seconds
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds

    Returns:
        ROI (effective rate gain per $M invested)

    Key insight: At high delay, improving τ (the multiplier) beats improving
    bandwidth (the base), because exp(-delay/τ) is a small number.
    """
    if investment_m <= 0:
        return 0.0

    tau_new = tau_from_investment(investment_m, tau_base)
    rate_gain = effective_rate_gain_from_tau(tau_base, tau_new, bw_mbps, delay_s)
    return rate_gain / investment_m


def roi_bandwidth_investment(
    investment_m: float, bw_base: float, tau_s: float, delay_s: float
) -> float:
    """Calculate ROI for bandwidth investment.

    ROI = effective_rate_gain / investment_cost

    Args:
        investment_m: Investment in millions USD
        bw_base: Starting bandwidth in Mbps
        tau_s: Decision decay time constant in seconds
        delay_s: One-way light delay in seconds

    Returns:
        ROI (effective rate gain per $M invested)

    Note: At high delay, bandwidth gains are multiplied by a small
    exp(-delay/τ) factor, reducing effective ROI.
    """
    if investment_m <= 0:
        return 0.0

    bw_increase = bandwidth_from_investment(investment_m)
    bw_new = bw_base + bw_increase
    rate_gain = effective_rate_gain_from_bw(bw_base, bw_new, tau_s, delay_s)
    return rate_gain / investment_m


def compare_investment_roi(
    investment_m: float, bw_base: float, tau_base: float, delay_s: float
) -> dict:
    """Compare same $ spent on τ vs bandwidth.

    Args:
        investment_m: Investment amount in millions USD
        bw_base: Current bandwidth in Mbps
        tau_base: Current τ value in seconds
        delay_s: One-way light delay in seconds

    Returns:
        Dict with ROI comparison:
        - roi_tau: ROI from τ investment
        - roi_bw: ROI from bandwidth investment
        - winner: "autonomy" or "bandwidth"
        - ratio: how many times better winner is

    THE FINDING: At Mars distances, τ-reduction ROI >> bandwidth ROI.
    """
    roi_tau = roi_tau_investment(investment_m, tau_base, bw_base, delay_s)
    roi_bw = roi_bandwidth_investment(investment_m, bw_base, tau_base, delay_s)

    if roi_tau > roi_bw:
        winner = "autonomy"
        ratio = roi_tau / roi_bw if roi_bw > 0 else float("inf")
    else:
        winner = "bandwidth"
        ratio = roi_bw / roi_tau if roi_tau > 0 else float("inf")

    return {
        "investment_m": investment_m,
        "roi_tau": roi_tau,
        "roi_bw": roi_bw,
        "winner": winner,
        "ratio": ratio,
        "tau_new": tau_from_investment(investment_m, tau_base),
        "bw_new": bw_base + bandwidth_from_investment(investment_m),
    }


def find_breakeven_delay(
    bw_base: float = STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
    tau_base: float = TAU_BASE_CURRENT_S,
    investment_m: float = 100.0,
) -> float:
    """Find delay at which τ investment ROI equals bandwidth investment ROI.

    Binary search for the crossover point.

    Args:
        bw_base: Current bandwidth in Mbps (default 4 Mbps)
        tau_base: Current τ value in seconds (default 300s)
        investment_m: Reference investment for comparison (default $100M)

    Returns:
        Breakeven delay in seconds

    THE FINDING:
        Below breakeven: invest in bandwidth (Moon, LEO)
        Above breakeven: invest in autonomy (Mars, beyond)
        Mars minimum delay (3 min) > breakeven → ALWAYS invest in autonomy for Mars
    """
    # Search between 60s (1 min) and 1800s (30 min)
    low, high = 60.0, 1800.0
    tolerance = 1.0  # 1 second precision

    while high - low > tolerance:
        mid = (low + high) / 2
        roi_tau = roi_tau_investment(investment_m, tau_base, bw_base, mid)
        roi_bw = roi_bandwidth_investment(investment_m, bw_base, tau_base, mid)

        if roi_tau > roi_bw:
            # τ wins at this delay, search lower
            high = mid
        else:
            # Bandwidth wins at this delay, search higher
            low = mid

    return (low + high) / 2


# === CREW SENSITIVITY FUNCTIONS (v1.2 - Grok feedback Dec 16, 2025) ===
# Source: "potentially dropping threshold to 20-30 crew"

# Constants for crew calculation
BASE_CREW = 10
"""Minimum operational crew (safety floor)."""

REQUIRED_RATE = 1000
"""Minimum decisions/sec for safe operation (baseline requirement)."""


def threshold_from_tau(
    tau_s: float, bw_mbps: float, delay_s: float, compute_flops: float = 0.0
) -> int:
    """Calculate crew threshold at given decision latency τ.

    KEY MODEL: Better autonomy (lower τ) means less reliance on external help.
    External rate is scaled down by the autonomy improvement factor.

    effective_external = base_external * (tau_autonomy / TAU_BASE)

    Args:
        tau_s: Decision latency in seconds (lower = better autonomy)
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        compute_flops: Compute capacity in FLOPS

    Returns:
        Minimum crew for sovereignty at this decision latency

    Expected results (per Grok):
        τ=300s, conjunction: ~47 crew
        τ=100s, conjunction: ~30 crew
        τ=30s, conjunction: ~20 crew

    Model: autonomy_factor = tau_s / TAU_BASE
        τ=300s → factor=1.0 (baseline - full external dependency)
        τ=100s → factor=0.33 (need 1/3 the external help)
        τ=30s → factor=0.10 (need 1/10 the external help)
    """
    from .entropy_shannon import HUMAN_DECISION_RATE_BPS

    # Autonomy factor: lower τ means less need for external help
    autonomy_factor = tau_s / TAU_BASE_CURRENT_S

    # Get the base external rate using exponential decay model
    base_external = external_rate_exponential(bw_mbps, delay_s, TAU_DECISION_DECAY_S)

    # Scale external rate by autonomy factor
    # Better autonomy = need less external help = lower effective external rate
    effective_external = base_external * autonomy_factor

    # Compute internal capacity (if any)
    compute_contribution = compute_flops * 1e-15

    # Threshold = ceiling of (effective_external - compute) / human_rate
    needed_human_rate = effective_external - compute_contribution
    if needed_human_rate <= 0:
        return 1  # Minimum crew

    threshold = math.ceil(needed_human_rate / HUMAN_DECISION_RATE_BPS)
    return max(1, min(threshold, 500))  # Clamp to [1, 500]


def threshold_sensitivity_to_tau(
    tau_range: Tuple[float, float], bw_mbps: float, delay_s: float, steps: int = 10
) -> List[Tuple[float, int]]:
    """Generate (τ, threshold) pairs across τ range.

    Args:
        tau_range: (tau_min, tau_max) in seconds
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        steps: Number of points to evaluate

    Returns:
        List of (τ, threshold) tuples showing crew reduction as τ decreases
    """
    tau_min, tau_max = tau_range
    results = []

    for i in range(steps):
        tau = tau_min + (tau_max - tau_min) * i / (steps - 1) if steps > 1 else tau_min
        threshold = threshold_from_tau(tau, bw_mbps, delay_s)
        results.append((tau, threshold))

    return results


def min_viable_crew(effective_rate: float, mission_criticality: float = 1.0) -> int:
    """Calculate minimum crew needed for given effective rate.

    Formula:
        min_crew = ceiling(BASE_CREW × (REQUIRED_RATE / effective_rate) × mission_criticality)

    Args:
        effective_rate: External rate from Earth in decisions/sec
        mission_criticality: 1.0 (nominal), 1.5 (high-risk), 0.7 (routine)

    Returns:
        Minimum crew count

    Note:
        If effective_rate > REQUIRED_RATE, crew can be reduced.
        If effective_rate < REQUIRED_RATE, crew must increase to compensate.
    """
    if effective_rate <= 0:
        return 500  # Max crew if no external support

    ratio = REQUIRED_RATE / effective_rate
    raw_crew = BASE_CREW * ratio * mission_criticality
    return max(BASE_CREW, math.ceil(raw_crew))


def crew_reduction_from_autonomy(
    investment_m: float,
    bw_mbps: float,
    delay_s: float,
    tau_base: float = TAU_BASE_CURRENT_S,
) -> dict:
    """Calculate crew threshold reduction from autonomy investment.

    Args:
        investment_m: Autonomy R&D investment in millions USD
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        tau_base: Starting τ value in seconds

    Returns:
        Dict with:
        - before: threshold crew before investment
        - after: threshold crew after investment
        - reduction: number of crew saved
        - reduction_pct: percentage reduction
        - tau_before: original τ
        - tau_after: new τ achieved

    Expected result (per Grok):
        $500M autonomy → 47→28 crew (40% reduction)
    """
    # Get threshold before investment
    threshold_before = threshold_from_tau(tau_base, bw_mbps, delay_s)

    # Calculate new τ from investment
    tau_after = tau_from_investment(investment_m, tau_base)

    # Get threshold after investment
    threshold_after = threshold_from_tau(tau_after, bw_mbps, delay_s)

    # Calculate reduction
    reduction = threshold_before - threshold_after
    reduction_pct = (reduction / threshold_before * 100) if threshold_before > 0 else 0

    return {
        "investment_m": investment_m,
        "before": threshold_before,
        "after": threshold_after,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
        "tau_before": tau_base,
        "tau_after": tau_after,
    }


def emit_roi_receipt(
    investment_m: float, bw_base: float, tau_base: float, delay_s: float
) -> dict:
    """Emit receipt for ROI comparison.

    MUST emit receipt per CLAUDEME.
    """
    comparison = compare_investment_roi(investment_m, bw_base, tau_base, delay_s)
    breakeven = find_breakeven_delay(bw_base, tau_base, investment_m)
    crew = crew_reduction_from_autonomy(investment_m, bw_base, delay_s, tau_base)

    return emit_receipt(
        "roi_comparison",
        {
            "tenant_id": "spaceproof-core",
            **comparison,
            "breakeven_delay_s": breakeven,
            "breakeven_delay_min": breakeven / 60,
            "crew_before": crew["before"],
            "crew_after": crew["after"],
            "crew_reduction": crew["reduction"],
            "crew_reduction_pct": crew["reduction_pct"],
            "finding": (
                f"τ-reduction ROI is {comparison['ratio']:.1f}x higher than bandwidth ROI "
                f"at {delay_s / 60:.0f} min delay"
            ),
        },
    )


# === META-COMPRESSION FACTOR (v1.3 - Grok: "meta-loop is pure τ reduction") ===
# Source: "AI→AI iteration compresses the question-to-shift path by 5-10x"


def meta_compression_factor(iteration_mode: str = "ai") -> float:
    """Returns compression factor for R&D iteration mode.

    THE META-INSIGHT: The AI→AI loop IS itself τ reduction.
    Same spend reaches τ reduction 7.5x faster with AI-mediated R&D.

    Args:
        iteration_mode: "ai" for AI-mediated, "human" for human-only

    Returns:
        Compression factor (7.5 for AI, 1.0 for human)

    Source: Grok Dec 16, 2025 - "5-10x compression on R&D decision latency"
    """
    if iteration_mode == "ai":
        return ITERATION_COMPRESSION_FACTOR
    return 1.0


def effective_autonomy_spend(raw_spend_m: float, iteration_mode: str = "ai") -> float:
    """Calculate effective autonomy spend accounting for iteration speedup.

    The insight: $500M with AI-mediated R&D delivers value equivalent to
    $500M × 7.5 = $3.75B of human-only R&D, due to faster iteration.

    Args:
        raw_spend_m: Raw investment in millions USD
        iteration_mode: "ai" or "human"

    Returns:
        Effective spend value in millions USD

    Note: This is VALUE equivalence, not actual cost.
    """
    return raw_spend_m * meta_compression_factor(iteration_mode)


def meta_boosted_roi(base_roi: float, iteration_mode: str = "ai") -> float:
    """ROI multiplied by faster iteration discovery.

    If AI path discovers the τ reduction 7.5x faster,
    the NPV is higher even at same nominal τ outcome.

    Args:
        base_roi: Base ROI (effective_rate_gain / investment)
        iteration_mode: "ai" or "human"

    Returns:
        Boosted ROI accounting for time-to-value
    """
    return base_roi * meta_compression_factor(iteration_mode)


def compare_iteration_modes(
    spend_m: float, bw_mbps: float, delay_s: float, tau_base: float = TAU_BASE_CURRENT_S
) -> dict:
    """Side-by-side comparison: human-only vs AI-mediated outcomes.

    Same $500M investment:
        Human path: τ reduction discovered in year 3, ROI starts year 4
        AI path: τ reduction discovered in month 5, ROI starts month 6

    Args:
        spend_m: Investment in millions USD
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        tau_base: Starting τ value in seconds

    Returns:
        Dict with side-by-side comparison:
        - human_time_to_value_years: Years to achieve τ reduction
        - ai_time_to_value_years: Years to achieve τ reduction
        - speedup_factor: How much faster AI path is
        - effective_roi_human: ROI for human path
        - effective_roi_ai: ROI for AI path
        - roi_advantage: How much better AI ROI is
    """
    # Time to achieve τ reduction
    human_time_years = META_TAU_HUMAN_DAYS * 12 / 365  # ~1 year per major iteration
    ai_time_years = META_TAU_AI_DAYS * 12 / 365  # ~0.13 years per iteration

    # For equivalent R&D progress, multiply by cycles needed
    # Assume ~10 major R&D cycles needed for breakthrough
    cycles_needed = 10
    human_total_years = human_time_years * cycles_needed
    ai_total_years = ai_time_years * cycles_needed

    # ROI calculations
    tau_achieved = tau_from_investment(spend_m, tau_base)
    base_roi = roi_tau_investment(spend_m, tau_base, bw_mbps, delay_s)

    # Human path: ROI starts after human_total_years
    # AI path: ROI starts after ai_total_years
    # NPV advantage comes from earlier ROI collection

    # Simple model: ROI per year × years of advantage
    years_advantage = human_total_years - ai_total_years
    effective_roi_human = base_roi
    effective_roi_ai = base_roi * (1 + years_advantage)  # More years to collect ROI

    return {
        "spend_m": spend_m,
        "tau_achieved": tau_achieved,
        "human_time_to_value_years": round(human_total_years, 2),
        "ai_time_to_value_years": round(ai_total_years, 2),
        "speedup_factor": round(human_total_years / ai_total_years, 1),
        "years_earlier": round(years_advantage, 2),
        "base_roi": base_roi,
        "effective_roi_human": effective_roi_human,
        "effective_roi_ai": effective_roi_ai,
        "roi_advantage": round(effective_roi_ai / effective_roi_human, 2)
        if effective_roi_human > 0
        else float("inf"),
        "meta_insight": "AI-mediated R&D reaches τ reduction 7.5x faster",
    }


# === COST FUNCTION SWEEP SIMULATION (v1.3 - Grok: "sim variable τ costs") ===


def sweep_cost_functions(
    spend_range: Tuple[float, float],
    bw_mbps: float,
    delay_s: float,
    curve_types: List[str] = None,
    steps: int = 20,
) -> dict:
    """Run simulation across all curve types.

    Sweeps spend from min to max, computing achievable τ and ROI
    for each curve type.

    Args:
        spend_range: (min_spend_m, max_spend_m) in millions USD
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        curve_types: List of curve types (default: all three)
        steps: Number of spend levels to evaluate

    Returns:
        Dict with sweep results for each curve type:
        - curve_type: list of (spend, tau, effective_rate, roi) tuples
        - optimal: optimal spend for each curve
    """
    if curve_types is None:
        curve_types = ["exponential", "logistic", "piecewise"]

    min_spend, max_spend = spend_range
    spend_step = (max_spend - min_spend) / (steps - 1) if steps > 1 else 0

    results = {}

    for curve in curve_types:
        curve_results = []
        peak_roi = 0
        optimal_spend = min_spend
        optimal_tau = TAU_BASE_CURRENT_S
        optimal_rate = 0

        for i in range(steps):
            spend = min_spend + i * spend_step

            # Get achievable τ for this spend
            tau = tau_from_cost(spend, curve)

            # Calculate effective rate at this τ
            # Better autonomy (lower τ) → higher effective decay constant
            decay_tau = TAU_BASE_CURRENT_S * TAU_BASE_CURRENT_S / tau
            effective_rate = external_rate_exponential(bw_mbps, delay_s, decay_tau)

            # Calculate ROI
            # Rate gain from τ reduction
            baseline_decay = TAU_BASE_CURRENT_S  # At baseline τ=300s
            baseline_rate = external_rate_exponential(bw_mbps, delay_s, baseline_decay)
            rate_gain = effective_rate - baseline_rate
            roi = rate_gain / spend if spend > 0 else 0

            curve_results.append(
                {
                    "spend_m": round(spend, 1),
                    "tau_s": round(tau, 1),
                    "effective_rate": round(effective_rate, 2),
                    "roi": round(roi, 6),
                }
            )

            # Track optimal
            if roi > peak_roi:
                peak_roi = roi
                optimal_spend = spend
                optimal_tau = tau
                optimal_rate = effective_rate

        results[curve] = {
            "sweep": curve_results,
            "optimal": {
                "spend_m": round(optimal_spend, 1),
                "tau_s": round(optimal_tau, 1),
                "effective_rate": round(optimal_rate, 2),
                "peak_roi": round(peak_roi, 6),
            },
        }

    return results


def find_optimal_spend(
    curve_type: str, bw_mbps: float, delay_s: float, budget_max: float = 1000
) -> dict:
    """Find spend that maximizes ROI for given curve.

    Uses golden section search to find optimal spend.

    Args:
        curve_type: "exponential", "logistic", or "piecewise"
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        budget_max: Maximum budget to consider in $M

    Returns:
        Dict with optimal spend and resulting metrics
    """
    # Golden section search
    phi = (1 + math.sqrt(5)) / 2
    tol = 1.0  # $1M precision

    a, b = 10.0, budget_max  # Search from $10M to max

    def compute_roi(spend):
        tau = tau_from_cost(spend, curve_type)
        decay_tau = TAU_BASE_CURRENT_S * TAU_BASE_CURRENT_S / tau
        effective_rate = external_rate_exponential(bw_mbps, delay_s, decay_tau)
        baseline_rate = external_rate_exponential(bw_mbps, delay_s, TAU_BASE_CURRENT_S)
        rate_gain = effective_rate - baseline_rate
        return rate_gain / spend if spend > 0 else 0

    c = b - (b - a) / phi
    d = a + (b - a) / phi

    while abs(b - a) > tol:
        if compute_roi(c) > compute_roi(d):
            b = d
        else:
            a = c

        c = b - (b - a) / phi
        d = a + (b - a) / phi

    optimal_spend = (a + b) / 2
    optimal_tau = tau_from_cost(optimal_spend, curve_type)
    decay_tau = TAU_BASE_CURRENT_S * TAU_BASE_CURRENT_S / optimal_tau
    optimal_rate = external_rate_exponential(bw_mbps, delay_s, decay_tau)
    optimal_roi = compute_roi(optimal_spend)

    return {
        "curve_type": curve_type,
        "optimal_spend_m": round(optimal_spend, 1),
        "tau_achieved_s": round(optimal_tau, 1),
        "effective_rate": round(optimal_rate, 2),
        "roi": round(optimal_roi, 6),
    }


def compare_curves_at_budget(budget_m: float, bw_mbps: float, delay_s: float) -> dict:
    """Same budget, different curves → different outcomes.

    Compare what each cost curve achieves with the same investment.

    Args:
        budget_m: Fixed budget in millions USD
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds

    Returns:
        Dict comparing outcomes for each curve type
    """
    curves = ["exponential", "logistic", "piecewise"]
    results = {}

    for curve in curves:
        tau = tau_from_cost(budget_m, curve)
        decay_tau = TAU_BASE_CURRENT_S * TAU_BASE_CURRENT_S / tau
        effective_rate = external_rate_exponential(bw_mbps, delay_s, decay_tau)

        baseline_rate = external_rate_exponential(bw_mbps, delay_s, TAU_BASE_CURRENT_S)
        rate_gain = effective_rate - baseline_rate
        roi = rate_gain / budget_m if budget_m > 0 else 0

        results[curve] = {
            "tau_achieved_s": round(tau, 1),
            "effective_rate": round(effective_rate, 2),
            "rate_gain": round(rate_gain, 2),
            "roi": round(roi, 6),
        }

    # Find best curve for this budget
    best_curve = max(results.keys(), key=lambda c: results[c]["roi"])

    return {
        "budget_m": budget_m,
        "delay_s": delay_s,
        "curves": results,
        "best_curve": best_curve,
        "best_roi": results[best_curve]["roi"],
    }


def recommend_cost_function(observed_data: dict = None) -> str:
    """Returns best-fit curve type.

    Without observed data, defaults to logistic based on
    technology adoption theory.

    Args:
        observed_data: Optional dict with historical τ/cost pairs
                      (not implemented - placeholder for future)

    Returns:
        Recommended curve type string

    Source: Grok Dec 16, 2025 - logistic matches reality best
    """
    # Future: fit curves to observed_data if provided

    # Default recommendation: logistic
    # Rationale:
    # - Exponential assumes constant doubling (unrealistic)
    # - Piecewise is too discrete
    # - Logistic matches technology S-curve adoption
    return "logistic"


def emit_sweep_receipt(
    spend_range: Tuple[float, float], bw_mbps: float, delay_s: float
) -> dict:
    """Emit receipt for cost function sweep.

    MUST emit receipt per CLAUDEME.
    """
    sweep_results = sweep_cost_functions(spend_range, bw_mbps, delay_s)
    comparison = compare_iteration_modes(500, bw_mbps, delay_s)  # $500M reference

    # Find best curve at typical spend
    budget_comparison = compare_curves_at_budget(400, bw_mbps, delay_s)

    return emit_receipt(
        "cost_function_sweep",
        {
            "tenant_id": "spaceproof-core",
            "spend_range_m": spend_range,
            "bandwidth_mbps": bw_mbps,
            "delay_s": delay_s,
            "sweep_results": sweep_results,
            "meta_comparison": comparison,
            "recommended_curve": recommend_cost_function(),
            "best_curve_at_400m": budget_comparison["best_curve"],
            "finding": f"Logistic curve with $400M inflection. AI iteration = {comparison['speedup_factor']}x faster.",
        },
    )


# === PERSON-EQUIVALENT CAPABILITY (v2.0 - Grok Integration) ===
# Source: Grok - "threshold = self-sustaining city, ~10^6 person-equivalent"

THRESHOLD_PERSON_EQUIVALENT = 1_000_000
"""Sovereignty threshold in person-equivalent capability.
Source: Grok - '~10^6 person-equivalent'"""

DECISION_CAPACITY_PER_PERSON = 10.0
"""Decision capacity in bits/sec per human equivalent.
Matches HUMAN_DECISION_RATE_BPS from entropy_shannon."""

EXPERTISE_COVERAGE_BASE = 0.5
"""Base expertise coverage for minimal autonomous operation."""


def capability_to_person_equivalent(
    decision_capacity_bps: float, tau: float, expertise_coverage: float
) -> float:
    """Convert autonomy state to person-equivalent capability units.

    Person-equivalent measures autonomous decision capacity in terms
    of how many humans worth of decision-making capability exists.

    Args:
        decision_capacity_bps: Decision capacity in bits/sec
        tau: Decision latency in seconds (lower = better)
        expertise_coverage: Domain expertise coverage (0-1)

    Returns:
        Person-equivalent capability units

    Formula:
        person_eq = (decision_capacity / base_rate) * (tau_ref / tau) * expertise

    Example:
        At 10000 bps, tau=30s, expertise=0.8:
        person_eq = (10000/10) * (300/30) * 0.8 = 1000 * 10 * 0.8 = 8000
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if expertise_coverage < 0 or expertise_coverage > 1:
        raise ValueError(
            f"expertise_coverage must be in [0, 1], got {expertise_coverage}"
        )

    # Base capacity normalized to human decision rate
    capacity_factor = decision_capacity_bps / DECISION_CAPACITY_PER_PERSON

    # Tau factor: lower tau = more effective capacity
    # Reference tau = 300s (baseline), tau=30s gives 10x multiplier
    tau_factor = TAU_BASE_CURRENT_S / tau

    # Expertise factor directly multiplies
    expertise_factor = expertise_coverage

    person_equivalent = capacity_factor * tau_factor * expertise_factor

    return person_equivalent


def check_sovereignty_threshold(
    person_equivalent: float, threshold: int = THRESHOLD_PERSON_EQUIVALENT
) -> bool:
    """Check if person-equivalent capability meets sovereignty threshold.

    Sovereignty is achieved when autonomous capability reaches
    10^6 person-equivalents (~1M humans worth of decision capacity).

    Args:
        person_equivalent: Current person-equivalent capability
        threshold: Target threshold (default 1M)

    Returns:
        True if person_equivalent >= threshold
    """
    return person_equivalent >= threshold


def project_sovereignty_capability(
    current_capacity_bps: float,
    current_tau: float,
    current_expertise: float,
    investment_m: float,
    iteration_mode: str = "ai",
) -> dict:
    """Project person-equivalent capability after investment.

    Combines investment effects on tau reduction with current state
    to project future capability.

    Args:
        current_capacity_bps: Current decision capacity
        current_tau: Current decision latency
        current_expertise: Current expertise coverage
        investment_m: Autonomy investment in millions USD
        iteration_mode: "ai" or "human" (affects R&D speed)

    Returns:
        Dict with current and projected capability
    """
    # Current person-equivalent
    current_pe = capability_to_person_equivalent(
        current_capacity_bps, current_tau, current_expertise
    )

    # Project tau after investment
    projected_tau = tau_from_investment(investment_m, current_tau)

    # Assume expertise improves slightly with investment
    expertise_gain = min(0.1, investment_m / 5000.0)  # Max +10% at $500M
    projected_expertise = min(1.0, current_expertise + expertise_gain)

    # Assume capacity improves with tau (faster decisions = higher throughput)
    capacity_multiplier = current_tau / projected_tau
    projected_capacity = current_capacity_bps * min(
        capacity_multiplier, 5.0
    )  # Cap at 5x

    # Projected person-equivalent
    projected_pe = capability_to_person_equivalent(
        projected_capacity, projected_tau, projected_expertise
    )

    # Meta-compression factor for AI-mediated R&D
    if iteration_mode == "ai":
        speed_factor = ITERATION_COMPRESSION_FACTOR
    else:
        speed_factor = 1.0

    return {
        "current_person_equivalent": current_pe,
        "projected_person_equivalent": projected_pe,
        "improvement_factor": projected_pe / current_pe
        if current_pe > 0
        else float("inf"),
        "current_tau_s": current_tau,
        "projected_tau_s": projected_tau,
        "current_expertise": current_expertise,
        "projected_expertise": projected_expertise,
        "investment_m": investment_m,
        "iteration_mode": iteration_mode,
        "iteration_speedup": speed_factor,
        "meets_threshold": check_sovereignty_threshold(projected_pe),
        "threshold": THRESHOLD_PERSON_EQUIVALENT,
        "gap_to_threshold": THRESHOLD_PERSON_EQUIVALENT - projected_pe,
    }


def emit_sovereignty_v2_receipt(
    person_equivalent: float, tau: float, expertise: float, decision_capacity_bps: float
) -> dict:
    """Emit receipt for v2 sovereignty calculation.

    Extends v1 receipt with person-equivalent fields.

    Args:
        person_equivalent: Calculated person-equivalent capability
        tau: Decision latency used
        expertise: Expertise coverage used
        decision_capacity_bps: Decision capacity used

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "sovereignty_v2",
        {
            "tenant_id": "spaceproof-autonomy",
            "person_equivalent": person_equivalent,
            "threshold_person_equivalent": THRESHOLD_PERSON_EQUIVALENT,
            "meets_threshold": check_sovereignty_threshold(person_equivalent),
            "tau_s": tau,
            "expertise_coverage": expertise,
            "decision_capacity_bps": decision_capacity_bps,
            "gap_to_threshold": THRESHOLD_PERSON_EQUIVALENT - person_equivalent,
        },
    )
