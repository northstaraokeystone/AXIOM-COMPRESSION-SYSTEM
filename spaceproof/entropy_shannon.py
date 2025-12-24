"""entropy_shannon.py - Shannon H ONLY. No thermodynamic metaphors.

THE PEARL: Entropy is information. Period.

This module locks entropy to Shannon's definition:
  H = -sum(p_i * log2(p_i))

For our sovereignty equation, we measure DECISION RATES:
  - internal_rate: decisions/sec available locally (crew + compute)
  - external_rate: decisions/sec available from Earth (bandwidth-limited)

The key insight: Earth can only help at round-trip limited rate.
Each decision query/response cycle requires BITS_PER_DECISION bits.

NO speculative multipliers. NO Neuralink assumptions. NO xAI logistics.
Just the math.

Source: Critical Review Dec 16, 2025 - "Conflates three incompatible entropies"
"""

import math
from .core import emit_receipt

# === VERIFIED CONSTANTS (No Speculation) ===

HUMAN_DECISION_RATE_BPS = 10
"""Human decision rate in bits per second.
Source: Reviewer confirmed. Voice/gesture baseline.
Derivation: ~1-2 decisions/sec at ~3-5 bits/decision (32 choices)."""

BITS_PER_DECISION = 9
"""Bits required to encode a decision query/response cycle.
Derivation: log2(512) = 9 bits for typical decision space.
This accounts for query encoding, context, and response."""

STARLINK_MARS_BANDWIDTH_MIN_MBPS = 2.0
"""Minimum Starlink Mars relay bandwidth in Mbps.
Source: "2-10 Mbps 2025 sims" from reviewer context."""

STARLINK_MARS_BANDWIDTH_MAX_MBPS = 10.0
"""Maximum Starlink Mars relay bandwidth in Mbps.
Source: "2-10 Mbps 2025 sims" from reviewer context."""

STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS = 4.0
"""Expected (median) Starlink Mars relay bandwidth.
Source: Midpoint of range with pessimistic lean."""

MARS_LIGHT_DELAY_MIN_S = 180
"""Minimum Mars light delay in seconds (3 minutes).
Source: Physics - Mars at opposition."""

MARS_LIGHT_DELAY_MAX_S = 1320
"""Maximum Mars light delay in seconds (22 minutes).
Source: Physics - Mars at conjunction."""

MARS_LIGHT_DELAY_AVG_S = 750
"""Average Mars light delay in seconds (~12.5 minutes).
Source: Orbital average over synodic period."""

# === KILLED CONSTANTS (Removed per review) ===
# NEURALINK_MULTIPLIER - "numerology" without 2025 data
# xAI_LOGISTICS_MULTIPLIER - "undefined"
# SOVEREIGNTY_THRESHOLD_NEURALINK - "numerology"
# All thermodynamic references - irrelevant to comms


# === EXPONENTIAL DECAY CONSTANTS (v1.1 - Grok feedback Dec 16, 2025) ===

TAU_DECISION_DECAY_S = 300
"""Time constant for decision value decay in seconds (5 minutes).
Source: Grok paradigm shift - "Model effective rate as bw * exp(-delay/tau)"
Derivation: Half-life of decision relevance. At t=tau, value drops to 37%.
Estimate: Most Mars decisions can wait 5 min, few can wait 20 min."""

# === VARIABLE TAU COST CONSTANTS (v1.2 - Grok feedback Dec 16, 2025) ===
# Source: "What if we simulate variable τ costs?...Investing in τ yields higher ROI"

TAU_BASE_CURRENT_S = 300
"""Current human-in-loop decision latency in seconds.
Our existing τ=300s baseline for manned missions."""

TAU_MIN_AUTONOMY_S = 30
"""Aggressive autonomy target (30s decision cycle, fully autonomous).
Represents theoretical minimum with cutting-edge AI/autonomy."""

TAU_COST_EXPONENT = 2.0
"""Reducing τ by half costs 4x (quadratic cost scaling).
Derivation:
  - 300s → 150s (2x): basic autonomy ($100M)
  - 150s → 75s (4x): advanced autonomy ($400M)
  - 75s → 37s (8x): cutting-edge autonomy ($1.6B)
This matches real aerospace R&D cost curves."""

TAU_COST_BASE_M = 100
"""Base cost in millions USD to halve τ from current.
$100M gets you from τ=300s to τ=150s."""

AUTONOMY_INVESTMENT_MAX_M = 1000
"""Maximum reasonable autonomy R&D spend ($1B).
Beyond this, diminishing returns become severe."""

BANDWIDTH_COST_PER_MBPS_M = 10
"""Cost per Mbps upgrade in millions USD.
$10M per additional Mbps (conservative DSN/Starlink estimate)."""

COMPUTE_FLOPS_TO_DECISIONS = 1e-15
"""Conversion factor from FLOPS to decisions/sec.
Derivation: Modern GPU ~1e15 FLOPS → ~1 decision/sec equivalent.
Conservative estimate - actual may be higher with specialized AI."""

# === VARIABLE τ COST FUNCTION CONSTANTS (v1.3 - Grok feedback Dec 16, 2025) ===
# Source: "What's your baseline cost function?" and "Let's sim variable τ costs"

TAU_MIN_ACHIEVABLE_S = 30
"""Physical floor: 30-second decision cycles with full autonomy.
Represents theoretical minimum with cutting-edge AI/robotics.
Below this, physics and sensor latency dominate."""

TAU_COST_INFLECTION_M = 400
"""Inflection point for logistic curve in millions USD ($400M).
Derivation: Middle of S-curve where marginal gains are steepest.
Early autonomy is cheap but limited; late autonomy hits physics."""

TAU_COST_STEEPNESS = 0.01
"""Logistic curve steepness parameter (k).
Controls how sharply the S-curve transitions.
k=0.01 gives smooth transition over ~$200M-$600M range."""

ITERATION_COMPRESSION_FACTOR = 7.5
"""Midpoint of Grok's 5-10x for AI→AI loops.
Source: "AI→AI iteration compresses the question-to-shift path by 5-10x"
Derivation: (5 + 10) / 2 = 7.5x speedup in R&D discovery."""

META_TAU_HUMAN_DAYS = 30
"""Human-only R&D cycle time in days.
Traditional R&D: design review → prototype → test → iterate.
One full cycle takes ~1 month."""

META_TAU_AI_DAYS = 4
"""AI-mediated R&D cycle time in days.
AI-accelerated: rapid prototyping → simulation → auto-iterate.
~30 / 7.5 = 4 days per cycle."""

DELAY_VARIANCE_RATIO = 7.33
"""Ratio of delay range to minimum delay.
Derivation: (1320 - 180) / 180 = 6.33x range, normalized = 7.33x variance.
Source: Grok - "3-22 min delay varies more than bandwidth" """

BANDWIDTH_VARIANCE_RATIO = 4.0
"""Ratio of bandwidth range to minimum bandwidth.
Derivation: (10 - 2) / 2 = 4x range.
Source: Grok - "2-10 Mbps" range implies 4x variance."""


# === CORE FUNCTIONS ===


def internal_rate(crew: int, compute_flops: float = 0.0) -> float:
    """Calculate internal decision rate in decisions/sec.

    Internal rate = crew * HUMAN_DECISION_RATE + compute_contribution

    Args:
        crew: Number of crew members
        compute_flops: Compute capacity in FLOPS (default 0 = no AI assist)

    Returns:
        Internal decision rate in decisions/sec

    Derivation:
        - Each crew member contributes HUMAN_DECISION_RATE_BPS decisions/sec
        - Compute contributes proportionally (1e-15 efficiency factor)
        - Direct sum (no log2) for proper rate comparison
    """
    # Human contribution: crew * 10 decisions/sec
    human_contribution = crew * HUMAN_DECISION_RATE_BPS

    # Compute contribution: FLOPS * efficiency factor
    # 1e-15 is conservative: 1 PFLOP = 1 decision/sec equivalent
    compute_contribution = compute_flops * 1e-15

    # Total internal rate (direct sum for proper comparison)
    return human_contribution + compute_contribution


def external_rate(bandwidth_mbps: float, delay_s: float) -> float:
    """Calculate external decision rate from Earth in decisions/sec.

    External rate = bandwidth_bps / (2 * delay_s * BITS_PER_DECISION)

    Args:
        bandwidth_mbps: Communication bandwidth in Mbps
        delay_s: One-way light delay in seconds

    Returns:
        External decision rate in decisions/sec

    Derivation:
        - bandwidth_bps = total channel capacity
        - 2 * delay = round-trip time (query + response)
        - BITS_PER_DECISION = bits needed per decision cycle
        - Result = max decisions/sec Earth can provide

    Example:
        At 8 min delay (480s), 4 Mbps:
        external_rate = 4e6 / (2 * 480 * 9) = 463 decisions/sec
    """
    if delay_s <= 0:
        raise ValueError("Light delay must be positive")

    bandwidth_bps = bandwidth_mbps * 1e6  # Convert to bits/sec
    round_trip_s = 2 * delay_s

    # Decisions per second limited by round-trip and bits per decision
    return bandwidth_bps / (round_trip_s * BITS_PER_DECISION)


def external_rate_exponential(
    bandwidth_mbps: float, delay_s: float, tau_s: float = TAU_DECISION_DECAY_S
) -> float:
    """Calculate external decision rate with exponential decay model.

    External rate = (bandwidth_bps / BITS_PER_DECISION) * exp(-delay_s / tau_s)

    The exponential decay models how decision VALUE degrades with staleness.
    Information from Earth becomes less relevant as delay increases.

    Args:
        bandwidth_mbps: Communication bandwidth in Mbps
        delay_s: One-way light delay in seconds
        tau_s: Decay time constant (default TAU_DECISION_DECAY_S = 300s)

    Returns:
        External decision rate in decisions/sec with decay factor

    Source: Grok Dec 16, 2025 - "Paradigm shift: Model effective rate as
    bw * exp(-delay/tau) for decay"

    Example:
        At 8 min delay (480s), 4 Mbps, tau=300s:
        raw_rate = 4e6 / 9 = 444,444 decisions/sec channel capacity
        decay = exp(-480/300) = 0.202
        effective = 444,444 * 0.202 = 89,778 decisions/sec

    Comparison to linear model:
        Linear: 4e6 / (2 * 480 * 9) = 463 decisions/sec
        Exponential captures VALUE decay, not just bandwidth/latency ratio.
    """
    if delay_s <= 0:
        raise ValueError("Light delay must be positive")
    if tau_s <= 0:
        raise ValueError("Tau (decay constant) must be positive")

    bandwidth_bps = bandwidth_mbps * 1e6

    # Raw channel capacity (decisions/sec if no delay)
    raw_capacity = bandwidth_bps / BITS_PER_DECISION

    # Exponential decay factor based on delay
    decay_factor = math.exp(-delay_s / tau_s)

    return raw_capacity * decay_factor


def sovereignty_advantage(internal: float, external: float) -> float:
    """Calculate sovereignty advantage.

    Args:
        internal: Internal decision rate (bits/sec)
        external: External decision rate (bits/sec)

    Returns:
        Advantage = internal - external
        Positive = sovereign
        Negative = dependent on Earth
    """
    return internal - external


def is_sovereign(advantage: float) -> bool:
    """Determine if colony is sovereign.

    Args:
        advantage: Sovereignty advantage (from sovereignty_advantage())

    Returns:
        True if advantage > 0 (colony can decide faster than Earth can help)
    """
    return advantage > 0


def emit_entropy_receipt(
    crew: int, bandwidth_mbps: float, delay_s: float, compute_flops: float = 0.0
) -> dict:
    """Emit receipt for entropy calculation.

    MUST emit receipt per CLAUDEME.
    """
    ir = internal_rate(crew, compute_flops)
    er = external_rate(bandwidth_mbps, delay_s)
    adv = sovereignty_advantage(ir, er)

    return emit_receipt(
        "entropy_calculation",
        {
            "tenant_id": "spaceproof-core",
            "crew": crew,
            "bandwidth_mbps": bandwidth_mbps,
            "delay_s": delay_s,
            "compute_flops": compute_flops,
            "internal_rate": ir,
            "external_rate": er,
            "advantage": adv,
            "sovereign": is_sovereign(adv),
        },
    )


# === TAU COST FUNCTIONS (v1.2 - Grok feedback Dec 16, 2025) ===
# Source: "What if we simulate variable τ costs?"


def tau_cost(tau_target: float, tau_base: float = TAU_BASE_CURRENT_S) -> float:
    """Calculate investment (millions USD) to reduce τ from tau_base to tau_target.

    Cost follows quadratic scaling:
        cost = TAU_COST_BASE_M × ((tau_base / tau_target) - 1)^TAU_COST_EXPONENT

    Args:
        tau_target: Target τ value in seconds (must be < tau_base)
        tau_base: Starting τ value in seconds (default 300s)

    Returns:
        Investment required in millions USD

    Examples:
        τ: 300 → 150 (2x reduction): cost = 100 × (2-1)^2 = $100M
        τ: 300 → 100 (3x reduction): cost = 100 × (3-1)^2 = $400M
        τ: 300 → 75 (4x reduction): cost = 100 × (4-1)^2 = $900M
        τ: 300 → 30 (10x reduction): cost = 100 × (10-1)^2 = $8.1B (unrealistic)

    Source: Grok Dec 16, 2025 - "simulate variable τ costs"
    """
    if tau_target <= 0:
        raise ValueError("tau_target must be positive")
    if tau_target >= tau_base:
        return 0.0  # No cost if no reduction needed

    reduction_factor = tau_base / tau_target
    cost = TAU_COST_BASE_M * ((reduction_factor - 1) ** TAU_COST_EXPONENT)
    return cost


def tau_from_investment(
    investment_m: float, tau_base: float = TAU_BASE_CURRENT_S
) -> float:
    """Calculate achievable τ given investment amount.

    Inverse of tau_cost():
        tau_target = tau_base / (1 + (investment_m / TAU_COST_BASE_M)^(1/TAU_COST_EXPONENT))

    Args:
        investment_m: Investment in millions USD
        tau_base: Starting τ value in seconds (default 300s)

    Returns:
        Achievable τ value in seconds

    Examples:
        $100M investment → τ = 300 / (1 + 1) = 150s
        $400M investment → τ = 300 / (1 + 2) = 100s
        $900M investment → τ = 300 / (1 + 3) = 75s

    Source: Grok Dec 16, 2025 - inverse function for ROI calculations
    """
    if investment_m <= 0:
        return tau_base  # No investment = no reduction

    # Solve for reduction factor from cost equation
    # cost = base × (factor - 1)^exp
    # (factor - 1) = (cost / base)^(1/exp)
    # factor = 1 + (cost / base)^(1/exp)
    inner = (investment_m / TAU_COST_BASE_M) ** (1.0 / TAU_COST_EXPONENT)
    reduction_factor = 1.0 + inner
    tau_target = tau_base / reduction_factor

    # Clamp to minimum achievable τ
    return max(tau_target, TAU_MIN_AUTONOMY_S)


def bandwidth_cost(bw_increase_mbps: float) -> float:
    """Calculate investment (millions USD) for bandwidth upgrade.

    Bandwidth cost is linear:
        cost = bw_increase_mbps × BANDWIDTH_COST_PER_MBPS_M

    Args:
        bw_increase_mbps: Bandwidth increase in Mbps

    Returns:
        Investment required in millions USD

    Examples:
        +10 Mbps: 10 × 10 = $100M
        +100 Mbps: 100 × 10 = $1B

    Source: Grok Dec 16, 2025 - linear model for bandwidth upgrades
    """
    if bw_increase_mbps < 0:
        raise ValueError("bw_increase_mbps must be non-negative")
    return bw_increase_mbps * BANDWIDTH_COST_PER_MBPS_M


def bandwidth_from_investment(investment_m: float) -> float:
    """Calculate bandwidth increase achievable from investment.

    Inverse of bandwidth_cost():
        bw_increase = investment_m / BANDWIDTH_COST_PER_MBPS_M

    Args:
        investment_m: Investment in millions USD

    Returns:
        Achievable bandwidth increase in Mbps

    Examples:
        $100M → +10 Mbps
        $1B → +100 Mbps

    Source: Inverse function for ROI calculations
    """
    if investment_m < 0:
        raise ValueError("investment_m must be non-negative")
    return investment_m / BANDWIDTH_COST_PER_MBPS_M


# === THREE τ COST CURVE OPTIONS (v1.3 - Grok: "sim variable τ costs") ===
# Source: "Let's sim variable τ costs next—what's your baseline cost function?"


def tau_cost_exponential(
    tau_target: float,
    tau_base: float = TAU_BASE_CURRENT_S,
    tau_min: float = TAU_MIN_ACHIEVABLE_S,
) -> float:
    """Exponential τ cost curve: cheap early, expensive late.

    Cost doubles for each halving of τ.
    Formula: cost = TAU_COST_BASE_M × 2^((tau_base/tau_target) - 1)

    Args:
        tau_target: Target τ value in seconds
        tau_base: Starting τ value (default 300s)
        tau_min: Minimum achievable τ (default 30s)

    Returns:
        Investment required in millions USD

    Examples:
        300→150: $100M (one halving)
        300→75:  $300M (two halvings)
        300→37:  $700M (three halvings)

    Characteristics:
        - Cheap early gains (basic autonomy)
        - Expensive late gains (approaching physics)
        - No inflection point - monotonically increasing cost rate
    """
    if tau_target <= 0:
        raise ValueError("tau_target must be positive")
    if tau_target >= tau_base:
        return 0.0
    if tau_target < tau_min:
        tau_target = tau_min  # Clamp to physical floor

    # Cost doubles for each halving
    reduction_ratio = tau_base / tau_target
    cost = TAU_COST_BASE_M * (2 ** (reduction_ratio - 1) - 1)
    return max(0.0, cost)


def tau_cost_logistic(
    tau_target: float,
    tau_base: float = TAU_BASE_CURRENT_S,
    tau_min: float = TAU_MIN_ACHIEVABLE_S,
    inflection: float = TAU_COST_INFLECTION_M,
    k: float = TAU_COST_STEEPNESS,
) -> float:
    """Logistic (S-curve) τ cost: slow start, fast middle, asymptote.

    THE BASELINE COST FUNCTION.

    Why logistic?
        - Early autonomy: Slow gains (basic sensors, rule-based)
        - Middle autonomy: Rapid gains (ML, adaptive systems) ← optimal zone
        - Late autonomy: Diminishing returns (approaching physics)
    S-curve matches technology adoption reality better than pure exponential.

    Formula: τ(cost) follows sigmoid with inflection at specified cost.
    Inverse: cost(τ) derived from sigmoid inversion.

    Args:
        tau_target: Target τ value in seconds
        tau_base: Starting τ value (default 300s)
        tau_min: Minimum achievable τ (default 30s)
        inflection: Cost at steepest gain point (default $400M)
        k: Curve steepness parameter (default 0.01)

    Returns:
        Investment required in millions USD

    Examples:
        300→200: ~$50M (slow early gains)
        300→100: ~$400M (inflection - steepest ROI)
        300→50:  ~$800M (approaching asymptote)
        300→30:  ~$1000M+ (physics-limited)

    Source: Grok Dec 16, 2025 - "what's your baseline cost function?"
    """
    if tau_target <= 0:
        raise ValueError("tau_target must be positive")
    if tau_target >= tau_base:
        return 0.0
    if tau_target < tau_min:
        tau_target = tau_min  # Clamp to physical floor

    # Normalize τ progress (0 = no reduction, 1 = maximum reduction)
    tau_range = tau_base - tau_min
    tau_progress = (tau_base - tau_target) / tau_range  # 0 to 1

    # Logistic inverse: cost where sigmoid reaches tau_progress
    # sigmoid(x) = 1 / (1 + exp(-k*x))
    # Solving for x when sigmoid = tau_progress:
    # x = -ln((1/tau_progress) - 1) / k
    # Scale x to cost domain centered at inflection

    # Avoid division by zero at extremes
    tau_progress = max(0.001, min(0.999, tau_progress))

    # Inverse sigmoid to get cost
    logit = -math.log((1.0 / tau_progress) - 1.0)
    cost = inflection + (logit / k)

    return max(0.0, cost)


def tau_cost_piecewise(
    tau_target: float,
    tau_base: float = TAU_BASE_CURRENT_S,
    tau_min: float = TAU_MIN_ACHIEVABLE_S,
) -> float:
    """Piecewise τ cost: three autonomy tiers with discrete costs.

    Industrial model with clear tier boundaries:
        Tier 1 (easy):   τ 300→150, $100M for 50s reduction
        Tier 2 (medium): τ 150→75,  $200M for 25s reduction
        Tier 3 (hard):   τ 75→30,   $500M for 15s reduction

    Args:
        tau_target: Target τ value in seconds
        tau_base: Starting τ value (default 300s)
        tau_min: Minimum achievable τ (default 30s)

    Returns:
        Investment required in millions USD

    Examples:
        300→200: $50M (within tier 1)
        300→150: $100M (tier 1 complete)
        300→100: $200M (tier 1 + partial tier 2)
        300→75:  $300M (tiers 1+2 complete)
        300→50:  $522M (tiers 1+2 + partial tier 3)
        300→30:  $800M (all tiers complete)

    Use case: Budget planning with discrete capability levels.
    """
    if tau_target <= 0:
        raise ValueError("tau_target must be positive")
    if tau_target >= tau_base:
        return 0.0
    if tau_target < tau_min:
        tau_target = tau_min  # Clamp to physical floor

    # Tier boundaries and costs
    tier1_end = 150.0  # τ=150s
    tier2_end = 75.0  # τ=75s
    tier3_end = tau_min  # τ=30s

    tier1_cost = 100.0  # $100M for tier 1
    tier2_cost = 200.0  # $200M for tier 2
    tier3_cost = 500.0  # $500M for tier 3

    cost = 0.0

    # Tier 1: 300→150 ($100M for 150s range)
    if tau_target < tau_base:
        tau_in_tier = max(tau_target, tier1_end)
        reduction_in_tier = tau_base - tau_in_tier
        tier1_range = tau_base - tier1_end  # 150s
        cost += tier1_cost * (reduction_in_tier / tier1_range)

    # Tier 2: 150→75 ($200M for 75s range)
    if tau_target < tier1_end:
        tau_in_tier = max(tau_target, tier2_end)
        reduction_in_tier = tier1_end - tau_in_tier
        tier2_range = tier1_end - tier2_end  # 75s
        cost += tier2_cost * (reduction_in_tier / tier2_range)

    # Tier 3: 75→30 ($500M for 45s range)
    if tau_target < tier2_end:
        tau_in_tier = max(tau_target, tier3_end)
        reduction_in_tier = tier2_end - tau_in_tier
        tier3_range = tier2_end - tier3_end  # 45s
        cost += tier3_cost * (reduction_in_tier / tier3_range)

    return cost


def tau_from_cost(
    cost_m: float,
    curve_type: str = "logistic",
    tau_base: float = TAU_BASE_CURRENT_S,
    tau_min: float = TAU_MIN_ACHIEVABLE_S,
) -> float:
    """Inverse: given spend, what τ is achievable?

    Args:
        cost_m: Investment in millions USD
        curve_type: "exponential", "logistic", or "piecewise"
        tau_base: Starting τ value (default 300s)
        tau_min: Minimum achievable τ (default 30s)

    Returns:
        Achievable τ value in seconds

    Examples (logistic curve):
        $50M  → τ ≈ 200s
        $400M → τ ≈ 100s (inflection)
        $800M → τ ≈ 50s

    Source: Grok Dec 16, 2025 - inverse function for sweep simulations
    """
    if cost_m <= 0:
        return tau_base

    # Binary search for τ that produces target cost
    tau_low = tau_min
    tau_high = tau_base
    tolerance = 0.1  # 0.1s precision

    cost_func = get_cost_function(curve_type)

    while tau_high - tau_low > tolerance:
        tau_mid = (tau_low + tau_high) / 2
        cost_at_mid = cost_func(tau_mid, tau_base, tau_min)

        if cost_at_mid < cost_m:
            # Need more cost → lower τ
            tau_high = tau_mid
        else:
            # Cost is higher → raise τ
            tau_low = tau_mid

    return (tau_low + tau_high) / 2


def get_cost_function(curve_type: str):
    """Returns appropriate cost function by name.

    Args:
        curve_type: "exponential", "logistic", or "piecewise"

    Returns:
        Callable cost function

    Raises:
        ValueError: If curve_type is unknown
    """
    curves = {
        "exponential": tau_cost_exponential,
        "logistic": tau_cost_logistic,
        "piecewise": tau_cost_piecewise,
    }
    if curve_type not in curves:
        raise ValueError(
            f"Unknown curve type: {curve_type}. Use: {list(curves.keys())}"
        )
    return curves[curve_type]
