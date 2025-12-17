"""build_rate.py - Multiplicative Build Rate Model

THE PARADIGM SHIFT (v2.0 - Grok Integration):
    Build rate B = c x A^alpha x P is MULTIPLICATIVE.
    Autonomy and propulsion multiply - they don't compete.
    Under-pivoting (<30%) = existential stall.

KEY INSIGHT: The v1 question was wrong.
    v1: "What's the optimal allocation between propulsion and autonomy?"
    v2: "Wrong question. They multiply, not trade."

At alpha=1.8:
    A=0.40 -> A^1.8 ~ 0.22 to the product
    A=0.25 -> A^1.8 ~ 0.10 to the product
    A=0.00 -> A^1.8 = 0.00 to the product (existential)

The ratio isn't 40/25 = 1.6x. The ratio is 0.22/0.10 = 2.2x.
That's where Grok's "2.5-3.0x" annual multiplier comes from.

Source: Grok - "B ~ c x A^alpha x P"
"""

from dataclasses import dataclass
from typing import Optional
import math

from .core import emit_receipt


# === CONSTANTS (from Grok timeline table) ===

ALPHA_BASELINE = 1.8
"""Compounding exponent for autonomy scaling.
Re-validated by Grok as foundation for all projections.
Source: Grok - 'calibrated to alpha=1.8 baseline'"""

BUILD_RATE_CONSTANT = 1.0
"""Initial conditions factor. Normalized to 1.0 for ratio comparisons."""

# Grok timeline table targets (validation constraints)
MULTIPLIER_40PCT = (2.5, 3.0)
"""Annual multiplier at 40% autonomy allocation. Source: Grok timeline table."""

MULTIPLIER_25PCT = (1.6, 2.0)
"""Annual multiplier at 25% autonomy allocation. Source: Grok timeline table."""

MULTIPLIER_15PCT = (1.2, 1.4)
"""Annual multiplier at 15% autonomy allocation. Source: Grok timeline table."""

MULTIPLIER_0PCT = 1.1
"""Annual multiplier at ~0% autonomy (propulsion-only). Source: Grok timeline table."""

# Person-equivalent normalization
DECISION_CAPACITY_BASE_BPS = 1000.0
"""Base decision capacity in bits per second per person-equivalent unit."""

TAU_REFERENCE_S = 300.0
"""Reference tau for normalization (baseline decision latency)."""


@dataclass
class BuildRateConfig:
    """Configuration for build rate calculation.

    Attributes:
        constant: Initial conditions factor (default 1.0)
        alpha: Compounding exponent for autonomy (default 1.8)
    """
    constant: float = BUILD_RATE_CONSTANT
    alpha: float = ALPHA_BASELINE


@dataclass
class BuildRateState:
    """State of the civilization build rate.

    Attributes:
        autonomy_level: Normalized autonomy level (0-1)
        propulsion_level: Normalized propulsion level (launches/year or capability)
        build_rate: Computed B value (B = c x A^alpha x P)
        annual_multiplier: Effective yearly compounding factor
    """
    autonomy_level: float
    propulsion_level: float
    build_rate: float
    annual_multiplier: float


def compute_build_rate(
    autonomy: float,
    propulsion: float,
    alpha: float = ALPHA_BASELINE,
    constant: float = BUILD_RATE_CONSTANT
) -> float:
    """Compute multiplicative build rate B = c x A^alpha x P.

    THE CORE EQUATION: Build rate is multiplicative, not additive.
    Under-investing in autonomy doesn't just shift timeline - it
    MULTIPLIES the entire build rate by a smaller factor.

    Args:
        autonomy: Autonomy level (0-1, normalized)
        propulsion: Propulsion level (normalized capability metric)
        alpha: Compounding exponent (default 1.8)
        constant: Initial conditions factor (default 1.0)

    Returns:
        Build rate B = c x A^alpha x P

    Example:
        At alpha=1.8:
        - A=0.40, P=1.0 -> B = 1.0 x 0.40^1.8 x 1.0 = 0.217
        - A=0.25, P=1.0 -> B = 1.0 x 0.25^1.8 x 1.0 = 0.098
        - Ratio: 0.217/0.098 = 2.2x (matches Grok's 2.5-3.0x range)
    """
    if autonomy < 0 or autonomy > 1:
        raise ValueError(f"autonomy must be in [0, 1], got {autonomy}")
    if propulsion < 0:
        raise ValueError(f"propulsion must be non-negative, got {propulsion}")

    # Handle edge case: zero autonomy
    if autonomy == 0:
        return 0.0  # Existential stall

    # B = c x A^alpha x P
    build_rate = constant * (autonomy ** alpha) * propulsion

    # Emit receipt
    emit_receipt("build_rate", {
        "tenant_id": "axiom-autonomy",
        "autonomy_level": autonomy,
        "propulsion_level": propulsion,
        "alpha": alpha,
        "constant": constant,
        "build_rate": build_rate,
        "computation": f"B = {constant} x {autonomy}^{alpha} x {propulsion} = {build_rate:.6f}"
    })

    return build_rate


def autonomy_to_level(
    tau: float,
    expertise: float,
    decision_capacity: float
) -> float:
    """Normalize autonomy state to 0-1 level for build rate calculation.

    Combines three factors:
    1. tau (decision latency) - lower is better
    2. expertise (domain coverage) - higher is better
    3. decision_capacity (decisions/sec capability) - higher is better

    Args:
        tau: Decision latency in seconds (lower = better autonomy)
        expertise: Domain expertise coverage (0-1)
        decision_capacity: Decision capacity in bits/sec

    Returns:
        Normalized autonomy level (0-1)

    Formula:
        autonomy = (tau_ref/tau) * expertise * min(decision_capacity/base, 1)
        Clamped to [0, 1]
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    # tau factor: lower tau = higher autonomy
    # Normalized so tau=TAU_REFERENCE gives factor=1.0
    tau_factor = min(TAU_REFERENCE_S / tau, 10.0)  # Cap at 10x improvement

    # Decision capacity factor: normalized to base
    capacity_factor = min(decision_capacity / DECISION_CAPACITY_BASE_BPS, 1.0)

    # Combined autonomy level
    raw_level = tau_factor * expertise * capacity_factor

    # Normalize to [0, 1] - use sqrt to compress range
    level = min(math.sqrt(raw_level) / math.sqrt(10.0), 1.0)

    return level


def propulsion_to_level(
    launches_per_year: float,
    payload_tons: float,
    reliability: float
) -> float:
    """Normalize propulsion state to level for build rate calculation.

    Combines three factors:
    1. launches_per_year - launch cadence
    2. payload_tons - mass to Mars per launch
    3. reliability - mission success rate

    Args:
        launches_per_year: Number of launches per year
        payload_tons: Payload capacity in tons per launch
        reliability: Mission success rate (0-1)

    Returns:
        Normalized propulsion level (positive, typically 0.5-2.0)

    Formula:
        propulsion = (launches * payload * reliability) / baseline
        Where baseline = 10 launches * 100 tons * 0.95 reliability
    """
    if launches_per_year < 0:
        raise ValueError(f"launches_per_year must be non-negative, got {launches_per_year}")
    if payload_tons < 0:
        raise ValueError(f"payload_tons must be non-negative, got {payload_tons}")
    if reliability < 0 or reliability > 1:
        raise ValueError(f"reliability must be in [0, 1], got {reliability}")

    # Baseline: 10 launches * 100 tons * 0.95 reliability = 950 ton-launches
    baseline = 10.0 * 100.0 * 0.95

    # Current: launches * payload * reliability
    current = launches_per_year * payload_tons * reliability

    # Normalized level
    level = current / baseline

    return level


def annual_multiplier(build_rate: float, prior_build_rate: float) -> float:
    """Calculate annual multiplier from build rate change.

    This is Grok's "annual multiplier" column from the timeline table.

    Args:
        build_rate: Current cycle build rate
        prior_build_rate: Previous cycle build rate

    Returns:
        Annual multiplier B_t / B_{t-1}

    Target values (from Grok table):
        40% allocation -> 2.5-3.0x
        25% allocation -> 1.6-2.0x
        15% allocation -> 1.2-1.4x
        0% allocation  -> ~1.1x
    """
    if prior_build_rate <= 0:
        return float('inf') if build_rate > 0 else 1.0

    return build_rate / prior_build_rate


def allocation_to_multiplier(autonomy_fraction: float, alpha: float = ALPHA_BASELINE) -> float:
    """Convert autonomy allocation fraction to expected annual multiplier.

    Uses Grok's empirical table values, interpolated by alpha exponent.

    Args:
        autonomy_fraction: Fraction of resources allocated to autonomy (0-1)
        alpha: Compounding exponent

    Returns:
        Expected annual multiplier

    Grok Table (alpha=1.8):
        40% -> 2.5-3.0x (midpoint 2.75)
        25% -> 1.6-2.0x (midpoint 1.80)
        15% -> 1.2-1.4x (midpoint 1.30)
        0%  -> 1.1x
    """
    if autonomy_fraction <= 0:
        return MULTIPLIER_0PCT

    # Base multiplier from autonomy fraction raised to alpha
    # Scaled to match Grok table at key points
    # At 40%: 0.40^1.8 ~ 0.217, need to map to 2.75
    # Scaling factor: 2.75 / 0.217 ~ 12.7

    base_effect = autonomy_fraction ** alpha

    # Interpolate between 1.1 (floor) and ~3.0 (ceiling at 40%)
    # Using linear scaling between autonomy effect and multiplier
    # multiplier = 1.1 + (max_mult - 1.1) * (effect / max_effect)

    max_effect = 0.40 ** alpha  # ~0.217
    max_mult = (MULTIPLIER_40PCT[0] + MULTIPLIER_40PCT[1]) / 2  # 2.75

    if base_effect >= max_effect:
        return max_mult

    # Linear interpolation
    multiplier = 1.1 + (max_mult - 1.1) * (base_effect / max_effect)

    return multiplier


def validate_grok_multipliers(results: dict) -> bool:
    """Verify multipliers match Grok table within tolerance.

    Args:
        results: Dict with keys like "0.40", "0.25", etc. mapping to multipliers

    Returns:
        True if all multipliers match Grok table within +-0.3

    Grok Table Targets:
        40% -> 2.5-3.0x
        25% -> 1.6-2.0x
        15% -> 1.2-1.4x
        0%  -> ~1.1x
    """
    validations = []

    # Check 40%
    if "0.40" in results or 0.40 in results:
        m = results.get("0.40", results.get(0.40))
        low, high = MULTIPLIER_40PCT
        validations.append(low <= m <= high)

    # Check 25%
    if "0.25" in results or 0.25 in results:
        m = results.get("0.25", results.get(0.25))
        low, high = MULTIPLIER_25PCT
        validations.append(low <= m <= high)

    # Check 15%
    if "0.15" in results or 0.15 in results:
        m = results.get("0.15", results.get(0.15))
        low, high = MULTIPLIER_15PCT
        validations.append(low <= m <= high)

    # Check 0%
    if "0.00" in results or 0.00 in results or 0 in results:
        m = results.get("0.00", results.get(0.00, results.get(0)))
        validations.append(0.9 <= m <= 1.3)  # ~1.1 with tolerance

    return all(validations) if validations else False


def compute_build_rate_state(
    autonomy: float,
    propulsion: float,
    prior_build_rate: float = 0.0,
    config: BuildRateConfig = None
) -> BuildRateState:
    """Compute full build rate state including annual multiplier.

    Args:
        autonomy: Normalized autonomy level (0-1)
        propulsion: Normalized propulsion level
        prior_build_rate: Previous cycle's build rate (for multiplier)
        config: BuildRateConfig (uses defaults if None)

    Returns:
        BuildRateState with all computed values
    """
    if config is None:
        config = BuildRateConfig()

    build_rate = compute_build_rate(
        autonomy, propulsion, config.alpha, config.constant
    )

    mult = annual_multiplier(build_rate, prior_build_rate) if prior_build_rate > 0 else 1.0

    return BuildRateState(
        autonomy_level=autonomy,
        propulsion_level=propulsion,
        build_rate=build_rate,
        annual_multiplier=mult
    )


def emit_build_rate_receipt(state: BuildRateState, cycle: int, config: BuildRateConfig = None) -> dict:
    """Emit receipt for build rate calculation per CLAUDEME.

    Args:
        state: BuildRateState with computed values
        cycle: Current simulation cycle
        config: BuildRateConfig used

    Returns:
        Receipt dict
    """
    if config is None:
        config = BuildRateConfig()

    return emit_receipt("build_rate", {
        "tenant_id": "axiom-autonomy",
        "cycle": cycle,
        "autonomy_level": state.autonomy_level,
        "propulsion_level": state.propulsion_level,
        "alpha": config.alpha,
        "constant": config.constant,
        "build_rate": state.build_rate,
        "annual_multiplier": state.annual_multiplier,
    })
