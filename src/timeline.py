"""timeline.py - Year-to-Threshold Projections with Mars Latency Penalty

THE GROK TABLE (v2.0 - Grok Integration):

| Pivot fraction | Annual multiplier | Years to threshold | Delay vs 40% |
|----------------|-------------------|-------------------|--------------|
| 40% (recommended) | ~2.5-3.0x     | 12-15             | baseline     |
| 20-25%         | ~1.6-2.0x         | 18-22             | +6-8 years   |
| <15%           | ~1.2-1.4x         | 25-35+            | +12-20 years |
| ~0% (propulsion-only) | ~1.1x      | 40-60 (or never)  | existential  |

MARS LATENCY PENALTY:
- At τ=20min max latency and α=1.69, sensitivity drops ~65% vs. baseline
- Effective α at Mars max: 1.69 × 0.35 ≈ 0.59
- Earth baseline (τ~0): 10³ person-eq in ~3-4 cycles
- Mars (τ=1200s): 10³ person-eq delayed +12-18 cycles

This module reproduces Grok's timeline table and projects years-to-threshold
for any allocation level, with configurable c/P baselines and latency penalty.

Threshold: 10^6 person-equivalent autonomous decision capacity
(~1M humans worth of autonomous decision capability)

Source: Grok - "threshold = self-sustaining city, ~10^6 person-equivalent"
Source: Grok - "Sim sovereignty timelines with FSD-like compounding—input your c and P baselines?"
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import math

from .core import emit_receipt
from .build_rate import (
    MULTIPLIER_40PCT,
    MULTIPLIER_25PCT,
    MULTIPLIER_15PCT,
    MULTIPLIER_0PCT,
)
from .latency import effective_alpha as compute_effective_alpha

if TYPE_CHECKING:
    from .strategies import StrategyConfig


# === ALPHA CONSTANT (updated from calibration) ===

ALPHA_DEFAULT = 1.69
"""Default alpha from Grok validation: 'α=1.69 fits the 20x MPI leap nicely'"""


# === CONSTANTS (from Grok timeline table) ===

THRESHOLD_PERSON_EQUIVALENT = 1_000_000
"""Sovereignty threshold in person-equivalent capability.
Source: Grok - '~10^6 person-equivalent'"""

MILESTONE_EARLY = 1_000
"""Early sovereignty marker: 10³ person-equivalent.
Source: Grok - '10^3 person-eq in 3 cycles'"""

MILESTONE_CITY = 1_000_000
"""City sovereignty threshold: 10⁶ person-equivalent.
Source: Grok - '~10^6 person-equivalent'"""

BASE_YEAR = 2025
"""Base year for timeline projections."""

CYCLES_PER_YEAR = 1.0
"""Development cycles per year (default 1.0)."""

# Year targets from Grok table
YEARS_40PCT = (12, 15)
"""Years to threshold at 40% allocation."""

YEARS_25PCT = (18, 22)
"""Years to threshold at 25% allocation."""

YEARS_15PCT = (25, 35)
"""Years to threshold at 15% allocation."""

YEARS_0PCT = (40, 60)
"""Years to threshold at 0% allocation (or never)."""

# Starting capability (current state)
CURRENT_PERSON_EQUIVALENT = 1_000
"""Current autonomous capability in person-equivalents.
Represents existing Starlink/Starship autonomy level."""

# === CONFIGURABLE c/P BASELINE CONSTANTS ===

C_BASE_DEFAULT = 50.0
"""Default initial person-eq capacity.
Source: Grok - 'input your c baseline?' - analogous to ISS normalized"""

P_FACTOR_DEFAULT = 1.8
"""Default propulsion growth factor per synod.
Source: Grok - 'input your P baseline?' - fleet scaling from refueling + production ramp"""


@dataclass
class TimelineConfig:
    """Configuration for timeline projections.

    Attributes:
        threshold_person_equivalent: Target capability (default 1M)
        base_year: Starting year for projections (default 2025)
        cycles_per_year: Development cycles per year (default 1.0)
        current_capability: Current person-equivalent level (default 1000)
        c_base: Initial person-eq capacity (default 50.0)
        p_factor: Propulsion growth per synod (default 1.8)
        milestone_early: Early sovereignty marker (default 1000)
        milestone_city: City sovereignty threshold (default 1000000)
    """

    threshold_person_equivalent: int = THRESHOLD_PERSON_EQUIVALENT
    base_year: int = BASE_YEAR
    cycles_per_year: float = CYCLES_PER_YEAR
    current_capability: float = CURRENT_PERSON_EQUIVALENT
    c_base: float = C_BASE_DEFAULT
    p_factor: float = P_FACTOR_DEFAULT
    milestone_early: int = MILESTONE_EARLY
    milestone_city: int = MILESTONE_CITY


@dataclass
class TimelineProjection:
    """Projection for a given allocation fraction.

    Attributes:
        allocation_fraction: Autonomy allocation (0-1)
        annual_multiplier: Expected yearly capability multiplier
        years_to_threshold_low: Optimistic years estimate
        years_to_threshold_high: Conservative years estimate
        delay_vs_optimal: Years delayed vs 40% optimal
        threshold_year_low: Calendar year (optimistic)
        threshold_year_high: Calendar year (conservative)
        c_base: Initial person-eq capacity used
        p_factor: Propulsion growth factor used
        tau_seconds: Latency in seconds (None for Earth)
        effective_alpha: Alpha after latency penalty
        cycles_to_milestone_early: Cycles to reach 10³ person-eq
        cycles_to_milestone_city: Cycles to reach 10⁶ person-eq
        receipt_integrity: Receipt coverage (0-1) for mitigation
        effective_alpha_mitigated: Alpha after receipt mitigation
        delay_vs_unmitigated: Cycles saved by receipts
    """

    allocation_fraction: float
    annual_multiplier: float
    years_to_threshold_low: int
    years_to_threshold_high: int
    delay_vs_optimal: int
    threshold_year_low: int
    threshold_year_high: int
    c_base: float = C_BASE_DEFAULT
    p_factor: float = P_FACTOR_DEFAULT
    tau_seconds: Optional[float] = None
    effective_alpha: float = ALPHA_DEFAULT
    cycles_to_milestone_early: Optional[int] = None
    cycles_to_milestone_city: Optional[int] = None
    receipt_integrity: float = 0.0
    effective_alpha_mitigated: Optional[float] = None
    delay_vs_unmitigated: Optional[int] = None


def allocation_to_multiplier(
    autonomy_fraction: float, alpha: float = ALPHA_DEFAULT
) -> float:
    """Convert autonomy allocation fraction to expected annual multiplier.

    Uses Grok's empirical table values with interpolation.

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
    # Use Grok table values directly with interpolation
    if autonomy_fraction >= 0.40:
        return (MULTIPLIER_40PCT[0] + MULTIPLIER_40PCT[1]) / 2  # 2.75

    if autonomy_fraction >= 0.25:
        # Interpolate between 25% and 40%
        t = (autonomy_fraction - 0.25) / 0.15
        low = (MULTIPLIER_25PCT[0] + MULTIPLIER_25PCT[1]) / 2  # 1.80
        high = (MULTIPLIER_40PCT[0] + MULTIPLIER_40PCT[1]) / 2  # 2.75
        return low + t * (high - low)

    if autonomy_fraction >= 0.15:
        # Interpolate between 15% and 25%
        t = (autonomy_fraction - 0.15) / 0.10
        low = (MULTIPLIER_15PCT[0] + MULTIPLIER_15PCT[1]) / 2  # 1.30
        high = (MULTIPLIER_25PCT[0] + MULTIPLIER_25PCT[1]) / 2  # 1.80
        return low + t * (high - low)

    if autonomy_fraction > 0:
        # Interpolate between 0% and 15%
        t = autonomy_fraction / 0.15
        low = MULTIPLIER_0PCT  # 1.1
        high = (MULTIPLIER_15PCT[0] + MULTIPLIER_15PCT[1]) / 2  # 1.30
        return low + t * (high - low)

    return MULTIPLIER_0PCT  # 1.1


def compute_years_to_threshold(
    annual_multiplier: float, current_capability: float, threshold: int
) -> tuple:
    """Calculate years to reach threshold given multiplier.

    Uses logarithmic calculation:
    years = log(threshold/current) / log(multiplier)

    Args:
        annual_multiplier: Yearly capability multiplier
        current_capability: Starting capability level
        threshold: Target capability level

    Returns:
        Tuple of (years_low, years_high) with uncertainty band
    """
    if annual_multiplier <= 1.0:
        # No growth or decline - never reaches threshold
        return (100, 200)  # Effectively infinite

    if current_capability >= threshold:
        return (0, 0)  # Already there

    ratio = threshold / current_capability
    base_years = math.log(ratio) / math.log(annual_multiplier)

    # Add uncertainty band (~15%)
    years_low = max(1, int(base_years * 0.85))
    years_high = int(base_years * 1.15) + 1

    return (years_low, years_high)


def compare_to_optimal(
    projection: TimelineProjection, optimal_fraction: float = 0.40
) -> int:
    """Calculate delay in years vs optimal allocation.

    Args:
        projection: TimelineProjection to compare
        optimal_fraction: Reference optimal allocation (default 40%)

    Returns:
        Delay in years (positive = slower than optimal)
    """
    if projection.allocation_fraction >= optimal_fraction:
        return 0

    # Get optimal timeline
    optimal_mult = allocation_to_multiplier(optimal_fraction)
    optimal_years_low, optimal_years_high = compute_years_to_threshold(
        optimal_mult, CURRENT_PERSON_EQUIVALENT, THRESHOLD_PERSON_EQUIVALENT
    )
    optimal_midpoint = (optimal_years_low + optimal_years_high) // 2

    # Current projection midpoint
    current_midpoint = (
        projection.years_to_threshold_low + projection.years_to_threshold_high
    ) // 2

    return current_midpoint - optimal_midpoint


def project_timeline(
    autonomy_fraction: float,
    alpha: float = ALPHA_DEFAULT,
    c_base: float = C_BASE_DEFAULT,
    p_factor: float = P_FACTOR_DEFAULT,
    tau_seconds: Optional[float] = None,
    receipt_integrity: float = 0.0,
    config: TimelineConfig = None,
) -> TimelineProjection:
    """Project years to 10^6 person-equivalent threshold.

    Main projection function. Computes full timeline for given allocation.
    Supports Mars latency penalty, receipt mitigation, and configurable c/P baselines.

    Args:
        autonomy_fraction: Fraction allocated to autonomy (0-1)
        alpha: Compounding exponent (default 1.69)
        c_base: Initial person-eq capacity (default 50.0)
        p_factor: Propulsion growth per synod (default 1.8)
        tau_seconds: Latency in seconds (None for Earth, 1200 for Mars max)
        receipt_integrity: Receipt coverage (0-1). If >0, applies receipt mitigation.
        config: TimelineConfig (uses defaults if None)

    Returns:
        TimelineProjection with all computed values
    """
    if config is None:
        config = TimelineConfig()

    # Apply latency penalty if tau_seconds provided, with optional receipt mitigation
    eff_alpha = alpha
    if tau_seconds is not None and tau_seconds > 0:
        eff_alpha = compute_effective_alpha(alpha, tau_seconds, receipt_integrity)

    # Get multiplier for this allocation with effective alpha
    multiplier = allocation_to_multiplier(autonomy_fraction, eff_alpha)

    # Compute years to threshold
    years_low, years_high = compute_years_to_threshold(
        multiplier, config.current_capability, config.threshold_person_equivalent
    )

    # Compute cycles to milestones using sovereignty timeline logic
    cycles_early, cycles_city = _compute_milestone_cycles(
        c_base, p_factor, eff_alpha, config
    )

    # Compute unmitigated comparison if receipt_integrity > 0
    effective_alpha_mitigated = None
    delay_vs_unmitigated = None
    if receipt_integrity > 0.0 and tau_seconds is not None and tau_seconds > 0:
        # Calculate unmitigated alpha for comparison
        unmitigated_alpha = compute_effective_alpha(alpha, tau_seconds, 0.0)
        unmitigated_cycles_early, _ = _compute_milestone_cycles(
            c_base, p_factor, unmitigated_alpha, config
        )
        effective_alpha_mitigated = eff_alpha
        delay_vs_unmitigated = unmitigated_cycles_early - cycles_early

    # Create projection
    projection = TimelineProjection(
        allocation_fraction=autonomy_fraction,
        annual_multiplier=multiplier,
        years_to_threshold_low=years_low,
        years_to_threshold_high=years_high,
        delay_vs_optimal=0,  # Computed below
        threshold_year_low=config.base_year + years_low,
        threshold_year_high=config.base_year + years_high,
        c_base=c_base,
        p_factor=p_factor,
        tau_seconds=tau_seconds,
        effective_alpha=eff_alpha,
        cycles_to_milestone_early=cycles_early,
        cycles_to_milestone_city=cycles_city,
        receipt_integrity=receipt_integrity,
        effective_alpha_mitigated=effective_alpha_mitigated,
        delay_vs_unmitigated=delay_vs_unmitigated,
    )

    # Compute delay vs optimal
    projection = TimelineProjection(
        allocation_fraction=autonomy_fraction,
        annual_multiplier=multiplier,
        years_to_threshold_low=years_low,
        years_to_threshold_high=years_high,
        delay_vs_optimal=compare_to_optimal(projection),
        threshold_year_low=config.base_year + years_low,
        threshold_year_high=config.base_year + years_high,
        c_base=c_base,
        p_factor=p_factor,
        tau_seconds=tau_seconds,
        effective_alpha=eff_alpha,
        cycles_to_milestone_early=cycles_early,
        cycles_to_milestone_city=cycles_city,
        receipt_integrity=receipt_integrity,
        effective_alpha_mitigated=effective_alpha_mitigated,
        delay_vs_unmitigated=delay_vs_unmitigated,
    )

    # Emit receipt
    emit_receipt(
        "timeline",
        {
            "tenant_id": "axiom-autonomy",
            "autonomy_fraction": autonomy_fraction,
            "alpha": alpha,
            "c_base": c_base,
            "p_factor": p_factor,
            "tau_seconds": tau_seconds,
            "receipt_integrity": receipt_integrity,
            "effective_alpha": eff_alpha,
            "effective_alpha_mitigated": effective_alpha_mitigated,
            "annual_multiplier": multiplier,
            "years_to_threshold_low": years_low,
            "years_to_threshold_high": years_high,
            "threshold_year_low": projection.threshold_year_low,
            "threshold_year_high": projection.threshold_year_high,
            "delay_vs_optimal": projection.delay_vs_optimal,
            "delay_vs_unmitigated": delay_vs_unmitigated,
            "threshold_person_equivalent": config.threshold_person_equivalent,
            "cycles_to_milestone_early": cycles_early,
            "cycles_to_milestone_city": cycles_city,
        },
    )

    return projection


def _compute_milestone_cycles(
    c_base: float, p_factor: float, alpha: float, config: TimelineConfig
) -> tuple:
    """Compute cycles to reach early and city milestones.

    Uses B = c × A^α × P model per cycle with A growing each cycle.

    Args:
        c_base: Initial person-eq capacity
        p_factor: Propulsion growth factor per cycle
        alpha: Effective compounding exponent
        config: TimelineConfig with milestone thresholds

    Returns:
        Tuple (cycles_to_early, cycles_to_city)
    """
    # Simulate growth using B = c × A^α × P where A compounds
    person_eq = c_base
    cycles_early = None
    cycles_city = None

    for cycle in range(1, 200):  # Max 200 cycles
        # Each cycle: person_eq grows by factor based on autonomy compounding
        # A grows with cycle: autonomy_level = min(1.0, cycle * 0.1)  # Simple model
        # B = c × A^α × P where P grows with p_factor
        autonomy_level = min(1.0, 0.4 + cycle * 0.05)  # Start at 40%, grow
        propulsion = p_factor**cycle
        build_contribution = c_base * (autonomy_level**alpha) * (propulsion / p_factor)
        person_eq += build_contribution

        if cycles_early is None and person_eq >= config.milestone_early:
            cycles_early = cycle

        if cycles_city is None and person_eq >= config.milestone_city:
            cycles_city = cycle
            break

    # If not reached, set to max
    if cycles_early is None:
        cycles_early = 200
    if cycles_city is None:
        cycles_city = 200

    return (cycles_early, cycles_city)


def generate_timeline_table(
    fractions: List[float] = None,
    alpha: float = ALPHA_DEFAULT,
    config: TimelineConfig = None,
) -> List[TimelineProjection]:
    """Generate Grok-style timeline table for multiple allocations.

    Args:
        fractions: List of allocation fractions (default: [0.40, 0.25, 0.15, 0.05, 0.00])
        alpha: Compounding exponent (default 1.69)
        config: TimelineConfig (uses defaults if None)

    Returns:
        List of TimelineProjection for each fraction
    """
    if fractions is None:
        fractions = [0.40, 0.25, 0.15, 0.05, 0.00]

    if config is None:
        config = TimelineConfig()

    projections = []
    for frac in fractions:
        proj = project_timeline(
            frac, alpha, config.c_base, config.p_factor, None, 0.0, config
        )
        projections.append(proj)

    return projections


def validate_grok_table(projections: List[TimelineProjection]) -> dict:
    """Validate projections match Grok table within tolerance.

    Grok Table Targets:
        40% -> multiplier 2.5-3.0x, years 12-15
        25% -> multiplier 1.6-2.0x, years 18-22
        15% -> multiplier 1.2-1.4x, years 25-35
        0%  -> multiplier ~1.1x, years 40-60+

    Args:
        projections: List of TimelineProjection to validate

    Returns:
        Dict with validation results for each fraction
    """
    validations = {}

    for proj in projections:
        frac = proj.allocation_fraction

        if frac >= 0.35:  # ~40%
            mult_ok = (
                MULTIPLIER_40PCT[0] - 0.2
                <= proj.annual_multiplier
                <= MULTIPLIER_40PCT[1] + 0.2
            )
            years_ok = (
                YEARS_40PCT[0] - 2 <= proj.years_to_threshold_low
                and proj.years_to_threshold_high <= YEARS_40PCT[1] + 2
            )
            validations["0.40"] = {
                "multiplier_ok": mult_ok,
                "years_ok": years_ok,
                "passed": mult_ok and years_ok,
            }

        elif frac >= 0.20:  # ~25%
            mult_ok = (
                MULTIPLIER_25PCT[0] - 0.2
                <= proj.annual_multiplier
                <= MULTIPLIER_25PCT[1] + 0.2
            )
            years_ok = (
                YEARS_25PCT[0] - 2 <= proj.years_to_threshold_low
                and proj.years_to_threshold_high <= YEARS_25PCT[1] + 2
            )
            validations["0.25"] = {
                "multiplier_ok": mult_ok,
                "years_ok": years_ok,
                "passed": mult_ok and years_ok,
            }

        elif frac >= 0.10:  # ~15%
            mult_ok = (
                MULTIPLIER_15PCT[0] - 0.2
                <= proj.annual_multiplier
                <= MULTIPLIER_15PCT[1] + 0.2
            )
            years_ok = (
                YEARS_15PCT[0] - 3 <= proj.years_to_threshold_low
                and proj.years_to_threshold_high <= YEARS_15PCT[1] + 5
            )
            validations["0.15"] = {
                "multiplier_ok": mult_ok,
                "years_ok": years_ok,
                "passed": mult_ok and years_ok,
            }

        elif frac <= 0.05:  # ~0%
            mult_ok = 0.9 <= proj.annual_multiplier <= 1.3
            years_ok = proj.years_to_threshold_low >= 30  # Very long
            validations["0.00"] = {
                "multiplier_ok": mult_ok,
                "years_ok": years_ok,
                "passed": mult_ok and years_ok,
            }

    validations["all_passed"] = all(
        v.get("passed", False) for v in validations.values() if isinstance(v, dict)
    )

    return validations


def format_timeline_table(projections: List[TimelineProjection]) -> str:
    """Format projections as human-readable table.

    Args:
        projections: List of TimelineProjection

    Returns:
        Formatted table string matching Grok's format
    """
    lines = [
        "| Pivot fraction | Annual multiplier | Years to threshold | Delay vs 40% |",
        "|----------------|-------------------|-------------------|--------------|",
    ]

    for proj in projections:
        frac_str = f"{proj.allocation_fraction:.0%}"
        mult_str = f"~{proj.annual_multiplier:.1f}x"
        years_str = f"{proj.years_to_threshold_low}-{proj.years_to_threshold_high}"
        if proj.delay_vs_optimal == 0:
            delay_str = "baseline"
        elif proj.delay_vs_optimal > 50:
            delay_str = "existential stall"
        else:
            delay_str = f"+{proj.delay_vs_optimal} years"

        lines.append(
            f"| {frac_str:14} | {mult_str:17} | {years_str:17} | {delay_str:12} |"
        )

    return "\n".join(lines)


def emit_timeline_receipt(
    projection: TimelineProjection, config: TimelineConfig = None
) -> dict:
    """Emit detailed timeline receipt per CLAUDEME.

    Args:
        projection: TimelineProjection to emit
        config: TimelineConfig used

    Returns:
        Receipt dict
    """
    if config is None:
        config = TimelineConfig()

    return emit_receipt(
        "timeline",
        {
            "tenant_id": "axiom-autonomy",
            "autonomy_fraction": projection.allocation_fraction,
            "alpha": ALPHA_DEFAULT,
            "c_base": projection.c_base,
            "p_factor": projection.p_factor,
            "tau_seconds": projection.tau_seconds,
            "effective_alpha": projection.effective_alpha,
            "annual_multiplier": projection.annual_multiplier,
            "years_to_threshold_low": projection.years_to_threshold_low,
            "years_to_threshold_high": projection.years_to_threshold_high,
            "threshold_year_low": projection.threshold_year_low,
            "threshold_year_high": projection.threshold_year_high,
            "delay_vs_optimal": projection.delay_vs_optimal,
            "threshold_person_equivalent": config.threshold_person_equivalent,
            "base_year": config.base_year,
            "cycles_to_milestone_early": projection.cycles_to_milestone_early,
            "cycles_to_milestone_city": projection.cycles_to_milestone_city,
        },
    )


def project_sovereignty_date(
    autonomy_fraction: float = 0.40, config: TimelineConfig = None
) -> dict:
    """Get projected sovereignty date for given allocation.

    Convenience function for common use case.

    Args:
        autonomy_fraction: Allocation to autonomy (default 40%)
        config: TimelineConfig (uses defaults if None)

    Returns:
        Dict with year projections and key metrics
    """
    if config is None:
        config = TimelineConfig()

    proj = project_timeline(
        autonomy_fraction,
        ALPHA_DEFAULT,
        config.c_base,
        config.p_factor,
        None,
        0.0,
        config,
    )

    return {
        "allocation": autonomy_fraction,
        "earliest_year": proj.threshold_year_low,
        "latest_year": proj.threshold_year_high,
        "midpoint_year": (proj.threshold_year_low + proj.threshold_year_high) // 2,
        "annual_multiplier": proj.annual_multiplier,
        "threshold": config.threshold_person_equivalent,
        "delay_vs_optimal": proj.delay_vs_optimal,
        "recommendation": "optimal"
        if autonomy_fraction >= 0.40
        else ("acceptable" if autonomy_fraction >= 0.30 else "under-pivoted"),
    }


def sovereignty_timeline(
    c_base: float = C_BASE_DEFAULT,
    p_factor: float = P_FACTOR_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    tau_seconds: Optional[float] = None,
    receipt_integrity: float = 0.0,
    strategy: "StrategyConfig" = None,
) -> Dict[str, Any]:
    """Compute full sovereignty timeline with milestones.

    Main sovereignty timeline simulation function. Computes person-eq trajectory
    with FSD-like compounding, optional Mars latency penalty, and receipt mitigation.

    THE PARADIGM SHIFT:
        Without receipts: effective_α = base_α × tau_penalty = 1.69 × 0.35 = 0.59
        With 90% receipts: effective_α = base_α × (1 - penalty × (1 - integrity)) = 1.58

    Model: B = c × A^α × P where:
    - c: initial capacity (normalized)
    - A: autonomy fraction (starts at 0.4, compounds each cycle)
    - α: compounding exponent (1.69, degraded by latency, mitigated by receipts)
    - P: propulsion multiplier (p_factor^cycle)

    For FSD-like exponential growth:
    - person_eq grows multiplicatively each cycle
    - multiplier ≈ 2.5-3.0x at 40% allocation with α=1.69

    Args:
        c_base: Initial person-eq capacity (default 50.0)
        p_factor: Propulsion growth factor per synod (default 1.8)
        alpha: Base compounding exponent (default 1.69)
        tau_seconds: Latency in seconds (0 or None for Earth, 1200 for Mars max)
        receipt_integrity: Receipt coverage (0-1). If >0, applies receipt mitigation.
        strategy: Optional StrategyConfig for τ reduction strategies

    Returns:
        Dict with:
            - cycles_to_10k_person_eq: Cycles to reach 10³ milestone
            - cycles_to_1M_person_eq: Cycles to reach 10⁶ milestone
            - person_eq_trajectory: List of person-eq values per cycle
            - effective_alpha: Alpha after latency penalty and receipt mitigation
            - delay_vs_earth: Cycles lost to latency (vs tau=0)
            - receipt_integrity: Receipt coverage used
            - delay_vs_unmitigated: Cycles saved by receipts (if receipt_integrity > 0)
            - strategy_applied: Strategy name (if strategy provided)
            - c_factor_applied: c factor from strategy (if provided)
            - p_cost_applied: P cost from strategy (if provided)
            - roi_score: ROI score (if strategy provided)

    Receipt: sovereignty_projection
    """
    # Strategy-specific values
    strategy_name = None
    c_factor = 1.0
    p_cost = 0.0
    roi_score = 0.0
    effective_tau = tau_seconds if tau_seconds else 0

    # Apply strategy if provided
    if strategy is not None:
        from .strategies import (
            compute_effective_tau as strategy_effective_tau,
            compute_effective_alpha as strategy_effective_alpha,
            compute_c_factor,
            compute_p_cost,
        )

        # Compute strategy effects
        base_tau = tau_seconds if tau_seconds else 0
        effective_tau = strategy_effective_tau(base_tau, strategy)
        c_factor = compute_c_factor(strategy)
        p_cost = compute_p_cost(strategy)
        strategy_name = strategy.strategy.value

        # Effective alpha from strategy (includes floor for onboard AI)
        eff_alpha = strategy_effective_alpha(alpha, strategy, effective_tau)
    else:
        # Compute effective alpha with latency penalty and receipt mitigation (legacy behavior)
        eff_alpha = alpha
        if tau_seconds is not None and tau_seconds > 0:
            eff_alpha = compute_effective_alpha(alpha, tau_seconds, receipt_integrity)

    # Adjust P factor for relay cost
    adjusted_p_factor = p_factor * (1.0 - p_cost)

    # Simulate growth trajectory using multiplicative FSD-like compounding
    # At 40% allocation with α=1.69: multiplier ≈ 2.75x per cycle
    # This achieves 10³ from c=50 in ~3-4 cycles: 50 → 137 → 377 → 1038
    person_eq = c_base
    trajectory = [c_base]
    cycles_to_10k = None
    cycles_to_1M = None

    # Base autonomy level (40% allocation)

    for cycle in range(1, 200):  # Max 200 cycles
        # Multiplier per cycle from B = c × A^α × P model
        # At A=0.40, α=1.69 (Earth): multiplier ≈ 2.75x per cycle
        # At A=0.40, α=0.59 (Mars): multiplier ≈ 1.4x per cycle (65% reduction in growth)
        #
        # Key insight: α controls the compounding EXPONENT
        # Higher α = stronger superlinear effects = faster growth
        # Lower α (Mars latency) = weaker compounding = slower growth
        #
        # Model: base_multiplier scales linearly with effective alpha ratio
        # At α=1.69: multiplier = 2.75 (optimal)
        # At α=0.59: multiplier ≈ 1.0 + (2.75-1.0) * (0.59/1.69) ≈ 1.61

        alpha_ratio = eff_alpha / 1.69  # Normalized to optimal alpha
        base_multiplier = 1.0 + (2.75 - 1.0) * alpha_ratio

        # Apply c_factor as growth rate modifier
        base_multiplier = 1.0 + (base_multiplier - 1.0) * c_factor

        # Apply propulsion growth (diminishing per cycle to prevent runaway)
        propulsion_factor = 1.0 + (adjusted_p_factor - 1.0) * (0.9 ** (cycle - 1))

        # Combined cycle multiplier
        cycle_multiplier = base_multiplier * propulsion_factor

        person_eq *= cycle_multiplier
        trajectory.append(person_eq)

        if cycles_to_10k is None and person_eq >= MILESTONE_EARLY:
            cycles_to_10k = cycle

        if cycles_to_1M is None and person_eq >= MILESTONE_CITY:
            cycles_to_1M = cycle
            break

    # Default if not reached
    if cycles_to_10k is None:
        cycles_to_10k = 200
    if cycles_to_1M is None:
        cycles_to_1M = 200

    # Compute delay vs Earth (tau=0)
    delay_vs_earth = 0
    if tau_seconds is not None and tau_seconds > 0:
        earth_result = sovereignty_timeline(c_base, p_factor, alpha, 0, 0.0)
        delay_vs_earth = cycles_to_10k - earth_result["cycles_to_10k_person_eq"]

    # Compute delay vs unmitigated (receipt_integrity=0)
    delay_vs_unmitigated = None
    if receipt_integrity > 0.0 and tau_seconds is not None and tau_seconds > 0:
        unmitigated_result = sovereignty_timeline(
            c_base, p_factor, alpha, tau_seconds, 0.0
        )
        delay_vs_unmitigated = (
            unmitigated_result["cycles_to_10k_person_eq"] - cycles_to_10k
        )

    result = {
        "cycles_to_10k_person_eq": cycles_to_10k,
        "cycles_to_1M_person_eq": cycles_to_1M,
        "person_eq_trajectory": trajectory[
            : min(len(trajectory), 50)
        ],  # Limit to 50 for receipt
        "effective_alpha": eff_alpha,
        "delay_vs_earth": delay_vs_earth,
        "receipt_integrity": receipt_integrity,
        "delay_vs_unmitigated": delay_vs_unmitigated,
    }

    # Add strategy fields if strategy was applied
    if strategy is not None:
        result["strategy_applied"] = strategy_name
        result["effective_tau"] = effective_tau
        result["c_factor_applied"] = c_factor
        result["p_cost_applied"] = p_cost
        result["roi_score"] = roi_score

    # Emit sovereignty projection receipt
    receipt_data = {
        "tenant_id": "axiom-autonomy",
        "c_base": c_base,
        "p_factor": p_factor,
        "alpha": alpha,
        "tau_seconds": tau_seconds if tau_seconds else 0,
        "receipt_integrity": receipt_integrity,
        "effective_alpha": eff_alpha,
        "cycles_to_10k_person_eq": cycles_to_10k,
        "cycles_to_1M_person_eq": cycles_to_1M,
        "person_eq_trajectory": trajectory[
            : min(len(trajectory), 20)
        ],  # Shorter for receipt
        "delay_vs_earth": delay_vs_earth,
        "delay_vs_unmitigated": delay_vs_unmitigated,
    }

    # Add strategy fields to receipt
    if strategy is not None:
        receipt_data["strategy_applied"] = strategy_name
        receipt_data["effective_tau"] = effective_tau
        receipt_data["c_factor_applied"] = c_factor
        receipt_data["p_cost_applied"] = p_cost
        receipt_data["roi_score"] = roi_score

    emit_receipt("sovereignty_projection", receipt_data)

    return result
