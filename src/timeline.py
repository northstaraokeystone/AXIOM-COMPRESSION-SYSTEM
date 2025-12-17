"""timeline.py - Year-to-Threshold Projections

THE GROK TABLE (v2.0 - Grok Integration):

| Pivot fraction | Annual multiplier | Years to threshold | Delay vs 40% |
|----------------|-------------------|-------------------|--------------|
| 40% (recommended) | ~2.5-3.0x     | 12-15             | baseline     |
| 20-25%         | ~1.6-2.0x         | 18-22             | +6-8 years   |
| <15%           | ~1.2-1.4x         | 25-35+            | +12-20 years |
| ~0% (propulsion-only) | ~1.1x      | 40-60 (or never)  | existential  |

This module reproduces Grok's timeline table and projects years-to-threshold
for any allocation level.

Threshold: 10^6 person-equivalent autonomous decision capacity
(~1M humans worth of autonomous decision capability)

Source: Grok - "threshold = self-sustaining city, ~10^6 person-equivalent"
"""

from dataclasses import dataclass
from typing import List, Optional
import math

from .core import emit_receipt
from .build_rate import (
    ALPHA_BASELINE,
    MULTIPLIER_40PCT,
    MULTIPLIER_25PCT,
    MULTIPLIER_15PCT,
    MULTIPLIER_0PCT,
)


# === CONSTANTS (from Grok timeline table) ===

THRESHOLD_PERSON_EQUIVALENT = 1_000_000
"""Sovereignty threshold in person-equivalent capability.
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


@dataclass
class TimelineConfig:
    """Configuration for timeline projections.

    Attributes:
        threshold_person_equivalent: Target capability (default 1M)
        base_year: Starting year for projections (default 2025)
        cycles_per_year: Development cycles per year (default 1.0)
        current_capability: Current person-equivalent level (default 1000)
    """
    threshold_person_equivalent: int = THRESHOLD_PERSON_EQUIVALENT
    base_year: int = BASE_YEAR
    cycles_per_year: float = CYCLES_PER_YEAR
    current_capability: float = CURRENT_PERSON_EQUIVALENT


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
    """
    allocation_fraction: float
    annual_multiplier: float
    years_to_threshold_low: int
    years_to_threshold_high: int
    delay_vs_optimal: int
    threshold_year_low: int
    threshold_year_high: int


def allocation_to_multiplier(autonomy_fraction: float, alpha: float = ALPHA_BASELINE) -> float:
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
    annual_multiplier: float,
    current_capability: float,
    threshold: int
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


def compare_to_optimal(projection: TimelineProjection, optimal_fraction: float = 0.40) -> int:
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
        optimal_mult,
        CURRENT_PERSON_EQUIVALENT,
        THRESHOLD_PERSON_EQUIVALENT
    )
    optimal_midpoint = (optimal_years_low + optimal_years_high) // 2

    # Current projection midpoint
    current_midpoint = (projection.years_to_threshold_low + projection.years_to_threshold_high) // 2

    return current_midpoint - optimal_midpoint


def project_timeline(
    autonomy_fraction: float,
    alpha: float = ALPHA_BASELINE,
    config: TimelineConfig = None
) -> TimelineProjection:
    """Project years to 10^6 person-equivalent threshold.

    Main projection function. Computes full timeline for given allocation.

    Args:
        autonomy_fraction: Fraction allocated to autonomy (0-1)
        alpha: Compounding exponent (default 1.8)
        config: TimelineConfig (uses defaults if None)

    Returns:
        TimelineProjection with all computed values
    """
    if config is None:
        config = TimelineConfig()

    # Get multiplier for this allocation
    multiplier = allocation_to_multiplier(autonomy_fraction, alpha)

    # Compute years to threshold
    years_low, years_high = compute_years_to_threshold(
        multiplier,
        config.current_capability,
        config.threshold_person_equivalent
    )

    # Create projection
    projection = TimelineProjection(
        allocation_fraction=autonomy_fraction,
        annual_multiplier=multiplier,
        years_to_threshold_low=years_low,
        years_to_threshold_high=years_high,
        delay_vs_optimal=0,  # Computed below
        threshold_year_low=config.base_year + years_low,
        threshold_year_high=config.base_year + years_high
    )

    # Compute delay vs optimal
    projection = TimelineProjection(
        allocation_fraction=autonomy_fraction,
        annual_multiplier=multiplier,
        years_to_threshold_low=years_low,
        years_to_threshold_high=years_high,
        delay_vs_optimal=compare_to_optimal(projection),
        threshold_year_low=config.base_year + years_low,
        threshold_year_high=config.base_year + years_high
    )

    # Emit receipt
    emit_receipt("timeline", {
        "tenant_id": "axiom-autonomy",
        "autonomy_fraction": autonomy_fraction,
        "alpha": alpha,
        "annual_multiplier": multiplier,
        "years_to_threshold_low": years_low,
        "years_to_threshold_high": years_high,
        "threshold_year_low": projection.threshold_year_low,
        "threshold_year_high": projection.threshold_year_high,
        "delay_vs_optimal": projection.delay_vs_optimal,
        "threshold_person_equivalent": config.threshold_person_equivalent,
    })

    return projection


def generate_timeline_table(
    fractions: List[float] = None,
    alpha: float = ALPHA_BASELINE,
    config: TimelineConfig = None
) -> List[TimelineProjection]:
    """Generate Grok-style timeline table for multiple allocations.

    Args:
        fractions: List of allocation fractions (default: [0.40, 0.25, 0.15, 0.05, 0.00])
        alpha: Compounding exponent
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
        proj = project_timeline(frac, alpha, config)
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
            mult_ok = MULTIPLIER_40PCT[0] - 0.2 <= proj.annual_multiplier <= MULTIPLIER_40PCT[1] + 0.2
            years_ok = YEARS_40PCT[0] - 2 <= proj.years_to_threshold_low and proj.years_to_threshold_high <= YEARS_40PCT[1] + 2
            validations["0.40"] = {"multiplier_ok": mult_ok, "years_ok": years_ok, "passed": mult_ok and years_ok}

        elif frac >= 0.20:  # ~25%
            mult_ok = MULTIPLIER_25PCT[0] - 0.2 <= proj.annual_multiplier <= MULTIPLIER_25PCT[1] + 0.2
            years_ok = YEARS_25PCT[0] - 2 <= proj.years_to_threshold_low and proj.years_to_threshold_high <= YEARS_25PCT[1] + 2
            validations["0.25"] = {"multiplier_ok": mult_ok, "years_ok": years_ok, "passed": mult_ok and years_ok}

        elif frac >= 0.10:  # ~15%
            mult_ok = MULTIPLIER_15PCT[0] - 0.2 <= proj.annual_multiplier <= MULTIPLIER_15PCT[1] + 0.2
            years_ok = YEARS_15PCT[0] - 3 <= proj.years_to_threshold_low and proj.years_to_threshold_high <= YEARS_15PCT[1] + 5
            validations["0.15"] = {"multiplier_ok": mult_ok, "years_ok": years_ok, "passed": mult_ok and years_ok}

        elif frac <= 0.05:  # ~0%
            mult_ok = 0.9 <= proj.annual_multiplier <= 1.3
            years_ok = proj.years_to_threshold_low >= 30  # Very long
            validations["0.00"] = {"multiplier_ok": mult_ok, "years_ok": years_ok, "passed": mult_ok and years_ok}

    validations["all_passed"] = all(v.get("passed", False) for v in validations.values() if isinstance(v, dict))

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

        lines.append(f"| {frac_str:14} | {mult_str:17} | {years_str:17} | {delay_str:12} |")

    return "\n".join(lines)


def emit_timeline_receipt(projection: TimelineProjection, config: TimelineConfig = None) -> dict:
    """Emit detailed timeline receipt per CLAUDEME.

    Args:
        projection: TimelineProjection to emit
        config: TimelineConfig used

    Returns:
        Receipt dict
    """
    if config is None:
        config = TimelineConfig()

    return emit_receipt("timeline", {
        "tenant_id": "axiom-autonomy",
        "autonomy_fraction": projection.allocation_fraction,
        "alpha": ALPHA_BASELINE,
        "annual_multiplier": projection.annual_multiplier,
        "years_to_threshold_low": projection.years_to_threshold_low,
        "years_to_threshold_high": projection.years_to_threshold_high,
        "threshold_year_low": projection.threshold_year_low,
        "threshold_year_high": projection.threshold_year_high,
        "delay_vs_optimal": projection.delay_vs_optimal,
        "threshold_person_equivalent": config.threshold_person_equivalent,
        "base_year": config.base_year,
    })


def project_sovereignty_date(
    autonomy_fraction: float = 0.40,
    config: TimelineConfig = None
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

    proj = project_timeline(autonomy_fraction, ALPHA_BASELINE, config)

    return {
        "allocation": autonomy_fraction,
        "earliest_year": proj.threshold_year_low,
        "latest_year": proj.threshold_year_high,
        "midpoint_year": (proj.threshold_year_low + proj.threshold_year_high) // 2,
        "annual_multiplier": proj.annual_multiplier,
        "threshold": config.threshold_person_equivalent,
        "delay_vs_optimal": proj.delay_vs_optimal,
        "recommendation": "optimal" if autonomy_fraction >= 0.40 else (
            "acceptable" if autonomy_fraction >= 0.30 else "under-pivoted"
        )
    }
