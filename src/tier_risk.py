"""tier_risk.py - 3-Tier Probability x Impact Risk Model

THE RISK FRAMEWORK (v2.0 - Grok Integration):

Tier 1 (60-80% probability, medium impact):
    - 5-10 year delay to sovereignty
    - Forced over-investment in fragile bandwidth
    - Higher cumulative launch costs

Tier 2 (30-50% probability, high impact):
    - Program stall (post-Apollo precedent)
    - Technical debt lock-in from tele-op architecture
    - Political/funding window closure on 20+ year timelines

Tier 3 (5-15% probability, existential impact):
    - Never escapes Earth dependency
    - Cascading failure during solar conjunction blackout
    - Single-planet species risk unchanged

Source: Grok - "Risk tiers" with probability x impact matrix
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from .core import emit_receipt


# === CONSTANTS (from Grok risk framework) ===

TIER_1_PROB_RANGE = (0.60, 0.80)
"""Tier 1 probability range. Source: Grok - '60-80%'"""

TIER_2_PROB_RANGE = (0.30, 0.50)
"""Tier 2 probability range. Source: Grok - '30-50%'"""

TIER_3_PROB_RANGE = (0.05, 0.15)
"""Tier 3 probability range. Source: Grok - '5-15%'"""

# Impact classes
IMPACT_MEDIUM = "medium"
IMPACT_HIGH = "high"
IMPACT_EXISTENTIAL = "existential"

# Allocation thresholds that trigger risk tiers
UNDER_PIVOT_THRESHOLD = 0.30
"""Below this autonomy allocation, all risk tiers activate."""

SEVERE_UNDER_PIVOT_THRESHOLD = 0.20
"""Below this, Tier 2 and 3 probabilities increase."""

CRITICAL_UNDER_PIVOT_THRESHOLD = 0.10
"""Below this, existential risk probability maximizes."""


class RiskTier(Enum):
    """Risk tier classification."""
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


@dataclass
class TierRiskProfile:
    """Risk profile for a single tier.

    Attributes:
        tier: RiskTier enum value
        probability_low: Lower bound of probability range
        probability_high: Upper bound of probability range
        impact_class: "medium", "high", or "existential"
        failure_modes: List of failure mode descriptions
        mitigation_available: True if mitigation exists
    """
    tier: RiskTier
    probability_low: float
    probability_high: float
    impact_class: str
    failure_modes: List[str]
    mitigation_available: bool


def tier_1_risk(autonomy_fraction: float) -> TierRiskProfile:
    """Generate Tier 1 risk profile.

    Tier 1: High probability (60-80%), medium impact
    - 5-10 year delay to sovereignty
    - Forced bandwidth over-investment
    - Higher launch costs

    Args:
        autonomy_fraction: Current autonomy allocation (0-1)

    Returns:
        TierRiskProfile for Tier 1
    """
    # Base probabilities from Grok
    prob_low, prob_high = TIER_1_PROB_RANGE

    # Adjust based on allocation (worse allocation = higher probability)
    if autonomy_fraction >= UNDER_PIVOT_THRESHOLD:
        # Adequate allocation - reduce probability
        prob_low = 0.20
        prob_high = 0.40
    elif autonomy_fraction < CRITICAL_UNDER_PIVOT_THRESHOLD:
        # Critical under-pivot - max probability
        prob_low = 0.80
        prob_high = 0.95

    failure_modes = [
        "5-10 year delay to sovereignty threshold",
        "Forced over-investment in fragile bandwidth infrastructure",
        "Higher cumulative launch costs due to tele-op dependency",
        "Reduced mission flexibility during communication blackouts",
    ]

    return TierRiskProfile(
        tier=RiskTier.TIER_1,
        probability_low=prob_low,
        probability_high=prob_high,
        impact_class=IMPACT_MEDIUM,
        failure_modes=failure_modes,
        mitigation_available=True  # Can increase autonomy allocation
    )


def tier_2_risk(years_to_threshold: int) -> TierRiskProfile:
    """Generate Tier 2 risk profile.

    Tier 2: Medium probability (30-50%), high impact
    - Program stall (post-Apollo precedent)
    - Technical debt lock-in
    - Political/funding window closure

    Args:
        years_to_threshold: Projected years to sovereignty

    Returns:
        TierRiskProfile for Tier 2
    """
    # Base probabilities from Grok
    prob_low, prob_high = TIER_2_PROB_RANGE

    # Adjust based on timeline (longer = higher probability)
    if years_to_threshold <= 15:
        # Fast timeline - low risk
        prob_low = 0.10
        prob_high = 0.25
    elif years_to_threshold >= 25:
        # Slow timeline - high risk (post-Apollo precedent)
        prob_low = 0.50
        prob_high = 0.70

    failure_modes = [
        "Program stall following post-Apollo precedent",
        "Technical debt lock-in from early tele-op architecture",
        "Political/funding window closure on 20+ year timelines",
        "Loss of institutional knowledge and momentum",
        "Retrofit nightmares from architecture decisions",
    ]

    return TierRiskProfile(
        tier=RiskTier.TIER_2,
        probability_low=prob_low,
        probability_high=prob_high,
        impact_class=IMPACT_HIGH,
        failure_modes=failure_modes,
        mitigation_available=True  # Can accelerate with more autonomy
    )


def tier_3_risk(autonomy_fraction: float) -> TierRiskProfile:
    """Generate Tier 3 risk profile.

    Tier 3: Low probability (5-15%), existential impact
    - Never escapes Earth dependency
    - Cascading failure during conjunction
    - Single-planet species unchanged

    Args:
        autonomy_fraction: Current autonomy allocation (0-1)

    Returns:
        TierRiskProfile for Tier 3
    """
    # Base probabilities from Grok
    prob_low, prob_high = TIER_3_PROB_RANGE

    # Adjust based on allocation (worse = higher existential risk)
    if autonomy_fraction >= UNDER_PIVOT_THRESHOLD:
        # Adequate - minimal existential risk
        prob_low = 0.01
        prob_high = 0.05
    elif autonomy_fraction < CRITICAL_UNDER_PIVOT_THRESHOLD:
        # Critical - elevated existential risk
        prob_low = 0.15
        prob_high = 0.25
    elif autonomy_fraction == 0:
        # Zero autonomy - maximum existential risk
        prob_low = 0.30
        prob_high = 0.50

    failure_modes = [
        "Never escapes Earth dependency - permanent tether",
        "Cascading failure during solar conjunction blackout",
        "Single-planet species risk remains unchanged",
        "Civilization capability permanently Earth-bound",
        "Mars colony unable to self-sustain after Earth disruption",
    ]

    return TierRiskProfile(
        tier=RiskTier.TIER_3,
        probability_low=prob_low,
        probability_high=prob_high,
        impact_class=IMPACT_EXISTENTIAL,
        failure_modes=failure_modes,
        mitigation_available=True  # Still possible to pivot
    )


def assess_tier_risk(
    autonomy_fraction: float,
    years_to_threshold: int
) -> List[TierRiskProfile]:
    """Assess all applicable risk tiers for given allocation.

    Main risk assessment function. Returns all three tiers with
    probability adjustments based on current state.

    Args:
        autonomy_fraction: Current autonomy allocation (0-1)
        years_to_threshold: Projected years to sovereignty

    Returns:
        List of TierRiskProfile for all three tiers
    """
    profiles = [
        tier_1_risk(autonomy_fraction),
        tier_2_risk(years_to_threshold),
        tier_3_risk(autonomy_fraction),
    ]

    # Emit receipts for each tier
    for profile in profiles:
        emit_receipt("tier_risk", {
            "tenant_id": "axiom-autonomy",
            "autonomy_fraction": autonomy_fraction,
            "tier": profile.tier.value,
            "probability_low": profile.probability_low,
            "probability_high": profile.probability_high,
            "impact_class": profile.impact_class,
            "failure_modes": profile.failure_modes,
            "mitigation_available": profile.mitigation_available,
        })

    return profiles


def aggregate_risk_score(profiles: List[TierRiskProfile]) -> float:
    """Calculate aggregate risk score from tier profiles.

    Uses expected value calculation:
    score = sum(probability_midpoint * impact_weight)

    Impact weights:
    - medium: 1.0
    - high: 3.0
    - existential: 10.0

    Args:
        profiles: List of TierRiskProfile

    Returns:
        Aggregate risk score (higher = worse)
    """
    impact_weights = {
        IMPACT_MEDIUM: 1.0,
        IMPACT_HIGH: 3.0,
        IMPACT_EXISTENTIAL: 10.0,
    }

    total_score = 0.0

    for profile in profiles:
        prob_mid = (profile.probability_low + profile.probability_high) / 2
        weight = impact_weights.get(profile.impact_class, 1.0)
        total_score += prob_mid * weight

    return total_score


def is_existential(profiles: List[TierRiskProfile]) -> bool:
    """Check if any tier has existential impact.

    Args:
        profiles: List of TierRiskProfile

    Returns:
        True if any tier has existential impact class
    """
    return any(p.impact_class == IMPACT_EXISTENTIAL for p in profiles)


def get_highest_probability_tier(profiles: List[TierRiskProfile]) -> TierRiskProfile:
    """Get the tier with highest probability midpoint.

    Args:
        profiles: List of TierRiskProfile

    Returns:
        TierRiskProfile with highest probability
    """
    return max(profiles, key=lambda p: (p.probability_low + p.probability_high) / 2)


def get_highest_impact_tier(profiles: List[TierRiskProfile]) -> TierRiskProfile:
    """Get the tier with highest impact.

    Args:
        profiles: List of TierRiskProfile

    Returns:
        TierRiskProfile with highest impact (existential > high > medium)
    """
    impact_order = {IMPACT_EXISTENTIAL: 3, IMPACT_HIGH: 2, IMPACT_MEDIUM: 1}
    return max(profiles, key=lambda p: impact_order.get(p.impact_class, 0))


def risk_summary(profiles: List[TierRiskProfile]) -> dict:
    """Generate risk summary for reporting.

    Args:
        profiles: List of TierRiskProfile

    Returns:
        Dict with risk summary metrics
    """
    aggregate = aggregate_risk_score(profiles)
    existential = is_existential(profiles)
    highest_prob = get_highest_probability_tier(profiles)
    highest_impact = get_highest_impact_tier(profiles)

    # Risk level classification
    if aggregate >= 3.0:
        risk_level = "CRITICAL"
    elif aggregate >= 1.5:
        risk_level = "HIGH"
    elif aggregate >= 0.5:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "aggregate_score": aggregate,
        "risk_level": risk_level,
        "has_existential_risk": existential,
        "highest_probability_tier": highest_prob.tier.value,
        "highest_impact_tier": highest_impact.tier.value,
        "tier_1_prob_range": (profiles[0].probability_low, profiles[0].probability_high),
        "tier_2_prob_range": (profiles[1].probability_low, profiles[1].probability_high),
        "tier_3_prob_range": (profiles[2].probability_low, profiles[2].probability_high),
        "mitigation_possible": all(p.mitigation_available for p in profiles),
    }


def format_risk_assessment(profiles: List[TierRiskProfile]) -> str:
    """Format risk assessment as human-readable report.

    Args:
        profiles: List of TierRiskProfile

    Returns:
        Formatted risk report string
    """
    summary = risk_summary(profiles)

    lines = [
        "=" * 60,
        f"RISK ASSESSMENT - Level: {summary['risk_level']}",
        "=" * 60,
        "",
        f"Aggregate Risk Score: {summary['aggregate_score']:.2f}",
        f"Existential Risk Present: {summary['has_existential_risk']}",
        "",
    ]

    for profile in profiles:
        lines.append(f"TIER {profile.tier.value} ({profile.impact_class.upper()} IMPACT)")
        lines.append(f"  Probability: {profile.probability_low:.0%} - {profile.probability_high:.0%}")
        lines.append(f"  Failure Modes:")
        for mode in profile.failure_modes[:3]:  # Top 3
            lines.append(f"    - {mode}")
        lines.append(f"  Mitigation Available: {profile.mitigation_available}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def emit_tier_risk_receipt(
    profile: TierRiskProfile,
    autonomy_fraction: float
) -> dict:
    """Emit detailed tier risk receipt per CLAUDEME.

    Args:
        profile: TierRiskProfile to emit
        autonomy_fraction: Current autonomy allocation

    Returns:
        Receipt dict
    """
    return emit_receipt("tier_risk", {
        "tenant_id": "axiom-autonomy",
        "autonomy_fraction": autonomy_fraction,
        "tier": profile.tier.value,
        "probability_low": profile.probability_low,
        "probability_high": profile.probability_high,
        "impact_class": profile.impact_class,
        "failure_modes": profile.failure_modes,
        "mitigation_available": profile.mitigation_available,
    })
