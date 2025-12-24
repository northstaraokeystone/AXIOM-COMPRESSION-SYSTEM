"""tiers - Multi-Tier Autonomy Framework

LEO → Mars → Deep-space autonomy tier transitions.

Light-delay creates FORCED isolation zones. Each zone must compress
reality locally or die. This IS the evolutionary pressure.
"""

from .autonomy_tiers import (
    AutonomyTier,
    TierConfig,
    TierTransitionResult,
    calculate_tier_decision_capacity,
    tier_transition,
    earth_input_by_tier,
    get_tier_from_delay,
    LIGHT_DELAY_LEO_SEC,
    LIGHT_DELAY_MARS_SEC,
    LIGHT_DELAY_DEEP_SPACE_SEC,
)

__all__ = [
    "AutonomyTier",
    "TierConfig",
    "TierTransitionResult",
    "calculate_tier_decision_capacity",
    "tier_transition",
    "earth_input_by_tier",
    "get_tier_from_delay",
    "LIGHT_DELAY_LEO_SEC",
    "LIGHT_DELAY_MARS_SEC",
    "LIGHT_DELAY_DEEP_SPACE_SEC",
]
