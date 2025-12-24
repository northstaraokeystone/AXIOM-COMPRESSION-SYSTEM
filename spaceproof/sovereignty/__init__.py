"""SpaceProof Sovereignty Module.

Calculates autonomy thresholds for space operations.
Includes Mars-specific computational sovereignty simulator.
"""

# Re-export from core sovereignty module
from spaceproof.sovereignty_core import (
    HUMAN_DECISION_RATE_BPS,
    MARS_LIGHT_DELAY_AVG_S,
    MARS_LIGHT_DELAY_MAX_S,
    MARS_LIGHT_DELAY_MIN_S,
    STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
    STARLINK_MARS_BANDWIDTH_MAX_MBPS,
    STARLINK_MARS_BANDWIDTH_MIN_MBPS,
    TENANT_ID,
    SovereigntyConfig,
    SovereigntyResult,
    compute_sovereignty,
    external_rate,
    find_threshold,
    internal_rate,
    is_sovereign,
    sensitivity_analysis,
    sovereignty_advantage,
)

# Import Mars submodule
from . import mars

__all__ = [
    # Core sovereignty
    "SovereigntyConfig",
    "SovereigntyResult",
    "compute_sovereignty",
    "find_threshold",
    "internal_rate",
    "external_rate",
    "sovereignty_advantage",
    "is_sovereign",
    "sensitivity_analysis",
    # Constants
    "HUMAN_DECISION_RATE_BPS",
    "STARLINK_MARS_BANDWIDTH_MIN_MBPS",
    "STARLINK_MARS_BANDWIDTH_MAX_MBPS",
    "STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS",
    "MARS_LIGHT_DELAY_MIN_S",
    "MARS_LIGHT_DELAY_MAX_S",
    "MARS_LIGHT_DELAY_AVG_S",
    "TENANT_ID",
    # Mars submodule
    "mars",
]
