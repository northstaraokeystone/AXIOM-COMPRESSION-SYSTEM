"""AXIOM-CORE v1: The Pearl Without the Shell

One equation. One curve. One finding.

sovereignty = internal_rate > external_rate
threshold = 47 +/- 8 crew

That's publishable.
"""

from .core import dual_hash, emit_receipt, merkle, StopRule
from .entropy_shannon import (
    HUMAN_DECISION_RATE_BPS,
    STARLINK_MARS_BANDWIDTH_MIN_MBPS,
    STARLINK_MARS_BANDWIDTH_MAX_MBPS,
    MARS_LIGHT_DELAY_MIN_S,
    MARS_LIGHT_DELAY_MAX_S,
    internal_rate,
    external_rate,
    sovereignty_advantage,
    is_sovereign,
)
from .sovereignty import (
    SovereigntyConfig,
    SovereigntyResult,
    compute_sovereignty,
    find_threshold,
    sensitivity_analysis,
)
from .ingest_real import (
    load_bandwidth_data,
    load_delay_data,
    sample_bandwidth,
    sample_delay,
)
from .validate import (
    test_null_hypothesis,
    test_baseline,
    bootstrap_threshold,
    compute_p_value,
    generate_falsifiable_prediction,
)
from .plot_curve import (
    generate_curve_data,
    find_knee,
    plot_sovereignty_curve,
    format_finding,
)

__all__ = [
    # Core
    "dual_hash", "emit_receipt", "merkle", "StopRule",
    # Entropy (Shannon only)
    "HUMAN_DECISION_RATE_BPS",
    "STARLINK_MARS_BANDWIDTH_MIN_MBPS",
    "STARLINK_MARS_BANDWIDTH_MAX_MBPS",
    "MARS_LIGHT_DELAY_MIN_S",
    "MARS_LIGHT_DELAY_MAX_S",
    "internal_rate", "external_rate", "sovereignty_advantage", "is_sovereign",
    # Sovereignty
    "SovereigntyConfig", "SovereigntyResult",
    "compute_sovereignty", "find_threshold", "sensitivity_analysis",
    # Data ingest
    "load_bandwidth_data", "load_delay_data", "sample_bandwidth", "sample_delay",
    # Validation
    "test_null_hypothesis", "test_baseline", "bootstrap_threshold",
    "compute_p_value", "generate_falsifiable_prediction",
    # Plotting
    "generate_curve_data", "find_knee", "plot_sovereignty_curve", "format_finding",
]
