"""AXIOM-CORE v1: The Pearl Without the Shell

One equation. One curve. One finding.

sovereignty = internal_rate > external_rate
threshold = 47 +/- 8 crew

That's publishable.

v2: Helper layer + Support infrastructure + Optimization agent
- optimize.py: Thompson sampling selection pressure
- helper.py: HARVEST → HYPOTHESIZE → GATE → ACTUATE
- support.py: L0-L4 receipt level infrastructure
- sim.py: Integrated simulation framework

v3: Modular architecture refactor (Dec 2025)
- constants.py: Centralized physics constants
- receipts.py: Receipt emission helpers (DRY)
- stoprules.py: Centralized stoprule registry
- pruning_*.py: Split entropy pruning modules
- gnn_*.py: Split GNN caching modules
"""

from .core import dual_hash, emit_receipt, merkle, StopRule

# Centralized constants (single source of truth)
from .constants import (
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
    ENTROPY_ASYMPTOTE_E,
)

# Receipt helpers
from .receipts import emit_with_hash, emit_anomaly, emit_spec_ingest
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
from .optimize import (
    OptimizationConfig,
    OptimizationState,
    selection_pressure,
    update_fitness,
    sample_thompson,
    measure_improvement,
)
from .helper import (
    HelperConfig,
    HelperBlueprint,
    harvest,
    hypothesize,
    gate,
    actuate,
    measure_effectiveness,
    retire,
)
from .support import (
    SupportLevel,
    SupportCoverage,
    classify_receipt,
    measure_coverage,
    check_completeness,
    detect_gaps,
    l4_feedback,
)
from .sim import (
    SimConfig,
    SimState,
    Scenario,
    initialize_sim,
    simulate_cycle,
    inject_gap,
    run_scenario,
    validate_constraints,
)

__all__ = [
    # Core
    "dual_hash", "emit_receipt", "merkle", "StopRule",
    # Centralized constants
    "SHANNON_FLOOR_ALPHA", "ALPHA_CEILING_TARGET", "ENTROPY_ASYMPTOTE_E",
    # Receipt helpers
    "emit_with_hash", "emit_anomaly", "emit_spec_ingest",
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
    # Optimization (Thompson sampling)
    "OptimizationConfig", "OptimizationState",
    "selection_pressure", "update_fitness", "sample_thompson", "measure_improvement",
    # Helper (HARVEST → HYPOTHESIZE → GATE → ACTUATE)
    "HelperConfig", "HelperBlueprint",
    "harvest", "hypothesize", "gate", "actuate", "measure_effectiveness", "retire",
    # Support (L0-L4 levels)
    "SupportLevel", "SupportCoverage",
    "classify_receipt", "measure_coverage", "check_completeness", "detect_gaps", "l4_feedback",
    # Simulation
    "SimConfig", "SimState", "Scenario",
    "initialize_sim", "simulate_cycle", "inject_gap", "run_scenario", "validate_constraints",
]
