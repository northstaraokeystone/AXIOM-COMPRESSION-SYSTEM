"""AXIOM-CORE v4: D20 Production Evolution

One equation. One curve. One finding.

sovereignty = internal_rate > external_rate
threshold = 47 +/- 8 crew

D20 PRODUCTION EVOLUTION:
Names are not labels. Names are sales.

STAKEHOLDER-INTUITIVE MODULES:
- compress.py: Telemetry compression (10x+, 0.999 recall)
- witness.py: KAN/MDL law discovery
- sovereignty.py: Autonomy threshold calculator
- detect.py: Entropy-based anomaly detection
- ledger.py: Append-only receipt storage
- anchor.py: Merkle proofs
- loop.py: 60-second SENSE→ACTUATE cycle

DOMAIN GENERATORS:
- domains/galaxy.py: Galaxy rotation curves
- domains/colony.py: Mars colony simulation
- domains/telemetry.py: Fleet telemetry (Tesla/Starlink/SpaceX)

STAKEHOLDER CONFIGS:
- configs/xai.yaml: Elon/xAI
- configs/doge.yaml: DOGE
- configs/dot.yaml: DOT
- configs/defense.yaml: Defense
- configs/nro.yaml: NRO

Source: D20 Production Evolution (Dec 2025)
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
    "dual_hash",
    "emit_receipt",
    "merkle",
    "StopRule",
    # Centralized constants
    "SHANNON_FLOOR_ALPHA",
    "ALPHA_CEILING_TARGET",
    "ENTROPY_ASYMPTOTE_E",
    # Receipt helpers
    "emit_with_hash",
    "emit_anomaly",
    "emit_spec_ingest",
    # Entropy (Shannon only)
    "HUMAN_DECISION_RATE_BPS",
    "STARLINK_MARS_BANDWIDTH_MIN_MBPS",
    "STARLINK_MARS_BANDWIDTH_MAX_MBPS",
    "MARS_LIGHT_DELAY_MIN_S",
    "MARS_LIGHT_DELAY_MAX_S",
    "internal_rate",
    "external_rate",
    "sovereignty_advantage",
    "is_sovereign",
    # Sovereignty
    "SovereigntyConfig",
    "SovereigntyResult",
    "compute_sovereignty",
    "find_threshold",
    "sensitivity_analysis",
    # Data ingest
    "load_bandwidth_data",
    "load_delay_data",
    "sample_bandwidth",
    "sample_delay",
    # Validation
    "test_null_hypothesis",
    "test_baseline",
    "bootstrap_threshold",
    "compute_p_value",
    "generate_falsifiable_prediction",
    # Plotting
    "generate_curve_data",
    "find_knee",
    "plot_sovereignty_curve",
    "format_finding",
    # Optimization (Thompson sampling)
    "OptimizationConfig",
    "OptimizationState",
    "selection_pressure",
    "update_fitness",
    "sample_thompson",
    "measure_improvement",
    # Helper (HARVEST → HYPOTHESIZE → GATE → ACTUATE)
    "HelperConfig",
    "HelperBlueprint",
    "harvest",
    "hypothesize",
    "gate",
    "actuate",
    "measure_effectiveness",
    "retire",
    # Support (L0-L4 levels)
    "SupportLevel",
    "SupportCoverage",
    "classify_receipt",
    "measure_coverage",
    "check_completeness",
    "detect_gaps",
    "l4_feedback",
    # Simulation
    "SimConfig",
    "SimState",
    "Scenario",
    "initialize_sim",
    "simulate_cycle",
    "inject_gap",
    "run_scenario",
    "validate_constraints",
    # D20 Production Evolution - Stakeholder-Intuitive Modules
    "compress",
    "detect",
    "anchor",
    "loop",
    # D20 Domain Generators
    "domains",
]

# D20 Production Evolution - New Modules
from . import compress
from . import detect
from . import anchor
from . import loop

# D20 Domain Generators
from . import domains
