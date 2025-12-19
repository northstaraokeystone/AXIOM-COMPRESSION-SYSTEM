"""reasoning/ - Sovereignty Timeline Projections Package.

Re-exports all reasoning functions and constants.
"""

# Constants
from .constants import (
    MIN_EFF_ALPHA_BOUND,
    CYCLES_THRESHOLD_EARLY,
    CYCLES_THRESHOLD_CITY,
    PILOT_RETENTION_TARGET,
    EXPECTED_FINAL_RETENTION,
    EXPECTED_EFF_ALPHA,
    SCALABILITY_GATE_THRESHOLD,
    SCALABILITY_INSTABILITY_TOLERANCE,
    SCALABILITY_DEGRADATION_TOLERANCE,
)

# Partition/Resilience functions
from .partition import (
    partition_sweep,
    project_with_resilience,
    sovereignty_projection_with_partition,
    validate_resilience_slo,
)

# Blackout/Reroute functions
from .blackout import (
    blackout_sweep,
    project_with_reroute,
    sovereignty_timeline,
    extended_blackout_sweep,
    extreme_blackout_sweep_200d,
    project_with_asymptote,
    project_with_degradation,
)

# Pruning sovereignty
from .pruning_sovereignty import (
    extended_250d_sovereignty,
    validate_pruning_slos,
)

# Ablation testing
from .ablation import (
    ablation_sweep,
    compute_alpha_with_isolation,
    get_layer_contributions,
)

# Dynamic/RL functions
from .dynamic import (
    sovereignty_timeline_dynamic,
    continued_ablation_loop,
    validate_no_static_configs,
    get_rl_integration_status,
)

# Pipeline functions
from .pipeline import (
    execute_full_pipeline,
    get_pipeline_info,
)

# Scalability gate
from .scalability import (
    enforce_scalability_gate,
    get_31_push_readiness,
)


__all__ = [
    # Constants
    "MIN_EFF_ALPHA_BOUND",
    "CYCLES_THRESHOLD_EARLY",
    "CYCLES_THRESHOLD_CITY",
    "PILOT_RETENTION_TARGET",
    "EXPECTED_FINAL_RETENTION",
    "EXPECTED_EFF_ALPHA",
    "SCALABILITY_GATE_THRESHOLD",
    "SCALABILITY_INSTABILITY_TOLERANCE",
    "SCALABILITY_DEGRADATION_TOLERANCE",
    # Partition/Resilience
    "partition_sweep",
    "project_with_resilience",
    "sovereignty_projection_with_partition",
    "validate_resilience_slo",
    # Blackout/Reroute
    "blackout_sweep",
    "project_with_reroute",
    "sovereignty_timeline",
    "extended_blackout_sweep",
    "extreme_blackout_sweep_200d",
    "project_with_asymptote",
    "project_with_degradation",
    # Pruning sovereignty
    "extended_250d_sovereignty",
    "validate_pruning_slos",
    # Ablation testing
    "ablation_sweep",
    "compute_alpha_with_isolation",
    "get_layer_contributions",
    # Dynamic/RL
    "sovereignty_timeline_dynamic",
    "continued_ablation_loop",
    "validate_no_static_configs",
    "get_rl_integration_status",
    # Pipeline
    "execute_full_pipeline",
    "get_pipeline_info",
    # Scalability gate
    "enforce_scalability_gate",
    "get_31_push_readiness",
]
