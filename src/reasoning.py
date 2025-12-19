"""reasoning.py - Re-export wrapper for backward compatibility.

All implementation moved to src/reasoning/ package.
This file exists ONLY to maintain import compatibility.

CLAUDEME COMPLIANT: ≤50 lines
"""

from .reasoning.constants import (
    MIN_EFF_ALPHA_BOUND, CYCLES_THRESHOLD_EARLY, CYCLES_THRESHOLD_CITY,
    PILOT_RETENTION_TARGET, EXPECTED_FINAL_RETENTION, EXPECTED_EFF_ALPHA,
    SCALABILITY_GATE_THRESHOLD, SCALABILITY_INSTABILITY_TOLERANCE,
    SCALABILITY_DEGRADATION_TOLERANCE,
)
from .reasoning.partition import (
    partition_sweep, project_with_resilience,
    sovereignty_projection_with_partition, validate_resilience_slo,
)
from .reasoning.blackout import (
    blackout_sweep, project_with_reroute, sovereignty_timeline,
    extended_blackout_sweep, extreme_blackout_sweep_200d,
    project_with_asymptote, project_with_degradation,
)
from .reasoning.pruning_sovereignty import extended_250d_sovereignty, validate_pruning_slos
from .reasoning.ablation import ablation_sweep, compute_alpha_with_isolation, get_layer_contributions
from .reasoning.dynamic import (
    sovereignty_timeline_dynamic, continued_ablation_loop,
    validate_no_static_configs, get_rl_integration_status,
)
from .reasoning.pipeline import execute_full_pipeline, get_pipeline_info
from .reasoning.scalability import enforce_scalability_gate, get_31_push_readiness

__all__ = [
    "MIN_EFF_ALPHA_BOUND", "CYCLES_THRESHOLD_EARLY", "CYCLES_THRESHOLD_CITY",
    "PILOT_RETENTION_TARGET", "EXPECTED_FINAL_RETENTION", "EXPECTED_EFF_ALPHA",
    "SCALABILITY_GATE_THRESHOLD", "SCALABILITY_INSTABILITY_TOLERANCE",
    "SCALABILITY_DEGRADATION_TOLERANCE", "partition_sweep", "project_with_resilience",
    "sovereignty_projection_with_partition", "validate_resilience_slo", "blackout_sweep",
    "project_with_reroute", "sovereignty_timeline", "extended_blackout_sweep",
    "extreme_blackout_sweep_200d", "project_with_asymptote", "project_with_degradation",
    "extended_250d_sovereignty", "validate_pruning_slos", "ablation_sweep",
    "compute_alpha_with_isolation", "get_layer_contributions", "sovereignty_timeline_dynamic",
    "continued_ablation_loop", "validate_no_static_configs", "get_rl_integration_status",
    "execute_full_pipeline", "get_pipeline_info", "enforce_scalability_gate", "get_31_push_readiness",
]
