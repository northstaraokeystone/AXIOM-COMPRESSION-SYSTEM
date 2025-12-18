"""AXIOM-CORE CLI command modules.

This package provides modular command handlers for the CLI.
Each module contains related commands grouped by domain.
"""

# Core commands (baseline, bootstrap, curve, full)
from cli.core import (
    cmd_baseline,
    cmd_bootstrap,
    cmd_curve,
    cmd_full,
)

# Partition commands
from cli.partition import (
    cmd_partition,
    cmd_stress_quorum,
)

# Blackout commands (including reroute)
from cli.blackout import (
    cmd_reroute,
    cmd_algo_info,
    cmd_blackout,
    cmd_blackout_sweep,
    cmd_simulate_timeline,
    cmd_extended_sweep,
    cmd_retention_curve,
    cmd_gnn_stub,
    cmd_gnn_nonlinear,
    cmd_cache_sweep,
    cmd_extreme_sweep,
    cmd_overflow_test,
    cmd_innovation_stubs,
)

# Pruning commands
from cli.pruning import (
    cmd_entropy_prune,
    cmd_pruning_sweep,
    cmd_extended_250d,
    cmd_verify_chain,
    cmd_pruning_info,
)

# Ablation commands
from cli.ablation import (
    cmd_ablate,
    cmd_ablation_sweep,
    cmd_ceiling_track,
    cmd_formula_check,
    cmd_isolate_layers,
)

# Depth commands
from cli.depth import (
    cmd_adaptive_depth_run,
    cmd_depth_scaling_test,
    cmd_compute_depth_single,
    cmd_depth_scaling_info,
    cmd_efficient_sweep_info,
)

# RL commands
from cli.rl import (
    cmd_rl_info,
    cmd_adaptive_info,
    cmd_rl_status,
    cmd_validate_no_static,
    cmd_rl_tune,
    cmd_dynamic_mode,
    cmd_tune_sweep,
    cmd_rl_500_sweep,
    cmd_rl_500_sweep_info,
)

# Quantum commands
from cli.quantum import (
    cmd_quantum_estimate,
    cmd_quantum_sim,
    cmd_quantum_rl_hybrid_info,
)

# Pipeline commands
from cli.pipeline import (
    cmd_lr_pilot,
    cmd_post_tune_execute,
    cmd_full_pipeline,
    cmd_pilot_info,
    cmd_pipeline_info,
)

# Scale commands
from cli.scale import (
    cmd_multi_scale_sweep,
    cmd_scalability_gate_test,
    cmd_scale_info,
    cmd_fractal_info,
)

# Fractal ceiling breach commands
from cli.fractal import (
    cmd_fractal_push,
    cmd_alpha_boost,
    cmd_fractal_info_hybrid,
)

# Full sweep commands
from cli.sweep import (
    cmd_full_500_sweep,
)

# Info commands
from cli.info import (
    cmd_hybrid_boost_info,
)

# Benchmark commands
from cli.benchmark import (
    cmd_hybrid_10e12,
    cmd_release_gate,
    cmd_fractal_recursion,
    cmd_fractal_recursion_sweep,
    cmd_benchmark_info,
)

# Path commands (exploration paths)
from cli.paths import (
    cmd_path_status,
    cmd_path_list,
    cmd_path_run,
    cmd_path_commands,
    cmd_mars_status,
    cmd_multiplanet_status,
    cmd_agi_status,
    cmd_d4_push,
    cmd_d4_info,
    cmd_registry_info,
)

# ISRU hybrid commands (D5 + MOXIE)
from cli.isru import (
    cmd_moxie_info,
    cmd_isru_simulate,
    cmd_isru_closure,
    cmd_d5_isru_hybrid,
    cmd_d5_push_isru,
    cmd_d5_info_isru,
    cmd_isru_info,
)

# D6 + Titan hybrid commands
from cli.titan import (
    cmd_d6_info,
    cmd_d6_push,
    cmd_d6_titan_hybrid,
    cmd_titan_info,
    cmd_titan_config,
    cmd_titan_simulate,
    cmd_titan_autonomy,
    cmd_perovskite_info,
    cmd_perovskite_project,
)

# Adversarial audit commands
from cli.audit import (
    cmd_audit_info,
    cmd_audit_config,
    cmd_audit_run,
    cmd_audit_stress,
    cmd_audit_classify,
)

# D7 + Europa hybrid commands
from cli.europa import (
    cmd_d7_info,
    cmd_d7_push,
    cmd_d7_europa_hybrid,
    cmd_europa_info,
    cmd_europa_config,
    cmd_europa_simulate,
    cmd_europa_autonomy,
)

# NREL perovskite validation commands
from cli.nrel import (
    cmd_nrel_info,
    cmd_nrel_config,
    cmd_nrel_validate,
    cmd_nrel_project,
    cmd_nrel_compare,
)

# D8 + Multi-planet sync commands
from cli.sync import (
    cmd_sync_info,
    cmd_sync_run,
    cmd_sync_efficiency,
    cmd_d8_multi_sync,
)

# D8 + Fractal encryption commands
from cli.encrypt import (
    cmd_encrypt_info,
    cmd_encrypt_keygen,
    cmd_encrypt_audit,
    cmd_encrypt_side_channel,
    cmd_encrypt_inversion,
)

# D8 + Atacama validation commands
from cli.atacama import (
    cmd_atacama_info,
    cmd_atacama_validate,
)

# D8 commands
from cli.d8 import (
    cmd_d8_info,
    cmd_d8_push,
)

__all__ = [
    # Core
    "cmd_baseline",
    "cmd_bootstrap",
    "cmd_curve",
    "cmd_full",
    # Partition
    "cmd_partition",
    "cmd_stress_quorum",
    # Blackout
    "cmd_reroute",
    "cmd_algo_info",
    "cmd_blackout",
    "cmd_blackout_sweep",
    "cmd_simulate_timeline",
    "cmd_extended_sweep",
    "cmd_retention_curve",
    "cmd_gnn_stub",
    "cmd_gnn_nonlinear",
    "cmd_cache_sweep",
    "cmd_extreme_sweep",
    "cmd_overflow_test",
    "cmd_innovation_stubs",
    # Pruning
    "cmd_entropy_prune",
    "cmd_pruning_sweep",
    "cmd_extended_250d",
    "cmd_verify_chain",
    "cmd_pruning_info",
    # Ablation
    "cmd_ablate",
    "cmd_ablation_sweep",
    "cmd_ceiling_track",
    "cmd_formula_check",
    "cmd_isolate_layers",
    # Depth
    "cmd_adaptive_depth_run",
    "cmd_depth_scaling_test",
    "cmd_compute_depth_single",
    "cmd_depth_scaling_info",
    "cmd_efficient_sweep_info",
    # RL
    "cmd_rl_info",
    "cmd_adaptive_info",
    "cmd_rl_status",
    "cmd_validate_no_static",
    "cmd_rl_tune",
    "cmd_dynamic_mode",
    "cmd_tune_sweep",
    "cmd_rl_500_sweep",
    "cmd_rl_500_sweep_info",
    # Quantum
    "cmd_quantum_estimate",
    "cmd_quantum_sim",
    "cmd_quantum_rl_hybrid_info",
    # Pipeline
    "cmd_lr_pilot",
    "cmd_post_tune_execute",
    "cmd_full_pipeline",
    "cmd_pilot_info",
    "cmd_pipeline_info",
    # Scale
    "cmd_multi_scale_sweep",
    "cmd_scalability_gate_test",
    "cmd_scale_info",
    "cmd_fractal_info",
    # Fractal ceiling breach
    "cmd_fractal_push",
    "cmd_alpha_boost",
    "cmd_fractal_info_hybrid",
    # Full sweep
    "cmd_full_500_sweep",
    # Info
    "cmd_hybrid_boost_info",
    # Benchmark
    "cmd_hybrid_10e12",
    "cmd_release_gate",
    "cmd_fractal_recursion",
    "cmd_fractal_recursion_sweep",
    "cmd_benchmark_info",
    # Paths
    "cmd_path_status",
    "cmd_path_list",
    "cmd_path_run",
    "cmd_path_commands",
    "cmd_mars_status",
    "cmd_multiplanet_status",
    "cmd_agi_status",
    "cmd_d4_push",
    "cmd_d4_info",
    "cmd_registry_info",
    # ISRU hybrid (D5 + MOXIE)
    "cmd_moxie_info",
    "cmd_isru_simulate",
    "cmd_isru_closure",
    "cmd_d5_isru_hybrid",
    "cmd_d5_push_isru",
    "cmd_d5_info_isru",
    "cmd_isru_info",
    # D6 + Titan hybrid
    "cmd_d6_info",
    "cmd_d6_push",
    "cmd_d6_titan_hybrid",
    "cmd_titan_info",
    "cmd_titan_config",
    "cmd_titan_simulate",
    "cmd_titan_autonomy",
    "cmd_perovskite_info",
    "cmd_perovskite_project",
    # Adversarial audit
    "cmd_audit_info",
    "cmd_audit_config",
    "cmd_audit_run",
    "cmd_audit_stress",
    "cmd_audit_classify",
    # D7 + Europa hybrid
    "cmd_d7_info",
    "cmd_d7_push",
    "cmd_d7_europa_hybrid",
    "cmd_europa_info",
    "cmd_europa_config",
    "cmd_europa_simulate",
    "cmd_europa_autonomy",
    # NREL perovskite validation
    "cmd_nrel_info",
    "cmd_nrel_config",
    "cmd_nrel_validate",
    "cmd_nrel_project",
    "cmd_nrel_compare",
    # D8 + Multi-planet sync
    "cmd_sync_info",
    "cmd_sync_run",
    "cmd_sync_efficiency",
    "cmd_d8_multi_sync",
    # D8 + Fractal encryption
    "cmd_encrypt_info",
    "cmd_encrypt_keygen",
    "cmd_encrypt_audit",
    "cmd_encrypt_side_channel",
    "cmd_encrypt_inversion",
    # D8 + Atacama validation
    "cmd_atacama_info",
    "cmd_atacama_validate",
    # D8
    "cmd_d8_info",
    "cmd_d8_push",
]
