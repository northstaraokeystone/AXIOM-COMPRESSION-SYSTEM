#!/usr/bin/env python3
"""AXIOM-CORE CLI - The Sovereignty Calculator

One equation. One curve. One finding.

Usage:
    python cli.py baseline      # Run baseline test
    python cli.py bootstrap     # Run bootstrap analysis
    python cli.py curve         # Generate sovereignty curve
    python cli.py full          # Run full integration test
    python cli.py --simulate_timeline --c_base 50 --p_factor 1.8 --tau 0     # Earth timeline
    python cli.py --simulate_timeline --c_base 50 --p_factor 1.8 --tau 1200  # Mars timeline

    # Partition resilience testing (Dec 2025)
    python cli.py --partition 0.4 --nodes 5 --simulate    # Single partition test
    python cli.py --stress_quorum                          # Full 1000-iteration stress test

    # Adaptive rerouting and blackout testing (Dec 2025)
    python cli.py --reroute --simulate                     # Single reroute test
    python cli.py --blackout 43 --reroute --simulate       # Blackout with reroute
    python cli.py --blackout_sweep --reroute               # Full blackout sweep (43-60d, 1000 iterations)
    python cli.py --algo_info                              # Output reroute algorithm spec

    # Extended blackout sweep and retention curve (Dec 2025)
    python cli.py --extended_sweep 43 90 --simulate        # Extended sweep (43-90d)
    python cli.py --retention_curve                        # Output retention curve as JSON
    python cli.py --blackout_sweep 60 --simulate           # Single-point extended blackout test
    python cli.py --gnn_stub                               # Echo GNN sensitivity stub config

    # GNN nonlinear caching (Dec 2025)
    python cli.py --gnn_nonlinear --blackout 150 --simulate  # GNN nonlinear at 150d
    python cli.py --cache_depth 1000000000 --blackout 200 --gnn_nonlinear  # Custom cache depth
    python cli.py --cache_sweep --simulate                   # Cache depth sensitivity sweep
    python cli.py --extreme_sweep 200 --simulate             # Extreme sweep to 200d
    python cli.py --overflow_test --simulate                 # Test cache overflow detection
    python cli.py --innovation_stubs                         # Echo innovation stub status

    # Entropy pruning (Dec 2025)
    python cli.py --entropy_prune --blackout 150 --simulate     # Single pruning test
    python cli.py --trim_factor 0.4 --entropy_prune --simulate  # Custom trim factor
    python cli.py --hybrid_prune --blackout 200 --simulate      # Hybrid dedup + predictive
    python cli.py --pruning_sweep --simulate                    # Pruning sensitivity sweep
    python cli.py --extended_250d --simulate                    # 250d with pruning
    python cli.py --verify_chain --entropy_prune --simulate     # Verify chain integrity
    python cli.py --pruning_info                                # Echo pruning configuration
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import constants needed for argument defaults
from src.partition import NODE_BASELINE
from src.timeline import C_BASE_DEFAULT, P_FACTOR_DEFAULT
from src.rl_tune import RL_LR_MIN, RL_LR_MAX

# Import all command handlers from cli modules
from cli import (
    # Core
    cmd_baseline,
    cmd_bootstrap,
    cmd_curve,
    cmd_full,
    # Partition
    cmd_partition,
    cmd_stress_quorum,
    # Blackout
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
    # Pruning
    cmd_entropy_prune,
    cmd_pruning_sweep,
    cmd_extended_250d,
    cmd_verify_chain,
    cmd_pruning_info,
    # Ablation
    cmd_ablate,
    cmd_ablation_sweep,
    cmd_ceiling_track,
    cmd_formula_check,
    cmd_isolate_layers,
    # Depth
    cmd_adaptive_depth_run,
    cmd_depth_scaling_test,
    cmd_compute_depth_single,
    cmd_depth_scaling_info,
    cmd_efficient_sweep_info,
    # RL
    cmd_rl_info,
    cmd_adaptive_info,
    cmd_rl_status,
    cmd_validate_no_static,
    cmd_rl_tune,
    cmd_dynamic_mode,
    cmd_tune_sweep,
    cmd_rl_500_sweep,
    cmd_rl_500_sweep_info,
    # Quantum
    cmd_quantum_estimate,
    cmd_quantum_sim,
    cmd_quantum_rl_hybrid_info,
    # Pipeline
    cmd_lr_pilot,
    cmd_post_tune_execute,
    cmd_full_pipeline,
    cmd_pilot_info,
    cmd_pipeline_info,
    # Scale
    cmd_multi_scale_sweep,
    cmd_scalability_gate_test,
    cmd_scale_info,
    cmd_fractal_info,
    # Fractal ceiling breach
    cmd_fractal_push,
    cmd_alpha_boost,
    cmd_fractal_info_hybrid,
    # Full sweep
    cmd_full_500_sweep,
    # Info
    cmd_hybrid_boost_info,
    # Benchmark (10^12 scale)
    cmd_hybrid_10e12,
    cmd_release_gate,
    cmd_fractal_recursion,
    cmd_fractal_recursion_sweep,
    cmd_benchmark_info,
    # Paths
    cmd_path_status,
    cmd_path_list,
    cmd_path_run,
    cmd_mars_status,
    cmd_multiplanet_status,
    cmd_agi_status,
    cmd_d4_push,
    cmd_d4_info,
    cmd_registry_info,
    # ISRU hybrid (D5 + MOXIE)
    cmd_moxie_info,
    cmd_isru_simulate,
    cmd_isru_closure,
    cmd_d5_isru_hybrid,
    cmd_d5_push_isru,
    cmd_d5_info_isru,
    cmd_isru_info,
    # D6 + Titan hybrid
    cmd_d6_info,
    cmd_d6_push,
    cmd_d6_titan_hybrid,
    cmd_titan_info,
    cmd_titan_config,
    cmd_titan_simulate,
    cmd_perovskite_info,
    cmd_perovskite_project,
    # Adversarial audit
    cmd_audit_info,
    cmd_audit_config,
    cmd_audit_run,
    cmd_audit_stress,
    cmd_d7_info,
    cmd_d7_push,
    cmd_d7_europa_hybrid,
    cmd_europa_info,
    cmd_europa_config,
    cmd_europa_simulate,
    cmd_nrel_info,
    cmd_nrel_config,
    cmd_nrel_validate,
    cmd_nrel_project,
    cmd_nrel_compare,
    # D8 + Multi-planet sync
    cmd_sync_info,
    cmd_sync_run,
    cmd_sync_efficiency,
    cmd_d8_multi_sync,
    # D8 + Fractal encryption
    cmd_encrypt_info,
    cmd_encrypt_keygen,
    cmd_encrypt_audit,
    cmd_encrypt_side_channel,
    cmd_encrypt_inversion,
    # D8 + Atacama validation
    cmd_atacama_info,
    cmd_atacama_validate,
    # D8
    cmd_d8_info,
    cmd_d8_push,
    # D9 + Ganymede
    cmd_d9_info,
    cmd_d9_push,
    cmd_d9_ganymede_hybrid,
    cmd_ganymede_info,
    cmd_ganymede_config,
    cmd_ganymede_navigate,
    cmd_ganymede_autonomy,
    # Atacama drone arrays
    cmd_drone_info,
    cmd_drone_config,
    cmd_drone_coverage,
    cmd_drone_sample,
    cmd_drone_validate,
    # Randomized execution paths
    cmd_randomized_info,
    cmd_randomized_config,
    cmd_randomized_generate,
    cmd_randomized_audit,
    cmd_randomized_timing,
    cmd_randomized_power,
    cmd_randomized_cache,
    # D10 + Jovian hub
    cmd_d10_info,
    cmd_d10_push,
    cmd_d10_jovian_hub,
    cmd_jovian_info,
    cmd_jovian_sync,
    cmd_jovian_autonomy,
    cmd_jovian_coordinate,
    # Callisto
    cmd_callisto_info,
    cmd_callisto_config,
    cmd_callisto_ice,
    cmd_callisto_extract,
    cmd_callisto_radiation,
    cmd_callisto_autonomy,
    cmd_callisto_hub_suitability,
    # Quantum-resistant
    cmd_quantum_resist_info,
    cmd_quantum_resist_config,
    cmd_quantum_keygen,
    cmd_quantum_resist_audit,
    cmd_quantum_spectre,
    cmd_quantum_resist_cache,
    cmd_quantum_spectre_v1,
    cmd_quantum_spectre_v2,
    cmd_quantum_spectre_v4,
    # Dust dynamics
    cmd_dust_dynamics_info,
    cmd_dust_dynamics_config,
    cmd_dust_dynamics,
    cmd_dust_settling,
    cmd_dust_particle,
    cmd_dust_solar_impact,
    cmd_dust_mars_projection,
    # D11 + Venus + CFD + enclave
    cmd_d11_info,
    cmd_d11_push,
    cmd_d11_venus_hybrid,
    cmd_venus_info,
    cmd_venus_cloud,
    cmd_venus_acid,
    cmd_venus_ops,
    cmd_venus_autonomy,
    cmd_cfd_info,
    cmd_cfd_reynolds,
    cmd_cfd_settling,
    cmd_cfd_storm,
    cmd_cfd_validate,
    cmd_enclave_info,
    cmd_enclave_init,
    cmd_enclave_audit,
    cmd_enclave_btb,
    cmd_enclave_pht,
    cmd_enclave_rsb,
    cmd_enclave_overhead,
)


def main():
    parser = argparse.ArgumentParser(
        description="AXIOM-CORE CLI - The Sovereignty Calculator"
    )
    parser.add_argument(
        "command", nargs="?", help="Command: baseline, bootstrap, curve, full"
    )

    # Timeline args
    parser.add_argument(
        "--c_base",
        type=float,
        default=C_BASE_DEFAULT,
        help="Initial person-eq capacity (default: 50)",
    )
    parser.add_argument(
        "--p_factor",
        type=float,
        default=P_FACTOR_DEFAULT,
        help="Propulsion growth factor (default: 1.8)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0,
        help="Latency in seconds (0=Earth, 1200=Mars max)",
    )
    parser.add_argument(
        "--simulate_timeline",
        action="store_true",
        help="Run sovereignty timeline simulation",
    )

    # Partition flags
    parser.add_argument(
        "--partition",
        type=float,
        default=None,
        help="Run single partition simulation at specified loss percentage (0-1)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=NODE_BASELINE,
        help="Specify node count for simulation (default: 5)",
    )
    parser.add_argument(
        "--stress_quorum",
        action="store_true",
        help="Run full stress sweep (1000 iterations, 0-40%% loss)",
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Output simulation receipt to stdout"
    )

    # Reroute/blackout flags
    parser.add_argument(
        "--reroute", action="store_true", help="Enable adaptive rerouting"
    )
    parser.add_argument(
        "--reroute_enabled", action="store_true", help="Alias for --reroute"
    )
    parser.add_argument(
        "--blackout", type=int, default=None, help="Blackout duration in days"
    )
    parser.add_argument(
        "--blackout_sweep", action="store_true", help="Run full blackout sweep"
    )
    parser.add_argument(
        "--algo_info", action="store_true", help="Output reroute algorithm spec"
    )

    # Extended sweep/retention flags
    parser.add_argument(
        "--extended_sweep",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Extended blackout sweep range",
    )
    parser.add_argument(
        "--retention_curve", action="store_true", help="Output retention curve JSON"
    )
    parser.add_argument(
        "--gnn_stub", action="store_true", help="Echo GNN sensitivity stub"
    )

    # GNN/cache flags
    parser.add_argument(
        "--gnn_nonlinear", action="store_true", help="Use GNN nonlinear model"
    )
    parser.add_argument(
        "--cache_depth", type=int, default=int(1e8), help="Cache depth (default: 1e8)"
    )
    parser.add_argument(
        "--cache_sweep", action="store_true", help="Run cache depth sweep"
    )
    parser.add_argument(
        "--extreme_sweep",
        type=int,
        default=None,
        metavar="DAYS",
        help="Run extreme blackout sweep to days",
    )
    parser.add_argument(
        "--overflow_test", action="store_true", help="Test cache overflow"
    )
    parser.add_argument(
        "--innovation_stubs", action="store_true", help="Echo innovation stubs"
    )

    # Pruning flags
    parser.add_argument(
        "--entropy_prune", action="store_true", help="Enable entropy pruning"
    )
    parser.add_argument(
        "--trim_factor", type=float, default=0.3, help="Trim factor (default: 0.3)"
    )
    parser.add_argument(
        "--hybrid_prune", action="store_true", help="Enable hybrid pruning"
    )
    parser.add_argument(
        "--pruning_sweep", action="store_true", help="Run pruning sweep"
    )
    parser.add_argument(
        "--extended_250d", action="store_true", help="Run 250d simulation"
    )
    parser.add_argument(
        "--verify_chain", action="store_true", help="Verify chain integrity"
    )
    parser.add_argument(
        "--pruning_info", action="store_true", help="Output pruning config"
    )

    # Ablation flags
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        metavar="MODE",
        help="Ablation mode (full/no_cache/no_prune/baseline)",
    )
    parser.add_argument(
        "--ablation_sweep", action="store_true", help="Run ablation sweep"
    )
    parser.add_argument(
        "--ceiling_track",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Ceiling gap analysis for alpha",
    )
    parser.add_argument(
        "--formula_check", action="store_true", help="Validate alpha formula"
    )
    parser.add_argument(
        "--isolate_layers", action="store_true", help="Isolate layer contributions"
    )

    # RL flags
    parser.add_argument("--rl_tune", action="store_true", help="Enable RL auto-tuning")
    parser.add_argument(
        "--rl_episodes", type=int, default=100, help="RL episodes (default: 100)"
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Enable adaptive depth/config"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable all dynamic features"
    )
    parser.add_argument("--tune_sweep", action="store_true", help="Run retention sweep")
    parser.add_argument(
        "--show_rl_history", action="store_true", help="Output RL history"
    )
    parser.add_argument(
        "--validate_no_static", action="store_true", help="Verify no static configs"
    )
    parser.add_argument(
        "--rl_info", action="store_true", help="Output RL tuning config"
    )
    parser.add_argument(
        "--adaptive_info", action="store_true", help="Output adaptive config"
    )
    parser.add_argument(
        "--rl_status", action="store_true", help="Output RL integration status"
    )

    # Adaptive depth flags
    parser.add_argument(
        "--adaptive_depth_run", action="store_true", help="Run with adaptive depth"
    )
    parser.add_argument(
        "--rl_sweep", type=int, default=500, help="Informed RL runs (default: 500)"
    )
    parser.add_argument(
        "--depth_scaling_test", action="store_true", help="Run depth scaling test"
    )
    parser.add_argument(
        "--compute_depth", action="store_true", help="Show computed depth"
    )
    parser.add_argument(
        "--tree_size", type=int, default=int(1e6), help="Tree size (default: 1e6)"
    )
    parser.add_argument(
        "--depth_scaling_info", action="store_true", help="Output depth scaling config"
    )
    parser.add_argument(
        "--efficient_sweep_info", action="store_true", help="Output sweep config"
    )

    # 500-run sweep flags
    parser.add_argument(
        "--rl_500_sweep", action="store_true", help="Run 500-run RL sweep"
    )
    parser.add_argument(
        "--lr_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override LR bounds",
    )
    parser.add_argument(
        "--retention_target", type=float, default=1.05, help="Retention target"
    )
    parser.add_argument(
        "--quantum_estimate", action="store_true", help="Show quantum estimate"
    )
    parser.add_argument(
        "--rl_500_sweep_info", action="store_true", help="Output 500-run config"
    )

    # Pipeline flags
    parser.add_argument(
        "--lr_pilot",
        type=int,
        default=None,
        const=50,
        nargs="?",
        metavar="RUNS",
        help="Run LR pilot (default: 50 runs)",
    )
    parser.add_argument(
        "--quantum_sim",
        type=int,
        default=None,
        const=10,
        nargs="?",
        metavar="RUNS",
        help="Run quantum sim (default: 10 runs)",
    )
    parser.add_argument(
        "--post_tune_execute", action="store_true", help="Execute tuned sweep"
    )
    parser.add_argument(
        "--full_pipeline", action="store_true", help="Run complete pipeline"
    )
    parser.add_argument(
        "--show_bounds", action="store_true", help="Show LR bounds comparison"
    )
    parser.add_argument("--pilot_info", action="store_true", help="Output pilot config")
    parser.add_argument(
        "--quantum_rl_hybrid_info", action="store_true", help="Output quantum-RL config"
    )
    parser.add_argument(
        "--pipeline_info", action="store_true", help="Output pipeline config"
    )
    parser.add_argument(
        "--pilot_runs", type=int, default=50, help="Pilot runs (default: 50)"
    )
    parser.add_argument(
        "--quantum_runs", type=int, default=10, help="Quantum runs (default: 10)"
    )
    parser.add_argument(
        "--sweep_runs", type=int, default=500, help="Sweep runs (default: 500)"
    )

    # Scale flags
    parser.add_argument(
        "--multi_scale_sweep",
        type=str,
        default=None,
        metavar="TYPE",
        help='Run multi-scale sweep ("all" or "1e9")',
    )
    parser.add_argument(
        "--scalability_gate_test",
        action="store_true",
        help="Test scalability gate status",
    )
    parser.add_argument(
        "--scale_info", action="store_true", help="Show scale validation configuration"
    )
    parser.add_argument(
        "--fractal_info", action="store_true", help="Show fractal layer configuration"
    )

    # Fractal ceiling breach flags
    parser.add_argument(
        "--alpha_boost",
        type=str,
        default=None,
        metavar="MODE",
        help="Alpha boost mode (off, quantum, fractal, hybrid)",
    )
    parser.add_argument(
        "--fractal_push", action="store_true", help="Run fractal ceiling breach"
    )
    parser.add_argument(
        "--fractal_info_hybrid",
        action="store_true",
        help="Show fractal hybrid configuration",
    )
    parser.add_argument(
        "--base_alpha",
        type=float,
        default=2.99,
        help="Base alpha for fractal/hybrid (default: 2.99)",
    )

    # Full sweep flags
    parser.add_argument(
        "--full_500_sweep",
        action="store_true",
        help="Run full 500-sweep with quantum-fractal hybrid",
    )

    # Hybrid info flags
    parser.add_argument(
        "--hybrid_boost_info",
        action="store_true",
        help="Show hybrid boost configuration",
    )

    # Benchmark (10^12 scale) flags
    parser.add_argument(
        "--hybrid_10e12", action="store_true", help="Run 10^12 hybrid benchmark"
    )
    parser.add_argument(
        "--release_gate", action="store_true", help="Check release gate 3.1 status"
    )
    parser.add_argument(
        "--fractal_recursion",
        action="store_true",
        help="Run fractal recursion ceiling breach",
    )
    parser.add_argument(
        "--fractal_recursion_sweep",
        action="store_true",
        help="Sweep through all recursion depths",
    )
    parser.add_argument(
        "--benchmark_info", action="store_true", help="Show benchmark configuration"
    )
    parser.add_argument(
        "--recursion_depth",
        type=int,
        default=3,
        help="Fractal recursion depth (1-5, default: 3)",
    )

    # Path exploration flags
    parser.add_argument(
        "--d4_push", action="store_true", help="Run D4 recursion for alpha>=3.2"
    )
    parser.add_argument("--d4_info", action="store_true", help="Show D4 configuration")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Select exploration path (mars/multiplanet/agi)",
    )
    parser.add_argument(
        "--path_status", action="store_true", help="Show all path statuses"
    )
    parser.add_argument(
        "--path_list", action="store_true", help="List registered paths"
    )
    parser.add_argument(
        "--path_cmd", type=str, default=None, help="Run command on selected path"
    )
    parser.add_argument(
        "--mars_status",
        action="store_true",
        help="Shortcut: --path mars --path_cmd status",
    )
    parser.add_argument(
        "--multiplanet_status",
        action="store_true",
        help="Shortcut: --path multiplanet --path_cmd status",
    )
    parser.add_argument(
        "--agi_status",
        action="store_true",
        help="Shortcut: --path agi --path_cmd status",
    )
    parser.add_argument(
        "--registry_info", action="store_true", help="Show path registry info"
    )

    # D5 + ISRU hybrid flags
    parser.add_argument(
        "--d5_push", action="store_true", help="Run D5 recursion for alpha>=3.25"
    )
    parser.add_argument(
        "--d5_info", action="store_true", help="Show D5 + ISRU configuration"
    )
    parser.add_argument(
        "--d5_isru_hybrid", action="store_true", help="Run integrated D5+ISRU hybrid"
    )
    parser.add_argument(
        "--moxie_info", action="store_true", help="Show MOXIE calibration data"
    )
    parser.add_argument(
        "--isru_simulate", action="store_true", help="Run ISRU O2 production simulation"
    )
    parser.add_argument(
        "--isru_closure", action="store_true", help="Show ISRU closure metrics"
    )
    parser.add_argument(
        "--isru_info", action="store_true", help="Show ISRU module info"
    )
    parser.add_argument(
        "--crew", type=int, default=4, help="Crew size for ISRU simulation (default: 4)"
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="Simulation hours for ISRU (default: 24)"
    )
    parser.add_argument(
        "--moxie_units",
        type=int,
        default=10,
        help="Number of MOXIE units (default: 10)",
    )

    # D6 + Titan hybrid flags
    parser.add_argument(
        "--d6_push", action="store_true", help="Run D6 recursion for alpha>=3.33"
    )
    parser.add_argument(
        "--d6_info", action="store_true", help="Show D6 + Titan configuration"
    )
    parser.add_argument(
        "--d6_titan_hybrid", action="store_true", help="Run integrated D6+Titan hybrid"
    )
    parser.add_argument(
        "--titan_info", action="store_true", help="Show Titan methane hybrid info"
    )
    parser.add_argument(
        "--titan_config", action="store_true", help="Show Titan configuration from spec"
    )
    parser.add_argument(
        "--titan_simulate",
        action="store_true",
        help="Run Titan methane harvest simulation",
    )
    parser.add_argument(
        "--titan_duration",
        type=int,
        default=30,
        help="Titan simulation duration in days (default: 30)",
    )
    parser.add_argument(
        "--titan_extraction_rate",
        type=float,
        default=10.0,
        help="Titan extraction rate kg/hr (default: 10.0)",
    )
    parser.add_argument(
        "--perovskite_info",
        action="store_true",
        help="Show perovskite efficiency configuration",
    )
    parser.add_argument(
        "--perovskite_project",
        action="store_true",
        help="Project perovskite efficiency timeline",
    )
    parser.add_argument(
        "--perovskite_years",
        type=int,
        default=10,
        help="Perovskite projection years (default: 10)",
    )
    parser.add_argument(
        "--perovskite_growth",
        type=float,
        default=0.10,
        help="Perovskite annual growth rate (default: 0.10)",
    )

    # Adversarial audit flags
    parser.add_argument(
        "--audit_info", action="store_true", help="Show adversarial audit configuration"
    )
    parser.add_argument(
        "--audit_config", action="store_true", help="Show adversarial config from spec"
    )
    parser.add_argument(
        "--audit_run", action="store_true", help="Run adversarial audit"
    )
    parser.add_argument(
        "--audit_noise",
        type=float,
        default=0.05,
        help="Adversarial noise level (default: 0.05)",
    )
    parser.add_argument(
        "--audit_iterations",
        type=int,
        default=100,
        help="Adversarial test iterations (default: 100)",
    )
    parser.add_argument(
        "--audit_stress", action="store_true", help="Run adversarial stress test"
    )

    # D7 + Europa hybrid flags
    parser.add_argument(
        "--d7_push", action="store_true", help="Run D7 recursion for alpha>=3.40"
    )
    parser.add_argument(
        "--d7_info", action="store_true", help="Show D7 + Europa configuration"
    )
    parser.add_argument(
        "--d7_europa_hybrid",
        action="store_true",
        help="Run integrated D7+Europa hybrid",
    )
    parser.add_argument(
        "--europa_info", action="store_true", help="Show Europa ice drilling info"
    )
    parser.add_argument(
        "--europa_config",
        action="store_true",
        help="Show Europa configuration from spec",
    )
    parser.add_argument(
        "--europa_simulate",
        action="store_true",
        help="Run Europa ice drilling simulation",
    )
    parser.add_argument(
        "--europa_depth",
        type=int,
        default=1000,
        help="Europa drilling depth in meters (default: 1000)",
    )
    parser.add_argument(
        "--europa_duration",
        type=int,
        default=30,
        help="Europa simulation duration in days (default: 30)",
    )
    parser.add_argument(
        "--europa_drill_rate",
        type=float,
        default=2.0,
        help="Europa drill rate m/hr (default: 2.0)",
    )
    parser.add_argument(
        "--europa_resupply",
        type=float,
        default=180.0,
        help="Europa resupply interval in days (default: 180)",
    )

    # NREL perovskite validation flags
    parser.add_argument(
        "--nrel_info", action="store_true", help="Show NREL validation configuration"
    )
    parser.add_argument(
        "--nrel_config", action="store_true", help="Show NREL configuration from spec"
    )
    parser.add_argument(
        "--nrel_validate", action="store_true", help="Run NREL efficiency validation"
    )
    parser.add_argument(
        "--nrel_efficiency",
        type=float,
        default=0.256,
        help="NREL efficiency to validate (default: 0.256)",
    )
    parser.add_argument(
        "--nrel_project", action="store_true", help="Project NREL degradation over time"
    )
    parser.add_argument(
        "--nrel_years", type=int, default=25, help="NREL projection years (default: 25)"
    )
    parser.add_argument(
        "--nrel_initial",
        type=float,
        default=0.0,
        help="NREL initial efficiency for projection (default: lab value)",
    )
    parser.add_argument(
        "--nrel_compare", action="store_true", help="Compare NREL to MOXIE efficiency"
    )
    parser.add_argument(
        "--moxie_efficiency",
        type=float,
        default=0.0,
        help="MOXIE efficiency for comparison (default: baseline)",
    )

    # Expanded AGI audit flags
    parser.add_argument(
        "--audit_injection", action="store_true", help="Run injection attack audit"
    )
    parser.add_argument(
        "--audit_poisoning", action="store_true", help="Run poisoning attack audit"
    )
    parser.add_argument(
        "--audit_expanded", action="store_true", help="Run all expanded audits"
    )

    # D8 + Multi-planet sync flags
    parser.add_argument(
        "--d8_push", action="store_true", help="Run D8 recursion for alpha>=3.45"
    )
    parser.add_argument(
        "--d8_info", action="store_true", help="Show D8 + unified RL configuration"
    )
    parser.add_argument(
        "--d8_multi_sync", action="store_true", help="Run integrated D8+sync"
    )
    parser.add_argument(
        "--sync_info", action="store_true", help="Show sync configuration"
    )
    parser.add_argument("--sync_run", action="store_true", help="Run sync cycle")
    parser.add_argument(
        "--sync_efficiency", action="store_true", help="Show efficiency metrics"
    )

    # D8 + Atacama validation flags
    parser.add_argument(
        "--atacama_info", action="store_true", help="Show Atacama configuration"
    )
    parser.add_argument(
        "--atacama_validate", action="store_true", help="Run Atacama validation"
    )

    # D8 + Fractal encryption flags
    parser.add_argument(
        "--encrypt_info", action="store_true", help="Show encryption configuration"
    )
    parser.add_argument(
        "--encrypt_keygen", action="store_true", help="Generate fractal key"
    )
    parser.add_argument(
        "--encrypt_audit", action="store_true", help="Run full encryption audit"
    )
    parser.add_argument(
        "--encrypt_side_channel", action="store_true", help="Test side-channel defense"
    )
    parser.add_argument(
        "--encrypt_inversion", action="store_true", help="Test model inversion defense"
    )
    parser.add_argument(
        "--encrypt_key_depth",
        type=int,
        default=6,
        help="Fractal key depth (default: 6)",
    )
    parser.add_argument(
        "--encrypt_iterations",
        type=int,
        default=100,
        help="Encryption test iterations (default: 100)",
    )

    # D9 + Ganymede + randomized paths flags
    parser.add_argument(
        "--d9_push", action="store_true", help="Run D9 recursion for alpha>=3.50"
    )
    parser.add_argument("--d9_info", action="store_true", help="Show D9 configuration")
    parser.add_argument(
        "--d9_ganymede_hybrid", action="store_true", help="Run integrated D9+Ganymede"
    )
    parser.add_argument(
        "--ganymede_info", action="store_true", help="Show Ganymede configuration"
    )
    parser.add_argument(
        "--ganymede_config", action="store_true", help="Show Ganymede config from spec"
    )
    parser.add_argument(
        "--ganymede_navigate", action="store_true", help="Run navigation simulation"
    )
    parser.add_argument(
        "--ganymede_nav_mode",
        type=str,
        default="field_following",
        help="Navigation mode (field_following, magnetopause_crossing, polar_transit)",
    )
    parser.add_argument(
        "--ganymede_duration",
        type=int,
        default=24,
        help="Navigation duration in hours (default: 24)",
    )
    parser.add_argument(
        "--ganymede_autonomy", action="store_true", help="Show autonomy metrics"
    )

    # Atacama drone array flags
    parser.add_argument(
        "--drone_info", action="store_true", help="Show drone array configuration"
    )
    parser.add_argument(
        "--drone_config", action="store_true", help="Show drone config from spec"
    )
    parser.add_argument(
        "--drone_coverage", action="store_true", help="Run drone coverage simulation"
    )
    parser.add_argument(
        "--drone_sample", action="store_true", help="Run drone dust sampling"
    )
    parser.add_argument(
        "--drone_validate", action="store_true", help="Run full drone validation"
    )
    parser.add_argument(
        "--drone_count", type=int, default=10, help="Number of drones (default: 10)"
    )
    parser.add_argument(
        "--drone_area",
        type=float,
        default=1000.0,
        help="Area to cover in km2 (default: 1000)",
    )
    parser.add_argument(
        "--drone_duration",
        type=int,
        default=60,
        help="Sampling duration in seconds (default: 60)",
    )

    # Randomized execution paths flags
    parser.add_argument(
        "--randomized_info",
        action="store_true",
        help="Show randomized paths configuration",
    )
    parser.add_argument(
        "--randomized_config",
        action="store_true",
        help="Show randomized config from spec",
    )
    parser.add_argument(
        "--randomized_generate", action="store_true", help="Generate execution tree"
    )
    parser.add_argument(
        "--randomized_audit", action="store_true", help="Run full randomized audit"
    )
    parser.add_argument(
        "--randomized_timing", action="store_true", help="Test timing resilience"
    )
    parser.add_argument(
        "--randomized_power", action="store_true", help="Test power resilience"
    )
    parser.add_argument(
        "--randomized_cache", action="store_true", help="Test cache resilience"
    )
    parser.add_argument(
        "--randomized_depth",
        type=int,
        default=8,
        help="Execution tree depth (default: 8)",
    )
    parser.add_argument(
        "--randomized_iterations",
        type=int,
        default=100,
        help="Randomized test iterations (default: 100)",
    )
    parser.add_argument(
        "--threat_level",
        type=str,
        default="medium",
        help="Threat level for path depth recommendation (low, medium, high, critical)",
    )

    # D10 + Jovian hub flags
    parser.add_argument(
        "--d10_push", action="store_true", help="Run D10 recursion for alpha>=3.55"
    )
    parser.add_argument(
        "--d10_info", action="store_true", help="Show D10 configuration"
    )
    parser.add_argument(
        "--d10_jovian_hub", action="store_true", help="Run integrated D10+Jovian hub"
    )
    parser.add_argument(
        "--jovian_info", action="store_true", help="Show Jovian hub configuration"
    )
    parser.add_argument(
        "--jovian_sync", action="store_true", help="Run Jovian sync cycle"
    )
    parser.add_argument(
        "--jovian_autonomy", action="store_true", help="Show Jovian system autonomy"
    )
    parser.add_argument(
        "--jovian_coordinate", action="store_true", help="Run full Jovian coordination"
    )

    # Callisto flags
    parser.add_argument(
        "--callisto_info", action="store_true", help="Show Callisto configuration"
    )
    parser.add_argument(
        "--callisto_config", action="store_true", help="Show Callisto config from spec"
    )
    parser.add_argument(
        "--callisto_ice", action="store_true", help="Show Callisto ice availability"
    )
    parser.add_argument(
        "--callisto_extract", action="store_true", help="Run Callisto extraction sim"
    )
    parser.add_argument(
        "--callisto_radiation",
        action="store_true",
        help="Show Callisto radiation advantage",
    )
    parser.add_argument(
        "--callisto_autonomy", action="store_true", help="Show Callisto autonomy"
    )
    parser.add_argument(
        "--callisto_hub", action="store_true", help="Evaluate Callisto hub suitability"
    )
    parser.add_argument(
        "--callisto_rate",
        type=float,
        default=100.0,
        help="Callisto extraction rate kg/hr (default: 100)",
    )
    parser.add_argument(
        "--callisto_duration",
        type=int,
        default=30,
        help="Callisto extraction duration days (default: 30)",
    )

    # Quantum-resistant flags
    parser.add_argument(
        "--quantum_resist_info",
        action="store_true",
        help="Show quantum-resistant config",
    )
    parser.add_argument(
        "--quantum_resist_config",
        action="store_true",
        help="Show quantum config from spec",
    )
    parser.add_argument(
        "--quantum_keygen", action="store_true", help="Generate quantum-resistant key"
    )
    parser.add_argument(
        "--quantum_key_size",
        type=int,
        default=256,
        help="Quantum key size in bits (default: 256)",
    )
    parser.add_argument(
        "--quantum_audit", action="store_true", help="Run full quantum-resistant audit"
    )
    parser.add_argument(
        "--quantum_spectre", action="store_true", help="Test Spectre defense"
    )
    parser.add_argument(
        "--quantum_cache", action="store_true", help="Test quantum cache timing defense"
    )
    parser.add_argument(
        "--spectre_v1", action="store_true", help="Test Spectre v1 defense"
    )
    parser.add_argument(
        "--spectre_v2", action="store_true", help="Test Spectre v2 defense"
    )
    parser.add_argument(
        "--spectre_v4", action="store_true", help="Test Spectre v4 defense"
    )
    parser.add_argument(
        "--quantum_iterations",
        type=int,
        default=100,
        help="Quantum test iterations (default: 100)",
    )

    # Dust dynamics flags
    parser.add_argument(
        "--dust_info", action="store_true", help="Show dust dynamics configuration"
    )
    parser.add_argument(
        "--dust_config", action="store_true", help="Show dust config from spec"
    )
    parser.add_argument(
        "--dust_dynamics", action="store_true", help="Run dust dynamics validation"
    )
    parser.add_argument(
        "--dust_settling", action="store_true", help="Simulate dust settling"
    )
    parser.add_argument(
        "--dust_particle", action="store_true", help="Analyze particle distribution"
    )
    parser.add_argument(
        "--dust_solar_impact", action="store_true", help="Compute solar panel impact"
    )
    parser.add_argument(
        "--dust_mars", action="store_true", help="Project Mars conditions from Atacama"
    )
    parser.add_argument(
        "--dust_depth_mm",
        type=float,
        default=1.0,
        help="Dust depth in mm for solar impact (default: 1.0)",
    )
    parser.add_argument(
        "--dust_duration",
        type=int,
        default=30,
        help="Dust settling duration days (default: 30)",
    )

    # D11 + Venus acid-cloud flags
    parser.add_argument(
        "--d11_push", action="store_true", help="Run D11 recursion for alpha>=3.60"
    )
    parser.add_argument("--d11_info", action="store_true", help="Show D11 configuration")
    parser.add_argument(
        "--d11_venus_hybrid", action="store_true", help="Run integrated D11+Venus hybrid"
    )
    parser.add_argument(
        "--venus_info", action="store_true", help="Show Venus configuration"
    )
    parser.add_argument(
        "--venus_cloud", action="store_true", help="Show Venus cloud zone analysis"
    )
    parser.add_argument(
        "--venus_acid", action="store_true", help="Test Venus acid resistance"
    )
    parser.add_argument(
        "--venus_ops", action="store_true", help="Run Venus operations simulation"
    )
    parser.add_argument(
        "--venus_autonomy", action="store_true", help="Show Venus autonomy metrics"
    )
    parser.add_argument(
        "--venus_altitude",
        type=float,
        default=55.0,
        help="Venus altitude in km (default: 55.0)",
    )
    parser.add_argument(
        "--venus_material",
        type=str,
        default="ptfe",
        help="Venus material for acid test (default: ptfe)",
    )
    parser.add_argument(
        "--venus_duration",
        type=int,
        default=30,
        help="Venus simulation duration days (default: 30)",
    )

    # CFD Navier-Stokes flags
    parser.add_argument(
        "--cfd_info", action="store_true", help="Show CFD configuration"
    )
    parser.add_argument(
        "--cfd_reynolds", action="store_true", help="Compute Reynolds number"
    )
    parser.add_argument(
        "--cfd_settling", action="store_true", help="Compute Stokes settling velocity"
    )
    parser.add_argument(
        "--cfd_storm", action="store_true", help="Run dust storm simulation"
    )
    parser.add_argument(
        "--cfd_validate", action="store_true", help="Run full CFD validation"
    )
    parser.add_argument(
        "--cfd_velocity",
        type=float,
        default=1.0,
        help="CFD flow velocity m/s (default: 1.0)",
    )
    parser.add_argument(
        "--cfd_length",
        type=float,
        default=0.001,
        help="CFD characteristic length m (default: 0.001)",
    )
    parser.add_argument(
        "--cfd_particle_size",
        type=float,
        default=10.0,
        help="CFD particle size um (default: 10.0)",
    )
    parser.add_argument(
        "--cfd_intensity",
        type=float,
        default=0.5,
        help="CFD storm intensity 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--cfd_duration",
        type=float,
        default=24.0,
        help="CFD storm duration hrs (default: 24.0)",
    )

    # Secure enclave flags
    parser.add_argument(
        "--enclave_info", action="store_true", help="Show secure enclave configuration"
    )
    parser.add_argument(
        "--enclave_init", action="store_true", help="Initialize secure enclave"
    )
    parser.add_argument(
        "--enclave_audit", action="store_true", help="Run full secure enclave audit"
    )
    parser.add_argument(
        "--enclave_btb", action="store_true", help="Test BTB injection defense"
    )
    parser.add_argument(
        "--enclave_pht", action="store_true", help="Test PHT poisoning defense"
    )
    parser.add_argument(
        "--enclave_rsb", action="store_true", help="Test RSB stuffing defense"
    )
    parser.add_argument(
        "--enclave_overhead", action="store_true", help="Measure enclave defense overhead"
    )
    parser.add_argument(
        "--enclave_memory",
        type=int,
        default=128,
        help="Enclave memory MB (default: 128)",
    )
    parser.add_argument(
        "--enclave_iterations",
        type=int,
        default=100,
        help="Enclave test iterations (default: 100)",
    )

    args = parser.parse_args()
    reroute_enabled = args.reroute or args.reroute_enabled

    # === DISPATCH ===

    # Info commands (no args needed)
    if args.rl_info:
        return cmd_rl_info()
    if args.adaptive_info:
        return cmd_adaptive_info()
    if args.rl_status:
        return cmd_rl_status()
    if args.validate_no_static:
        return cmd_validate_no_static()
    if args.depth_scaling_info:
        return cmd_depth_scaling_info()
    if args.efficient_sweep_info:
        return cmd_efficient_sweep_info()
    if args.rl_500_sweep_info:
        return cmd_rl_500_sweep_info()
    if args.pilot_info:
        return cmd_pilot_info()
    if args.quantum_rl_hybrid_info:
        return cmd_quantum_rl_hybrid_info()
    if args.pipeline_info:
        return cmd_pipeline_info()
    if args.scale_info:
        return cmd_scale_info()
    if args.fractal_info:
        return cmd_fractal_info()
    if args.algo_info:
        return cmd_algo_info()
    if args.gnn_stub:
        return cmd_gnn_stub()
    if args.innovation_stubs:
        return cmd_innovation_stubs()
    if args.pruning_info:
        return cmd_pruning_info()
    if args.formula_check:
        return cmd_formula_check()
    if args.retention_curve:
        return cmd_retention_curve()

    # Benchmark (10^12 scale) commands
    if args.benchmark_info:
        return cmd_benchmark_info()
    if args.hybrid_10e12:
        return cmd_hybrid_10e12(args.tree_size, args.base_alpha, args.simulate)
    if args.release_gate:
        return cmd_release_gate(args.simulate)
    if args.fractal_recursion_sweep:
        return cmd_fractal_recursion_sweep(
            args.tree_size, args.base_alpha, args.simulate
        )
    if args.fractal_recursion:
        return cmd_fractal_recursion(
            args.tree_size, args.base_alpha, args.recursion_depth, args.simulate
        )

    # Path exploration commands
    if args.d4_info:
        return cmd_d4_info()
    if args.registry_info:
        return cmd_registry_info()
    if args.d4_push:
        return cmd_d4_push(args.tree_size, args.base_alpha, args.simulate)
    if args.path_status:
        return cmd_path_status(args.path)
    if args.path_list:
        return cmd_path_list()
    if args.mars_status:
        return cmd_mars_status()
    if args.multiplanet_status:
        return cmd_multiplanet_status()
    if args.agi_status:
        return cmd_agi_status()
    if args.path is not None and args.path_cmd is not None:
        return cmd_path_run(args.path, args.path_cmd)

    # D5 + ISRU hybrid commands
    if args.moxie_info:
        return cmd_moxie_info()
    if args.isru_info:
        return cmd_isru_info()
    if args.d5_info:
        return cmd_d5_info_isru()
    if args.isru_closure:
        return cmd_isru_closure(args.simulate)
    if args.isru_simulate:
        return cmd_isru_simulate(args.hours, args.crew, args.moxie_units, args.simulate)
    if args.d5_push:
        return cmd_d5_push_isru(args.tree_size, args.base_alpha, args.simulate)
    if args.d5_isru_hybrid:
        return cmd_d5_isru_hybrid(
            args.tree_size,
            args.base_alpha,
            args.crew,
            args.hours,
            args.moxie_units,
            args.simulate,
        )

    # D6 + Titan hybrid commands
    if args.d6_info:
        return cmd_d6_info()
    if args.titan_info:
        return cmd_titan_info()
    if args.titan_config:
        return cmd_titan_config()
    if args.perovskite_info:
        return cmd_perovskite_info()
    if args.d6_push:
        return cmd_d6_push(args.tree_size, args.base_alpha, args.simulate)
    if args.titan_simulate:
        return cmd_titan_simulate(
            args.titan_duration, args.titan_extraction_rate, args.simulate
        )
    if args.d6_titan_hybrid:
        return cmd_d6_titan_hybrid(
            args.tree_size, args.base_alpha, args.titan_duration, args.simulate
        )
    if args.perovskite_project:
        return cmd_perovskite_project(args.perovskite_years, args.perovskite_growth)

    # Adversarial audit commands
    if args.audit_info:
        return cmd_audit_info()
    if args.audit_config:
        return cmd_audit_config()
    if args.audit_run:
        return cmd_audit_run(args.audit_noise, args.audit_iterations, args.simulate)
    if args.audit_stress:
        return cmd_audit_stress(None, 50, args.simulate)

    # D7 + Europa hybrid commands
    if args.d7_info:
        return cmd_d7_info()
    if args.europa_info:
        return cmd_europa_info()
    if args.europa_config:
        return cmd_europa_config()
    if args.d7_push:
        return cmd_d7_push(args.tree_size, args.base_alpha, args.simulate)
    if args.europa_simulate:
        return cmd_europa_simulate(
            args.europa_depth,
            args.europa_duration,
            args.europa_drill_rate,
            args.simulate,
        )
    if args.d7_europa_hybrid:
        return cmd_d7_europa_hybrid(
            args.tree_size,
            args.base_alpha,
            args.europa_depth,
            args.europa_duration,
            args.simulate,
        )

    # NREL perovskite validation commands
    if args.nrel_info:
        return cmd_nrel_info()
    if args.nrel_config:
        return cmd_nrel_config()
    if args.nrel_validate:
        return cmd_nrel_validate(args.nrel_efficiency, args.simulate)
    if args.nrel_project:
        return cmd_nrel_project(args.nrel_years, args.nrel_initial, args.simulate)
    if args.nrel_compare:
        return cmd_nrel_compare(
            args.nrel_efficiency, args.moxie_efficiency, args.simulate
        )

    # D8 + Multi-planet sync commands
    if args.d8_info:
        return cmd_d8_info()
    if args.sync_info:
        return cmd_sync_info()
    if args.encrypt_info:
        return cmd_encrypt_info()
    if args.atacama_info:
        return cmd_atacama_info()
    if args.d8_push:
        return cmd_d8_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d8_multi_sync:
        return cmd_d8_multi_sync(args.tree_size, args.base_alpha, args.simulate)
    if args.sync_run:
        return cmd_sync_run(args.simulate)
    if args.sync_efficiency:
        return cmd_sync_efficiency()
    if args.atacama_validate:
        return cmd_atacama_validate(args.simulate)
    if args.encrypt_keygen:
        return cmd_encrypt_keygen(args.encrypt_key_depth)
    if args.encrypt_audit:
        return cmd_encrypt_audit(args.simulate)
    if args.encrypt_side_channel:
        return cmd_encrypt_side_channel(args.encrypt_iterations)
    if args.encrypt_inversion:
        return cmd_encrypt_inversion(args.encrypt_iterations)

    # D9 + Ganymede + randomized paths commands
    if args.d9_info:
        return cmd_d9_info()
    if args.ganymede_info:
        return cmd_ganymede_info()
    if args.ganymede_config:
        return cmd_ganymede_config()
    if args.drone_info:
        return cmd_drone_info()
    if args.drone_config:
        return cmd_drone_config()
    if args.randomized_info:
        return cmd_randomized_info()
    if args.randomized_config:
        return cmd_randomized_config()
    if args.d9_push:
        return cmd_d9_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d9_ganymede_hybrid:
        return cmd_d9_ganymede_hybrid(
            args.tree_size,
            args.base_alpha,
            args.ganymede_nav_mode,
            args.ganymede_duration,
            args.simulate,
        )
    if args.ganymede_navigate:
        return cmd_ganymede_navigate(
            args.ganymede_nav_mode, args.ganymede_duration, args.simulate
        )
    if args.ganymede_autonomy:
        return cmd_ganymede_autonomy(args.simulate)
    if args.drone_coverage:
        return cmd_drone_coverage(args.drone_count, args.drone_area, args.simulate)
    if args.drone_sample:
        return cmd_drone_sample(10, args.drone_duration, args.simulate)
    if args.drone_validate:
        return cmd_drone_validate(
            args.drone_count, args.drone_area, args.drone_duration, args.simulate
        )
    if args.randomized_generate:
        return cmd_randomized_generate(args.randomized_depth)
    if args.randomized_audit:
        return cmd_randomized_audit(args.randomized_iterations, args.simulate)
    if args.randomized_timing:
        return cmd_randomized_timing(args.randomized_iterations, args.simulate)
    if args.randomized_power:
        return cmd_randomized_power(args.randomized_iterations, args.simulate)
    if args.randomized_cache:
        return cmd_randomized_cache(args.randomized_iterations, args.simulate)

    # D10 + Jovian hub commands
    if args.d10_info:
        return cmd_d10_info()
    if args.d10_push:
        return cmd_d10_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d10_jovian_hub:
        return cmd_d10_jovian_hub(args.tree_size, args.base_alpha, args.simulate)
    if args.jovian_info:
        return cmd_jovian_info()
    if args.jovian_sync:
        return cmd_jovian_sync(args.simulate)
    if args.jovian_autonomy:
        return cmd_jovian_autonomy(args.simulate)
    if args.jovian_coordinate:
        return cmd_jovian_coordinate(args.simulate)

    # Callisto commands
    if args.callisto_info:
        return cmd_callisto_info()
    if args.callisto_config:
        return cmd_callisto_config()
    if args.callisto_ice:
        return cmd_callisto_ice(args.simulate)
    if args.callisto_extract:
        return cmd_callisto_extract(
            args.callisto_duration, args.callisto_rate, args.simulate
        )
    if args.callisto_radiation:
        return cmd_callisto_radiation(args.simulate)
    if args.callisto_autonomy:
        return cmd_callisto_autonomy(args.simulate)
    if args.callisto_hub:
        return cmd_callisto_hub_suitability(args.simulate)

    # Quantum-resistant commands
    if args.quantum_resist_info:
        return cmd_quantum_resist_info()
    if args.quantum_resist_config:
        return cmd_quantum_resist_config()
    if args.quantum_keygen:
        return cmd_quantum_keygen(args.quantum_key_size)
    if args.quantum_audit:
        return cmd_quantum_resist_audit(args.quantum_iterations, args.simulate)
    if args.quantum_spectre:
        return cmd_quantum_spectre(args.quantum_iterations, args.simulate)
    if args.quantum_cache:
        return cmd_quantum_resist_cache(args.quantum_iterations, args.simulate)
    if args.spectre_v1:
        return cmd_quantum_spectre_v1(args.quantum_iterations, args.simulate)
    if args.spectre_v2:
        return cmd_quantum_spectre_v2(args.quantum_iterations, args.simulate)
    if args.spectre_v4:
        return cmd_quantum_spectre_v4(args.quantum_iterations, args.simulate)

    # Dust dynamics commands
    if args.dust_info:
        return cmd_dust_dynamics_info()
    if args.dust_config:
        return cmd_dust_dynamics_config()
    if args.dust_dynamics:
        return cmd_dust_dynamics(args.simulate)
    if args.dust_settling:
        return cmd_dust_settling(args.dust_duration, args.simulate)
    if args.dust_particle:
        return cmd_dust_particle(args.simulate)
    if args.dust_solar_impact:
        return cmd_dust_solar_impact(args.dust_depth_mm, args.simulate)
    if args.dust_mars:
        return cmd_dust_mars_projection(args.simulate)

    # D11 + Venus acid-cloud commands
    if args.d11_info:
        return cmd_d11_info()
    if args.d11_push:
        return cmd_d11_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d11_venus_hybrid:
        return cmd_d11_venus_hybrid(args.tree_size, args.base_alpha, args.simulate)
    if args.venus_info:
        return cmd_venus_info()
    if args.venus_cloud:
        return cmd_venus_cloud(args.venus_altitude)
    if args.venus_acid:
        return cmd_venus_acid(args.venus_material)
    if args.venus_ops:
        return cmd_venus_ops(args.venus_duration, args.venus_altitude)
    if args.venus_autonomy:
        return cmd_venus_autonomy()

    # CFD Navier-Stokes commands
    if args.cfd_info:
        return cmd_cfd_info()
    if args.cfd_reynolds:
        return cmd_cfd_reynolds(args.cfd_velocity, args.cfd_length)
    if args.cfd_settling:
        return cmd_cfd_settling(args.cfd_particle_size)
    if args.cfd_storm:
        return cmd_cfd_storm(args.cfd_intensity, args.cfd_duration)
    if args.cfd_validate:
        return cmd_cfd_validate()

    # Secure enclave commands
    if args.enclave_info:
        return cmd_enclave_info()
    if args.enclave_init:
        return cmd_enclave_init(args.enclave_memory)
    if args.enclave_audit:
        return cmd_enclave_audit(args.enclave_iterations)
    if args.enclave_btb:
        return cmd_enclave_btb(args.enclave_iterations)
    if args.enclave_pht:
        return cmd_enclave_pht(args.enclave_iterations)
    if args.enclave_rsb:
        return cmd_enclave_rsb(args.enclave_iterations)
    if args.enclave_overhead:
        return cmd_enclave_overhead()

    # Expanded AGI audit commands
    if args.audit_expanded:
        from src.agi_audit_expanded import run_expanded_audit
        import json

        result = run_expanded_audit(attack_type="all", iterations=args.audit_iterations)
        print(json.dumps(result, indent=2))
        return
    if args.audit_injection:
        from src.agi_audit_expanded import run_expanded_audit
        import json

        result = run_expanded_audit(
            attack_type="injection", iterations=args.audit_iterations
        )
        print(json.dumps(result, indent=2))
        return
    if args.audit_poisoning:
        from src.agi_audit_expanded import run_expanded_audit
        import json

        result = run_expanded_audit(
            attack_type="poisoning", iterations=args.audit_iterations
        )
        print(json.dumps(result, indent=2))
        return

    # Scale commands
    if args.scalability_gate_test:
        return cmd_scalability_gate_test()
    if args.multi_scale_sweep is not None:
        return cmd_multi_scale_sweep(args.multi_scale_sweep, args.simulate)

    # Fractal ceiling breach commands
    if args.fractal_info_hybrid:
        return cmd_fractal_info_hybrid()
    if args.hybrid_boost_info:
        return cmd_hybrid_boost_info()
    if args.fractal_push:
        return cmd_fractal_push(args.tree_size, args.base_alpha, args.simulate)
    if args.alpha_boost is not None:
        return cmd_alpha_boost(
            args.alpha_boost, args.tree_size, args.base_alpha, args.simulate
        )
    if args.full_500_sweep:
        lr_range = tuple(args.lr_range) if args.lr_range else (RL_LR_MIN, RL_LR_MAX)
        return cmd_full_500_sweep(
            args.tree_size, lr_range, args.retention_target, args.simulate
        )

    # RL commands
    if args.rl_tune and args.blackout is not None:
        return cmd_rl_tune(
            args.blackout,
            args.rl_episodes,
            True,
            args.adaptive or args.dynamic,
            args.simulate,
        )
    if args.dynamic and args.blackout is not None:
        return cmd_dynamic_mode(args.blackout, args.rl_episodes, args.simulate)
    if args.tune_sweep:
        return cmd_tune_sweep(args.simulate)
    if args.quantum_estimate:
        return cmd_quantum_estimate(args.retention_target)
    if args.rl_500_sweep:
        lr_range = tuple(args.lr_range) if args.lr_range else (RL_LR_MIN, RL_LR_MAX)
        return cmd_rl_500_sweep(
            args.tree_size, lr_range, args.retention_target, args.simulate
        )

    # Pipeline commands
    if args.full_pipeline:
        return cmd_full_pipeline(
            args.pilot_runs,
            args.quantum_runs,
            args.sweep_runs,
            args.tree_size,
            args.simulate,
        )
    if args.post_tune_execute:
        return cmd_post_tune_execute(args.tree_size, args.simulate)
    if args.lr_pilot is not None:
        runs = args.lr_pilot if args.lr_pilot else 50
        return cmd_lr_pilot(runs, args.tree_size, args.simulate, args.show_bounds)
    if args.quantum_sim is not None:
        runs = args.quantum_sim if args.quantum_sim else 10
        return cmd_quantum_sim(runs, args.simulate)

    # Depth commands
    if args.compute_depth:
        return cmd_compute_depth_single(args.tree_size)
    if args.depth_scaling_test:
        return cmd_depth_scaling_test()
    if args.adaptive_depth_run:
        return cmd_adaptive_depth_run(args.rl_sweep, args.tree_size, args.simulate)

    # Ablation commands
    if args.ceiling_track is not None:
        return cmd_ceiling_track(args.ceiling_track)
    if args.isolate_layers:
        blackout_days = args.blackout if args.blackout is not None else 150
        return cmd_isolate_layers(blackout_days, args.simulate)
    if args.ablation_sweep:
        blackout_days = args.blackout if args.blackout is not None else 150
        return cmd_ablation_sweep(blackout_days, args.simulate)
    if args.ablate is not None:
        blackout_days = args.blackout if args.blackout is not None else 150
        return cmd_ablate(args.ablate, blackout_days, args.simulate)

    # Pruning commands
    if args.pruning_sweep:
        return cmd_pruning_sweep(args.simulate)
    if args.extended_250d:
        return cmd_extended_250d(args.simulate)
    if args.verify_chain:
        return cmd_verify_chain(args.trim_factor, args.simulate)
    if args.entropy_prune and args.blackout is not None:
        hybrid = args.hybrid_prune or True
        return cmd_entropy_prune(args.blackout, args.trim_factor, hybrid, args.simulate)

    # GNN/cache commands
    if args.cache_sweep:
        return cmd_cache_sweep(args.simulate)
    if args.overflow_test:
        return cmd_overflow_test(args.simulate)
    if args.extreme_sweep is not None:
        return cmd_extreme_sweep(args.extreme_sweep, args.cache_depth, args.simulate)
    if args.gnn_nonlinear and args.blackout is not None:
        return cmd_gnn_nonlinear(args.blackout, args.cache_depth, args.simulate)

    # Extended sweep
    if args.extended_sweep is not None:
        return cmd_extended_sweep(
            args.extended_sweep[0], args.extended_sweep[1], args.simulate
        )

    # Blackout commands
    if args.blackout_sweep:
        return cmd_blackout_sweep(reroute_enabled)
    if args.blackout is not None:
        return cmd_blackout(args.blackout, reroute_enabled, args.simulate)
    if reroute_enabled and args.partition is None and args.blackout is None:
        return cmd_reroute(args.simulate)

    # Partition commands
    if args.stress_quorum:
        return cmd_stress_quorum()
    if args.partition is not None:
        return cmd_partition(args.partition, args.nodes, args.simulate)

    # Timeline
    if args.simulate_timeline:
        return cmd_simulate_timeline(args.c_base, args.p_factor, args.tau)

    # Positional commands
    if args.command is None:
        print(__doc__)
        return

    cmd = args.command.lower()
    if cmd == "baseline":
        cmd_baseline()
    elif cmd == "bootstrap":
        cmd_bootstrap()
    elif cmd == "curve":
        cmd_curve()
    elif cmd == "full":
        cmd_full()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
