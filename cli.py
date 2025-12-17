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
)


def main():
    parser = argparse.ArgumentParser(description="AXIOM-CORE CLI - The Sovereignty Calculator")
    parser.add_argument('command', nargs='?', help='Command: baseline, bootstrap, curve, full')

    # Timeline args
    parser.add_argument('--c_base', type=float, default=C_BASE_DEFAULT,
                        help='Initial person-eq capacity (default: 50)')
    parser.add_argument('--p_factor', type=float, default=P_FACTOR_DEFAULT,
                        help='Propulsion growth factor (default: 1.8)')
    parser.add_argument('--tau', type=float, default=0,
                        help='Latency in seconds (0=Earth, 1200=Mars max)')
    parser.add_argument('--simulate_timeline', action='store_true',
                        help='Run sovereignty timeline simulation')

    # Partition flags
    parser.add_argument('--partition', type=float, default=None,
                        help='Run single partition simulation at specified loss percentage (0-1)')
    parser.add_argument('--nodes', type=int, default=NODE_BASELINE,
                        help='Specify node count for simulation (default: 5)')
    parser.add_argument('--stress_quorum', action='store_true',
                        help='Run full stress sweep (1000 iterations, 0-40%% loss)')
    parser.add_argument('--simulate', action='store_true',
                        help='Output simulation receipt to stdout')

    # Reroute/blackout flags
    parser.add_argument('--reroute', action='store_true', help='Enable adaptive rerouting')
    parser.add_argument('--reroute_enabled', action='store_true', help='Alias for --reroute')
    parser.add_argument('--blackout', type=int, default=None, help='Blackout duration in days')
    parser.add_argument('--blackout_sweep', action='store_true', help='Run full blackout sweep')
    parser.add_argument('--algo_info', action='store_true', help='Output reroute algorithm spec')

    # Extended sweep/retention flags
    parser.add_argument('--extended_sweep', nargs=2, type=int, default=None,
                        metavar=('START', 'END'), help='Extended blackout sweep range')
    parser.add_argument('--retention_curve', action='store_true', help='Output retention curve JSON')
    parser.add_argument('--gnn_stub', action='store_true', help='Echo GNN sensitivity stub')

    # GNN/cache flags
    parser.add_argument('--gnn_nonlinear', action='store_true', help='Use GNN nonlinear model')
    parser.add_argument('--cache_depth', type=int, default=int(1e8), help='Cache depth (default: 1e8)')
    parser.add_argument('--cache_sweep', action='store_true', help='Run cache depth sweep')
    parser.add_argument('--extreme_sweep', type=int, default=None, metavar='DAYS',
                        help='Run extreme blackout sweep to days')
    parser.add_argument('--overflow_test', action='store_true', help='Test cache overflow')
    parser.add_argument('--innovation_stubs', action='store_true', help='Echo innovation stubs')

    # Pruning flags
    parser.add_argument('--entropy_prune', action='store_true', help='Enable entropy pruning')
    parser.add_argument('--trim_factor', type=float, default=0.3, help='Trim factor (default: 0.3)')
    parser.add_argument('--hybrid_prune', action='store_true', help='Enable hybrid pruning')
    parser.add_argument('--pruning_sweep', action='store_true', help='Run pruning sweep')
    parser.add_argument('--extended_250d', action='store_true', help='Run 250d simulation')
    parser.add_argument('--verify_chain', action='store_true', help='Verify chain integrity')
    parser.add_argument('--pruning_info', action='store_true', help='Output pruning config')

    # Ablation flags
    parser.add_argument('--ablate', type=str, default=None, metavar='MODE',
                        help='Ablation mode (full/no_cache/no_prune/baseline)')
    parser.add_argument('--ablation_sweep', action='store_true', help='Run ablation sweep')
    parser.add_argument('--ceiling_track', type=float, default=None, metavar='ALPHA',
                        help='Ceiling gap analysis for alpha')
    parser.add_argument('--formula_check', action='store_true', help='Validate alpha formula')
    parser.add_argument('--isolate_layers', action='store_true', help='Isolate layer contributions')

    # RL flags
    parser.add_argument('--rl_tune', action='store_true', help='Enable RL auto-tuning')
    parser.add_argument('--rl_episodes', type=int, default=100, help='RL episodes (default: 100)')
    parser.add_argument('--adaptive', action='store_true', help='Enable adaptive depth/config')
    parser.add_argument('--dynamic', action='store_true', help='Enable all dynamic features')
    parser.add_argument('--tune_sweep', action='store_true', help='Run retention sweep')
    parser.add_argument('--show_rl_history', action='store_true', help='Output RL history')
    parser.add_argument('--validate_no_static', action='store_true', help='Verify no static configs')
    parser.add_argument('--rl_info', action='store_true', help='Output RL tuning config')
    parser.add_argument('--adaptive_info', action='store_true', help='Output adaptive config')
    parser.add_argument('--rl_status', action='store_true', help='Output RL integration status')

    # Adaptive depth flags
    parser.add_argument('--adaptive_depth_run', action='store_true', help='Run with adaptive depth')
    parser.add_argument('--rl_sweep', type=int, default=500, help='Informed RL runs (default: 500)')
    parser.add_argument('--depth_scaling_test', action='store_true', help='Run depth scaling test')
    parser.add_argument('--compute_depth', action='store_true', help='Show computed depth')
    parser.add_argument('--tree_size', type=int, default=int(1e6), help='Tree size (default: 1e6)')
    parser.add_argument('--depth_scaling_info', action='store_true', help='Output depth scaling config')
    parser.add_argument('--efficient_sweep_info', action='store_true', help='Output sweep config')

    # 500-run sweep flags
    parser.add_argument('--rl_500_sweep', action='store_true', help='Run 500-run RL sweep')
    parser.add_argument('--lr_range', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'),
                        help='Override LR bounds')
    parser.add_argument('--retention_target', type=float, default=1.05, help='Retention target')
    parser.add_argument('--quantum_estimate', action='store_true', help='Show quantum estimate')
    parser.add_argument('--rl_500_sweep_info', action='store_true', help='Output 500-run config')

    # Pipeline flags
    parser.add_argument('--lr_pilot', type=int, default=None, const=50, nargs='?', metavar='RUNS',
                        help='Run LR pilot (default: 50 runs)')
    parser.add_argument('--quantum_sim', type=int, default=None, const=10, nargs='?', metavar='RUNS',
                        help='Run quantum sim (default: 10 runs)')
    parser.add_argument('--post_tune_execute', action='store_true', help='Execute tuned sweep')
    parser.add_argument('--full_pipeline', action='store_true', help='Run complete pipeline')
    parser.add_argument('--show_bounds', action='store_true', help='Show LR bounds comparison')
    parser.add_argument('--pilot_info', action='store_true', help='Output pilot config')
    parser.add_argument('--quantum_rl_hybrid_info', action='store_true', help='Output quantum-RL config')
    parser.add_argument('--pipeline_info', action='store_true', help='Output pipeline config')
    parser.add_argument('--pilot_runs', type=int, default=50, help='Pilot runs (default: 50)')
    parser.add_argument('--quantum_runs', type=int, default=10, help='Quantum runs (default: 10)')
    parser.add_argument('--sweep_runs', type=int, default=500, help='Sweep runs (default: 500)')

    # Scale flags
    parser.add_argument('--multi_scale_sweep', type=str, default=None, metavar='TYPE',
                        help='Run multi-scale sweep ("all" or "1e9")')
    parser.add_argument('--scalability_gate_test', action='store_true',
                        help='Test scalability gate status')
    parser.add_argument('--scale_info', action='store_true',
                        help='Show scale validation configuration')
    parser.add_argument('--fractal_info', action='store_true',
                        help='Show fractal layer configuration')

    # Fractal ceiling breach flags
    parser.add_argument('--alpha_boost', type=str, default=None, metavar='MODE',
                        help='Alpha boost mode (off, quantum, fractal, hybrid)')
    parser.add_argument('--fractal_push', action='store_true',
                        help='Run fractal ceiling breach')
    parser.add_argument('--fractal_info_hybrid', action='store_true',
                        help='Show fractal hybrid configuration')
    parser.add_argument('--base_alpha', type=float, default=2.99,
                        help='Base alpha for fractal/hybrid (default: 2.99)')

    # Full sweep flags
    parser.add_argument('--full_500_sweep', action='store_true',
                        help='Run full 500-sweep with quantum-fractal hybrid')

    # Hybrid info flags
    parser.add_argument('--hybrid_boost_info', action='store_true',
                        help='Show hybrid boost configuration')

    # Benchmark (10^12 scale) flags
    parser.add_argument('--hybrid_10e12', action='store_true',
                        help='Run 10^12 hybrid benchmark')
    parser.add_argument('--release_gate', action='store_true',
                        help='Check release gate 3.1 status')
    parser.add_argument('--fractal_recursion', action='store_true',
                        help='Run fractal recursion ceiling breach')
    parser.add_argument('--fractal_recursion_sweep', action='store_true',
                        help='Sweep through all recursion depths')
    parser.add_argument('--benchmark_info', action='store_true',
                        help='Show benchmark configuration')
    parser.add_argument('--recursion_depth', type=int, default=3,
                        help='Fractal recursion depth (1-5, default: 3)')

    # Path exploration flags
    parser.add_argument('--d4_push', action='store_true',
                        help='Run D4 recursion for alpha>=3.2')
    parser.add_argument('--d4_info', action='store_true',
                        help='Show D4 configuration')
    parser.add_argument('--path', type=str, default=None,
                        help='Select exploration path (mars/multiplanet/agi)')
    parser.add_argument('--path_status', action='store_true',
                        help='Show all path statuses')
    parser.add_argument('--path_list', action='store_true',
                        help='List registered paths')
    parser.add_argument('--path_cmd', type=str, default=None,
                        help='Run command on selected path')
    parser.add_argument('--mars_status', action='store_true',
                        help='Shortcut: --path mars --path_cmd status')
    parser.add_argument('--multiplanet_status', action='store_true',
                        help='Shortcut: --path multiplanet --path_cmd status')
    parser.add_argument('--agi_status', action='store_true',
                        help='Shortcut: --path agi --path_cmd status')
    parser.add_argument('--registry_info', action='store_true',
                        help='Show path registry info')

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
        return cmd_fractal_recursion_sweep(args.tree_size, args.base_alpha, args.simulate)
    if args.fractal_recursion:
        return cmd_fractal_recursion(args.tree_size, args.base_alpha, args.recursion_depth, args.simulate)

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
        return cmd_alpha_boost(args.alpha_boost, args.tree_size, args.base_alpha, args.simulate)
    if args.full_500_sweep:
        lr_range = tuple(args.lr_range) if args.lr_range else (RL_LR_MIN, RL_LR_MAX)
        return cmd_full_500_sweep(args.tree_size, lr_range, args.retention_target, args.simulate)

    # RL commands
    if args.rl_tune and args.blackout is not None:
        return cmd_rl_tune(args.blackout, args.rl_episodes, True, args.adaptive or args.dynamic, args.simulate)
    if args.dynamic and args.blackout is not None:
        return cmd_dynamic_mode(args.blackout, args.rl_episodes, args.simulate)
    if args.tune_sweep:
        return cmd_tune_sweep(args.simulate)
    if args.quantum_estimate:
        return cmd_quantum_estimate(args.retention_target)
    if args.rl_500_sweep:
        lr_range = tuple(args.lr_range) if args.lr_range else (RL_LR_MIN, RL_LR_MAX)
        return cmd_rl_500_sweep(args.tree_size, lr_range, args.retention_target, args.simulate)

    # Pipeline commands
    if args.full_pipeline:
        return cmd_full_pipeline(args.pilot_runs, args.quantum_runs, args.sweep_runs, args.tree_size, args.simulate)
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
        return cmd_extended_sweep(args.extended_sweep[0], args.extended_sweep[1], args.simulate)

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
