"""Adaptive depth CLI commands for AXIOM-CORE.

Commands: adaptive_depth_run, depth_scaling_test, compute_depth_single,
          depth_scaling_info, efficient_sweep_info
"""

import math

from src.adaptive_depth import (
    load_depth_spec,
    compute_depth as compute_adaptive_n_depth,
    get_depth_scaling_info,
)
from src.rl_tune import (
    run_sweep,
    compare_sweep_efficiency,
    get_efficient_sweep_info,
    RETENTION_QUICK_WIN_TARGET,
)

from cli.base import print_header


def cmd_adaptive_depth_run(rl_sweep_runs: int, tree_size: int, simulate: bool):
    """Run with adaptive depth enabled.

    Args:
        rl_sweep_runs: Number of informed RL runs
        tree_size: Merkle tree size for depth calculation
        simulate: Whether to output simulation receipt
    """
    print_header(f"ADAPTIVE DEPTH RUN ({rl_sweep_runs} runs)")

    # Load and display spec
    spec = load_depth_spec()
    print("\nSpec loaded:")
    print(f"  base_layers: {spec['base_layers']}")
    print(f"  scale_factor: {spec['scale_factor']}")
    print(f"  baseline_n: {spec['baseline_n']}")
    print(f"  max_layers: {spec['max_layers']}")
    print(f"  sweep_limit: {spec['sweep_limit']}")

    # Compute depth for given tree size
    depth = compute_adaptive_n_depth(tree_size, 0.5)
    print(f"\nComputed depth for n={tree_size:.2e}:")
    print(f"  layers: {depth}")

    print(f"\nRunning {rl_sweep_runs}-run informed sweep...")

    result = run_sweep(
        runs=rl_sweep_runs,
        tree_size=tree_size,
        adaptive_depth=True,
        early_stopping=True,
        seed=42
    )

    print("\nRESULTS:")
    print(f"  Runs completed: {result['runs_completed']}/{result['runs_limit']}")
    print(f"  Best retention: {result['best_retention']:.5f}")
    print(f"  Best alpha: {result['best_alpha']:.5f}")
    print(f"  Depth used: {result['depth_used']}")
    print(f"  Target achieved: {'PASS' if result['target_achieved'] else 'PENDING'}")
    print(f"  Early stopped: {result['early_stopped']}")

    if result['convergence_run']:
        print(f"  Convergence run: {result['convergence_run']}")

    print("\nSLO VALIDATION:")
    ret_ok = result['best_retention'] >= RETENTION_QUICK_WIN_TARGET
    print(f"  Retention >= {RETENTION_QUICK_WIN_TARGET}: {'PASS' if ret_ok else 'FAIL'} ({result['best_retention']:.5f})")

    if simulate:
        print("\n[efficient_rl_sweep receipt emitted above]")

    print("=" * 60)


def cmd_depth_scaling_test():
    """Run depth scaling efficiency comparison."""
    print_header("DEPTH SCALING EFFICIENCY TEST")

    print("\nComparing 500 informed vs 300 blind runs...")

    result = compare_sweep_efficiency(
        informed_runs=500,
        blind_runs=300,
        tree_size=int(1e8),
        seed=42
    )

    print("\nRESULTS:")
    print(f"  Informed (500 runs): {result['informed_retention']:.5f}")
    print(f"  Blind (300 runs): {result['blind_retention']:.5f}")
    print(f"  Efficiency gain: {result['efficiency_gain']:.5f}")
    print(f"  Informed better: {'YES' if result['informed_better'] else 'NO'}")

    print(f"\nCONCLUSION: {result['conclusion']}")

    print("\n[sweep_efficiency_comparison receipt emitted above]")
    print("=" * 60)


def cmd_compute_depth_single(tree_size: int):
    """Show computed depth for given tree size.

    Args:
        tree_size: Tree size to compute depth for
    """
    print_header(f"COMPUTE DEPTH (n={tree_size:.2e})")

    # Load spec
    spec = load_depth_spec()
    print("\nSpec:")
    print(f"  base_layers: {spec['base_layers']}")
    print(f"  scale_factor: {spec['scale_factor']}")
    print(f"  baseline_n: {spec['baseline_n']}")
    print(f"  max_layers: {spec['max_layers']}")

    # Compute depth
    depth = compute_adaptive_n_depth(tree_size, 0.5)

    print("\nFormula: layers = base + scale * log(n / baseline)")
    print(f"  base_layers: {spec['base_layers']}")
    print(f"  scale_factor * log({tree_size} / {spec['baseline_n']})")

    if tree_size > 0:
        raw = spec['base_layers'] + spec['scale_factor'] * math.log(tree_size / spec['baseline_n'])
        print(f"  Raw value: {raw:.4f}")

    print(f"\nRESULT: {depth} layers")

    # Show expected examples
    info = get_depth_scaling_info()
    print("\nExpected depths table:")
    for key, val in info['example_depths'].items():
        print(f"  {key}: {val} layers")

    print("\n[adaptive_depth receipt emitted above]")
    print("=" * 60)


def cmd_depth_scaling_info():
    """Output adaptive depth module configuration."""
    print_header("ADAPTIVE DEPTH SCALING CONFIGURATION")

    info = get_depth_scaling_info()

    print("\nScaling Parameters:")
    print(f"  base_layers: {info['base_layers']}")
    print(f"  scale_factor: {info['scale_factor']}")
    print(f"  baseline_n: {info['baseline_n']}")
    print(f"  max_layers: {info['max_layers']}")
    print(f"  min_layers: {info['min_layers']}")

    print("\nSweep Configuration:")
    print(f"  sweep_limit: {info['sweep_limit']}")
    print(f"  quick_target: {info['quick_target']}")

    print(f"\nFormula: {info['formula']}")

    print("\nExample Depths:")
    for key, val in info['example_depths'].items():
        print(f"  {key}: {val} layers")

    print(f"\nDescription: {info['description']}")

    print("\n[depth_scaling_info receipt emitted above]")
    print("=" * 60)


def cmd_efficient_sweep_info():
    """Output efficient sweep configuration."""
    print_header("EFFICIENT SWEEP CONFIGURATION")

    info = get_efficient_sweep_info()

    print("\nConfiguration:")
    print(f"  sweep_limit: {info['sweep_limit']}")
    print(f"  quick_win_target: {info['quick_win_target']}")
    print(f"  convergence_check_interval: {info['convergence_check_interval']}")
    print(f"  early_stopping_threshold: {info['early_stopping_threshold']}")

    print("\nExpected Behavior:")
    print(f"  Expected convergence: {info['expected_convergence']}")
    print(f"  vs blind: {info['vs_blind']}")

    print(f"\nDescription: {info['description']}")

    print("\n[efficient_sweep_info receipt emitted above]")
    print("=" * 60)
