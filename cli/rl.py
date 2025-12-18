"""RL auto-tuning CLI commands for AXIOM-CORE.

Commands: rl_info, adaptive_info, rl_status, validate_no_static, rl_tune,
          dynamic_mode, tune_sweep, rl_500_sweep, rl_500_sweep_info
"""

from src.rl_tune import (
    rl_auto_tune,
    get_rl_tune_info,
    RETENTION_MILESTONE_1,
    ALPHA_TARGET_M1,
    run_500_sweep,
    load_sweep_spec,
    get_500_sweep_info,
    RL_SWEEP_RUNS,
    RETENTION_TARGET,
)
from src.adaptive import get_adaptive_info
from src.adaptive_depth import compute_depth as compute_adaptive_n_depth
from src.reasoning import (
    sovereignty_timeline_dynamic,
    validate_no_static_configs,
    get_rl_integration_status,
)

from cli.base import print_header


def cmd_rl_info():
    """Output RL tuning configuration."""
    print_header("RL AUTO-TUNING CONFIGURATION")

    info = get_rl_tune_info()

    print("\nTargets:")
    print(f"  Retention milestone 1: {info['retention_milestone_1']} (α = {info['alpha_target_m1']})")
    print(f"  Retention milestone 2: {info['retention_milestone_2']} (α = {info['alpha_target_m2']})")
    print(f"  Retention ceiling: {info['retention_ceiling']} (α = {info['alpha_ceiling']})")

    print("\nSafety Bounds:")
    print(f"  Alpha drop threshold: {info['alpha_drop_threshold']}")
    print(f"  Exploration bound: {info['exploration_bound']} (±15%)")
    print(f"  Max episodes without improvement: {info['max_episodes_without_improvement']}")

    print("\nAction Space:")
    print(f"  GNN layers range: {info['gnn_layers_range']}")
    print(f"  LR decay range: {info['lr_decay_range']}")
    print(f"  Prune aggressiveness range: {info['prune_aggressiveness_range']}")

    print("\nPer-layer contribution (validated):")
    print(f"  Range: {info['layer_retention_range']}")

    print(f"\nDescription: {info['description']}")

    print("\n[rl_tune_info receipt emitted above]")
    print("=" * 60)


def cmd_adaptive_info():
    """Output adaptive module configuration."""
    print_header("ADAPTIVE MODULE CONFIGURATION")

    info = get_adaptive_info()

    print("\nDepth Scaling:")
    print(f"  Base depth: {info['adaptive_depth_base']}")
    print(f"  Min depth: {info['adaptive_depth_min']}")
    print(f"  Max depth: {info['adaptive_depth_max']}")
    print(f"  Formula: {info['formulas']['depth']}")

    print("\nLearning Rate Scaling:")
    print(f"  Base LR: {info['lr_base']}")
    print(f"  Min LR: {info['lr_min']}")
    print(f"  Max LR: {info['lr_max']}")
    print(f"  Formula: {info['formulas']['lr']}")

    print("\nPrune Factor Scaling:")
    print(f"  Min prune factor: {info['prune_factor_min']}")
    print(f"  Max prune factor: {info['prune_factor_max']}")
    print(f"  Formula: {info['formulas']['prune']}")

    print(f"\nDescription: {info['description']}")

    print("\n[adaptive_info receipt emitted above]")
    print("=" * 60)


def cmd_rl_status():
    """Output RL integration status."""
    print_header("RL INTEGRATION STATUS")

    status = get_rl_integration_status()

    print("\nModule Readiness:")
    print(f"  RL tune ready: {'PASS' if status['rl_tune_ready'] else 'FAIL'}")
    print(f"  Adaptive ready: {'PASS' if status['adaptive_ready'] else 'FAIL'}")
    print(f"  GNN dynamic ready: {'PASS' if status['gnn_dynamic_ready'] else 'FAIL'}")
    print(f"  Pruning dynamic ready: {'PASS' if status['pruning_dynamic_ready'] else 'FAIL'}")

    print("\nOverall Status:")
    all_ready = status['all_modules_ready']
    print(f"  All modules ready: {'PASS' if all_ready else 'FAIL'}")

    print("\nTargets:")
    targets = status['targets']
    print(f"  Retention milestone 1: {targets['retention_milestone_1']}")
    print(f"  Retention milestone 2: {targets['retention_milestone_2']}")

    print("\n[rl_integration_status receipt emitted above]")
    print("=" * 60)


def cmd_validate_no_static():
    """Verify no static configs remain."""
    print_header("STATIC CONFIG VALIDATION")

    validations = validate_no_static_configs()

    print("\nValidations:")
    for key, value in validations.items():
        if key != "all_pass":
            print(f"  {key}: {'PASS' if value else 'FAIL'}")

    print(f"\nOverall: {'ALL PASS' if validations.get('all_pass', False) else 'SOME FAILED'}")

    print("\n[no_static_configs_validation receipt emitted above]")
    print("=" * 60)


def cmd_rl_tune(blackout_days: int, episodes: int, rl_enabled: bool, adaptive_enabled: bool, simulate: bool):
    """Run RL auto-tuning simulation.

    Args:
        blackout_days: Blackout duration in days
        episodes: Number of RL episodes
        rl_enabled: Whether RL is enabled
        adaptive_enabled: Whether adaptive scaling is enabled
        simulate: Whether to output receipts
    """
    print_header(f"RL AUTO-TUNING ({blackout_days} days, {episodes} episodes)")

    print("\nConfiguration:")
    print(f"  Blackout duration: {blackout_days} days")
    print(f"  RL episodes: {episodes}")
    print(f"  RL enabled: {rl_enabled}")
    print(f"  Adaptive enabled: {adaptive_enabled}")
    print(f"  Target retention: {RETENTION_MILESTONE_1} (α = {ALPHA_TARGET_M1})")

    print("\nRunning RL auto-tuning...")

    result = sovereignty_timeline_dynamic(
        blackout_days=blackout_days,
        rl_enabled=rl_enabled,
        rl_episodes=episodes,
        adaptive_enabled=adaptive_enabled
    )

    print("\nRESULTS:")
    print(f"  Base alpha: {result['base_alpha']}")
    print(f"  Effective alpha: {result['effective_alpha']}")
    if result['rl_best_retention']:
        print(f"  Best retention: {result['rl_best_retention']}")
    print(f"  Target achieved: {'PASS' if result['rl_target_achieved'] else 'PENDING'}")
    if result['adaptive_depth']:
        print(f"  Adaptive depth: {result['adaptive_depth']}")

    print("\nSOVEREIGNTY TIMELINE:")
    print(f"  Cycles to 10K: {result['cycles_to_10k_person_eq']}")
    print(f"  Cycles to 1M: {result['cycles_to_1M_person_eq']}")
    print(f"  Dynamic mode: {result['dynamic_mode']}")

    print("\nSLO VALIDATION:")
    if result['rl_best_retention']:
        ret_ok = result['rl_best_retention'] >= RETENTION_MILESTONE_1
        print(f"  Retention >= {RETENTION_MILESTONE_1}: {'PASS' if ret_ok else 'FAIL'} ({result['rl_best_retention']})")

    if simulate:
        print("\n[sovereignty_timeline_dynamic receipt emitted above]")

    print("=" * 60)


def cmd_dynamic_mode(blackout_days: int, episodes: int, simulate: bool):
    """Run full dynamic mode (RL + adaptive).

    Args:
        blackout_days: Blackout duration in days
        episodes: Number of RL episodes
        simulate: Whether to output receipts
    """
    print_header(f"FULL DYNAMIC MODE ({blackout_days} days)")

    print("\nConfiguration:")
    print("  RL enabled: True")
    print("  Adaptive enabled: True")
    print(f"  Episodes: {episodes}")

    result = sovereignty_timeline_dynamic(
        blackout_days=blackout_days,
        rl_enabled=True,
        rl_episodes=episodes,
        adaptive_enabled=True
    )

    print("\nRESULTS:")
    print(f"  Effective alpha: {result['effective_alpha']}")
    print(f"  Best retention: {result['rl_best_retention']}")
    print(f"  Adaptive depth: {result['adaptive_depth']}")
    print(f"  Target achieved: {result['rl_target_achieved']}")

    if simulate:
        print("\n[sovereignty_timeline_dynamic receipt emitted above]")

    print("=" * 60)


def cmd_tune_sweep(simulate: bool):
    """Run retention sweep from 1.01 to target.

    Args:
        simulate: Whether to output receipts
    """
    print_header("RETENTION TUNING SWEEP")

    print("\nConfiguration:")
    print("  Start retention: 1.01")
    print(f"  Target retention: {RETENTION_MILESTONE_1}")
    print("  Episodes per step: 50")

    print("\nRunning sweep...")

    episodes_list = [10, 50, 100, 200, 500]
    results = []

    for eps in episodes_list:
        result = rl_auto_tune(
            current_retention=1.01,
            blackout_days=150,
            episodes=eps,
            seed=42
        )
        results.append({
            "episodes": eps,
            "best_retention": result["best_retention"],
            "best_alpha": result["best_alpha"],
            "target_achieved": result["target_achieved"]
        })

    print("\nRESULTS:")
    print(f"  {'Episodes':>10} | {'Retention':>10} | {'Alpha':>8} | {'Target':>8}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for r in results:
        target_str = "PASS" if r["target_achieved"] else "PENDING"
        print(f"  {r['episodes']:>10} | {r['best_retention']:>10.4f} | {r['best_alpha']:>8.4f} | {target_str:>8}")

    # Check if target achieved at any episode count
    any_target = any(r["target_achieved"] for r in results)
    print(f"\nTarget {RETENTION_MILESTONE_1} achieved: {'YES' if any_target else 'NOT YET'}")

    if simulate:
        print("\n[rl_tune_receipt emitted per episode]")

    print("=" * 60)


def cmd_rl_500_sweep(tree_size: int, lr_range: tuple, retention_target: float, simulate: bool):
    """Run 500-run RL sweep for 1.05 retention.

    Args:
        tree_size: Merkle tree size for depth calculation
        lr_range: Optional LR range override (min, max)
        retention_target: Override retention target
        simulate: Whether to output simulation receipt
    """
    print_header("500-RUN RL SWEEP FOR 1.05 RETENTION")

    # Load and display spec
    try:
        spec = load_sweep_spec()
        print("\nSpec loaded:")
        print(f"  sweep_runs: {spec['sweep_runs']}")
        print(f"  lr_min: {spec['lr_min']}")
        print(f"  lr_max: {spec['lr_max']}")
        print(f"  retention_target: {spec['retention_target']}")
        print(f"  seed: {spec['seed']}")
    except FileNotFoundError:
        print("\nSpec file not found, using defaults")

    # Compute depth for given tree size
    depth = compute_adaptive_n_depth(tree_size, 0.5)
    print(f"\nComputed depth for n={tree_size:.2e}:")
    print(f"  layers: {depth}")

    print("\nRunning 500-run informed sweep...")

    result = run_500_sweep(
        runs=RL_SWEEP_RUNS,
        tree_size=tree_size,
        adaptive_depth=True,
        early_stopping=True,
        seed=42
    )

    print("\nRESULTS:")
    print(f"  Final retention: {result['final_retention']}")
    print(f"  Best retention: {result['best_retention']}")
    print(f"  Target achieved: {'PASS' if result['target_achieved'] else 'PENDING'}")
    print(f"  Convergence run: {result['convergence_run']}")
    print(f"  Runs completed: {result['runs_completed']}/{result['runs_limit']}")
    print(f"  Depth used: {result['depth_used']}")
    print(f"  LR range: {result['lr_range']}")

    if result['best_action']:
        print("\nBest Action:")
        print(f"  layers_delta: {result['best_action']['layers_delta']}")
        print(f"  lr: {result['best_action']['lr']}")
        print(f"  prune_factor: {result['best_action']['prune_factor']}")

    print("\nSLO VALIDATION:")
    ret_ok = result['best_retention'] >= RETENTION_TARGET
    print(f"  Retention >= {RETENTION_TARGET}: {'PASS' if ret_ok else 'FAIL'} ({result['best_retention']:.5f})")

    if simulate:
        print("\n[rl_500_sweep_receipt emitted above]")

    print("=" * 60)


def cmd_rl_500_sweep_info():
    """Output 500-run sweep configuration."""
    print_header("500-RUN RL SWEEP CONFIGURATION")

    info = get_500_sweep_info()

    print("\nConfiguration:")
    print(f"  sweep_runs: {info['sweep_runs']}")
    print(f"  lr_min: {info['lr_min']}")
    print(f"  lr_max: {info['lr_max']}")
    print(f"  retention_target: {info['retention_target']}")
    print(f"  seed: {info['seed']}")
    print(f"  divergence_threshold: {info['divergence_threshold']}")

    print("\nState Components:")
    for comp in info['state_components']:
        print(f"  - {comp}")

    print("\nAction Components:")
    for comp in info['action_components']:
        print(f"  - {comp}")

    print("\nExpected Behavior:")
    print(f"  Expected convergence: {info['expected_convergence']}")
    print(f"  vs blind: {info['vs_blind']}")

    print(f"\nDescription: {info['description']}")

    print("\n[rl_500_sweep_info receipt emitted above]")
    print("=" * 60)
