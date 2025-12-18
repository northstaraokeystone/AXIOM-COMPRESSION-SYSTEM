"""Pipeline execution CLI commands for AXIOM-CORE.

Commands: lr_pilot, post_tune_execute, full_pipeline, pilot_info, pipeline_info
"""

from src.rl_tune import (
    pilot_lr_narrow,
    run_tuned_sweep,
    load_pilot_spec,
    get_pilot_info,
    PILOT_LR_RUNS,
    INITIAL_LR_RANGE,
    TARGET_NARROWED_LR,
    FULL_TUNED_RUNS,
    RETENTION_TARGET,
)
from src.reasoning import execute_full_pipeline, get_pipeline_info

from cli.base import print_header


def cmd_lr_pilot(runs: int, tree_size: int, simulate: bool, show_bounds: bool):
    """Run LR pilot narrowing.

    Args:
        runs: Number of pilot iterations
        tree_size: Merkle tree size for depth calculation
        simulate: Whether to output simulation receipt
        show_bounds: Show narrowed LR bounds comparison
    """
    print_header(f"LR PILOT NARROWING ({runs} runs)")

    # Load and display spec
    try:
        spec = load_pilot_spec()
        print("\nSpec loaded:")
        print(f"  pilot_runs: {spec.get('pilot_runs', PILOT_LR_RUNS)}")
        lr_min = spec.get("initial_lr_min", INITIAL_LR_RANGE[0])
        lr_max = spec.get("initial_lr_max", INITIAL_LR_RANGE[1])
        print(f"  initial_lr_range: ({lr_min}, {lr_max})")
        target_min = spec.get("target_narrow_min", TARGET_NARROWED_LR[0])
        target_max = spec.get("target_narrow_max", TARGET_NARROWED_LR[1])
        print(f"  target_narrowed_lr: ({target_min}, {target_max})")
        print(f"  full_tuned_runs: {spec.get('full_tuned_runs', FULL_TUNED_RUNS)}")
    except FileNotFoundError:
        print("\nSpec file not found, using defaults")

    print("\nConfiguration:")
    print(f"  Pilot runs: {runs}")
    print(f"  Initial LR range: {INITIAL_LR_RANGE}")
    print(f"  Target narrowed LR: {TARGET_NARROWED_LR}")
    print(f"  Tree size: {tree_size}")

    print("\nRunning LR pilot narrowing...")

    result = pilot_lr_narrow(runs=runs, tree_size=tree_size, seed=42)

    print("\nRESULTS:")
    print(
        f"  Runs completed: {result.get('runs_completed', result.get('pilot_runs', 'N/A'))}"
    )
    print(
        f"  Best LR found: {result.get('best_lr', result.get('optimal_lr_found', 0)):.6f}"
    )
    print(
        f"  Best retention: {result.get('best_retention', result.get('reward_improvement_pct', 0) / 100 + 1):.5f}"
    )
    print(f"  Narrowed range: {result.get('narrowed_range', 'N/A')}")
    range_reduction = result.get("range_reduction_pct", 0)
    if not range_reduction and "narrowed_range" in result:
        initial_span = INITIAL_LR_RANGE[1] - INITIAL_LR_RANGE[0]
        narrowed = result["narrowed_range"]
        if isinstance(narrowed, (list, tuple)) and len(narrowed) == 2:
            narrowed_span = narrowed[1] - narrowed[0]
            range_reduction = (1 - narrowed_span / initial_span) * 100
    print(f"  Range reduction: {range_reduction:.1f}%")

    if show_bounds:
        print("\nBounds Comparison:")
        print(f"  Initial: {INITIAL_LR_RANGE}")
        print(f"  Narrowed: {result['narrowed_range']}")
        print(f"  Target: {TARGET_NARROWED_LR}")
        target_met = (
            result["narrowed_range"][1] - result["narrowed_range"][0]
            <= TARGET_NARROWED_LR
        )
        print(f"  Target met: {'PASS' if target_met else 'PENDING'}")

    print("\nSLO VALIDATION:")
    ret_ok = result["best_retention"] >= 1.0
    print(
        f"  Retention >= 1.0: {'PASS' if ret_ok else 'FAIL'} ({result['best_retention']:.5f})"
    )

    if simulate:
        print("\n[lr_pilot_receipt emitted above]")

    print("=" * 60)


def cmd_post_tune_execute(tree_size: int, simulate: bool):
    """Execute full tuned 500-run sweep (after pilot).

    Args:
        tree_size: Merkle tree size for depth calculation
        simulate: Whether to output simulation receipt
    """
    print_header(f"POST-TUNE EXECUTION ({FULL_TUNED_RUNS} runs)")

    print("\nConfiguration:")
    print(f"  Full tuned runs: {FULL_TUNED_RUNS}")
    print(f"  Tree size: {tree_size}")
    print("  Using narrowed LR from pilot phase")

    print("\nRunning tuned sweep...")

    result = run_tuned_sweep(runs=FULL_TUNED_RUNS, tree_size=tree_size, seed=42)

    print("\nRESULTS:")
    print(f"  Runs completed: {result['runs_completed']}")
    print(f"  Final retention: {result['final_retention']:.5f}")
    print(f"  Best retention: {result['best_retention']:.5f}")
    print(f"  LR used: {result['lr_used']:.6f}")
    print(f"  Target achieved: {'PASS' if result['target_achieved'] else 'PENDING'}")

    if result.get("convergence_run"):
        print(f"  Convergence run: {result['convergence_run']}")

    print("\nSLO VALIDATION:")
    ret_ok = result["best_retention"] >= RETENTION_TARGET
    print(
        f"  Retention >= {RETENTION_TARGET}: {'PASS' if ret_ok else 'FAIL'} ({result['best_retention']:.5f})"
    )

    if simulate:
        print("\n[post_tune_receipt emitted above]")

    print("=" * 60)


def cmd_full_pipeline(
    pilot_runs: int, quantum_runs: int, sweep_runs: int, tree_size: int, simulate: bool
):
    """Run complete pilot -> quantum -> sweep chain.

    Args:
        pilot_runs: Number of pilot runs
        quantum_runs: Number of quantum simulation runs
        sweep_runs: Number of tuned sweep runs
        tree_size: Merkle tree size
        simulate: Whether to output simulation receipt
    """
    print_header("FULL PIPELINE EXECUTION")

    print("\nConfiguration:")
    print(f"  Pilot runs: {pilot_runs}")
    print(f"  Quantum sim runs: {quantum_runs}")
    print(f"  Sweep runs: {sweep_runs}")
    print(f"  Tree size: {tree_size}")

    print("\nExecuting full pipeline...")
    print("  Phase 1: LR Pilot Narrowing")
    print("  Phase 2: Quantum Policy Simulation")
    print("  Phase 3: Tuned Sweep Execution")

    result = execute_full_pipeline(
        pilot_runs=pilot_runs,
        quantum_runs=quantum_runs,
        sweep_runs=sweep_runs,
        tree_size=tree_size,
        seed=42,
    )

    print("\nPIPELINE RESULTS:")

    print("\n  [Phase 1] LR Pilot:")
    pilot = result["pilot_phase"]
    print(f"    Best LR: {pilot['best_lr']:.6f}")
    print(f"    Narrowed range: {pilot['narrowed_range']}")
    print(f"    Best retention: {pilot['best_retention']:.5f}")

    print("\n  [Phase 2] Quantum Sim:")
    quantum = result["quantum_phase"]
    print(f"    Avg retention: {quantum['avg_retention']:.5f}")
    print(f"    Quantum boost: {quantum['quantum_boost_applied']}")

    print("\n  [Phase 3] Tuned Sweep:")
    sweep = result["sweep_phase"]
    print(f"    Final retention: {sweep['final_retention']:.5f}")
    print(f"    Best retention: {sweep['best_retention']:.5f}")
    print(f"    Target achieved: {'PASS' if sweep['target_achieved'] else 'PENDING'}")

    print("\nOVERALL RESULTS:")
    print(f"  Pipeline complete: {result['pipeline_complete']}")
    print(f"  Final retention: {result['final_retention']:.5f}")
    print(f"  SLO met: {result['slo_met']}")

    print("\nSLO VALIDATION:")
    ret_ok = result["final_retention"] >= RETENTION_TARGET
    print(
        f"  Retention >= {RETENTION_TARGET}: {'PASS' if ret_ok else 'FAIL'} ({result['final_retention']:.5f})"
    )

    if simulate:
        print("\n[full_pipeline_receipt emitted above]")

    print("=" * 60)


def cmd_pilot_info():
    """Output LR pilot narrowing configuration."""
    print_header("LR PILOT NARROWING CONFIGURATION")

    info = get_pilot_info()

    print("\nConfiguration:")
    print(f"  Pilot runs: {info['pilot_runs']}")
    print(f"  Initial LR range: {info['initial_lr_range']}")
    print(f"  Target narrowed LR: {info['target_narrowed_lr']}")
    print(f"  Full tuned runs: {info['full_tuned_runs']}")

    print("\nStrategy:")
    print(f"  Method: {info.get('narrowing_strategy', 'top_percentile_reward')}")
    print(
        f"  Expected improvement: {info.get('expected_improvement', '~10% faster convergence')}"
    )

    print("\nExpected Behavior:")
    if "expected_results" in info:
        print(
            f"  Narrowed range: {info['expected_results'].get('narrowed_range', 'N/A')}"
        )
        print(
            f"  Final retention: {info['expected_results'].get('final_retention', 'N/A')}"
        )

    print(f"\nDescription: {info['description']}")

    print("\n[pilot_info receipt emitted above]")
    print("=" * 60)


def cmd_pipeline_info():
    """Output full pipeline configuration."""
    print_header("FULL PIPELINE CONFIGURATION")

    info = get_pipeline_info()

    print("\nPhase Configuration:")
    print(f"  Phase 1 (Pilot): {info.get('phase_1_pilot', 'LR narrowing')}")
    print(f"  Phase 2 (Quantum): {info.get('phase_2_quantum', 'Quantum simulation')}")
    print(f"  Phase 3 (Sweep): {info.get('phase_3_sweep', 'Tuned sweep')}")

    print("\nDefaults:")
    print(f"  Pilot runs: {info.get('default_pilot_runs', info.get('pilot_runs', 50))}")
    print(
        f"  Quantum runs: {info.get('default_quantum_runs', info.get('quantum_runs', 10))}"
    )
    print(
        f"  Sweep runs: {info.get('default_sweep_runs', info.get('sweep_runs', 500))}"
    )

    print("\nTarget:")
    print(f"  Retention target: {info.get('retention_target', 1.05)}")
    print(f"  Alpha target: {info.get('alpha_target', 2.86)}")

    print(
        f"\nDescription: {info.get('description', 'Full pipeline: pilot -> quantum -> sweep')}"
    )

    print("\n[pipeline_info receipt emitted above]")
    print("=" * 60)
