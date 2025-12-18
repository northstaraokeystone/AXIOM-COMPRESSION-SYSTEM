"""Ablation testing CLI commands for AXIOM-CORE.

Commands: ablate, ablation_sweep, ceiling_track, formula_check, isolate_layers
"""

import math

from src.gnn_cache import (
    nonlinear_retention_with_pruning,
    CACHE_DEPTH_BASELINE,
)
from src.reasoning import ablation_sweep, get_layer_contributions
from src.alpha_compute import (
    alpha_calc,
    ceiling_gap,
    get_ablation_expected,
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
    ABLATION_MODES,
)

from cli.base import print_header


def cmd_ablate(mode: str, blackout_days: int, simulate: bool):
    """Run simulation in specified ablation mode.

    Args:
        mode: Ablation mode (full/no_cache/no_prune/baseline)
        blackout_days: Blackout duration in days
        simulate: Whether to output simulation receipt
    """
    print_header(f"ABLATION TEST: {mode.upper()} ({blackout_days} days)")

    print("\nConfiguration:")
    print(f"  Ablation mode: {mode}")
    print(f"  Blackout duration: {blackout_days} days")
    print(f"  Shannon floor: {SHANNON_FLOOR_ALPHA}")
    print(f"  Ceiling target: {ALPHA_CEILING_TARGET}")

    expected = get_ablation_expected(mode)
    print(f"\nExpected results ({mode}):")
    print(f"  Alpha range: {expected['alpha_range']}")
    print(f"  Retention: {expected['retention']}")
    print(f"  Description: {expected['description']}")

    try:
        result = nonlinear_retention_with_pruning(
            blackout_days,
            CACHE_DEPTH_BASELINE,
            pruning_enabled=(mode != "no_prune" and mode != "baseline"),
            trim_factor=0.3,
            ablation_mode=mode,
        )

        print("\nRESULTS:")
        print(f"  Effective alpha: {result['eff_alpha']}")
        print(f"  Retention factor: {result['retention_factor']}")
        print(f"  GNN boost: {result['gnn_boost']}")
        print(f"  Pruning boost: {result['pruning_boost']}")
        print(f"  Model: {result['model']}")

        # Validate against expected
        alpha_min, alpha_max = expected["alpha_range"]
        alpha_ok = alpha_min <= result["eff_alpha"] <= alpha_max

        print("\nVALIDATION:")
        print(
            f"  Alpha in expected range: {'PASS' if alpha_ok else 'FAIL'} ({result['eff_alpha']} in {expected['alpha_range']})"
        )

        if simulate:
            print("\n[ablation_test receipt emitted above]")

    except Exception as e:
        print(f"\nERROR: {e}")
        if "overflow" in str(e).lower():
            print("Cache overflow - StopRule triggered")

    print("=" * 60)


def cmd_ablation_sweep(blackout_days: int, simulate: bool):
    """Run all 4 ablation modes and compare.

    Args:
        blackout_days: Blackout duration in days
        simulate: Whether to output simulation receipt
    """
    print_header(f"ABLATION SWEEP ({blackout_days} days)")

    print("\nConfiguration:")
    print(f"  Ablation modes: {ABLATION_MODES}")
    print(f"  Blackout duration: {blackout_days} days")
    print("  Iterations per mode: 100")
    print(f"  Shannon floor: {SHANNON_FLOOR_ALPHA}")
    print(f"  Ceiling target: {ALPHA_CEILING_TARGET}")

    print("\nRunning ablation sweep...")

    result = ablation_sweep(
        modes=ABLATION_MODES, blackout_days=blackout_days, iterations=100, seed=42
    )

    print("\nRESULTS BY MODE:")
    print(
        f"  {'Mode':<12} | {'Avg Alpha':>10} | {'Min':>8} | {'Max':>8} | {'Success':>8}"
    )
    print(f"  {'-' * 12}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}")

    for mode in ["baseline", "no_prune", "no_cache", "full"]:
        if mode in result["results_by_mode"]:
            m = result["results_by_mode"][mode]
            print(
                f"  {mode:<12} | {m['avg_alpha']:>10.4f} | {m['min_alpha']:>8.4f} | {m['max_alpha']:>8.4f} | {m['successful']:>8}"
            )

    print("\nORDERING VALIDATION:")
    print("  Expected: baseline < no_prune < no_cache < full")
    print(f"  Ordering valid: {'PASS' if result['ordering_valid'] else 'FAIL'}")

    print("\nLAYER CONTRIBUTIONS:")
    lc = result["layer_contributions"]
    print(f"  GNN contribution: {lc['gnn_contribution']:.4f}")
    print(f"  Prune contribution: {lc['prune_contribution']:.4f}")
    print(f"  Total uplift: {lc['total_uplift']}")

    print("\nCEILING ANALYSIS:")
    gap = result["gap_to_ceiling"]
    print(f"  Current alpha: {gap['current_alpha']}")
    print(f"  Ceiling target: {gap['ceiling_target']}")
    print(f"  Gap: {gap['gap_pct']:.1f}%")
    print(f"  Path: {gap['path_to_ceiling']}")

    if simulate:
        print("\n[ablation_sweep receipt emitted above]")

    print("=" * 60)


def cmd_ceiling_track(current_alpha: float):
    """Output ceiling gap analysis.

    Args:
        current_alpha: Current alpha value to analyze
    """
    print_header("CEILING GAP ANALYSIS")

    result = ceiling_gap(current_alpha, ALPHA_CEILING_TARGET)

    print("\nCurrent Status:")
    print(f"  Current alpha: {result['current_alpha']}")
    print(f"  Ceiling target: {result['ceiling_target']}")
    print(f"  Shannon floor: {SHANNON_FLOOR_ALPHA}")

    print("\nGap Analysis:")
    print(f"  Gap absolute: {result['gap_absolute']}")
    print(f"  Gap percentage: {result['gap_pct']:.1f}%")

    print("\nRetention Analysis:")
    print(f"  Current retention factor: {result['retention_factor_current']}")
    print(f"  Retention needed: {result['retention_factor_needed']}")
    print(f"  Retention delta: {result['retention_factor_delta']}")

    print("\nPath to Ceiling:")
    print(f"  {result['path_to_ceiling']}")

    print("\n[ceiling_track receipt emitted above]")
    print("=" * 60)


def cmd_formula_check():
    """Validate alpha formula with example values."""
    print_header("ALPHA FORMULA VALIDATION")

    print("\nFormula: alpha = (min_eff / baseline) * retention_factor")
    print(f"Shannon floor (e): {SHANNON_FLOOR_ALPHA}")
    print(f"Ceiling target: {ALPHA_CEILING_TARGET}")

    test_cases = [
        (math.e, 1.0, 1.0, math.e, "Identity at baseline"),
        (2.7185, 1.0, 1.01, 2.745, "Standard case"),
        (math.e, 1.0, 1.10, ALPHA_CEILING_TARGET, "Ceiling case"),
    ]

    print("\nTest Cases:")
    print(
        f"  {'min_eff':>10} | {'baseline':>10} | {'retention':>10} | {'expected':>10} | {'computed':>10} | {'status':>8}"
    )
    print(
        f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}"
    )

    all_pass = True
    for min_eff, baseline, retention, expected, description in test_cases:
        try:
            result = alpha_calc(min_eff, baseline, retention, validate=False)
            computed = result["computed_alpha"]
            passed = abs(computed - expected) < 0.01
            all_pass = all_pass and passed
            print(
                f"  {min_eff:>10.4f} | {baseline:>10.4f} | {retention:>10.4f} | {expected:>10.4f} | {computed:>10.4f} | {'PASS' if passed else 'FAIL':>8}"
            )
        except Exception:
            print(
                f"  {min_eff:>10.4f} | {baseline:>10.4f} | {retention:>10.4f} | {expected:>10.4f} | {'ERROR':>10} | {'FAIL':>8}"
            )
            all_pass = False

    print(f"\nOVERALL: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")

    print("\n[formula_check complete]")
    print("=" * 60)


def cmd_isolate_layers(blackout_days: int, simulate: bool):
    """Output isolated contribution from each layer.

    Args:
        blackout_days: Blackout duration in days
        simulate: Whether to output simulation receipt
    """
    print_header(f"LAYER ISOLATION ANALYSIS ({blackout_days} days)")

    result = get_layer_contributions(blackout_days, 0.3)

    print("\nGNN Layer:")
    gnn = result["gnn_layer"]
    print(f"  Retention factor: {gnn['retention_factor']}")
    print(f"  Contribution: {gnn['contribution_pct']}%")
    print(f"  Alpha with GNN only: {gnn['alpha_with_gnn_only']}")
    print(f"  Expected range: {gnn['range_expected']}")

    print("\nPruning Layer:")
    prune = result["prune_layer"]
    print(f"  Retention factor: {prune['retention_factor']}")
    print(f"  Contribution: {prune['contribution_pct']}%")
    print(f"  Alpha with prune only: {prune['alpha_with_prune_only']}")
    print(f"  Expected range: {prune['range_expected']}")

    print("\nCompound:")
    compound = result["compound"]
    print(f"  Compound retention: {compound['compound_retention']}")
    print(f"  Full alpha: {compound['full_alpha']}")
    print(f"  Total uplift from floor: {compound['total_uplift_from_floor']}")

    print("\nCeiling Analysis:")
    ceiling = result["ceiling_analysis"]
    print(f"  Gap to ceiling: {ceiling['gap_pct']:.1f}%")
    print(f"  Path: {ceiling['path_to_ceiling']}")

    if simulate:
        print("\n[layer_contributions receipt emitted above]")

    print("=" * 60)
