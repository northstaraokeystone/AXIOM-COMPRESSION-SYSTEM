"""Benchmark CLI commands for SpaceProof-CORE.

Commands: hybrid_10e12, release_gate, fractal_recursion, benchmark_info

BENCHMARK GATE:
    - 10^12 scale validation
    - eff_alpha floor: 3.05
    - instability: 0.00 sustained
    - scale decay: <= 0.02
"""

from spaceproof.hybrid_benchmark import (
    benchmark_10e12,
    check_release_gate_3_1,
    get_benchmark_info,
    ALPHA_10E12_FLOOR,
    ALPHA_10E12_TARGET,
    SCALE_DECAY_MAX,
)
from spaceproof.fractal_layers import (
    recursive_fractal,
    recursive_fractal_sweep,
    get_recursive_fractal_info,
    FRACTAL_RECURSION_MAX_DEPTH,
)

from cli.base import print_header, print_receipt_note


def cmd_hybrid_10e12(tree_size: int, base_alpha: float, simulate: bool):
    """Run 10^12 hybrid benchmark.

    Args:
        tree_size: Number of nodes (default: 10^12)
        base_alpha: Base alpha before contributions
        simulate: Whether to output simulation receipt
    """
    print_header("10^12 HYBRID BENCHMARK")

    print("\nConfiguration:")
    print(f"  Tree size: {tree_size:,}")
    print(f"  Base alpha: {base_alpha}")
    print(f"  Alpha floor: {ALPHA_10E12_FLOOR}")
    print(f"  Alpha target: {ALPHA_10E12_TARGET}")

    print("\nRunning benchmark...")

    result = benchmark_10e12(tree_size=tree_size, base_alpha=base_alpha)

    print("\nRESULTS:")
    print(f"  Effective alpha: {result['eff_alpha']}")
    print(f"  Instability: {result['instability']}")
    print(f"  Scale decay: {result['scale_decay']}")
    print(f"  Scale factor: {result['scale_factor']}")

    print("\nCONTRIBUTIONS:")
    print(f"  Quantum: +{result['quantum_contrib']}")
    print(f"  Fractal: +{result['fractal_contrib']}")
    print(f"  Hybrid total: +{result['hybrid_total']}")

    print("\nVALIDATION:")
    val = result["validation"]
    print(f"  alpha >= {ALPHA_10E12_FLOOR}: {'PASS' if val['alpha_ok'] else 'FAIL'}")
    print(f"  instability == 0.00: {'PASS' if val['instability_ok'] else 'FAIL'}")
    print(f"  decay <= {SCALE_DECAY_MAX}: {'PASS' if val['decay_ok'] else 'FAIL'}")

    print(f"\nGATE STATUS: {'PASS' if result['gate_pass'] else 'FAIL'}")

    if simulate:
        print_receipt_note("hybrid_10e12_benchmark")

    print("=" * 60)


def cmd_release_gate(simulate: bool):
    """Check release gate 3.1 status.

    Args:
        simulate: Whether to output simulation receipt
    """
    print_header("RELEASE GATE 3.1 CHECK")

    print("\nChecking gate conditions...")

    result = check_release_gate_3_1()

    print("\nBENCHMARK RESULTS:")
    bm = result["benchmark_result"]
    print(f"  eff_alpha: {bm['eff_alpha']}")
    print(f"  instability: {bm['instability']}")
    print(f"  scale_decay: {bm['scale_decay']}")

    if result["blockers"]:
        print("\nBLOCKERS:")
        for blocker in result["blockers"]:
            print(f"  - {blocker}")
    else:
        print("\nBLOCKERS: None")

    print(f"\nGATE STATUS: {'PASS' if result['gate_pass'] else 'FAIL'}")
    if result["gate_pass"]:
        print(f"VERSION: {result['version']} UNLOCKED")

    if simulate:
        print_receipt_note("release_gate_3_1")

    print("=" * 60)


def cmd_fractal_recursion(
    tree_size: int, base_alpha: float, depth: int, simulate: bool
):
    """Run fractal recursion for ceiling breach.

    Args:
        tree_size: Number of nodes in the tree
        base_alpha: Base alpha before recursion
        depth: Recursion depth (1-5)
        simulate: Whether to output simulation receipt
    """
    print_header("FRACTAL RECURSION CEILING BREACH")

    print("\nConfiguration:")
    print(f"  Tree size: {tree_size:,}")
    print(f"  Base alpha: {base_alpha}")
    print(f"  Recursion depth: {depth}")
    print(f"  Max depth: {FRACTAL_RECURSION_MAX_DEPTH}")

    print("\nRunning recursive fractal...")

    result = recursive_fractal(tree_size, base_alpha, depth=depth)

    print("\nDEPTH CONTRIBUTIONS:")
    for contrib in result["depth_contributions"]:
        print(
            f"  Depth {contrib['depth']}: +{contrib['contribution']} (decay: {contrib['decay_factor']})"
        )

    print("\nRESULTS:")
    print(f"  Total uplift: {result['total_uplift']}")
    print(f"  Scale-adjusted uplift: {result['adjusted_uplift']}")
    print(f"  Final alpha: {result['final_alpha']}")
    print(f"  Ceiling breached (>3.0): {'YES' if result['ceiling_breached'] else 'NO'}")
    print(f"  Target 3.1 reached: {'YES' if result['target_3_1_reached'] else 'NO'}")

    if simulate:
        print_receipt_note("fractal_recursion")

    print("=" * 60)


def cmd_fractal_recursion_sweep(tree_size: int, base_alpha: float, simulate: bool):
    """Sweep through all recursion depths.

    Args:
        tree_size: Number of nodes in the tree
        base_alpha: Base alpha before recursion
        simulate: Whether to output simulation receipt
    """
    print_header("FRACTAL RECURSION DEPTH SWEEP")

    print("\nConfiguration:")
    print(f"  Tree size: {tree_size:,}")
    print(f"  Base alpha: {base_alpha}")
    print(f"  Max depth: {FRACTAL_RECURSION_MAX_DEPTH}")

    print("\nRunning sweep...")

    result = recursive_fractal_sweep(tree_size, base_alpha)

    print("\nSWEEP RESULTS:")
    print(f"  {'Depth':>6} | {'Alpha':>8} | {'Uplift':>8} | {'3.1 Target':>10}")
    print(f"  {'-' * 6} | {'-' * 8} | {'-' * 8} | {'-' * 10}")
    for r in result["sweep_results"]:
        target = "YES" if r["target_3_1"] else "NO"
        print(
            f"  {r['depth']:>6} | {r['final_alpha']:>8.4f} | {r['uplift']:>8.4f} | {target:>10}"
        )

    print("\nOPTIMAL:")
    print(f"  Depth: {result['optimal_depth']}")
    print(f"  Alpha: {result['optimal_alpha']}")
    print(f"  3.1 achievable: {'YES' if result['target_3_1_achievable'] else 'NO'}")

    if simulate:
        print_receipt_note("fractal_recursion_sweep")

    print("=" * 60)


def cmd_benchmark_info():
    """Show benchmark module configuration."""
    print_header("BENCHMARK CONFIGURATION")

    # Get benchmark info
    bm_info = get_benchmark_info()

    print("\n10^12 Benchmark Constants:")
    print(f"  Tree size: {bm_info['tree_10e12']:,}")
    print(f"  Alpha floor: {bm_info['alpha_10e12_floor']}")
    print(f"  Alpha target: {bm_info['alpha_10e12_target']}")
    print(f"  Scale decay max: {bm_info['scale_decay_max']}")
    print(f"  Instability max: {bm_info['instability_max']}")

    print("\nSLO Requirements:")
    for key, value in bm_info["slo"].items():
        print(f"  {key}: {value}")

    print("\nExpected Results:")
    for key, value in bm_info["expected_results"].items():
        print(f"  {key}: {value}")

    # Get recursive fractal info
    rf_info = get_recursive_fractal_info()

    print("\nRecursive Fractal Configuration:")
    print(f"  Max depth: {rf_info['max_depth']}")
    print(f"  Default depth: {rf_info['default_depth']}")
    print(f"  Decay per depth: {rf_info['decay_per_depth']}")
    print(f"  Base uplift: {rf_info['base_uplift']}")

    print("\nExpected Uplifts by Depth:")
    for key, value in rf_info["expected_uplifts"].items():
        print(f"  {key}: +{value}")

    print_receipt_note("benchmark_info")
    print_receipt_note("recursive_fractal_info")
    print("=" * 60)
