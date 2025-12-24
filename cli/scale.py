"""Scale validation CLI commands for SpaceProof-CORE.

Commands: multi_scale_sweep, scalability_gate_test, scale_info
"""

from spaceproof.multi_scale_sweep import (
    run_scale_sweep,
    check_degradation,
    scalability_gate,
    run_multi_scale_validation,
    get_multi_scale_info,
    TREE_SCALES,
    ALPHA_BASELINE,
    DEGRADATION_TOLERANCE,
    SCALABILITY_GATE_THRESHOLD,
)
from spaceproof.reasoning import (
    get_31_push_readiness,
)
from spaceproof.fractal_layers import (
    get_fractal_layers_info,
    validate_scale_physics,
)

from cli.base import print_header, print_receipt_note


def cmd_multi_scale_sweep(sweep_type: str, simulate: bool):
    """Run multi-scale validation across tree sizes.

    Args:
        sweep_type: "all" for full sweep, "1e9" for just 10^9
        simulate: Whether to output simulation receipt
    """
    print_header(f"MULTI-SCALE VALIDATION ({sweep_type.upper()})")

    print("\nConfiguration:")
    print(f"  Alpha baseline (10^6): {ALPHA_BASELINE}")
    print(f"  Degradation tolerance: {DEGRADATION_TOLERANCE * 100:.1f}%")
    print(f"  Gate threshold: {SCALABILITY_GATE_THRESHOLD}")
    print(f"  Scales: {TREE_SCALES}")

    if sweep_type.lower() == "1e9":
        scales = [1_000_000_000]
        print("\nRunning 10^9 only validation...")
    else:
        scales = TREE_SCALES
        print("\nRunning full sweep [10^6, 10^8, 10^9]...")

    results = run_scale_sweep(scales)

    print("\nRESULTS:")
    for scale_key, data in results["results"].items():
        alpha = data["alpha"]
        instability = data["instability"]
        scale_factor = data.get("scale_factor", 1.0)
        status = "PASS" if alpha >= SCALABILITY_GATE_THRESHOLD else "FAIL"
        print(
            f"  {scale_key}: alpha={alpha:.4f}, instability={instability:.2f}, "
            f"scale_factor={scale_factor:.6f} [{status}]"
        )

    if sweep_type.lower() != "1e9":
        # Check degradation
        degradation = check_degradation(results)
        print("\nDEGRADATION:")
        print(f"  Baseline alpha: {degradation['baseline_alpha']:.4f}")
        print(f"  Target alpha (10^9): {degradation['target_alpha']:.4f}")
        print(f"  Degradation: {degradation['degradation_pct'] * 100:.2f}%")
        print(
            f"  Acceptable: {'YES' if degradation['degradation_acceptable'] else 'NO'}"
        )

    # Run gate check
    gate = scalability_gate(results)
    print("\nSCALABILITY GATE:")
    print(f"  Gate threshold: {gate['gate_threshold']}")
    print(f"  Alpha at 10^9: {gate['alpha_at_10e9']:.4f}")
    print(f"  Instability at 10^9: {gate['instability_at_10e9']:.2f}")
    print(f"  Gate passed: {'PASS' if gate['gate_passed'] else 'FAIL'}")
    print(f"  Ready for 3.1 push: {'YES' if gate['ready_for_31_push'] else 'NO'}")

    if simulate:
        print_receipt_note("multi_scale_sweep")
        print_receipt_note("scalability_gate")

    print("=" * 60)


def cmd_scalability_gate_test():
    """Check scalability gate status."""
    print_header("SCALABILITY GATE TEST")

    print("\nConfiguration:")
    print(f"  Gate threshold: {SCALABILITY_GATE_THRESHOLD}")
    print(f"  Degradation tolerance: {DEGRADATION_TOLERANCE * 100:.1f}%")

    print("\nRunning full validation...")

    # Run full validation
    results = run_multi_scale_validation()

    print("\nRESULTS:")
    print(f"  Alpha at 10^9: {results['alpha_at_10e9']:.4f}")
    print(f"  Instability at 10^9: {results['instability_at_10e9']:.2f}")
    print(f"  Degradation: {results['degradation_pct'] * 100:.2f}%")
    print(
        f"  Degradation acceptable: {'YES' if results['degradation_acceptable'] else 'NO'}"
    )

    print("\nGATE STATUS:")
    print(f"  Scalability valid: {'YES' if results['scalability_valid'] else 'NO'}")
    print(f"  Gate passed: {'PASS' if results['gate_passed'] else 'FAIL'}")
    print(f"  Ready for 3.1 push: {'YES' if results['ready_for_31_push'] else 'NO'}")

    # Get full readiness check
    print("\n3.1 PUSH READINESS:")
    readiness = get_31_push_readiness()
    for key, value in readiness["prerequisites"].items():
        status = "PASS" if value else "FAIL"
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {status}")

    print(f"\n  OVERALL: {'READY' if readiness['ready_for_31_push'] else 'NOT READY'}")

    print_receipt_note("scalability_gate")
    print_receipt_note("push_31_readiness")
    print("=" * 60)


def cmd_scale_info():
    """Show scale validation configuration."""
    print_header("SCALE VALIDATION CONFIGURATION")

    # Multi-scale info
    info = get_multi_scale_info()

    print("\nMulti-Scale Parameters:")
    print(f"  Tree scales: {info['tree_scales']}")
    print(f"  Alpha baseline: {info['alpha_baseline']}")
    print(f"  Degradation tolerance: {info['degradation_tolerance'] * 100:.1f}%")
    print(f"  Gate threshold: {info['scalability_gate_threshold']}")
    print(f"  Instability tolerance: {info['instability_tolerance']}")

    print("\nExpected Results:")
    for scale_key, data in info["expected_results"].items():
        print(
            f"  {scale_key}: alpha={data['alpha']:.3f}, instability={data['instability']:.2f}"
        )

    print("\nStoprules:")
    for stoprule in info["stoprules"]:
        print(f"  - {stoprule}")

    # Fractal layers info
    fractal_info = get_fractal_layers_info()

    print("\nFractal Layer Parameters:")
    print(f"  Base tree size: {fractal_info['base_tree_size']:,}")
    print(f"  Correlation decay factor: {fractal_info['correlation_decay_factor']}")
    print(f"  Base correlation: {fractal_info['fractal_base_correlation']}")

    print("\nScale Factors:")
    for scale_key, factor in fractal_info["scale_factors"].items():
        print(f"  {scale_key}: {factor:.6f}")

    print("\nExpected Alphas:")
    for scale_key, alpha in fractal_info["expected_alphas"].items():
        print(f"  {scale_key}: {alpha:.4f}")

    print(f"\nPhysics Formula: {fractal_info['physics_formula']}")

    # Validate scale physics
    print("\nScale Physics Validation:")
    physics = validate_scale_physics()
    for result in physics["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {result['tree_size']:.0e}: alpha={result['expected_alpha']:.4f} "
            f">= {result['min_required']:.3f} [{status}]"
        )

    print(f"\n  ALL PASSED: {'YES' if physics['all_passed'] else 'NO'}")

    print_receipt_note("multi_scale_info")
    print_receipt_note("fractal_layers_info")
    print("=" * 60)


def cmd_fractal_info():
    """Show fractal layer configuration."""
    print_header("FRACTAL LAYER CONFIGURATION")

    info = get_fractal_layers_info()

    print("\nConstants:")
    print(f"  Base tree size: {info['base_tree_size']:,}")
    print(f"  Correlation decay factor: {info['correlation_decay_factor']}")
    print(f"  Base correlation: {info['fractal_base_correlation']}")
    print(f"  Alpha contribution: {info['fractal_alpha_contribution']}")

    print("\nScale Factors:")
    for scale_key, factor in info["scale_factors"].items():
        print(f"  {scale_key}: {factor:.6f}")

    print("\nExpected Alphas:")
    for scale_key, alpha in info["expected_alphas"].items():
        print(f"  {scale_key}: {alpha:.4f}")

    print(f"\nFormula: {info['physics_formula']}")
    print(f"\nDescription: {info['description']}")

    print_receipt_note("fractal_layers_info")
    print("=" * 60)
