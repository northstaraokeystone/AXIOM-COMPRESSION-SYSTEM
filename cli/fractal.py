"""Fractal ceiling breach CLI commands for SpaceProof-CORE.

Commands: fractal_push, alpha_boost, fractal_info_hybrid
"""

from spaceproof.fractal_layers import (
    multi_scale_fractal,
    get_fractal_hybrid_spec,
    get_fractal_layers_info,
    FRACTAL_SCALES,
    FRACTAL_UPLIFT,
    FRACTAL_DIM_MIN,
    FRACTAL_DIM_MAX,
)
from spaceproof.quantum_rl_hybrid import (
    quantum_fractal_hybrid,
    QUANTUM_RETENTION_BOOST,
)

from cli.base import print_header, print_receipt_note


def cmd_fractal_push(tree_size: int, base_alpha: float, simulate: bool):
    """Run fractal ceiling breach.

    Args:
        tree_size: Number of nodes in the tree
        base_alpha: Base alpha before fractal contribution
        simulate: Whether to output simulation receipt
    """
    print_header("FRACTAL CEILING BREACH")

    print("\nConfiguration:")
    print(f"  Tree size: {tree_size:,}")
    print(f"  Base alpha: {base_alpha}")
    print(f"  Scales: {FRACTAL_SCALES}")
    print(f"  Target uplift: {FRACTAL_UPLIFT}")

    print("\nRunning multi-scale fractal analysis...")

    result = multi_scale_fractal(tree_size, base_alpha)

    print("\nRESULTS:")
    print(f"  Fractal dimension: {result['fractal_dimension']}")
    print(f"  Uplift achieved: {result['uplift_achieved']}")
    print(f"  Fractal alpha: {result['fractal_alpha']}")
    print(f"  Ceiling breached: {'YES' if result['ceiling_breached'] else 'NO'}")

    print("\nScale Entropies:")
    for scale_key, entropy in result["scale_entropies"].items():
        print(f"  {scale_key}: {entropy}")

    print(f"\nCross-scale correlation: {result['cross_scale_corr']}")

    print("\nSLO VALIDATION:")
    alpha_ok = result["fractal_alpha"] > 3.0
    print(
        f"  fractal_alpha > 3.0: {'PASS' if alpha_ok else 'FAIL'} ({result['fractal_alpha']})"
    )

    if simulate:
        print_receipt_note("fractal_layer")

    print("=" * 60)


def cmd_alpha_boost(mode: str, tree_size: int, base_alpha: float, simulate: bool):
    """Run specified alpha boost mode.

    Args:
        mode: Boost mode (off, quantum, fractal, hybrid)
        tree_size: Number of nodes in the tree
        base_alpha: Base alpha before boost
        simulate: Whether to output simulation receipt
    """
    print_header(f"ALPHA BOOST: {mode.upper()}")

    print("\nConfiguration:")
    print(f"  Mode: {mode}")
    print(f"  Tree size: {tree_size:,}")
    print(f"  Base alpha: {base_alpha}")

    if mode == "off":
        print("\nRESULTS:")
        print(f"  Final alpha: {base_alpha} (no boost)")
        print(f"  Ceiling breached: {'YES' if base_alpha > 3.0 else 'NO'}")
        print("=" * 60)
        return

    if mode == "quantum":
        boost = QUANTUM_RETENTION_BOOST
        final_alpha = base_alpha + boost
        print("\nQuantum Boost:")
        print(f"  Quantum contribution: +{boost}")
        print(f"  Final alpha: {final_alpha}")
        print(f"  Ceiling breached: {'YES' if final_alpha > 3.0 else 'NO'}")

    elif mode == "fractal":
        result = multi_scale_fractal(tree_size, base_alpha)
        print("\nFractal Boost:")
        print(f"  Fractal uplift: +{result['uplift_achieved']}")
        print(f"  Final alpha: {result['fractal_alpha']}")
        print(f"  Ceiling breached: {'YES' if result['ceiling_breached'] else 'NO'}")

        if simulate:
            print_receipt_note("fractal_layer")

    elif mode == "hybrid":
        # First get fractal result
        fractal_result = multi_scale_fractal(tree_size, base_alpha)

        # Then combine with quantum
        state = {"alpha": base_alpha}
        hybrid_result = quantum_fractal_hybrid(state, fractal_result)

        print("\nHybrid Boost (Quantum + Fractal):")
        print(f"  Quantum contribution: +{hybrid_result['quantum_contribution']}")
        print(f"  Fractal contribution: +{hybrid_result['fractal_contribution']}")
        print(f"  Total boost: +{hybrid_result['hybrid_total']}")
        print(f"  Final alpha: {hybrid_result['final_alpha']}")
        print(f"  Instability: {hybrid_result['instability']}")
        print(
            f"  Ceiling breached: {'YES' if hybrid_result['ceiling_breached'] else 'NO'}"
        )

        if simulate:
            print_receipt_note("fractal_layer")
            print_receipt_note("quantum_fractal_hybrid")

    else:
        print(f"\nERROR: Unknown mode '{mode}'")
        print("Valid modes: off, quantum, fractal, hybrid")

    print("=" * 60)


def cmd_fractal_info_hybrid():
    """Show fractal hybrid configuration."""
    print_header("FRACTAL HYBRID CONFIGURATION")

    # Get fractal layers info
    layers_info = get_fractal_layers_info()

    print("\nFractal Layer Constants:")
    print(f"  Scales: {FRACTAL_SCALES}")
    print(f"  Dimension range: [{FRACTAL_DIM_MIN}, {FRACTAL_DIM_MAX}]")
    print(f"  Base uplift: {FRACTAL_UPLIFT}")

    print("\nScale Factors:")
    for scale_key, factor in layers_info["scale_factors"].items():
        print(f"  {scale_key}: {factor:.6f}")

    # Load spec
    try:
        spec = get_fractal_hybrid_spec()

        print("\nHybrid Spec (from JSON):")
        print(f"  Fractal uplift target: {spec['fractal_uplift_target']}")
        print(f"  Ceiling break target: {spec['ceiling_break_target']}")
        print(f"  Quantum contribution: {spec['quantum_contribution']}")
        print(f"  Hybrid total: {spec['hybrid_total']}")

        print("\nExpected Results:")
        expected = spec.get("expected_results", {})
        print(f"  eff_alpha before: {expected.get('eff_alpha_before', 2.99)}")
        print(f"  eff_alpha after: {expected.get('eff_alpha_after', 3.07)}")
        print(f"  instability: {expected.get('instability', 0.00)}")

    except FileNotFoundError:
        print("\nWARNING: fractal_hybrid_spec.json not found")

    print_receipt_note("fractal_layers_info")
    print_receipt_note("fractal_hybrid_spec_load")
    print("=" * 60)
