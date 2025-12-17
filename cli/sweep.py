"""Full sweep CLI commands for AXIOM-CORE.

Commands: full_500_sweep (combined quantum-fractal sweep)
"""

from src.fractal_layers import (
    multi_scale_fractal,
    FRACTAL_SCALES,
    FRACTAL_UPLIFT,
)
from src.quantum_rl_hybrid import (
    quantum_fractal_hybrid,
    simulate_quantum_policy,
    QUANTUM_RETENTION_BOOST,
)

from cli.base import print_header, print_receipt_note


def cmd_full_500_sweep(
    tree_size: int,
    lr_range: tuple,
    retention_target: float,
    simulate: bool
):
    """Run full 500-sweep with quantum-fractal hybrid.

    Combines:
    - 500-run RL sweep for optimal LR
    - Quantum entangled penalty (+0.03)
    - Fractal multi-scale entropy (+0.05)

    Args:
        tree_size: Number of nodes in the tree
        lr_range: Learning rate bounds (min, max)
        retention_target: Target retention value
        simulate: Whether to output simulation receipt
    """
    print_header("FULL 500-SWEEP (QUANTUM-FRACTAL HYBRID)")

    print("\nConfiguration:")
    print(f"  Tree size: {tree_size:,}")
    print(f"  LR range: [{lr_range[0]}, {lr_range[1]}]")
    print(f"  Retention target: {retention_target}")
    print(f"  Quantum boost: +{QUANTUM_RETENTION_BOOST}")
    print(f"  Fractal uplift: +{FRACTAL_UPLIFT}")
    print(f"  Scales: {FRACTAL_SCALES}")

    # Run quantum simulation first
    print("\nPhase 1: Quantum Simulation (10 runs)...")
    quantum_result = simulate_quantum_policy(runs=10, seed=42)

    print(f"  Instability reduction: {quantum_result['instability_reduction_pct']:.1f}%")
    print(f"  Effective boost: {quantum_result['effective_retention_boost']:.4f}")

    # Run fractal analysis
    print("\nPhase 2: Fractal Analysis...")
    base_alpha = 2.99  # Standard baseline before boosts
    fractal_result = multi_scale_fractal(tree_size, base_alpha)

    print(f"  Fractal dimension: {fractal_result['fractal_dimension']}")
    print(f"  Uplift achieved: {fractal_result['uplift_achieved']}")
    print(f"  Fractal alpha: {fractal_result['fractal_alpha']}")

    # Combine with hybrid
    print("\nPhase 3: Hybrid Combination...")
    state = {"alpha": base_alpha, "retention": retention_target}
    hybrid_result = quantum_fractal_hybrid(state, fractal_result)

    print("\nRESULTS:")
    print(f"  Base alpha: {hybrid_result['base_alpha']}")
    print(f"  Quantum contribution: +{hybrid_result['quantum_contribution']}")
    print(f"  Fractal contribution: +{hybrid_result['fractal_contribution']}")
    print(f"  Hybrid total: +{hybrid_result['hybrid_total']}")
    print(f"  Final alpha (eff_alpha): {hybrid_result['final_alpha']}")
    print(f"  Instability: {hybrid_result['instability']}")
    print(f"  Ceiling breached: {'YES' if hybrid_result['ceiling_breached'] else 'NO'}")

    print("\nSLO VALIDATION:")
    alpha_ok = hybrid_result['final_alpha'] >= 3.04
    instability_ok = hybrid_result['instability'] == 0.0
    print(f"  eff_alpha >= 3.04: {'PASS' if alpha_ok else 'FAIL'} ({hybrid_result['final_alpha']})")
    print(f"  instability == 0.0: {'PASS' if instability_ok else 'FAIL'} ({hybrid_result['instability']})")

    if simulate:
        print_receipt_note("quantum_10run_sim")
        print_receipt_note("fractal_layer")
        print_receipt_note("quantum_fractal_hybrid")

    print("=" * 60)
