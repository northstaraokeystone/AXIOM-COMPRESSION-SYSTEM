"""Info and status CLI commands for SpaceProof-CORE.

Commands: hybrid_boost_info
"""

from spaceproof.fractal_layers import (
    get_fractal_hybrid_spec,
    FRACTAL_SCALES,
    FRACTAL_UPLIFT,
    FRACTAL_DIM_MIN,
    FRACTAL_DIM_MAX,
)
from spaceproof.quantum_rl_hybrid import (
    QUANTUM_RETENTION_BOOST,
    ENTANGLED_PENALTY_FACTOR,
)

from cli.base import print_header, print_receipt_note


def cmd_hybrid_boost_info():
    """Show hybrid boost configuration (quantum + fractal)."""
    print_header("HYBRID BOOST CONFIGURATION")

    print("\nQuantum Contribution:")
    print(f"  Retention boost: +{QUANTUM_RETENTION_BOOST}")
    print(f"  Entanglement factor: {ENTANGLED_PENALTY_FACTOR}")
    print("  Mechanism: Reduced instability penalty (-1.0 → -0.92)")

    print("\nFractal Contribution:")
    print(f"  Alpha uplift: +{FRACTAL_UPLIFT}")
    print(f"  Scales: {FRACTAL_SCALES}")
    print(f"  Dimension range: [{FRACTAL_DIM_MIN}, {FRACTAL_DIM_MAX}]")
    print("  Mechanism: Multi-scale entropy contribution")

    print("\nCombined Hybrid:")
    total = QUANTUM_RETENTION_BOOST + FRACTAL_UPLIFT
    print(f"  Total boost: +{total}")
    print(f"  Expected eff_alpha: 2.99 + {total} = {2.99 + total:.2f}")
    print("  Instability: 0.00 (sustained)")

    # Load spec if available
    try:
        spec = get_fractal_hybrid_spec()
        print("\nSpec Validation:")
        print(f"  fractal_uplift_target: {spec['fractal_uplift_target']}")
        print(f"  quantum_contribution: {spec['quantum_contribution']}")
        print(f"  hybrid_total: {spec['hybrid_total']}")
        print(f"  ceiling_break_target: {spec['ceiling_break_target']}")
    except FileNotFoundError:
        print("\nWARNING: fractal_hybrid_spec.json not found")

    print("\nSequencing:")
    print("  1. 50-run pilot → narrow LR")
    print("  2. 10-run quantum sim → validate entanglement")
    print("  3. Fractal analysis → multi-scale entropy")
    print("  4. 500-run tuned sweep → hybrid ceiling breach")

    print_receipt_note("fractal_hybrid_spec_load")
    print("=" * 60)
