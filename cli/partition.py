"""Partition resilience CLI commands for AXIOM-CORE.

Commands: partition, stress_quorum
"""

from src.partition import (
    partition_sim,
    stress_sweep,
    NODE_BASELINE,
    QUORUM_THRESHOLD,
    PARTITION_MAX_TEST_PCT,
    BASE_ALPHA
)
from src.ledger import LEDGER_ALPHA_BOOST_VALIDATED

from cli.base import print_header


def cmd_partition(loss_pct: float, nodes: int, simulate: bool):
    """Run single partition simulation.

    Args:
        loss_pct: Partition loss percentage (0-1)
        nodes: Node count for simulation
        simulate: Whether to output simulation receipt
    """
    print_header("PARTITION RESILIENCE TEST")

    print("\nConfiguration:")
    print(f"  Nodes total: {nodes}")
    print(f"  Loss percentage: {loss_pct * 100:.0f}%")
    print(f"  Base alpha: {BASE_ALPHA}")
    print(f"  Ledger boost: +{LEDGER_ALPHA_BOOST_VALIDATED}")

    try:
        # Run partition simulation
        result = partition_sim(
            nodes_total=nodes,
            loss_pct=loss_pct,
            base_alpha=BASE_ALPHA,
            emit=simulate
        )

        print("\nRESULTS:")
        print(f"  Nodes surviving: {result['nodes_surviving']}")
        print(f"  Quorum status: {'INTACT' if result['quorum_status'] else 'FAILED'}")
        print(f"  Effective α drop: {result['eff_alpha_drop']:.4f}")
        print(f"  Effective α: {result['eff_alpha']:.4f}")

        # Validate SLOs
        print("\nSLO VALIDATION:")
        alpha_ok = result['eff_alpha'] >= 2.63
        drop_ok = result['eff_alpha_drop'] <= 0.05  # At boundary at 40% (exactly 0.05)
        quorum_ok = result['quorum_status']

        print(f"  eff_α >= 2.63: {'PASS' if alpha_ok else 'FAIL'} ({result['eff_alpha']:.4f})")
        print(f"  α drop <= 0.05: {'PASS' if drop_ok else 'FAIL'} ({result['eff_alpha_drop']:.4f})")
        print(f"  Quorum intact: {'PASS' if quorum_ok else 'FAIL'}")

        if simulate:
            print("\n[partition_stress receipt emitted above]")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Quorum failure - partition exceeded safe limits")

    print("=" * 60)


def cmd_stress_quorum():
    """Run full stress quorum test (1000 iterations, 0-40% loss)."""
    print_header("QUORUM STRESS TEST (1000 iterations)")

    print("\nConfiguration:")
    print(f"  Nodes baseline: {NODE_BASELINE}")
    print(f"  Quorum threshold: {QUORUM_THRESHOLD}")
    print(f"  Loss range: 0-{PARTITION_MAX_TEST_PCT * 100:.0f}%")
    print("  Iterations: 1000")
    print(f"  Base alpha: {BASE_ALPHA}")

    print("\nRunning stress sweep...")

    # Run stress sweep
    results = stress_sweep(
        nodes_total=NODE_BASELINE,
        loss_range=(0.0, PARTITION_MAX_TEST_PCT),
        n_iterations=1000,
        base_alpha=BASE_ALPHA,
        seed=42
    )

    # Compute stats
    quorum_successes = [r for r in results if r["quorum_status"]]
    success_rate = len(quorum_successes) / len(results)
    avg_drop = sum(r["eff_alpha_drop"] for r in quorum_successes) / len(quorum_successes)
    max_drop = max(r["eff_alpha_drop"] for r in quorum_successes)
    min_alpha = min(r["eff_alpha"] for r in quorum_successes)

    print("\nRESULTS:")
    print(f"  Success rate: {success_rate * 100:.1f}%")
    print(f"  Avg α drop: {avg_drop:.4f}")
    print(f"  Max α drop: {max_drop:.4f}")
    print(f"  Min effective α: {min_alpha:.4f}")

    print("\nSLO VALIDATION:")
    print(f"  100% quorum survival: {'PASS' if success_rate == 1.0 else 'FAIL'}")
    print(f"  Avg drop < 0.05: {'PASS' if avg_drop < 0.05 else 'FAIL'}")
    print(f"  Min α >= 2.63: {'PASS' if min_alpha >= 2.63 else 'FAIL'}")

    print("\n[quorum_resilience receipt emitted above]")
    print("=" * 60)
