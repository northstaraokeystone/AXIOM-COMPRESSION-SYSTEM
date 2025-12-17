#!/usr/bin/env python3
"""AXIOM-CORE CLI - The Sovereignty Calculator

One equation. One curve. One finding.

Usage:
    python cli.py baseline      # Run baseline test
    python cli.py bootstrap     # Run bootstrap analysis
    python cli.py curve         # Generate sovereignty curve
    python cli.py full          # Run full integration test
    python cli.py --simulate_timeline --c_base 50 --p_factor 1.8 --tau 0     # Earth timeline
    python cli.py --simulate_timeline --c_base 50 --p_factor 1.8 --tau 1200  # Mars timeline

    # Partition resilience testing (Dec 2025)
    python cli.py --partition 0.4 --nodes 5 --simulate    # Single partition test
    python cli.py --stress_quorum                          # Full 1000-iteration stress test
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sovereignty import compute_sovereignty, find_threshold, SovereigntyConfig
from src.validate import test_null_hypothesis, test_baseline, bootstrap_threshold, generate_falsifiable_prediction
from src.plot_curve import generate_curve_data, find_knee, plot_sovereignty_curve, format_finding
from src.entropy_shannon import (
    HUMAN_DECISION_RATE_BPS,
    STARLINK_MARS_BANDWIDTH_MIN_MBPS,
    STARLINK_MARS_BANDWIDTH_MAX_MBPS,
)
from src.timeline import sovereignty_timeline, C_BASE_DEFAULT, P_FACTOR_DEFAULT, ALPHA_DEFAULT
from src.latency import tau_penalty, effective_alpha
from src.partition import (
    partition_sim,
    stress_sweep,
    load_partition_spec,
    NODE_BASELINE,
    QUORUM_THRESHOLD,
    PARTITION_MAX_TEST_PCT,
    BASE_ALPHA
)
from src.ledger import LEDGER_ALPHA_BOOST_VALIDATED
from src.reasoning import sovereignty_projection_with_partition, validate_resilience_slo


def cmd_baseline():
    """Run baseline sovereignty test."""
    print("=" * 60)
    print("BASELINE SOVEREIGNTY TEST")
    print("=" * 60)

    result = test_baseline()
    print(f"\nConfiguration:")
    print(f"  Bandwidth: {result['bandwidth_mbps']} Mbps")
    print(f"  Delay: {result['delay_s']} seconds")
    print(f"  Compute: {result['compute_flops']} FLOPS (no AI assist)")

    print(f"\nRESULT: Threshold = {result['threshold']} crew")
    print("=" * 60)


def cmd_bootstrap():
    """Run bootstrap statistical analysis."""
    print("=" * 60)
    print("BOOTSTRAP STATISTICAL ANALYSIS")
    print("=" * 60)

    print("\nRunning 100 bootstrap iterations...")
    result = bootstrap_threshold(100, 42)

    print(f"\nRESULTS:")
    print(f"  Mean threshold: {result['mean']:.1f} crew")
    print(f"  Std deviation:  {result['std']:.1f} crew")
    print(f"  Range: [{result['min']}, {result['max']}] crew")
    print(f"  P-value: {result['p_value']:.6f}")

    print("\n" + generate_falsifiable_prediction(result))
    print("=" * 60)


def cmd_curve():
    """Generate sovereignty curve."""
    print("=" * 60)
    print("SOVEREIGNTY CURVE GENERATION")
    print("=" * 60)

    # Parameters
    bandwidth = 4.0  # Expected Mbps
    delay = 480.0    # 8 min average

    print(f"\nConfiguration:")
    print(f"  Bandwidth: {bandwidth} Mbps")
    print(f"  Delay: {delay} seconds ({delay/60:.0f} minutes)")

    # Generate curve
    data = generate_curve_data((10, 100), bandwidth, delay)
    knee = find_knee(data)

    # Bootstrap for uncertainty
    bootstrap = bootstrap_threshold(100, 42)
    uncertainty = bootstrap["std"]

    print(f"\nTHRESHOLD: {knee} +/- {uncertainty:.0f} crew")

    # Plot
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "sovereignty_curve.png")
    plot_sovereignty_curve(data, knee, output_path, uncertainty=uncertainty)
    print(f"\nCurve saved to: {output_path}")

    # The finding
    print("\n" + "-" * 60)
    print("THE FINDING:")
    print("-" * 60)
    print(format_finding(knee, bandwidth, delay / 60, uncertainty))
    print("=" * 60)


def cmd_full():
    """Run full integration test."""
    print("=" * 60)
    print("AXIOM-CORE v1 INTEGRATION TEST")
    print("=" * 60)

    # 1. Null hypothesis
    print("\n[1] NULL HYPOTHESIS TEST")
    null_result = test_null_hypothesis()
    status = "PASS" if null_result["passed"] else "FAIL"
    print(f"    Status: {status}")
    print(f"    Threshold with infinite bandwidth: {null_result['threshold']}")

    # 2. Baseline
    print("\n[2] BASELINE TEST")
    baseline = test_baseline()
    print(f"    Threshold (no tech assist): {baseline['threshold']} crew")

    # 3. Bootstrap
    print("\n[3] BOOTSTRAP ANALYSIS")
    bootstrap = bootstrap_threshold(100, 42)
    print(f"    Mean: {bootstrap['mean']:.1f} +/- {bootstrap['std']:.1f} crew")
    print(f"    P-value: {bootstrap['p_value']:.6f}")

    # 4. Curve
    print("\n[4] SOVEREIGNTY CURVE")
    data = generate_curve_data((10, 100), 4.0, 480)
    knee = find_knee(data)
    print(f"    Knee at: {knee} crew")

    # Generate plot
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "sovereignty_curve.png")
    plot_sovereignty_curve(data, knee, output_path, uncertainty=bootstrap["std"])
    print(f"    Saved: {output_path}")

    # 5. The finding
    print("\n" + "=" * 60)
    print("THE FINDING:")
    print("=" * 60)
    print(format_finding(
        knee=int(bootstrap["mean"]),
        bandwidth=4.0,
        delay=8.0,  # minutes
        uncertainty=bootstrap["std"]
    ))

    # 6. Falsifiable prediction
    print("\n" + "-" * 60)
    print(generate_falsifiable_prediction(bootstrap))

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


def cmd_partition(loss_pct: float, nodes: int, simulate: bool):
    """Run single partition simulation.

    Args:
        loss_pct: Partition loss percentage (0-1)
        nodes: Node count for simulation
        simulate: Whether to output simulation receipt
    """
    print("=" * 60)
    print("PARTITION RESILIENCE TEST")
    print("=" * 60)

    print(f"\nConfiguration:")
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

        print(f"\nRESULTS:")
        print(f"  Nodes surviving: {result['nodes_surviving']}")
        print(f"  Quorum status: {'INTACT' if result['quorum_status'] else 'FAILED'}")
        print(f"  Effective α drop: {result['eff_alpha_drop']:.4f}")
        print(f"  Effective α: {result['eff_alpha']:.4f}")

        # Validate SLOs
        print(f"\nSLO VALIDATION:")
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
    print("=" * 60)
    print("QUORUM STRESS TEST (1000 iterations)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Nodes baseline: {NODE_BASELINE}")
    print(f"  Quorum threshold: {QUORUM_THRESHOLD}")
    print(f"  Loss range: 0-{PARTITION_MAX_TEST_PCT * 100:.0f}%")
    print(f"  Iterations: 1000")
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

    print(f"\nRESULTS:")
    print(f"  Success rate: {success_rate * 100:.1f}%")
    print(f"  Avg α drop: {avg_drop:.4f}")
    print(f"  Max α drop: {max_drop:.4f}")
    print(f"  Min effective α: {min_alpha:.4f}")

    print(f"\nSLO VALIDATION:")
    print(f"  100% quorum survival: {'PASS' if success_rate == 1.0 else 'FAIL'}")
    print(f"  Avg drop < 0.05: {'PASS' if avg_drop < 0.05 else 'FAIL'}")
    print(f"  Min α >= 2.63: {'PASS' if min_alpha >= 2.63 else 'FAIL'}")

    print("\n[quorum_resilience receipt emitted above]")
    print("=" * 60)


def cmd_simulate_timeline(c_base: float, p_factor: float, tau: float):
    """Run sovereignty timeline simulation with Mars latency penalty.

    Args:
        c_base: Initial person-eq capacity
        p_factor: Propulsion growth factor per synod
        tau: Latency in seconds (0=Earth, 1200=Mars max)
    """
    print("=" * 60)
    print("SOVEREIGNTY TIMELINE SIMULATION")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  c_base (initial capacity): {c_base} person-eq")
    print(f"  p_factor (propulsion growth): {p_factor}x per synod")
    print(f"  tau (latency): {tau}s ({tau/60:.1f} min)")
    print(f"  alpha (base): {ALPHA_DEFAULT}")

    # Compute effective alpha
    eff_alpha = effective_alpha(ALPHA_DEFAULT, tau) if tau > 0 else ALPHA_DEFAULT
    print(f"  effective_alpha: {eff_alpha:.3f}")

    if tau > 0:
        penalty = tau_penalty(tau)
        print(f"  latency_penalty: {penalty:.2f} ({(1-penalty)*100:.0f}% drop)")

    # Run simulation
    result = sovereignty_timeline(c_base, p_factor, ALPHA_DEFAULT, tau)

    print(f"\nRESULTS:")
    print(f"  Cycles to 10³ person-eq: {result['cycles_to_10k_person_eq']}")
    print(f"  Cycles to 10⁶ person-eq: {result['cycles_to_1M_person_eq']}")

    if tau > 0:
        print(f"  Delay vs Earth: +{result['delay_vs_earth']} cycles")

    # Show trajectory summary
    traj = result['person_eq_trajectory']
    print(f"\nTrajectory (first 10 cycles):")
    for i, val in enumerate(traj[:10]):
        marker = ""
        if val >= 1000 and (i == 0 or traj[i-1] < 1000):
            marker = " <- 10³ milestone"
        if val >= 1000000 and (i == 0 or traj[i-1] < 1000000):
            marker = " <- 10⁶ milestone"
        print(f"    Cycle {i}: {val:.0f} person-eq{marker}")

    print("=" * 60)

    # The receipt is emitted by sovereignty_timeline, search for it
    print("\n[sovereignty_projection receipt emitted above]")


def main():
    # Check for flag-based invocation
    parser = argparse.ArgumentParser(description="AXIOM-CORE CLI - The Sovereignty Calculator")
    parser.add_argument('command', nargs='?', help='Command: baseline, bootstrap, curve, full')
    parser.add_argument('--c_base', type=float, default=C_BASE_DEFAULT,
                        help=f'Initial person-eq capacity (default: {C_BASE_DEFAULT})')
    parser.add_argument('--p_factor', type=float, default=P_FACTOR_DEFAULT,
                        help=f'Propulsion growth factor (default: {P_FACTOR_DEFAULT})')
    parser.add_argument('--tau', type=float, default=0,
                        help='Latency in seconds (0=Earth, 1200=Mars max)')
    parser.add_argument('--simulate_timeline', action='store_true',
                        help='Run sovereignty timeline simulation')

    # Partition resilience testing flags (Dec 2025)
    parser.add_argument('--partition', type=float, default=None,
                        help='Run single partition simulation at specified loss percentage (0-1)')
    parser.add_argument('--nodes', type=int, default=NODE_BASELINE,
                        help=f'Specify node count for simulation (default: {NODE_BASELINE})')
    parser.add_argument('--stress_quorum', action='store_true',
                        help='Run full stress sweep (1000 iterations, 0-40% loss)')
    parser.add_argument('--simulate', action='store_true',
                        help='Output simulation receipt to stdout')

    args = parser.parse_args()

    # Handle partition stress test
    if args.stress_quorum:
        cmd_stress_quorum()
        return

    # Handle single partition test
    if args.partition is not None:
        cmd_partition(args.partition, args.nodes, args.simulate)
        return

    # Handle timeline simulation
    if args.simulate_timeline:
        cmd_simulate_timeline(args.c_base, args.p_factor, args.tau)
        return

    # Handle positional command
    if args.command is None:
        print(__doc__)
        return

    cmd = args.command.lower()

    if cmd == "baseline":
        cmd_baseline()
    elif cmd == "bootstrap":
        cmd_bootstrap()
    elif cmd == "curve":
        cmd_curve()
    elif cmd == "full":
        cmd_full()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
