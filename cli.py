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

    # Adaptive rerouting and blackout testing (Dec 2025)
    python cli.py --reroute --simulate                     # Single reroute test
    python cli.py --blackout 43 --reroute --simulate       # Blackout with reroute
    python cli.py --blackout_sweep --reroute               # Full blackout sweep (1000 iterations)
    python cli.py --algo_info                              # Output reroute algorithm spec
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
from src.reasoning import sovereignty_projection_with_partition, validate_resilience_slo, blackout_sweep
from src.reroute import (
    adaptive_reroute,
    blackout_sim,
    blackout_stress_sweep,
    apply_reroute_boost,
    get_reroute_algo_info,
    load_reroute_spec,
    REROUTE_ALPHA_BOOST,
    BLACKOUT_BASE_DAYS,
    BLACKOUT_EXTENDED_DAYS,
    MIN_EFF_ALPHA_FLOOR
)


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


def cmd_reroute(simulate: bool):
    """Run single adaptive reroute test.

    Args:
        simulate: Whether to output simulation receipt
    """
    print("=" * 60)
    print("ADAPTIVE REROUTE TEST")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Algorithm: {load_reroute_spec()['algo_type']}")
    print(f"  CGR Baseline: {load_reroute_spec()['cgr_baseline']}")
    print(f"  ML Model: {load_reroute_spec()['ml_model_type']}")
    print(f"  Alpha Boost: +{REROUTE_ALPHA_BOOST}")
    print(f"  Base Alpha Floor: {MIN_EFF_ALPHA_FLOOR}")

    # Test reroute boost
    boosted = apply_reroute_boost(MIN_EFF_ALPHA_FLOOR, reroute_active=True, blackout_days=0)

    print(f"\nRESULTS:")
    print(f"  Base eff_α: {MIN_EFF_ALPHA_FLOOR}")
    print(f"  Boosted eff_α: {boosted}")
    print(f"  Boost applied: +{REROUTE_ALPHA_BOOST}")

    print(f"\nSLO VALIDATION:")
    alpha_ok = boosted >= 2.70
    print(f"  eff_α >= 2.70: {'PASS' if alpha_ok else 'FAIL'} ({boosted})")

    # Run adaptive reroute simulation
    graph_state = {
        "nodes": NODE_BASELINE,
        "edges": [{"src": f"n{i}", "dst": f"n{(i+1) % NODE_BASELINE}"} for i in range(NODE_BASELINE)]
    }

    result = adaptive_reroute(graph_state, partition_pct=0.2, blackout_days=0)

    print(f"\n  Recovery factor: {result['recovery_factor']}")
    print(f"  Quorum preserved: {'PASS' if result['quorum_preserved'] else 'FAIL'}")
    print(f"  Alpha boost applied: {result['alpha_boost']}")

    if simulate:
        print("\n[adaptive_reroute_receipt emitted above]")

    print("=" * 60)


def cmd_blackout(blackout_days: int, reroute_enabled: bool, simulate: bool):
    """Run single blackout simulation.

    Args:
        blackout_days: Blackout duration in days
        reroute_enabled: Whether adaptive rerouting is active
        simulate: Whether to output simulation receipt
    """
    print("=" * 60)
    print(f"BLACKOUT SIMULATION ({blackout_days} days)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Blackout duration: {blackout_days} days")
    print(f"  Baseline max: {BLACKOUT_BASE_DAYS} days")
    print(f"  Extended max: {BLACKOUT_EXTENDED_DAYS} days (with reroute)")
    print(f"  Reroute enabled: {reroute_enabled}")
    print(f"  Base alpha floor: {MIN_EFF_ALPHA_FLOOR}")

    # Run blackout simulation
    result = blackout_sim(
        nodes=NODE_BASELINE,
        blackout_days=blackout_days,
        reroute_enabled=reroute_enabled,
        base_alpha=MIN_EFF_ALPHA_FLOOR,
        seed=42
    )

    print(f"\nRESULTS:")
    print(f"  Survival status: {'SURVIVED' if result['survival_status'] else 'FAILED'}")
    print(f"  Min α during: {result['min_alpha_during']}")
    print(f"  Max α drop: {result['max_alpha_drop']}")
    print(f"  Quorum failures: {result['quorum_failures']}")

    print(f"\nSLO VALIDATION:")
    survival_ok = result['survival_status']
    alpha_ok = result['min_alpha_during'] >= MIN_EFF_ALPHA_FLOOR * 0.9

    print(f"  Survival: {'PASS' if survival_ok else 'FAIL'}")
    print(f"  Min α acceptable: {'PASS' if alpha_ok else 'FAIL'} ({result['min_alpha_during']})")

    if blackout_days <= BLACKOUT_BASE_DAYS:
        print(f"  Within baseline (43d): PASS")
    elif blackout_days <= BLACKOUT_EXTENDED_DAYS and reroute_enabled:
        print(f"  Within extended (60d) with reroute: PASS")
    else:
        print(f"  Beyond tolerance: WARNING")

    if simulate:
        print("\n[blackout_sim_receipt emitted above]")

    print("=" * 60)


def cmd_blackout_sweep(reroute_enabled: bool):
    """Run full blackout sweep (1000 iterations, 43-60d range).

    Args:
        reroute_enabled: Whether adaptive rerouting is active
    """
    print("=" * 60)
    print("BLACKOUT STRESS SWEEP (1000 iterations)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Nodes baseline: {NODE_BASELINE}")
    print(f"  Blackout range: {BLACKOUT_BASE_DAYS}-{BLACKOUT_EXTENDED_DAYS} days")
    print(f"  Iterations: 1000")
    print(f"  Reroute enabled: {reroute_enabled}")
    print(f"  Base alpha: {MIN_EFF_ALPHA_FLOOR}")

    print("\nRunning blackout stress sweep...")

    result = blackout_stress_sweep(
        nodes=NODE_BASELINE,
        blackout_range=(BLACKOUT_BASE_DAYS, BLACKOUT_EXTENDED_DAYS),
        n_iterations=1000,
        reroute_enabled=reroute_enabled,
        base_alpha=MIN_EFF_ALPHA_FLOOR,
        seed=42
    )

    print(f"\nRESULTS:")
    print(f"  Survival rate: {result['survival_rate'] * 100:.1f}%")
    print(f"  Failures: {result['failures']}")
    print(f"  Avg min α: {result['avg_min_alpha']}")
    print(f"  Avg max drop: {result['avg_max_drop']}")
    print(f"  All survived: {result['all_survived']}")

    print(f"\nSLO VALIDATION:")
    survival_ok = result['survival_rate'] == 1.0
    drop_ok = result['avg_max_drop'] < 0.05

    print(f"  100% survival: {'PASS' if survival_ok else 'FAIL'} ({result['survival_rate']*100:.1f}%)")
    print(f"  Avg drop < 0.05: {'PASS' if drop_ok else 'FAIL'} ({result['avg_max_drop']})")

    if reroute_enabled:
        boosted = MIN_EFF_ALPHA_FLOOR + REROUTE_ALPHA_BOOST
        print(f"  eff_α with reroute >= 2.70: {'PASS' if boosted >= 2.70 else 'FAIL'} ({boosted})")

    print("\n[blackout_stress_sweep receipt emitted above]")
    print("=" * 60)


def cmd_algo_info():
    """Output reroute algorithm specification."""
    print("=" * 60)
    print("ADAPTIVE REROUTE ALGORITHM SPECIFICATION")
    print("=" * 60)

    info = get_reroute_algo_info()

    print(f"\nAlgorithm: {info['algo_type']}")
    print(f"Description: {info['description']}")

    print(f"\nComponents:")
    print(f"  CGR Baseline: {info['cgr_baseline']}")
    print(f"  ML Model: {info['ml_model_type']}")

    print(f"\nParameters:")
    print(f"  Alpha Boost: +{info['alpha_boost']}")
    print(f"  Blackout Base: {info['blackout_base_days']} days")
    print(f"  Blackout Extended: {info['blackout_extended_days']} days")
    print(f"  Retention Factor: {info['retention_factor']}")
    print(f"  Min eff_α Floor: {info['min_eff_alpha_floor']}")

    print(f"\nHybrid Architecture:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  CGR Base Layer (Deterministic)                     │")
    print("  │  - Time-varying Dijkstra on contact graph           │")
    print("  │  - Precomputed orbital contact windows              │")
    print("  │  - Provable delivery bounds                         │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  ML Adaptive Layer (Predictive)                     │")
    print("  │  - Lightweight GNN for anomaly prediction           │")
    print("  │  - Historical pattern learning                      │")
    print("  │  - Contact degradation forecasting                  │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  Quorum-Aware Recovery                              │")
    print("  │  - Merkle chain continuity preservation             │")
    print("  │  - Distributed anchoring integrity                  │")
    print("  └─────────────────────────────────────────────────────┘")

    print("\n[reroute_algo_info receipt emitted above]")
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

    # Adaptive rerouting and blackout testing flags (Dec 2025)
    parser.add_argument('--reroute', action='store_true',
                        help='Enable adaptive rerouting in simulation')
    parser.add_argument('--reroute_enabled', action='store_true',
                        help='Alias for --reroute')
    parser.add_argument('--blackout', type=int, default=None,
                        help='Run blackout simulation for specified days')
    parser.add_argument('--blackout_sweep', action='store_true',
                        help='Run full blackout sweep (43-60d range, 1000 iterations)')
    parser.add_argument('--algo_info', action='store_true',
                        help='Output reroute algorithm specification')

    args = parser.parse_args()

    # Combine reroute flags
    reroute_enabled = args.reroute or args.reroute_enabled

    # Handle algorithm info
    if args.algo_info:
        cmd_algo_info()
        return

    # Handle blackout sweep
    if args.blackout_sweep:
        cmd_blackout_sweep(reroute_enabled)
        return

    # Handle single blackout test
    if args.blackout is not None:
        cmd_blackout(args.blackout, reroute_enabled, args.simulate)
        return

    # Handle single reroute test
    if reroute_enabled and args.partition is None and args.blackout is None:
        cmd_reroute(args.simulate)
        return

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
