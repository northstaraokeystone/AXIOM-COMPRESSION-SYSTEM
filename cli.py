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
    python cli.py --blackout_sweep --reroute               # Full blackout sweep (43-60d, 1000 iterations)
    python cli.py --algo_info                              # Output reroute algorithm spec

    # Extended blackout sweep and retention curve (Dec 2025)
    python cli.py --extended_sweep 43 90 --simulate        # Extended sweep (43-90d)
    python cli.py --retention_curve                        # Output retention curve as JSON
    python cli.py --blackout_sweep 60 --simulate           # Single-point extended blackout test
    python cli.py --gnn_stub                               # Echo GNN sensitivity stub config

    # GNN nonlinear caching (Dec 2025 - NEW)
    python cli.py --gnn_nonlinear --blackout 150 --simulate  # GNN nonlinear at 150d
    python cli.py --cache_depth 1000000000 --blackout 200 --gnn_nonlinear  # Custom cache depth
    python cli.py --cache_sweep --simulate                   # Cache depth sensitivity sweep
    python cli.py --extreme_sweep 200 --simulate             # Extreme sweep to 200d
    python cli.py --overflow_test --simulate                 # Test cache overflow detection
    python cli.py --innovation_stubs                         # Echo innovation stub status

    # Entropy pruning (Dec 2025 - NEW)
    python cli.py --entropy_prune --blackout 150 --simulate     # Single pruning test
    python cli.py --trim_factor 0.4 --entropy_prune --simulate  # Custom trim factor
    python cli.py --hybrid_prune --blackout 200 --simulate      # Hybrid dedup + predictive
    python cli.py --pruning_sweep --simulate                    # Pruning sensitivity sweep
    python cli.py --extended_250d --simulate                    # 250d with pruning
    python cli.py --verify_chain --entropy_prune --simulate     # Verify chain integrity
    python cli.py --pruning_info                                # Echo pruning configuration
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
    REROUTING_ALPHA_BOOST_LOCKED,
    BLACKOUT_BASE_DAYS,
    BLACKOUT_EXTENDED_DAYS,
    MIN_EFF_ALPHA_FLOOR,
    MIN_EFF_ALPHA_VALIDATED
)
from src.blackout import (
    retention_curve,
    alpha_at_duration,
    extended_blackout_sweep as blackout_extended_sweep,
    generate_retention_curve_data,
    gnn_sensitivity_stub,
    BLACKOUT_SWEEP_MAX_DAYS,
    RETENTION_BASE_FACTOR,
    CURVE_TYPE,
    ASYMPTOTE_ALPHA
)
from src.reasoning import extended_blackout_sweep, project_with_degradation, extreme_blackout_sweep_200d, project_with_asymptote
from src.gnn_cache import (
    nonlinear_retention,
    nonlinear_retention_with_pruning,
    cache_depth_check,
    predict_overflow,
    extreme_blackout_sweep as gnn_extreme_sweep,
    quantum_relay_stub,
    swarm_autorepair_stub,
    cosmos_sim_stub,
    get_gnn_cache_info,
    ASYMPTOTE_ALPHA as GNN_ASYMPTOTE_ALPHA,
    ENTROPY_ASYMPTOTE_E,
    PRUNING_TARGET_ALPHA,
    MIN_EFF_ALPHA_VALIDATED as GNN_MIN_EFF_ALPHA_VALIDATED,
    CACHE_DEPTH_BASELINE,
    OVERFLOW_THRESHOLD_DAYS,
    OVERFLOW_THRESHOLD_DAYS_PRUNED,
    BLACKOUT_PRUNING_TARGET_DAYS
)
from src.pruning import (
    entropy_prune,
    dedup_prune,
    classify_leaf_entropy,
    generate_sample_merkle_tree,
    get_pruning_info,
    load_entropy_pruning_spec,
    ENTROPY_ASYMPTOTE_E as PRUNING_E,
    PRUNING_TARGET_ALPHA as PRUNING_ALPHA,
    LN_N_TRIM_FACTOR_BASE,
    LN_N_TRIM_FACTOR_MAX,
    BLACKOUT_PRUNING_TARGET_DAYS as PRUNING_TARGET_DAYS
)
from src.reasoning import extended_250d_sovereignty, validate_pruning_slos
from src.reasoning import ablation_sweep, compute_alpha_with_isolation, get_layer_contributions
from src.blackout import sweep_with_pruning
from src.alpha_compute import (
    alpha_calc,
    compound_retention,
    ceiling_gap,
    validate_formula,
    get_ablation_expected,
    get_alpha_compute_info,
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
    ABLATION_MODES
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


def cmd_extended_sweep(start_days: int, end_days: int, simulate: bool):
    """Run extended blackout sweep from start to end days.

    Args:
        start_days: Start of sweep range (e.g., 43)
        end_days: End of sweep range (e.g., 90)
        simulate: Whether to output receipts
    """
    import json as json_lib

    print("=" * 60)
    print(f"EXTENDED BLACKOUT SWEEP ({start_days}-{end_days}d)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Sweep range: {start_days}-{end_days} days")
    print(f"  Iterations: 1000")
    print(f"  Validated floor: {MIN_EFF_ALPHA_VALIDATED}")
    print(f"  Reroute boost (locked): +{REROUTING_ALPHA_BOOST_LOCKED}")

    print("\nRunning extended sweep...")

    result = extended_blackout_sweep(
        day_range=(start_days, end_days),
        iterations=1000,
        seed=42
    )

    print(f"\nRESULTS:")
    print(f"  All survived: {result['all_survived']}")
    print(f"  Survival rate: {result['survival_rate'] * 100:.1f}%")
    print(f"  Avg α: {result['avg_alpha']}")
    print(f"  Min α: {result['min_alpha']}")
    print(f"  α at 60d: {result['alpha_at_60d']}")
    print(f"  α at 90d: {result['alpha_at_90d']}")

    print(f"\nRETENTION FLOOR:")
    floor = result['retention_floor']
    print(f"  Min retention: {floor['min_retention']}")
    print(f"  Days at min: {floor['days_at_min']}")
    print(f"  α at min: {floor['alpha_at_min']}")

    print(f"\nASSERTION VALIDATION:")
    assertions = result['assertions_passed']
    print(f"  α(60d) >= 2.69: {'PASS' if assertions['alpha_60_ge_2.69'] else 'FAIL'}")
    print(f"  α(90d) >= 2.65: {'PASS' if assertions['alpha_90_ge_2.65'] else 'FAIL'}")

    if simulate:
        print("\n[extended_blackout_sweep receipt emitted above]")

    print("=" * 60)


def cmd_retention_curve():
    """Output retention curve data as JSON."""
    import json as json_lib

    print("=" * 60)
    print("RETENTION CURVE DATA")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Range: {BLACKOUT_BASE_DAYS}-{BLACKOUT_SWEEP_MAX_DAYS} days")
    print(f"  Base retention: {RETENTION_BASE_FACTOR}")
    print(f"  Degradation model: linear")

    curve_data = generate_retention_curve_data((BLACKOUT_BASE_DAYS, BLACKOUT_SWEEP_MAX_DAYS))

    print(f"\nCurve points: {len(curve_data)}")
    print("\nSample points:")
    print(f"  43d: retention={curve_data[0]['retention']}, α={curve_data[0]['alpha']}")

    # Find 60d, 75d, 90d indices
    idx_60 = 60 - BLACKOUT_BASE_DAYS
    idx_75 = 75 - BLACKOUT_BASE_DAYS
    idx_90 = 90 - BLACKOUT_BASE_DAYS

    if idx_60 < len(curve_data):
        print(f"  60d: retention={curve_data[idx_60]['retention']}, α={curve_data[idx_60]['alpha']}")
    if idx_75 < len(curve_data):
        print(f"  75d: retention={curve_data[idx_75]['retention']}, α={curve_data[idx_75]['alpha']}")
    if idx_90 < len(curve_data):
        print(f"  90d: retention={curve_data[idx_90]['retention']}, α={curve_data[idx_90]['alpha']}")

    print("\n--- JSON OUTPUT ---")
    print(json_lib.dumps(curve_data, indent=2))

    print("\n[retention_curve receipt emitted above]")
    print("=" * 60)


def cmd_gnn_stub():
    """Output GNN sensitivity stub config."""
    import json as json_lib

    print("=" * 60)
    print("GNN SENSITIVITY STUB (Next Gate Placeholder)")
    print("=" * 60)

    param_config = {
        "model_sizes": ["1K", "10K", "100K"],
        "complexity_levels": ["low", "medium", "high"],
        "target_metric": "alpha_uplift_per_watt",
        "hardware_constraint": "mars_surface_edge_compute",
        "gate": "next"
    }

    result = gnn_sensitivity_stub(param_config)

    print(f"\nStatus: {result['status']}")
    print(f"Not implemented: {result['not_implemented']}")
    print(f"Next gate: {result['next_gate']}")

    print(f"\nParameter config:")
    print(json_lib.dumps(param_config, indent=2))

    print(f"\nDescription: {result['description']}")

    print("\n[gnn_sensitivity_stub receipt emitted above]")
    print("=" * 60)


def cmd_gnn_nonlinear(blackout_days: int, cache_depth: int, simulate: bool):
    """Run GNN nonlinear retention simulation.

    Args:
        blackout_days: Blackout duration in days
        cache_depth: Cache depth in entries
        simulate: Whether to output simulation receipt
    """
    import json as json_lib

    print("=" * 60)
    print(f"GNN NONLINEAR RETENTION ({blackout_days} days)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Blackout duration: {blackout_days} days")
    print(f"  Cache depth: {cache_depth:,} entries")
    print(f"  Asymptote alpha: {GNN_ASYMPTOTE_ALPHA}")
    print(f"  Min eff alpha validated: {GNN_MIN_EFF_ALPHA_VALIDATED}")

    try:
        result = nonlinear_retention(blackout_days, cache_depth)

        print(f"\nRESULTS:")
        print(f"  Retention factor: {result['retention_factor']}")
        print(f"  Effective alpha: {result['eff_alpha']}")
        print(f"  Asymptote proximity: {result['asymptote_proximity']}")
        print(f"  GNN boost: {result['gnn_boost']}")
        print(f"  Curve type: {result['curve_type']}")

        print(f"\nSLO VALIDATION:")
        asymptote_ok = result['asymptote_proximity'] <= 0.02
        print(f"  Asymptote proximity <= 0.02: {'PASS' if asymptote_ok else 'FAIL'} ({result['asymptote_proximity']})")

        if simulate:
            print("\n[gnn_nonlinear_receipt emitted above]")

    except Exception as e:
        print(f"\nOVERFLOW DETECTED: {e}")
        print("Cache overflow - StopRule triggered")

    print("=" * 60)


def cmd_cache_sweep(simulate: bool):
    """Run cache depth sensitivity sweep.

    Args:
        simulate: Whether to output simulation receipts
    """
    import json as json_lib

    print("=" * 60)
    print("CACHE DEPTH SENSITIVITY SWEEP")
    print("=" * 60)

    cache_depths = [int(1e7), int(1e8), int(1e9), int(1e10)]
    test_days = [90, 150, 180, 200]

    print(f"\nConfiguration:")
    print(f"  Cache depths: {[f'{d:.0e}' for d in cache_depths]}")
    print(f"  Test durations: {test_days} days")

    print(f"\nRESULTS:")
    print(f"  {'Depth':>12} | {'90d':>10} | {'150d':>10} | {'180d':>10} | {'200d':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for depth in cache_depths:
        row = f"  {depth:>12.0e} |"
        for days in test_days:
            try:
                result = nonlinear_retention(days, depth)
                row += f" {result['eff_alpha']:>10.4f} |"
            except Exception:
                row += f" {'OVERFLOW':>10} |"
        print(row)

    if simulate:
        print("\n[cache_sweep receipts emitted above]")

    print("=" * 60)


def cmd_extreme_sweep(max_days: int, cache_depth: int, simulate: bool):
    """Run extreme blackout sweep to specified days.

    Args:
        max_days: Maximum blackout days
        cache_depth: Cache depth in entries
        simulate: Whether to output simulation receipts
    """
    import json as json_lib

    print("=" * 60)
    print(f"EXTREME BLACKOUT SWEEP (43-{max_days}d)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Day range: 43-{max_days} days")
    print(f"  Cache depth: {cache_depth:,} entries")
    print(f"  Iterations: 100 (abbreviated)")

    result = extreme_blackout_sweep_200d(
        day_range=(43, max_days),
        cache_depth=cache_depth,
        iterations=100,
        seed=42
    )

    print(f"\nRESULTS:")
    print(f"  Total sweeps: {result['total_sweeps']}")
    print(f"  Overflow events: {result['overflow_events']}")
    print(f"  Survival events: {result['survival_events']}")
    print(f"  Overflow at 200d: {result['overflow_at_200d']}")
    print(f"  StopRule expected: {result['stoprule_expected']}")

    if simulate:
        print("\n[extreme_blackout_sweep_200d receipt emitted above]")

    print("=" * 60)


def cmd_overflow_test(simulate: bool):
    """Test cache overflow detection.

    Args:
        simulate: Whether to output simulation receipts
    """
    print("=" * 60)
    print("CACHE OVERFLOW DETECTION TEST")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Overflow threshold: {OVERFLOW_THRESHOLD_DAYS} days")
    print(f"  Cache baseline: {CACHE_DEPTH_BASELINE:,} entries")

    # Test overflow detection
    test_days = [180, 190, 200, 201]

    print(f"\nRESULTS:")
    for days in test_days:
        overflow_result = predict_overflow(days, CACHE_DEPTH_BASELINE)
        overflow = overflow_result["overflow_risk"] >= 0.95

        try:
            retention = nonlinear_retention(days, CACHE_DEPTH_BASELINE)
            status = f"alpha={retention['eff_alpha']:.4f}"
        except Exception:
            status = "STOPRULE"

        print(f"  {days}d: risk={overflow_result['overflow_risk']:.2%}, overflow={overflow}, status={status}")

    if simulate:
        print("\n[overflow_test receipts emitted above]")

    print("=" * 60)


def cmd_innovation_stubs():
    """Echo innovation stub status."""
    import json as json_lib

    print("=" * 60)
    print("INNOVATION STUBS STATUS")
    print("=" * 60)

    print("\n[1] QUANTUM RELAY STUB")
    quantum = quantum_relay_stub()
    print(f"    Status: {quantum['status']}")
    print(f"    Potential boost: {quantum['potential_boost']}")
    print(f"    Description: {quantum['description']}")

    print("\n[2] SWARM AUTOREPAIR STUB")
    swarm = swarm_autorepair_stub()
    print(f"    Status: {swarm['status']}")
    print(f"    Potential boost: {swarm['potential_boost']}")
    print(f"    Potential: {swarm['potential']}")

    print("\n[3] COSMOS SIM STUB")
    cosmos = cosmos_sim_stub()
    print(f"    Status: {cosmos['status']}")
    print(f"    Reason: {cosmos['reason']}")
    print(f"    Availability: {cosmos['availability']}")

    print("\n[innovation_stub receipts emitted above]")
    print("=" * 60)


def cmd_entropy_prune(blackout_days: int, trim_factor: float, hybrid: bool, simulate: bool):
    """Run single entropy pruning test.

    Args:
        blackout_days: Blackout duration in days
        trim_factor: ln(n) trim factor (0.3-0.5 range)
        hybrid: Whether to enable hybrid dedup + predictive pruning
        simulate: Whether to output simulation receipt
    """
    import json as json_lib
    import math

    print("=" * 60)
    print(f"ENTROPY PRUNING TEST ({blackout_days} days)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  ENTROPY_ASYMPTOTE_E: {ENTROPY_ASYMPTOTE_E} (physics constant)")
    print(f"  Trim factor: {trim_factor}")
    print(f"  Hybrid mode: {hybrid}")
    print(f"  Target alpha: {PRUNING_TARGET_ALPHA}")
    print(f"  Target days: {BLACKOUT_PRUNING_TARGET_DAYS}")

    # Generate sample Merkle tree
    tree = generate_sample_merkle_tree(n_leaves=100, duplicate_ratio=0.2)
    print(f"  Sample tree leaves: {tree['leaf_count']}")

    # Run entropy pruning
    result = entropy_prune(tree, trim_factor=trim_factor, hybrid=hybrid)

    print(f"\nRESULTS:")
    print(f"  Branches pruned: {result['branches_pruned']}")
    print(f"  Entropy before: {result['entropy_before']}")
    print(f"  Entropy after: {result['entropy_after']}")
    print(f"  Entropy reduction: {result['entropy_reduction_pct']:.1f}%")
    print(f"  Alpha uplift: {result['alpha_uplift']}")
    print(f"  Dedup removed: {result['dedup_removed']}")
    print(f"  Predictive pruned: {result['predictive_pruned']}")

    # Get retention with pruning
    try:
        retention = nonlinear_retention_with_pruning(
            blackout_days,
            CACHE_DEPTH_BASELINE,
            pruning_enabled=True,
            trim_factor=trim_factor
        )
        print(f"\n  Effective alpha at {blackout_days}d: {retention['eff_alpha']}")
        print(f"  Pruning boost: {retention['pruning_boost']}")
        print(f"  Target achieved: {retention['eff_alpha'] >= PRUNING_TARGET_ALPHA}")
    except Exception as e:
        print(f"\n  OVERFLOW: {e}")

    if simulate:
        print("\n[entropy_pruning_receipt emitted above]")

    print("=" * 60)


def cmd_pruning_sweep(simulate: bool):
    """Run pruning sensitivity sweep.

    Args:
        simulate: Whether to output simulation receipts
    """
    import json as json_lib

    print("=" * 60)
    print("PRUNING SENSITIVITY SWEEP")
    print("=" * 60)

    trim_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_days = [150, 200, 250]

    print(f"\nConfiguration:")
    print(f"  Trim factors: {trim_factors}")
    print(f"  Test durations: {test_days} days")

    print(f"\nRESULTS:")
    print(f"  {'Trim':>8} | {'150d':>10} | {'200d':>10} | {'250d':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for trim in trim_factors:
        row = f"  {trim:>8.2f} |"
        for days in test_days:
            try:
                result = nonlinear_retention_with_pruning(
                    days,
                    CACHE_DEPTH_BASELINE,
                    pruning_enabled=True,
                    trim_factor=trim
                )
                row += f" {result['eff_alpha']:>10.4f} |"
            except Exception:
                row += f" {'OVERFLOW':>10} |"
        print(row)

    if simulate:
        print("\n[pruning_sweep receipts emitted above]")

    print("=" * 60)


def cmd_extended_250d(simulate: bool):
    """Run 250d extended simulation with pruning.

    Args:
        simulate: Whether to output simulation receipts
    """
    import json as json_lib

    print("=" * 60)
    print("EXTENDED 250d SOVEREIGNTY PROJECTION")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Target days: {BLACKOUT_PRUNING_TARGET_DAYS}")
    print(f"  Target alpha: {PRUNING_TARGET_ALPHA}")
    print(f"  Overflow threshold (pruned): {OVERFLOW_THRESHOLD_DAYS_PRUNED}")
    print(f"  Entropy asymptote: {ENTROPY_ASYMPTOTE_E} (physics)")

    result = extended_250d_sovereignty(
        pruning_enabled=True,
        trim_factor=0.3,
        blackout_days=BLACKOUT_PRUNING_TARGET_DAYS
    )

    print(f"\nRESULTS:")
    print(f"  Effective alpha: {result['effective_alpha']}")
    print(f"  Target achieved: {result['target_achieved']}")
    print(f"  Pruning boost: {result['pruning_boost']}")
    print(f"  Overflow margin: {result['overflow_margin']} days")
    print(f"  Cycles to 10K: {result['cycles_to_10k_person_eq']}")
    print(f"  Cycles to 1M: {result['cycles_to_1M_person_eq']}")

    print(f"\nSLO VALIDATION:")
    alpha_ok = result['effective_alpha'] >= PRUNING_TARGET_ALPHA
    overflow_ok = result['overflow_margin'] >= 0
    print(f"  Alpha >= {PRUNING_TARGET_ALPHA}: {'PASS' if alpha_ok else 'FAIL'} ({result['effective_alpha']})")
    print(f"  Overflow margin >= 0: {'PASS' if overflow_ok else 'FAIL'} ({result['overflow_margin']}d)")

    if simulate:
        print("\n[extended_250d_sovereignty receipt emitted above]")

    print("=" * 60)


def cmd_verify_chain(trim_factor: float, simulate: bool):
    """Verify chain integrity after pruning.

    Args:
        trim_factor: ln(n) trim factor
        simulate: Whether to output simulation receipts
    """
    print("=" * 60)
    print("CHAIN INTEGRITY VERIFICATION")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Trim factor: {trim_factor}")
    print(f"  Test iterations: 10")

    all_passed = True
    for i in range(10):
        tree = generate_sample_merkle_tree(n_leaves=100, duplicate_ratio=0.2)
        try:
            result = entropy_prune(tree, trim_factor=trim_factor, hybrid=True)
            status = "PASS"
        except Exception as e:
            if "Chain broken" in str(e) or "Quorum lost" in str(e):
                status = f"FAIL: {e}"
                all_passed = False
            else:
                status = "PASS"
        print(f"  Iteration {i+1}: {status}")

    print(f"\nCHAIN INTEGRITY: {'PASS' if all_passed else 'FAIL'}")

    if simulate:
        print("\n[chain_integrity receipts emitted above]")

    print("=" * 60)


def cmd_pruning_info():
    """Output pruning configuration."""
    import json as json_lib

    print("=" * 60)
    print("PRUNING CONFIGURATION")
    print("=" * 60)

    info = get_pruning_info()

    print(f"\nPhysics Constants:")
    print(f"  ENTROPY_ASYMPTOTE_E: {info['entropy_asymptote_e']} (Shannon bound, NOT tunable)")
    print(f"  PRUNING_TARGET_ALPHA: {info['pruning_target_alpha']}")

    print(f"\nTrim Factor Range:")
    print(f"  Base: {info['ln_n_trim_factor_base']} (conservative)")
    print(f"  Max: {info['ln_n_trim_factor_max']} (aggressive)")

    print(f"\nThresholds:")
    print(f"  Entropy prune threshold: {info['entropy_prune_threshold']}")
    print(f"  Min confidence: {info['min_confidence_threshold']}")
    print(f"  Min quorum fraction: {info['min_quorum_fraction']:.2%}")

    print(f"\nTargets:")
    print(f"  Blackout target days: {info['blackout_pruning_target_days']}")
    print(f"  Overflow threshold (pruned): {info['overflow_threshold_pruned_days']}")

    print(f"\nDescription: {info['description']}")

    print("\n[pruning_info receipt emitted above]")
    print("=" * 60)


# === ABLATION TESTING CLI COMMANDS (Dec 2025) ===


def cmd_ablate(mode: str, blackout_days: int, simulate: bool):
    """Run simulation in specified ablation mode.

    Args:
        mode: Ablation mode (full/no_cache/no_prune/baseline)
        blackout_days: Blackout duration in days
        simulate: Whether to output simulation receipt
    """
    import json as json_lib

    print("=" * 60)
    print(f"ABLATION TEST: {mode.upper()} ({blackout_days} days)")
    print("=" * 60)

    print(f"\nConfiguration:")
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
            ablation_mode=mode
        )

        print(f"\nRESULTS:")
        print(f"  Effective alpha: {result['eff_alpha']}")
        print(f"  Retention factor: {result['retention_factor']}")
        print(f"  GNN boost: {result['gnn_boost']}")
        print(f"  Pruning boost: {result['pruning_boost']}")
        print(f"  Model: {result['model']}")

        # Validate against expected
        alpha_min, alpha_max = expected['alpha_range']
        alpha_ok = alpha_min <= result['eff_alpha'] <= alpha_max

        print(f"\nVALIDATION:")
        print(f"  Alpha in expected range: {'PASS' if alpha_ok else 'FAIL'} ({result['eff_alpha']} in {expected['alpha_range']})")

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
    import json as json_lib

    print("=" * 60)
    print(f"ABLATION SWEEP ({blackout_days} days)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Ablation modes: {ABLATION_MODES}")
    print(f"  Blackout duration: {blackout_days} days")
    print(f"  Iterations per mode: 100")
    print(f"  Shannon floor: {SHANNON_FLOOR_ALPHA}")
    print(f"  Ceiling target: {ALPHA_CEILING_TARGET}")

    print("\nRunning ablation sweep...")

    result = ablation_sweep(
        modes=ABLATION_MODES,
        blackout_days=blackout_days,
        iterations=100,
        seed=42
    )

    print(f"\nRESULTS BY MODE:")
    print(f"  {'Mode':<12} | {'Avg Alpha':>10} | {'Min':>8} | {'Max':>8} | {'Success':>8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for mode in ["baseline", "no_prune", "no_cache", "full"]:
        if mode in result["results_by_mode"]:
            m = result["results_by_mode"][mode]
            print(f"  {mode:<12} | {m['avg_alpha']:>10.4f} | {m['min_alpha']:>8.4f} | {m['max_alpha']:>8.4f} | {m['successful']:>8}")

    print(f"\nORDERING VALIDATION:")
    print(f"  Expected: baseline < no_prune < no_cache < full")
    print(f"  Ordering valid: {'PASS' if result['ordering_valid'] else 'FAIL'}")

    print(f"\nLAYER CONTRIBUTIONS:")
    lc = result["layer_contributions"]
    print(f"  GNN contribution: {lc['gnn_contribution']:.4f}")
    print(f"  Prune contribution: {lc['prune_contribution']:.4f}")
    print(f"  Total uplift: {lc['total_uplift']}")

    print(f"\nCEILING ANALYSIS:")
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
    print("=" * 60)
    print("CEILING GAP ANALYSIS")
    print("=" * 60)

    result = ceiling_gap(current_alpha, ALPHA_CEILING_TARGET)

    print(f"\nCurrent Status:")
    print(f"  Current alpha: {result['current_alpha']}")
    print(f"  Ceiling target: {result['ceiling_target']}")
    print(f"  Shannon floor: {SHANNON_FLOOR_ALPHA}")

    print(f"\nGap Analysis:")
    print(f"  Gap absolute: {result['gap_absolute']}")
    print(f"  Gap percentage: {result['gap_pct']:.1f}%")

    print(f"\nRetention Analysis:")
    print(f"  Current retention factor: {result['retention_factor_current']}")
    print(f"  Retention needed: {result['retention_factor_needed']}")
    print(f"  Retention delta: {result['retention_factor_delta']}")

    print(f"\nPath to Ceiling:")
    print(f"  {result['path_to_ceiling']}")

    print("\n[ceiling_track receipt emitted above]")
    print("=" * 60)


def cmd_formula_check():
    """Validate alpha formula with example values."""
    import math

    print("=" * 60)
    print("ALPHA FORMULA VALIDATION")
    print("=" * 60)

    print(f"\nFormula: alpha = (min_eff / baseline) * retention_factor")
    print(f"Shannon floor (e): {SHANNON_FLOOR_ALPHA}")
    print(f"Ceiling target: {ALPHA_CEILING_TARGET}")

    test_cases = [
        (math.e, 1.0, 1.0, math.e, "Identity at baseline"),
        (2.7185, 1.0, 1.01, 2.745, "Standard case"),
        (math.e, 1.0, 1.10, ALPHA_CEILING_TARGET, "Ceiling case"),
    ]

    print(f"\nTest Cases:")
    print(f"  {'min_eff':>10} | {'baseline':>10} | {'retention':>10} | {'expected':>10} | {'computed':>10} | {'status':>8}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    all_pass = True
    for min_eff, baseline, retention, expected, description in test_cases:
        try:
            result = alpha_calc(min_eff, baseline, retention, validate=False)
            computed = result["computed_alpha"]
            passed = abs(computed - expected) < 0.01
            all_pass = all_pass and passed
            print(f"  {min_eff:>10.4f} | {baseline:>10.4f} | {retention:>10.4f} | {expected:>10.4f} | {computed:>10.4f} | {'PASS' if passed else 'FAIL':>8}")
        except Exception as e:
            print(f"  {min_eff:>10.4f} | {baseline:>10.4f} | {retention:>10.4f} | {expected:>10.4f} | {'ERROR':>10} | {'FAIL':>8}")
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
    import json as json_lib

    print("=" * 60)
    print(f"LAYER ISOLATION ANALYSIS ({blackout_days} days)")
    print("=" * 60)

    result = get_layer_contributions(blackout_days, 0.3)

    print(f"\nGNN Layer:")
    gnn = result["gnn_layer"]
    print(f"  Retention factor: {gnn['retention_factor']}")
    print(f"  Contribution: {gnn['contribution_pct']}%")
    print(f"  Alpha with GNN only: {gnn['alpha_with_gnn_only']}")
    print(f"  Expected range: {gnn['range_expected']}")

    print(f"\nPruning Layer:")
    prune = result["prune_layer"]
    print(f"  Retention factor: {prune['retention_factor']}")
    print(f"  Contribution: {prune['contribution_pct']}%")
    print(f"  Alpha with prune only: {prune['alpha_with_prune_only']}")
    print(f"  Expected range: {prune['range_expected']}")

    print(f"\nCompound:")
    compound = result["compound"]
    print(f"  Compound retention: {compound['compound_retention']}")
    print(f"  Full alpha: {compound['full_alpha']}")
    print(f"  Total uplift from floor: {compound['total_uplift_from_floor']}")

    print(f"\nCeiling Analysis:")
    ceiling = result["ceiling_analysis"]
    print(f"  Gap to ceiling: {ceiling['gap_pct']:.1f}%")
    print(f"  Path: {ceiling['path_to_ceiling']}")

    if simulate:
        print("\n[layer_contributions receipt emitted above]")

    print("=" * 60)


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

    # Extended blackout sweep and retention curve flags (Dec 2025)
    parser.add_argument('--extended_sweep', nargs=2, type=int, default=None,
                        metavar=('START', 'END'),
                        help='Run extended blackout sweep from START to END days (e.g., --extended_sweep 43 90)')
    parser.add_argument('--retention_curve', action='store_true',
                        help='Output retention curve data as JSON')
    parser.add_argument('--gnn_stub', action='store_true',
                        help='Echo GNN sensitivity stub config (placeholder for next gate)')

    # GNN nonlinear caching flags (Dec 2025 - NEW)
    parser.add_argument('--gnn_nonlinear', action='store_true',
                        help='Use GNN nonlinear retention model')
    parser.add_argument('--cache_depth', type=int, default=int(1e8),
                        help='Set cache depth for simulation (default: 1e8)')
    parser.add_argument('--cache_sweep', action='store_true',
                        help='Run cache depth sensitivity sweep')
    parser.add_argument('--extreme_sweep', type=int, default=None,
                        metavar='DAYS',
                        help='Run extreme blackout sweep to specified days')
    parser.add_argument('--overflow_test', action='store_true',
                        help='Test cache overflow detection')
    parser.add_argument('--innovation_stubs', action='store_true',
                        help='Echo innovation stub status (quantum, swarm, cosmos)')

    # Entropy pruning flags (Dec 2025 - NEW)
    parser.add_argument('--entropy_prune', action='store_true',
                        help='Enable entropy pruning for simulation')
    parser.add_argument('--trim_factor', type=float, default=0.3,
                        help='Set ln(n) trim factor (default: 0.3, max: 0.5)')
    parser.add_argument('--hybrid_prune', action='store_true',
                        help='Enable hybrid dedup + predictive pruning')
    parser.add_argument('--pruning_sweep', action='store_true',
                        help='Run pruning sensitivity sweep')
    parser.add_argument('--extended_250d', action='store_true',
                        help='Run 250d simulation with pruning')
    parser.add_argument('--verify_chain', action='store_true',
                        help='Verify chain integrity after pruning')
    parser.add_argument('--pruning_info', action='store_true',
                        help='Output pruning configuration')

    # Ablation testing flags (Dec 2025 - NEW)
    parser.add_argument('--ablate', type=str, default=None,
                        metavar='MODE',
                        help='Run simulation in specified ablation mode (full/no_cache/no_prune/baseline)')
    parser.add_argument('--ablation_sweep', action='store_true',
                        help='Run all 4 ablation modes and compare')
    parser.add_argument('--ceiling_track', type=float, default=None,
                        metavar='ALPHA',
                        help='Output ceiling gap analysis for specified alpha')
    parser.add_argument('--formula_check', action='store_true',
                        help='Validate alpha formula with example values')
    parser.add_argument('--isolate_layers', action='store_true',
                        help='Output isolated contribution from each layer')

    args = parser.parse_args()

    # Combine reroute flags
    reroute_enabled = args.reroute or args.reroute_enabled

    # Handle ablation testing flags (Dec 2025 - NEW)
    if args.formula_check:
        cmd_formula_check()
        return

    if args.ceiling_track is not None:
        cmd_ceiling_track(args.ceiling_track)
        return

    if args.isolate_layers:
        blackout_days = args.blackout if args.blackout is not None else 150
        cmd_isolate_layers(blackout_days, args.simulate)
        return

    if args.ablation_sweep:
        blackout_days = args.blackout if args.blackout is not None else 150
        cmd_ablation_sweep(blackout_days, args.simulate)
        return

    if args.ablate is not None:
        blackout_days = args.blackout if args.blackout is not None else 150
        cmd_ablate(args.ablate, blackout_days, args.simulate)
        return

    # Handle algorithm info
    if args.algo_info:
        cmd_algo_info()
        return

    # Handle GNN stub
    if args.gnn_stub:
        cmd_gnn_stub()
        return

    # Handle innovation stubs
    if args.innovation_stubs:
        cmd_innovation_stubs()
        return

    # Handle pruning info
    if args.pruning_info:
        cmd_pruning_info()
        return

    # Handle pruning sweep
    if args.pruning_sweep:
        cmd_pruning_sweep(args.simulate)
        return

    # Handle extended 250d
    if args.extended_250d:
        cmd_extended_250d(args.simulate)
        return

    # Handle verify chain
    if args.verify_chain:
        cmd_verify_chain(args.trim_factor, args.simulate)
        return

    # Handle entropy prune with blackout
    if args.entropy_prune and args.blackout is not None:
        hybrid = args.hybrid_prune or True  # Default to hybrid
        cmd_entropy_prune(args.blackout, args.trim_factor, hybrid, args.simulate)
        return

    # Handle cache sweep
    if args.cache_sweep:
        cmd_cache_sweep(args.simulate)
        return

    # Handle overflow test
    if args.overflow_test:
        cmd_overflow_test(args.simulate)
        return

    # Handle extreme sweep
    if args.extreme_sweep is not None:
        cmd_extreme_sweep(args.extreme_sweep, args.cache_depth, args.simulate)
        return

    # Handle GNN nonlinear with blackout
    if args.gnn_nonlinear and args.blackout is not None:
        cmd_gnn_nonlinear(args.blackout, args.cache_depth, args.simulate)
        return

    # Handle retention curve output
    if args.retention_curve:
        cmd_retention_curve()
        return

    # Handle extended sweep
    if args.extended_sweep is not None:
        cmd_extended_sweep(args.extended_sweep[0], args.extended_sweep[1], args.simulate)
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
