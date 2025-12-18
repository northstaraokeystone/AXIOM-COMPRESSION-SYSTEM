"""Blackout and GNN CLI commands for AXIOM-CORE.

Commands: blackout, blackout_sweep, simulate_timeline, extended_sweep,
          retention_curve, gnn_stub, gnn_nonlinear, cache_sweep,
          extreme_sweep, overflow_test, innovation_stubs, reroute, algo_info
"""

import json as json_lib

from src.partition import NODE_BASELINE
from src.timeline import sovereignty_timeline, ALPHA_DEFAULT
from src.latency import tau_penalty, effective_alpha
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
    MIN_EFF_ALPHA_FLOOR,
    MIN_EFF_ALPHA_VALIDATED,
    REROUTING_ALPHA_BOOST_LOCKED
)
from src.blackout import (
    generate_retention_curve_data,
    gnn_sensitivity_stub,
    BLACKOUT_SWEEP_MAX_DAYS,
    RETENTION_BASE_FACTOR,
)
from src.reasoning import extended_blackout_sweep, extreme_blackout_sweep_200d
from src.gnn_cache import (
    nonlinear_retention,
    predict_overflow,
    quantum_relay_stub,
    swarm_autorepair_stub,
    cosmos_sim_stub,
    ASYMPTOTE_ALPHA as GNN_ASYMPTOTE_ALPHA,
    MIN_EFF_ALPHA_VALIDATED as GNN_MIN_EFF_ALPHA_VALIDATED,
    CACHE_DEPTH_BASELINE,
    OVERFLOW_THRESHOLD_DAYS,
)

from cli.base import print_header


def cmd_reroute(simulate: bool):
    """Run single adaptive reroute test.

    Args:
        simulate: Whether to output simulation receipt
    """
    print_header("ADAPTIVE REROUTE TEST")

    print("\nConfiguration:")
    print(f"  Algorithm: {load_reroute_spec()['algo_type']}")
    print(f"  CGR Baseline: {load_reroute_spec()['cgr_baseline']}")
    print(f"  ML Model: {load_reroute_spec()['ml_model_type']}")
    print(f"  Alpha Boost: +{REROUTE_ALPHA_BOOST}")
    print(f"  Base Alpha Floor: {MIN_EFF_ALPHA_FLOOR}")

    # Test reroute boost
    boosted = apply_reroute_boost(MIN_EFF_ALPHA_FLOOR, reroute_active=True, blackout_days=0)

    print("\nRESULTS:")
    print(f"  Base eff_α: {MIN_EFF_ALPHA_FLOOR}")
    print(f"  Boosted eff_α: {boosted}")
    print(f"  Boost applied: +{REROUTE_ALPHA_BOOST}")

    print("\nSLO VALIDATION:")
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


def cmd_algo_info():
    """Output reroute algorithm specification."""
    print_header("ADAPTIVE REROUTE ALGORITHM SPECIFICATION")

    info = get_reroute_algo_info()

    print(f"\nAlgorithm: {info['algo_type']}")
    print(f"Description: {info['description']}")

    print("\nComponents:")
    print(f"  CGR Baseline: {info['cgr_baseline']}")
    print(f"  ML Model: {info['ml_model_type']}")

    print("\nParameters:")
    print(f"  Alpha Boost: +{info['alpha_boost']}")
    print(f"  Blackout Base: {info['blackout_base_days']} days")
    print(f"  Blackout Extended: {info['blackout_extended_days']} days")
    print(f"  Retention Factor: {info['retention_factor']}")
    print(f"  Min eff_α Floor: {info['min_eff_alpha_floor']}")

    print("\nHybrid Architecture:")
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


def cmd_blackout(blackout_days: int, reroute_enabled: bool, simulate: bool):
    """Run single blackout simulation.

    Args:
        blackout_days: Blackout duration in days
        reroute_enabled: Whether adaptive rerouting is active
        simulate: Whether to output simulation receipt
    """
    print_header(f"BLACKOUT SIMULATION ({blackout_days} days)")

    print("\nConfiguration:")
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

    print("\nRESULTS:")
    print(f"  Survival status: {'SURVIVED' if result['survival_status'] else 'FAILED'}")
    print(f"  Min α during: {result['min_alpha_during']}")
    print(f"  Max α drop: {result['max_alpha_drop']}")
    print(f"  Quorum failures: {result['quorum_failures']}")

    print("\nSLO VALIDATION:")
    survival_ok = result['survival_status']
    alpha_ok = result['min_alpha_during'] >= MIN_EFF_ALPHA_FLOOR * 0.9

    print(f"  Survival: {'PASS' if survival_ok else 'FAIL'}")
    print(f"  Min α acceptable: {'PASS' if alpha_ok else 'FAIL'} ({result['min_alpha_during']})")

    if blackout_days <= BLACKOUT_BASE_DAYS:
        print("  Within baseline (43d): PASS")
    elif blackout_days <= BLACKOUT_EXTENDED_DAYS and reroute_enabled:
        print("  Within extended (60d) with reroute: PASS")
    else:
        print("  Beyond tolerance: WARNING")

    if simulate:
        print("\n[blackout_sim_receipt emitted above]")

    print("=" * 60)


def cmd_blackout_sweep(reroute_enabled: bool):
    """Run full blackout sweep (1000 iterations, 43-60d range).

    Args:
        reroute_enabled: Whether adaptive rerouting is active
    """
    print_header("BLACKOUT STRESS SWEEP (1000 iterations)")

    print("\nConfiguration:")
    print(f"  Nodes baseline: {NODE_BASELINE}")
    print(f"  Blackout range: {BLACKOUT_BASE_DAYS}-{BLACKOUT_EXTENDED_DAYS} days")
    print("  Iterations: 1000")
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

    print("\nRESULTS:")
    print(f"  Survival rate: {result['survival_rate'] * 100:.1f}%")
    print(f"  Failures: {result['failures']}")
    print(f"  Avg min α: {result['avg_min_alpha']}")
    print(f"  Avg max drop: {result['avg_max_drop']}")
    print(f"  All survived: {result['all_survived']}")

    print("\nSLO VALIDATION:")
    survival_ok = result['survival_rate'] == 1.0
    drop_ok = result['avg_max_drop'] < 0.05

    print(f"  100% survival: {'PASS' if survival_ok else 'FAIL'} ({result['survival_rate']*100:.1f}%)")
    print(f"  Avg drop < 0.05: {'PASS' if drop_ok else 'FAIL'} ({result['avg_max_drop']})")

    if reroute_enabled:
        boosted = MIN_EFF_ALPHA_FLOOR + REROUTE_ALPHA_BOOST
        print(f"  eff_α with reroute >= 2.70: {'PASS' if boosted >= 2.70 else 'FAIL'} ({boosted})")

    print("\n[blackout_stress_sweep receipt emitted above]")
    print("=" * 60)


def cmd_simulate_timeline(c_base: float, p_factor: float, tau: float):
    """Run sovereignty timeline simulation with Mars latency penalty.

    Args:
        c_base: Initial person-eq capacity
        p_factor: Propulsion growth factor per synod
        tau: Latency in seconds (0=Earth, 1200=Mars max)
    """
    print_header("SOVEREIGNTY TIMELINE SIMULATION")

    print("\nConfiguration:")
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

    print("\nRESULTS:")
    print(f"  Cycles to 10³ person-eq: {result['cycles_to_10k_person_eq']}")
    print(f"  Cycles to 10⁶ person-eq: {result['cycles_to_1M_person_eq']}")

    if tau > 0:
        print(f"  Delay vs Earth: +{result['delay_vs_earth']} cycles")

    # Show trajectory summary
    traj = result['person_eq_trajectory']
    print("\nTrajectory (first 10 cycles):")
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
    print_header(f"EXTENDED BLACKOUT SWEEP ({start_days}-{end_days}d)")

    print("\nConfiguration:")
    print(f"  Sweep range: {start_days}-{end_days} days")
    print("  Iterations: 1000")
    print(f"  Validated floor: {MIN_EFF_ALPHA_VALIDATED}")
    print(f"  Reroute boost (locked): +{REROUTING_ALPHA_BOOST_LOCKED}")

    print("\nRunning extended sweep...")

    result = extended_blackout_sweep(
        day_range=(start_days, end_days),
        iterations=1000,
        seed=42
    )

    print("\nRESULTS:")
    print(f"  All survived: {result['all_survived']}")
    print(f"  Survival rate: {result['survival_rate'] * 100:.1f}%")
    print(f"  Avg α: {result['avg_alpha']}")
    print(f"  Min α: {result['min_alpha']}")
    print(f"  α at 60d: {result['alpha_at_60d']}")
    print(f"  α at 90d: {result['alpha_at_90d']}")

    print("\nRETENTION FLOOR:")
    floor = result['retention_floor']
    print(f"  Min retention: {floor['min_retention']}")
    print(f"  Days at min: {floor['days_at_min']}")
    print(f"  α at min: {floor['alpha_at_min']}")

    print("\nASSERTION VALIDATION:")
    assertions = result['assertions_passed']
    print(f"  α(60d) >= 2.69: {'PASS' if assertions['alpha_60_ge_2.69'] else 'FAIL'}")
    print(f"  α(90d) >= 2.65: {'PASS' if assertions['alpha_90_ge_2.65'] else 'FAIL'}")

    if simulate:
        print("\n[extended_blackout_sweep receipt emitted above]")

    print("=" * 60)


def cmd_retention_curve():
    """Output retention curve data as JSON."""
    print_header("RETENTION CURVE DATA")

    print("\nConfiguration:")
    print(f"  Range: {BLACKOUT_BASE_DAYS}-{BLACKOUT_SWEEP_MAX_DAYS} days")
    print(f"  Base retention: {RETENTION_BASE_FACTOR}")
    print("  Degradation model: linear")

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
    print_header("GNN SENSITIVITY STUB (Next Gate Placeholder)")

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

    print("\nParameter config:")
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
    print_header(f"GNN NONLINEAR RETENTION ({blackout_days} days)")

    print("\nConfiguration:")
    print(f"  Blackout duration: {blackout_days} days")
    print(f"  Cache depth: {cache_depth:,} entries")
    print(f"  Asymptote alpha: {GNN_ASYMPTOTE_ALPHA}")
    print(f"  Min eff alpha validated: {GNN_MIN_EFF_ALPHA_VALIDATED}")

    try:
        result = nonlinear_retention(blackout_days, cache_depth)

        print("\nRESULTS:")
        print(f"  Retention factor: {result['retention_factor']}")
        print(f"  Effective alpha: {result['eff_alpha']}")
        print(f"  Asymptote proximity: {result['asymptote_proximity']}")
        print(f"  GNN boost: {result['gnn_boost']}")
        print(f"  Curve type: {result['curve_type']}")

        print("\nSLO VALIDATION:")
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
    print_header("CACHE DEPTH SENSITIVITY SWEEP")

    cache_depths = [int(1e7), int(1e8), int(1e9), int(1e10)]
    test_days = [90, 150, 180, 200]

    print("\nConfiguration:")
    print(f"  Cache depths: {[f'{d:.0e}' for d in cache_depths]}")
    print(f"  Test durations: {test_days} days")

    print("\nRESULTS:")
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
    print_header(f"EXTREME BLACKOUT SWEEP (43-{max_days}d)")

    print("\nConfiguration:")
    print(f"  Day range: 43-{max_days} days")
    print(f"  Cache depth: {cache_depth:,} entries")
    print("  Iterations: 100 (abbreviated)")

    result = extreme_blackout_sweep_200d(
        day_range=(43, max_days),
        cache_depth=cache_depth,
        iterations=100,
        seed=42
    )

    print("\nRESULTS:")
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
    print_header("CACHE OVERFLOW DETECTION TEST")

    print("\nConfiguration:")
    print(f"  Overflow threshold: {OVERFLOW_THRESHOLD_DAYS} days")
    print(f"  Cache baseline: {CACHE_DEPTH_BASELINE:,} entries")

    # Test overflow detection
    test_days = [180, 190, 200, 201]

    print("\nRESULTS:")
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
    print_header("INNOVATION STUBS STATUS")

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
