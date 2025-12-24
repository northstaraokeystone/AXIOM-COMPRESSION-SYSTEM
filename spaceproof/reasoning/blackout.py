"""reasoning/blackout.py - Blackout Sweep and Reroute Projections.

Functions for blackout simulations, reroute projections, and extended blackout sweeps.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import math

from ..core import emit_receipt, StopRule, dual_hash
from ..partition import partition_sim, NODE_BASELINE, BASE_ALPHA
from ..reroute import (
    blackout_sim,
    blackout_stress_sweep,
    apply_reroute_boost,
    REROUTE_ALPHA_BOOST,
    BLACKOUT_BASE_DAYS,
    BLACKOUT_EXTENDED_DAYS,
    MIN_EFF_ALPHA_FLOOR,
    MIN_EFF_ALPHA_VALIDATED,
)
from ..blackout import (
    retention_curve,
    generate_retention_curve_data,
    find_retention_floor,
    BLACKOUT_SWEEP_MAX_DAYS,
    RETENTION_BASE_FACTOR,
    CURVE_TYPE,
    ASYMPTOTE_ALPHA,
)
from ..gnn_cache import (
    extreme_blackout_sweep as gnn_extreme_sweep,
    nonlinear_retention as gnn_nonlinear_retention,
    predict_overflow,
    CACHE_DEPTH_BASELINE,
    OVERFLOW_THRESHOLD_DAYS,
    QUORUM_FAIL_DAYS,
    CACHE_BREAK_DAYS,
)
from .constants import (
    MIN_EFF_ALPHA_BOUND,
    CYCLES_THRESHOLD_EARLY,
    CYCLES_THRESHOLD_CITY,
)


def blackout_sweep(
    nodes: int = NODE_BASELINE,
    blackout_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_EXTENDED_DAYS),
    reroute_enabled: bool = True,
    iterations: int = 1000,
    base_alpha: float = MIN_EFF_ALPHA_BOUND,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run blackout simulation sweep across duration range.

    Runs blackout_sim across the specified range and collects
    resilience metrics for projection adjustment.

    Args:
        nodes: Total node count (default: 5)
        blackout_range: Tuple of (min_days, max_days) (default: 43-60)
        reroute_enabled: Whether adaptive rerouting is active
        iterations: Number of iterations (default: 1000)
        base_alpha: Baseline effective alpha (default: 2.63)
        seed: Random seed for reproducibility

    Returns:
        Dict with:
            - survival_rate: Fraction of successful blackouts
            - avg_min_alpha: Average minimum alpha during blackouts
            - avg_max_drop: Average maximum alpha drop
            - worst_case_alpha: Minimum alpha across all iterations
            - blackout_tolerance: Maximum days survived

    Assertions:
        - assert eff_alpha(reroute=True) >= 2.70
        - assert blackout_survival(days=60, reroute=True) == True

    Receipt: blackout_sweep
    """
    # Use reroute module's stress sweep
    result = blackout_stress_sweep(
        nodes=nodes,
        blackout_range=blackout_range,
        n_iterations=iterations,
        reroute_enabled=reroute_enabled,
        base_alpha=base_alpha,
        seed=seed,
    )

    # Compute additional metrics
    if reroute_enabled:
        # Verify reroute boost pushes alpha to 2.70+
        boosted_alpha = apply_reroute_boost(base_alpha, True, 0)
        assert boosted_alpha >= 2.70, (
            f"eff_alpha(reroute=True) = {boosted_alpha} < 2.70"
        )

        # Verify 60-day survival with reroute
        extended_sim = blackout_sim(nodes, 60, True, base_alpha, seed)
        assert extended_sim["survival_status"], (
            "blackout_survival(days=60, reroute=True) failed"
        )

    # Find blackout tolerance (max days survived)
    blackout_tolerance = (
        blackout_range[1] if result["all_survived"] else blackout_range[0]
    )

    report = {
        "nodes": nodes,
        "blackout_range": list(blackout_range),
        "iterations": iterations,
        "reroute_enabled": reroute_enabled,
        "base_alpha": base_alpha,
        "survival_rate": result["survival_rate"],
        "avg_min_alpha": result["avg_min_alpha"],
        "avg_max_drop": result["avg_max_drop"],
        "failures": result["failures"],
        "all_survived": result["all_survived"],
        "blackout_tolerance": blackout_tolerance,
        "reroute_boost": REROUTE_ALPHA_BOOST if reroute_enabled else 0.0,
        "boosted_alpha": base_alpha + REROUTE_ALPHA_BOOST
        if reroute_enabled
        else base_alpha,
    }

    emit_receipt("blackout_sweep", {"tenant_id": "spaceproof-reasoning", **report})

    return report


def project_with_reroute(
    base_projection: Dict[str, Any],
    reroute_results: Dict[str, Any],
    blackout_days: int = 0,
) -> Dict[str, Any]:
    """Adjust sovereignty timeline by reroute boost and blackout tolerance.

    Takes a base projection and applies reroute enhancements.

    Args:
        base_projection: Base sovereignty projection dict with:
            - cycles_to_10k_person_eq
            - cycles_to_1M_person_eq
            - effective_alpha
        reroute_results: Output from adaptive_reroute or blackout_sweep
        blackout_days: Current blackout duration (default: 0)

    Returns:
        Dict with adjusted projection including reroute metrics

    Receipt: reroute_projection
    """
    # Extract base values
    base_cycles_10k = base_projection.get("cycles_to_10k_person_eq", 4)
    base_cycles_1M = base_projection.get("cycles_to_1M_person_eq", 15)
    base_alpha = base_projection.get("effective_alpha", MIN_EFF_ALPHA_BOUND)

    # Get reroute impact
    alpha_boost = reroute_results.get("alpha_boost", REROUTE_ALPHA_BOOST)
    recovery_factor = reroute_results.get("recovery_factor", 0.8)
    reroute_results.get("survival_rate", 1.0)

    # Apply reroute boost
    boosted_alpha = apply_reroute_boost(base_alpha, True, blackout_days)

    # Calculate adjusted cycles
    # Higher alpha means fewer cycles needed
    # Formula: cycles_adjusted = cycles_base * (base_alpha / boosted_alpha)
    alpha_ratio = base_alpha / max(0.1, boosted_alpha)

    adjusted_cycles_10k = max(1, math.ceil(base_cycles_10k * alpha_ratio))
    adjusted_cycles_1M = max(1, math.ceil(base_cycles_1M * alpha_ratio))

    # Cycles saved by reroute
    cycles_saved_10k = base_cycles_10k - adjusted_cycles_10k
    cycles_saved_1M = base_cycles_1M - adjusted_cycles_1M

    # Blackout resilience metrics
    blackout_resilience = {
        "base_days": BLACKOUT_BASE_DAYS,
        "extended_days": BLACKOUT_EXTENDED_DAYS,
        "current_days": blackout_days,
        "survival_status": blackout_days <= BLACKOUT_EXTENDED_DAYS,
        "tolerance_factor": 1.0
        if blackout_days <= BLACKOUT_BASE_DAYS
        else (
            max(
                0.7, 1.0 - (blackout_days - BLACKOUT_BASE_DAYS) / BLACKOUT_EXTENDED_DAYS
            )
        ),
    }

    projection = {
        "base_cycles_10k": base_cycles_10k,
        "base_cycles_1M": base_cycles_1M,
        "base_alpha": base_alpha,
        "reroute_alpha_boost": alpha_boost,
        "boosted_alpha": boosted_alpha,
        "alpha_ratio": round(alpha_ratio, 4),
        "adjusted_cycles_10k": adjusted_cycles_10k,
        "adjusted_cycles_1M": adjusted_cycles_1M,
        "cycles_saved_10k": cycles_saved_10k,
        "cycles_saved_1M": cycles_saved_1M,
        "recovery_factor": recovery_factor,
        "blackout_resilience": blackout_resilience,
        "reroute_validated": boosted_alpha >= 2.70,
    }

    emit_receipt("reroute_projection", {"tenant_id": "spaceproof-reasoning", **projection})

    return projection


def sovereignty_timeline(
    c_base: float = 50.0,
    p_factor: float = 1.8,
    alpha: float = BASE_ALPHA,
    loss_pct: float = 0.0,
    reroute_enabled: bool = False,
    blackout_days: int = 0,
) -> Dict[str, Any]:
    """Compute sovereignty timeline with optional reroute and blackout.

    Full sovereignty projection including:
    1. Base timeline calculation
    2. Partition impact
    3. Reroute boost (if enabled)
    4. Blackout resilience (if applicable)

    Args:
        c_base: Initial person-eq capacity (default: 50.0)
        p_factor: Propulsion growth factor (default: 1.8)
        alpha: Base effective alpha (default: 2.68)
        loss_pct: Partition loss percentage (default: 0.0)
        reroute_enabled: Whether adaptive rerouting is active
        blackout_days: Current blackout duration in days

    Returns:
        Dict with complete sovereignty timeline

    Receipt: sovereignty_timeline
    """
    # Apply partition if specified
    if loss_pct > 0:
        partition_result = partition_sim(
            NODE_BASELINE, loss_pct, alpha, emit=False, reroute_enabled=reroute_enabled
        )
        effective_alpha = partition_result["eff_alpha"]
    else:
        effective_alpha = alpha

    # Apply reroute boost if enabled
    if reroute_enabled:
        effective_alpha = apply_reroute_boost(effective_alpha, True, blackout_days)

    # Compute base timeline
    alpha_ratio = effective_alpha / 1.69
    base_multiplier = 1.0 + (2.75 - 1.0) * alpha_ratio

    person_eq = c_base
    cycles_10k = None
    cycles_1M = None

    for cycle in range(1, 200):
        propulsion = 1.0 + (p_factor - 1.0) * (0.9 ** (cycle - 1))
        person_eq *= base_multiplier * propulsion

        if cycles_10k is None and person_eq >= CYCLES_THRESHOLD_EARLY:
            cycles_10k = cycle
        if cycles_1M is None and person_eq >= CYCLES_THRESHOLD_CITY:
            cycles_1M = cycle
            break

    cycles_10k = cycles_10k or 200
    cycles_1M = cycles_1M or 200

    result = {
        "c_base": c_base,
        "p_factor": p_factor,
        "base_alpha": alpha,
        "effective_alpha": round(effective_alpha, 4),
        "loss_pct": loss_pct,
        "reroute_enabled": reroute_enabled,
        "blackout_days": blackout_days,
        "cycles_to_10k_person_eq": cycles_10k,
        "cycles_to_1M_person_eq": cycles_1M,
        "reroute_boost_applied": reroute_enabled and REROUTE_ALPHA_BOOST > 0,
        "min_eff_alpha_floor": MIN_EFF_ALPHA_FLOOR,
    }

    emit_receipt("sovereignty_timeline", {"tenant_id": "spaceproof-reasoning", **result})

    return result


def extended_blackout_sweep(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_SWEEP_MAX_DAYS),
    iterations: int = 1000,
    seed: Optional[int] = None,
    cache_depth: int = CACHE_DEPTH_BASELINE,
) -> Dict[str, Any]:
    """Run extended blackout sweep across day range with GNN nonlinear retention curve.

    UPGRADED Dec 2025: Uses GNN nonlinear model with overflow detection.
    Runs extreme sweeps to 200d+, detects overflow.

    Assertions (UPDATED):
        - assert eff_alpha(blackout=150) >= 2.70 (asymptote approach)
        - assert eff_alpha(blackout=180) >= 2.50 or overflow_detected

    Args:
        day_range: Tuple of (min_days, max_days) (default: 43-200)
        iterations: Number of iterations (default: 1000)
        seed: Random seed for reproducibility
        cache_depth: Cache depth in entries (default: CACHE_DEPTH_BASELINE)

    Returns:
        Dict with sweep results, overflow detection, and retention curve receipt

    Receipt: extended_blackout_sweep
    """
    # Run GNN extreme sweep (handles overflow detection)
    sweep_results = gnn_extreme_sweep(day_range, cache_depth, iterations, seed)

    # Validate GNN nonlinear assertions
    overflow_detected = False
    alpha_150 = None
    alpha_180 = None

    try:
        result_150 = gnn_nonlinear_retention(150, cache_depth)
        alpha_150 = result_150["eff_alpha"]
        assert alpha_150 >= 2.70, f"eff_alpha(blackout=150) = {alpha_150} < 2.70"
    except StopRule:
        overflow_detected = True
        alpha_150 = None

    try:
        result_180 = gnn_nonlinear_retention(180, cache_depth)
        alpha_180 = result_180["eff_alpha"]
        # Either alpha >= 2.50 OR overflow is expected
        if alpha_180 < 2.50 and not overflow_detected:
            overflow_result = predict_overflow(180, cache_depth)
            if overflow_result["overflow_risk"] < 0.95:
                assert False, (
                    f"eff_alpha(blackout=180) = {alpha_180} < 2.50 without overflow"
                )
    except StopRule:
        overflow_detected = True
        alpha_180 = None

    # Generate retention curve data (may stop at overflow)
    curve_data = generate_retention_curve_data(day_range)

    # Find floor from sweep results
    surviving_results = [r for r in sweep_results if r.get("survival_status", False)]
    if surviving_results:
        floor_data = find_retention_floor(surviving_results)
    else:
        floor_data = {
            "min_retention": RETENTION_BASE_FACTOR,
            "days_at_min": day_range[0],
        }

    # Compute stats
    all_survived = all(r.get("survival_status", False) for r in sweep_results)
    survival_count = len([r for r in sweep_results if r.get("survival_status", False)])
    survival_rate = survival_count / max(1, len(sweep_results))

    alpha_values = [r["eff_alpha"] for r in sweep_results if "eff_alpha" in r]
    avg_alpha = sum(alpha_values) / max(1, len(alpha_values)) if alpha_values else 0.0
    min_alpha = min(alpha_values) if alpha_values else 0.0

    # Count overflow triggers
    overflow_count = len(
        [r for r in sweep_results if r.get("overflow_triggered", False)]
    )

    result = {
        "day_range": list(day_range),
        "iterations": iterations,
        "cache_depth": cache_depth,
        "all_survived": all_survived,
        "survival_rate": round(survival_rate, 4),
        "avg_alpha": round(avg_alpha, 4),
        "min_alpha": round(min_alpha, 4),
        "alpha_at_150d": alpha_150,
        "alpha_at_180d": alpha_180,
        "overflow_detected": overflow_detected,
        "overflow_count": overflow_count,
        "retention_floor": floor_data,
        "curve_points_count": len(curve_data),
        "curve_type": CURVE_TYPE,
        "asymptote_alpha": ASYMPTOTE_ALPHA,
        "assertions_passed": {
            "alpha_150_ge_2.70": alpha_150 is not None and alpha_150 >= 2.70,
            "alpha_180_ge_2.50_or_overflow": (
                alpha_180 is not None and alpha_180 >= 2.50
            )
            or overflow_detected,
        },
    }

    emit_receipt(
        "extended_blackout_sweep",
        {
            "tenant_id": "spaceproof-reasoning",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def extreme_blackout_sweep_200d(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, OVERFLOW_THRESHOLD_DAYS),
    cache_depth: int = CACHE_DEPTH_BASELINE,
    iterations: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run extreme blackout sweep to 200d+ with overflow detection.

    Specifically tests the 200d overflow threshold.

    Args:
        day_range: Tuple of (min_days, max_days) (default: 43-200)
        cache_depth: Cache depth in entries (default: CACHE_DEPTH_BASELINE)
        iterations: Number of iterations (default: 1000)
        seed: Random seed for reproducibility

    Returns:
        Dict with extreme sweep results and overflow stoprule receipts

    Receipt: extreme_blackout_sweep_200d
    """
    # Run GNN extreme sweep
    sweep_results = gnn_extreme_sweep(day_range, cache_depth, iterations, seed)

    # Count overflow events
    overflow_events = [r for r in sweep_results if r.get("overflow_triggered", False)]
    survival_events = [r for r in sweep_results if r.get("survival_status", False)]

    # Check overflow at boundary
    overflow_at_200d = False
    try:
        gnn_nonlinear_retention(200, cache_depth)
    except StopRule:
        overflow_at_200d = True

    result = {
        "day_range": list(day_range),
        "iterations": iterations,
        "cache_depth": cache_depth,
        "overflow_threshold_days": OVERFLOW_THRESHOLD_DAYS,
        "total_sweeps": len(sweep_results),
        "overflow_events": len(overflow_events),
        "survival_events": len(survival_events),
        "overflow_at_200d": overflow_at_200d,
        "stoprule_expected": day_range[1] >= CACHE_BREAK_DAYS,
        "quorum_fail_days": QUORUM_FAIL_DAYS,
    }

    emit_receipt(
        "extreme_blackout_sweep_200d",
        {
            "tenant_id": "spaceproof-reasoning",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def project_with_asymptote(
    base_projection: Dict[str, Any], gnn_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Adjust sovereignty timeline by asymptotic alpha ceiling.

    Takes a base projection and adjusts by GNN asymptotic alpha.

    Args:
        base_projection: Base sovereignty projection dict with:
            - cycles_to_10k_person_eq
            - cycles_to_1M_person_eq
            - effective_alpha
        gnn_results: Output from GNN nonlinear retention with:
            - eff_alpha
            - asymptote_proximity

    Returns:
        Dict with adjusted projection including asymptote metrics

    Receipt: asymptote_projection
    """
    # Extract base values
    base_cycles_10k = base_projection.get("cycles_to_10k_person_eq", 4)
    base_cycles_1M = base_projection.get("cycles_to_1M_person_eq", 15)
    base_alpha = base_projection.get("effective_alpha", MIN_EFF_ALPHA_VALIDATED)

    # Get GNN asymptote impact
    gnn_alpha = gnn_results.get("eff_alpha", base_alpha)
    asymptote_proximity = gnn_results.get(
        "asymptote_proximity", abs(ASYMPTOTE_ALPHA - gnn_alpha)
    )

    # Calculate adjusted cycles using asymptotic alpha
    # Higher alpha means fewer cycles needed
    alpha_ratio = base_alpha / max(0.1, gnn_alpha)

    adjusted_cycles_10k = max(1, math.ceil(base_cycles_10k * alpha_ratio))
    adjusted_cycles_1M = max(1, math.ceil(base_cycles_1M * alpha_ratio))

    # Cycles saved by asymptote approach
    cycles_saved_10k = base_cycles_10k - adjusted_cycles_10k
    cycles_saved_1M = base_cycles_1M - adjusted_cycles_1M

    projection = {
        "base_cycles_10k": base_cycles_10k,
        "base_cycles_1M": base_cycles_1M,
        "base_alpha": base_alpha,
        "gnn_alpha": gnn_alpha,
        "asymptote_alpha": ASYMPTOTE_ALPHA,
        "asymptote_proximity": round(asymptote_proximity, 4),
        "alpha_ratio": round(alpha_ratio, 4),
        "adjusted_cycles_10k": adjusted_cycles_10k,
        "adjusted_cycles_1M": adjusted_cycles_1M,
        "cycles_saved_10k": cycles_saved_10k,
        "cycles_saved_1M": cycles_saved_1M,
        "asymptote_validated": asymptote_proximity <= 0.02,
    }

    emit_receipt(
        "asymptote_projection",
        {
            "tenant_id": "spaceproof-reasoning",
            **projection,
            "payload_hash": dual_hash(json.dumps(projection, sort_keys=True)),
        },
    )

    return projection


def project_with_degradation(
    base_projection: Dict[str, Any],
    retention_curve_data: List[Dict[str, float]],
    target_blackout_days: int = 60,
    cache_depth: int = CACHE_DEPTH_BASELINE,
) -> Dict[str, Any]:
    """Adjust sovereignty timeline projection by GNN nonlinear alpha degradation.

    UPGRADED Dec 2025: Uses GNN nonlinear model (replaces linear).
    Takes a base projection and applies retention curve degradation at specified
    blackout duration.

    Args:
        base_projection: Base sovereignty projection dict with:
            - cycles_to_10k_person_eq
            - cycles_to_1M_person_eq
            - effective_alpha
        retention_curve_data: Output from generate_retention_curve_data
        target_blackout_days: Blackout duration to project (default: 60)
        cache_depth: Cache depth in entries (default: CACHE_DEPTH_BASELINE)

    Returns:
        Dict with adjusted projection including GNN nonlinear degradation metrics

    Receipt: degradation_projection
    """
    # Extract base values
    base_cycles_10k = base_projection.get("cycles_to_10k_person_eq", 4)
    base_cycles_1M = base_projection.get("cycles_to_1M_person_eq", 15)
    base_alpha = base_projection.get("effective_alpha", MIN_EFF_ALPHA_VALIDATED)

    # Get GNN nonlinear retention at target duration
    overflow_detected = False
    try:
        curve_point = retention_curve(target_blackout_days, cache_depth)
        degraded_alpha = curve_point["eff_alpha"]
        retention_factor = curve_point["retention_factor"]
        degradation_pct = curve_point["degradation_pct"]
        (curve_point.get("gnn_boost", 0.0) if "gnn_boost" in curve_point else 0.0)
    except StopRule:
        overflow_detected = True
        degraded_alpha = 0.0
        retention_factor = RETENTION_BASE_FACTOR
        degradation_pct = 100.0

    # Calculate adjusted cycles
    # Lower alpha means more cycles needed
    if not overflow_detected and degraded_alpha > 0:
        alpha_ratio = base_alpha / max(0.1, degraded_alpha)
        adjusted_cycles_10k = math.ceil(base_cycles_10k * alpha_ratio)
        adjusted_cycles_1M = math.ceil(base_cycles_1M * alpha_ratio)
    else:
        alpha_ratio = float("inf")
        adjusted_cycles_10k = 999
        adjusted_cycles_1M = 999

    # Delta calculation
    cycles_delay_10k = adjusted_cycles_10k - base_cycles_10k
    cycles_delay_1M = adjusted_cycles_1M - base_cycles_1M

    projection = {
        "base_cycles_10k": base_cycles_10k,
        "base_cycles_1M": base_cycles_1M,
        "base_alpha": base_alpha,
        "target_blackout_days": target_blackout_days,
        "degraded_alpha": degraded_alpha,
        "retention_factor": retention_factor,
        "degradation_pct": degradation_pct,
        "alpha_ratio": round(alpha_ratio, 4) if not overflow_detected else "overflow",
        "adjusted_cycles_10k": adjusted_cycles_10k,
        "adjusted_cycles_1M": adjusted_cycles_1M,
        "cycles_delay_10k": cycles_delay_10k,
        "cycles_delay_1M": cycles_delay_1M,
        "degradation_model": CURVE_TYPE,
        "overflow_detected": overflow_detected,
        "asymptote_alpha": ASYMPTOTE_ALPHA,
        "validated": not overflow_detected and degraded_alpha >= 2.50,
    }

    emit_receipt(
        "degradation_projection",
        {
            "tenant_id": "spaceproof-reasoning",
            **projection,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        k: v
                        for k, v in projection.items()
                        if k != "alpha_ratio" or not overflow_detected
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return projection


__all__ = [
    "blackout_sweep",
    "project_with_reroute",
    "sovereignty_timeline",
    "extended_blackout_sweep",
    "extreme_blackout_sweep_200d",
    "project_with_asymptote",
    "project_with_degradation",
]
