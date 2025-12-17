"""reasoning.py - Sovereignty Timeline Projections with Partition Stress Testing

Extends sovereignty timeline projections with:
1. Partition stress sweep integration
2. Resilience-adjusted projections
3. Worst-case α drop calculations
4. Blackout sweeps and reroute projection (Dec 2025)
5. Extended blackout sweep (43-90d) with retention curve modeling

THE PHYSICS:
    eff_alpha(partition=0.4, nodes=5) >= 2.63 (per Grok validation)
    Worst-case drop at 40% partition: ~0.05 from baseline 2.68

REROUTE INTEGRATION (Dec 2025):
    eff_alpha(reroute=True) >= 2.70 (+0.07 boost)
    blackout_survival(days=60, reroute=True) == True
    Blackout resilience metrics in projection receipt

EXTENDED BLACKOUT (Dec 2025):
    eff_alpha(blackout=60) >= 2.69 (assert)
    eff_alpha(blackout=90) >= 2.65 (assert)
    Retention curve: 1.4 @ 43d → 1.25 @ 90d (linear, no cliff)

Source: Grok - "Variable partitions (e.g., 40%)", "eff_α to 2.68", "+0.07 to 2.7+"
"""

from typing import Dict, Any, List, Tuple, Optional
import math

from .core import emit_receipt
from .partition import (
    partition_sim,
    stress_sweep,
    validate_partition_bounds,
    NODE_BASELINE,
    QUORUM_THRESHOLD,
    PARTITION_MAX_TEST_PCT,
    BASE_ALPHA,
    ALPHA_DROP_FACTOR
)
from .ledger import (
    LEDGER_ALPHA_BOOST_VALIDATED,
    BASE_ALPHA_PRE_BOOST,
    get_effective_alpha_with_partition
)
from .reroute import (
    blackout_sim,
    blackout_stress_sweep,
    apply_reroute_boost,
    adaptive_reroute,
    REROUTE_ALPHA_BOOST,
    REROUTING_ALPHA_BOOST_LOCKED,
    BLACKOUT_BASE_DAYS,
    BLACKOUT_EXTENDED_DAYS,
    MIN_EFF_ALPHA_FLOOR,
    MIN_EFF_ALPHA_VALIDATED
)
from .blackout import (
    retention_curve,
    alpha_at_duration,
    extended_blackout_sweep as blackout_extended_sweep,
    generate_retention_curve_data,
    find_retention_floor,
    BLACKOUT_SWEEP_MAX_DAYS,
    RETENTION_BASE_FACTOR,
    DEGRADATION_RATE
)


# === CONSTANTS ===

MIN_EFF_ALPHA_BOUND = 2.63
"""Minimum effective α at 40% partition per Grok validation."""

CYCLES_THRESHOLD_EARLY = 1000
"""Early sovereignty marker: 10³ person-equivalent."""

CYCLES_THRESHOLD_CITY = 1_000_000
"""City sovereignty threshold: 10⁶ person-equivalent."""


def partition_sweep(
    nodes: int = NODE_BASELINE,
    loss_range: Tuple[float, float] = (0.0, PARTITION_MAX_TEST_PCT),
    iterations: int = 100,
    base_alpha: float = BASE_ALPHA
) -> Dict[str, Any]:
    """Run partition simulation across loss range.

    Performs partition_sim across the specified range and collects
    resilience metrics for projection adjustment.

    Args:
        nodes: Total node count (default: 5)
        loss_range: Tuple of (min_loss, max_loss) as floats (default: 0-40%)
        iterations: Number of samples across range (default: 100)
        base_alpha: Baseline effective α (default: 2.68)

    Returns:
        Dict with:
            - samples: List of partition results
            - worst_case_drop: Maximum α drop observed
            - avg_drop: Average α drop across samples
            - min_eff_alpha: Minimum effective α observed
            - quorum_success_rate: Fraction with quorum intact

    Receipt: partition_sweep
    """
    samples = []
    total_drop = 0.0
    worst_drop = 0.0
    min_alpha = base_alpha
    quorum_successes = 0

    # Linear samples across range
    step = (loss_range[1] - loss_range[0]) / max(1, iterations - 1)

    for i in range(iterations):
        loss_pct = loss_range[0] + i * step
        loss_pct = min(loss_pct, loss_range[1])  # Clamp to max

        try:
            result = partition_sim(nodes, loss_pct, base_alpha, emit=False)
            samples.append(result)
            total_drop += result["eff_alpha_drop"]
            worst_drop = max(worst_drop, result["eff_alpha_drop"])
            min_alpha = min(min_alpha, result["eff_alpha"])
            if result["quorum_status"]:
                quorum_successes += 1
        except Exception:
            # Quorum failed - shouldn't happen in valid range
            samples.append({
                "nodes_total": nodes,
                "loss_pct": loss_pct,
                "quorum_status": False
            })

    avg_drop = total_drop / max(1, len(samples))
    quorum_rate = quorum_successes / max(1, len(samples))

    report = {
        "nodes": nodes,
        "loss_range": list(loss_range),
        "iterations": iterations,
        "base_alpha": base_alpha,
        "worst_case_drop": round(worst_drop, 4),
        "avg_drop": round(avg_drop, 4),
        "min_eff_alpha": round(min_alpha, 4),
        "quorum_success_rate": round(quorum_rate, 4),
        "samples_count": len(samples)
    }

    emit_receipt("partition_sweep", {
        "tenant_id": "axiom-reasoning",
        **report
    })

    return report


def project_with_resilience(
    base_projection: Dict[str, Any],
    partition_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Adjust sovereignty timeline projection by worst-case α drop.

    Takes a base projection and partition sweep results, adjusts
    the timeline by the worst-case effective α.

    Args:
        base_projection: Base sovereignty projection dict with:
            - cycles_to_10k_person_eq
            - cycles_to_1M_person_eq
            - effective_alpha
        partition_results: Output from partition_sweep with:
            - worst_case_drop
            - min_eff_alpha

    Returns:
        Dict with adjusted projection including resilience metrics

    Receipt: resilience_projection
    """
    # Extract base values
    base_cycles_10k = base_projection.get("cycles_to_10k_person_eq", 4)
    base_cycles_1M = base_projection.get("cycles_to_1M_person_eq", 15)
    base_alpha = base_projection.get("effective_alpha", BASE_ALPHA)

    # Get partition impact
    worst_drop = partition_results.get("worst_case_drop", 0.0)
    min_alpha = partition_results.get("min_eff_alpha", base_alpha)

    # Calculate adjusted cycles
    # Formula: cycles_adjusted = cycles_base × (base_alpha / min_alpha)
    # Lower α means more cycles needed
    alpha_ratio = base_alpha / max(0.1, min_alpha)

    adjusted_cycles_10k = math.ceil(base_cycles_10k * alpha_ratio)
    adjusted_cycles_1M = math.ceil(base_cycles_1M * alpha_ratio)

    # Delta calculation
    cycles_delay_10k = adjusted_cycles_10k - base_cycles_10k
    cycles_delay_1M = adjusted_cycles_1M - base_cycles_1M

    projection = {
        "base_cycles_10k": base_cycles_10k,
        "base_cycles_1M": base_cycles_1M,
        "base_alpha": base_alpha,
        "worst_case_drop": worst_drop,
        "min_eff_alpha": min_alpha,
        "alpha_ratio": round(alpha_ratio, 4),
        "adjusted_cycles_10k": adjusted_cycles_10k,
        "adjusted_cycles_1M": adjusted_cycles_1M,
        "cycles_delay_10k": cycles_delay_10k,
        "cycles_delay_1M": cycles_delay_1M,
        "resilience_validated": min_alpha >= MIN_EFF_ALPHA_BOUND
    }

    emit_receipt("resilience_projection", {
        "tenant_id": "axiom-reasoning",
        **projection
    })

    return projection


def sovereignty_projection_with_partition(
    c_base: float = 50.0,
    p_factor: float = 1.8,
    alpha: float = BASE_ALPHA,
    loss_pct: float = PARTITION_MAX_TEST_PCT,
    include_sweep: bool = True
) -> Dict[str, Any]:
    """Full sovereignty projection with partition resilience.

    Combines:
    1. Partition impact at specified loss percentage
    2. Optional sweep for worst-case analysis
    3. Adjusted timeline projections

    Args:
        c_base: Initial person-eq capacity (default: 50.0)
        p_factor: Propulsion growth factor (default: 1.8)
        alpha: Base effective α (default: 2.68)
        loss_pct: Partition loss to simulate (default: 40%)
        include_sweep: Whether to run full sweep (default: True)

    Returns:
        Dict with complete projection including:
            - base_projection: Original timeline
            - partition_impact: Single partition result
            - sweep_results: Full sweep analysis (if include_sweep)
            - resilience_projection: Adjusted timeline

    Receipt: sovereignty_partition_projection

    Assertion: eff_alpha(partition=0.4, nodes=5) >= 2.63
    """
    # Validate bounds assertion
    partition_result = partition_sim(NODE_BASELINE, loss_pct, alpha, emit=False)
    assert partition_result["eff_alpha"] >= MIN_EFF_ALPHA_BOUND, \
        f"eff_alpha {partition_result['eff_alpha']} < {MIN_EFF_ALPHA_BOUND} at {loss_pct*100}% partition"

    # Compute base projection (simplified model)
    # B = c × A^α × P where multiplier ≈ 2.75x at α=1.69, scales with α
    alpha_ratio = alpha / 1.69
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

    base_projection = {
        "c_base": c_base,
        "p_factor": p_factor,
        "effective_alpha": alpha,
        "cycles_to_10k_person_eq": cycles_10k,
        "cycles_to_1M_person_eq": cycles_1M
    }

    # Partition impact
    partition_impact = {
        "loss_pct": loss_pct,
        "nodes_total": partition_result["nodes_total"],
        "nodes_surviving": partition_result["nodes_surviving"],
        "eff_alpha_drop": partition_result["eff_alpha_drop"],
        "eff_alpha": partition_result["eff_alpha"],
        "quorum_status": partition_result["quorum_status"]
    }

    # Optional sweep
    sweep_results = None
    if include_sweep:
        sweep_results = partition_sweep(NODE_BASELINE, (0.0, loss_pct), 100, alpha)

    # Resilience projection
    partition_for_projection = sweep_results if sweep_results else {
        "worst_case_drop": partition_result["eff_alpha_drop"],
        "min_eff_alpha": partition_result["eff_alpha"]
    }
    resilience = project_with_resilience(base_projection, partition_for_projection)

    full_projection = {
        "base_projection": base_projection,
        "partition_impact": partition_impact,
        "sweep_results": sweep_results,
        "resilience_projection": resilience,
        "slo_validated": partition_result["eff_alpha"] >= MIN_EFF_ALPHA_BOUND
    }

    emit_receipt("sovereignty_partition_projection", {
        "tenant_id": "axiom-reasoning",
        "c_base": c_base,
        "p_factor": p_factor,
        "base_alpha": alpha,
        "loss_pct": loss_pct,
        "base_cycles_10k": cycles_10k,
        "adjusted_cycles_10k": resilience["adjusted_cycles_10k"],
        "min_eff_alpha": resilience["min_eff_alpha"],
        "cycles_delay": resilience["cycles_delay_10k"],
        "slo_validated": full_projection["slo_validated"]
    })

    return full_projection


def validate_resilience_slo(
    nodes: int = NODE_BASELINE,
    max_loss: float = PARTITION_MAX_TEST_PCT,
    min_alpha: float = MIN_EFF_ALPHA_BOUND,
    max_drop: float = 0.05
) -> Dict[str, Any]:
    """Validate resilience SLOs for partition testing.

    SLOs:
    1. eff_alpha >= 2.63 at 40% partition
    2. α drop < 0.05 from baseline
    3. Quorum survives at 40% (2 node loss)

    Args:
        nodes: Baseline node count (default: 5)
        max_loss: Maximum partition percentage (default: 40%)
        min_alpha: Minimum required α (default: 2.63)
        max_drop: Maximum allowed α drop (default: 0.05)

    Returns:
        Dict with validation results

    Receipt: resilience_slo_validation
    """
    result = partition_sim(nodes, max_loss, BASE_ALPHA, emit=False)

    validations = {
        "alpha_slo": result["eff_alpha"] >= min_alpha,
        "drop_slo": result["eff_alpha_drop"] <= max_drop,  # At boundary at 40%
        "quorum_slo": result["quorum_status"]
    }

    all_passed = all(validations.values())

    report = {
        "nodes": nodes,
        "loss_pct": max_loss,
        "eff_alpha": result["eff_alpha"],
        "eff_alpha_drop": result["eff_alpha_drop"],
        "quorum_status": result["quorum_status"],
        "min_alpha_required": min_alpha,
        "max_drop_allowed": max_drop,
        **validations,
        "all_passed": all_passed
    }

    emit_receipt("resilience_slo_validation", {
        "tenant_id": "axiom-reasoning",
        **report
    })

    return report


def blackout_sweep(
    nodes: int = NODE_BASELINE,
    blackout_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_EXTENDED_DAYS),
    reroute_enabled: bool = True,
    iterations: int = 1000,
    base_alpha: float = MIN_EFF_ALPHA_BOUND,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run blackout simulation sweep across duration range.

    Runs blackout_sim across the specified range and collects
    resilience metrics for projection adjustment.

    Args:
        nodes: Total node count (default: 5)
        blackout_range: Tuple of (min_days, max_days) (default: 43-60)
        reroute_enabled: Whether adaptive rerouting is active
        iterations: Number of iterations (default: 1000)
        base_alpha: Baseline effective α (default: 2.63)
        seed: Random seed for reproducibility

    Returns:
        Dict with:
            - survival_rate: Fraction of successful blackouts
            - avg_min_alpha: Average minimum α during blackouts
            - avg_max_drop: Average maximum α drop
            - worst_case_alpha: Minimum α across all iterations
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
        seed=seed
    )

    # Compute additional metrics
    if reroute_enabled:
        # Verify reroute boost pushes alpha to 2.70+
        boosted_alpha = apply_reroute_boost(base_alpha, True, 0)
        assert boosted_alpha >= 2.70, \
            f"eff_alpha(reroute=True) = {boosted_alpha} < 2.70"

        # Verify 60-day survival with reroute
        extended_sim = blackout_sim(nodes, 60, True, base_alpha, seed)
        assert extended_sim["survival_status"], \
            "blackout_survival(days=60, reroute=True) failed"

    # Find blackout tolerance (max days survived)
    blackout_tolerance = blackout_range[1] if result["all_survived"] else blackout_range[0]

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
        "boosted_alpha": base_alpha + REROUTE_ALPHA_BOOST if reroute_enabled else base_alpha
    }

    emit_receipt("blackout_sweep", {
        "tenant_id": "axiom-reasoning",
        **report
    })

    return report


def project_with_reroute(
    base_projection: Dict[str, Any],
    reroute_results: Dict[str, Any],
    blackout_days: int = 0
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
    survival_rate = reroute_results.get("survival_rate", 1.0)

    # Apply reroute boost
    boosted_alpha = apply_reroute_boost(base_alpha, True, blackout_days)

    # Calculate adjusted cycles
    # Higher α means fewer cycles needed
    # Formula: cycles_adjusted = cycles_base × (base_alpha / boosted_alpha)
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
        "tolerance_factor": 1.0 if blackout_days <= BLACKOUT_BASE_DAYS else (
            max(0.7, 1.0 - (blackout_days - BLACKOUT_BASE_DAYS) / BLACKOUT_EXTENDED_DAYS)
        )
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
        "reroute_validated": boosted_alpha >= 2.70
    }

    emit_receipt("reroute_projection", {
        "tenant_id": "axiom-reasoning",
        **projection
    })

    return projection


def sovereignty_timeline(
    c_base: float = 50.0,
    p_factor: float = 1.8,
    alpha: float = BASE_ALPHA,
    loss_pct: float = 0.0,
    reroute_enabled: bool = False,
    blackout_days: int = 0
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
        alpha: Base effective α (default: 2.68)
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
        "min_eff_alpha_floor": MIN_EFF_ALPHA_FLOOR
    }

    emit_receipt("sovereignty_timeline", {
        "tenant_id": "axiom-reasoning",
        **result
    })

    return result


def extended_blackout_sweep(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_SWEEP_MAX_DAYS),
    iterations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run extended blackout sweep across day range with retention curve modeling.

    Runs blackout_extended_sweep from blackout.py and returns comprehensive results
    including retention curve data.

    Assertions:
        - assert eff_alpha(blackout=60) >= 2.69
        - assert eff_alpha(blackout=90) >= 2.65

    Args:
        day_range: Tuple of (min_days, max_days) (default: 43-90)
        iterations: Number of iterations (default: 1000)
        seed: Random seed for reproducibility

    Returns:
        Dict with sweep results and retention curve receipt

    Receipt: extended_blackout_sweep
    """
    # Run extended sweep from blackout module
    sweep_results = blackout_extended_sweep(day_range, iterations, seed)

    # Validate assertions
    alpha_60 = alpha_at_duration(60)
    alpha_90 = alpha_at_duration(90)

    assert alpha_60 >= 2.69, f"eff_alpha(blackout=60) = {alpha_60} < 2.69"
    assert alpha_90 >= 2.65, f"eff_alpha(blackout=90) = {alpha_90} < 2.65"

    # Generate retention curve data
    curve_data = generate_retention_curve_data(day_range)

    # Find floor
    floor_data = find_retention_floor(sweep_results)

    # Compute stats
    all_survived = all(r["survival_status"] for r in sweep_results)
    survival_rate = len([r for r in sweep_results if r["survival_status"]]) / max(1, len(sweep_results))
    avg_alpha = sum(r["eff_alpha"] for r in sweep_results) / max(1, len(sweep_results))
    min_alpha = min(r["eff_alpha"] for r in sweep_results) if sweep_results else 0.0

    result = {
        "day_range": list(day_range),
        "iterations": iterations,
        "all_survived": all_survived,
        "survival_rate": round(survival_rate, 4),
        "avg_alpha": round(avg_alpha, 4),
        "min_alpha": round(min_alpha, 4),
        "alpha_at_60d": alpha_60,
        "alpha_at_90d": alpha_90,
        "retention_floor": floor_data,
        "curve_points_count": len(curve_data),
        "assertions_passed": {
            "alpha_60_ge_2.69": alpha_60 >= 2.69,
            "alpha_90_ge_2.65": alpha_90 >= 2.65
        }
    }

    emit_receipt("extended_blackout_sweep", {
        "tenant_id": "axiom-reasoning",
        **result
    })

    return result


def project_with_degradation(
    base_projection: Dict[str, Any],
    retention_curve_data: List[Dict[str, float]],
    target_blackout_days: int = 60
) -> Dict[str, Any]:
    """Adjust sovereignty timeline projection by duration-dependent α degradation.

    Takes a base projection and applies retention curve degradation at specified
    blackout duration.

    Args:
        base_projection: Base sovereignty projection dict with:
            - cycles_to_10k_person_eq
            - cycles_to_1M_person_eq
            - effective_alpha
        retention_curve_data: Output from generate_retention_curve_data
        target_blackout_days: Blackout duration to project (default: 60)

    Returns:
        Dict with adjusted projection including degradation metrics

    Receipt: degradation_projection
    """
    # Extract base values
    base_cycles_10k = base_projection.get("cycles_to_10k_person_eq", 4)
    base_cycles_1M = base_projection.get("cycles_to_1M_person_eq", 15)
    base_alpha = base_projection.get("effective_alpha", MIN_EFF_ALPHA_VALIDATED)

    # Get retention curve point at target duration
    curve_point = retention_curve(target_blackout_days)
    degraded_alpha = curve_point["eff_alpha"]
    retention_factor = curve_point["retention_factor"]
    degradation_pct = curve_point["degradation_pct"]

    # Calculate adjusted cycles
    # Lower α means more cycles needed
    alpha_ratio = base_alpha / max(0.1, degraded_alpha)

    adjusted_cycles_10k = math.ceil(base_cycles_10k * alpha_ratio)
    adjusted_cycles_1M = math.ceil(base_cycles_1M * alpha_ratio)

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
        "alpha_ratio": round(alpha_ratio, 4),
        "adjusted_cycles_10k": adjusted_cycles_10k,
        "adjusted_cycles_1M": adjusted_cycles_1M,
        "cycles_delay_10k": cycles_delay_10k,
        "cycles_delay_1M": cycles_delay_1M,
        "degradation_model": "linear",
        "validated": degraded_alpha >= MIN_EFF_ALPHA_VALIDATED
    }

    emit_receipt("degradation_projection", {
        "tenant_id": "axiom-reasoning",
        **projection
    })

    return projection
