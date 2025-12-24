"""reasoning/partition.py - Partition Sweep and Resilience Projections.

Functions for partition simulation sweeps and resilience-adjusted projections.
"""

from typing import Any, Dict, Tuple
import math

from ..core import emit_receipt
from ..partition import (
    partition_sim,
    NODE_BASELINE,
    PARTITION_MAX_TEST_PCT,
    BASE_ALPHA,
)
from .constants import (
    MIN_EFF_ALPHA_BOUND,
    CYCLES_THRESHOLD_EARLY,
    CYCLES_THRESHOLD_CITY,
)


def partition_sweep(
    nodes: int = NODE_BASELINE,
    loss_range: Tuple[float, float] = (0.0, PARTITION_MAX_TEST_PCT),
    iterations: int = 100,
    base_alpha: float = BASE_ALPHA,
) -> Dict[str, Any]:
    """Run partition simulation across loss range.

    Performs partition_sim across the specified range and collects
    resilience metrics for projection adjustment.

    Args:
        nodes: Total node count (default: 5)
        loss_range: Tuple of (min_loss, max_loss) as floats (default: 0-40%)
        iterations: Number of samples across range (default: 100)
        base_alpha: Baseline effective alpha (default: 2.68)

    Returns:
        Dict with:
            - samples: List of partition results
            - worst_case_drop: Maximum alpha drop observed
            - avg_drop: Average alpha drop across samples
            - min_eff_alpha: Minimum effective alpha observed
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
            samples.append(
                {"nodes_total": nodes, "loss_pct": loss_pct, "quorum_status": False}
            )

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
        "samples_count": len(samples),
    }

    emit_receipt("partition_sweep", {"tenant_id": "axiom-reasoning", **report})

    return report


def project_with_resilience(
    base_projection: Dict[str, Any], partition_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Adjust sovereignty timeline projection by worst-case alpha drop.

    Takes a base projection and partition sweep results, adjusts
    the timeline by the worst-case effective alpha.

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
    # Formula: cycles_adjusted = cycles_base * (base_alpha / min_alpha)
    # Lower alpha means more cycles needed
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
        "resilience_validated": min_alpha >= MIN_EFF_ALPHA_BOUND,
    }

    emit_receipt(
        "resilience_projection", {"tenant_id": "axiom-reasoning", **projection}
    )

    return projection


def sovereignty_projection_with_partition(
    c_base: float = 50.0,
    p_factor: float = 1.8,
    alpha: float = BASE_ALPHA,
    loss_pct: float = PARTITION_MAX_TEST_PCT,
    include_sweep: bool = True,
) -> Dict[str, Any]:
    """Full sovereignty projection with partition resilience.

    Combines:
    1. Partition impact at specified loss percentage
    2. Optional sweep for worst-case analysis
    3. Adjusted timeline projections

    Args:
        c_base: Initial person-eq capacity (default: 50.0)
        p_factor: Propulsion growth factor (default: 1.8)
        alpha: Base effective alpha (default: 2.68)
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
    assert partition_result["eff_alpha"] >= MIN_EFF_ALPHA_BOUND, (
        f"eff_alpha {partition_result['eff_alpha']} < {MIN_EFF_ALPHA_BOUND} at {loss_pct * 100}% partition"
    )

    # Compute base projection (simplified model)
    # B = c * A^alpha * P where multiplier ~= 2.75x at alpha=1.69, scales with alpha
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
        "cycles_to_1M_person_eq": cycles_1M,
    }

    # Partition impact
    partition_impact = {
        "loss_pct": loss_pct,
        "nodes_total": partition_result["nodes_total"],
        "nodes_surviving": partition_result["nodes_surviving"],
        "eff_alpha_drop": partition_result["eff_alpha_drop"],
        "eff_alpha": partition_result["eff_alpha"],
        "quorum_status": partition_result["quorum_status"],
    }

    # Optional sweep
    sweep_results = None
    if include_sweep:
        sweep_results = partition_sweep(NODE_BASELINE, (0.0, loss_pct), 100, alpha)

    # Resilience projection
    partition_for_projection = (
        sweep_results
        if sweep_results
        else {
            "worst_case_drop": partition_result["eff_alpha_drop"],
            "min_eff_alpha": partition_result["eff_alpha"],
        }
    )
    resilience = project_with_resilience(base_projection, partition_for_projection)

    full_projection = {
        "base_projection": base_projection,
        "partition_impact": partition_impact,
        "sweep_results": sweep_results,
        "resilience_projection": resilience,
        "slo_validated": partition_result["eff_alpha"] >= MIN_EFF_ALPHA_BOUND,
    }

    emit_receipt(
        "sovereignty_partition_projection",
        {
            "tenant_id": "axiom-reasoning",
            "c_base": c_base,
            "p_factor": p_factor,
            "base_alpha": alpha,
            "loss_pct": loss_pct,
            "base_cycles_10k": cycles_10k,
            "adjusted_cycles_10k": resilience["adjusted_cycles_10k"],
            "min_eff_alpha": resilience["min_eff_alpha"],
            "cycles_delay": resilience["cycles_delay_10k"],
            "slo_validated": full_projection["slo_validated"],
        },
    )

    return full_projection


def validate_resilience_slo(
    nodes: int = NODE_BASELINE,
    max_loss: float = PARTITION_MAX_TEST_PCT,
    min_alpha: float = MIN_EFF_ALPHA_BOUND,
    max_drop: float = 0.05,
) -> Dict[str, Any]:
    """Validate resilience SLOs for partition testing.

    SLOs:
    1. eff_alpha >= 2.63 at 40% partition
    2. alpha drop < 0.05 from baseline
    3. Quorum survives at 40% (2 node loss)

    Args:
        nodes: Baseline node count (default: 5)
        max_loss: Maximum partition percentage (default: 40%)
        min_alpha: Minimum required alpha (default: 2.63)
        max_drop: Maximum allowed alpha drop (default: 0.05)

    Returns:
        Dict with validation results

    Receipt: resilience_slo_validation
    """
    result = partition_sim(nodes, max_loss, BASE_ALPHA, emit=False)

    validations = {
        "alpha_slo": result["eff_alpha"] >= min_alpha,
        "drop_slo": result["eff_alpha_drop"] <= max_drop,  # At boundary at 40%
        "quorum_slo": result["quorum_status"],
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
        "all_passed": all_passed,
    }

    emit_receipt(
        "resilience_slo_validation", {"tenant_id": "axiom-reasoning", **report}
    )

    return report


__all__ = [
    "partition_sweep",
    "project_with_resilience",
    "sovereignty_projection_with_partition",
    "validate_resilience_slo",
]
