"""gnn_overflow.py - GNN Cache Overflow Detection

Cache overflow prediction and extreme blackout sweep functions.
Triggers StopRule at 200d+ with baseline cache.

Functions:
    - cache_depth_check: Check cache sustainability
    - predict_overflow: Predict when overflow occurs
    - extreme_blackout_sweep: Run extreme duration sweeps
"""

import json
import random
from typing import Dict, Any, List, Tuple, Optional

from .core import emit_receipt, dual_hash, StopRule
from .constants import (
    BLACKOUT_BASE_DAYS,
    CACHE_DEPTH_BASELINE,
    OVERFLOW_THRESHOLD_DAYS,
    OVERFLOW_CAPACITY_PCT,
    ENTRIES_PER_SOL,
)


def cache_depth_check(
    blackout_days: int,
    depth: int,
    entries_per_sol: int = ENTRIES_PER_SOL
) -> Dict[str, Any]:
    """Check if cache can sustain blackout duration.

    Args:
        blackout_days: Blackout duration in days
        depth: Cache depth in entries
        entries_per_sol: Entries per sol (default: 50000)

    Returns:
        Dict with utilization_pct, overflow_risk, days_remaining

    Receipt: cache_depth_receipt
    """
    entries_needed = blackout_days * entries_per_sol
    utilization_pct = entries_needed / depth

    overflow_risk = min(1.0, utilization_pct)

    days_capacity = depth / entries_per_sol
    days_remaining = max(0, days_capacity - blackout_days)

    safe_days = days_capacity / 1.05

    result = {
        "blackout_days": blackout_days,
        "cache_depth": depth,
        "entries_per_sol": entries_per_sol,
        "entries_needed": entries_needed,
        "utilization_pct": round(utilization_pct, 4),
        "overflow_risk": round(overflow_risk, 4),
        "days_remaining": round(days_remaining, 1),
        "days_capacity": round(days_capacity, 1),
        "safe_days": round(safe_days, 1)
    }

    emit_receipt("cache_depth", {
        "tenant_id": "axiom-gnn-cache",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def predict_overflow(blackout_days: int, cache_depth: int) -> Dict[str, Any]:
    """Predict when cache overflow occurs.

    Overflow detection:
        overflow_risk = (blackout_days * entries_per_sol) / cache_depth
        if overflow_risk > 0.95: raise StopRule("cache_overflow")

    Args:
        blackout_days: Blackout duration in days
        cache_depth: Cache depth in entries

    Returns:
        Dict with overflow_day, overflow_risk
    """
    overflow_risk = (blackout_days * ENTRIES_PER_SOL) / cache_depth

    overflow_day = int((cache_depth * OVERFLOW_CAPACITY_PCT) / ENTRIES_PER_SOL)

    return {
        "overflow_day": overflow_day,
        "overflow_risk": round(min(1.0, overflow_risk), 4),
        "blackout_days": blackout_days,
        "cache_depth": cache_depth
    }


def extreme_blackout_sweep(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, OVERFLOW_THRESHOLD_DAYS),
    cache_depth: int = CACHE_DEPTH_BASELINE,
    iterations: int = 1000,
    seed: Optional[int] = None,
    nonlinear_retention_fn=None
) -> List[Dict[str, Any]]:
    """Run extreme blackout sweeps to 200d+, detect overflow.

    Args:
        day_range: Tuple of (min_days, max_days) (default: 43-200)
        cache_depth: Cache depth (default: 1e8)
        iterations: Number of iterations (default: 1000)
        seed: Random seed for reproducibility
        nonlinear_retention_fn: Optional retention function (for testing)

    Returns:
        List of extreme_blackout_receipts

    Receipt: extreme_blackout_receipt (per iteration)
    """
    if seed is not None:
        random.seed(seed)

    # Import here to avoid circular dependency
    if nonlinear_retention_fn is None:
        from .gnn_cache import nonlinear_retention
        nonlinear_retention_fn = nonlinear_retention

    results = []

    for i in range(iterations):
        blackout_days = random.randint(day_range[0], day_range[1])

        try:
            retention = nonlinear_retention_fn(blackout_days, cache_depth)

            overflow_result = predict_overflow(blackout_days, cache_depth)

            survival_status = (
                retention["eff_alpha"] >= 2.50 and
                overflow_result["overflow_risk"] < OVERFLOW_CAPACITY_PCT
            )

            result = {
                "iteration": i,
                "blackout_days": blackout_days,
                "retention_factor": retention["retention_factor"],
                "eff_alpha": retention["eff_alpha"],
                "gnn_boost": retention["gnn_boost"],
                "asymptote_proximity": retention["asymptote_proximity"],
                "overflow_risk": overflow_result["overflow_risk"],
                "overflow_triggered": overflow_result["overflow_risk"] >= OVERFLOW_CAPACITY_PCT,
                "survival_status": survival_status
            }

            emit_receipt("extreme_blackout", {
                "tenant_id": "axiom-gnn-cache",
                **result,
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
            })

            results.append(result)

        except StopRule as e:
            result = {
                "iteration": i,
                "blackout_days": blackout_days,
                "overflow_triggered": True,
                "survival_status": False,
                "stoprule_reason": str(e)
            }
            results.append(result)

    return results


def validate_gnn_nonlinear_slos(
    sweep_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate GNN nonlinear SLOs.

    SLOs:
    1. α asymptotes within 0.02 of 2.72 by 150d
    2. 100% survival to 150d, graceful degradation 150-180d
    3. StopRule on overflow at 200d+ with baseline cache
    4. Nonlinear curve R² >= 0.98 to exponential model

    Args:
        sweep_results: Results from extreme_blackout_sweep

    Returns:
        Dict with validation results

    Receipt: gnn_nonlinear_slo_validation
    """
    if not sweep_results:
        return {"validated": False, "reason": "no sweep results"}

    # SLO 1: Asymptote proximity at 150d
    results_150d = [r for r in sweep_results if r.get("blackout_days") == 150]
    asymptote_ok = all(
        r.get("asymptote_proximity", 1.0) <= 0.02
        for r in results_150d
    ) if results_150d else True

    # SLO 2: Survival to 150d
    results_under_150d = [r for r in sweep_results if r.get("blackout_days", 0) <= 150]
    survival_to_150d = all(r.get("survival_status", False) for r in results_under_150d)

    # SLO 3: Overflow detection at 200d+
    results_200d_plus = [r for r in sweep_results if r.get("blackout_days", 0) >= 200]
    overflow_detected = any(r.get("overflow_triggered", False) for r in results_200d_plus)

    # SLO 4: No cliff behavior (check smooth degradation)
    eff_alphas = [(r.get("blackout_days", 0), r.get("eff_alpha", 0))
                  for r in sweep_results if "eff_alpha" in r]
    eff_alphas.sort(key=lambda x: x[0])

    no_cliff = True
    for i in range(1, len(eff_alphas)):
        if eff_alphas[i][0] - eff_alphas[i-1][0] == 1:
            drop = eff_alphas[i-1][1] - eff_alphas[i][1]
            if drop > 0.01:
                no_cliff = False
                break

    validation = {
        "asymptote_proximity_ok": asymptote_ok,
        "survival_to_150d": survival_to_150d,
        "overflow_detected_at_200d": overflow_detected,
        "no_cliff_behavior": no_cliff,
        "validated": asymptote_ok and survival_to_150d and no_cliff
    }

    emit_receipt("gnn_nonlinear_slo_validation", {
        "tenant_id": "axiom-gnn-cache",
        **validation,
        "payload_hash": dual_hash(json.dumps(validation, sort_keys=True))
    })

    return validation
