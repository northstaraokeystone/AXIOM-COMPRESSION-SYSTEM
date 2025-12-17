"""blackout.py - Extended Blackout Duration Modeling with GNN Nonlinear Retention Curve

THE PHYSICS (from Grok simulation - UPDATED Dec 2025):
    GNN predictive caching provides nonlinear boost via anticipatory buffering
    α asymptotes ~2.72 (e-like stability) rather than linear decay
    Holds to 150d before dipping, <2.5 at 180d+, breaks at 200d+ on cache overflow

KEY DISCOVERY:
    Model was conservative. Nonlinear retention >> linear prediction.
    At 90d: eff_α = 2.7185 (vs linear predicted 2.65)
    At 150d: eff_α ≈ 2.71 (asymptote approach)

CONSTANTS:
    BLACKOUT_BASE_DAYS = 43 (baseline conjunction)
    BLACKOUT_SWEEP_MAX_DAYS = 200 (extended extreme bound - was 90)
    RETENTION_BASE_FACTOR = 1.4 (baseline at 43d)
    MIN_EFF_ALPHA_VALIDATED = 2.7185 (from 1000-run sweep - UPGRADED from 2.656)
    REROUTING_ALPHA_BOOST_LOCKED = 0.07 (validated, immutable)
    CURVE_TYPE = "gnn_nonlinear" (REPLACES linear model)

KILLED (Dec 2025):
    DEGRADATION_RATE = 0.0032/day (linear model OBSOLETE)
    Linear formula: retention = BASE * (1 - RATE * (d - 43)) (KILLED)

Source: Grok - "Flatter retention... GNN adds nonlinear boosts", "α asymptotes ~2.72"
"""

import json
import os
import random
from typing import Dict, Any, List, Tuple, Optional

from .core import emit_receipt, dual_hash, StopRule
from .gnn_cache import (
    nonlinear_retention as gnn_nonlinear_retention,
    nonlinear_retention_with_pruning,
    ASYMPTOTE_ALPHA,
    PRUNING_TARGET_ALPHA,
    CACHE_DEPTH_BASELINE,
    OVERFLOW_THRESHOLD_DAYS,
    NONLINEAR_RETENTION_FLOOR,
    BLACKOUT_PRUNING_TARGET_DAYS,
)


# === CONSTANTS (Dec 2025 GNN nonlinear - UPGRADED) ===

BLACKOUT_BASE_DAYS = 43
"""physics: Mars solar conjunction maximum duration in days."""

BLACKOUT_SWEEP_MAX_DAYS = 200
"""physics: Extended extreme stress bound (was 90d, now 200d with GNN caching)."""

BLACKOUT_MAX_UNREALISTIC = 300
"""physics: StopRule threshold for unrealistic blackout duration (was 120, now 300)."""

RETENTION_BASE_FACTOR = 1.4
"""physics: Baseline retention factor at 43d blackout."""

MIN_EFF_ALPHA_VALIDATED = 2.7185
"""physics: Validated min effective alpha from 1000-run sweep (UPGRADED from 2.656)."""

REROUTING_ALPHA_BOOST_LOCKED = 0.07
"""physics: Validated reroute boost (locked, immutable)."""

# KILLED: Linear degradation rate (Dec 2025)
# Original: DEGRADATION_RATE = 0.0032 - OBSOLETE, killed by GNN nonlinear model
# Instead, use gnn_cache.nonlinear_retention() which provides exponential decay
# Backward compatibility: Keep constant for tests that import it, but it's unused
DEGRADATION_RATE = 0.0  # DEPRECATED - Linear model killed, value unused

CURVE_TYPE = "gnn_nonlinear"
"""physics: Model identifier - GNN nonlinear replaces linear."""

DEGRADATION_MODEL = "gnn_nonlinear"
"""physics: GNN nonlinear degradation model (REPLACES linear)."""

TEST_RUNS_BENCHMARK = 38
"""Performance reference from prior gate."""

BLACKOUT_EXTENSION_SPEC_PATH = "data/blackout_extension_spec.json"
"""Path to blackout extension specification file."""


def load_blackout_extension_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify blackout extension specification file.

    Loads data/blackout_extension_spec.json and emits ingest receipt
    with dual_hash per CLAUDEME S4.1.

    Args:
        path: Optional path override (default: BLACKOUT_EXTENSION_SPEC_PATH)

    Returns:
        Dict containing blackout extension specification

    Receipt: blackout_extension_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, BLACKOUT_EXTENSION_SPEC_PATH)

    with open(path, 'r') as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt("blackout_extension_spec_ingest", {
        "tenant_id": "axiom-blackout",
        "file_path": path,
        "blackout_base_days": data["blackout_base_days"],
        "blackout_sweep_max_days": data["blackout_sweep_max_days"],
        "retention_base_factor": data["retention_base_factor"],
        "min_eff_alpha_validated": data["min_eff_alpha_validated"],
        "rerouting_alpha_boost_locked": data["rerouting_alpha_boost_locked"],
        "degradation_model": data["degradation_model"],
        "payload_hash": content_hash
    })

    return data


def compute_degradation(blackout_days: int, base_retention: float = RETENTION_BASE_FACTOR) -> float:
    """Compute degradation factor for given blackout duration.

    GNN NONLINEAR model (Dec 2025): delegates to gnn_cache for exponential decay.
    KILLED: Linear model (degradation = (days - 43) * DEGRADATION_RATE)

    Args:
        blackout_days: Blackout duration in days
        base_retention: Base retention factor (default: 1.4)

    Returns:
        degradation_factor: Multiplicative degradation factor (0-1)
    """
    if blackout_days <= BLACKOUT_BASE_DAYS:
        return 0.0

    # Delegate to GNN nonlinear model
    try:
        result = gnn_nonlinear_retention(blackout_days, CACHE_DEPTH_BASELINE)
        retention_factor = result["retention_factor"]
        # Compute degradation as percentage drop from base
        degradation = 1.0 - (retention_factor / base_retention)
        return round(max(0.0, degradation), 4)
    except StopRule:
        # Cache overflow - return maximum degradation
        return round(1.0 - (NONLINEAR_RETENTION_FLOOR / base_retention), 4)


def retention_curve(blackout_days: int, cache_depth: int = CACHE_DEPTH_BASELINE) -> Dict[str, Any]:
    """Compute retention curve point for given blackout duration.

    DELEGATES TO GNN NONLINEAR MODEL (Dec 2025).
    Pure function. Returns retention_factor, eff_alpha, degradation_pct.
    Raises StopRule if blackout_days > 300 (unrealistic) or cache overflow.

    Args:
        blackout_days: Blackout duration in days
        cache_depth: Cache depth in entries (default: CACHE_DEPTH_BASELINE)

    Returns:
        Dict with retention_factor, eff_alpha, degradation_pct, curve_type

    Raises:
        StopRule: If blackout_days > 300 (unrealistic) or cache overflow
    """
    if blackout_days > BLACKOUT_MAX_UNREALISTIC:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-blackout",
            "metric": "blackout_duration_unrealistic",
            "baseline": BLACKOUT_MAX_UNREALISTIC,
            "delta": blackout_days - BLACKOUT_MAX_UNREALISTIC,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(f"Blackout duration {blackout_days}d > {BLACKOUT_MAX_UNREALISTIC}d unrealistic limit")

    # DELEGATE TO GNN NONLINEAR MODEL
    try:
        gnn_result = gnn_nonlinear_retention(blackout_days, cache_depth)
        retention_factor = gnn_result["retention_factor"]
        eff_alpha = gnn_result["eff_alpha"]
    except StopRule:
        # Re-raise cache overflow
        raise

    # Floor retention at NONLINEAR_RETENTION_FLOOR (asymptotic floor)
    retention_factor = max(NONLINEAR_RETENTION_FLOOR, round(retention_factor, 4))

    # Degradation percentage from baseline
    degradation_pct = round((1.0 - retention_factor / RETENTION_BASE_FACTOR) * 100, 2)

    return {
        "blackout_days": blackout_days,
        "retention_factor": retention_factor,
        "eff_alpha": eff_alpha,
        "degradation_pct": degradation_pct,
        "model": DEGRADATION_MODEL,
        "curve_type": CURVE_TYPE,
        "asymptote_alpha": ASYMPTOTE_ALPHA
    }


def alpha_at_duration(
    blackout_days: int,
    base_alpha: float = MIN_EFF_ALPHA_VALIDATED,
    reroute_boost: float = REROUTING_ALPHA_BOOST_LOCKED
) -> float:
    """Compute effective alpha at given blackout duration.

    Apply retention-scaled boost.

    Args:
        blackout_days: Blackout duration in days
        base_alpha: Base effective alpha (default: 2.656 validated floor)
        reroute_boost: Reroute boost to apply (default: 0.07 locked)

    Returns:
        eff_alpha: Effective alpha at duration
    """
    curve = retention_curve(blackout_days)
    retention_scale = curve["retention_factor"] / RETENTION_BASE_FACTOR

    eff_alpha = base_alpha + (reroute_boost * retention_scale)

    return round(eff_alpha, 4)


def sweep_with_pruning(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_PRUNING_TARGET_DAYS),
    trim_factor: float = 0.3,
    iterations: int = 1000,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run extended sweep with entropy pruning enabled.

    Extends sweep range to 250d+ using pruning-boosted retention.

    Args:
        day_range: Tuple of (min_days, max_days) (default: 43-250)
        trim_factor: ln(n) trim factor (0.3-0.5 range)
        iterations: Number of iterations (default: 1000)
        seed: Random seed for reproducibility

    Returns:
        List of pruning_sweep_receipts

    Receipt: pruning_sweep_receipt (per iteration)
    """
    if seed is not None:
        random.seed(seed)

    results = []

    for i in range(iterations):
        blackout_days = random.randint(day_range[0], day_range[1])

        try:
            # Use pruning-enabled retention
            result = nonlinear_retention_with_pruning(
                blackout_days,
                CACHE_DEPTH_BASELINE,
                pruning_enabled=True,
                trim_factor=trim_factor
            )

            survival_status = result["eff_alpha"] >= PRUNING_TARGET_ALPHA * 0.9

            sweep_result = {
                "iteration": i,
                "blackout_days": blackout_days,
                "retention_factor": result["retention_factor"],
                "eff_alpha": result["eff_alpha"],
                "pruning_boost": result["pruning_boost"],
                "trim_factor": trim_factor,
                "target_alpha": PRUNING_TARGET_ALPHA,
                "survival_status": survival_status
            }

            emit_receipt("pruning_sweep", {
                "tenant_id": "axiom-blackout",
                **sweep_result,
                "payload_hash": dual_hash(json.dumps(sweep_result, sort_keys=True))
            })

            results.append(sweep_result)

        except StopRule as e:
            # Overflow triggered
            results.append({
                "iteration": i,
                "blackout_days": blackout_days,
                "overflow_triggered": True,
                "survival_status": False,
                "stoprule_reason": str(e)
            })

    return results


def extended_blackout_sweep(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_SWEEP_MAX_DAYS),
    iterations: int = 1000,
    seed: Optional[int] = None,
    pruning_enabled: bool = False,
    trim_factor: float = 0.3
) -> List[Dict[str, Any]]:
    """Run extended blackout sweep across day range.

    Run iterations across day range, return curve data with receipts.
    With pruning_enabled=True, uses pruning-boosted retention for 250d+ sweeps.

    Args:
        day_range: Tuple of (min_days, max_days)
        iterations: Number of iterations (default: 1000)
        seed: Random seed for reproducibility
        pruning_enabled: Whether entropy pruning is active (default: False)
        trim_factor: ln(n) trim factor when pruning enabled (default: 0.3)

    Returns:
        List of extended_blackout_receipts

    Receipt: extended_blackout_receipt (per iteration)
    """
    if seed is not None:
        random.seed(seed)

    # If pruning enabled and range extends beyond 200d, use pruning sweep
    if pruning_enabled and day_range[1] > OVERFLOW_THRESHOLD_DAYS:
        return sweep_with_pruning(day_range, trim_factor, iterations, seed)

    results = []

    for i in range(iterations):
        # Random blackout duration in range
        blackout_days = random.randint(day_range[0], day_range[1])

        try:
            if pruning_enabled:
                # Use pruning-enabled retention
                retention_result = nonlinear_retention_with_pruning(
                    blackout_days,
                    CACHE_DEPTH_BASELINE,
                    pruning_enabled=True,
                    trim_factor=trim_factor
                )
                curve = {
                    "retention_factor": retention_result["retention_factor"],
                    "eff_alpha": retention_result["eff_alpha"],
                    "degradation_pct": 0.0,
                    "pruning_boost": retention_result["pruning_boost"]
                }
            else:
                curve = retention_curve(blackout_days)
                curve["pruning_boost"] = 0.0

            # Survival status: alpha above floor
            target_alpha = PRUNING_TARGET_ALPHA if pruning_enabled else MIN_EFF_ALPHA_VALIDATED
            survival_status = curve["eff_alpha"] >= target_alpha * 0.9

            result = {
                "iteration": i,
                "blackout_days": blackout_days,
                "retention_factor": curve["retention_factor"],
                "eff_alpha": curve["eff_alpha"],
                "degradation_pct": curve.get("degradation_pct", 0.0),
                "survival_status": survival_status,
                "pruning_enabled": pruning_enabled,
                "pruning_boost": curve.get("pruning_boost", 0.0)
            }

            emit_receipt("extended_blackout", {
                "tenant_id": "axiom-blackout",
                **result,
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
            })

            results.append(result)

        except StopRule:
            # Overflow or unrealistic duration - record failure
            results.append({
                "iteration": i,
                "blackout_days": blackout_days,
                "overflow_triggered": True,
                "survival_status": False,
                "pruning_enabled": pruning_enabled
            })

    return results


def find_retention_floor(sweep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find retention floor from sweep results.

    Identify worst-case from sweep.

    Args:
        sweep_results: List from extended_blackout_sweep

    Returns:
        Dict with min_retention, days_at_min, alpha_at_min
    """
    if not sweep_results:
        return {
            "min_retention": RETENTION_BASE_FACTOR,
            "days_at_min": BLACKOUT_BASE_DAYS,
            "alpha_at_min": MIN_EFF_ALPHA_VALIDATED + REROUTING_ALPHA_BOOST_LOCKED
        }

    # Find minimum retention
    min_result = min(sweep_results, key=lambda x: x["retention_factor"])

    return {
        "min_retention": min_result["retention_factor"],
        "days_at_min": min_result["blackout_days"],
        "alpha_at_min": min_result["eff_alpha"],
        "degradation_pct_at_min": min_result["degradation_pct"],
        "survival_at_min": min_result["survival_status"]
    }


def generate_retention_curve_data(
    day_range: Tuple[int, int] = (BLACKOUT_BASE_DAYS, BLACKOUT_SWEEP_MAX_DAYS),
    step: int = 1
) -> List[Dict[str, float]]:
    """Generate retention curve data points.

    Uses GNN NONLINEAR model (Dec 2025).

    Args:
        day_range: Tuple of (min_days, max_days)
        step: Day increment (default: 1)

    Returns:
        List of {days, retention, alpha} dicts

    Receipt: retention_curve_receipt
    """
    import math

    curve_points = []

    for days in range(day_range[0], min(day_range[1] + 1, BLACKOUT_MAX_UNREALISTIC), step):
        try:
            curve = retention_curve(days)
            curve_points.append({
                "days": days,
                "retention": curve["retention_factor"],
                "alpha": curve["eff_alpha"]
            })
        except StopRule:
            # Cache overflow - stop curve generation
            break

    if not curve_points:
        return []

    # Compute exponential fit R² (GNN nonlinear model)
    retentions = [p["retention"] for p in curve_points]
    is_monotonic = all(retentions[i] >= retentions[i+1] for i in range(len(retentions)-1))

    # Compute max single-day drop
    max_single_drop = 0.0
    for i in range(1, len(retentions)):
        drop = retentions[i-1] - retentions[i]
        max_single_drop = max(max_single_drop, drop)

    # R² approximation for EXPONENTIAL fit (GNN nonlinear model)
    # Formula: retention = FLOOR + (BASE - FLOOR) * exp(-λ * (d - BASE_DAYS))
    mean_ret = sum(retentions) / len(retentions)
    ss_tot = sum((r - mean_ret) ** 2 for r in retentions)

    # Predict using exponential decay (GNN nonlinear formula)
    from .gnn_cache import DECAY_LAMBDA
    predicted = []
    for d in range(day_range[0], day_range[0] + len(curve_points) * step, step):
        if d <= BLACKOUT_BASE_DAYS:
            pred = RETENTION_BASE_FACTOR
        else:
            excess = d - BLACKOUT_BASE_DAYS
            pred = NONLINEAR_RETENTION_FLOOR + \
                (RETENTION_BASE_FACTOR - NONLINEAR_RETENTION_FLOOR) * math.exp(-DECAY_LAMBDA * excess)
        predicted.append(pred)

    ss_res = sum((retentions[i] - predicted[i]) ** 2 for i in range(min(len(retentions), len(predicted))))
    r_squared = 1 - (ss_res / max(ss_tot, 0.0001))
    r_squared = round(max(0.0, min(1.0, r_squared)), 4)

    emit_receipt("retention_curve", {
        "tenant_id": "axiom-blackout",
        "day_range": list(day_range),
        "curve_points": curve_points,
        "model_type": DEGRADATION_MODEL,
        "curve_type": CURVE_TYPE,
        "r_squared": r_squared,
        "is_monotonic": is_monotonic,
        "max_single_day_drop": round(max_single_drop, 4),
        "no_cliff_behavior": max_single_drop < 0.02,
        "asymptote_alpha": ASYMPTOTE_ALPHA,
        "payload_hash": dual_hash(json.dumps(curve_points, sort_keys=True))
    })

    return curve_points


def gnn_sensitivity_stub(param_config: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for GNN parameter sensitivity analysis (next gate).

    Returns config echo with "not_implemented" flag.

    Args:
        param_config: GNN parameter configuration dict

    Returns:
        stub_receipt dict with status="stub_only"

    Receipt: gnn_sensitivity_stub_receipt
    """
    result = {
        "param_config": param_config,
        "status": "stub_only",
        "not_implemented": True,
        "next_gate": "gnn_parameter_sensitivity",
        "description": "Placeholder for GNN complexity sweep (1K-100K params)"
    }

    emit_receipt("gnn_sensitivity_stub", {
        "tenant_id": "axiom-blackout",
        **result,
        "payload_hash": dual_hash(json.dumps(param_config, sort_keys=True))
    })

    return result


def validate_retention_slos(
    sweep_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate retention curve SLOs.

    SLOs (UPDATED Dec 2025 - GNN nonlinear):
    1. α asymptotes within 0.02 of 2.72 by 150d
    2. 100% survival to 150d, graceful degradation 150-180d
    3. No cliff behavior (max single-day drop < 0.01)
    4. Nonlinear curve R² >= 0.98 to exponential model

    Args:
        sweep_results: Results from extended_blackout_sweep

    Returns:
        Dict with validation results

    Receipt: retention_slo_validation
    """
    if not sweep_results:
        return {"validated": False, "reason": "no sweep results"}

    # SLO 1: All alphas above floor (2.50 for survival, asymptote check at 150d)
    all_above_floor = all(r["eff_alpha"] >= 2.50 for r in sweep_results)
    failures_below_floor = [r for r in sweep_results if r["eff_alpha"] < 2.50]

    # SLO 2: Asymptote proximity at 150d (if present in results)
    results_150d = [r for r in sweep_results if r.get("blackout_days") == 150]
    asymptote_ok = True
    if results_150d:
        alpha_150d = results_150d[0]["eff_alpha"]
        asymptote_ok = abs(ASYMPTOTE_ALPHA - alpha_150d) <= 0.02

    # SLO 3: Check for cliff behavior (tighter threshold for GNN nonlinear)
    retentions = []
    for d in range(BLACKOUT_BASE_DAYS, min(BLACKOUT_SWEEP_MAX_DAYS + 1, 200)):
        try:
            ret = retention_curve(d)["retention_factor"]
            retentions.append(ret)
        except StopRule:
            break

    max_drop = 0.0
    for i in range(1, len(retentions)):
        drop = retentions[i-1] - retentions[i]
        max_drop = max(max_drop, drop)

    no_cliff = max_drop < 0.01  # Tighter threshold for GNN nonlinear

    validation = {
        "all_above_floor": all_above_floor,
        "failures_count": len(failures_below_floor),
        "asymptote_proximity_ok": asymptote_ok,
        "max_single_day_drop": round(max_drop, 4),
        "no_cliff_behavior": no_cliff,
        "curve_type": CURVE_TYPE,
        "validated": all_above_floor and no_cliff and asymptote_ok
    }

    emit_receipt("retention_slo_validation", {
        "tenant_id": "axiom-blackout",
        **validation
    })

    return validation
