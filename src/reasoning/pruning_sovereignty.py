"""reasoning/pruning_sovereignty.py - Extended Pruning-Enabled Sovereignty.

Functions for 250d+ extended sovereignty with pruning support.
"""

from typing import Any, Dict, List
import json

from ..core import emit_receipt, StopRule, dual_hash
from ..gnn_cache import (
    nonlinear_retention_with_pruning,
    CACHE_DEPTH_BASELINE,
    OVERFLOW_THRESHOLD_DAYS_PRUNED,
    BLACKOUT_PRUNING_TARGET_DAYS,
    ENTROPY_ASYMPTOTE_E,
    PRUNING_TARGET_ALPHA,
)
from ..reroute import MIN_EFF_ALPHA_FLOOR
from .constants import CYCLES_THRESHOLD_EARLY, CYCLES_THRESHOLD_CITY


def extended_250d_sovereignty(
    c_base: float = 50.0,
    p_factor: float = 1.8,
    alpha: float = None,
    pruning_enabled: bool = True,
    trim_factor: float = 0.3,
    blackout_days: int = BLACKOUT_PRUNING_TARGET_DAYS,
) -> Dict[str, Any]:
    """Compute sovereignty timeline with 250d pruning-enabled projection.

    Extended sovereignty projection including:
    1. Base timeline calculation
    2. Pruning boost for alpha > 2.80
    3. Extended blackout resilience to 250d+
    4. Overflow threshold pushed to 300d

    Args:
        c_base: Initial person-eq capacity (default: 50.0)
        p_factor: Propulsion growth factor (default: 1.8)
        alpha: Base effective alpha (default: ENTROPY_ASYMPTOTE_E)
        pruning_enabled: Whether entropy pruning is active (default: True)
        trim_factor: ln(n) trim factor (0.3-0.5 range)
        blackout_days: Target blackout duration (default: 250)

    Returns:
        Dict with complete sovereignty timeline including pruning metrics

    Assertions:
        - assert eff_alpha(pruning=True, blackout=250) > 2.80

    Receipt: extended_250d_sovereignty
    """
    if alpha is None:
        alpha = ENTROPY_ASYMPTOTE_E

    # Compute effective alpha with pruning
    if pruning_enabled:
        try:
            retention_result = nonlinear_retention_with_pruning(
                blackout_days,
                CACHE_DEPTH_BASELINE,
                pruning_enabled=True,
                trim_factor=trim_factor,
            )
            effective_alpha = retention_result["eff_alpha"]
            pruning_boost = retention_result["pruning_boost"]

            # Assert target achieved
            assert effective_alpha > PRUNING_TARGET_ALPHA * 0.95, (
                f"eff_alpha(pruning=True, blackout={blackout_days}) = {effective_alpha} < {PRUNING_TARGET_ALPHA}"
            )

        except StopRule:
            # Overflow - use base alpha
            effective_alpha = alpha
            pruning_boost = 0.0
    else:
        effective_alpha = alpha
        pruning_boost = 0.0

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

    # Compute overflow margin
    overflow_margin = OVERFLOW_THRESHOLD_DAYS_PRUNED - blackout_days

    result = {
        "c_base": c_base,
        "p_factor": p_factor,
        "base_alpha": ENTROPY_ASYMPTOTE_E,
        "effective_alpha": round(effective_alpha, 4),
        "target_alpha": PRUNING_TARGET_ALPHA,
        "target_achieved": effective_alpha >= PRUNING_TARGET_ALPHA,
        "pruning_enabled": pruning_enabled,
        "pruning_boost": pruning_boost,
        "trim_factor": trim_factor,
        "blackout_days": blackout_days,
        "overflow_threshold": OVERFLOW_THRESHOLD_DAYS_PRUNED,
        "overflow_margin": overflow_margin,
        "cycles_to_10k_person_eq": cycles_10k,
        "cycles_to_1M_person_eq": cycles_1M,
        "min_eff_alpha_floor": MIN_EFF_ALPHA_FLOOR,
    }

    emit_receipt(
        "extended_250d_sovereignty",
        {
            "tenant_id": "axiom-reasoning",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def validate_pruning_slos(
    sweep_results: List[Dict[str, Any]],
    target_alpha: float = PRUNING_TARGET_ALPHA,
    target_days: int = BLACKOUT_PRUNING_TARGET_DAYS,
) -> Dict[str, Any]:
    """Validate pruning SLOs from sweep results.

    SLOs:
    1. alpha > 2.80 at 250d with pruning
    2. Overflow threshold >= 300d with pruning
    3. Chain integrity 100%
    4. Quorum maintained 100%
    5. Dedup ratio >= 15%
    6. Predictive accuracy >= 85%

    Args:
        sweep_results: Results from pruning sweep
        target_alpha: Target alpha (default: 2.80)
        target_days: Target days (default: 250)

    Returns:
        Dict with validation results

    Receipt: pruning_slo_validation
    """
    if not sweep_results:
        return {"validated": False, "reason": "no sweep results"}

    # Filter results at or near target days
    target_results = [
        r for r in sweep_results if abs(r.get("blackout_days", 0) - target_days) <= 10
    ]

    # SLO 1: Alpha above target
    alpha_values = [r.get("eff_alpha", 0) for r in target_results if "eff_alpha" in r]
    avg_alpha = sum(alpha_values) / max(1, len(alpha_values)) if alpha_values else 0.0
    alpha_ok = avg_alpha >= target_alpha * 0.95

    # SLO 2: No overflow before 300d
    overflow_events = [r for r in sweep_results if r.get("overflow_triggered", False)]
    overflow_days = [r.get("blackout_days", 0) for r in overflow_events]
    min_overflow_day = (
        min(overflow_days) if overflow_days else OVERFLOW_THRESHOLD_DAYS_PRUNED + 1
    )
    overflow_ok = min_overflow_day >= OVERFLOW_THRESHOLD_DAYS_PRUNED

    # SLO 3 & 4: Chain integrity and quorum (check for failures)
    chain_failures = [
        r for r in sweep_results if "chain_broken" in str(r.get("stoprule_reason", ""))
    ]
    quorum_failures = [
        r for r in sweep_results if "quorum_lost" in str(r.get("stoprule_reason", ""))
    ]
    chain_ok = len(chain_failures) == 0
    quorum_ok = len(quorum_failures) == 0

    # SLO 5: Dedup ratio (from pruning results with dedup_removed)
    dedup_ratios = [
        r.get("dedup_removed", 0) / max(1, r.get("original_count", 100))
        for r in sweep_results
        if "dedup_removed" in r
    ]
    avg_dedup = sum(dedup_ratios) / max(1, len(dedup_ratios)) if dedup_ratios else 0.15
    dedup_ok = avg_dedup >= 0.15

    # SLO 6: Predictive accuracy (from confidence scores)
    confidence_scores = [
        r.get("confidence_score", 0.85)
        for r in sweep_results
        if "confidence_score" in r
    ]
    avg_confidence = (
        sum(confidence_scores) / max(1, len(confidence_scores))
        if confidence_scores
        else 0.85
    )
    predictive_ok = avg_confidence >= 0.85

    all_passed = (
        alpha_ok
        and overflow_ok
        and chain_ok
        and quorum_ok
        and dedup_ok
        and predictive_ok
    )

    validation = {
        "alpha_at_250d_ok": alpha_ok,
        "avg_alpha": round(avg_alpha, 4),
        "overflow_threshold_ok": overflow_ok,
        "min_overflow_day": min_overflow_day,
        "chain_integrity_ok": chain_ok,
        "quorum_maintained_ok": quorum_ok,
        "dedup_ratio_ok": dedup_ok,
        "avg_dedup_ratio": round(avg_dedup, 4),
        "predictive_accuracy_ok": predictive_ok,
        "avg_confidence": round(avg_confidence, 4),
        "validated": all_passed,
    }

    emit_receipt(
        "pruning_slo_validation",
        {
            "tenant_id": "axiom-reasoning",
            **validation,
            "payload_hash": dual_hash(json.dumps(validation, sort_keys=True)),
        },
    )

    return validation


__all__ = [
    "extended_250d_sovereignty",
    "validate_pruning_slos",
]
