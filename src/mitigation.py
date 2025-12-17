"""mitigation.py - Partition Resilience in Mitigation Stack

Incorporates partition tolerance into the overall mitigation scoring
and weights mitigation by quorum health.

THE PHYSICS:
    - PARTITION_MITIGATION_FACTOR = 0.05 (max expected α drop)
    - Quorum health weight: 1.0 if intact, degraded otherwise
    - Combined mitigation includes τ-penalty + partition + quorum + reroute
    - Duration-dependent degradation for extended blackouts (43-90d)

REROUTE INTEGRATION (Dec 2025 adaptive rerouting):
    - REROUTING_ALPHA_BOOST_LOCKED = 0.07 (validated, immutable)
    - Reroute boost applied multiplicatively in mitigation stack
    - Blackout factor scales mitigation by duration (graceful degradation beyond 43d)

EXTENDED BLACKOUT DEGRADATION (Dec 2025):
    - DEGRADATION_RATE = 0.0032/day (calibrated: 1.4 @ 43d → 1.25 @ 90d)
    - Duration degradation applied to mitigation stack when blackout_days > 43
    - Linear model, no cliff behavior

Source: Grok - "Variable partitions (e.g., 40%)", "quorum intact", "+0.07 to 2.7+"
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .core import emit_receipt
from .partition import (
    partition_sim,
    quorum_check,
    NODE_BASELINE,
    QUORUM_THRESHOLD,
    BASE_ALPHA,
    ALPHA_DROP_FACTOR,
    PARTITION_MAX_TEST_PCT
)
from .ledger import (
    LEDGER_ALPHA_BOOST_VALIDATED,
    apply_quorum_factor
)


# === CONSTANTS ===

PARTITION_MITIGATION_FACTOR = 0.05
"""Maximum expected α drop from partition (at 40% loss)."""

QUORUM_HEALTH_FULL = 1.0
"""Full quorum weight (all nodes operational)."""

QUORUM_HEALTH_DEGRADED_BASE = 0.9
"""Base weight for degraded quorum."""

QUORUM_HEALTH_DEGRADED_PER_NODE = 0.05
"""Additional degradation per missing node."""

TAU_MITIGATION_FACTOR = 0.889
"""Receipt-based τ-penalty mitigation factor (from receipt_params.json)."""

REROUTING_ALPHA_BOOST_LOCKED = 0.07
"""physics: LOCKED. Validated reroute boost. 2.656 + 0.07 = 2.726"""

# Backward compatibility alias
REROUTE_ALPHA_BOOST = REROUTING_ALPHA_BOOST_LOCKED

BLACKOUT_BASE_DAYS = 43
"""physics: Mars solar conjunction maximum duration in days."""

BLACKOUT_EXTENDED_DAYS = 60
"""physics: Extended blackout tolerance with reroute (43d * 1.4 retention)."""

BLACKOUT_SWEEP_MAX_DAYS = 90
"""physics: Extreme stress bound (2× conjunction duration)."""

DEGRADATION_RATE = 0.0032
"""physics: Per-day degradation rate beyond 43d (calibrated: 1.4 @ 43d → 1.25 @ 90d)."""

RETENTION_BASE_FACTOR = 1.4
"""physics: Baseline retention factor at 43d blackout."""


@dataclass
class MitigationScore:
    """Combined mitigation score.

    Attributes:
        partition_score: Partition tolerance score (0-1)
        quorum_score: Quorum health score (0-1)
        tau_score: τ-penalty mitigation score (0-1)
        reroute_score: Reroute mitigation score (0-1)
        combined_score: Weighted combination
        effective_alpha: Final effective α after all mitigations
    """
    partition_score: float
    quorum_score: float
    tau_score: float
    reroute_score: float
    combined_score: float
    effective_alpha: float


def compute_partition_tolerance(
    loss_pct: float = PARTITION_MAX_TEST_PCT,
    base_alpha: float = BASE_ALPHA
) -> float:
    """Compute partition tolerance score.

    Score = 1.0 - (actual_drop / max_expected_drop)
    At 40% loss with drop ~0.05, score ≈ 0.0 (at limit)
    At 0% loss, score = 1.0 (full tolerance)

    Args:
        loss_pct: Partition loss percentage (0-1)
        base_alpha: Baseline α (default: 2.68)

    Returns:
        Partition tolerance score (0-1)
    """
    if loss_pct <= 0:
        return 1.0

    try:
        result = partition_sim(NODE_BASELINE, loss_pct, base_alpha, emit=False)
        actual_drop = result["eff_alpha_drop"]

        # Score is inverse of drop ratio
        # If actual_drop == PARTITION_MITIGATION_FACTOR, score = 0
        # If actual_drop == 0, score = 1
        drop_ratio = actual_drop / PARTITION_MITIGATION_FACTOR
        score = max(0.0, 1.0 - drop_ratio)

        return round(score, 4)
    except Exception:
        # Quorum failed
        return 0.0


def compute_quorum_health(
    nodes_surviving: int,
    nodes_baseline: int = NODE_BASELINE
) -> float:
    """Compute quorum health weight.

    Weight = 1.0 if full quorum
    Weight = DEGRADED_BASE - (missing * DEGRADED_PER_NODE) if degraded

    Args:
        nodes_surviving: Number of operational nodes
        nodes_baseline: Expected baseline (default: 5)

    Returns:
        Quorum health weight (0-1)
    """
    if nodes_surviving >= nodes_baseline:
        return QUORUM_HEALTH_FULL

    if nodes_surviving < QUORUM_THRESHOLD:
        return 0.0

    # Degraded quorum
    nodes_missing = nodes_baseline - nodes_surviving
    weight = QUORUM_HEALTH_DEGRADED_BASE - (nodes_missing * QUORUM_HEALTH_DEGRADED_PER_NODE)

    return max(0.0, round(weight, 4))


def compute_tau_mitigation(
    receipt_integrity: float = 0.9,
    tau_factor: float = TAU_MITIGATION_FACTOR
) -> float:
    """Compute τ-penalty mitigation score.

    Score based on receipt coverage and mitigation factor.
    At 90% receipt integrity, mitigation ≈ 88.9%

    Args:
        receipt_integrity: Receipt coverage (0-1)
        tau_factor: Base mitigation factor (default: 0.889)

    Returns:
        τ-penalty mitigation score (0-1)
    """
    return round(receipt_integrity * tau_factor, 4)


def compute_reroute_mitigation(
    reroute_enabled: bool = False,
    reroute_result: Optional[Dict[str, Any]] = None
) -> float:
    """Compute reroute mitigation score.

    Score based on reroute effectiveness and recovery factor.

    Args:
        reroute_enabled: Whether adaptive rerouting is active
        reroute_result: Result from adaptive_reroute (optional)

    Returns:
        Reroute mitigation score (0-1)

    Receipt: reroute_mitigation
    """
    if not reroute_enabled:
        return 0.0

    if reroute_result is None:
        # Default score when reroute is enabled but no specific result
        score = 0.7  # Conservative default
    else:
        recovery_factor = reroute_result.get("recovery_factor", 0.0)
        quorum_preserved = reroute_result.get("quorum_preserved", False)

        if not quorum_preserved:
            score = 0.0
        else:
            # Score based on recovery factor
            score = recovery_factor * 0.9 + 0.1  # 0.1 base for quorum preserved

    emit_receipt("reroute_mitigation", {
        "tenant_id": "axiom-mitigation",
        "reroute_enabled": reroute_enabled,
        "score": round(score, 4),
        "recovery_factor": reroute_result.get("recovery_factor", 0.0) if reroute_result else 0.0
    })

    return round(score, 4)


def compute_blackout_factor(
    blackout_days: int = 0,
    reroute_enabled: bool = False
) -> float:
    """Compute blackout factor for graceful degradation.

    Factor degrades gracefully beyond base blackout duration.
    With reroute enabled, can extend tolerance from 43d to 60d+.

    Args:
        blackout_days: Current blackout duration in days
        reroute_enabled: Whether adaptive rerouting is active

    Returns:
        Blackout factor (0-1), where 1.0 = no degradation

    Receipt: blackout_factor
    """
    if blackout_days == 0:
        factor = 1.0
    elif not reroute_enabled:
        # Without reroute, degradation starts immediately
        if blackout_days <= BLACKOUT_BASE_DAYS:
            factor = 1.0 - (blackout_days / BLACKOUT_BASE_DAYS) * 0.3
        else:
            # Beyond base, severe degradation
            excess = blackout_days - BLACKOUT_BASE_DAYS
            factor = 0.7 - min(0.5, excess * 0.02)
    else:
        # With reroute, extended tolerance
        if blackout_days <= BLACKOUT_BASE_DAYS:
            # Minimal degradation within base period
            factor = 1.0 - (blackout_days / BLACKOUT_BASE_DAYS) * 0.1
        elif blackout_days <= BLACKOUT_EXTENDED_DAYS:
            # Gradual degradation in extended period
            excess = blackout_days - BLACKOUT_BASE_DAYS
            max_excess = BLACKOUT_EXTENDED_DAYS - BLACKOUT_BASE_DAYS
            factor = 0.9 - (excess / max_excess) * 0.2
        else:
            # Beyond extended, moderate degradation
            excess = blackout_days - BLACKOUT_EXTENDED_DAYS
            factor = 0.7 - min(0.4, excess * 0.01)

    emit_receipt("blackout_factor", {
        "tenant_id": "axiom-mitigation",
        "blackout_days": blackout_days,
        "reroute_enabled": reroute_enabled,
        "factor": round(max(0.0, factor), 4),
        "blackout_base_days": BLACKOUT_BASE_DAYS,
        "blackout_extended_days": BLACKOUT_EXTENDED_DAYS
    })

    return round(max(0.0, factor), 4)


def apply_duration_degradation(
    base_mitigation: float,
    blackout_days: int
) -> Dict[str, Any]:
    """Apply duration-dependent degradation to mitigation stack.

    Scales mitigation by retention factor when blackout_days > 43.
    Uses linear degradation model with DEGRADATION_RATE = 0.0032/day.

    Formula: degraded_mitigation = base_mitigation * retention_factor
    Where: retention_factor = RETENTION_BASE * (1 - (days - 43) * DEGRADATION_RATE)

    Args:
        base_mitigation: Base mitigation value (0-1)
        blackout_days: Current blackout duration in days

    Returns:
        Dict with degraded_mitigation, retention_factor, degradation_pct

    Receipt: duration_degradation
    """
    if blackout_days <= BLACKOUT_BASE_DAYS:
        retention_factor = RETENTION_BASE_FACTOR
        degradation_pct = 0.0
    else:
        # Linear degradation beyond base
        excess_days = blackout_days - BLACKOUT_BASE_DAYS
        degradation = excess_days * DEGRADATION_RATE

        # Retention formula: retention = RETENTION_BASE * (1 - degradation / RETENTION_BASE)
        retention_factor = RETENTION_BASE_FACTOR - degradation
        retention_factor = max(1.0, round(retention_factor, 4))  # Floor at 1.0

        # Degradation percentage
        degradation_pct = round((1.0 - retention_factor / RETENTION_BASE_FACTOR) * 100, 2)

    # Scale mitigation by retention factor (normalized to base)
    retention_scale = retention_factor / RETENTION_BASE_FACTOR
    degraded_mitigation = base_mitigation * retention_scale

    result = {
        "base_mitigation": round(base_mitigation, 4),
        "blackout_days": blackout_days,
        "retention_factor": retention_factor,
        "retention_scale": round(retention_scale, 4),
        "degradation_pct": degradation_pct,
        "degraded_mitigation": round(degraded_mitigation, 4)
    }

    emit_receipt("duration_degradation", {
        "tenant_id": "axiom-mitigation",
        **result
    })

    return result


def apply_reroute_mitigation(
    base_mitigation: MitigationScore,
    reroute_result: Dict[str, Any],
    blackout_days: int = 0
) -> Dict[str, Any]:
    """Apply reroute boost multiplicatively to mitigation stack.

    Enhances base mitigation with reroute boost and blackout factor.

    Args:
        base_mitigation: Base MitigationScore to enhance
        reroute_result: Result from adaptive_reroute
        blackout_days: Current blackout duration in days

    Returns:
        Dict with enhanced mitigation including +0.07 boost

    Receipt: reroute_enhanced_mitigation
    """
    # Get alpha boost from reroute result
    alpha_boost = reroute_result.get("alpha_boost", 0.0)
    recovery_factor = reroute_result.get("recovery_factor", 0.0)
    quorum_preserved = reroute_result.get("quorum_preserved", False)

    # Compute blackout factor
    blackout_factor = compute_blackout_factor(blackout_days, reroute_enabled=True)

    # Apply boost multiplicatively
    if quorum_preserved and recovery_factor > 0.5:
        effective_boost = alpha_boost * blackout_factor
        enhanced_alpha = base_mitigation.effective_alpha + effective_boost
        enhanced_combined = min(1.0, base_mitigation.combined_score + recovery_factor * 0.1)
    else:
        effective_boost = 0.0
        enhanced_alpha = base_mitigation.effective_alpha
        enhanced_combined = base_mitigation.combined_score

    result = {
        "base_effective_alpha": base_mitigation.effective_alpha,
        "reroute_alpha_boost": alpha_boost,
        "blackout_factor": blackout_factor,
        "effective_boost": round(effective_boost, 4),
        "enhanced_alpha": round(enhanced_alpha, 4),
        "base_combined_score": base_mitigation.combined_score,
        "enhanced_combined_score": round(enhanced_combined, 4),
        "recovery_factor": recovery_factor,
        "quorum_preserved": quorum_preserved,
        "blackout_days": blackout_days
    }

    emit_receipt("reroute_enhanced_mitigation", {
        "tenant_id": "axiom-mitigation",
        **result
    })

    return result


def compute_mitigation_score(
    loss_pct: float = 0.0,
    nodes_surviving: Optional[int] = None,
    receipt_integrity: float = 0.9,
    base_alpha: float = BASE_ALPHA,
    weights: Optional[Dict[str, float]] = None,
    reroute_enabled: bool = False,
    reroute_result: Optional[Dict[str, Any]] = None,
    blackout_days: int = 0
) -> MitigationScore:
    """Compute combined mitigation score.

    Combines partition tolerance, quorum health, τ-penalty mitigation,
    and reroute mitigation into a single weighted score.

    Args:
        loss_pct: Partition loss percentage (0-1)
        nodes_surviving: Operational node count (default: baseline)
        receipt_integrity: Receipt coverage (default: 0.9)
        base_alpha: Baseline α (default: 2.68)
        weights: Optional weight overrides (default: equal weights)
        reroute_enabled: Whether adaptive rerouting is active
        reroute_result: Result from adaptive_reroute (optional)
        blackout_days: Current blackout duration in days

    Returns:
        MitigationScore with all components

    Receipt: mitigation_score
    """
    if nodes_surviving is None:
        nodes_surviving = NODE_BASELINE

    if weights is None:
        if reroute_enabled:
            weights = {"partition": 0.25, "quorum": 0.25, "tau": 0.25, "reroute": 0.25}
        else:
            weights = {"partition": 0.33, "quorum": 0.34, "tau": 0.33, "reroute": 0.0}

    # Compute individual scores
    partition_score = compute_partition_tolerance(loss_pct, base_alpha)
    quorum_score = compute_quorum_health(nodes_surviving, NODE_BASELINE)
    tau_score = compute_tau_mitigation(receipt_integrity)
    reroute_score = compute_reroute_mitigation(reroute_enabled, reroute_result)

    # Weighted combination
    combined = (
        partition_score * weights.get("partition", 0.33) +
        quorum_score * weights.get("quorum", 0.34) +
        tau_score * weights.get("tau", 0.33) +
        reroute_score * weights.get("reroute", 0.0)
    )
    combined = round(combined, 4)

    # Compute effective alpha with all factors
    try:
        partition_result = partition_sim(NODE_BASELINE, loss_pct, base_alpha, emit=False, reroute_enabled=reroute_enabled)
        eff_alpha = partition_result["eff_alpha"]

        # Apply quorum factor
        quorum_degradation = (NODE_BASELINE - nodes_surviving) * 0.02
        eff_alpha = eff_alpha - quorum_degradation

        # Apply τ-mitigation boost
        tau_boost = (1.0 - tau_score) * 0.1  # Up to 10% recovery
        eff_alpha = eff_alpha + tau_boost

        # Apply blackout factor if applicable
        if blackout_days > 0:
            blackout_factor = compute_blackout_factor(blackout_days, reroute_enabled)
            eff_alpha = eff_alpha * blackout_factor
    except Exception:
        eff_alpha = 0.0

    score = MitigationScore(
        partition_score=partition_score,
        quorum_score=quorum_score,
        tau_score=tau_score,
        reroute_score=reroute_score,
        combined_score=combined,
        effective_alpha=round(eff_alpha, 4)
    )

    emit_receipt("mitigation_score", {
        "tenant_id": "axiom-mitigation",
        "loss_pct": loss_pct,
        "nodes_surviving": nodes_surviving,
        "receipt_integrity": receipt_integrity,
        "partition_score": partition_score,
        "quorum_score": quorum_score,
        "tau_score": tau_score,
        "reroute_score": reroute_score,
        "combined_score": combined,
        "effective_alpha": score.effective_alpha,
        "weights": weights,
        "reroute_enabled": reroute_enabled,
        "blackout_days": blackout_days
    })

    return score


def apply_mitigation_to_projection(
    base_projection: Dict[str, Any],
    mitigation: MitigationScore
) -> Dict[str, Any]:
    """Apply mitigation score to sovereignty projection.

    Adjusts timeline based on combined mitigation effectiveness.

    Args:
        base_projection: Base projection dict
        mitigation: MitigationScore to apply

    Returns:
        Adjusted projection with mitigation applied

    Receipt: mitigated_projection
    """
    base_cycles = base_projection.get("cycles_to_10k_person_eq", 4)
    base_alpha = base_projection.get("effective_alpha", BASE_ALPHA)

    # Mitigation adjusts cycles inversely to combined score
    # High score = fewer additional cycles
    mitigation_penalty = 1.0 + (1.0 - mitigation.combined_score) * 0.5
    adjusted_cycles = int(base_cycles * mitigation_penalty)

    # Cycles saved by effective mitigation
    cycles_saved = max(0, adjusted_cycles - base_cycles - int((1.0 - mitigation.combined_score) * 2))

    result = {
        "base_cycles": base_cycles,
        "base_alpha": base_alpha,
        "mitigation_combined_score": mitigation.combined_score,
        "mitigation_effective_alpha": mitigation.effective_alpha,
        "mitigation_penalty_factor": round(mitigation_penalty, 4),
        "adjusted_cycles": adjusted_cycles,
        "cycles_saved_by_mitigation": cycles_saved
    }

    emit_receipt("mitigated_projection", {
        "tenant_id": "axiom-mitigation",
        **result
    })

    return result


def get_mitigation_summary(
    loss_pct: float = PARTITION_MAX_TEST_PCT,
    nodes_surviving: Optional[int] = None,
    receipt_integrity: float = 0.9
) -> Dict[str, Any]:
    """Get comprehensive mitigation summary.

    Convenience function for full mitigation analysis.

    Args:
        loss_pct: Partition loss percentage
        nodes_surviving: Operational node count
        receipt_integrity: Receipt coverage

    Returns:
        Complete mitigation summary dict
    """
    if nodes_surviving is None:
        nodes_surviving = NODE_BASELINE - int(NODE_BASELINE * loss_pct)

    score = compute_mitigation_score(
        loss_pct=loss_pct,
        nodes_surviving=nodes_surviving,
        receipt_integrity=receipt_integrity
    )

    summary = {
        "inputs": {
            "loss_pct": loss_pct,
            "nodes_surviving": nodes_surviving,
            "nodes_baseline": NODE_BASELINE,
            "receipt_integrity": receipt_integrity
        },
        "scores": {
            "partition": score.partition_score,
            "quorum": score.quorum_score,
            "tau": score.tau_score,
            "combined": score.combined_score
        },
        "effective_alpha": score.effective_alpha,
        "status": "healthy" if score.combined_score >= 0.7 else (
            "degraded" if score.combined_score >= 0.4 else "critical"
        )
    }

    emit_receipt("mitigation_summary", {
        "tenant_id": "axiom-mitigation",
        **summary
    })

    return summary
