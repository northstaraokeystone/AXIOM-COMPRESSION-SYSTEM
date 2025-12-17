"""mitigation.py - Partition Resilience in Mitigation Stack

Incorporates partition tolerance into the overall mitigation scoring
and weights mitigation by quorum health.

THE PHYSICS:
    - PARTITION_MITIGATION_FACTOR = 0.05 (max expected α drop)
    - Quorum health weight: 1.0 if intact, degraded otherwise
    - Combined mitigation includes τ-penalty + partition + quorum

Source: Grok - "Variable partitions (e.g., 40%)", "quorum intact"
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


@dataclass
class MitigationScore:
    """Combined mitigation score.

    Attributes:
        partition_score: Partition tolerance score (0-1)
        quorum_score: Quorum health score (0-1)
        tau_score: τ-penalty mitigation score (0-1)
        combined_score: Weighted combination
        effective_alpha: Final effective α after all mitigations
    """
    partition_score: float
    quorum_score: float
    tau_score: float
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


def compute_mitigation_score(
    loss_pct: float = 0.0,
    nodes_surviving: Optional[int] = None,
    receipt_integrity: float = 0.9,
    base_alpha: float = BASE_ALPHA,
    weights: Optional[Dict[str, float]] = None
) -> MitigationScore:
    """Compute combined mitigation score.

    Combines partition tolerance, quorum health, and τ-penalty mitigation
    into a single weighted score.

    Args:
        loss_pct: Partition loss percentage (0-1)
        nodes_surviving: Operational node count (default: baseline)
        receipt_integrity: Receipt coverage (default: 0.9)
        base_alpha: Baseline α (default: 2.68)
        weights: Optional weight overrides (default: equal weights)

    Returns:
        MitigationScore with all components

    Receipt: mitigation_score
    """
    if nodes_surviving is None:
        nodes_surviving = NODE_BASELINE

    if weights is None:
        weights = {"partition": 0.33, "quorum": 0.34, "tau": 0.33}

    # Compute individual scores
    partition_score = compute_partition_tolerance(loss_pct, base_alpha)
    quorum_score = compute_quorum_health(nodes_surviving, NODE_BASELINE)
    tau_score = compute_tau_mitigation(receipt_integrity)

    # Weighted combination
    combined = (
        partition_score * weights["partition"] +
        quorum_score * weights["quorum"] +
        tau_score * weights["tau"]
    )
    combined = round(combined, 4)

    # Compute effective alpha with all factors
    try:
        partition_result = partition_sim(NODE_BASELINE, loss_pct, base_alpha, emit=False)
        eff_alpha = partition_result["eff_alpha"]

        # Apply quorum factor
        quorum_degradation = (NODE_BASELINE - nodes_surviving) * 0.02
        eff_alpha = eff_alpha - quorum_degradation

        # Apply τ-mitigation boost
        tau_boost = (1.0 - tau_score) * 0.1  # Up to 10% recovery
        eff_alpha = eff_alpha + tau_boost
    except Exception:
        eff_alpha = 0.0

    score = MitigationScore(
        partition_score=partition_score,
        quorum_score=quorum_score,
        tau_score=tau_score,
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
        "combined_score": combined,
        "effective_alpha": score.effective_alpha,
        "weights": weights
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
