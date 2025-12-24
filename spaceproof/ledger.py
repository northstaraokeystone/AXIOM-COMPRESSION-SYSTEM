"""ledger.py - Distributed Anchor Boost with Quorum Integration

THE PHYSICS (Dec 2025 distributed anchoring):
    Base α = 2.56 (τ-penalty adjusted)
    Ledger boost = +0.12 (validated)
    Effective α = 2.68 with distributed anchoring

QUORUM INTEGRATION:
    Before anchoring, check quorum status.
    If quorum degraded (< baseline but >= threshold), apply degradation factor.
    If quorum fails (< threshold), halt via StopRule.

Source: Grok - "eff_α to 2.68 (+0.12)", "quorum intact for 5-node eg."
"""

from typing import Dict, Any, Optional

from .core import emit_receipt
from .partition import quorum_check, NODE_BASELINE, QUORUM_THRESHOLD, partition_sim


# === CONSTANTS (Dec 2025 distributed anchoring validated) ===

LEDGER_ALPHA_BOOST = 0.12
"""Legacy constant for backwards compatibility."""

LEDGER_ALPHA_BOOST_VALIDATED = 0.12
"""physics: Dec 2025 distributed anchoring boost. Validated by stress tests."""

BASE_ALPHA_PRE_BOOST = 2.56
"""Base α before distributed anchoring boost (τ-penalty adjusted)."""

EFFECTIVE_ALPHA_WITH_BOOST = 2.68
"""Effective α with distributed anchoring: 2.56 + 0.12 = 2.68"""

QUORUM_DEGRADATION_FACTOR = 0.02
"""Alpha degradation per missing node above threshold."""


def apply_ledger_boost(
    base_alpha: float = BASE_ALPHA_PRE_BOOST,
    boost: float = LEDGER_ALPHA_BOOST_VALIDATED,
) -> float:
    """Apply distributed anchoring boost to base alpha.

    Args:
        base_alpha: Base α before boost (default: 2.56)
        boost: Ledger boost amount (default: 0.12)

    Returns:
        Effective α with distributed anchoring

    Receipt: ledger_boost
    """
    eff_alpha = base_alpha + boost

    emit_receipt(
        "ledger_boost",
        {
            "tenant_id": "spaceproof-ledger",
            "base_alpha": base_alpha,
            "boost": boost,
            "effective_alpha": eff_alpha,
            "boost_validated": True,
            "physics": "Dec 2025 distributed anchoring",
        },
    )

    return eff_alpha


def apply_quorum_factor(
    base_alpha: float,
    nodes_surviving: int,
    nodes_baseline: int = NODE_BASELINE,
    quorum_min: int = QUORUM_THRESHOLD,
) -> float:
    """Apply quorum factor to alpha based on node health.

    If quorum is degraded (surviving < baseline but >= threshold),
    apply degradation factor. If quorum fails, raise StopRule.

    Args:
        base_alpha: Effective α to adjust
        nodes_surviving: Number of nodes still operational
        nodes_baseline: Expected baseline node count (default: 5)
        quorum_min: Minimum required for quorum (default: 3)

    Returns:
        Alpha with quorum degradation applied (if any)

    Raises:
        StopRule: If quorum fails (surviving < threshold)

    Receipt: quorum_factor (if degraded), warning if < baseline
    """
    # Check quorum - raises StopRule if failed
    quorum_check(nodes_surviving, quorum_min)

    # Calculate degradation
    nodes_missing = nodes_baseline - nodes_surviving
    degradation = 0.0
    quorum_status = "intact"

    if nodes_missing > 0:
        # Apply degradation per missing node
        degradation = nodes_missing * QUORUM_DEGRADATION_FACTOR
        quorum_status = "degraded"

        # Emit warning receipt
        emit_receipt(
            "quorum_warning",
            {
                "tenant_id": "spaceproof-ledger",
                "nodes_baseline": nodes_baseline,
                "nodes_surviving": nodes_surviving,
                "nodes_missing": nodes_missing,
                "degradation": degradation,
                "alpha_before": base_alpha,
                "alpha_after": base_alpha - degradation,
                "quorum_status": quorum_status,
            },
        )

    adjusted_alpha = base_alpha - degradation

    emit_receipt(
        "quorum_factor",
        {
            "tenant_id": "spaceproof-ledger",
            "base_alpha": base_alpha,
            "nodes_surviving": nodes_surviving,
            "nodes_baseline": nodes_baseline,
            "quorum_min": quorum_min,
            "degradation": degradation,
            "adjusted_alpha": adjusted_alpha,
            "quorum_status": quorum_status,
        },
    )

    return adjusted_alpha


def anchor_with_quorum(
    data: Dict[str, Any],
    nodes_surviving: Optional[int] = None,
    base_alpha: float = BASE_ALPHA_PRE_BOOST,
) -> Dict[str, Any]:
    """Anchor data with quorum check and ledger boost.

    Performs full anchoring pipeline:
    1. Apply ledger boost (+0.12)
    2. Check quorum status
    3. Apply quorum degradation if needed
    4. Emit anchor receipt

    Args:
        data: Data to anchor
        nodes_surviving: Current node count (default: baseline, no degradation)
        base_alpha: Pre-boost α (default: 2.56)

    Returns:
        Anchor receipt with effective α and quorum status

    Raises:
        StopRule: If quorum fails

    Receipt: distributed_anchor
    """
    # Step 1: Apply ledger boost
    boosted_alpha = apply_ledger_boost(base_alpha)

    # Step 2 & 3: Check quorum and apply factor
    if nodes_surviving is None:
        nodes_surviving = NODE_BASELINE

    final_alpha = apply_quorum_factor(
        boosted_alpha, nodes_surviving, NODE_BASELINE, QUORUM_THRESHOLD
    )

    # Determine quorum status string
    if nodes_surviving >= NODE_BASELINE:
        quorum_status = "full"
    elif nodes_surviving >= QUORUM_THRESHOLD:
        quorum_status = "degraded"
    else:
        quorum_status = "failed"  # Won't reach here due to StopRule

    # Step 4: Emit anchor receipt
    receipt = emit_receipt(
        "distributed_anchor",
        {
            "tenant_id": "spaceproof-ledger",
            "data_hash": data.get("payload_hash", "uncomputed"),
            "base_alpha": base_alpha,
            "ledger_boost": LEDGER_ALPHA_BOOST_VALIDATED,
            "boosted_alpha": boosted_alpha,
            "nodes_surviving": nodes_surviving,
            "nodes_baseline": NODE_BASELINE,
            "quorum_threshold": QUORUM_THRESHOLD,
            "quorum_status": quorum_status,
            "effective_alpha": final_alpha,
            "physics": "Dec 2025 distributed anchoring + quorum",
        },
    )

    return receipt


def get_effective_alpha_with_partition(
    loss_pct: float = 0.0, base_alpha: float = BASE_ALPHA_PRE_BOOST
) -> Dict[str, Any]:
    """Get effective alpha considering partition loss.

    Convenience function that combines ledger boost and partition impact.

    Args:
        loss_pct: Partition loss percentage (0-1)
        base_alpha: Pre-boost α (default: 2.56)

    Returns:
        Dict with effective α after all factors applied
    """
    # Apply ledger boost first
    boosted_alpha = base_alpha + LEDGER_ALPHA_BOOST_VALIDATED

    # Simulate partition impact
    result = partition_sim(
        nodes_total=NODE_BASELINE,
        loss_pct=loss_pct,
        base_alpha=boosted_alpha,
        emit=False,
    )

    return {
        "base_alpha": base_alpha,
        "ledger_boost": LEDGER_ALPHA_BOOST_VALIDATED,
        "boosted_alpha": boosted_alpha,
        "loss_pct": loss_pct,
        "eff_alpha_drop": result["eff_alpha_drop"],
        "effective_alpha": result["eff_alpha"],
        "quorum_status": result["quorum_status"],
    }
