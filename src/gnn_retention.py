"""gnn_retention.py - GNN Nonlinear Retention Curve Functions

Nonlinear retention curve modeling for GNN predictive caching.
α asymptotes ~e (2.71828) - Shannon entropy bound.

Functions:
    - gnn_boost_factor: Compute nonlinear boost from GNN caching
    - compute_asymptote: Effective alpha with asymptotic formula
    - compute_asymptote_with_pruning: Alpha with pruning boost
    - nonlinear_retention: Full retention curve computation
    - get_retention_factor_gnn_isolated: Isolated GNN contribution
"""

import json
import math
from typing import Dict, Any

from .core import emit_receipt, dual_hash
from .constants import (
    ENTROPY_ASYMPTOTE_E,
    ASYMPTOTE_ALPHA,
    PRUNING_TARGET_ALPHA,
    MIN_EFF_ALPHA_VALIDATED,
    CACHE_DEPTH_BASELINE,
    BLACKOUT_BASE_DAYS,
    SATURATION_KAPPA,
    RETENTION_FACTOR_GNN_RANGE,
)


def gnn_boost_factor(blackout_days: int) -> float:
    """Compute nonlinear boost from GNN predictive caching.

    Nonlinear boost from predictive caching. Saturates toward asymptote.
    GNN provides anticipatory buffering that increases with duration.

    Formula: boost = 1 - exp(-κ * max(0, days - BASE_DAYS) / 50)
    Saturates at ~1.0 for very long durations.

    Args:
        blackout_days: Blackout duration in days

    Returns:
        boost: float (0 to ~1.0)
    """
    if blackout_days <= BLACKOUT_BASE_DAYS:
        return 0.0

    excess_days = blackout_days - BLACKOUT_BASE_DAYS
    boost = 1.0 - math.exp(-SATURATION_KAPPA * excess_days)

    return round(min(1.0, boost), 4)


def compute_asymptote(blackout_days: int, base_alpha: float = MIN_EFF_ALPHA_VALIDATED) -> float:
    """Compute effective alpha with asymptotic formula.

    Asymptotic formula: α = ASYMPTOTE - decay_term
    where decay_term → 0 as GNN caching saturates

    Formula: eff_alpha = ASYMPTOTE - (ASYMPTOTE - base_alpha) * exp(-κ * gnn_boost)

    Args:
        blackout_days: Blackout duration in days
        base_alpha: Base effective alpha (default: MIN_EFF_ALPHA_VALIDATED)

    Returns:
        eff_alpha: float
    """
    boost = gnn_boost_factor(blackout_days)

    decay_term = (ASYMPTOTE_ALPHA - base_alpha) * math.exp(-3.0 * boost)
    eff_alpha = ASYMPTOTE_ALPHA - decay_term

    eff_alpha = min(ASYMPTOTE_ALPHA, max(base_alpha, eff_alpha))

    return round(eff_alpha, 4)


def compute_asymptote_with_pruning(
    blackout_days: int,
    base_alpha: float = MIN_EFF_ALPHA_VALIDATED,
    pruning_uplift: float = 0.0
) -> float:
    """Compute effective alpha with asymptotic formula and pruning boost.

    Formula: eff_alpha = ASYMPTOTE + pruning_uplift - decay_term
    where pruning_uplift comes from ln(n) compression.

    Args:
        blackout_days: Blackout duration in days
        base_alpha: Base effective alpha (default: MIN_EFF_ALPHA_VALIDATED)
        pruning_uplift: Alpha uplift from pruning (default: 0.0)

    Returns:
        eff_alpha: float (may exceed ASYMPTOTE_ALPHA with pruning)
    """
    boost = gnn_boost_factor(blackout_days)

    decay_term = (ENTROPY_ASYMPTOTE_E - base_alpha) * math.exp(-3.0 * boost)
    eff_alpha = ENTROPY_ASYMPTOTE_E - decay_term

    if pruning_uplift > 0:
        eff_alpha = min(PRUNING_TARGET_ALPHA, eff_alpha + pruning_uplift)

    eff_alpha = max(base_alpha, eff_alpha)

    return round(eff_alpha, 4)


def get_retention_factor_gnn_isolated(blackout_days: int) -> Dict[str, Any]:
    """Get isolated GNN retention factor contribution.

    Returns the GNN-only contribution (1.008-1.015 typical).
    Used for ablation testing to isolate layer contributions.

    Args:
        blackout_days: Blackout duration in days

    Returns:
        Dict with retention_factor_gnn, contribution_pct, range_expected

    Receipt: retention_gnn_isolated
    """
    gnn_boost = gnn_boost_factor(blackout_days)

    min_retention, max_retention = RETENTION_FACTOR_GNN_RANGE
    retention_range = max_retention - min_retention

    retention_factor_gnn = min_retention + (gnn_boost * retention_range)
    retention_factor_gnn = round(min(max_retention, max(min_retention, retention_factor_gnn)), 4)

    contribution_pct = round((retention_factor_gnn - 1.0) * 100, 3)

    result = {
        "blackout_days": blackout_days,
        "retention_factor_gnn": retention_factor_gnn,
        "contribution_pct": contribution_pct,
        "gnn_boost": gnn_boost,
        "range_expected": RETENTION_FACTOR_GNN_RANGE,
        "layer": "gnn_cache"
    }

    emit_receipt("retention_gnn_isolated", {
        "tenant_id": "axiom-gnn-cache",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def apply_gnn_nonlinear_boost(
    base_mitigation: float,
    blackout_days: int,
    cache_depth: int = CACHE_DEPTH_BASELINE
) -> Dict[str, Any]:
    """Apply GNN nonlinear boost to mitigation stack.

    Returns boosted mitigation with asymptotic ceiling.

    Args:
        base_mitigation: Base mitigation value (0-1)
        blackout_days: Current blackout duration in days
        cache_depth: Cache depth in entries

    Returns:
        Dict with boosted_mitigation, gnn_boost, asymptote_factor
    """
    gnn_boost = gnn_boost_factor(blackout_days)

    asymptote_factor = 1.0 - 0.1 * math.exp(-3.0 * gnn_boost)

    boosted_mitigation = min(1.0, base_mitigation * asymptote_factor + gnn_boost * 0.05)

    return {
        "base_mitigation": round(base_mitigation, 4),
        "blackout_days": blackout_days,
        "gnn_boost": gnn_boost,
        "asymptote_factor": round(asymptote_factor, 4),
        "boosted_mitigation": round(boosted_mitigation, 4)
    }
