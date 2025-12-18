"""stoprules.py - Centralized StopRule Registry

All stoprule handlers in one place. Fail fast, fail loud.
Per CLAUDEME: NEVER catch StopRule silently.

Usage:
    from .stoprules import (
        stoprule_overflow,
        stoprule_over_prune,
        stoprule_low_confidence,
        stoprule_quorum_lost,
        stoprule_chain_broken
    )

Registry:
    - stoprule_overflow: Cache overflow detected
    - stoprule_over_prune: Trim factor too aggressive
    - stoprule_low_confidence: GNN confidence below threshold
    - stoprule_quorum_lost: Quorum fraction below minimum
    - stoprule_chain_broken: Merkle chain integrity violation
    - stoprule_hash_mismatch: Hash verification failed
    - stoprule_invalid_receipt: Receipt schema violation
    - stoprule_alpha_violation: Alpha outside valid range
"""

from typing import NoReturn

from .core import emit_receipt, dual_hash, StopRule
from .constants import (
    OVER_PRUNE_STOPRULE_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_QUORUM_FRACTION,
    MIN_PROOF_PATHS_RETAINED,
    OVERFLOW_CAPACITY_PCT,
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
)


def stoprule_overflow(
    tenant_id: str,
    blackout_days: int,
    cache_depth: int,
    overflow_pct: float,
    pruning_enabled: bool = False,
) -> NoReturn:
    """StopRule for cache overflow.

    Args:
        tenant_id: Tenant identifier
        blackout_days: Current blackout duration
        cache_depth: Cache depth in entries
        overflow_pct: Current overflow percentage (0-1)
        pruning_enabled: Whether pruning is active

    Raises:
        StopRule: Always raises with overflow details
    """
    import json

    emit_receipt(
        "overflow_stoprule",
        {
            "tenant_id": tenant_id,
            "blackout_days": blackout_days,
            "cache_depth": cache_depth,
            "overflow_pct": overflow_pct,
            "pruning_enabled": pruning_enabled,
            "action": "halt",
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "blackout_days": blackout_days,
                        "cache_depth": cache_depth,
                        "overflow_pct": overflow_pct,
                        "pruning_enabled": pruning_enabled,
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    raise StopRule(
        f"Cache overflow at {blackout_days}d (pruning={pruning_enabled}): "
        f"{overflow_pct * 100:.1f}% > {OVERFLOW_CAPACITY_PCT * 100:.0f}%"
    )


def stoprule_over_prune(tenant_id: str, trim_factor: float) -> NoReturn:
    """StopRule for excessive pruning.

    Args:
        tenant_id: Tenant identifier
        trim_factor: The trim factor being applied

    Raises:
        StopRule: If trim_factor > OVER_PRUNE_STOPRULE_THRESHOLD (0.6)
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": "trim_factor",
            "baseline": OVER_PRUNE_STOPRULE_THRESHOLD,
            "delta": trim_factor - OVER_PRUNE_STOPRULE_THRESHOLD,
            "classification": "violation",
            "action": "halt",
        },
    )

    raise StopRule(
        f"Over-prune: trim_factor {trim_factor} > "
        f"{OVER_PRUNE_STOPRULE_THRESHOLD} threshold"
    )


def stoprule_low_confidence(
    tenant_id: str, confidence: float, threshold: float = MIN_CONFIDENCE_THRESHOLD
) -> NoReturn:
    """StopRule for low GNN prediction confidence.

    Args:
        tenant_id: Tenant identifier
        confidence: Actual confidence score
        threshold: Minimum required confidence

    Raises:
        StopRule: If confidence < threshold
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": "predictive_confidence",
            "baseline": threshold,
            "delta": confidence - threshold,
            "classification": "deviation",
            "action": "skip_predictive",
        },
    )

    raise StopRule(f"Predictive confidence {confidence:.3f} < {threshold} threshold")


def stoprule_quorum_lost(
    tenant_id: str, retention_ratio: float, required: float = MIN_QUORUM_FRACTION
) -> NoReturn:
    """StopRule for lost quorum.

    Args:
        tenant_id: Tenant identifier
        retention_ratio: Current retention ratio (0-1)
        required: Minimum required quorum fraction

    Raises:
        StopRule: If retention_ratio < required
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": "quorum",
            "baseline": required,
            "delta": retention_ratio - required,
            "classification": "violation",
            "action": "halt",
        },
    )

    raise StopRule(
        f"Quorum lost: {retention_ratio:.2%} retention < {required:.2%} required"
    )


def stoprule_chain_broken(
    tenant_id: str, proof_paths_count: int, reason: str = ""
) -> NoReturn:
    """StopRule for broken Merkle chain.

    Args:
        tenant_id: Tenant identifier
        proof_paths_count: Number of valid proof paths
        reason: Additional context

    Raises:
        StopRule: If chain integrity compromised
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": "proof_paths",
            "baseline": MIN_PROOF_PATHS_RETAINED,
            "delta": proof_paths_count - MIN_PROOF_PATHS_RETAINED,
            "classification": "violation",
            "action": "halt",
        },
    )

    msg = f"Chain broken: only {proof_paths_count} proof paths (need {MIN_PROOF_PATHS_RETAINED})"
    if reason:
        msg += f" - {reason}"

    raise StopRule(msg)


def stoprule_hash_mismatch(tenant_id: str, expected: str, actual: str) -> NoReturn:
    """StopRule for hash verification failure.

    Args:
        tenant_id: Tenant identifier
        expected: Expected hash value
        actual: Actual computed hash

    Raises:
        StopRule: Always raises with hash details
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": "hash_mismatch",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "violation",
            "action": "halt",
            "expected": expected,
            "actual": actual,
        },
    )

    raise StopRule(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(tenant_id: str, reason: str) -> NoReturn:
    """StopRule for invalid receipt structure.

    Args:
        tenant_id: Tenant identifier
        reason: Description of validation failure

    Raises:
        StopRule: Always raises with reason
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": "invalid_receipt",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "anti_pattern",
            "action": "halt",
            "reason": reason,
        },
    )

    raise StopRule(f"Invalid receipt: {reason}")


def stoprule_alpha_violation(
    tenant_id: str, alpha: float, violation_type: str
) -> NoReturn:
    """StopRule for alpha outside valid range.

    Args:
        tenant_id: Tenant identifier
        alpha: The computed alpha value
        violation_type: "below_floor" or "above_ceiling"

    Raises:
        StopRule: If alpha violates bounds
    """
    if violation_type == "below_floor":
        baseline = SHANNON_FLOOR_ALPHA
        classification = "investigate"
        msg = f"Alpha {alpha:.4f} below Shannon floor {baseline}"
    else:
        baseline = ALPHA_CEILING_TARGET
        classification = "investigate"
        msg = f"Alpha {alpha:.4f} exceeds theoretical ceiling {baseline}"

    emit_receipt(
        "anomaly",
        {
            "tenant_id": tenant_id,
            "metric": f"alpha_{violation_type}",
            "baseline": baseline,
            "delta": alpha - baseline,
            "classification": classification,
            "action": "investigate",
        },
    )

    raise StopRule(msg)


def check_trim_factor(tenant_id: str, trim_factor: float) -> None:
    """Check trim factor and raise StopRule if too aggressive.

    Use this for validation before applying pruning.

    Args:
        tenant_id: Tenant identifier
        trim_factor: The trim factor to validate

    Raises:
        StopRule: If trim_factor > threshold
    """
    if trim_factor > OVER_PRUNE_STOPRULE_THRESHOLD:
        stoprule_over_prune(tenant_id, trim_factor)


def check_confidence(tenant_id: str, confidence: float) -> None:
    """Check confidence and raise StopRule if too low.

    Use this for validation before predictive pruning.

    Args:
        tenant_id: Tenant identifier
        confidence: The confidence score to validate

    Raises:
        StopRule: If confidence < threshold
    """
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        stoprule_low_confidence(tenant_id, confidence)


def check_quorum(tenant_id: str, retention_ratio: float) -> None:
    """Check quorum and raise StopRule if lost.

    Use this for validation after pruning operations.

    Args:
        tenant_id: Tenant identifier
        retention_ratio: The retention ratio to validate

    Raises:
        StopRule: If retention_ratio < MIN_QUORUM_FRACTION
    """
    if retention_ratio < MIN_QUORUM_FRACTION:
        stoprule_quorum_lost(tenant_id, retention_ratio)
