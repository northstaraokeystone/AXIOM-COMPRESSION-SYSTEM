"""latency.py - Mars Latency Penalty Calculation

THE PHYSICS CONSTRAINT:
    At τ=20min (Mars max), effective α drops from 1.69 to ~0.59
    This means compounding rate drops by 65%
    Every cycle of delay at Mars costs MORE than a cycle at Earth

The latency penalty doesn't just slow the build—it changes the physics
of what's possible.

Source: Grok - "At τ=20min max latency and α=1.69, sensitivity drops ~65% vs. baseline"
"""

import json
import os
from typing import Dict, Any

from .core import emit_receipt, dual_hash, StopRule


# === CONSTANTS ===

TAU_MARS_MAX = 1200
"""Maximum Mars latency in seconds (20 minutes one-way light delay)."""

TAU_MARS_MIN = 180
"""Minimum Mars latency in seconds (3 minutes at closest approach)."""

LATENCY_PENALTY_MAX = 0.35
"""Penalty multiplier at maximum latency. 65% drop → retains 35%."""

TAU_EARTH = 0.1
"""Earth-based operations baseline latency (~100ms)."""

MARS_PARAMS_PATH = "data/verified/mars_params.json"
"""Path to Mars latency physics parameters file."""


def tau_penalty(tau_seconds: float, relay_factor: float = 1.0) -> float:
    """Calculate latency penalty multiplier for given tau.

    Linear interpolation between:
    - τ=0 returns 1.0 (no penalty)
    - τ=TAU_MARS_MAX (1200s) returns LATENCY_PENALTY_MAX (0.35)

    Args:
        tau_seconds: Latency in seconds
        relay_factor: Multiplier for relay reduction (0.5 for swarm, 1.0 default)

    Returns:
        Penalty multiplier (0-1). At τ=1200s returns ~0.35 (65% drop).

    Example:
        >>> tau_penalty(0)
        1.0
        >>> tau_penalty(1200)
        0.35
        >>> tau_penalty(600)  # midpoint
        0.675
        >>> tau_penalty(1200, relay_factor=0.5)  # with relay swarm
        0.675  # effective tau = 600s
    """
    # Apply relay factor to get effective tau
    effective_tau = tau_seconds * relay_factor

    if effective_tau <= 0:
        return 1.0

    if effective_tau >= TAU_MARS_MAX:
        penalty = LATENCY_PENALTY_MAX
    else:
        # Linear interpolation: 1.0 at τ=0, LATENCY_PENALTY_MAX at τ=TAU_MARS_MAX
        # penalty = 1.0 - (1.0 - LATENCY_PENALTY_MAX) * (tau / TAU_MARS_MAX)
        fraction = effective_tau / TAU_MARS_MAX
        penalty = 1.0 - (1.0 - LATENCY_PENALTY_MAX) * fraction

    # Determine regime
    if effective_tau < 1.0:
        regime = "earth"
    elif effective_tau <= TAU_MARS_MIN:
        regime = "mars_min"
    else:
        regime = "mars_max"

    # Emit receipt
    emit_receipt(
        "latency_penalty",
        {
            "tenant_id": "spaceproof-autonomy",
            "tau_seconds": tau_seconds,
            "relay_factor": relay_factor,
            "effective_tau": effective_tau,
            "penalty_multiplier": penalty,
            "effective_autonomy_retained": penalty,
            "regime": regime,
        },
    )

    return penalty


def load_mars_params(path: str = None) -> Dict[str, Any]:
    """Load and verify Mars latency physics parameters.

    Loads data/verified/mars_params.json, verifies payload_hash,
    and emits mars_params_ingest_receipt.

    Args:
        path: Optional path override (default: MARS_PARAMS_PATH)

    Returns:
        Dict containing verified Mars parameters

    Raises:
        StopRule: If payload_hash doesn't match computed hash
        FileNotFoundError: If data file doesn't exist

    Receipt: mars_params_ingest
    """
    if path is None:
        # Resolve path relative to repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, MARS_PARAMS_PATH)

    with open(path, "r") as f:
        data = json.load(f)

    # Extract stored hash
    stored_hash = data.pop("payload_hash", None)
    if stored_hash is None:
        raise StopRule("mars_params.json missing payload_hash field")

    # Compute expected hash from data (without payload_hash)
    computed_hash = dual_hash(json.dumps(data, sort_keys=True))

    # Verify hash
    hash_verified = stored_hash == computed_hash

    if not hash_verified:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": "spaceproof-autonomy",
                "metric": "hash_mismatch",
                "classification": "violation",
                "action": "halt",
                "expected": stored_hash,
                "actual": computed_hash,
                "file_path": path,
            },
        )
        raise StopRule(
            f"Mars params hash mismatch: expected {stored_hash}, got {computed_hash}"
        )

    # Emit ingest receipt
    emit_receipt(
        "mars_params_ingest",
        {
            "tenant_id": "spaceproof-autonomy",
            "file_path": path,
            "tau_mars_max_seconds": data["tau_mars_max_seconds"],
            "tau_mars_min_seconds": data["tau_mars_min_seconds"],
            "latency_penalty_at_max": data["latency_penalty_at_max"],
            "person_eq_milestone_early": data["person_eq_milestone_early"],
            "person_eq_milestone_city": data["person_eq_milestone_city"],
            "hash_verified": hash_verified,
            "payload_hash": stored_hash,
        },
    )

    # Restore hash to data for downstream use
    data["payload_hash"] = stored_hash

    return data


def effective_alpha(
    alpha: float,
    tau_seconds: float,
    receipt_integrity: float = 0.0,
    relay_factor: float = 1.0,
    onboard_alpha_floor: float = 0.0,
) -> float:
    """Calculate effective alpha after latency penalty with receipt mitigation.

    THE PARADIGM SHIFT:
        Without receipts: effective_α = base_α × tau_penalty = 1.69 × 0.35 = 0.59
        With 90% receipts: effective_α = base_α × (1 - penalty × (1 - integrity)) = 1.58

    That's a 2.7× improvement in effective compounding from receipts alone.

    Formula:
        effective_α_mars = base_α × (1 - penalty × (1 - receipt_integrity))

    Where:
        - penalty = 1 - tau_penalty(tau) = latency drop (0.65 at τ=1200s)
        - receipt_integrity = fraction of decisions with receipts (0-1)

    Args:
        alpha: Base compounding exponent (e.g., 1.69)
        tau_seconds: Latency in seconds
        receipt_integrity: Receipt coverage (0-1). If >0, applies receipt mitigation.
        relay_factor: Multiplier for relay reduction (0.5 for swarm)
        onboard_alpha_floor: Floor for effective α (1.2 for onboard AI)

    Returns:
        Effective alpha after latency penalty and receipt mitigation, or floor if higher

    Example:
        >>> effective_alpha(1.69, 0)
        1.69
        >>> effective_alpha(1.69, 1200, 0.0)  # Mars max, no receipts
        0.5915  # ~0.59
        >>> effective_alpha(1.69, 1200, 0.9)  # Mars max, 90% receipts
        1.58  # ~1.58
        >>> effective_alpha(1.69, 1200, onboard_alpha_floor=1.2)  # with onboard AI
        1.2  # floor applied

    Receipt: effective_alpha_receipt (when receipt_integrity > 0)
    """
    # Apply relay factor to get raw penalty
    raw_penalty = tau_penalty(tau_seconds, relay_factor)
    latency_drop = 1.0 - raw_penalty  # How much we lose to latency

    if receipt_integrity > 0.0:
        # Receipt mitigation formula:
        # effective_α = base_α × (1 - penalty × (1 - receipt_integrity))
        # Where penalty = latency_drop (the loss, not the retained)
        mitigation_factor = 1.0 - latency_drop * (1.0 - receipt_integrity)
        eff_alpha = alpha * mitigation_factor

        # Calculate unmitigated for comparison
        unmitigated = alpha * raw_penalty
        mitigation_benefit = eff_alpha - unmitigated

        # Emit effective_alpha_receipt
        emit_receipt(
            "effective_alpha",
            {
                "tenant_id": "spaceproof-autonomy",
                "base_alpha": alpha,
                "tau_seconds": tau_seconds,
                "relay_factor": relay_factor,
                "tau_penalty": raw_penalty,
                "receipt_integrity": receipt_integrity,
                "effective_alpha": eff_alpha,
                "mitigation_benefit": mitigation_benefit,
                "unmitigated_alpha": unmitigated,
            },
        )
    else:
        # No receipt mitigation - original formula
        eff_alpha = alpha * raw_penalty

    # Apply onboard AI floor if specified
    if onboard_alpha_floor > 0:
        eff_alpha = max(eff_alpha, onboard_alpha_floor)

    return eff_alpha
