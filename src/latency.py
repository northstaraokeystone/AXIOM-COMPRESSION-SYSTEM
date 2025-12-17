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


def tau_penalty(tau_seconds: float) -> float:
    """Calculate latency penalty multiplier for given tau.

    Linear interpolation between:
    - τ=0 returns 1.0 (no penalty)
    - τ=TAU_MARS_MAX (1200s) returns LATENCY_PENALTY_MAX (0.35)

    Args:
        tau_seconds: Latency in seconds

    Returns:
        Penalty multiplier (0-1). At τ=1200s returns ~0.35 (65% drop).

    Example:
        >>> tau_penalty(0)
        1.0
        >>> tau_penalty(1200)
        0.35
        >>> tau_penalty(600)  # midpoint
        0.675
    """
    if tau_seconds <= 0:
        return 1.0

    if tau_seconds >= TAU_MARS_MAX:
        penalty = LATENCY_PENALTY_MAX
    else:
        # Linear interpolation: 1.0 at τ=0, LATENCY_PENALTY_MAX at τ=TAU_MARS_MAX
        # penalty = 1.0 - (1.0 - LATENCY_PENALTY_MAX) * (tau / TAU_MARS_MAX)
        fraction = tau_seconds / TAU_MARS_MAX
        penalty = 1.0 - (1.0 - LATENCY_PENALTY_MAX) * fraction

    # Determine regime
    if tau_seconds < 1.0:
        regime = "earth"
    elif tau_seconds <= TAU_MARS_MIN:
        regime = "mars_min"
    else:
        regime = "mars_max"

    # Emit receipt
    emit_receipt("latency_penalty", {
        "tenant_id": "axiom-autonomy",
        "tau_seconds": tau_seconds,
        "penalty_multiplier": penalty,
        "effective_autonomy_retained": penalty,
        "regime": regime,
    })

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

    with open(path, 'r') as f:
        data = json.load(f)

    # Extract stored hash
    stored_hash = data.pop('payload_hash', None)
    if stored_hash is None:
        raise StopRule("mars_params.json missing payload_hash field")

    # Compute expected hash from data (without payload_hash)
    computed_hash = dual_hash(json.dumps(data, sort_keys=True))

    # Verify hash
    hash_verified = (stored_hash == computed_hash)

    if not hash_verified:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-autonomy",
            "metric": "hash_mismatch",
            "classification": "violation",
            "action": "halt",
            "expected": stored_hash,
            "actual": computed_hash,
            "file_path": path
        })
        raise StopRule(f"Mars params hash mismatch: expected {stored_hash}, got {computed_hash}")

    # Emit ingest receipt
    emit_receipt("mars_params_ingest", {
        "tenant_id": "axiom-autonomy",
        "file_path": path,
        "tau_mars_max_seconds": data['tau_mars_max_seconds'],
        "tau_mars_min_seconds": data['tau_mars_min_seconds'],
        "latency_penalty_at_max": data['latency_penalty_at_max'],
        "person_eq_milestone_early": data['person_eq_milestone_early'],
        "person_eq_milestone_city": data['person_eq_milestone_city'],
        "hash_verified": hash_verified,
        "payload_hash": stored_hash
    })

    # Restore hash to data for downstream use
    data['payload_hash'] = stored_hash

    return data


def effective_alpha(alpha: float, tau_seconds: float) -> float:
    """Calculate effective alpha after latency penalty.

    Represents α degradation under latency constraints.

    Args:
        alpha: Base compounding exponent (e.g., 1.69)
        tau_seconds: Latency in seconds

    Returns:
        Effective alpha = alpha × tau_penalty(tau)

    Example:
        >>> effective_alpha(1.69, 0)
        1.69
        >>> effective_alpha(1.69, 1200)  # Mars max
        0.5915  # ~0.59
    """
    penalty = tau_penalty(tau_seconds)
    eff_alpha = alpha * penalty
    return eff_alpha
