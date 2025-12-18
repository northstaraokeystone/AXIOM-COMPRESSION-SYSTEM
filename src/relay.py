"""relay.py - Relay Swarm Configuration and τ Reduction

THE RELAY INSIGHT:
    Relay swarms halve τ to 10min via midpoint satellites.
    Physical τ reduction compounds cycles to Earth-like (2 vs. 3).
    P cost per satellite trades infrastructure for latency.

Source: Grok - "Relay swarms: Midpoint satellites halve τ to 10min"
Source: Grok - "What's your P baseline for swarm costs?"
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any

from .core import emit_receipt, dual_hash, StopRule


# === CONSTANTS ===

RELAY_TAU_FACTOR = 0.5
"""Relay swarm halves τ. Source: Grok - 'halve τ to 10min'"""

RELAY_P_COST_PER_SAT = 0.05
"""P factor cost per relay satellite. Infrastructure investment."""

RELAY_SWARM_MIN = 3
"""Minimum satellites for coverage."""

RELAY_SWARM_OPTIMAL = 6
"""Full Mars-Earth corridor coverage."""

TAU_EARTH_TARGET = 600
"""10min = Earth-like latency target (down from 20min = 1200s)."""

TAU_STRATEGIES_PATH = "data/verified/tau_strategies.json"
"""Path to strategy constants file."""


# === DATACLASS ===


@dataclass
class RelayConfig:
    """Configuration for relay swarm.

    Attributes:
        swarm_size: Number of relay satellites (3-12)
        p_cost_per_sat: P factor cost per satellite (default 0.05)
        tau_reduction_factor: τ multiplier (default 0.5)
        operational: Whether swarm is active (default True)
    """

    swarm_size: int = RELAY_SWARM_OPTIMAL
    p_cost_per_sat: float = RELAY_P_COST_PER_SAT
    tau_reduction_factor: float = RELAY_TAU_FACTOR
    operational: bool = True

    def __post_init__(self):
        """Validate swarm size."""
        if self.swarm_size < 0:
            raise ValueError(f"swarm_size must be non-negative, got {self.swarm_size}")
        if self.swarm_size > 0 and self.swarm_size < RELAY_SWARM_MIN:
            # Warn but don't fail - partial coverage still helps
            pass


# === FUNCTIONS ===


def compute_relay_tau(base_tau: float, config: RelayConfig) -> float:
    """Compute reduced τ with relay swarm.

    Physical τ reduction via midpoint satellites.
    τ_reduced = base_tau × tau_reduction_factor

    Args:
        base_tau: Base latency in seconds (e.g., 1200 for Mars max)
        config: RelayConfig with swarm parameters

    Returns:
        Reduced τ in seconds

    Receipt: relay_tau_receipt
    """
    if not config.operational or config.swarm_size == 0:
        reduced_tau = base_tau
    else:
        reduced_tau = base_tau * config.tau_reduction_factor

    # Emit receipt
    emit_receipt(
        "relay_tau",
        {
            "tenant_id": "axiom-autonomy",
            "base_tau": base_tau,
            "reduced_tau": reduced_tau,
            "tau_reduction_factor": config.tau_reduction_factor,
            "swarm_size": config.swarm_size,
            "operational": config.operational,
        },
    )

    return reduced_tau


def compute_relay_p_cost(config: RelayConfig) -> float:
    """Compute total P factor cost for relay swarm.

    P_cost = swarm_size × p_cost_per_sat
    This reduces net P factor available for other operations.

    Args:
        config: RelayConfig with swarm parameters

    Returns:
        Total P factor cost

    Receipt: relay_p_cost_receipt
    """
    if not config.operational or config.swarm_size == 0:
        p_cost = 0.0
    else:
        p_cost = config.swarm_size * config.p_cost_per_sat

    emit_receipt(
        "relay_p_cost",
        {
            "tenant_id": "axiom-autonomy",
            "swarm_size": config.swarm_size,
            "p_cost_per_sat": config.p_cost_per_sat,
            "p_cost_total": p_cost,
            "operational": config.operational,
        },
    )

    return p_cost


def optimal_swarm_size(
    budget_p: float, target_tau: float, base_tau: float = 1200
) -> int:
    """Compute optimal swarm size within P budget to achieve target τ.

    Finds minimum swarm size that:
    1. Achieves target_tau (or as close as possible)
    2. Stays within budget_p

    Args:
        budget_p: Maximum P factor to spend on relay infrastructure
        target_tau: Target latency in seconds
        base_tau: Base latency before relay (default 1200s)

    Returns:
        Optimal swarm size (0 if budget insufficient)

    Receipt: relay_optimization_receipt
    """
    # Maximum affordable satellites
    max_sats = int(budget_p / RELAY_P_COST_PER_SAT)

    # Check if any relay helps achieve target
    if max_sats < RELAY_SWARM_MIN:
        # Can't afford minimum swarm
        optimal = 0
        achieved_tau = base_tau
    else:
        # With relay factor 0.5, any swarm >= MIN achieves halved τ
        achieved_tau = base_tau * RELAY_TAU_FACTOR

        if achieved_tau <= target_tau:
            # Minimum swarm is sufficient
            optimal = RELAY_SWARM_MIN
        else:
            # Target not achievable with single relay layer
            # Use optimal for best reduction within budget
            optimal = min(max_sats, RELAY_SWARM_OPTIMAL)
            achieved_tau = base_tau * RELAY_TAU_FACTOR

    actual_cost = optimal * RELAY_P_COST_PER_SAT

    emit_receipt(
        "relay_optimization",
        {
            "tenant_id": "axiom-autonomy",
            "budget_p": budget_p,
            "target_tau": target_tau,
            "base_tau": base_tau,
            "optimal_swarm_size": optimal,
            "achieved_tau": achieved_tau,
            "actual_p_cost": actual_cost,
            "budget_remaining": budget_p - actual_cost,
        },
    )

    return optimal


def load_relay_params(path: str = None) -> Dict[str, Any]:
    """Load and verify τ strategy parameters.

    Loads data/verified/tau_strategies.json, verifies payload_hash,
    and emits tau_strategies_ingest_receipt.

    Args:
        path: Optional path override (default: TAU_STRATEGIES_PATH)

    Returns:
        Dict containing verified strategy parameters

    Raises:
        StopRule: If payload_hash doesn't match computed hash
        FileNotFoundError: If data file doesn't exist

    Receipt: tau_strategies_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, TAU_STRATEGIES_PATH)

    with open(path, "r") as f:
        data = json.load(f)

    # Extract and verify hash
    stored_hash = data.pop("payload_hash", None)
    if stored_hash is None:
        raise StopRule("tau_strategies.json missing payload_hash field")

    computed_hash = dual_hash(json.dumps(data, sort_keys=True))

    hash_verified = stored_hash == computed_hash

    if not hash_verified:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": "axiom-autonomy",
                "metric": "hash_mismatch",
                "classification": "violation",
                "action": "halt",
                "expected": stored_hash,
                "actual": computed_hash,
                "file_path": path,
            },
        )
        raise StopRule(
            f"Strategy params hash mismatch: expected {stored_hash}, got {computed_hash}"
        )

    emit_receipt(
        "tau_strategies_ingest",
        {
            "tenant_id": "axiom-autonomy",
            "file_path": path,
            "relay_tau_factor": data["relay_tau_factor"],
            "relay_p_cost_per_sat": data["relay_p_cost_per_sat"],
            "relay_swarm_min": data["relay_swarm_min"],
            "relay_swarm_optimal": data["relay_swarm_optimal"],
            "onboard_ai_eff_alpha_floor": data["onboard_ai_eff_alpha_floor"],
            "predictive_tau_reduction": data["predictive_tau_reduction"],
            "hash_verified": hash_verified,
            "payload_hash": stored_hash,
        },
    )

    # Restore hash for downstream use
    data["payload_hash"] = stored_hash

    return data


def emit_relay_config_receipt(config: RelayConfig, base_tau: float) -> dict:
    """Emit comprehensive relay configuration receipt.

    Args:
        config: RelayConfig to document
        base_tau: Base τ before relay

    Returns:
        Receipt dict

    Receipt: relay_config_receipt
    """
    reduced_tau = compute_relay_tau(base_tau, config)
    p_cost = compute_relay_p_cost(config)

    return emit_receipt(
        "relay_config",
        {
            "tenant_id": "axiom-autonomy",
            "swarm_size": config.swarm_size,
            "tau_base": base_tau,
            "tau_reduced": reduced_tau,
            "p_cost_total": p_cost,
            "operational": config.operational,
            "tau_reduction_factor": config.tau_reduction_factor,
            "p_cost_per_sat": config.p_cost_per_sat,
        },
    )
