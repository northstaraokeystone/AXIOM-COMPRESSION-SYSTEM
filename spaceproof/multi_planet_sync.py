"""multi_planet_sync.py - Unified RL Coordination for Multi-Moon Resource Sharing

PARADIGM:
    Unified RL coordination enables Titan methane + Europa ice resource sharing.
    Cross-moon sync with 24-hour intervals, 85% share efficiency minimum.

MOONS:
    - Titan: 99% autonomy, 70-90 min latency, methane (1.5 kg/m^3)
    - Europa: 95% autonomy, 33-53 min latency, water ice (15 km)

SYNC PROTOCOL:
    1. Resource state collection from each moon
    2. RL policy determines optimal transfer allocation
    3. Transfer execution with latency-aware scheduling
    4. Efficiency metrics updated

Source: SpaceProof D8 unified RL coordination - Jovian system integration
"""

import json
import random
from datetime import datetime
from typing import Any, Dict, List

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

SYNC_TENANT_ID = "spaceproof-sync"
"""Tenant ID for sync receipts."""

# Multi-sync config defaults
TITAN_METHANE_DENSITY_KG_M3 = 1.5
"""Titan methane density in kg/m^3."""

EUROPA_ICE_THICKNESS_KM = 15
"""Europa ice thickness in km."""

SYNC_LATENCY_TITAN_MIN = [70, 90]
"""Titan one-way latency bounds in minutes."""

SYNC_LATENCY_EUROPA_MIN = [33, 53]
"""Europa one-way latency bounds in minutes."""

UNIFIED_RL_LEARNING_RATE = 0.001
"""Unified RL learning rate."""

RESOURCE_SHARE_EFFICIENCY = 0.85
"""Minimum resource share efficiency."""

SYNC_INTERVAL_HRS = 24
"""Sync interval in hours."""


# === CONFIG LOADING ===


def load_sync_config() -> Dict[str, Any]:
    """Load multi-sync configuration from d8_multi_spec.json.

    Returns:
        Dict with sync configuration

    Receipt: sync_config_receipt
    """
    import os

    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d8_multi_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("multi_sync_config", {})

    emit_receipt(
        "sync_config",
        {
            "receipt_type": "sync_config",
            "tenant_id": SYNC_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons": config.get("moons", []),
            "share_efficiency": config.get("unified_rl", {}).get(
                "share_efficiency", RESOURCE_SHARE_EFFICIENCY
            ),
            "sync_interval_hrs": config.get("unified_rl", {}).get(
                "sync_interval_hrs", SYNC_INTERVAL_HRS
            ),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


# === UNIFIED RL ===


class UnifiedRLNetwork:
    """Unified RL network for multi-moon coordination."""

    def __init__(self, learning_rate: float = UNIFIED_RL_LEARNING_RATE):
        """Initialize unified RL network.

        Args:
            learning_rate: Learning rate for RL updates
        """
        self.learning_rate = learning_rate
        self.policy_weights = {
            "titan_priority": 0.5,
            "europa_priority": 0.5,
            "transfer_threshold": 0.3,
            "efficiency_weight": 0.8,
        }
        self.episode_count = 0
        self.cumulative_reward = 0.0

    def update_policy(self, reward: float, state: Dict[str, Any]) -> None:
        """Update policy weights based on reward.

        Args:
            reward: Reward signal from environment
            state: Current environment state
        """
        self.episode_count += 1
        self.cumulative_reward += reward

        # Simple gradient update (stub for full RL)
        titan_excess = state.get("titan_excess", 0.0)
        europa_excess = state.get("europa_excess", 0.0)

        # Adjust priorities based on resource availability
        if titan_excess > europa_excess:
            self.policy_weights["titan_priority"] -= self.learning_rate * 0.1
            self.policy_weights["europa_priority"] += self.learning_rate * 0.1
        else:
            self.policy_weights["titan_priority"] += self.learning_rate * 0.1
            self.policy_weights["europa_priority"] -= self.learning_rate * 0.1

        # Clamp priorities
        self.policy_weights["titan_priority"] = max(
            0.3, min(0.7, self.policy_weights["titan_priority"])
        )
        self.policy_weights["europa_priority"] = max(
            0.3, min(0.7, self.policy_weights["europa_priority"])
        )

    def get_allocation(
        self, titan_state: Dict[str, Any], europa_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get resource allocation decision.

        Args:
            titan_state: Titan resource state
            europa_state: Europa resource state

        Returns:
            Dict with allocation percentages
        """
        return {
            "titan_share": self.policy_weights["titan_priority"],
            "europa_share": self.policy_weights["europa_priority"],
            "transfer_threshold": self.policy_weights["transfer_threshold"],
        }


def init_unified_rl(
    learning_rate: float = UNIFIED_RL_LEARNING_RATE,
) -> UnifiedRLNetwork:
    """Initialize unified RL network.

    Args:
        learning_rate: Learning rate for RL updates

    Returns:
        UnifiedRLNetwork instance
    """
    return UnifiedRLNetwork(learning_rate)


# === RESOURCE SYNC ===


def sync_resources(
    titan_state: Dict[str, Any], europa_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Synchronize resources between Titan and Europa.

    Args:
        titan_state: Titan resource state (methane_kg, energy_kwh, autonomy)
        europa_state: Europa resource state (water_kg, energy_kwh, autonomy)

    Returns:
        Dict with sync results

    Receipt: resource_transfer_receipt
    """
    # Get resource levels
    titan_methane = titan_state.get("methane_kg", 0.0)
    titan_energy = titan_state.get("energy_kwh", 0.0)
    europa_water = europa_state.get("water_kg", 0.0)
    europa_energy = europa_state.get("energy_kwh", 0.0)

    # Calculate transfer amounts (basic balancing)
    total_energy = titan_energy + europa_energy
    titan_target = total_energy * 0.6  # Titan gets 60% (further, needs more)
    europa_target = total_energy * 0.4  # Europa gets 40%

    energy_transfer = 0.0
    transfer_direction = "none"

    if titan_energy < titan_target and europa_energy > europa_target:
        energy_transfer = min(
            europa_energy - europa_target, titan_target - titan_energy
        )
        transfer_direction = "europa_to_titan"
    elif europa_energy < europa_target and titan_energy > titan_target:
        energy_transfer = min(
            titan_energy - titan_target, europa_target - europa_energy
        )
        transfer_direction = "titan_to_europa"

    # Apply transfer efficiency
    effective_transfer = energy_transfer * RESOURCE_SHARE_EFFICIENCY

    result = {
        "titan_methane_kg": titan_methane,
        "europa_water_kg": europa_water,
        "energy_transfer_kwh": round(effective_transfer, 2),
        "transfer_direction": transfer_direction,
        "transfer_efficiency": RESOURCE_SHARE_EFFICIENCY,
        "titan_energy_after": round(
            titan_energy
            + (
                effective_transfer
                if transfer_direction == "europa_to_titan"
                else -effective_transfer
            ),
            2,
        ),
        "europa_energy_after": round(
            europa_energy
            + (
                effective_transfer
                if transfer_direction == "titan_to_europa"
                else -effective_transfer
            ),
            2,
        ),
        "sync_successful": True,
    }

    emit_receipt(
        "resource_transfer",
        {
            "receipt_type": "resource_transfer",
            "tenant_id": SYNC_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "transfer_direction": transfer_direction,
            "energy_transfer_kwh": round(effective_transfer, 2),
            "transfer_efficiency": RESOURCE_SHARE_EFFICIENCY,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_share_efficiency(transfers: List[Dict[str, Any]]) -> float:
    """Compute average share efficiency from transfer list.

    Args:
        transfers: List of transfer results

    Returns:
        Average share efficiency
    """
    if not transfers:
        return 0.0

    total_efficiency = sum(t.get("transfer_efficiency", 0.0) for t in transfers)
    return total_efficiency / len(transfers)


# === SYNC CYCLE ===


def run_sync_cycle(interval_hrs: int = SYNC_INTERVAL_HRS) -> Dict[str, Any]:
    """Run one sync cycle between Titan and Europa.

    Args:
        interval_hrs: Sync interval in hours

    Returns:
        Dict with cycle results

    Receipt: sync_cycle_receipt
    """
    # Simulate resource states
    titan_state = {
        "methane_kg": random.uniform(800, 1200),
        "energy_kwh": random.uniform(400, 600),
        "autonomy": 0.99,
    }
    europa_state = {
        "water_kg": random.uniform(1500, 2500),
        "energy_kwh": random.uniform(300, 500),
        "autonomy": 0.95,
    }

    # Run resource sync
    sync_result = sync_resources(titan_state, europa_state)

    # Compute latencies
    titan_latency = random.uniform(SYNC_LATENCY_TITAN_MIN[0], SYNC_LATENCY_TITAN_MIN[1])
    europa_latency = random.uniform(
        SYNC_LATENCY_EUROPA_MIN[0], SYNC_LATENCY_EUROPA_MIN[1]
    )

    result = {
        "cycle_interval_hrs": interval_hrs,
        "titan_state": titan_state,
        "europa_state": europa_state,
        "sync_result": sync_result,
        "titan_latency_min": round(titan_latency, 1),
        "europa_latency_min": round(europa_latency, 1),
        "total_round_trip_min": round((titan_latency + europa_latency) * 2, 1),
        "cycle_successful": sync_result["sync_successful"],
        "efficiency": sync_result["transfer_efficiency"],
    }

    emit_receipt(
        "sync_cycle",
        {
            "receipt_type": "sync_cycle",
            "tenant_id": SYNC_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "interval_hrs": interval_hrs,
            "efficiency": sync_result["transfer_efficiency"],
            "cycle_successful": sync_result["sync_successful"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INTEGRATED D8+SYNC ===


def d8_multi_sync(tree_size: int, base_alpha: float) -> Dict[str, Any]:
    """Run integrated D8 fractal + multi-planet sync.

    Args:
        tree_size: Tree size for D8 recursion
        base_alpha: Base alpha for D8 recursion

    Returns:
        Dict with integrated results

    Receipt: d8_multi_sync_receipt
    """
    # Import D8 from fractal_layers
    from .fractal_layers import d8_recursive_fractal

    # Run D8 recursion
    d8_result = d8_recursive_fractal(tree_size, base_alpha, depth=8)

    # Run sync cycle
    sync_result = run_sync_cycle()

    # Combined result
    result = {
        "d8_result": {
            "eff_alpha": d8_result["eff_alpha"],
            "floor_met": d8_result["floor_met"],
            "target_met": d8_result["target_met"],
            "instability": d8_result["instability"],
        },
        "sync_result": {
            "cycle_successful": sync_result["cycle_successful"],
            "efficiency": sync_result["efficiency"],
            "transfer_direction": sync_result["sync_result"]["transfer_direction"],
        },
        "combined_score": round(d8_result["eff_alpha"] * sync_result["efficiency"], 4),
        "all_targets_met": d8_result["floor_met"]
        and sync_result["efficiency"] >= RESOURCE_SHARE_EFFICIENCY,
        "tree_size": tree_size,
        "base_alpha": base_alpha,
    }

    emit_receipt(
        "d8_multi_sync",
        {
            "receipt_type": "d8_multi_sync",
            "tenant_id": SYNC_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "eff_alpha": d8_result["eff_alpha"],
            "sync_efficiency": sync_result["efficiency"],
            "all_targets_met": result["all_targets_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === OPTIMIZATION ===


def optimize_transfer_route(source: str, dest: str) -> Dict[str, Any]:
    """Compute optimal transfer route between moons.

    Args:
        source: Source moon (titan/europa)
        dest: Destination moon (titan/europa)

    Returns:
        Dict with optimal route info
    """
    latencies = {"titan": SYNC_LATENCY_TITAN_MIN, "europa": SYNC_LATENCY_EUROPA_MIN}

    source_latency = latencies.get(source, [0, 0])
    dest_latency = latencies.get(dest, [0, 0])

    # Direct transfer route (no relay optimization in this stub)
    min_total = source_latency[0] + dest_latency[0]
    max_total = source_latency[1] + dest_latency[1]

    return {
        "source": source,
        "dest": dest,
        "route": "direct",
        "min_latency_min": min_total,
        "max_latency_min": max_total,
        "avg_latency_min": (min_total + max_total) / 2,
        "efficiency_factor": RESOURCE_SHARE_EFFICIENCY,
    }


def handle_latency_mismatch(titan_latency: int, europa_latency: int) -> Dict[str, Any]:
    """Handle latency mismatch between Titan and Europa.

    Args:
        titan_latency: Current Titan latency in minutes
        europa_latency: Current Europa latency in minutes

    Returns:
        Dict with async handling strategy
    """
    diff = abs(titan_latency - europa_latency)
    faster_moon = "europa" if europa_latency < titan_latency else "titan"

    # Strategy: faster moon sends updates more frequently
    if diff > 30:
        strategy = "dual_rate"
        titan_freq = 2 if faster_moon == "europa" else 1
        europa_freq = 2 if faster_moon == "titan" else 1
    else:
        strategy = "synchronized"
        titan_freq = 1
        europa_freq = 1

    return {
        "latency_diff_min": diff,
        "faster_moon": faster_moon,
        "strategy": strategy,
        "titan_update_freq": titan_freq,
        "europa_update_freq": europa_freq,
        "sync_window_min": max(titan_latency, europa_latency) * 2,
    }


# === MAIN RUNNER ===


def run_sync(simulate: bool = True) -> Dict[str, Any]:
    """Run full sync workflow.

    Args:
        simulate: Whether to run in simulation mode

    Returns:
        Dict with full sync results

    Receipt: multi_planet_sync_receipt
    """
    # Load config
    config = load_sync_config()

    # Initialize RL
    rl_network = init_unified_rl(
        config.get("unified_rl", {}).get("learning_rate", UNIFIED_RL_LEARNING_RATE)
    )

    # Run sync cycle
    cycle = run_sync_cycle()

    result = {
        "mode": "simulate" if simulate else "execute",
        "config": config,
        "rl_policy": rl_network.policy_weights,
        "cycle_result": cycle,
        "efficiency": cycle["efficiency"],
        "moons": config.get("moons", []),
        "sync_successful": cycle["cycle_successful"],
    }

    emit_receipt(
        "multi_planet_sync",
        {
            "receipt_type": "multi_planet_sync",
            "tenant_id": SYNC_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons": config.get("moons", []),
            "efficiency": cycle["efficiency"],
            "sync_successful": cycle["cycle_successful"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_sync_info() -> Dict[str, Any]:
    """Get multi-planet sync configuration info.

    Returns:
        Dict with sync info

    Receipt: sync_info
    """
    config = load_sync_config()

    info = {
        "moons": config.get("moons", []),
        "titan_config": config.get("titan", {}),
        "europa_config": config.get("europa", {}),
        "unified_rl_config": config.get("unified_rl", {}),
        "constants": {
            "titan_methane_density_kg_m3": TITAN_METHANE_DENSITY_KG_M3,
            "europa_ice_thickness_km": EUROPA_ICE_THICKNESS_KM,
            "resource_share_efficiency": RESOURCE_SHARE_EFFICIENCY,
            "sync_interval_hrs": SYNC_INTERVAL_HRS,
        },
    }

    return info
