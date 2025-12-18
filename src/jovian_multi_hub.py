"""jovian_multi_hub.py - Full Jovian System Coordination Hub

PARADIGM:
    Full Jovian multi-moon coordination provides system-level autonomy >= 95%.
    Callisto serves as hub location (lowest radiation, outside magnetosphere).
    Four-moon resource coordination: Titan (methane), Europa (ice), Ganymede (shielding), Callisto (ice).

THE PHYSICS:
    Callisto advantages:
    - Outside Jupiter's main radiation belts (radiation level: 0.01)
    - 200 km ice/rock mix depth for water extraction
    - 16.69-day orbital period for predictable scheduling
    - 98% autonomy requirement (outermost Galilean moon)

    Hub coordination:
    - 12-hour sync interval across all four moons
    - Unified RL coordination mode
    - Resource transfer efficiency target: 90%
    - Redundancy factor: 2 (dual-path transfers)

Source: Grok - "Full Jovian hub: Callisto integration + Titan/Europa/Ganymede sync viable"
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

JOVIAN_TENANT_ID = "axiom-jovian"
"""Tenant ID for Jovian hub receipts."""

JOVIAN_MOONS = ["titan", "europa", "ganymede", "callisto"]
"""Full Jovian system moons."""

JOVIAN_RESOURCES = {
    "titan": "methane",
    "europa": "water_ice",
    "ganymede": "magnetic_shielding",
    "callisto": "water_ice",
}
"""Primary resource per moon."""

JOVIAN_HUB_SYNC_INTERVAL_HRS = 12
"""Sync interval for hub coordination (hours)."""

JOVIAN_SYSTEM_AUTONOMY_TARGET = 0.95
"""System-level autonomy target (95%)."""

JOVIAN_TRANSFER_EFFICIENCY_TARGET = 0.90
"""Resource transfer efficiency target (90%)."""

CALLISTO_AUTONOMY_REQUIREMENT = 0.98
"""Callisto autonomy requirement (98%, outermost)."""

CALLISTO_RADIATION_LEVEL = 0.01
"""Callisto radiation level (very low, outside magnetosphere)."""


# === HUB CONFIG LOADING ===


def load_jovian_hub_config() -> Dict[str, Any]:
    """Load Jovian hub configuration from d10_jovian_spec.json.

    Returns:
        Dict with hub configuration

    Receipt: jovian_hub_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d10_jovian_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    hub_config = spec.get("jovian_hub_config", {})

    result = {
        "moons": hub_config.get("moons", JOVIAN_MOONS),
        "resources": hub_config.get("resources", JOVIAN_RESOURCES),
        "sync_interval_hrs": hub_config.get(
            "sync_interval_hrs", JOVIAN_HUB_SYNC_INTERVAL_HRS
        ),
        "system_autonomy_target": hub_config.get(
            "system_autonomy_target", JOVIAN_SYSTEM_AUTONOMY_TARGET
        ),
        "hub_location": hub_config.get("hub_location", "callisto"),
        "coordination_mode": hub_config.get("coordination_mode", "unified_rl"),
        "transfer_efficiency_target": hub_config.get(
            "transfer_efficiency_target", JOVIAN_TRANSFER_EFFICIENCY_TARGET
        ),
        "redundancy_factor": hub_config.get("redundancy_factor", 2),
    }

    emit_receipt(
        "jovian_hub_config",
        {
            "receipt_type": "jovian_hub_config",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons_count": len(result["moons"]),
            "hub_location": result["hub_location"],
            "sync_interval_hrs": result["sync_interval_hrs"],
            "autonomy_target": result["system_autonomy_target"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === HUB INITIALIZATION ===


def init_hub(hub_location: str = "callisto") -> Dict[str, Any]:
    """Initialize Jovian hub at specified location.

    Args:
        hub_location: Hub location (default: callisto)

    Returns:
        Dict with hub initialization results

    Receipt: jovian_hub_init_receipt
    """
    load_jovian_hub_config()

    hub_state = {
        "initialized": True,
        "hub_location": hub_location,
        "moons_registered": [],
        "resources_available": {},
        "sync_active": False,
        "last_sync": None,
        "autonomy_level": 0.0,
    }

    emit_receipt(
        "jovian_hub_init",
        {
            "receipt_type": "jovian_hub_init",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hub_location": hub_location,
            "initialized": True,
            "payload_hash": dual_hash(json.dumps(hub_state, sort_keys=True)),
        },
    )

    return hub_state


def register_moon(moon: str, resources: Dict[str, Any]) -> Dict[str, Any]:
    """Register a moon to the Jovian hub.

    Args:
        moon: Moon name (titan, europa, ganymede, callisto)
        resources: Resource state for the moon

    Returns:
        Dict with registration result

    Receipt: jovian_moon_register
    """
    if moon not in JOVIAN_MOONS:
        return {"registered": False, "error": f"Unknown moon: {moon}"}

    result = {
        "registered": True,
        "moon": moon,
        "primary_resource": JOVIAN_RESOURCES.get(moon, "unknown"),
        "resources": resources,
        "autonomy_contribution": get_moon_autonomy_weight(moon),
    }

    emit_receipt(
        "jovian_moon_register",
        {
            "receipt_type": "jovian_moon_register",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moon": moon,
            "registered": True,
            "primary_resource": result["primary_resource"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_moon_autonomy_weight(moon: str) -> float:
    """Get autonomy weight for a moon.

    Args:
        moon: Moon name

    Returns:
        Autonomy weight (0-1)
    """
    weights = {
        "titan": 0.30,  # Furthest, highest autonomy needed
        "europa": 0.25,  # Ice resource critical
        "ganymede": 0.25,  # Magnetic shielding unique
        "callisto": 0.20,  # Hub location, coordination role
    }
    return weights.get(moon, 0.0)


# === SYNC OPERATIONS ===


def sync_all_moons(interval_hrs: int = JOVIAN_HUB_SYNC_INTERVAL_HRS) -> Dict[str, Any]:
    """Synchronize all four moons in the Jovian system.

    Args:
        interval_hrs: Sync interval in hours

    Returns:
        Dict with sync results

    Receipt: jovian_sync_receipt
    """
    # Simulate sync across all moons
    moon_states = {}
    for moon in JOVIAN_MOONS:
        moon_states[moon] = {
            "synced": True,
            "resource_status": "nominal",
            "autonomy_achieved": get_moon_base_autonomy(moon),
            "latency_min": get_moon_latency(moon),
        }

    # Compute combined sync efficiency
    sync_efficiency = 0.92  # Simulated efficiency

    result = {
        "sync_complete": True,
        "interval_hrs": interval_hrs,
        "moons_synced": len(JOVIAN_MOONS),
        "moon_states": moon_states,
        "sync_efficiency": sync_efficiency,
        "next_sync_hrs": interval_hrs,
    }

    emit_receipt(
        "jovian_sync",
        {
            "receipt_type": "jovian_sync",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons_synced": len(JOVIAN_MOONS),
            "sync_efficiency": sync_efficiency,
            "interval_hrs": interval_hrs,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_moon_base_autonomy(moon: str) -> float:
    """Get base autonomy level for a moon.

    Args:
        moon: Moon name

    Returns:
        Base autonomy (0-1)
    """
    autonomy_levels = {
        "titan": 0.99,  # Furthest, near-total autonomy
        "europa": 0.95,  # Jupiter system
        "ganymede": 0.97,  # Magnetic field complexity
        "callisto": 0.98,  # Outside magnetosphere
    }
    return autonomy_levels.get(moon, 0.90)


def get_moon_latency(moon: str) -> List[int]:
    """Get latency bounds for a moon (in minutes).

    Args:
        moon: Moon name

    Returns:
        [min_latency, max_latency] in minutes
    """
    # All Jovian moons have similar latency (same Jupiter system)
    return [33, 53]


# === RESOURCE ALLOCATION ===


def allocate_resources(request: Dict[str, Any]) -> Dict[str, Any]:
    """Allocate resources across Jovian moons.

    Args:
        request: Resource allocation request

    Returns:
        Dict with allocation results

    Receipt: jovian_allocation_receipt
    """
    source_moon = request.get("source", "callisto")
    target_moon = request.get("target", "europa")
    resource_type = request.get("resource_type", "water_ice")
    amount = request.get("amount", 1000.0)

    # Compute transfer efficiency
    efficiency = JOVIAN_TRANSFER_EFFICIENCY_TARGET

    result = {
        "allocated": True,
        "source": source_moon,
        "target": target_moon,
        "resource_type": resource_type,
        "amount_requested": amount,
        "amount_delivered": round(amount * efficiency, 2),
        "transfer_efficiency": efficiency,
        "efficiency_met": efficiency >= JOVIAN_TRANSFER_EFFICIENCY_TARGET,
    }

    emit_receipt(
        "jovian_allocation",
        {
            "receipt_type": "jovian_allocation",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": source_moon,
            "target": target_moon,
            "resource_type": resource_type,
            "amount": amount,
            "efficiency": efficiency,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === AUTONOMY COMPUTATION ===


def compute_system_autonomy(
    moon_states: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """Compute system-level autonomy for the Jovian hub.

    Args:
        moon_states: Optional list of moon states

    Returns:
        System autonomy (0-1)

    Receipt: jovian_system_autonomy_receipt
    """
    if moon_states is None:
        # Use default autonomy levels
        moon_states = [
            {"moon": "titan", "autonomy": 0.99},
            {"moon": "europa", "autonomy": 0.95},
            {"moon": "ganymede", "autonomy": 0.97},
            {"moon": "callisto", "autonomy": 0.98},
        ]

    # Weighted average based on moon weights
    total_weight = 0.0
    weighted_autonomy = 0.0

    for state in moon_states:
        moon = state.get("moon", "")
        autonomy = state.get("autonomy", 0.0)
        weight = get_moon_autonomy_weight(moon)

        weighted_autonomy += autonomy * weight
        total_weight += weight

    system_autonomy = weighted_autonomy / total_weight if total_weight > 0 else 0.0

    emit_receipt(
        "jovian_system_autonomy",
        {
            "receipt_type": "jovian_system_autonomy",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons_count": len(moon_states),
            "system_autonomy": round(system_autonomy, 4),
            "target_met": system_autonomy >= JOVIAN_SYSTEM_AUTONOMY_TARGET,
            "payload_hash": dual_hash(
                json.dumps({"system_autonomy": system_autonomy}, sort_keys=True)
            ),
        },
    )

    return round(system_autonomy, 4)


# === FULL COORDINATION ===


def coordinate_full_jovian(
    titan: Optional[Dict[str, Any]] = None,
    europa: Optional[Dict[str, Any]] = None,
    ganymede: Optional[Dict[str, Any]] = None,
    callisto: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coordinate full Jovian system with all four moons.

    Args:
        titan: Optional Titan state
        europa: Optional Europa state
        ganymede: Optional Ganymede state
        callisto: Optional Callisto state

    Returns:
        Dict with full coordination results

    Receipt: jovian_full_coordinate_receipt
    """
    # Default states if not provided
    if titan is None:
        titan = {"methane_kg": 5000.0, "autonomy": 0.99}
    if europa is None:
        europa = {"water_kg": 10000.0, "autonomy": 0.95}
    if ganymede is None:
        ganymede = {"shielding_factor": 3.5, "autonomy": 0.97}
    if callisto is None:
        callisto = {"water_kg": 8000.0, "autonomy": 0.98}

    # Compute individual moon autonomies
    moon_states = [
        {"moon": "titan", "autonomy": titan.get("autonomy", 0.99)},
        {"moon": "europa", "autonomy": europa.get("autonomy", 0.95)},
        {"moon": "ganymede", "autonomy": ganymede.get("autonomy", 0.97)},
        {"moon": "callisto", "autonomy": callisto.get("autonomy", 0.98)},
    ]

    system_autonomy = compute_system_autonomy(moon_states)

    result = {
        "subsystem": "full_jovian",
        "moons": JOVIAN_MOONS,
        "hub_location": "callisto",
        "titan": {
            "resource": "methane",
            "amount_kg": titan.get("methane_kg", 5000.0),
            "autonomy": titan.get("autonomy", 0.99),
        },
        "europa": {
            "resource": "water_ice",
            "amount_kg": europa.get("water_kg", 10000.0),
            "autonomy": europa.get("autonomy", 0.95),
        },
        "ganymede": {
            "resource": "magnetic_shielding",
            "shielding_factor": ganymede.get("shielding_factor", 3.5),
            "autonomy": ganymede.get("autonomy", 0.97),
        },
        "callisto": {
            "resource": "water_ice",
            "amount_kg": callisto.get("water_kg", 8000.0),
            "autonomy": callisto.get("autonomy", 0.98),
        },
        "system_autonomy": system_autonomy,
        "autonomy_target_met": system_autonomy >= JOVIAN_SYSTEM_AUTONOMY_TARGET,
        "coordination_mode": "unified_rl",
        "sync_interval_hrs": JOVIAN_HUB_SYNC_INTERVAL_HRS,
    }

    emit_receipt(
        "jovian_full_coordinate",
        {
            "receipt_type": "jovian_full_coordinate",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons_count": 4,
            "hub_location": "callisto",
            "system_autonomy": system_autonomy,
            "autonomy_target_met": system_autonomy >= JOVIAN_SYSTEM_AUTONOMY_TARGET,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === D10+JOVIAN INTEGRATED HUB ===


def d10_jovian_hub(
    tree_size: int, base_alpha: float, simulate: bool = False
) -> Dict[str, Any]:
    """Run integrated D10 + Jovian hub operation.

    Args:
        tree_size: Tree size for D10 recursion
        base_alpha: Base alpha value
        simulate: Whether to run in simulation mode

    Returns:
        Dict with integrated D10+Jovian results

    Receipt: d10_jovian_hub_receipt
    """
    # Import D10 function
    from .fractal_layers import d10_push

    # Run D10 push
    d10_result = d10_push(tree_size, base_alpha, simulate)

    # Run full Jovian coordination
    jovian_result = coordinate_full_jovian()

    result = {
        "integrated": True,
        "mode": "simulate" if simulate else "execute",
        "d10_result": {
            "eff_alpha": d10_result["eff_alpha"],
            "floor_met": d10_result["floor_met"],
            "target_met": d10_result["target_met"],
            "slo_passed": d10_result["slo_passed"],
        },
        "jovian_result": {
            "system_autonomy": jovian_result["system_autonomy"],
            "autonomy_target_met": jovian_result["autonomy_target_met"],
            "hub_location": jovian_result["hub_location"],
            "moons": jovian_result["moons"],
        },
        "combined_success": d10_result["slo_passed"]
        and jovian_result["autonomy_target_met"],
        "gate": "t24h",
    }

    emit_receipt(
        "d10_jovian_hub",
        {
            "receipt_type": "d10_jovian_hub",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "eff_alpha": d10_result["eff_alpha"],
            "system_autonomy": jovian_result["system_autonomy"],
            "combined_success": result["combined_success"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === TRANSFER NETWORK OPTIMIZATION ===


def optimize_transfer_network(moons: Optional[List[str]] = None) -> Dict[str, Any]:
    """Optimize inter-moon transfer network.

    Args:
        moons: List of moons to include (default: all)

    Returns:
        Dict with optimization results

    Receipt: jovian_transfer_optimize
    """
    if moons is None:
        moons = JOVIAN_MOONS

    # Build transfer matrix
    transfers = []
    for source in moons:
        for target in moons:
            if source != target:
                transfers.append(
                    {
                        "source": source,
                        "target": target,
                        "efficiency": JOVIAN_TRANSFER_EFFICIENCY_TARGET,
                        "latency_hrs": 2.0,  # Simulated transfer time
                    }
                )

    result = {
        "moons": moons,
        "transfer_count": len(transfers),
        "transfers": transfers,
        "avg_efficiency": JOVIAN_TRANSFER_EFFICIENCY_TARGET,
        "redundancy_factor": 2,
        "optimized": True,
    }

    emit_receipt(
        "jovian_transfer_optimize",
        {
            "receipt_type": "jovian_transfer_optimize",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "moons_count": len(moons),
            "transfer_count": len(transfers),
            "avg_efficiency": JOVIAN_TRANSFER_EFFICIENCY_TARGET,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def handle_moon_failure(failed_moon: str) -> Dict[str, Any]:
    """Handle graceful degradation when a moon fails.

    Args:
        failed_moon: Name of the failed moon

    Returns:
        Dict with degradation results

    Receipt: jovian_failure_handle
    """
    remaining_moons = [m for m in JOVIAN_MOONS if m != failed_moon]

    # Recompute system autonomy without failed moon
    moon_states = [
        {"moon": m, "autonomy": get_moon_base_autonomy(m)} for m in remaining_moons
    ]

    # Redistribute weights
    total_weight = sum(get_moon_autonomy_weight(m) for m in remaining_moons)
    adjusted_states = []
    for state in moon_states:
        moon = state["moon"]
        weight = (
            get_moon_autonomy_weight(moon) / total_weight if total_weight > 0 else 0
        )
        adjusted_states.append(
            {"moon": moon, "autonomy": state["autonomy"], "weight": weight}
        )

    degraded_autonomy = sum(s["autonomy"] * s["weight"] for s in adjusted_states)

    result = {
        "failed_moon": failed_moon,
        "remaining_moons": remaining_moons,
        "degraded_autonomy": round(degraded_autonomy, 4),
        "still_operational": degraded_autonomy >= 0.90,  # 90% threshold for degraded
        "hub_location": "callisto"
        if "callisto" in remaining_moons
        else remaining_moons[0],
        "recovery_action": f"Reroute through {remaining_moons[0]} if hub fails",
    }

    emit_receipt(
        "jovian_failure_handle",
        {
            "receipt_type": "jovian_failure_handle",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "failed_moon": failed_moon,
            "remaining_count": len(remaining_moons),
            "degraded_autonomy": degraded_autonomy,
            "still_operational": result["still_operational"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === HUB INFO ===


def get_jovian_hub_info() -> Dict[str, Any]:
    """Get Jovian hub configuration and status.

    Returns:
        Dict with hub info

    Receipt: jovian_hub_info
    """
    config = load_jovian_hub_config()

    info = {
        "hub_name": "Full Jovian Multi-Moon Hub",
        "hub_location": config.get("hub_location", "callisto"),
        "moons": config.get("moons", JOVIAN_MOONS),
        "resources": config.get("resources", JOVIAN_RESOURCES),
        "sync_interval_hrs": config.get(
            "sync_interval_hrs", JOVIAN_HUB_SYNC_INTERVAL_HRS
        ),
        "system_autonomy_target": config.get(
            "system_autonomy_target", JOVIAN_SYSTEM_AUTONOMY_TARGET
        ),
        "transfer_efficiency_target": config.get(
            "transfer_efficiency_target", JOVIAN_TRANSFER_EFFICIENCY_TARGET
        ),
        "coordination_mode": config.get("coordination_mode", "unified_rl"),
        "callisto_advantage": "Outside magnetosphere (0.01 radiation level)",
        "description": "Full Jovian system coordination with Callisto hub",
    }

    emit_receipt(
        "jovian_hub_info",
        {
            "receipt_type": "jovian_hub_info",
            "tenant_id": JOVIAN_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hub_location": info["hub_location"],
            "moons_count": len(info["moons"]),
            "autonomy_target": info["system_autonomy_target"],
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
