"""Jovian hub integration module.

This module handles coordination of Jovian moons (Titan, Europa, Ganymede, Callisto) as a unified system.
The Jovian hub manages cross-moon resource sharing, unified RL coordination, and hub location selection.

Functions:
- coordinate_jovian_moons: Coordinate Titan and Europa operations as Jovian subsystem
- integrate_unified_rl: Wire unified RL coordination to multi-planet path
- coordinate_titan_europa: Coordinate Titan and Europa via unified RL
- compute_jovian_autonomy: Compute system-level autonomy for Jovian moons
- coordinate_jovian_system: Coordinate Titan, Europa, and Ganymede as full Jovian system
- compute_system_autonomy: Compute system-level autonomy for all Jovian moons
- integrate_jovian_hub: Wire full Jovian multi-moon hub to multi-planet path
- coordinate_four_moons: Coordinate all four Jovian moons as complete system
- select_hub_location: Select optimal hub location from available moons
"""

from typing import Dict, Any, Optional, List

from ...base import emit_path_receipt
from ..core import MULTIPLANET_TENANT_ID


# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA = {
    "jovian_coordinate": "mp_jovian_coordinate",
    "unified_rl_integrate": "mp_unified_rl_integrate",
    "titan_europa_coordinate": "mp_titan_europa_coordinate",
    "jovian_autonomy": "mp_jovian_autonomy",
    "jovian_system_coordinate": "mp_jovian_system_coordinate",
    "system_autonomy": "mp_system_autonomy",
    "jovian_hub_integrate": "mp_jovian_hub_integrate",
    "four_moon_coordinate": "mp_four_moon_coordinate",
    "hub_select": "mp_hub_select",
}
"""Receipt types emitted by Jovian hub module."""


# === EXPORTS ===

__all__ = [
    "coordinate_jovian_moons",
    "integrate_unified_rl",
    "coordinate_titan_europa",
    "compute_jovian_autonomy",
    "coordinate_jovian_system",
    "compute_system_autonomy",
    "integrate_jovian_hub",
    "coordinate_four_moons",
    "select_hub_location",
    "RECEIPT_SCHEMA",
]


# === JOVIAN MOON COORDINATION ===


def coordinate_jovian_moons(
    titan_config: Optional[Dict[str, Any]] = None,
    europa_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coordinate Titan and Europa operations as Jovian subsystem.

    Args:
        titan_config: Optional Titan config override
        europa_config: Optional Europa config override

    Returns:
        Dict with coordination results

    Receipt: mp_jovian_coordinate
    """
    # Import modules
    from ...titan_methane_hybrid import (
        load_titan_config,
        simulate_harvest,
        TITAN_AUTONOMY_REQUIREMENT,
    )
    from ...europa_ice_hybrid import (
        load_europa_config,
        simulate_drilling,
        EUROPA_AUTONOMY_REQUIREMENT,
    )

    if titan_config is None:
        titan_config = load_titan_config()
    if europa_config is None:
        europa_config = load_europa_config()

    # Run simulations
    titan_result = simulate_harvest(duration_days=30)
    europa_result = simulate_drilling(depth_m=1000, duration_days=30)

    # Compute combined Jovian autonomy
    titan_autonomy = titan_result["autonomy_achieved"]
    europa_autonomy = europa_result["autonomy_achieved"]

    # Combined autonomy is weighted by distance (Titan further)
    combined_autonomy = (titan_autonomy * 0.6) + (europa_autonomy * 0.4)

    result = {
        "subsystem": "jovian",
        "bodies": ["titan", "europa"],
        "titan": {
            "autonomy_achieved": titan_autonomy,
            "autonomy_required": TITAN_AUTONOMY_REQUIREMENT,
            "autonomy_met": titan_autonomy >= TITAN_AUTONOMY_REQUIREMENT,
            "resource": "methane",
            "processed_kg": titan_result["processed_kg"],
        },
        "europa": {
            "autonomy_achieved": europa_autonomy,
            "autonomy_required": EUROPA_AUTONOMY_REQUIREMENT,
            "autonomy_met": europa_autonomy >= EUROPA_AUTONOMY_REQUIREMENT,
            "resource": "water_ice",
            "water_kg": europa_result["water_extracted_kg"],
        },
        "combined_autonomy": round(combined_autonomy, 4),
        "all_targets_met": (
            titan_autonomy >= TITAN_AUTONOMY_REQUIREMENT
            and europa_autonomy >= EUROPA_AUTONOMY_REQUIREMENT
        ),
        "coordination_status": "operational",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "jovian_coordinate", result)
    return result


# === UNIFIED RL INTEGRATION ===


def integrate_unified_rl(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire unified RL coordination to multi-planet path.

    Args:
        config: Optional unified RL config override

    Returns:
        Dict with unified RL integration results

    Receipt: mp_unified_rl_integrate
    """
    # Import unified RL module
    from ...multi_planet_sync import (
        load_sync_config,
        init_unified_rl,
        run_sync_cycle,
        RESOURCE_SHARE_EFFICIENCY,
    )

    if config is None:
        config = load_sync_config()

    # Initialize unified RL network
    rl_config = config.get("unified_rl", {})
    rl_network = init_unified_rl(rl_config.get("learning_rate", 0.001))

    # Run sync cycle
    cycle = run_sync_cycle()

    result = {
        "integrated": True,
        "unified_rl_config": rl_config,
        "rl_policy_weights": rl_network.policy_weights,
        "sync_cycle": {
            "successful": cycle["cycle_successful"],
            "efficiency": cycle["efficiency"],
            "titan_latency_min": cycle["titan_latency_min"],
            "europa_latency_min": cycle["europa_latency_min"],
        },
        "efficiency_target": RESOURCE_SHARE_EFFICIENCY,
        "efficiency_met": cycle["efficiency"] >= RESOURCE_SHARE_EFFICIENCY,
        "moons": config.get("moons", []),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "unified_rl_integrate", result)
    return result


def coordinate_titan_europa(
    titan: Optional[Dict[str, Any]] = None, europa: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Coordinate Titan and Europa via unified RL.

    Args:
        titan: Optional Titan state override
        europa: Optional Europa state override

    Returns:
        Dict with coordination results

    Receipt: mp_titan_europa_coordinate
    """
    # Import unified RL module
    from ...multi_planet_sync import (
        sync_resources,
        RESOURCE_SHARE_EFFICIENCY,
    )

    if titan is None:
        titan = {"methane_kg": 1000.0, "energy_kwh": 500.0, "autonomy": 0.99}
    if europa is None:
        europa = {"water_kg": 2000.0, "energy_kwh": 400.0, "autonomy": 0.95}

    # Run resource sync
    sync_result = sync_resources(titan, europa)

    result = {
        "titan_input": titan,
        "europa_input": europa,
        "sync_result": sync_result,
        "coordination_status": "operational"
        if sync_result["sync_successful"]
        else "degraded",
        "efficiency": sync_result["transfer_efficiency"],
        "efficiency_met": sync_result["transfer_efficiency"]
        >= RESOURCE_SHARE_EFFICIENCY,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_europa_coordinate", result)
    return result


def compute_jovian_autonomy(moons: Optional[List[str]] = None) -> float:
    """Compute system-level autonomy for Jovian moons.

    Args:
        moons: List of moons to include (default: titan, europa, ganymede)

    Returns:
        Combined autonomy score (0-1)

    Receipt: mp_jovian_autonomy
    """
    if moons is None:
        moons = ["titan", "europa", "ganymede"]

    autonomy_weights = {
        "titan": (0.99, 0.4),  # (autonomy, weight) - Titan weighted for distance
        "europa": (0.95, 0.3),
        "ganymede": (0.97, 0.3),  # Ganymede weighted for magnetic complexity
    }

    total_weight = 0.0
    weighted_autonomy = 0.0

    for moon in moons:
        if moon in autonomy_weights:
            autonomy, weight = autonomy_weights[moon]
            weighted_autonomy += autonomy * weight
            total_weight += weight

    combined = weighted_autonomy / total_weight if total_weight > 0 else 0.0

    result = {
        "moons": moons,
        "autonomy_by_moon": {m: autonomy_weights.get(m, (0, 0))[0] for m in moons},
        "weights_by_moon": {m: autonomy_weights.get(m, (0, 0))[1] for m in moons},
        "combined_autonomy": round(combined, 4),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "jovian_autonomy", result)
    return combined


# === FULL JOVIAN SYSTEM COORDINATION ===


def coordinate_jovian_system(
    titan: Optional[Dict[str, Any]] = None,
    europa: Optional[Dict[str, Any]] = None,
    ganymede: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coordinate Titan, Europa, and Ganymede operations as full Jovian system.

    Args:
        titan: Optional Titan state override
        europa: Optional Europa state override
        ganymede: Optional Ganymede state override

    Returns:
        Dict with full system coordination results

    Receipt: mp_jovian_system_coordinate
    """
    # Import modules
    from ...titan_methane_hybrid import (
        load_titan_config,
        simulate_harvest,
        TITAN_AUTONOMY_REQUIREMENT,
    )
    from ...europa_ice_hybrid import (
        load_europa_config,
        simulate_drilling,
        EUROPA_AUTONOMY_REQUIREMENT,
    )
    from ...ganymede_mag_hybrid import (
        load_ganymede_config,
        simulate_navigation,
        GANYMEDE_AUTONOMY_REQUIREMENT,
    )

    if titan is None:
        load_titan_config()
    if europa is None:
        load_europa_config()
    if ganymede is None:
        load_ganymede_config()

    # Run simulations
    titan_result = simulate_harvest(duration_days=30)
    europa_result = simulate_drilling(depth_m=1000, duration_days=30)
    ganymede_result = simulate_navigation(mode="field_following", duration_hrs=24)

    # Compute individual autonomies
    titan_autonomy = titan_result["autonomy_achieved"]
    europa_autonomy = europa_result["autonomy_achieved"]
    ganymede_autonomy = ganymede_result["autonomy"]

    # Combined autonomy (weighted by mission complexity)
    combined_autonomy = (
        titan_autonomy * 0.4 + europa_autonomy * 0.3 + ganymede_autonomy * 0.3
    )

    result = {
        "subsystem": "full_jovian",
        "bodies": ["titan", "europa", "ganymede"],
        "titan": {
            "autonomy_achieved": titan_autonomy,
            "autonomy_required": TITAN_AUTONOMY_REQUIREMENT,
            "autonomy_met": titan_autonomy >= TITAN_AUTONOMY_REQUIREMENT,
            "resource": "methane",
            "processed_kg": titan_result["processed_kg"],
        },
        "europa": {
            "autonomy_achieved": europa_autonomy,
            "autonomy_required": EUROPA_AUTONOMY_REQUIREMENT,
            "autonomy_met": europa_autonomy >= EUROPA_AUTONOMY_REQUIREMENT,
            "resource": "water_ice",
            "water_kg": europa_result["water_extracted_kg"],
        },
        "ganymede": {
            "autonomy_achieved": ganymede_autonomy,
            "autonomy_required": GANYMEDE_AUTONOMY_REQUIREMENT,
            "autonomy_met": ganymede_autonomy >= GANYMEDE_AUTONOMY_REQUIREMENT,
            "resource": "magnetic_shielding",
            "mode": ganymede_result["mode"],
        },
        "combined_autonomy": round(combined_autonomy, 4),
        "all_targets_met": (
            titan_autonomy >= TITAN_AUTONOMY_REQUIREMENT
            and europa_autonomy >= EUROPA_AUTONOMY_REQUIREMENT
            and ganymede_autonomy >= GANYMEDE_AUTONOMY_REQUIREMENT
        ),
        "coordination_status": "operational",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "jovian_system_coordinate", result)
    return result


def compute_system_autonomy(moons: Optional[List[str]] = None) -> float:
    """Compute system-level autonomy for all Jovian moons.

    Args:
        moons: List of moons to include (default: titan, europa, ganymede, callisto)

    Returns:
        Combined autonomy score (0-1)

    Receipt: mp_system_autonomy
    """
    if moons is None:
        moons = ["titan", "europa", "ganymede", "callisto"]

    # Use compute_jovian_autonomy with full moon list
    combined = compute_jovian_autonomy(moons)

    result = {
        "moons": moons,
        "system_autonomy": round(combined, 4),
        "full_jovian": len(moons) == 4,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "system_autonomy", result)
    return combined


# === FULL JOVIAN HUB INTEGRATION ===


def integrate_jovian_hub(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire full Jovian multi-moon hub to multi-planet path.

    Args:
        config: Optional hub config override

    Returns:
        Dict with Jovian hub integration results

    Receipt: mp_jovian_hub_integrate
    """
    # Import Jovian hub module
    from ...jovian_multi_hub import (
        load_jovian_hub_config,
        coordinate_full_jovian,
        JOVIAN_SYSTEM_AUTONOMY_TARGET,
    )

    if config is None:
        config = load_jovian_hub_config()

    # Run full Jovian coordination
    coordination = coordinate_full_jovian()

    result = {
        "integrated": True,
        "subsystem": "full_jovian_hub",
        "hub_config": config,
        "coordination_result": {
            "system_autonomy": coordination["system_autonomy"],
            "autonomy_target_met": coordination["autonomy_target_met"],
            "hub_location": coordination["hub_location"],
            "moons": coordination["moons"],
        },
        "autonomy_target": JOVIAN_SYSTEM_AUTONOMY_TARGET,
        "autonomy_met": coordination["autonomy_target_met"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "jovian_hub_integrate", result)
    return result


def coordinate_four_moons(
    titan: Optional[Dict[str, Any]] = None,
    europa: Optional[Dict[str, Any]] = None,
    ganymede: Optional[Dict[str, Any]] = None,
    callisto: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coordinate all four Jovian moons as complete system.

    Args:
        titan: Optional Titan state override
        europa: Optional Europa state override
        ganymede: Optional Ganymede state override
        callisto: Optional Callisto state override

    Returns:
        Dict with four-moon coordination results

    Receipt: mp_four_moon_coordinate
    """
    # Import Jovian hub module
    from ...jovian_multi_hub import (
        coordinate_full_jovian,
        JOVIAN_SYSTEM_AUTONOMY_TARGET,
    )

    # Run full coordination
    result = coordinate_full_jovian(titan, europa, ganymede, callisto)

    # Add multiplanet wrapper
    mp_result = {
        "subsystem": "four_moon",
        "moons": ["titan", "europa", "ganymede", "callisto"],
        "titan_result": result.get("titan", {}),
        "europa_result": result.get("europa", {}),
        "ganymede_result": result.get("ganymede", {}),
        "callisto_result": result.get("callisto", {}),
        "system_autonomy": result["system_autonomy"],
        "autonomy_target": JOVIAN_SYSTEM_AUTONOMY_TARGET,
        "all_targets_met": result["autonomy_target_met"],
        "hub_location": result["hub_location"],
        "coordination_mode": result.get("coordination_mode", "unified_rl"),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "four_moon_coordinate", mp_result)
    return mp_result


def select_hub_location(moons: Optional[List[str]] = None) -> str:
    """Select optimal hub location from available moons.

    Args:
        moons: List of available moons

    Returns:
        Optimal hub location (moon name)

    Receipt: mp_hub_select
    """
    if moons is None:
        moons = ["titan", "europa", "ganymede", "callisto"]

    # Hub scoring based on radiation (lower is better) and autonomy
    hub_scores = {
        "callisto": 10,  # Lowest radiation, optimal hub
        "ganymede": 7,  # Low radiation, magnetic protection
        "europa": 4,  # High radiation, ice resource
        "titan": 6,  # Far, but good autonomy
    }

    # Find best hub among available moons
    available_scores = {m: hub_scores.get(m, 0) for m in moons if m in hub_scores}
    best_hub = (
        max(available_scores, key=available_scores.get)
        if available_scores
        else "callisto"
    )

    result = {
        "available_moons": moons,
        "hub_scores": available_scores,
        "selected_hub": best_hub,
        "selection_rationale": "Lowest radiation, highest hub suitability score",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "hub_select", result)
    return best_hub
