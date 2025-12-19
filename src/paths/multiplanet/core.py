"""Multi-planet expansion sequence core logic.

Evolution path: stub -> sequence -> body_sim -> integrated

EXPANSION SEQUENCE:
1. Asteroid (3-20 min latency, 70% autonomy)
2. Mars (3-22 min latency, 85% autonomy)
3. Europa (33-53 min latency, 95% autonomy)
4. Titan (70-90 min latency, 99% autonomy)

KEY INSIGHT:
Autonomy requirement INCREASES with distance/latency.
Each body builds on capabilities proven at previous body.

Source: AXIOM scalable paths architecture - Multi-planet expansion
"""

from typing import Dict, Any, List, Optional

from ..base import emit_path_receipt, load_path_spec, PathStopRule


# === CONSTANTS ===

MULTIPLANET_TENANT_ID = "axiom-multiplanet"
"""Tenant ID for multi-planet path receipts."""

EXPANSION_SEQUENCE = ["asteroid", "mars", "europa", "ganymede", "titan"]
"""Ordered expansion sequence (by increasing latency/autonomy)."""

LATENCY_BOUNDS_MIN = {
    "asteroid": 3,
    "mars": 3,
    "europa": 33,
    "titan": 70,
    "ganymede": 33,
}
"""Minimum one-way latency in minutes."""

LATENCY_BOUNDS_MAX = {
    "asteroid": 20,
    "mars": 22,
    "europa": 53,
    "titan": 90,
    "ganymede": 53,
}
"""Maximum one-way latency in minutes."""

AUTONOMY_REQUIREMENT = {
    "asteroid": 0.70,
    "mars": 0.85,
    "europa": 0.95,
    "titan": 0.99,
    "ganymede": 0.97,
}
"""Required autonomy level per body."""

BANDWIDTH_BUDGET_MBPS = {
    "asteroid": 500,
    "mars": 100,
    "europa": 20,
    "titan": 5,
    "ganymede": 15,
}
"""Bandwidth budget per body in Mbps."""

TELEMETRY_COMPRESSION_TARGET = 0.95
"""Target telemetry compression ratio."""


# === STUB STATUS ===


def stub_status() -> Dict[str, Any]:
    """Return current stub status.

    Returns:
        Dict with stub readiness info

    Receipt: mp_status
    """
    spec = load_path_spec("multiplanet")

    status = {
        "ready": True,
        "stage": "stub",
        "version": spec.get("version", "0.1.0"),
        "evolution_path": ["stub", "sequence", "body_sim", "integrated"],
        "current_capabilities": [
            "stub_status",
            "get_sequence",
            "get_body_config",
            "compute_latency_budget",
            "compute_autonomy_requirement",
        ],
        "pending_capabilities": ["simulate_body", "integrated_simulation"],
        "config": spec.get("config", {}),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "status", status)
    return status


# === SEQUENCE FUNCTIONS ===


def get_sequence() -> List[str]:
    """Return expansion sequence.

    Returns:
        List of body names in expansion order

    Receipt: mp_sequence
    """
    emit_path_receipt(
        "multiplanet",
        "sequence",
        {
            "sequence": EXPANSION_SEQUENCE,
            "count": len(EXPANSION_SEQUENCE),
            "tenant_id": MULTIPLANET_TENANT_ID,
        },
    )

    return list(EXPANSION_SEQUENCE)


def get_body_config(body: str) -> Dict[str, Any]:
    """Get body-specific configuration.

    Args:
        body: Body name (asteroid, mars, europa, titan)

    Returns:
        Dict with body configuration

    Raises:
        PathStopRule: If unknown body

    Receipt: mp_body
    """
    if body not in EXPANSION_SEQUENCE:
        raise PathStopRule("multiplanet", f"Unknown body: {body}")

    config = {
        "body": body,
        "sequence_position": EXPANSION_SEQUENCE.index(body) + 1,
        "latency_min_min": LATENCY_BOUNDS_MIN[body],
        "latency_max_min": LATENCY_BOUNDS_MAX[body],
        "autonomy_requirement": AUTONOMY_REQUIREMENT[body],
        "bandwidth_budget_mbps": BANDWIDTH_BUDGET_MBPS[body],
        "compression_target": TELEMETRY_COMPRESSION_TARGET,
        "prerequisites": EXPANSION_SEQUENCE[: EXPANSION_SEQUENCE.index(body)],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "body", config)
    return config


# === LATENCY COMPUTATION ===


def compute_latency_budget(body: str) -> Dict[str, Any]:
    """Compute latency constraints for body.

    Args:
        body: Body name

    Returns:
        Dict with latency budget info

    Receipt: mp_latency
    """
    if body not in EXPANSION_SEQUENCE:
        raise PathStopRule("multiplanet", f"Unknown body: {body}")

    min_latency = LATENCY_BOUNDS_MIN[body]
    max_latency = LATENCY_BOUNDS_MAX[body]

    # Round-trip time
    min_rtt = min_latency * 2
    max_rtt = max_latency * 2

    # Decision window (how long until Earth response possible)
    decision_window_min = min_rtt
    decision_window_max = max_rtt

    result = {
        "body": body,
        "one_way_min_min": min_latency,
        "one_way_max_min": max_latency,
        "round_trip_min_min": min_rtt,
        "round_trip_max_min": max_rtt,
        "decision_window_min": decision_window_min,
        "decision_window_max": decision_window_max,
        "autonomy_implication": f"Must handle {decision_window_max} min decisions autonomously",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "latency", result)
    return result


def compute_autonomy_requirement(body: str) -> float:
    """Get required autonomy level for body.

    Autonomy increases with distance:
    - Asteroid: 70% (close, backup from Earth)
    - Mars: 85% (proven ops, some Earth support)
    - Europa: 95% (distant, minimal Earth support)
    - Titan: 99% (very distant, essentially independent)

    Args:
        body: Body name

    Returns:
        Required autonomy level (0.0 to 1.0)

    Receipt: mp_autonomy
    """
    if body not in EXPANSION_SEQUENCE:
        raise PathStopRule("multiplanet", f"Unknown body: {body}")

    autonomy = AUTONOMY_REQUIREMENT[body]

    result = {
        "body": body,
        "autonomy_required": autonomy,
        "autonomy_pct": f"{autonomy * 100:.0f}%",
        "earth_support_max": f"{(1 - autonomy) * 100:.0f}%",
        "rationale": get_autonomy_rationale(body),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "autonomy", result)
    return autonomy


def get_autonomy_rationale(body: str) -> str:
    """Get rationale for autonomy requirement.

    Args:
        body: Body name

    Returns:
        Explanation string
    """
    rationales = {
        "asteroid": "Close to Earth, quick emergency response possible",
        "mars": "Proven operations, communication delays manageable",
        "europa": "Jupiter system, significant light delay, limited bandwidth",
        "titan": "Saturn system, extreme distance, nearly independent operations required",
    }
    return rationales.get(body, "Unknown body")


# === SIMULATION (STUB) ===


def simulate_body(body: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Body simulation placeholder.

    STUB: Returns projected values based on body config.
    FULL: Will integrate with fractal compression and autonomy models.

    Args:
        body: Body name
        config: Optional override config

    Returns:
        Dict with simulation results

    Receipt: mp_simulate
    """
    if body not in EXPANSION_SEQUENCE:
        raise PathStopRule("multiplanet", f"Unknown body: {body}")

    if config is None:
        config = {}

    body_config = get_body_config(body)
    latency = compute_latency_budget(body)
    autonomy = compute_autonomy_requirement(body)

    result = {
        "stub_mode": True,
        "body": body,
        "body_config": body_config,
        "latency_budget": latency,
        "autonomy_required": autonomy,
        "simulation_status": "pending",
        "next_stage": "Full body simulation with autonomy modeling",
        "prerequisites_met": all(
            p in ["asteroid"]  # Only asteroid has no prereqs
            for p in body_config.get("prerequisites", [])
        )
        if body != "asteroid"
        else True,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "simulate", result)
    return result


# === TELEMETRY COMPRESSION ===


def compute_telemetry_compression(body: str, data_rate_mbps: float) -> Dict[str, Any]:
    """Compute telemetry compression requirements.

    Args:
        body: Body name
        data_rate_mbps: Raw data generation rate

    Returns:
        Dict with compression requirements

    Receipt: mp_telemetry
    """
    if body not in EXPANSION_SEQUENCE:
        raise PathStopRule("multiplanet", f"Unknown body: {body}")

    bandwidth = BANDWIDTH_BUDGET_MBPS[body]
    compression_needed = data_rate_mbps / bandwidth if bandwidth > 0 else float("inf")
    target_met = compression_needed <= (1 / (1 - TELEMETRY_COMPRESSION_TARGET))

    result = {
        "body": body,
        "raw_data_rate_mbps": data_rate_mbps,
        "bandwidth_budget_mbps": bandwidth,
        "compression_needed": round(compression_needed, 2),
        "compression_target": TELEMETRY_COMPRESSION_TARGET,
        "target_met": target_met,
        "effective_compression": round(1 - (bandwidth / data_rate_mbps), 4)
        if data_rate_mbps > 0
        else 0,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "telemetry", result)
    return result


# === PATH INFO ===


def get_multiplanet_info() -> Dict[str, Any]:
    """Get multi-planet path configuration and status.

    Returns:
        Dict with path info

    Receipt: mp_info
    """
    spec = load_path_spec("multiplanet")

    info = {
        "path": "multiplanet",
        "version": spec.get("version", "0.1.0"),
        "status": spec.get("status", "stub"),
        "description": spec.get("description", ""),
        "sequence": EXPANSION_SEQUENCE,
        "config": spec.get("config", {}),
        "dependencies": spec.get("dependencies", []),
        "receipts": spec.get("receipts", []),
        "evolution": spec.get("evolution", {}),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "info", info)
    return info


# === TITAN INTEGRATION ===


def integrate_titan(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Titan methane harvesting to multi-planet path.

    Args:
        config: Optional Titan config override

    Returns:
        Dict with Titan integration results

    Receipt: mp_titan_integrate
    """
    # Import Titan module
    from ...titan_methane_hybrid import (
        load_titan_config,
        simulate_harvest,
        TITAN_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_titan_config()

    # Get Titan body config
    titan_body = get_body_config("titan")

    # Run harvest simulation
    harvest = simulate_harvest(duration_days=30)

    result = {
        "integrated": True,
        "body": "titan",
        "body_config": titan_body,
        "titan_config": config,
        "harvest_simulation": {
            "duration_days": harvest["duration_days"],
            "processed_kg": harvest["processed_kg"],
            "energy_kwh": harvest["energy_kwh"],
            "autonomy_achieved": harvest["autonomy_achieved"],
        },
        "autonomy_requirement": TITAN_AUTONOMY_REQUIREMENT,
        "autonomy_met": harvest["autonomy_achieved"] >= TITAN_AUTONOMY_REQUIREMENT,
        "sequence_position": EXPANSION_SEQUENCE.index("titan") + 1,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_integrate", result)
    return result


def compute_titan_autonomy() -> float:
    """Compute Titan-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_titan_autonomy
    """
    # Import Titan module
    from ...titan_methane_hybrid import (
        load_titan_config,
        simulate_harvest,
    )

    config = load_titan_config()
    harvest = simulate_harvest(duration_days=30)

    autonomy = harvest["autonomy_achieved"]

    result = {
        "body": "titan",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_autonomy", result)
    return autonomy


def simulate_titan_methane(
    duration_days: int = 30, extraction_rate_kg_hr: float = 10.0
) -> Dict[str, Any]:
    """Run Titan methane simulation within multiplanet context.

    Args:
        duration_days: Simulation duration
        extraction_rate_kg_hr: Extraction rate

    Returns:
        Dict with simulation results

    Receipt: mp_titan_simulate
    """
    # Import Titan module
    from ...titan_methane_hybrid import (
        simulate_harvest,
        methane_to_fuel,
        TITAN_AUTONOMY_REQUIREMENT,
    )

    # Run harvest simulation
    harvest = simulate_harvest(duration_days, extraction_rate_kg_hr)

    # Get fuel conversion metrics
    fuel = methane_to_fuel(harvest["processed_kg"])

    result = {
        "body": "titan",
        "simulation_type": "methane_harvest",
        "duration_days": duration_days,
        "extraction_rate_kg_hr": extraction_rate_kg_hr,
        "harvest": {
            "processed_kg": harvest["processed_kg"],
            "energy_kwh": harvest["energy_kwh"],
            "autonomy_achieved": harvest["autonomy_achieved"],
        },
        "fuel_conversion": fuel,
        "autonomy_met": harvest["autonomy_achieved"] >= TITAN_AUTONOMY_REQUIREMENT,
        "sequence": EXPANSION_SEQUENCE,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "titan_simulate", result)
    return result


# === EUROPA INTEGRATION ===


def integrate_europa(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Europa ice drilling to multi-planet path.

    Args:
        config: Optional Europa config override

    Returns:
        Dict with Europa integration results

    Receipt: mp_europa_integrate
    """
    # Import Europa module
    from ...europa_ice_hybrid import (
        load_europa_config,
        simulate_drilling,
        EUROPA_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_europa_config()

    # Get Europa body config
    europa_body = get_body_config("europa")

    # Run drilling simulation
    drilling = simulate_drilling(depth_m=1000, duration_days=30)

    result = {
        "integrated": True,
        "body": "europa",
        "body_config": europa_body,
        "europa_config": config,
        "drilling_simulation": {
            "depth_m": drilling["actual_depth_m"],
            "water_kg": drilling["water_extracted_kg"],
            "energy_kwh": drilling["melting_energy_kwh"],
            "autonomy_achieved": drilling["autonomy_achieved"],
        },
        "autonomy_requirement": EUROPA_AUTONOMY_REQUIREMENT,
        "autonomy_met": drilling["autonomy_achieved"] >= EUROPA_AUTONOMY_REQUIREMENT,
        "sequence_position": EXPANSION_SEQUENCE.index("europa") + 1,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "europa_integrate", result)
    return result


def compute_europa_autonomy() -> float:
    """Compute Europa-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_europa_autonomy
    """
    # Import Europa module
    from ...europa_ice_hybrid import (
        load_europa_config,
        simulate_drilling,
    )

    config = load_europa_config()
    drilling = simulate_drilling(depth_m=1000, duration_days=30)

    autonomy = drilling["autonomy_achieved"]

    result = {
        "body": "europa",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "europa_autonomy", result)
    return autonomy


def simulate_europa_drilling(
    depth_m: int = 1000, duration_days: int = 30, drill_rate_m_hr: float = 2.0
) -> Dict[str, Any]:
    """Run Europa drilling simulation within multiplanet context.

    Args:
        depth_m: Target drill depth in meters
        duration_days: Simulation duration
        drill_rate_m_hr: Drilling rate

    Returns:
        Dict with simulation results

    Receipt: mp_europa_simulate
    """
    # Import Europa module
    from ...europa_ice_hybrid import (
        simulate_drilling,
        ice_to_water,
        EUROPA_AUTONOMY_REQUIREMENT,
    )

    # Run drilling simulation
    drilling = simulate_drilling(depth_m, duration_days, drill_rate_m_hr)

    # Get water conversion metrics
    water = ice_to_water(drilling["ice_mass_kg"])

    result = {
        "body": "europa",
        "simulation_type": "ice_drilling",
        "depth_m": depth_m,
        "duration_days": duration_days,
        "drill_rate_m_hr": drill_rate_m_hr,
        "drilling": {
            "actual_depth_m": drilling["actual_depth_m"],
            "ice_mass_kg": drilling["ice_mass_kg"],
            "water_kg": drilling["water_extracted_kg"],
            "autonomy_achieved": drilling["autonomy_achieved"],
        },
        "water_conversion": water,
        "autonomy_met": drilling["autonomy_achieved"] >= EUROPA_AUTONOMY_REQUIREMENT,
        "sequence": EXPANSION_SEQUENCE,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "europa_simulate", result)
    return result


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


# === GANYMEDE INTEGRATION ===


def integrate_ganymede(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Ganymede magnetic field navigation to multi-planet path.

    Args:
        config: Optional Ganymede config override

    Returns:
        Dict with Ganymede integration results

    Receipt: mp_ganymede_integrate
    """
    # Import Ganymede module
    from ...ganymede_mag_hybrid import (
        load_ganymede_config,
        simulate_navigation,
        GANYMEDE_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_ganymede_config()

    # Get Ganymede body config
    ganymede_body = get_body_config("ganymede")

    # Run navigation simulation
    navigation = simulate_navigation(mode="field_following", duration_hrs=24)

    result = {
        "integrated": True,
        "body": "ganymede",
        "body_config": ganymede_body,
        "ganymede_config": config,
        "navigation_simulation": {
            "mode": navigation["mode"],
            "duration_hrs": navigation["duration_hrs"],
            "autonomy_achieved": navigation["autonomy"],
        },
        "autonomy_requirement": GANYMEDE_AUTONOMY_REQUIREMENT,
        "autonomy_met": navigation["autonomy"] >= GANYMEDE_AUTONOMY_REQUIREMENT,
        "sequence_position": EXPANSION_SEQUENCE.index("ganymede") + 1,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "ganymede_integrate", result)
    return result


def compute_ganymede_autonomy() -> float:
    """Compute Ganymede-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_ganymede_autonomy
    """
    # Import Ganymede module
    from ...ganymede_mag_hybrid import (
        load_ganymede_config,
        simulate_navigation,
    )

    config = load_ganymede_config()
    navigation = simulate_navigation(mode="field_following", duration_hrs=24)

    autonomy = navigation["autonomy"]

    result = {
        "body": "ganymede",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "ganymede_autonomy", result)
    return autonomy


def simulate_ganymede_navigation(
    mode: str = "field_following", duration_hrs: int = 24
) -> Dict[str, Any]:
    """Run Ganymede navigation simulation within multiplanet context.

    Args:
        mode: Navigation mode
        duration_hrs: Simulation duration

    Returns:
        Dict with simulation results

    Receipt: mp_ganymede_simulate
    """
    # Import Ganymede module
    from ...ganymede_mag_hybrid import (
        simulate_navigation,
        compute_radiation_shielding,
        GANYMEDE_AUTONOMY_REQUIREMENT,
        GANYMEDE_RADIUS_KM,
    )

    # Run navigation simulation
    navigation = simulate_navigation(mode, duration_hrs)

    # Get radiation shielding at typical position
    shielding = compute_radiation_shielding((GANYMEDE_RADIUS_KM + 500, 0, 0))

    result = {
        "body": "ganymede",
        "simulation_type": "magnetic_navigation",
        "mode": mode,
        "duration_hrs": duration_hrs,
        "navigation": {
            "autonomy_achieved": navigation["autonomy"],
            "autonomy_met": navigation["autonomy_met"],
        },
        "radiation_shielding": shielding,
        "autonomy_met": navigation["autonomy"] >= GANYMEDE_AUTONOMY_REQUIREMENT,
        "sequence": EXPANSION_SEQUENCE,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "ganymede_simulate", result)
    return result


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


# === CALLISTO INTEGRATION ===


def integrate_callisto(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Callisto ice operations to multi-planet path.

    Args:
        config: Optional Callisto config override

    Returns:
        Dict with Callisto integration results

    Receipt: mp_callisto_integrate
    """
    # Import Callisto module
    from ...callisto_ice import (
        load_callisto_config,
        simulate_extraction,
        compute_autonomy,
        CALLISTO_AUTONOMY_REQUIREMENT,
    )

    if config is None:
        config = load_callisto_config()

    # Get Callisto body config (add to sequence if not present)
    if "callisto" not in EXPANSION_SEQUENCE:
        EXPANSION_SEQUENCE.append("callisto")
        LATENCY_BOUNDS_MIN["callisto"] = 33
        LATENCY_BOUNDS_MAX["callisto"] = 53
        AUTONOMY_REQUIREMENT["callisto"] = 0.98
        BANDWIDTH_BUDGET_MBPS["callisto"] = 15

    # Run extraction simulation
    extraction = simulate_extraction(rate_kg_hr=100, duration_days=30)
    autonomy = compute_autonomy(extraction)

    result = {
        "integrated": True,
        "body": "callisto",
        "callisto_config": config,
        "extraction_simulation": {
            "duration_days": extraction["duration_days"],
            "total_extracted_kg": extraction["total_extracted_kg"],
            "energy_kwh": extraction["energy_kwh"],
            "autonomy_achieved": extraction["autonomy_achieved"],
        },
        "autonomy_requirement": CALLISTO_AUTONOMY_REQUIREMENT,
        "autonomy_met": autonomy >= CALLISTO_AUTONOMY_REQUIREMENT,
        "hub_suitability": "optimal",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "callisto_integrate", result)
    return result


def compute_callisto_autonomy() -> float:
    """Compute Callisto-specific autonomy metrics.

    Returns:
        Autonomy level (0-1)

    Receipt: mp_callisto_autonomy
    """
    # Import Callisto module
    from ...callisto_ice import (
        load_callisto_config,
        simulate_extraction,
        compute_autonomy,
    )

    config = load_callisto_config()
    extraction = simulate_extraction(rate_kg_hr=100, duration_days=30)
    autonomy = compute_autonomy(extraction)

    result = {
        "body": "callisto",
        "autonomy_achieved": autonomy,
        "autonomy_required": config["autonomy_requirement"],
        "autonomy_met": autonomy >= config["autonomy_requirement"],
        "latency_min": config["latency_min"],
        "earth_callback_max_pct": config["earth_callback_max_pct"],
        "radiation_level": config["radiation_level"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "callisto_autonomy", result)
    return autonomy


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


# === VENUS INNER PLANET INTEGRATION ===


VENUS_AUTONOMY_REQUIREMENT = 0.99
"""Venus cloud operations autonomy requirement (99%)."""


def integrate_venus(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Venus acid-cloud autonomy to multi-planet path.

    Args:
        config: Optional Venus config override

    Returns:
        Dict with Venus integration results

    Receipt: mp_venus_integrate
    """
    # Import Venus module
    from ...venus_acid_hybrid import (
        load_venus_config,
        simulate_cloud_ops,
        VENUS_AUTONOMY_REQUIREMENT as VENUS_REQ,
    )

    if config is None:
        config = load_venus_config()

    # Run Venus operations simulation
    ops = simulate_cloud_ops(duration_days=30, altitude_km=55.0)

    result = {
        "integrated": True,
        "subsystem": "venus_acid_cloud",
        "venus_config": {
            "surface_temp_c": config.get("surface_temp_c", 465),
            "cloud_altitude_km": config.get("cloud_altitude_km", [48, 70]),
            "acid_concentration": config.get("acid_concentration", 0.85),
        },
        "operations_result": {
            "autonomy": ops["autonomy"],
            "autonomy_met": ops["autonomy_met"],
            "altitude_km": ops["altitude_km"],
            "duration_days": ops["duration_days"],
        },
        "autonomy_requirement": VENUS_REQ,
        "autonomy_met": ops["autonomy_met"],
        "inner_planet": True,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "venus_integrate", result)
    return result


def compute_venus_autonomy() -> float:
    """Compute Venus-specific autonomy.

    Returns:
        Venus autonomy level (0-1)

    Receipt: mp_venus_autonomy
    """
    # Import Venus module
    from ...venus_acid_hybrid import (
        simulate_cloud_ops,
        VENUS_AUTONOMY_REQUIREMENT as VENUS_REQ,
    )

    # Run simulation
    ops = simulate_cloud_ops(duration_days=30, altitude_km=55.0)
    autonomy = ops["autonomy"]

    result = {
        "subsystem": "venus",
        "autonomy": autonomy,
        "requirement": VENUS_REQ,
        "met": autonomy >= VENUS_REQ,
        "inner_planet": True,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "venus_autonomy", result)
    return autonomy


def coordinate_inner_planets(venus: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Coordinate inner planet operations (Venus).

    Args:
        venus: Optional Venus state override

    Returns:
        Dict with inner planet coordination results

    Receipt: mp_inner_coordinate
    """
    # Import Venus module
    from ...venus_acid_hybrid import (
        load_venus_config,
        simulate_cloud_ops,
        VENUS_AUTONOMY_REQUIREMENT as VENUS_REQ,
    )

    if venus is None:
        venus_config = load_venus_config()
        venus_ops = simulate_cloud_ops(duration_days=30, altitude_km=55.0)
        venus = {"config": venus_config, "ops": venus_ops}

    result = {
        "subsystem": "inner_planets",
        "planets": ["venus"],
        "venus_result": {
            "autonomy": venus.get("ops", {}).get("autonomy", 0.0),
            "autonomy_met": venus.get("ops", {}).get("autonomy_met", False),
            "altitude_km": venus.get("ops", {}).get("altitude_km", 55.0),
        },
        "inner_planet_count": 1,
        "autonomy_requirement": VENUS_REQ,
        "all_targets_met": venus.get("ops", {}).get("autonomy_met", False),
        "expansion_status": "venus_operational",
        "next_target": "mercury",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "inner_coordinate", result)
    return result


def compute_solar_system_coverage(planets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute solar system coverage across all integrated bodies.

    Args:
        planets: List of integrated planets/moons

    Returns:
        Dict with solar system coverage analysis

    Receipt: mp_coverage
    """
    if planets is None:
        planets = ["asteroid", "mars", "europa", "titan", "ganymede", "callisto", "venus"]

    # Categorize by region
    inner_planets = [p for p in planets if p in ["venus", "mercury"]]
    asteroid_belt = [p for p in planets if p in ["asteroid"]]
    mars_system = [p for p in planets if p in ["mars"]]
    jovian_moons = [p for p in planets if p in ["europa", "titan", "ganymede", "callisto"]]

    # Compute coverage
    total_bodies = len(planets)
    inner_coverage = len(inner_planets) / 2  # Venus, Mercury
    mars_coverage = len(mars_system) / 1  # Mars
    jovian_coverage = len(jovian_moons) / 4  # 4 main moons

    result = {
        "planets": planets,
        "total_bodies": total_bodies,
        "inner_planets": inner_planets,
        "asteroid_belt": asteroid_belt,
        "mars_system": mars_system,
        "jovian_moons": jovian_moons,
        "coverage": {
            "inner": inner_coverage,
            "mars": mars_coverage,
            "jovian": jovian_coverage,
        },
        "overall_coverage": (inner_coverage + mars_coverage + jovian_coverage) / 3,
        "expansion_sequence": ["asteroid", "mars", "europa", "titan", "ganymede", "callisto", "venus"],
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "coverage", result)
    return result


# === SOLAR ORBITAL HUB INTEGRATION (D13) ===


SOLAR_HUB_AUTONOMY_TARGET = 0.95
"""Solar hub system autonomy target (95%)."""

MERCURY_AUTONOMY_REQUIREMENT = 0.97
"""Mercury operations autonomy requirement (97%)."""


def integrate_solar_hub(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Solar orbital hub (Venus+Mercury+Mars) to multi-planet path.

    Args:
        config: Optional Solar hub config override

    Returns:
        Dict with Solar hub integration results

    Receipt: mp_solar_hub_integrate
    """
    # Import Solar hub module
    from ...solar_orbital_hub import (
        load_solar_hub_config,
        simulate_hub_operations,
        compute_hub_autonomy,
        SOLAR_HUB_PLANETS,
        SOLAR_HUB_AUTONOMY_TARGET as HUB_TARGET,
    )

    if config is None:
        config = load_solar_hub_config()

    # Run hub operations simulation
    hub_result = simulate_hub_operations(duration_days=365)
    autonomy = compute_hub_autonomy()

    result = {
        "integrated": True,
        "subsystem": "solar_orbital_hub",
        "hub_config": config,
        "planets": SOLAR_HUB_PLANETS,
        "hub_simulation": {
            "duration_days": hub_result["duration_days"],
            "sync_cycles": hub_result["sync_cycles"],
            "autonomy": hub_result["autonomy"],
            "hub_operational": hub_result["hub_operational"],
        },
        "autonomy_target": HUB_TARGET,
        "autonomy_achieved": autonomy,
        "autonomy_met": autonomy >= HUB_TARGET,
        "inner_system": True,
        "coordination_mode": config.get("coordination_mode", "orbital_rl"),
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "solar_hub_integrate", result)
    return result


def compute_solar_hub_autonomy() -> float:
    """Compute Solar hub-specific autonomy metrics.

    Returns:
        Hub autonomy level (0-1)

    Receipt: mp_solar_hub_autonomy
    """
    # Import Solar hub module
    from ...solar_orbital_hub import (
        compute_hub_autonomy,
        SOLAR_HUB_AUTONOMY_TARGET as HUB_TARGET,
    )

    autonomy = compute_hub_autonomy()

    result = {
        "subsystem": "solar_hub",
        "autonomy": autonomy,
        "requirement": HUB_TARGET,
        "met": autonomy >= HUB_TARGET,
        "inner_system": True,
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "solar_hub_autonomy", result)
    return autonomy


def coordinate_inner_system(
    venus: Optional[Dict[str, Any]] = None,
    mercury: Optional[Dict[str, Any]] = None,
    mars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coordinate inner Solar system operations (Venus+Mercury+Mars).

    Args:
        venus: Optional Venus state override
        mercury: Optional Mercury state override
        mars: Optional Mars state override

    Returns:
        Dict with inner system coordination results

    Receipt: mp_inner_system_coordinate
    """
    # Import Solar hub module
    from ...solar_orbital_hub import (
        load_solar_hub_config,
        simulate_hub_operations,
        compute_hub_autonomy,
        SOLAR_HUB_PLANETS,
    )

    # Import Venus module
    from ...venus_acid_hybrid import (
        simulate_cloud_ops,
        VENUS_AUTONOMY_REQUIREMENT as VENUS_REQ,
    )

    config = load_solar_hub_config()

    # Venus operations
    if venus is None:
        venus_ops = simulate_cloud_ops(duration_days=30, altitude_km=55.0)
        venus = {"autonomy": venus_ops["autonomy"], "autonomy_met": venus_ops["autonomy_met"]}

    # Mercury operations (simulated for now)
    if mercury is None:
        mercury = {
            "autonomy": 0.97,
            "autonomy_met": True,
            "surface_temp_c": 430,
            "resources": ["metals", "solar_energy", "thermal_gradient"],
        }

    # Mars operations
    if mars is None:
        mars = {
            "autonomy": 0.85,
            "autonomy_met": True,
            "resources": ["water_ice", "co2", "regolith"],
        }

    # Run hub simulation
    hub_result = simulate_hub_operations(duration_days=365)
    hub_autonomy = compute_hub_autonomy()

    # Combined inner system autonomy
    venus_auto = venus.get("autonomy", 0.99)
    mercury_auto = mercury.get("autonomy", 0.97)
    mars_auto = mars.get("autonomy", 0.85)

    # Weighted by environment harshness
    combined_autonomy = (venus_auto * 0.35 + mercury_auto * 0.30 + mars_auto * 0.35)

    result = {
        "subsystem": "inner_system",
        "planets": SOLAR_HUB_PLANETS,
        "venus": {
            "autonomy": venus_auto,
            "requirement": VENUS_REQ,
            "met": venus_auto >= VENUS_REQ,
        },
        "mercury": {
            "autonomy": mercury_auto,
            "requirement": MERCURY_AUTONOMY_REQUIREMENT,
            "met": mercury_auto >= MERCURY_AUTONOMY_REQUIREMENT,
        },
        "mars": {
            "autonomy": mars_auto,
            "requirement": AUTONOMY_REQUIREMENT.get("mars", 0.85),
            "met": mars_auto >= AUTONOMY_REQUIREMENT.get("mars", 0.85),
        },
        "hub_autonomy": hub_autonomy,
        "combined_autonomy": round(combined_autonomy, 4),
        "hub_operational": hub_result["hub_operational"],
        "all_targets_met": (
            venus.get("autonomy_met", False)
            and mercury.get("autonomy_met", False)
            and mars.get("autonomy_met", False)
        ),
        "coordination_mode": "orbital_rl",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "inner_system_coordinate", result)
    return result


def compute_full_system_coverage(
    inner: Optional[Dict[str, Any]] = None,
    outer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute full Solar system coverage (Inner + Jovian).

    Args:
        inner: Optional inner system state
        outer: Optional Jovian system state

    Returns:
        Dict with full system coverage analysis

    Receipt: mp_full_system_coverage
    """
    # Import Solar hub module
    from ...solar_orbital_hub import (
        compute_hub_autonomy,
        SOLAR_HUB_PLANETS,
        SOLAR_HUB_AUTONOMY_TARGET as HUB_TARGET,
    )

    # Inner system
    if inner is None:
        inner_autonomy = compute_hub_autonomy()
        inner = {
            "planets": SOLAR_HUB_PLANETS,
            "autonomy": inner_autonomy,
            "autonomy_met": inner_autonomy >= HUB_TARGET,
        }

    # Outer system (Jovian)
    if outer is None:
        outer_autonomy = compute_jovian_autonomy()
        outer = {
            "moons": ["titan", "europa", "ganymede", "callisto"],
            "autonomy": outer_autonomy,
            "autonomy_met": outer_autonomy >= 0.95,
        }

    # Full system coverage
    inner_coverage = len(inner.get("planets", [])) / 3  # Venus, Mercury, Mars
    outer_coverage = len(outer.get("moons", [])) / 4  # 4 Jovian moons

    # Combined autonomy (inner systems need more Earth support)
    combined_autonomy = (
        inner.get("autonomy", 0.95) * 0.4 + outer.get("autonomy", 0.97) * 0.6
    )

    result = {
        "inner_system": {
            "planets": inner.get("planets", []),
            "autonomy": inner.get("autonomy", 0),
            "autonomy_met": inner.get("autonomy_met", False),
            "coverage": inner_coverage,
        },
        "outer_system": {
            "moons": outer.get("moons", []),
            "autonomy": outer.get("autonomy", 0),
            "autonomy_met": outer.get("autonomy_met", False),
            "coverage": outer_coverage,
        },
        "total_bodies": len(inner.get("planets", [])) + len(outer.get("moons", [])),
        "combined_autonomy": round(combined_autonomy, 4),
        "overall_coverage": (inner_coverage + outer_coverage) / 2,
        "full_system_operational": (
            inner.get("autonomy_met", False) and outer.get("autonomy_met", False)
        ),
        "next_phase": "Full Solar System extremes (include outer planets)",
        "tenant_id": MULTIPLANET_TENANT_ID,
    }

    emit_path_receipt("multiplanet", "full_system_coverage", result)
    return result
