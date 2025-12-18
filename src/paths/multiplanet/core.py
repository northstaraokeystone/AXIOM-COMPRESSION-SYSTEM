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

EXPANSION_SEQUENCE = ["asteroid", "mars", "europa", "titan"]
"""Ordered expansion sequence."""

LATENCY_BOUNDS_MIN = {"asteroid": 3, "mars": 3, "europa": 33, "titan": 70}
"""Minimum one-way latency in minutes."""

LATENCY_BOUNDS_MAX = {"asteroid": 20, "mars": 22, "europa": 53, "titan": 90}
"""Maximum one-way latency in minutes."""

AUTONOMY_REQUIREMENT = {"asteroid": 0.70, "mars": 0.85, "europa": 0.95, "titan": 0.99}
"""Required autonomy level per body."""

BANDWIDTH_BUDGET_MBPS = {"asteroid": 500, "mars": 100, "europa": 20, "titan": 5}
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
        moons: List of moons to include (default: titan, europa)

    Returns:
        Combined autonomy score (0-1)

    Receipt: mp_jovian_autonomy
    """
    if moons is None:
        moons = ["titan", "europa"]

    autonomy_weights = {
        "titan": (0.99, 0.6),  # (autonomy, weight) - Titan weighted higher (further)
        "europa": (0.95, 0.4),
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
