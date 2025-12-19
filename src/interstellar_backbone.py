"""Interstellar backbone for full Solar System 7-body RL coordination.

PARADIGM:
    Full Solar System coordination via interstellar RL backbone.
    Unifies Jovian moons (Titan, Europa, Ganymede, Callisto) with
    inner planets (Venus, Mercury, Mars) for system-wide autonomy.

THE PHYSICS:
    - Jovian moon orbital periods: 2-17 days
    - Inner planet periods: 88-687 days
    - Maximum latency: 90 min (Saturn at conjunction)
    - Sync interval: 60 days (system-wide)
    - Total bodies: 7

ORBITAL MECHANICS:
    Extended Keplerian model for multi-body coordination.
    Accounts for light-time delays and conjunction windows.
    Resource pooling across system boundaries.

Source: Grok - "Interstellar backbone: Full Jovian+Inner merge viable"
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

INTERSTELLAR_TENANT_ID = "axiom-interstellar"
"""Tenant ID for interstellar backbone receipts."""

INTERSTELLAR_JOVIAN_BODIES = ["titan", "europa", "ganymede", "callisto"]
"""Jovian moons in the interstellar backbone."""

INTERSTELLAR_INNER_BODIES = ["venus", "mercury", "mars"]
"""Inner planets in the interstellar backbone."""

INTERSTELLAR_ALL_BODIES = INTERSTELLAR_JOVIAN_BODIES + INTERSTELLAR_INNER_BODIES
"""All bodies in the interstellar backbone (7 total)."""

INTERSTELLAR_BODY_COUNT = 7
"""Total body count in the interstellar backbone."""

INTERSTELLAR_SYNC_INTERVAL_DAYS = 60
"""System-wide sync interval in days."""

INTERSTELLAR_RL_LEARNING_RATE = 0.0001
"""RL learning rate for interstellar coordination (very slow for stability)."""

INTERSTELLAR_AUTONOMY_TARGET = 0.98
"""System-wide autonomy target."""

MAX_INTERSTELLAR_LATENCY_MIN = 90
"""Maximum latency in minutes (Saturn at conjunction)."""

# Orbital parameters for Jovian moons (relative to Jupiter)
JOVIAN_MOON_PERIODS_DAYS = {
    "titan": 15.945,  # Saturn's moon, but included for Saturn subsystem
    "europa": 3.551,
    "ganymede": 7.155,
    "callisto": 16.689,
}
"""Orbital periods for Jovian moons in days."""

JOVIAN_MOON_SEMI_MAJOR_AU = {
    "titan": 0.00817,  # ~1.22M km from Saturn
    "europa": 0.00448,  # ~671k km from Jupiter
    "ganymede": 0.00715,  # ~1.07M km from Jupiter
    "callisto": 0.01259,  # ~1.88M km from Jupiter
}
"""Semi-major axis relative to parent body in AU."""

# Inner planet parameters
INNER_PLANET_PERIODS_DAYS = {
    "venus": 225,
    "mercury": 88,
    "mars": 687,
}
"""Orbital periods for inner planets in days."""

INNER_PLANET_SEMI_MAJOR_AU = {
    "venus": 0.723,
    "mercury": 0.387,
    "mars": 1.524,
}
"""Semi-major axis in AU from Sun."""

# Parent body distances from Sun
JUPITER_SEMI_MAJOR_AU = 5.203
"""Jupiter's semi-major axis in AU."""

SATURN_SEMI_MAJOR_AU = 9.537
"""Saturn's semi-major axis in AU."""

# Resources by body
BODY_RESOURCES = {
    "titan": ["hydrocarbons", "nitrogen", "water_ice"],
    "europa": ["water_ice", "oxygen", "organics"],
    "ganymede": ["water_ice", "silicates", "magnetic_field"],
    "callisto": ["water_ice", "silicates", "ancient_surface"],
    "venus": ["atmospheric_chemicals", "solar_energy"],
    "mercury": ["metals", "solar_energy", "thermal_gradient"],
    "mars": ["water_ice", "co2", "regolith"],
}
"""Available resources per body."""


# === CONFIGURATION FUNCTIONS ===


def load_interstellar_config() -> Dict[str, Any]:
    """Load interstellar backbone configuration from d14_interstellar_spec.json.

    Returns:
        Dict with interstellar backbone configuration

    Receipt: interstellar_config_receipt
    """
    import os

    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d14_interstellar_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("interstellar_config", {})

    emit_receipt(
        "interstellar_config",
        {
            "receipt_type": "interstellar_config",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": config.get("body_count", INTERSTELLAR_BODY_COUNT),
            "sync_interval_days": config.get(
                "sync_interval_days", INTERSTELLAR_SYNC_INTERVAL_DAYS
            ),
            "autonomy_target": config.get(
                "autonomy_target", INTERSTELLAR_AUTONOMY_TARGET
            ),
            "coordination_mode": config.get("coordination_mode", "interstellar_rl"),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


def get_all_bodies() -> List[str]:
    """Return all 7 bodies in the interstellar backbone.

    Returns:
        List of body names (4 Jovian moons + 3 inner planets)

    Receipt: interstellar_bodies_receipt
    """
    bodies = INTERSTELLAR_ALL_BODIES.copy()

    emit_receipt(
        "interstellar_bodies",
        {
            "receipt_type": "interstellar_bodies",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "bodies": bodies,
            "count": len(bodies),
            "jovian_count": len(INTERSTELLAR_JOVIAN_BODIES),
            "inner_count": len(INTERSTELLAR_INNER_BODIES),
            "payload_hash": dual_hash(json.dumps(bodies, sort_keys=True)),
        },
    )

    return bodies


def get_interstellar_info() -> Dict[str, Any]:
    """Get interstellar backbone configuration summary.

    Returns:
        Dict with interstellar backbone info

    Receipt: interstellar_info_receipt
    """
    config = load_interstellar_config()

    info = {
        "bodies": {
            "jovian": INTERSTELLAR_JOVIAN_BODIES,
            "inner": INTERSTELLAR_INNER_BODIES,
            "total": INTERSTELLAR_ALL_BODIES,
        },
        "body_count": INTERSTELLAR_BODY_COUNT,
        "sync_interval_days": INTERSTELLAR_SYNC_INTERVAL_DAYS,
        "rl_learning_rate": INTERSTELLAR_RL_LEARNING_RATE,
        "autonomy_target": INTERSTELLAR_AUTONOMY_TARGET,
        "max_latency_min": MAX_INTERSTELLAR_LATENCY_MIN,
        "resources": BODY_RESOURCES,
        "config": config,
    }

    emit_receipt(
        "interstellar_info",
        {
            "receipt_type": "interstellar_info",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": INTERSTELLAR_BODY_COUNT,
            "autonomy_target": INTERSTELLAR_AUTONOMY_TARGET,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === ORBITAL POSITION COMPUTATION ===


def compute_body_positions(timestamp: float = 0.0) -> Dict[str, Any]:
    """Compute positions of all bodies in the interstellar backbone.

    Uses simplified Keplerian model with mean anomaly.

    Args:
        timestamp: Days since epoch (default: 0)

    Returns:
        Dict with positions for each body

    Receipt: interstellar_position_receipt
    """
    positions = {}

    # Compute inner planet positions (heliocentric)
    for planet in INTERSTELLAR_INNER_BODIES:
        period = INNER_PLANET_PERIODS_DAYS[planet]
        semi_major = INNER_PLANET_SEMI_MAJOR_AU[planet]

        mean_anomaly = (2 * math.pi * timestamp / period) % (2 * math.pi)
        x = semi_major * math.cos(mean_anomaly)
        y = semi_major * math.sin(mean_anomaly)

        positions[planet] = {
            "type": "inner_planet",
            "semi_major_au": semi_major,
            "period_days": period,
            "mean_anomaly_rad": round(mean_anomaly, 4),
            "x_au": round(x, 4),
            "y_au": round(y, 4),
            "distance_from_sun_au": semi_major,
        }

    # Compute Jovian moon positions (relative to parent, then heliocentric)
    # Jupiter position
    jupiter_period = 4332.59  # days
    jupiter_anomaly = (2 * math.pi * timestamp / jupiter_period) % (2 * math.pi)
    jupiter_x = JUPITER_SEMI_MAJOR_AU * math.cos(jupiter_anomaly)
    jupiter_y = JUPITER_SEMI_MAJOR_AU * math.sin(jupiter_anomaly)

    # Saturn position (for Titan)
    saturn_period = 10759.22  # days
    saturn_anomaly = (2 * math.pi * timestamp / saturn_period) % (2 * math.pi)
    saturn_x = SATURN_SEMI_MAJOR_AU * math.cos(saturn_anomaly)
    saturn_y = SATURN_SEMI_MAJOR_AU * math.sin(saturn_anomaly)

    for moon in INTERSTELLAR_JOVIAN_BODIES:
        period = JOVIAN_MOON_PERIODS_DAYS[moon]
        semi_major = JOVIAN_MOON_SEMI_MAJOR_AU[moon]

        mean_anomaly = (2 * math.pi * timestamp / period) % (2 * math.pi)

        # Titan orbits Saturn, others orbit Jupiter
        if moon == "titan":
            parent_x, parent_y = saturn_x, saturn_y
            parent_distance = SATURN_SEMI_MAJOR_AU
        else:
            parent_x, parent_y = jupiter_x, jupiter_y
            parent_distance = JUPITER_SEMI_MAJOR_AU

        # Moon position relative to parent
        moon_rel_x = semi_major * math.cos(mean_anomaly)
        moon_rel_y = semi_major * math.sin(mean_anomaly)

        # Heliocentric position
        x = parent_x + moon_rel_x
        y = parent_y + moon_rel_y

        positions[moon] = {
            "type": "jovian_moon",
            "parent": "saturn" if moon == "titan" else "jupiter",
            "semi_major_au": semi_major,
            "period_days": period,
            "mean_anomaly_rad": round(mean_anomaly, 4),
            "x_au": round(x, 4),
            "y_au": round(y, 4),
            "distance_from_sun_au": round(parent_distance, 4),
        }

    emit_receipt(
        "interstellar_position",
        {
            "receipt_type": "interstellar_position",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "timestamp_days": timestamp,
            "bodies_computed": list(positions.keys()),
            "payload_hash": dual_hash(json.dumps(positions, sort_keys=True)),
        },
    )

    return positions


# === COMMUNICATION WINDOWS ===


def compute_interstellar_windows(
    bodies: Optional[List[str]] = None, timestamp: float = 0.0
) -> Dict[str, Any]:
    """Compute communication windows between bodies.

    Args:
        bodies: List of bodies to compute windows for (default: all)
        timestamp: Days since epoch (default: 0)

    Returns:
        Dict with communication window information

    Receipt: interstellar_window_receipt
    """
    if bodies is None:
        bodies = INTERSTELLAR_ALL_BODIES

    positions = compute_body_positions(timestamp)

    # Compute windows between all pairs
    windows = {}
    for i, body1 in enumerate(bodies):
        for body2 in bodies[i + 1 :]:
            pos1 = positions.get(body1, {})
            pos2 = positions.get(body2, {})

            # Compute distance between bodies
            dx = pos1.get("x_au", 0) - pos2.get("x_au", 0)
            dy = pos1.get("y_au", 0) - pos2.get("y_au", 0)
            distance_au = math.sqrt(dx * dx + dy * dy)

            # Light time in minutes (1 AU = 8.317 light minutes)
            light_time_min = distance_au * 8.317

            # Communication window quality (inverse of distance)
            window_quality = 1.0 / (1.0 + distance_au)

            window_key = f"{body1}-{body2}"
            windows[window_key] = {
                "body1": body1,
                "body2": body2,
                "distance_au": round(distance_au, 4),
                "light_time_min": round(light_time_min, 2),
                "window_quality": round(window_quality, 4),
                "window_open": window_quality > 0.1,
            }

    emit_receipt(
        "interstellar_window",
        {
            "receipt_type": "interstellar_window",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "timestamp_days": timestamp,
            "windows_computed": len(windows),
            "payload_hash": dual_hash(json.dumps(windows, sort_keys=True)),
        },
    )

    return windows


# === TRANSFER COSTS ===


def compute_transfer_costs(from_body: str, to_body: str) -> Dict[str, Any]:
    """Compute delta-v transfer costs between bodies.

    Uses simplified Hohmann transfer approximation.

    Args:
        from_body: Source body name
        to_body: Destination body name

    Returns:
        Dict with transfer cost information

    Receipt: interstellar_transfer_receipt
    """
    # Get orbital parameters
    if from_body in INNER_PLANET_SEMI_MAJOR_AU:
        r1 = INNER_PLANET_SEMI_MAJOR_AU[from_body]
    elif from_body in JOVIAN_MOON_SEMI_MAJOR_AU:
        # Use parent body distance for moons
        r1 = JUPITER_SEMI_MAJOR_AU if from_body != "titan" else SATURN_SEMI_MAJOR_AU
    else:
        r1 = 1.0  # Default to Earth orbit

    if to_body in INNER_PLANET_SEMI_MAJOR_AU:
        r2 = INNER_PLANET_SEMI_MAJOR_AU[to_body]
    elif to_body in JOVIAN_MOON_SEMI_MAJOR_AU:
        r2 = JUPITER_SEMI_MAJOR_AU if to_body != "titan" else SATURN_SEMI_MAJOR_AU
    else:
        r2 = 1.0

    # Simplified Hohmann transfer delta-v (km/s)
    # Using vis-viva equation approximation
    mu_sun = 1.327e11  # km^3/s^2
    au_km = 1.496e8

    r1_km = r1 * au_km
    r2_km = r2 * au_km

    # Transfer orbit semi-major axis
    a_transfer = (r1_km + r2_km) / 2

    # Delta-v at departure
    v_circular_1 = math.sqrt(mu_sun / r1_km)
    v_transfer_1 = math.sqrt(mu_sun * (2 / r1_km - 1 / a_transfer))
    dv1 = abs(v_transfer_1 - v_circular_1)

    # Delta-v at arrival
    v_circular_2 = math.sqrt(mu_sun / r2_km)
    v_transfer_2 = math.sqrt(mu_sun * (2 / r2_km - 1 / a_transfer))
    dv2 = abs(v_circular_2 - v_transfer_2)

    # Total delta-v
    total_dv = dv1 + dv2

    # Transfer time (Hohmann half-period)
    transfer_time_s = math.pi * math.sqrt(a_transfer**3 / mu_sun)
    transfer_time_days = transfer_time_s / 86400

    result = {
        "from_body": from_body,
        "to_body": to_body,
        "r1_au": r1,
        "r2_au": r2,
        "delta_v_departure_km_s": round(dv1, 3),
        "delta_v_arrival_km_s": round(dv2, 3),
        "total_delta_v_km_s": round(total_dv, 3),
        "transfer_time_days": round(transfer_time_days, 1),
        "transfer_type": "hohmann",
    }

    emit_receipt(
        "interstellar_transfer",
        {
            "receipt_type": "interstellar_transfer",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "from_body": from_body,
            "to_body": to_body,
            "total_delta_v_km_s": result["total_delta_v_km_s"],
            "transfer_time_days": result["transfer_time_days"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === RL COORDINATION ===


def interstellar_rl_step(
    state: Dict[str, Any], action: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute one RL coordination step across the backbone.

    Args:
        state: Current system state
        action: Action to execute

    Returns:
        Dict with new state and reward

    Receipt: interstellar_rl_receipt
    """
    # Extract state components
    body_states = state.get("body_states", {})
    timestamp = state.get("timestamp", 0.0)

    # Execute action (simplified)
    action_type = action.get("type", "noop")
    target_body = action.get("target", None)

    reward = 0.0
    new_body_states = body_states.copy()

    if action_type == "sync" and target_body:
        # Sync increases autonomy slightly
        if target_body in new_body_states:
            current_autonomy = new_body_states[target_body].get("autonomy", 0.9)
            new_autonomy = min(1.0, current_autonomy + INTERSTELLAR_RL_LEARNING_RATE)
            new_body_states[target_body]["autonomy"] = new_autonomy
            reward = new_autonomy - current_autonomy
    elif action_type == "transfer":
        # Resource transfer
        from_body = action.get("from_body")
        to_body = action.get("to_body")
        if from_body and to_body:
            reward = 0.01  # Small reward for successful transfer

    # Advance timestamp
    new_timestamp = timestamp + 1.0  # 1 day step

    new_state = {
        "body_states": new_body_states,
        "timestamp": new_timestamp,
        "last_action": action,
    }

    result = {
        "old_state": state,
        "action": action,
        "new_state": new_state,
        "reward": reward,
        "done": False,
    }

    emit_receipt(
        "interstellar_rl",
        {
            "receipt_type": "interstellar_rl",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action_type": action_type,
            "reward": reward,
            "timestamp": new_timestamp,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === RESOURCE POOLING ===


def simulate_resource_pooling(
    bodies: Optional[List[str]] = None, resource: str = "water_ice"
) -> Dict[str, Any]:
    """Simulate resource pooling across specified bodies.

    Args:
        bodies: List of bodies to pool (default: all)
        resource: Resource type to pool

    Returns:
        Dict with pooling results

    Receipt: interstellar_pool_receipt
    """
    if bodies is None:
        bodies = INTERSTELLAR_ALL_BODIES

    # Find bodies with the resource
    providers = []
    consumers = []

    for body in bodies:
        body_resources = BODY_RESOURCES.get(body, [])
        if resource in body_resources:
            providers.append(body)
        else:
            consumers.append(body)

    # Compute pooling efficiency
    provider_count = len(providers)
    consumer_count = len(consumers)
    total_count = len(bodies)

    # Efficiency based on coverage
    efficiency = provider_count / total_count if total_count > 0 else 0.0

    result = {
        "resource": resource,
        "bodies": bodies,
        "providers": providers,
        "consumers": consumers,
        "provider_count": provider_count,
        "consumer_count": consumer_count,
        "efficiency": round(efficiency, 4),
        "pooling_viable": efficiency >= 0.3,  # At least 30% coverage
    }

    emit_receipt(
        "interstellar_pool",
        {
            "receipt_type": "interstellar_pool",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "resource": resource,
            "provider_count": provider_count,
            "efficiency": result["efficiency"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === AUTONOMY COMPUTATION ===


def compute_backbone_autonomy(
    sync_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute 7-body autonomy across the backbone.

    Args:
        sync_results: Optional sync results to incorporate

    Returns:
        Dict with autonomy metrics

    Receipt: interstellar_autonomy_receipt
    """
    # Initialize body autonomies
    body_autonomies = {}

    # Base autonomy varies by body type
    for body in INTERSTELLAR_JOVIAN_BODIES:
        # Jovian moons have slightly lower base autonomy (farther from Sun)
        body_autonomies[body] = 0.95

    for body in INTERSTELLAR_INNER_BODIES:
        # Inner planets have higher base autonomy
        body_autonomies[body] = 0.97

    # Incorporate sync results if provided
    if sync_results:
        for body, result in sync_results.items():
            if body in body_autonomies:
                sync_boost = result.get("sync_quality", 0) * 0.03
                body_autonomies[body] = min(1.0, body_autonomies[body] + sync_boost)

    # Compute overall autonomy (weighted average)
    total_autonomy = sum(body_autonomies.values())
    avg_autonomy = total_autonomy / len(body_autonomies) if body_autonomies else 0.0

    # Check if target met
    target_met = avg_autonomy >= INTERSTELLAR_AUTONOMY_TARGET

    result = {
        "body_autonomies": body_autonomies,
        "avg_autonomy": round(avg_autonomy, 4),
        "autonomy": round(avg_autonomy, 4),  # Alias for compatibility
        "target": INTERSTELLAR_AUTONOMY_TARGET,
        "target_met": target_met,
        "body_count": len(body_autonomies),
    }

    emit_receipt(
        "interstellar_autonomy",
        {
            "receipt_type": "interstellar_autonomy",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "autonomy": result["autonomy"],
            "target_met": target_met,
            "body_count": result["body_count"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === SIMULATION ===


def simulate_backbone_operations(duration_days: int = 60) -> Dict[str, Any]:
    """Run full backbone simulation for specified duration.

    Args:
        duration_days: Simulation duration in days (default: 60)

    Returns:
        Dict with simulation results

    Receipt: interstellar_sim_receipt
    """
    # Initialize state
    body_states = {}
    for body in INTERSTELLAR_ALL_BODIES:
        body_states[body] = {
            "autonomy": 0.90,
            "resources": BODY_RESOURCES.get(body, []),
            "sync_count": 0,
        }

    state = {
        "body_states": body_states,
        "timestamp": 0.0,
    }

    # Run simulation
    sync_cycles = 0
    total_reward = 0.0

    for day in range(duration_days):
        # Periodic sync (every SYNC_INTERVAL)
        if day % INTERSTELLAR_SYNC_INTERVAL_DAYS == 0 and day > 0:
            sync_cycles += 1
            for body in INTERSTELLAR_ALL_BODIES:
                action = {"type": "sync", "target": body}
                result = interstellar_rl_step(state, action)
                state = result["new_state"]
                total_reward += result["reward"]

        # Daily operations
        state["timestamp"] = float(day)

    # Final autonomy computation
    autonomy_result = compute_backbone_autonomy()

    result = {
        "duration_days": duration_days,
        "sync_cycles": sync_cycles,
        "total_reward": round(total_reward, 6),
        "final_autonomy": autonomy_result["autonomy"],
        "target_met": autonomy_result["target_met"],
        "bodies_simulated": len(INTERSTELLAR_ALL_BODIES),
        "simulation_complete": True,
    }

    emit_receipt(
        "interstellar_sim",
        {
            "receipt_type": "interstellar_sim",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "duration_days": duration_days,
            "sync_cycles": sync_cycles,
            "final_autonomy": result["final_autonomy"],
            "target_met": result["target_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === D14 HYBRID INTEGRATION ===


def d14_interstellar_hybrid(
    tree_size: int = 10**12, base_alpha: float = 3.41, simulate: bool = False
) -> Dict[str, Any]:
    """Run integrated D14 fractal + interstellar backbone.

    Args:
        tree_size: Tree size for D14 fractal
        base_alpha: Base alpha for D14 fractal
        simulate: Whether to run in simulation mode

    Returns:
        Dict with hybrid results

    Receipt: d14_interstellar_hybrid_receipt
    """
    from .fractal_layers import d14_push

    # Run D14 fractal
    d14_result = d14_push(tree_size=tree_size, base_alpha=base_alpha, simulate=simulate)

    # Run interstellar backbone
    backbone_result = simulate_backbone_operations(duration_days=60)
    autonomy_result = compute_backbone_autonomy()

    # Combine results
    result = {
        "mode": "simulate" if simulate else "execute",
        "d14_result": {
            "eff_alpha": d14_result["eff_alpha"],
            "floor_met": d14_result["floor_met"],
            "target_met": d14_result["target_met"],
            "ceiling_met": d14_result["ceiling_met"],
        },
        "backbone_result": {
            "sync_cycles": backbone_result["sync_cycles"],
            "autonomy": autonomy_result["autonomy"],
            "target_met": autonomy_result["target_met"],
        },
        "combined_alpha": d14_result["eff_alpha"],
        "combined_autonomy": autonomy_result["autonomy"],
        "integration_status": "operational"
        if d14_result["target_met"] and autonomy_result["target_met"]
        else "partial",
        "gate": "t24h",
    }

    emit_receipt(
        "d14_interstellar_hybrid",
        {
            "receipt_type": "d14_interstellar_hybrid",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "eff_alpha": result["combined_alpha"],
            "autonomy": result["combined_autonomy"],
            "integration_status": result["integration_status"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === EMERGENCY FAILOVER ===


def emergency_failover(failed_body: str) -> Dict[str, Any]:
    """Handle body failure with emergency failover.

    Args:
        failed_body: Name of the failed body

    Returns:
        Dict with failover results

    Receipt: interstellar_failover_receipt
    """
    if failed_body not in INTERSTELLAR_ALL_BODIES:
        return {"error": f"Unknown body: {failed_body}", "failover_success": False}

    # Identify backup bodies
    if failed_body in INTERSTELLAR_JOVIAN_BODIES:
        # Failover to another Jovian moon
        backups = [b for b in INTERSTELLAR_JOVIAN_BODIES if b != failed_body]
    else:
        # Failover to another inner planet
        backups = [b for b in INTERSTELLAR_INNER_BODIES if b != failed_body]

    # Select primary backup
    primary_backup = backups[0] if backups else None

    # Compute failover metrics
    if primary_backup:
        # Get resources that need to be covered
        failed_resources = BODY_RESOURCES.get(failed_body, [])
        backup_resources = BODY_RESOURCES.get(primary_backup, [])

        # Coverage ratio
        covered = sum(1 for r in failed_resources if r in backup_resources)
        coverage_ratio = covered / len(failed_resources) if failed_resources else 1.0

        failover_success = coverage_ratio >= 0.5
    else:
        coverage_ratio = 0.0
        failover_success = False

    result = {
        "failed_body": failed_body,
        "primary_backup": primary_backup,
        "all_backups": backups,
        "coverage_ratio": round(coverage_ratio, 4),
        "failover_success": failover_success,
        "remaining_bodies": [b for b in INTERSTELLAR_ALL_BODIES if b != failed_body],
        "remaining_count": len(INTERSTELLAR_ALL_BODIES) - 1,
    }

    emit_receipt(
        "interstellar_failover",
        {
            "receipt_type": "interstellar_failover",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "failed_body": failed_body,
            "primary_backup": primary_backup,
            "failover_success": failover_success,
            "coverage_ratio": result["coverage_ratio"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === CROSS-SYSTEM TRANSFER ===


def jovian_inner_handoff(
    data: Dict[str, Any], direction: str = "jovian_to_inner"
) -> Dict[str, Any]:
    """Execute cross-system data/resource handoff.

    Args:
        data: Data/resource package to transfer
        direction: "jovian_to_inner" or "inner_to_jovian"

    Returns:
        Dict with handoff results

    Receipt: interstellar_handoff_receipt
    """
    if direction == "jovian_to_inner":
        source_bodies = INTERSTELLAR_JOVIAN_BODIES
        dest_bodies = INTERSTELLAR_INNER_BODIES
    else:
        source_bodies = INTERSTELLAR_INNER_BODIES
        dest_bodies = INTERSTELLAR_JOVIAN_BODIES

    # Compute handoff metrics
    # Use first body in each group as representative
    source_body = source_bodies[0]
    dest_body = dest_bodies[0]

    # Get transfer costs
    transfer = compute_transfer_costs(source_body, dest_body)

    # Compute latency (based on distance)
    positions = compute_body_positions()
    source_pos = positions.get(source_body, {})
    dest_pos = positions.get(dest_body, {})

    dx = source_pos.get("x_au", 0) - dest_pos.get("x_au", 0)
    dy = source_pos.get("y_au", 0) - dest_pos.get("y_au", 0)
    distance_au = math.sqrt(dx * dx + dy * dy)
    latency_min = distance_au * 8.317  # Light minutes

    result = {
        "direction": direction,
        "source_bodies": source_bodies,
        "dest_bodies": dest_bodies,
        "representative_source": source_body,
        "representative_dest": dest_body,
        "transfer_delta_v_km_s": transfer["total_delta_v_km_s"],
        "transfer_time_days": transfer["transfer_time_days"],
        "light_time_latency_min": round(latency_min, 2),
        "data_size": len(json.dumps(data)),
        "handoff_success": True,
    }

    emit_receipt(
        "interstellar_handoff",
        {
            "receipt_type": "interstellar_handoff",
            "tenant_id": INTERSTELLAR_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "direction": direction,
            "latency_min": result["light_time_latency_min"],
            "handoff_success": result["handoff_success"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result
