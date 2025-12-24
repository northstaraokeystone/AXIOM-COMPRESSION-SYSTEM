"""Solar System inner planet orbital RL coordination hub.

PARADIGM:
    Orbital Reinforcement Learning hub for Venus, Mercury, and Mars coordination.
    Manages communication windows, transfer orbits, and resource sharing.

THE PHYSICS:
    - Venus orbital period: 225 days (0.723 AU)
    - Mercury orbital period: 88 days (0.387 AU)
    - Mars orbital period: 687 days (1.524 AU)
    - Maximum latency: 22 min (Mars at conjunction)
    - Sync interval: 30 days

ORBITAL MECHANICS:
    Simplified Keplerian model for coordination windows.
    Communication windows computed via angular separation.
    Hohmann transfers for resource movement.

Source: Grok - "Solar orbital hub: Venus/Mercury/Mars coordination viable"
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

SOLAR_HUB_TENANT_ID = "spaceproof-solar-hub"
"""Tenant ID for Solar hub receipts."""

SOLAR_HUB_PLANETS = ["venus", "mercury", "mars"]
"""Inner planets in the Solar hub."""

ORBITAL_SYNC_INTERVAL_DAYS = 30
"""Coordination window interval in days."""

ORBITAL_RL_LEARNING_RATE = 0.0005
"""RL learning rate for orbital coordination."""

SOLAR_HUB_AUTONOMY_TARGET = 0.95
"""System-level autonomy target."""

VENUS_ORBITAL_PERIOD_DAYS = 225
"""Venus orbital period in days."""

MERCURY_ORBITAL_PERIOD_DAYS = 88
"""Mercury orbital period in days."""

MARS_ORBITAL_PERIOD_DAYS = 687
"""Mars orbital period in days."""

MAX_INNER_SYSTEM_LATENCY_MIN = 22
"""Maximum latency in minutes (Mars at conjunction)."""

SEMI_MAJOR_AXIS_AU = {
    "venus": 0.723,
    "mercury": 0.387,
    "mars": 1.524,
    "earth": 1.0,
}
"""Semi-major axis in AU for each body."""

ORBITAL_PERIODS_DAYS = {
    "venus": VENUS_ORBITAL_PERIOD_DAYS,
    "mercury": MERCURY_ORBITAL_PERIOD_DAYS,
    "mars": MARS_ORBITAL_PERIOD_DAYS,
}
"""Orbital periods in days."""

PLANET_RESOURCES = {
    "venus": ["atmospheric_chemicals", "solar_energy"],
    "mercury": ["metals", "solar_energy", "thermal_gradient"],
    "mars": ["water_ice", "co2", "regolith"],
}
"""Available resources per planet."""


# === CONFIGURATION FUNCTIONS ===


def load_solar_hub_config() -> Dict[str, Any]:
    """Load Solar hub configuration from d13_solar_spec.json.

    Returns:
        Dict with Solar hub configuration

    Receipt: solar_hub_config_receipt
    """
    import os

    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d13_solar_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("solar_hub_config", {})

    emit_receipt(
        "solar_hub_config",
        {
            "receipt_type": "solar_hub_config",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "planets": config.get("planets", SOLAR_HUB_PLANETS),
            "sync_interval_days": config.get(
                "sync_interval_days", ORBITAL_SYNC_INTERVAL_DAYS
            ),
            "autonomy_target": config.get("autonomy_target", SOLAR_HUB_AUTONOMY_TARGET),
            "coordination_mode": config.get("coordination_mode", "orbital_rl"),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


def get_solar_hub_info() -> Dict[str, Any]:
    """Get Solar hub configuration summary.

    Returns:
        Dict with Solar hub info

    Receipt: solar_hub_info_receipt
    """
    config = load_solar_hub_config()

    info = {
        "planets": SOLAR_HUB_PLANETS,
        "orbital_periods_days": ORBITAL_PERIODS_DAYS,
        "semi_major_axis_au": SEMI_MAJOR_AXIS_AU,
        "sync_interval_days": ORBITAL_SYNC_INTERVAL_DAYS,
        "rl_learning_rate": ORBITAL_RL_LEARNING_RATE,
        "autonomy_target": SOLAR_HUB_AUTONOMY_TARGET,
        "max_latency_min": MAX_INNER_SYSTEM_LATENCY_MIN,
        "resources": PLANET_RESOURCES,
        "config": config,
    }

    emit_receipt(
        "solar_hub_info",
        {
            "receipt_type": "solar_hub_info",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "planets": SOLAR_HUB_PLANETS,
            "autonomy_target": SOLAR_HUB_AUTONOMY_TARGET,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === ORBITAL POSITION COMPUTATION ===


def compute_orbital_positions(timestamp: float = 0.0) -> Dict[str, Any]:
    """Compute orbital positions of inner planets.

    Uses simplified Keplerian model with mean anomaly.

    Args:
        timestamp: Days since epoch (default: 0)

    Returns:
        Dict with orbital positions for each planet

    Receipt: solar_hub_position_receipt
    """
    positions = {}

    for planet in SOLAR_HUB_PLANETS:
        period = ORBITAL_PERIODS_DAYS[planet]
        semi_major = SEMI_MAJOR_AXIS_AU[planet]

        # Mean anomaly (simplified circular orbit)
        mean_anomaly = (2 * math.pi * timestamp / period) % (2 * math.pi)

        # Position in heliocentric coordinates
        x = semi_major * math.cos(mean_anomaly)
        y = semi_major * math.sin(mean_anomaly)

        positions[planet] = {
            "semi_major_au": semi_major,
            "period_days": period,
            "mean_anomaly_rad": round(mean_anomaly, 4),
            "x_au": round(x, 4),
            "y_au": round(y, 4),
            "distance_from_sun_au": semi_major,  # Circular approximation
        }

    # Compute Earth position for reference
    earth_anomaly = (2 * math.pi * timestamp / 365.25) % (2 * math.pi)
    positions["earth"] = {
        "semi_major_au": 1.0,
        "period_days": 365.25,
        "mean_anomaly_rad": round(earth_anomaly, 4),
        "x_au": round(math.cos(earth_anomaly), 4),
        "y_au": round(math.sin(earth_anomaly), 4),
        "distance_from_sun_au": 1.0,
    }

    result = {
        "timestamp_days": timestamp,
        "positions": positions,
        "reference_frame": "heliocentric",
    }

    emit_receipt(
        "solar_hub_position",
        {
            "receipt_type": "solar_hub_position",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "timestamp_days": timestamp,
            "planets_computed": list(positions.keys()),
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_planet_distance(
    planet1: str, planet2: str, timestamp: float = 0.0
) -> float:
    """Compute distance between two planets.

    Args:
        planet1: First planet name
        planet2: Second planet name
        timestamp: Days since epoch

    Returns:
        Distance in AU
    """
    positions = compute_orbital_positions(timestamp)

    p1 = positions["positions"].get(planet1, {})
    p2 = positions["positions"].get(planet2, {})

    if not p1 or not p2:
        return 0.0

    dx = p1["x_au"] - p2["x_au"]
    dy = p1["y_au"] - p2["y_au"]

    return math.sqrt(dx**2 + dy**2)


def compute_communication_latency(
    planet1: str, planet2: str, timestamp: float = 0.0
) -> float:
    """Compute one-way communication latency in minutes.

    Light travels at ~8.317 minutes per AU.

    Args:
        planet1: First planet
        planet2: Second planet
        timestamp: Days since epoch

    Returns:
        Latency in minutes
    """
    distance_au = compute_planet_distance(planet1, planet2, timestamp)
    latency_min = distance_au * 8.317  # minutes per AU
    return round(latency_min, 2)


# === COMMUNICATION WINDOWS ===


def compute_communication_windows(
    planets: Optional[List[str]] = None, duration_days: int = 365
) -> Dict[str, Any]:
    """Compute communication windows between planets.

    Args:
        planets: List of planets to analyze (default: all)
        duration_days: Analysis duration

    Returns:
        Dict with communication window analysis

    Receipt: solar_hub_window_receipt
    """
    if planets is None:
        planets = SOLAR_HUB_PLANETS

    windows = {}
    min_latencies = {}
    max_latencies = {}

    # Sample every 10 days
    sample_interval = 10

    for i, p1 in enumerate(planets):
        for p2 in planets[i + 1 :]:
            pair_key = f"{p1}_{p2}"
            latencies = []

            for day in range(0, duration_days, sample_interval):
                lat = compute_communication_latency(p1, p2, day)
                latencies.append({"day": day, "latency_min": lat})

            min_lat = min(lat["latency_min"] for lat in latencies)
            max_lat = max(lat["latency_min"] for lat in latencies)
            avg_lat = sum(lat["latency_min"] for lat in latencies) / len(latencies)

            windows[pair_key] = {
                "min_latency_min": round(min_lat, 2),
                "max_latency_min": round(max_lat, 2),
                "avg_latency_min": round(avg_lat, 2),
                "sample_count": len(latencies),
            }
            min_latencies[pair_key] = min_lat
            max_latencies[pair_key] = max_lat

    # Earth-to-planet windows
    for planet in planets:
        pair_key = f"earth_{planet}"
        latencies = []

        for day in range(0, duration_days, sample_interval):
            lat = compute_communication_latency("earth", planet, day)
            latencies.append({"day": day, "latency_min": lat})

        min_lat = min(lat["latency_min"] for lat in latencies)
        max_lat = max(lat["latency_min"] for lat in latencies)
        avg_lat = sum(lat["latency_min"] for lat in latencies) / len(latencies)

        windows[pair_key] = {
            "min_latency_min": round(min_lat, 2),
            "max_latency_min": round(max_lat, 2),
            "avg_latency_min": round(avg_lat, 2),
            "sample_count": len(latencies),
        }

    result = {
        "planets": planets,
        "duration_days": duration_days,
        "windows": windows,
        "overall_max_latency_min": max(max_latencies.values()) if max_latencies else 0,
    }

    emit_receipt(
        "solar_hub_window",
        {
            "receipt_type": "solar_hub_window",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "planets": planets,
            "duration_days": duration_days,
            "window_count": len(windows),
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === TRANSFER WINDOWS ===


def compute_transfer_windows(
    from_planet: str, to_planet: str, duration_days: int = 365
) -> Dict[str, Any]:
    """Compute Hohmann transfer windows between planets.

    Args:
        from_planet: Origin planet
        to_planet: Destination planet
        duration_days: Analysis duration

    Returns:
        Dict with transfer window analysis

    Receipt: solar_hub_transfer_window_receipt
    """
    r1 = SEMI_MAJOR_AXIS_AU.get(from_planet, 1.0)
    r2 = SEMI_MAJOR_AXIS_AU.get(to_planet, 1.0)

    # Hohmann transfer semi-major axis
    a_transfer = (r1 + r2) / 2

    # Transfer time (half of elliptical orbit period)
    # T = 2*pi*sqrt(a^3/GM), GM_sun = 1 in AU^3/year^2
    transfer_period_years = math.sqrt(a_transfer**3)
    transfer_time_days = transfer_period_years * 365.25 / 2

    # Synodic period for window frequency
    p1 = ORBITAL_PERIODS_DAYS.get(from_planet, 365.25)
    p2 = ORBITAL_PERIODS_DAYS.get(to_planet, 365.25)

    if p1 != p2:
        synodic_period = abs(1 / (1 / p1 - 1 / p2))
    else:
        synodic_period = float("inf")

    # Delta-V estimate (simplified)
    mu = 1.0  # GM_sun normalized
    v1_circular = math.sqrt(mu / r1)
    v2_circular = math.sqrt(mu / r2)
    v_perihelion = math.sqrt(2 * mu * r2 / (r1 * (r1 + r2)))
    v_aphelion = math.sqrt(2 * mu * r1 / (r2 * (r1 + r2)))

    delta_v1 = abs(v_perihelion - v1_circular)
    delta_v2 = abs(v2_circular - v_aphelion)
    total_delta_v = delta_v1 + delta_v2

    result = {
        "from_planet": from_planet,
        "to_planet": to_planet,
        "r1_au": r1,
        "r2_au": r2,
        "transfer_semi_major_au": round(a_transfer, 4),
        "transfer_time_days": round(transfer_time_days, 1),
        "synodic_period_days": round(synodic_period, 1),
        "delta_v_au_per_year": round(total_delta_v, 4),
        "window_frequency_per_year": round(365.25 / synodic_period, 2)
        if synodic_period < float("inf")
        else 0,
    }

    emit_receipt(
        "solar_hub_transfer_window",
        {
            "receipt_type": "solar_hub_transfer_window",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "from_planet": from_planet,
            "to_planet": to_planet,
            "transfer_time_days": result["transfer_time_days"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === ORBITAL RL COORDINATION ===


class OrbitalRLNetwork:
    """Simple RL network for orbital coordination."""

    def __init__(self, learning_rate: float = ORBITAL_RL_LEARNING_RATE):
        self.learning_rate = learning_rate
        self.policy_weights = {
            planet: 1.0 / len(SOLAR_HUB_PLANETS) for planet in SOLAR_HUB_PLANETS
        }
        self.value_estimates = {planet: 0.5 for planet in SOLAR_HUB_PLANETS}
        self.episode_count = 0

    def step(self, state: Dict[str, Any], reward: float) -> Dict[str, float]:
        """Update policy based on reward."""
        self.episode_count += 1

        # Update value estimates
        for planet in SOLAR_HUB_PLANETS:
            old_value = self.value_estimates[planet]
            self.value_estimates[planet] = old_value + self.learning_rate * (
                reward - old_value
            )

        # Normalize policy weights
        total = sum(self.value_estimates.values())
        if total > 0:
            self.policy_weights = {
                p: v / total for p, v in self.value_estimates.items()
            }

        return self.policy_weights


def orbital_rl_step(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one RL coordination step.

    Args:
        state: Current system state
        action: Action to take

    Returns:
        Dict with step results

    Receipt: solar_hub_rl_step_receipt
    """
    # Compute reward based on coordination efficiency
    efficiency = state.get("efficiency", 0.8)
    latency_factor = 1.0 - (state.get("latency_min", 10) / MAX_INNER_SYSTEM_LATENCY_MIN)
    reward = efficiency * 0.7 + latency_factor * 0.3

    # Update network
    network = OrbitalRLNetwork()
    new_weights = network.step(state, reward)

    result = {
        "state": state,
        "action": action,
        "reward": round(reward, 4),
        "new_weights": new_weights,
        "episode": network.episode_count,
        "learning_rate": network.learning_rate,
    }

    emit_receipt(
        "solar_hub_rl_step",
        {
            "receipt_type": "solar_hub_rl_step",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "reward": result["reward"],
            "episode": result["episode"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === RESOURCE TRANSFER ===


def simulate_resource_transfer(
    from_planet: str, to_planet: str, resource: str, amount_kg: float = 1000.0
) -> Dict[str, Any]:
    """Simulate resource transfer between planets.

    Args:
        from_planet: Origin planet
        to_planet: Destination planet
        resource: Resource type
        amount_kg: Amount in kg

    Returns:
        Dict with transfer simulation results

    Receipt: solar_hub_transfer_receipt
    """
    # Check resource availability
    available_resources = PLANET_RESOURCES.get(from_planet, [])
    resource_available = resource in available_resources

    # Get transfer window
    transfer = compute_transfer_windows(from_planet, to_planet)
    transfer_time = transfer["transfer_time_days"]

    # Compute transfer cost (simplified)
    delta_v = transfer["delta_v_au_per_year"]
    fuel_ratio = 0.1 * delta_v  # Simplified mass ratio
    fuel_kg = amount_kg * fuel_ratio

    result = {
        "from_planet": from_planet,
        "to_planet": to_planet,
        "resource": resource,
        "amount_kg": amount_kg,
        "resource_available": resource_available,
        "transfer_time_days": transfer_time,
        "fuel_required_kg": round(fuel_kg, 2),
        "efficiency": 0.95 if resource_available else 0.0,
        "feasible": resource_available,
    }

    emit_receipt(
        "solar_hub_transfer",
        {
            "receipt_type": "solar_hub_transfer",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "from_planet": from_planet,
            "to_planet": to_planet,
            "resource": resource,
            "feasible": result["feasible"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === AUTONOMY COMPUTATION ===


def compute_hub_autonomy(sync_results: Optional[Dict[str, Any]] = None) -> float:
    """Compute system-level autonomy for Solar hub.

    Args:
        sync_results: Optional sync results to include

    Returns:
        Autonomy level (0-1)

    Receipt: solar_hub_autonomy_receipt
    """
    # Base autonomy per planet
    planet_autonomy = {
        "venus": 0.99,  # Extreme environment, high autonomy required
        "mercury": 0.97,  # High thermal stress, high autonomy
        "mars": 0.85,  # Proven ops, moderate autonomy
    }

    # Weighted average based on operational complexity
    weights = {
        "venus": 0.35,
        "mercury": 0.30,
        "mars": 0.35,
    }

    weighted_autonomy = sum(planet_autonomy[p] * weights[p] for p in SOLAR_HUB_PLANETS)

    # Apply sync efficiency bonus if provided
    if sync_results:
        sync_efficiency = sync_results.get("efficiency", 0.9)
        weighted_autonomy = weighted_autonomy * 0.8 + sync_efficiency * 0.2

    # Clamp to target
    autonomy = min(weighted_autonomy, SOLAR_HUB_AUTONOMY_TARGET)

    result = {
        "planet_autonomy": planet_autonomy,
        "weights": weights,
        "weighted_autonomy": round(weighted_autonomy, 4),
        "final_autonomy": round(autonomy, 4),
        "target": SOLAR_HUB_AUTONOMY_TARGET,
        "target_met": autonomy >= SOLAR_HUB_AUTONOMY_TARGET,
    }

    emit_receipt(
        "solar_hub_autonomy",
        {
            "receipt_type": "solar_hub_autonomy",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "autonomy": result["final_autonomy"],
            "target_met": result["target_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return autonomy


# === HUB SIMULATION ===


def simulate_hub_operations(duration_days: int = 365) -> Dict[str, Any]:
    """Simulate full Solar hub operations.

    Args:
        duration_days: Simulation duration in days

    Returns:
        Dict with simulation results

    Receipt: solar_hub_simulate_receipt
    """
    # Compute positions at start
    positions = compute_orbital_positions(0)

    # Compute communication windows
    windows = compute_communication_windows(duration_days=duration_days)

    # Compute transfer opportunities
    transfers = {}
    for i, p1 in enumerate(SOLAR_HUB_PLANETS):
        for p2 in SOLAR_HUB_PLANETS[i + 1 :]:
            transfers[f"{p1}_to_{p2}"] = compute_transfer_windows(p1, p2, duration_days)

    # Compute sync cycles
    sync_cycles = duration_days // ORBITAL_SYNC_INTERVAL_DAYS

    # Run RL episodes
    rl_rewards = []
    for cycle in range(min(sync_cycles, 10)):  # Sample first 10 cycles
        state = {"efficiency": 0.85 + 0.1 * (cycle / 10), "latency_min": 15 - cycle}
        action = {"sync": True, "cycle": cycle}
        step_result = orbital_rl_step(state, action)
        rl_rewards.append(step_result["reward"])

    avg_reward = sum(rl_rewards) / len(rl_rewards) if rl_rewards else 0

    # Compute autonomy
    autonomy = compute_hub_autonomy()

    result = {
        "duration_days": duration_days,
        "sync_cycles": sync_cycles,
        "planets": SOLAR_HUB_PLANETS,
        "initial_positions": positions,
        "communication_windows": windows,
        "transfer_opportunities": transfers,
        "rl_episodes": len(rl_rewards),
        "avg_rl_reward": round(avg_reward, 4),
        "autonomy": autonomy,
        "autonomy_met": autonomy >= SOLAR_HUB_AUTONOMY_TARGET,
        "hub_operational": True,
    }

    emit_receipt(
        "solar_hub_simulate",
        {
            "receipt_type": "solar_hub_simulate",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "duration_days": duration_days,
            "sync_cycles": sync_cycles,
            "autonomy": autonomy,
            "hub_operational": result["hub_operational"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === D13 SOLAR HYBRID ===


def d13_solar_hybrid(
    tree_size: int, base_alpha: float, simulate: bool = False
) -> Dict[str, Any]:
    """Integrated D13 fractal + Solar hub hybrid.

    Args:
        tree_size: Tree size for fractal
        base_alpha: Base alpha value
        simulate: Whether to run in simulation mode

    Returns:
        Dict with hybrid results

    Receipt: d13_solar_hybrid_receipt
    """
    # Import D13 functions
    from .fractal_layers import d13_push

    # Run D13 push
    d13_result = d13_push(tree_size, base_alpha, simulate=simulate)

    # Run Solar hub simulation
    hub_result = simulate_hub_operations(duration_days=365)

    # Combined metrics
    combined_autonomy = (
        d13_result.get("autonomy", 0.95) * 0.5 + hub_result["autonomy"] * 0.5
    )

    result = {
        "mode": "simulate" if simulate else "execute",
        "d13_result": {
            "eff_alpha": d13_result["eff_alpha"],
            "floor_met": d13_result["floor_met"],
            "target_met": d13_result["target_met"],
        },
        "hub_result": {
            "autonomy": hub_result["autonomy"],
            "sync_cycles": hub_result["sync_cycles"],
            "hub_operational": hub_result["hub_operational"],
        },
        "combined_autonomy": round(combined_autonomy, 4),
        "integration_status": "operational",
        "gate": "t24h",
    }

    emit_receipt(
        "d13_solar_hybrid",
        {
            "receipt_type": "d13_solar_hybrid",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "eff_alpha": d13_result["eff_alpha"],
            "hub_autonomy": hub_result["autonomy"],
            "combined_autonomy": result["combined_autonomy"],
            "integration_status": result["integration_status"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === JOVIAN COORDINATION (FUTURE) ===


def coordinate_with_jovian(
    jovian_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coordinate inner Solar hub with Jovian moons (future phase).

    Args:
        jovian_state: Optional Jovian system state

    Returns:
        Dict with coordination results

    Receipt: solar_jovian_coordinate_receipt
    """
    if jovian_state is None:
        jovian_state = {
            "moons": ["titan", "europa", "ganymede", "callisto"],
            "autonomy": 0.97,
            "hub_location": "callisto",
        }

    # Inner system state
    inner_autonomy = compute_hub_autonomy()

    # Combined system autonomy (weighted by distance)
    # Jovian is further, weight more heavily
    combined = inner_autonomy * 0.4 + jovian_state["autonomy"] * 0.6

    result = {
        "inner_planets": SOLAR_HUB_PLANETS,
        "inner_autonomy": inner_autonomy,
        "jovian_moons": jovian_state["moons"],
        "jovian_autonomy": jovian_state["autonomy"],
        "jovian_hub": jovian_state["hub_location"],
        "combined_autonomy": round(combined, 4),
        "full_system_operational": combined >= 0.95,
        "next_phase": "Full Solar System coordination",
    }

    emit_receipt(
        "solar_jovian_coordinate",
        {
            "receipt_type": "solar_jovian_coordinate",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "inner_autonomy": inner_autonomy,
            "jovian_autonomy": jovian_state["autonomy"],
            "combined_autonomy": result["combined_autonomy"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === EMERGENCY PROTOCOL ===


def emergency_protocol(failure_planet: str) -> Dict[str, Any]:
    """Handle failure of one planet in the hub.

    Args:
        failure_planet: Planet that has failed

    Returns:
        Dict with failover results

    Receipt: solar_hub_emergency_receipt
    """
    remaining_planets = [p for p in SOLAR_HUB_PLANETS if p != failure_planet]

    # Compute degraded autonomy
    degraded_autonomy = compute_hub_autonomy() * (
        len(remaining_planets) / len(SOLAR_HUB_PLANETS)
    )

    # Reroute resources
    reroute_plan = {}
    failed_resources = PLANET_RESOURCES.get(failure_planet, [])

    for resource in failed_resources:
        for planet in remaining_planets:
            if resource in PLANET_RESOURCES.get(planet, []):
                reroute_plan[resource] = planet
                break

    result = {
        "failure_planet": failure_planet,
        "remaining_planets": remaining_planets,
        "degraded_autonomy": round(degraded_autonomy, 4),
        "reroute_plan": reroute_plan,
        "system_operational": degraded_autonomy >= 0.80,
        "recovery_actions": [
            f"Reroute {r} from {failure_planet} to {p}" for r, p in reroute_plan.items()
        ],
    }

    emit_receipt(
        "solar_hub_emergency",
        {
            "receipt_type": "solar_hub_emergency",
            "tenant_id": SOLAR_HUB_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "failure_planet": failure_planet,
            "degraded_autonomy": result["degraded_autonomy"],
            "system_operational": result["system_operational"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result
