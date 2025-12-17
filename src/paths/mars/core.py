"""Mars habitat autonomous optimization core logic.

Evolution path: stub -> simulate -> optimize -> autonomous

CURRENT STAGE: stub
- stub_status(): Returns current stub status
- simulate_dome(): Placeholder for dome simulation
- compute_isru_closure(): ISRU closure ratio calculation
- compute_sovereignty(): Computational sovereignty check
- optimize_resources(): RL optimization placeholder

PHYSICS BASIS:
- ISRU (In-Situ Resource Utilization) closure target: 85%
- Remaining 15% from Earth uplift (diminishing over time)
- Decision rate: 1000 bps for autonomous operations
- Sovereignty threshold: internal_rate > external_rate

Source: AXIOM scalable paths architecture - Mars autonomous habitat
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..base import emit_path_receipt, load_path_spec, PathStopRule


# === CONSTANTS ===

MARS_TENANT_ID = "axiom-mars"
"""Tenant ID for Mars path receipts."""

DEFAULT_CREW = 50
"""Default crew size for simulations."""

ISRU_CLOSURE_TARGET = 0.85
"""Target ISRU closure ratio (85% self-sufficient)."""

ISRU_UPLIFT_TARGET = 0.15
"""Target Earth uplift ratio (15% from Earth)."""

DECISION_RATE_TARGET_BPS = 1000
"""Target decision rate in bits per second."""

DOME_RESOURCES = ["water", "o2", "power", "food"]
"""Resources tracked in dome simulation."""


# === STUB STATUS ===

def stub_status() -> Dict[str, Any]:
    """Return current stub status.

    Returns:
        Dict with stub readiness info

    Receipt: mars_status
    """
    spec = load_path_spec("mars")

    status = {
        "ready": True,
        "stage": "stub",
        "version": spec.get("version", "0.1.0"),
        "evolution_path": ["stub", "simulate", "optimize", "autonomous"],
        "current_capabilities": [
            "stub_status",
            "compute_isru_closure",
            "compute_sovereignty"
        ],
        "pending_capabilities": [
            "simulate_dome",
            "optimize_resources"
        ],
        "config": spec.get("config", {}),
        "tenant_id": MARS_TENANT_ID
    }

    emit_path_receipt("mars", "status", status)
    return status


# === DOME SIMULATION (STUB) ===

def simulate_dome(
    crew: int = DEFAULT_CREW,
    duration_days: int = 365,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Dome simulation placeholder.

    STUB: Returns projected values based on linear models.
    FULL: Will integrate with fractal compression and RL optimization.

    Args:
        crew: Number of crew members
        duration_days: Simulation duration in days
        config: Optional override config

    Returns:
        Dict with simulation results

    Receipt: mars_dome
    """
    if config is None:
        config = {}

    # Stub: Linear resource consumption model
    # Full: Fractal optimization with RL feedback
    daily_water_per_crew = 3.0  # liters
    daily_o2_per_crew = 0.84  # kg
    daily_power_per_crew = 10.0  # kWh
    daily_food_per_crew = 2.0  # kg

    total_water = crew * duration_days * daily_water_per_crew
    total_o2 = crew * duration_days * daily_o2_per_crew
    total_power = crew * duration_days * daily_power_per_crew
    total_food = crew * duration_days * daily_food_per_crew

    # ISRU projections (stub values)
    isru_water = total_water * 0.75  # 75% from ice extraction
    isru_o2 = total_o2 * 0.90  # 90% from electrolysis
    isru_power = total_power * 0.95  # 95% solar/nuclear
    isru_food = total_food * 0.30  # 30% greenhouse (improving)

    result = {
        "stub_mode": True,
        "crew": crew,
        "duration_days": duration_days,
        "resources_required": {
            "water_liters": round(total_water, 1),
            "o2_kg": round(total_o2, 1),
            "power_kwh": round(total_power, 1),
            "food_kg": round(total_food, 1)
        },
        "isru_projected": {
            "water_liters": round(isru_water, 1),
            "o2_kg": round(isru_o2, 1),
            "power_kwh": round(isru_power, 1),
            "food_kg": round(isru_food, 1)
        },
        "isru_closure_projected": round(
            (isru_water + isru_o2 + isru_power + isru_food) /
            (total_water + total_o2 + total_power + total_food),
            3
        ),
        "next_stage": "Full fractal optimization integration",
        "tenant_id": MARS_TENANT_ID
    }

    emit_path_receipt("mars", "dome", result)
    return result


# === ISRU COMPUTATION ===

def compute_isru_closure(resources: Dict[str, float]) -> float:
    """Compute ISRU closure ratio.

    ISRU closure = local_production / total_consumption

    Args:
        resources: Dict with {resource: (local, total)} values

    Returns:
        Closure ratio (0.0 to 1.0)

    Receipt: mars_isru
    """
    total_local = 0.0
    total_required = 0.0

    for resource, values in resources.items():
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            local, required = values[0], values[1]
        elif isinstance(values, dict):
            local = values.get("local", 0)
            required = values.get("required", 0)
        else:
            continue

        total_local += local
        total_required += required

    if total_required == 0:
        closure = 0.0
    else:
        closure = total_local / total_required

    closure = min(1.0, max(0.0, closure))

    result = {
        "closure_ratio": round(closure, 4),
        "target": ISRU_CLOSURE_TARGET,
        "gap": round(ISRU_CLOSURE_TARGET - closure, 4),
        "target_met": closure >= ISRU_CLOSURE_TARGET,
        "uplift_required": round(1.0 - closure, 4),
        "tenant_id": MARS_TENANT_ID
    }

    emit_path_receipt("mars", "isru", result)
    return closure


# === SOVEREIGNTY COMPUTATION ===

def compute_sovereignty(
    crew: int,
    bandwidth_mbps: float = 100.0,
    latency_s: float = 1200.0
) -> bool:
    """Computational sovereignty check.

    Sovereignty = internal_rate > external_rate

    Internal rate: crew * decision_rate_per_person
    External rate: bandwidth / (latency * 2)  # round-trip

    Args:
        crew: Number of crew members
        bandwidth_mbps: Available bandwidth in Mbps
        latency_s: One-way latency in seconds

    Returns:
        True if sovereign (internal > external)

    Receipt: mars_sovereignty
    """
    # Internal decision capacity
    internal_rate_bps = crew * DECISION_RATE_TARGET_BPS

    # External decision capacity (constrained by latency)
    # Round-trip delay limits effective bandwidth
    round_trip_s = latency_s * 2
    if round_trip_s > 0:
        effective_bandwidth_bps = (bandwidth_mbps * 1_000_000) / round_trip_s
    else:
        effective_bandwidth_bps = bandwidth_mbps * 1_000_000

    # Sovereignty check
    is_sovereign = internal_rate_bps > effective_bandwidth_bps

    result = {
        "crew": crew,
        "internal_rate_bps": internal_rate_bps,
        "external_rate_bps": round(effective_bandwidth_bps, 2),
        "bandwidth_mbps": bandwidth_mbps,
        "latency_s": latency_s,
        "is_sovereign": is_sovereign,
        "advantage_ratio": round(
            internal_rate_bps / effective_bandwidth_bps
            if effective_bandwidth_bps > 0 else float("inf"),
            3
        ),
        "tenant_id": MARS_TENANT_ID
    }

    emit_path_receipt("mars", "sovereignty", result)
    return is_sovereign


# === OPTIMIZATION (STUB) ===

def optimize_resources(dome_state: Dict[str, Any]) -> Dict[str, Any]:
    """RL optimization placeholder.

    STUB: Returns current state with optimization suggestions.
    FULL: Will use Thompson sampling + fractal layers.

    Args:
        dome_state: Current dome simulation state

    Returns:
        Dict with optimization suggestions

    Receipt: mars_optimize
    """
    result = {
        "stub_mode": True,
        "current_state": dome_state,
        "suggestions": [
            {"resource": "water", "action": "increase_recycling", "projected_gain": 0.05},
            {"resource": "food", "action": "expand_greenhouse", "projected_gain": 0.10},
            {"resource": "power", "action": "add_solar_panels", "projected_gain": 0.02}
        ],
        "optimization_ready": False,
        "next_stage": "RL integration with fractal_layers",
        "tenant_id": MARS_TENANT_ID
    }

    emit_path_receipt("mars", "optimize", result)
    return result


# === PATH INFO ===

def get_mars_info() -> Dict[str, Any]:
    """Get Mars path configuration and status.

    Returns:
        Dict with path info

    Receipt: mars_info
    """
    spec = load_path_spec("mars")

    info = {
        "path": "mars",
        "version": spec.get("version", "0.1.0"),
        "status": spec.get("status", "stub"),
        "description": spec.get("description", ""),
        "config": spec.get("config", {}),
        "dependencies": spec.get("dependencies", []),
        "receipts": spec.get("receipts", []),
        "evolution": spec.get("evolution", {}),
        "tenant_id": MARS_TENANT_ID
    }

    emit_path_receipt("mars", "info", info)
    return info
