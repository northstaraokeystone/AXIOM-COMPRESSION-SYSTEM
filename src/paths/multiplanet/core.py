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

import json
from typing import Dict, Any, List, Optional

from ..base import emit_path_receipt, load_path_spec, PathStopRule


# === CONSTANTS ===

MULTIPLANET_TENANT_ID = "axiom-multiplanet"
"""Tenant ID for multi-planet path receipts."""

EXPANSION_SEQUENCE = ["asteroid", "mars", "europa", "titan"]
"""Ordered expansion sequence."""

LATENCY_BOUNDS_MIN = {
    "asteroid": 3,
    "mars": 3,
    "europa": 33,
    "titan": 70
}
"""Minimum one-way latency in minutes."""

LATENCY_BOUNDS_MAX = {
    "asteroid": 20,
    "mars": 22,
    "europa": 53,
    "titan": 90
}
"""Maximum one-way latency in minutes."""

AUTONOMY_REQUIREMENT = {
    "asteroid": 0.70,
    "mars": 0.85,
    "europa": 0.95,
    "titan": 0.99
}
"""Required autonomy level per body."""

BANDWIDTH_BUDGET_MBPS = {
    "asteroid": 500,
    "mars": 100,
    "europa": 20,
    "titan": 5
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
            "compute_autonomy_requirement"
        ],
        "pending_capabilities": [
            "simulate_body",
            "integrated_simulation"
        ],
        "config": spec.get("config", {}),
        "tenant_id": MULTIPLANET_TENANT_ID
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
    emit_path_receipt("multiplanet", "sequence", {
        "sequence": EXPANSION_SEQUENCE,
        "count": len(EXPANSION_SEQUENCE),
        "tenant_id": MULTIPLANET_TENANT_ID
    })

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
        "prerequisites": EXPANSION_SEQUENCE[:EXPANSION_SEQUENCE.index(body)],
        "tenant_id": MULTIPLANET_TENANT_ID
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
        "tenant_id": MULTIPLANET_TENANT_ID
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
        "tenant_id": MULTIPLANET_TENANT_ID
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
        "titan": "Saturn system, extreme distance, nearly independent operations required"
    }
    return rationales.get(body, "Unknown body")


# === SIMULATION (STUB) ===

def simulate_body(
    body: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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
        ) if body != "asteroid" else True,
        "tenant_id": MULTIPLANET_TENANT_ID
    }

    emit_path_receipt("multiplanet", "simulate", result)
    return result


# === TELEMETRY COMPRESSION ===

def compute_telemetry_compression(
    body: str,
    data_rate_mbps: float
) -> Dict[str, Any]:
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
        "effective_compression": round(1 - (bandwidth / data_rate_mbps), 4) if data_rate_mbps > 0 else 0,
        "tenant_id": MULTIPLANET_TENANT_ID
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
        "tenant_id": MULTIPLANET_TENANT_ID
    }

    emit_path_receipt("multiplanet", "info", info)
    return info
