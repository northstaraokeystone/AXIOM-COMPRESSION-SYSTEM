"""Multi-planet federation protocol expansion.

Extends federation protocols for multi-planet coordination with
lag-tolerant consensus and cross-planet synchronization.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core import TENANT_ID, dual_hash, emit_receipt
from src.federation_multiplanet import (
    init_federation,
    sync_federation,
    run_consensus,
    get_federation_status,
    load_federation_config,
)


def get_multiplanet_status() -> Dict[str, Any]:
    """Get multi-planet federation status.

    Returns:
        dict: Status for all planets in federation.
    """
    status = get_federation_status()
    config = load_federation_config()

    return {
        "federation_active": status["initialized"],
        "planets": status["planets"],
        "planet_count": status["planet_count"],
        "consensus_rounds": status["consensus_round"],
        "health": status["health"],
        "protocol": {
            "consensus": "modified_raft_federated",
            "lag_tolerance": config.get("consensus_lag_tolerance", True),
            "sync_interval_hours": config.get("sync_interval_hours", 24),
        },
    }


def cross_planet_sync(planets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Perform cross-planet synchronization.

    Args:
        planets: Specific planets to sync (all if None).

    Returns:
        dict: Sync result.
    """
    # Initialize if needed
    status = get_federation_status()
    if not status["initialized"]:
        init_federation(planets)

    # Perform sync
    sync_result = sync_federation()

    emit_receipt(
        "federation_multiplanet_sync_receipt",
        {
            "receipt_type": "federation_multiplanet_sync_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "sync_complete": sync_result["sync_complete"],
            "planets_synced": sync_result["planets_synced"],
            "total_planets": sync_result["total_planets"],
            "payload_hash": dual_hash(json.dumps(sync_result, default=str)),
        },
    )

    return sync_result


def run_multiplanet_consensus(
    proposal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run consensus across all planets.

    Args:
        proposal: Proposal to vote on.

    Returns:
        dict: Consensus result.
    """
    # Initialize if needed
    status = get_federation_status()
    if not status["initialized"]:
        init_federation()

    # Run consensus
    result = run_consensus(proposal)

    emit_receipt(
        "federation_multiplanet_consensus_receipt",
        {
            "receipt_type": "federation_multiplanet_consensus_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "consensus_reached": result["consensus_reached"],
            "consensus_round": result["consensus_round"],
            "approval_rate": result["approval_rate"],
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )

    return result


def get_planet_latencies() -> Dict[str, Dict[str, float]]:
    """Get latency profiles for all federated planets.

    Returns:
        dict: Latency info for each planet.
    """
    config = load_federation_config()

    # Load planet profiles from spec
    import os

    spec_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ),
        "data",
        "federation_spec.json",
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    planet_profiles = spec.get("planet_profiles", {})

    latencies = {}
    for planet, profile in planet_profiles.items():
        latencies[planet] = {
            "latency_min": profile.get("latency_min", 3),
            "latency_max": profile.get("latency_max", 22),
            "latency_avg": (
                profile.get("latency_min", 3) + profile.get("latency_max", 22)
            )
            / 2,
            "round_trip_min": profile.get("latency_min", 3) * 2,
            "round_trip_max": profile.get("latency_max", 22) * 2,
        }

    return latencies


def calculate_federation_reach() -> Dict[str, Any]:
    """Calculate current federation reach and coverage.

    Returns:
        dict: Federation reach metrics.
    """
    status = get_federation_status()
    latencies = get_planet_latencies()

    if not latencies:
        return {"reach": "none", "planets": 0}

    max_latency = max(p["latency_max"] for p in latencies.values())
    avg_latency = sum(p["latency_avg"] for p in latencies.values()) / len(latencies)

    return {
        "planets": len(latencies),
        "max_latency_min": max_latency,
        "avg_latency_min": avg_latency,
        "max_round_trip_min": max_latency * 2,
        "reach": "solar_system"
        if max_latency > 30
        else "inner_solar"
        if max_latency > 10
        else "near_earth",
        "federation_health": status.get("health", {}).get("healthy", False),
    }
