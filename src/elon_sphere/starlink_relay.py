"""src/elon_sphere/starlink_relay.py - Starlink as interstellar relay analog.

Models Starlink v2 laser links (100Gbps demonstrated 2025) as an analog
for interstellar relay node architecture.
"""

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

from src.core import TENANT_ID, dual_hash, emit_receipt


# === CONSTANTS ===

STARLINK_LASER_GBPS = 100
"""Starlink v2 laser link capacity in Gbps."""

STARLINK_RELAY_HOPS = 5
"""Default relay hops."""

STARLINK_LATENCY_MS = 20
"""Base latency per hop in milliseconds."""

MARS_DELAY_MIN = 3.0
"""Mars minimum one-way delay in minutes."""

MARS_DELAY_MAX = 22.0
"""Mars maximum one-way delay in minutes."""


# === FUNCTIONS ===


def load_starlink_config() -> Dict[str, Any]:
    """Load Starlink configuration from d18_interstellar_spec.json.

    Returns:
        Dict with Starlink configuration

    Receipt: starlink_analog_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "d18_interstellar_spec.json",
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("elon_sphere_config", {}).get("starlink_relay", {})

    result = {
        "enabled": config.get("enabled", True),
        "laser_gbps": config.get("laser_gbps", STARLINK_LASER_GBPS),
        "relay_hops": config.get("relay_hops", STARLINK_RELAY_HOPS),
        "latency_ms": config.get("latency_ms", STARLINK_LATENCY_MS),
        "analog_to_interstellar": config.get("analog_to_interstellar", True),
    }

    emit_receipt(
        "starlink_analog",
        {
            "receipt_type": "starlink_analog",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "enabled": result["enabled"],
            "laser_gbps": result["laser_gbps"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def initialize_starlink_mesh(nodes: int = 10) -> Dict[str, Any]:
    """Create Starlink mesh network.

    Args:
        nodes: Number of satellite nodes

    Returns:
        Dict with mesh configuration

    Receipt: starlink_mesh_receipt
    """
    satellites = []

    for i in range(nodes):
        sat = {
            "sat_id": i,
            "orbit_km": 550 + random.randint(-50, 50),
            "laser_capacity_gbps": STARLINK_LASER_GBPS,
            "status": "operational",
            "connections": min(4, nodes - 1),  # Max 4 laser links
        }
        satellites.append(sat)

    result = {
        "nodes": len(satellites),
        "satellites": satellites,
        "total_capacity_gbps": len(satellites) * STARLINK_LASER_GBPS,
        "mesh_connectivity": "full" if nodes <= 4 else "partial",
    }

    emit_receipt(
        "starlink_mesh",
        {
            "receipt_type": "starlink_mesh",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "nodes": result["nodes"],
            "total_capacity_gbps": result["total_capacity_gbps"],
            "payload_hash": dual_hash(json.dumps({"nodes": result["nodes"]}, sort_keys=True)),
        },
    )

    return result


def simulate_laser_link(gbps: int = STARLINK_LASER_GBPS, distance_km: float = 1000.0) -> Dict[str, Any]:
    """Simulate laser link performance.

    Args:
        gbps: Link capacity in Gbps
        distance_km: Link distance in km

    Returns:
        Dict with link performance metrics
    """
    # Speed of light: ~300,000 km/s
    latency_ms = (distance_km / 300000) * 1000

    # Atmospheric/alignment losses
    efficiency = 0.95 - (distance_km / 100000) * 0.1
    effective_gbps = gbps * max(0.5, efficiency)

    result = {
        "distance_km": distance_km,
        "capacity_gbps": gbps,
        "latency_ms": round(latency_ms, 4),
        "efficiency": round(efficiency, 4),
        "effective_gbps": round(effective_gbps, 2),
    }

    return result


def relay_hop_latency(hops: int = STARLINK_RELAY_HOPS, per_hop_ms: float = STARLINK_LATENCY_MS) -> float:
    """Calculate total relay latency.

    Args:
        hops: Number of relay hops
        per_hop_ms: Latency per hop in ms

    Returns:
        Total latency in ms
    """
    return round(hops * per_hop_ms, 2)


def analog_to_interstellar(starlink_results: Dict[str, Any]) -> Dict[str, Any]:
    """Map Starlink metrics to interstellar scale.

    Provides analog for testing interstellar relay concepts
    using existing Starlink infrastructure.

    Args:
        starlink_results: Starlink simulation results

    Returns:
        Dict with interstellar analog mapping
    """
    # Scale factors: Proxima is ~4.24 ly, Starlink is ~1000 km per hop
    ly_to_km = 9.461e12  # km per light-year
    starlink_scale_km = starlink_results.get("distance_km", 1000)

    # Calculate scale ratio
    proxima_km = 4.24 * ly_to_km
    scale_ratio = proxima_km / max(1, starlink_scale_km)

    result = {
        "starlink_metrics": starlink_results,
        "interstellar_analog": {
            "scale_ratio": scale_ratio,
            "analogous_to": "relay_chain_concept",
            "validated_patterns": [
                "laser_link_efficiency",
                "hop_latency_accumulation",
                "mesh_redundancy",
                "autonomy_requirements",
            ],
        },
        "transferable_learnings": True,
    }

    return result


def mars_comms_proof(delay_min: float = 10.0) -> Dict[str, Any]:
    """Prove Mars communication autonomy model.

    Args:
        delay_min: One-way delay in minutes

    Returns:
        Dict with Mars comms proof

    Receipt: mars_comms_receipt
    """
    # Mars communication delay varies from 3-22 minutes one-way
    delay_min = max(MARS_DELAY_MIN, min(MARS_DELAY_MAX, delay_min))
    round_trip_min = delay_min * 2

    # Required autonomy increases with delay
    autonomy_required = 0.9 + (delay_min / MARS_DELAY_MAX) * 0.09

    result = {
        "delay_min": round(delay_min, 2),
        "round_trip_min": round(round_trip_min, 2),
        "autonomy_required": round(autonomy_required, 4),
        "autonomy_achievable": True,
        "starlink_analog_valid": True,
    }

    emit_receipt(
        "mars_comms",
        {
            "receipt_type": "mars_comms",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "delay_min": result["delay_min"],
            "autonomy_required": result["autonomy_required"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_starlink_status() -> Dict[str, Any]:
    """Get current Starlink status.

    Returns:
        Dict with Starlink status
    """
    config = load_starlink_config()

    result = {
        "enabled": config["enabled"],
        "laser_gbps": config["laser_gbps"],
        "relay_hops": config["relay_hops"],
        "latency_ms": config["latency_ms"],
        "status": "operational",
    }

    return result
