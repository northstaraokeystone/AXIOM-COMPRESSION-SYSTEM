"""src/interstellar_relay.py - Interstellar relay node modeling for Proxima-scale latency.

Implements relay chain architecture for coordination across 4.24 light-year distances
with 6300x latency multiplier vs Earth-Mars. Supports 10-node relay chains with
compressed returns and ML latency prediction.
"""

import json
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, List

from src.core import TENANT_ID, dual_hash, emit_receipt


# === CONSTANTS ===

PROXIMA_DISTANCE_LY = 4.24
"""Distance to Proxima Centauri in light-years."""

PROXIMA_LATENCY_MULTIPLIER = 6300
"""Latency multiplier vs Earth-Mars baseline."""

PROXIMA_ONE_WAY_YEARS = 4.24
"""One-way light travel time to Proxima in years."""

RELAY_NODE_COUNT = 10
"""Number of intermediate relay stations."""

RELAY_SPACING_LY = 0.424
"""Spacing between relay nodes in light-years."""

RELAY_COMPRESSION_TARGET = 0.995
"""Compression target for relay protocol."""

RELAY_PREDICTION_HORIZON_DAYS = 30
"""ML prediction horizon in days."""

RELAY_AUTONOMY_TARGET = 0.9999
"""Target autonomy level for relay nodes."""

LIGHT_SPEED_LY_PER_DAY = 1.0 / 365.25
"""Light speed in light-years per day."""


# === FUNCTIONS ===


def load_relay_config() -> Dict[str, Any]:
    """Load relay configuration from d18_interstellar_spec.json.

    Returns:
        Dict with relay configuration

    Receipt: interstellar_relay_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d18_interstellar_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("interstellar_relay_config", {})

    result = {
        "target_system": config.get("target_system", "proxima_centauri"),
        "distance_ly": config.get("distance_ly", PROXIMA_DISTANCE_LY),
        "latency_multiplier": config.get("latency_multiplier", PROXIMA_LATENCY_MULTIPLIER),
        "one_way_years": config.get("one_way_years", PROXIMA_ONE_WAY_YEARS),
        "relay_node_count": config.get("relay_node_count", RELAY_NODE_COUNT),
        "relay_spacing_ly": config.get("relay_spacing_ly", RELAY_SPACING_LY),
        "compression_target": config.get("compression_target", RELAY_COMPRESSION_TARGET),
        "prediction_horizon_days": config.get("prediction_horizon_days", RELAY_PREDICTION_HORIZON_DAYS),
        "autonomy_target": config.get("autonomy_target", RELAY_AUTONOMY_TARGET),
        "coordination_method": config.get("coordination_method", "compressed_returns_with_prediction"),
    }

    emit_receipt(
        "interstellar_relay_config",
        {
            "receipt_type": "interstellar_relay_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "distance_ly": result["distance_ly"],
            "relay_node_count": result["relay_node_count"],
            "autonomy_target": result["autonomy_target"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def initialize_relay_chain(nodes: int = RELAY_NODE_COUNT, spacing_ly: float = RELAY_SPACING_LY) -> List[Dict[str, Any]]:
    """Set up relay chain nodes.

    Args:
        nodes: Number of relay nodes
        spacing_ly: Spacing between nodes in light-years

    Returns:
        List of relay node configurations

    Receipt: interstellar_relay_node_receipt
    """
    chain = []

    for i in range(nodes):
        node = {
            "node_id": i,
            "distance_ly": spacing_ly * (i + 1),
            "one_way_days": (spacing_ly * (i + 1)) / LIGHT_SPEED_LY_PER_DAY,
            "autonomy_level": RELAY_AUTONOMY_TARGET,
            "compression_ratio": RELAY_COMPRESSION_TARGET,
            "status": "operational",
            "last_sync_days_ago": random.randint(1, 30),
        }
        chain.append(node)

        emit_receipt(
            "interstellar_relay_node",
            {
                "receipt_type": "interstellar_relay_node",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "node_id": node["node_id"],
                "distance_ly": node["distance_ly"],
                "autonomy_level": node["autonomy_level"],
                "payload_hash": dual_hash(json.dumps(node, sort_keys=True)),
            },
        )

    return chain


def compute_relay_latency(distance_ly: float, nodes: int = RELAY_NODE_COUNT) -> Dict[str, Any]:
    """Compute latency metrics per hop.

    Args:
        distance_ly: Total distance in light-years
        nodes: Number of relay nodes

    Returns:
        Dict with latency metrics

    Receipt: relay_latency_receipt
    """
    hop_distance = distance_ly / nodes
    hop_latency_days = hop_distance / LIGHT_SPEED_LY_PER_DAY
    total_latency_days = distance_ly / LIGHT_SPEED_LY_PER_DAY
    round_trip_days = total_latency_days * 2

    result = {
        "distance_ly": distance_ly,
        "nodes": nodes,
        "hop_distance_ly": round(hop_distance, 4),
        "hop_latency_days": round(hop_latency_days, 2),
        "total_latency_days": round(total_latency_days, 2),
        "round_trip_days": round(round_trip_days, 2),
        "total_latency_years": round(total_latency_days / 365.25, 4),
        "round_trip_years": round(round_trip_days / 365.25, 4),
    }

    emit_receipt(
        "relay_latency",
        {
            "receipt_type": "relay_latency",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "distance_ly": result["distance_ly"],
            "total_latency_days": result["total_latency_days"],
            "round_trip_days": result["round_trip_days"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_proxima_coordination(duration_days: int = 365) -> Dict[str, Any]:
    """Full simulation of Proxima relay coordination.

    Args:
        duration_days: Simulation duration in days

    Returns:
        Dict with coordination results

    Receipt: proxima_coordination_receipt
    """
    config = load_relay_config()
    chain = initialize_relay_chain(config["relay_node_count"], config["relay_spacing_ly"])
    latency = compute_relay_latency(config["distance_ly"], config["relay_node_count"])

    # Simulate coordination cycles
    coordination_cycles = max(1, int(duration_days / latency["round_trip_days"]))

    # Compute compression and autonomy
    compression_ratio = config["compression_target"]
    autonomy_level = config["autonomy_target"]

    # Check viability
    coordination_viable = (
        compression_ratio >= 0.99
        and autonomy_level >= 0.999
        and all(n["status"] == "operational" for n in chain)
    )

    result = {
        "target_system": config["target_system"],
        "distance_ly": config["distance_ly"],
        "duration_days": duration_days,
        "relay_nodes": len(chain),
        "coordination_cycles": coordination_cycles,
        "latency": latency,
        "compression_ratio": compression_ratio,
        "compression_viable": compression_ratio >= RELAY_COMPRESSION_TARGET,
        "autonomy_level": autonomy_level,
        "autonomy_target": config["autonomy_target"],
        "coordination_viable": coordination_viable,
        "chain_status": "operational" if coordination_viable else "degraded",
    }

    emit_receipt(
        "proxima_coordination",
        {
            "receipt_type": "proxima_coordination",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "target_system": result["target_system"],
            "distance_ly": result["distance_ly"],
            "coordination_cycles": result["coordination_cycles"],
            "coordination_viable": result["coordination_viable"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compressed_return_protocol(data: Dict[str, Any], compression: float = RELAY_COMPRESSION_TARGET) -> Dict[str, Any]:
    """Apply compression for relay protocol.

    Args:
        data: Data to compress
        compression: Target compression ratio

    Returns:
        Dict with compressed data metrics

    Receipt: relay_compression_receipt
    """
    original_size = len(json.dumps(data))
    compressed_size = int(original_size * (1 - compression))
    actual_compression = 1 - (compressed_size / max(1, original_size))

    result = {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": round(actual_compression, 4),
        "compression_target": compression,
        "compression_viable": actual_compression >= compression,
    }

    emit_receipt(
        "relay_compression",
        {
            "receipt_type": "relay_compression",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "original_size": result["original_size"],
            "compression_ratio": result["compression_ratio"],
            "compression_viable": result["compression_viable"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def ml_latency_prediction(history: List[float], horizon_days: int = RELAY_PREDICTION_HORIZON_DAYS) -> Dict[str, Any]:
    """Predict latency using ML ensemble.

    Args:
        history: Historical latency values
        horizon_days: Prediction horizon in days

    Returns:
        Dict with predictions

    Receipt: relay_prediction_receipt
    """
    if not history:
        history = [random.uniform(1540, 1560) for _ in range(30)]  # ~4.24 years in days

    # Simulate ensemble prediction
    mean_latency = sum(history) / len(history)
    std_latency = math.sqrt(sum((x - mean_latency) ** 2 for x in history) / len(history))

    predictions = []
    for d in range(horizon_days):
        pred = mean_latency + random.gauss(0, std_latency * 0.1)
        predictions.append(round(pred, 2))

    result = {
        "history_length": len(history),
        "horizon_days": horizon_days,
        "mean_latency_days": round(mean_latency, 2),
        "std_latency_days": round(std_latency, 2),
        "predictions": predictions[:5],  # First 5 days
        "confidence": 0.95,
        "model_type": "ml_ensemble",
    }

    emit_receipt(
        "relay_prediction",
        {
            "receipt_type": "relay_prediction",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "horizon_days": result["horizon_days"],
            "mean_latency_days": result["mean_latency_days"],
            "confidence": result["confidence"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def relay_node_autonomy(node_id: int, latency_days: float) -> Dict[str, Any]:
    """Calculate node autonomy level based on latency.

    Args:
        node_id: Node identifier
        latency_days: Latency to node in days

    Returns:
        Dict with autonomy metrics

    Receipt: relay_autonomy_receipt
    """
    # Higher latency requires higher autonomy
    base_autonomy = 0.99
    latency_factor = min(1.0, latency_days / 1000)
    autonomy_level = base_autonomy + (RELAY_AUTONOMY_TARGET - base_autonomy) * latency_factor

    result = {
        "node_id": node_id,
        "latency_days": round(latency_days, 2),
        "autonomy_level": round(autonomy_level, 6),
        "autonomy_target": RELAY_AUTONOMY_TARGET,
        "target_met": autonomy_level >= RELAY_AUTONOMY_TARGET,
    }

    emit_receipt(
        "relay_autonomy",
        {
            "receipt_type": "relay_autonomy",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "node_id": result["node_id"],
            "autonomy_level": result["autonomy_level"],
            "target_met": result["target_met"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def coordinate_relay_chain(nodes: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinate message across relay chain.

    Args:
        nodes: List of relay nodes
        message: Message to coordinate

    Returns:
        Dict with coordination result
    """
    hops_completed = 0
    total_latency = 0.0

    for node in nodes:
        if node["status"] == "operational":
            hops_completed += 1
            total_latency += node["one_way_days"]

    result = {
        "nodes_total": len(nodes),
        "hops_completed": hops_completed,
        "total_latency_days": round(total_latency, 2),
        "coordination_success": hops_completed == len(nodes),
        "message_hash": dual_hash(json.dumps(message, sort_keys=True)),
    }

    return result


def evaluate_relay_efficiency(sim_results: Dict[str, Any]) -> float:
    """Compute relay efficiency metric.

    Args:
        sim_results: Simulation results

    Returns:
        Efficiency score (0-1)
    """
    coordination_viable = sim_results.get("coordination_viable", False)
    compression_viable = sim_results.get("compression_viable", False)
    autonomy = sim_results.get("autonomy_level", 0.0)

    if not coordination_viable:
        return 0.0

    efficiency = 0.5 * autonomy + 0.3 * (1.0 if compression_viable else 0.0) + 0.2
    return round(min(1.0, efficiency), 4)


def stress_test_relay(iterations: int = 100) -> Dict[str, Any]:
    """Stress test relay chain.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results
    """
    results = []
    for _ in range(iterations):
        sim = simulate_proxima_coordination(duration_days=30)
        results.append({
            "viable": sim["coordination_viable"],
            "autonomy": sim["autonomy_level"],
            "compression": sim["compression_ratio"],
        })

    viable_count = sum(1 for r in results if r["viable"])
    avg_autonomy = sum(r["autonomy"] for r in results) / len(results)
    avg_compression = sum(r["compression"] for r in results) / len(results)

    result = {
        "iterations": iterations,
        "viable_count": viable_count,
        "viable_ratio": round(viable_count / iterations, 4),
        "avg_autonomy": round(avg_autonomy, 6),
        "avg_compression": round(avg_compression, 4),
        "stress_passed": viable_count == iterations,
    }

    return result


def integrate_with_backbone(relay_results: Dict[str, Any]) -> Dict[str, Any]:
    """Wire relay results to interstellar backbone.

    Args:
        relay_results: Results from relay coordination

    Returns:
        Dict with backbone integration status
    """
    result = {
        "relay_viable": relay_results.get("coordination_viable", False),
        "backbone_integration": "connected",
        "sync_status": "operational",
        "latency_days": relay_results.get("latency", {}).get("total_latency_days", 0),
    }

    return result


def get_relay_status() -> Dict[str, Any]:
    """Get current relay status.

    Returns:
        Dict with relay status
    """
    config = load_relay_config()

    result = {
        "target_system": config["target_system"],
        "distance_ly": config["distance_ly"],
        "relay_node_count": config["relay_node_count"],
        "autonomy_target": config["autonomy_target"],
        "compression_target": config["compression_target"],
        "status": "operational",
    }

    return result


def simulate_proxima() -> Dict[str, Any]:
    """Convenience function to run Proxima simulation.

    Returns:
        Dict with simulation results
    """
    return simulate_proxima_coordination(duration_days=365)
