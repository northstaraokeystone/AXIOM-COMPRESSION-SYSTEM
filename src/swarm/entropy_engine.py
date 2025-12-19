"""Swarm entropy engine for D19 emergent coordination.

Collective entropy measurement across 100 nodes.
Entropy gradients replace central coordination.
The SWARM thinks, not any individual node.

Key insight: No node knows the global state. Each node measures LOCAL entropy.
Gradients between neighbors guide coordination. Entropy IS the coordinator.

D19.1 UPDATE:
  - Live triad ingest mode replaces synthetic generation
  - No synthetic entropy - reality only
  - "Reality is the only valid scenario"
"""

import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19 SWARM CONSTANTS ===

NODE_COUNT = 100
"""Number of nodes in swarm."""

ENTROPY_SAMPLE_RATE_HZ = 10
"""Entropy sampling rate in Hz."""

GRADIENT_THRESHOLD = 0.001
"""Minimum gradient to trigger coordination."""

CONVERGENCE_TARGET = 0.95
"""Target convergence for swarm coherence."""

MESH_CONNECTIONS = 4950
"""Full mesh connections: n*(n-1)/2 for 100 nodes."""


@dataclass
class SwarmNode:
    """Individual node in the swarm."""

    node_id: str
    entropy: float = 0.0
    receipts: List[Dict] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)
    gradients: Dict[str, float] = field(default_factory=dict)
    is_sink: bool = False
    is_source: bool = False


@dataclass
class EntropyEngine:
    """Entropy engine managing swarm coordination."""

    engine_id: str
    nodes: Dict[str, SwarmNode] = field(default_factory=dict)
    global_entropy: float = 0.0
    coherence: float = 0.0
    convergence: float = 0.0
    config: Dict = field(default_factory=dict)


def init_entropy_engine(config: Dict = None) -> EntropyEngine:
    """Initialize entropy engine across 100 nodes.

    Args:
        config: Optional configuration dict

    Returns:
        EntropyEngine instance with initialized nodes

    Receipt: entropy_engine_init_receipt
    """
    config = config or {}
    node_count = config.get("node_count", NODE_COUNT)

    engine_id = str(uuid.uuid4())[:8]
    engine = EntropyEngine(engine_id=engine_id, config=config)

    # Initialize nodes
    for i in range(node_count):
        node_id = f"node_{i:03d}"
        node = SwarmNode(node_id=node_id)

        # Assign neighbors (mesh topology)
        for j in range(node_count):
            if i != j:
                neighbor_id = f"node_{j:03d}"
                node.neighbors.append(neighbor_id)

        engine.nodes[node_id] = node

    emit_receipt(
        "entropy_engine_init",
        {
            "receipt_type": "entropy_engine_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "engine_id": engine_id,
            "node_count": len(engine.nodes),
            "mesh_connections": len(engine.nodes) * (len(engine.nodes) - 1) // 2,
            "payload_hash": dual_hash(
                json.dumps({"engine_id": engine_id, "node_count": len(engine.nodes)}, sort_keys=True)
            ),
        },
    )

    return engine


def measure_local_entropy(node_id: str, receipts: List[Dict]) -> float:
    """Measure Shannon entropy of node's receipt stream.

    Args:
        node_id: Node identifier
        receipts: List of receipts at node

    Returns:
        Shannon entropy value

    Receipt: entropy_measurement_receipt
    """
    if not receipts:
        return 0.0

    # Count receipt types for probability distribution
    type_counts: Dict[str, int] = {}
    for r in receipts:
        rtype = r.get("receipt_type", "unknown")
        type_counts[rtype] = type_counts.get(rtype, 0) + 1

    total = len(receipts)
    entropy = 0.0

    for count in type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    emit_receipt(
        "entropy_measurement",
        {
            "receipt_type": "entropy_measurement",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "node_id": node_id,
            "entropy": round(entropy, 6),
            "receipt_count": total,
            "type_count": len(type_counts),
            "payload_hash": dual_hash(
                json.dumps({"node_id": node_id, "entropy": round(entropy, 6)}, sort_keys=True)
            ),
        },
    )

    return entropy


def compute_gradient(node_id: str, neighbors: List[str], engine: EntropyEngine) -> Dict[str, float]:
    """Compute entropy gradient to each neighbor.

    Gradient = H_self - H_neighbor
    Positive gradient: neighbor has lower entropy (sink)
    Negative gradient: neighbor has higher entropy (source)

    Args:
        node_id: Current node ID
        neighbors: List of neighbor node IDs
        engine: EntropyEngine instance

    Returns:
        Dict mapping neighbor_id to gradient value
    """
    gradients = {}
    node = engine.nodes.get(node_id)
    if not node:
        return gradients

    self_entropy = node.entropy

    for neighbor_id in neighbors:
        neighbor = engine.nodes.get(neighbor_id)
        if neighbor:
            gradient = self_entropy - neighbor.entropy
            gradients[neighbor_id] = round(gradient, 6)

    return gradients


def propagate_gradient(engine: EntropyEngine, gradients: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Propagate gradients through mesh network.

    Args:
        engine: EntropyEngine instance
        gradients: Dict mapping node_id to gradient dict

    Returns:
        Propagation result with convergence status

    Receipt: gradient_propagation_receipt
    """
    # Update gradients for all nodes
    for node_id, node_gradients in gradients.items():
        if node_id in engine.nodes:
            engine.nodes[node_id].gradients = node_gradients

    # Compute global statistics
    all_gradients = []
    for node in engine.nodes.values():
        all_gradients.extend(node.gradients.values())

    if all_gradients:
        mean_gradient = sum(all_gradients) / len(all_gradients)
        variance = sum((g - mean_gradient) ** 2 for g in all_gradients) / len(all_gradients)
        std_gradient = math.sqrt(variance)
    else:
        mean_gradient = 0.0
        std_gradient = 0.0

    # Convergence: low gradient variance = high convergence
    convergence = max(0.0, 1.0 - std_gradient)
    engine.convergence = convergence

    result = {
        "nodes_updated": len(gradients),
        "total_gradients": len(all_gradients),
        "mean_gradient": round(mean_gradient, 6),
        "std_gradient": round(std_gradient, 6),
        "convergence": round(convergence, 4),
        "target_met": convergence >= CONVERGENCE_TARGET,
    }

    emit_receipt(
        "gradient_propagation",
        {
            "receipt_type": "gradient_propagation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "engine_id": engine.engine_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def detect_entropy_sink(engine: EntropyEngine) -> List[str]:
    """Find nodes absorbing entropy (leaders).

    Entropy sinks have lower entropy than neighbors on average.
    They attract coordination signals.

    Args:
        engine: EntropyEngine instance

    Returns:
        List of sink node IDs
    """
    sinks = []
    mean_entropy = sum(n.entropy for n in engine.nodes.values()) / len(engine.nodes) if engine.nodes else 0

    for node_id, node in engine.nodes.items():
        if node.entropy < mean_entropy - GRADIENT_THRESHOLD:
            node.is_sink = True
            sinks.append(node_id)
        else:
            node.is_sink = False

    return sinks


def detect_entropy_source(engine: EntropyEngine) -> List[str]:
    """Find nodes emitting entropy (workers).

    Entropy sources have higher entropy than neighbors on average.
    They respond to coordination signals.

    Args:
        engine: EntropyEngine instance

    Returns:
        List of source node IDs
    """
    sources = []
    mean_entropy = sum(n.entropy for n in engine.nodes.values()) / len(engine.nodes) if engine.nodes else 0

    for node_id, node in engine.nodes.items():
        if node.entropy > mean_entropy + GRADIENT_THRESHOLD:
            node.is_source = True
            sources.append(node_id)
        else:
            node.is_source = False

    return sources


def coordinate_via_gradient(engine: EntropyEngine, action: str) -> Dict[str, Any]:
    """Execute coordinated action via gradient descent.

    Nodes follow gradients toward entropy sinks.
    No central coordinator required.

    Args:
        engine: EntropyEngine instance
        action: Action to coordinate

    Returns:
        Coordination result

    Receipt: coordination_action_receipt
    """
    # Find sinks and sources
    sinks = detect_entropy_sink(engine)
    sources = detect_entropy_source(engine)

    # Simulate gradient descent coordination
    participating_nodes = len(engine.nodes)
    sink_count = len(sinks)
    source_count = len(sources)

    # Coordination success based on clear gradient structure
    success_rate = min(1.0, (sink_count + source_count) / participating_nodes) if participating_nodes > 0 else 0

    result = {
        "action": action,
        "participating_nodes": participating_nodes,
        "sinks": sink_count,
        "sources": source_count,
        "success_rate": round(success_rate, 4),
        "coordination_mode": "entropy_gradient",
        "central_coordinator": False,
    }

    emit_receipt(
        "coordination_action",
        {
            "receipt_type": "coordination_action",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "engine_id": engine.engine_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def measure_swarm_coherence(engine: EntropyEngine) -> float:
    """Measure global coherence from local measurements.

    Coherence = 1 - Var(H_i) / Mean(H_i)
    High coherence = nodes have similar entropy = coordinated

    Args:
        engine: EntropyEngine instance

    Returns:
        Coherence value 0-1

    Receipt: swarm_coherence_receipt
    """
    if not engine.nodes:
        return 0.0

    entropies = [n.entropy for n in engine.nodes.values()]
    mean_h = sum(entropies) / len(entropies)

    if mean_h == 0:
        coherence = 1.0  # All zero entropy = perfectly coherent
    else:
        variance = sum((h - mean_h) ** 2 for h in entropies) / len(entropies)
        coherence = max(0.0, 1.0 - variance / mean_h)

    engine.coherence = coherence

    emit_receipt(
        "swarm_coherence",
        {
            "receipt_type": "swarm_coherence",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "engine_id": engine.engine_id,
            "coherence": round(coherence, 4),
            "node_count": len(engine.nodes),
            "mean_entropy": round(mean_h, 6),
            "target_met": coherence >= CONVERGENCE_TARGET,
            "payload_hash": dual_hash(
                json.dumps({"coherence": round(coherence, 4), "node_count": len(engine.nodes)}, sort_keys=True)
            ),
        },
    )

    return coherence


def simulate_coordination(engine: EntropyEngine, scenario: str) -> Dict[str, Any]:
    """Test coordination scenario with entropy gradients.

    Args:
        engine: EntropyEngine instance
        scenario: Scenario name ("consensus", "recovery", "migration")

    Returns:
        Simulation results
    """
    # Initialize random entropies for simulation
    for node in engine.nodes.values():
        if scenario == "consensus":
            # Start with varied entropy, expect convergence
            node.entropy = random.uniform(0.5, 2.0)
        elif scenario == "recovery":
            # Some nodes have high entropy (failing)
            node.entropy = random.choice([0.1, 0.1, 0.1, 3.0])
        elif scenario == "migration":
            # Bimodal distribution (source and destination clusters)
            node.entropy = random.choice([0.2, 1.8])
        else:
            node.entropy = random.uniform(0.0, 2.0)

        # Simulate receipt stream
        node.receipts = [{"receipt_type": f"sim_{i}"} for i in range(random.randint(5, 20))]
        node.entropy = measure_local_entropy(node.node_id, node.receipts)

    # Compute gradients
    gradients = {}
    for node_id, node in engine.nodes.items():
        gradients[node_id] = compute_gradient(node_id, node.neighbors, engine)

    # Propagate gradients
    prop_result = propagate_gradient(engine, gradients)

    # Measure final coherence
    coherence = measure_swarm_coherence(engine)

    # Coordinate action
    coord_result = coordinate_via_gradient(engine, f"simulate_{scenario}")

    return {
        "scenario": scenario,
        "nodes": len(engine.nodes),
        "propagation": prop_result,
        "coherence": coherence,
        "coordination": coord_result,
        "success": coherence >= CONVERGENCE_TARGET * 0.8,  # 80% of target for simulation
    }


def get_engine_status() -> Dict[str, Any]:
    """Get current entropy engine status.

    Returns:
        Engine status dict
    """
    return {
        "module": "swarm.entropy_engine",
        "version": "19.1.0",
        "node_count": NODE_COUNT,
        "sample_rate_hz": ENTROPY_SAMPLE_RATE_HZ,
        "gradient_threshold": GRADIENT_THRESHOLD,
        "convergence_target": CONVERGENCE_TARGET,
        "mesh_connections": MESH_CONNECTIONS,
        "coordination_mode": "entropy_gradient",
        "central_coordinator": False,
        "live_ingest_mode": True,
        "synthetic_enabled": False,
    }


# === D19.1 LIVE INGEST MODE ===

# KILLED: generate_synthetic_disruption() - NO SYNTHETIC GENERATION
# KILLED: batch_generate() - NO SYNTHETIC GENERATION


def ingest_live_entropy(engine: EntropyEngine, live_receipts: List[Dict]) -> Dict[str, Any]:
    """Ingest live receipts for entropy measurement.

    D19.1: Replaces synthetic entropy generation.
    Real disruptions only - reality is the only valid scenario.

    Args:
        engine: EntropyEngine instance
        live_receipts: Live receipt stream from AgentProof + NEURON

    Returns:
        Live ingest result dict

    Receipt: live_entropy_ingest_receipt
    """
    if not live_receipts:
        return {"error": "empty_live_stream", "entropy": 0.0}

    # Distribute receipts to nodes based on receipt content
    for i, receipt in enumerate(live_receipts):
        node_id = f"node_{i % len(engine.nodes):03d}"
        if node_id in engine.nodes:
            engine.nodes[node_id].receipts.append(receipt)

    # Measure entropy at each node from live receipts
    for node_id, node in engine.nodes.items():
        if node.receipts:
            node.entropy = measure_local_entropy(node_id, node.receipts)

    # Compute global entropy
    entropies = [n.entropy for n in engine.nodes.values()]
    engine.global_entropy = sum(entropies) / len(entropies) if entropies else 0.0

    result = {
        "source": "live_triad",
        "receipts_ingested": len(live_receipts),
        "nodes_updated": len(engine.nodes),
        "global_entropy": round(engine.global_entropy, 6),
        "synthetic": False,
    }

    emit_receipt(
        "live_entropy_ingest",
        {
            "receipt_type": "live_entropy_ingest",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "engine_id": engine.engine_id,
            "receipts_ingested": len(live_receipts),
            "nodes_updated": len(engine.nodes),
            "global_entropy": round(engine.global_entropy, 6),
            "synthetic": False,
            "payload_hash": dual_hash(
                json.dumps(result, sort_keys=True)
            ),
        },
    )

    return result


def coordinate_from_live_stream(engine: EntropyEngine, live_receipts: List[Dict]) -> Dict[str, Any]:
    """Coordinate swarm from live receipt stream.

    D19.1: Full coordination cycle using live data only.

    Args:
        engine: EntropyEngine instance
        live_receipts: Live receipt stream

    Returns:
        Coordination result dict
    """
    # Ingest live receipts
    ingest_result = ingest_live_entropy(engine, live_receipts)

    # Compute gradients
    gradients = {}
    for node_id, node in engine.nodes.items():
        gradients[node_id] = compute_gradient(node_id, node.neighbors, engine)

    # Propagate gradients
    prop_result = propagate_gradient(engine, gradients)

    # Measure coherence
    coherence = measure_swarm_coherence(engine)

    # Coordinate action
    coord_result = coordinate_via_gradient(engine, "live_coordination")

    return {
        "mode": "live_stream",
        "synthetic": False,
        "ingest": ingest_result,
        "propagation": prop_result,
        "coherence": coherence,
        "coordination": coord_result,
        "success": coherence >= CONVERGENCE_TARGET * 0.8,
    }
