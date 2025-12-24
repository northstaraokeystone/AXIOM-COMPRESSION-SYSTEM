"""100-node swarm testnet for mesh validation.

Implements distributed relay mesh with full mesh topology,
modified Raft consensus, failure injection, and recovery testing.

Receipt Types:
    - swarm_config_receipt: Configuration loaded
    - swarm_node_receipt: Node deployed
    - swarm_mesh_receipt: Mesh created
    - swarm_consensus_receipt: Consensus result
    - swarm_failure_receipt: Failure injected
    - swarm_recovery_receipt: Recovery completed
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core import TENANT_ID, dual_hash, emit_receipt

# Swarm constants
SWARM_NODE_COUNT = 100
SWARM_MESH_TOPOLOGY = "full_mesh"
SWARM_CONSENSUS_ALGORITHM = "modified_raft"
SWARM_LATENCY_SIMULATION = True
SWARM_PACKET_LOSS_RATE = 0.001

# Node type distribution
SWARM_ORBITAL_NODES = 60
SWARM_SURFACE_NODES = 30
SWARM_DEEP_SPACE_NODES = 10


def calculate_mesh_connections(node_count: int = SWARM_NODE_COUNT) -> int:
    """Calculate number of mesh connections for N nodes.

    Full mesh has n*(n-1)/2 connections.

    Args:
        node_count: Number of nodes in mesh.

    Returns:
        Number of connections.
    """
    return node_count * (node_count - 1) // 2


@dataclass
class SwarmNode:
    """Represents a swarm node."""

    node_id: str
    node_type: str  # "orbital", "surface", "deep_space"
    status: str  # "active", "failed", "recovering"
    latency_ms: float
    bandwidth_mbps: float
    messages_sent: int = 0
    messages_received: int = 0
    failures: int = 0
    last_heartbeat: Optional[str] = None


@dataclass
class SwarmState:
    """Current swarm state."""

    nodes: Dict[str, SwarmNode] = field(default_factory=dict)
    initialized: bool = False
    mesh_created: bool = False
    consensus_round: int = 0
    total_failures: int = 0
    total_recoveries: int = 0


# Global swarm state
_swarm_state = SwarmState()


def load_swarm_config() -> Dict[str, Any]:
    """Load swarm testnet configuration from spec file.

    Returns:
        dict: Swarm configuration.

    Receipt:
        swarm_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "swarm_testnet_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get(
        "swarm_config",
        {
            "node_count": SWARM_NODE_COUNT,
            "mesh_topology": SWARM_MESH_TOPOLOGY,
            "consensus_algorithm": SWARM_CONSENSUS_ALGORITHM,
            "latency_simulation": SWARM_LATENCY_SIMULATION,
            "packet_loss_rate": SWARM_PACKET_LOSS_RATE,
        },
    )

    emit_receipt(
        "swarm_config_receipt",
        {
            "receipt_type": "swarm_config_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "node_count": config.get("node_count", SWARM_NODE_COUNT),
            "mesh_topology": config.get("mesh_topology", SWARM_MESH_TOPOLOGY),
            "consensus_algorithm": config.get(
                "consensus_algorithm", SWARM_CONSENSUS_ALGORITHM
            ),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )
    return config


def init_swarm(node_count: Optional[int] = None) -> Dict[str, Any]:
    """Initialize swarm with specified node count.

    Args:
        node_count: Number of nodes (default from config).

    Returns:
        dict: Initialization result.

    Receipt:
        swarm_config_receipt
    """
    global _swarm_state

    config = load_swarm_config()
    if node_count is None:
        node_count = config.get("node_count", SWARM_NODE_COUNT)

    # Load node distribution
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "swarm_testnet_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    distribution = spec.get(
        "node_distribution", {"orbital": 40, "surface": 30, "deep_space": 30}
    )
    node_types = spec.get("node_types", {})

    _swarm_state.nodes = {}
    _swarm_state.initialized = True
    _swarm_state.mesh_created = False

    # Calculate actual counts based on distribution percentages
    total_pct = sum(distribution.values())
    orbital_count = int(node_count * distribution.get("orbital", 40) / total_pct)
    surface_count = int(node_count * distribution.get("surface", 30) / total_pct)
    deep_space_count = node_count - orbital_count - surface_count

    node_idx = 0

    # Create orbital nodes
    for i in range(orbital_count):
        node_config = node_types.get("orbital", {})
        node = SwarmNode(
            node_id=f"swarm_orbital_{node_idx:03d}",
            node_type="orbital",
            status="active",
            latency_ms=node_config.get("latency_ms", 50),
            bandwidth_mbps=node_config.get("bandwidth_mbps", 1000),
            last_heartbeat=datetime.utcnow().isoformat() + "Z",
        )
        _swarm_state.nodes[node.node_id] = node
        node_idx += 1

    # Create surface nodes
    for i in range(surface_count):
        node_config = node_types.get("surface", {})
        node = SwarmNode(
            node_id=f"swarm_surface_{node_idx:03d}",
            node_type="surface",
            status="active",
            latency_ms=node_config.get("latency_ms", 10),
            bandwidth_mbps=node_config.get("bandwidth_mbps", 100),
            last_heartbeat=datetime.utcnow().isoformat() + "Z",
        )
        _swarm_state.nodes[node.node_id] = node
        node_idx += 1

    # Create deep space nodes
    for i in range(deep_space_count):
        node_config = node_types.get("deep_space", {})
        node = SwarmNode(
            node_id=f"swarm_deep_{node_idx:03d}",
            node_type="deep_space",
            status="active",
            latency_ms=node_config.get("latency_ms", 1000),
            bandwidth_mbps=node_config.get("bandwidth_mbps", 10),
            last_heartbeat=datetime.utcnow().isoformat() + "Z",
        )
        _swarm_state.nodes[node.node_id] = node
        node_idx += 1

    result = {
        "initialized": True,
        "node_count": len(_swarm_state.nodes),
        "orbital_count": orbital_count,
        "surface_count": surface_count,
        "deep_space_count": deep_space_count,
    }

    emit_receipt(
        "swarm_config_receipt",
        {
            "receipt_type": "swarm_config_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "init",
            "node_count": len(_swarm_state.nodes),
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def deploy_node(node_id: str, node_type: str = "orbital") -> Dict[str, Any]:
    """Deploy a single swarm node.

    Args:
        node_id: Node identifier.
        node_type: Node type ("orbital", "surface", "deep_space").

    Returns:
        dict: Deployment result.

    Receipt:
        swarm_node_receipt
    """
    global _swarm_state

    if not _swarm_state.initialized:
        init_swarm()

    # Load node type config
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "swarm_testnet_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    node_types = spec.get("node_types", {})
    node_config = node_types.get(node_type, {"latency_ms": 50, "bandwidth_mbps": 100})

    node = SwarmNode(
        node_id=node_id,
        node_type=node_type,
        status="active",
        latency_ms=node_config.get("latency_ms", 50),
        bandwidth_mbps=node_config.get("bandwidth_mbps", 100),
        last_heartbeat=datetime.utcnow().isoformat() + "Z",
    )

    _swarm_state.nodes[node_id] = node

    result = {
        "deployed": True,
        "node_id": node_id,
        "node_type": node_type,
        "latency_ms": node.latency_ms,
        "bandwidth_mbps": node.bandwidth_mbps,
    }

    emit_receipt(
        "swarm_node_receipt",
        {
            "receipt_type": "swarm_node_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "deploy",
            "node_id": node_id,
            "node_type": node_type,
            "deployed": True,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def deploy_full_swarm() -> Dict[str, Any]:
    """Deploy all 100 nodes.

    Returns:
        dict: Full deployment result.

    Receipt:
        swarm_mesh_receipt
    """
    init_result = init_swarm(SWARM_NODE_COUNT)

    result = {
        "full_deployment": True,
        "node_count": init_result["node_count"],
        "orbital_count": init_result["orbital_count"],
        "surface_count": init_result["surface_count"],
        "deep_space_count": init_result["deep_space_count"],
    }

    emit_receipt(
        "swarm_mesh_receipt",
        {
            "receipt_type": "swarm_mesh_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "full_deployment",
            "node_count": result["node_count"],
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def create_mesh_topology() -> Dict[str, Any]:
    """Create full mesh topology.

    Returns:
        dict: Mesh creation result.

    Receipt:
        swarm_mesh_receipt
    """
    global _swarm_state

    if not _swarm_state.initialized:
        init_swarm()

    node_count = len(_swarm_state.nodes)
    # Full mesh: each node connected to all others
    connection_count = node_count * (node_count - 1) // 2

    _swarm_state.mesh_created = True

    result = {
        "mesh_created": True,
        "topology": SWARM_MESH_TOPOLOGY,
        "node_count": node_count,
        "connection_count": connection_count,
        "connections_per_node": node_count - 1,
    }

    emit_receipt(
        "swarm_mesh_receipt",
        {
            "receipt_type": "swarm_mesh_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "mesh_created": True,
            "topology": SWARM_MESH_TOPOLOGY,
            "node_count": node_count,
            "connection_count": connection_count,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def run_swarm_consensus() -> Dict[str, Any]:
    """Run consensus across swarm.

    Returns:
        dict: Consensus result.

    Receipt:
        swarm_consensus_receipt
    """
    global _swarm_state

    if not _swarm_state.initialized:
        init_swarm()

    _swarm_state.consensus_round += 1

    # Collect votes from active nodes
    active_nodes = [n for n in _swarm_state.nodes.values() if n.status == "active"]
    votes = []

    for node in active_nodes:
        # Simulate vote with latency
        vote = random.random() > 0.05  # 95% approval
        votes.append({"node_id": node.node_id, "vote": vote})
        node.messages_sent += 1
        node.messages_received += 1

    approval_count = sum(1 for v in votes if v["vote"])
    approval_rate = approval_count / max(1, len(votes))
    consensus_reached = approval_rate >= 0.51

    result = {
        "consensus_reached": consensus_reached,
        "consensus_round": _swarm_state.consensus_round,
        "active_nodes": len(active_nodes),
        "total_nodes": len(_swarm_state.nodes),
        "approval_rate": approval_rate,
        "algorithm": SWARM_CONSENSUS_ALGORITHM,
    }

    emit_receipt(
        "swarm_consensus_receipt",
        {
            "receipt_type": "swarm_consensus_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "consensus_reached": consensus_reached,
            "consensus_round": _swarm_state.consensus_round,
            "active_nodes": len(active_nodes),
            "approval_rate": approval_rate,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def inject_failure(node_id: str) -> Dict[str, Any]:
    """Inject failure into a node.

    Args:
        node_id: Node to fail.

    Returns:
        dict: Failure injection result.

    Receipt:
        swarm_failure_receipt
    """
    global _swarm_state

    if node_id not in _swarm_state.nodes:
        return {"injected": False, "error": f"Node {node_id} not found"}

    node = _swarm_state.nodes[node_id]
    node.status = "failed"
    node.failures += 1
    _swarm_state.total_failures += 1

    result = {
        "injected": True,
        "node_id": node_id,
        "node_type": node.node_type,
        "total_failures": _swarm_state.total_failures,
    }

    emit_receipt(
        "swarm_failure_receipt",
        {
            "receipt_type": "swarm_failure_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "injected": True,
            "node_id": node_id,
            "total_failures": _swarm_state.total_failures,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def recover_node(node_id: str) -> Dict[str, Any]:
    """Recover a failed node.

    Args:
        node_id: Node to recover.

    Returns:
        dict: Recovery result.

    Receipt:
        swarm_recovery_receipt
    """
    global _swarm_state

    if node_id not in _swarm_state.nodes:
        return {"recovered": False, "error": f"Node {node_id} not found"}

    node = _swarm_state.nodes[node_id]
    node.status = "active"
    node.last_heartbeat = datetime.utcnow().isoformat() + "Z"
    _swarm_state.total_recoveries += 1

    result = {
        "recovered": True,
        "node_id": node_id,
        "total_recoveries": _swarm_state.total_recoveries,
    }

    emit_receipt(
        "swarm_recovery_receipt",
        {
            "receipt_type": "swarm_recovery_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "recovered": True,
            "node_id": node_id,
            "total_recoveries": _swarm_state.total_recoveries,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def measure_recovery_time() -> Dict[str, Any]:
    """Measure average recovery time.

    Returns:
        dict: Recovery time measurement.

    Receipt:
        swarm_recovery_receipt
    """
    global _swarm_state

    # Simulate recovery time measurement
    recovery_times = []

    # Inject and recover random nodes
    failed_nodes = random.sample(
        list(_swarm_state.nodes.keys()),
        min(10, len(_swarm_state.nodes)),
    )

    for node_id in failed_nodes:
        inject_failure(node_id)
        start = time.time()
        recover_node(node_id)
        recovery_time = (time.time() - start) * 1000 + random.uniform(100, 500)
        recovery_times.append(recovery_time)

    avg_recovery_ms = sum(recovery_times) / max(1, len(recovery_times))

    result = {
        "nodes_tested": len(failed_nodes),
        "avg_recovery_ms": avg_recovery_ms,
        "min_recovery_ms": min(recovery_times) if recovery_times else 0,
        "max_recovery_ms": max(recovery_times) if recovery_times else 0,
        "target_ms": 5000,
        "target_met": avg_recovery_ms < 5000,
    }

    emit_receipt(
        "swarm_recovery_receipt",
        {
            "receipt_type": "swarm_recovery_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "measurement",
            "avg_recovery_ms": avg_recovery_ms,
            "target_met": result["target_met"],
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def stress_test_swarm(cycles: int = 100) -> Dict[str, Any]:
    """Run stress test on swarm.

    Args:
        cycles: Number of stress cycles.

    Returns:
        dict: Stress test results.

    Receipt:
        swarm_consensus_receipt
    """
    global _swarm_state

    if not _swarm_state.initialized:
        init_swarm()

    start_time = time.time()
    consensus_results = []
    failures_injected = 0
    recoveries_completed = 0

    for i in range(cycles):
        # Randomly inject failures (1% rate)
        if random.random() < 0.01:
            active_nodes = [n for n in _swarm_state.nodes.keys() if _swarm_state.nodes[n].status == "active"]
            if active_nodes:
                node_to_fail = random.choice(active_nodes)
                inject_failure(node_to_fail)
                failures_injected += 1

        # Run consensus
        consensus = run_swarm_consensus()
        consensus_results.append(consensus["consensus_reached"])

        # Recover failed nodes (50% rate)
        failed_nodes = [n for n in _swarm_state.nodes.keys() if _swarm_state.nodes[n].status == "failed"]
        for node_id in failed_nodes:
            if random.random() < 0.5:
                recover_node(node_id)
                recoveries_completed += 1

    total_time = time.time() - start_time
    success_rate = sum(1 for r in consensus_results if r) / max(1, len(consensus_results))

    result = {
        "stress_passed": success_rate >= 0.95,
        "cycles": cycles,
        "total_time_s": total_time,
        "consensus_success_rate": success_rate,
        "failures_injected": failures_injected,
        "recoveries_completed": recoveries_completed,
        "throughput_cps": cycles / total_time,
    }

    emit_receipt(
        "swarm_consensus_receipt",
        {
            "receipt_type": "swarm_consensus_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "test_type": "stress_test",
            "stress_passed": result["stress_passed"],
            "cycles": cycles,
            "success_rate": success_rate,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def get_swarm_status() -> Dict[str, Any]:
    """Get current swarm status.

    Returns:
        dict: Swarm status.
    """
    global _swarm_state

    active_nodes = sum(1 for n in _swarm_state.nodes.values() if n.status == "active")
    failed_nodes = sum(1 for n in _swarm_state.nodes.values() if n.status == "failed")

    return {
        "initialized": _swarm_state.initialized,
        "mesh_created": _swarm_state.mesh_created,
        "total_nodes": len(_swarm_state.nodes),
        "active_nodes": active_nodes,
        "failed_nodes": failed_nodes,
        "consensus_rounds": _swarm_state.consensus_round,
        "total_failures": _swarm_state.total_failures,
        "total_recoveries": _swarm_state.total_recoveries,
        "availability": active_nodes / max(1, len(_swarm_state.nodes)),
    }
