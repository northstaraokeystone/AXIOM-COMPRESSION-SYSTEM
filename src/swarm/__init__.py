"""Swarm intelligence package for D19 emergent coordination.

This package implements entropy-gradient based coordination across 100 nodes.
No central coordinator - entropy IS the coordinator.
"""

from .entropy_engine import (
    EntropyEngine,
    init_entropy_engine,
    measure_local_entropy,
    compute_gradient,
    propagate_gradient,
    detect_entropy_sink,
    detect_entropy_source,
    coordinate_via_gradient,
    measure_swarm_coherence,
    simulate_coordination,
    get_engine_status,
    NODE_COUNT,
    ENTROPY_SAMPLE_RATE_HZ,
    GRADIENT_THRESHOLD,
    CONVERGENCE_TARGET,
    MESH_CONNECTIONS,
)

from .gradient_coordinator import (
    GradientCoordinator,
    init_coordinator,
    propose_action,
    collect_votes,
    achieve_consensus,
    execute_coordinated,
    detect_partition,
    heal_partition,
    measure_coordination_latency,
    get_coordinator_status,
)

from .swarm_state import (
    SwarmState,
    init_swarm_state,
    get_node_state,
    update_node_state,
    broadcast_state,
    sync_states,
    get_global_state,
    compute_state_hash,
)

from .topology import (
    SwarmTopology,
    init_topology,
    add_node,
    remove_node,
    get_neighbors,
    compute_mesh_density,
    detect_clusters,
    rebalance_topology,
)

__all__ = [
    # Entropy Engine
    "EntropyEngine",
    "init_entropy_engine",
    "measure_local_entropy",
    "compute_gradient",
    "propagate_gradient",
    "detect_entropy_sink",
    "detect_entropy_source",
    "coordinate_via_gradient",
    "measure_swarm_coherence",
    "simulate_coordination",
    "get_engine_status",
    "NODE_COUNT",
    "ENTROPY_SAMPLE_RATE_HZ",
    "GRADIENT_THRESHOLD",
    "CONVERGENCE_TARGET",
    "MESH_CONNECTIONS",
    # Gradient Coordinator
    "GradientCoordinator",
    "init_coordinator",
    "propose_action",
    "collect_votes",
    "achieve_consensus",
    "execute_coordinated",
    "detect_partition",
    "heal_partition",
    "measure_coordination_latency",
    "get_coordinator_status",
    # Swarm State
    "SwarmState",
    "init_swarm_state",
    "get_node_state",
    "update_node_state",
    "broadcast_state",
    "sync_states",
    "get_global_state",
    "compute_state_hash",
    # Topology
    "SwarmTopology",
    "init_topology",
    "add_node",
    "remove_node",
    "get_neighbors",
    "compute_mesh_density",
    "detect_clusters",
    "rebalance_topology",
]

RECEIPT_SCHEMA = {
    "module": "src.swarm",
    "receipt_types": [
        "entropy_measurement_receipt",
        "gradient_propagation_receipt",
        "coordination_action_receipt",
        "swarm_coherence_receipt",
        "proposal_receipt",
        "vote_receipt",
        "consensus_receipt",
        "partition_receipt",
        "heal_receipt",
    ],
    "version": "19.0.0",
}
