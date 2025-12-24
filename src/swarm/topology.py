"""Swarm topology management for D19.

Dynamic topology for 100-node swarm mesh.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TopologyNode:
    """Node in topology graph."""

    node_id: str
    neighbors: Set[str] = field(default_factory=set)
    cluster_id: Optional[str] = None


@dataclass
class SwarmTopology:
    """Swarm network topology."""

    topology_id: str
    nodes: Dict[str, TopologyNode] = field(default_factory=dict)
    clusters: Dict[str, List[str]] = field(default_factory=dict)
    mesh_density: float = 0.0


def init_topology(node_count: int = 100, full_mesh: bool = True) -> SwarmTopology:
    """Initialize swarm topology.

    Args:
        node_count: Number of nodes
        full_mesh: Whether to create full mesh connections

    Returns:
        SwarmTopology instance
    """
    topology_id = str(uuid.uuid4())[:8]
    topology = SwarmTopology(topology_id=topology_id)

    # Create nodes
    for i in range(node_count):
        node_id = f"node_{i:03d}"
        topology.nodes[node_id] = TopologyNode(node_id=node_id)

    # Create mesh connections
    if full_mesh:
        for node_id, node in topology.nodes.items():
            for other_id in topology.nodes:
                if other_id != node_id:
                    node.neighbors.add(other_id)

    # Calculate mesh density
    topology.mesh_density = compute_mesh_density(topology)

    return topology


def add_node(
    topology: SwarmTopology, node_id: str, neighbors: List[str] = None
) -> Dict[str, Any]:
    """Add node to topology.

    Args:
        topology: SwarmTopology instance
        node_id: Node identifier
        neighbors: Initial neighbors

    Returns:
        Add result
    """
    if node_id in topology.nodes:
        return {"error": "node_exists", "node_id": node_id}

    node = TopologyNode(node_id=node_id)
    if neighbors:
        for neighbor_id in neighbors:
            if neighbor_id in topology.nodes:
                node.neighbors.add(neighbor_id)
                topology.nodes[neighbor_id].neighbors.add(node_id)

    topology.nodes[node_id] = node
    topology.mesh_density = compute_mesh_density(topology)

    return {
        "node_id": node_id,
        "neighbors": len(node.neighbors),
        "total_nodes": len(topology.nodes),
    }


def remove_node(topology: SwarmTopology, node_id: str) -> Dict[str, Any]:
    """Remove node from topology.

    Args:
        topology: SwarmTopology instance
        node_id: Node identifier

    Returns:
        Remove result
    """
    if node_id not in topology.nodes:
        return {"error": "node_not_found", "node_id": node_id}

    # Remove from all neighbor lists
    for node in topology.nodes.values():
        node.neighbors.discard(node_id)

    del topology.nodes[node_id]
    topology.mesh_density = compute_mesh_density(topology)

    return {
        "node_id": node_id,
        "removed": True,
        "total_nodes": len(topology.nodes),
    }


def get_neighbors(topology: SwarmTopology, node_id: str) -> List[str]:
    """Get neighbors for node.

    Args:
        topology: SwarmTopology instance
        node_id: Node identifier

    Returns:
        List of neighbor IDs
    """
    node = topology.nodes.get(node_id)
    return list(node.neighbors) if node else []


def compute_mesh_density(topology: SwarmTopology) -> float:
    """Compute mesh density of topology.

    Density = actual_edges / possible_edges

    Args:
        topology: SwarmTopology instance

    Returns:
        Mesh density 0-1
    """
    n = len(topology.nodes)
    if n < 2:
        return 0.0

    possible_edges = n * (n - 1) / 2
    actual_edges = sum(len(node.neighbors) for node in topology.nodes.values()) / 2

    return actual_edges / possible_edges if possible_edges > 0 else 0.0


def detect_clusters(topology: SwarmTopology, min_size: int = 5) -> Dict[str, List[str]]:
    """Detect clusters in topology.

    Uses connected components with minimum size filter.

    Args:
        topology: SwarmTopology instance
        min_size: Minimum cluster size

    Returns:
        Dict mapping cluster_id to node list
    """
    visited = set()
    clusters = {}
    cluster_idx = 0

    def dfs(node_id: str, cluster: List[str]):
        if node_id in visited:
            return
        visited.add(node_id)
        cluster.append(node_id)

        node = topology.nodes.get(node_id)
        if node:
            for neighbor_id in node.neighbors:
                dfs(neighbor_id, cluster)

    for node_id in topology.nodes:
        if node_id not in visited:
            cluster = []
            dfs(node_id, cluster)
            if len(cluster) >= min_size:
                cluster_id = f"cluster_{cluster_idx:02d}"
                clusters[cluster_id] = cluster
                for nid in cluster:
                    if nid in topology.nodes:
                        topology.nodes[nid].cluster_id = cluster_id
                cluster_idx += 1

    topology.clusters = clusters
    return clusters


def rebalance_topology(
    topology: SwarmTopology, target_density: float = 0.5
) -> Dict[str, Any]:
    """Rebalance topology toward target density.

    Args:
        topology: SwarmTopology instance
        target_density: Target mesh density

    Returns:
        Rebalance result
    """
    current_density = topology.mesh_density
    edges_added = 0
    edges_removed = 0

    n = len(topology.nodes)
    possible_edges = n * (n - 1) / 2
    target_edges = int(target_density * possible_edges)
    current_edges = int(current_density * possible_edges)

    if current_edges < target_edges:
        # Add edges
        nodes_list = list(topology.nodes.keys())
        for i, node_id in enumerate(nodes_list):
            if edges_added >= target_edges - current_edges:
                break
            node = topology.nodes[node_id]
            for other_id in nodes_list[i + 1 :]:
                if other_id not in node.neighbors:
                    node.neighbors.add(other_id)
                    topology.nodes[other_id].neighbors.add(node_id)
                    edges_added += 1
                    if edges_added >= target_edges - current_edges:
                        break

    topology.mesh_density = compute_mesh_density(topology)

    return {
        "previous_density": round(current_density, 4),
        "target_density": target_density,
        "new_density": round(topology.mesh_density, 4),
        "edges_added": edges_added,
        "edges_removed": edges_removed,
    }
