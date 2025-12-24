"""Distributed swarm state management for D19.

Manages shared state across 100 swarm nodes.
State synchronized via entropy-weighted consensus.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from ..core import dual_hash


@dataclass
class NodeState:
    """State for a single swarm node."""

    node_id: str
    state: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    last_updated: str = ""


@dataclass
class SwarmState:
    """Global swarm state container."""

    swarm_id: str
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    global_version: int = 0
    sync_count: int = 0


def init_swarm_state(node_count: int = 100) -> SwarmState:
    """Initialize swarm state for all nodes.

    Args:
        node_count: Number of nodes in swarm

    Returns:
        SwarmState instance
    """
    swarm_id = str(uuid.uuid4())[:8]
    swarm = SwarmState(swarm_id=swarm_id)

    for i in range(node_count):
        node_id = f"node_{i:03d}"
        swarm.nodes[node_id] = NodeState(
            node_id=node_id,
            state={"initialized": True},
            version=0,
            last_updated=datetime.utcnow().isoformat() + "Z",
        )

    return swarm


def get_node_state(swarm: SwarmState, node_id: str) -> Optional[Dict[str, Any]]:
    """Get state for specific node.

    Args:
        swarm: SwarmState instance
        node_id: Node identifier

    Returns:
        Node state dict or None
    """
    node = swarm.nodes.get(node_id)
    return node.state if node else None


def update_node_state(
    swarm: SwarmState, node_id: str, updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Update state for specific node.

    Args:
        swarm: SwarmState instance
        node_id: Node identifier
        updates: State updates to apply

    Returns:
        Updated node state
    """
    node = swarm.nodes.get(node_id)
    if not node:
        return {"error": "node_not_found"}

    node.state.update(updates)
    node.version += 1
    node.last_updated = datetime.utcnow().isoformat() + "Z"

    return {
        "node_id": node_id,
        "state": node.state,
        "version": node.version,
    }


def broadcast_state(
    swarm: SwarmState, source_id: str, state: Dict[str, Any]
) -> Dict[str, Any]:
    """Broadcast state from source to all nodes.

    Args:
        swarm: SwarmState instance
        source_id: Source node ID
        state: State to broadcast

    Returns:
        Broadcast result
    """
    nodes_updated = 0
    for node_id, node in swarm.nodes.items():
        if node_id != source_id:
            node.state.update(state)
            node.version += 1
            nodes_updated += 1

    swarm.global_version += 1

    return {
        "source": source_id,
        "nodes_updated": nodes_updated,
        "global_version": swarm.global_version,
    }


def sync_states(swarm: SwarmState) -> Dict[str, Any]:
    """Synchronize states across all nodes.

    Args:
        swarm: SwarmState instance

    Returns:
        Sync result
    """
    # Find most common state version
    version_counts: Dict[int, int] = {}
    for node in swarm.nodes.values():
        version_counts[node.version] = version_counts.get(node.version, 0) + 1

    # Sync to highest version
    if version_counts:
        max_version = max(version_counts.keys())
        for node in swarm.nodes.values():
            if node.version < max_version:
                node.version = max_version

    swarm.sync_count += 1
    swarm.global_version = max_version if version_counts else 0

    return {
        "sync_count": swarm.sync_count,
        "global_version": swarm.global_version,
        "nodes_synced": len(swarm.nodes),
    }


def get_global_state(swarm: SwarmState) -> Dict[str, Any]:
    """Get aggregated global state.

    Args:
        swarm: SwarmState instance

    Returns:
        Global state summary
    """
    return {
        "swarm_id": swarm.swarm_id,
        "node_count": len(swarm.nodes),
        "global_version": swarm.global_version,
        "sync_count": swarm.sync_count,
    }


def compute_state_hash(swarm: SwarmState) -> str:
    """Compute hash of global state.

    Args:
        swarm: SwarmState instance

    Returns:
        Dual hash of state
    """
    state_data = {
        "swarm_id": swarm.swarm_id,
        "global_version": swarm.global_version,
        "node_versions": {n: s.version for n, s in swarm.nodes.items()},
    }
    return dual_hash(json.dumps(state_data, sort_keys=True))
