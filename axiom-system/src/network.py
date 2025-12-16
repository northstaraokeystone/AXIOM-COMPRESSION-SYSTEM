"""AXIOM-SYSTEM v2 Network Module - NetworkX-based relay topology.

Status: NEW
Purpose: Relay graph management, bandwidth allocation, congestion

Grok: "relay_topology graph (nx.DiGraph)" for topology
"""

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .core import emit_receipt
from .entropy import MOON_RELAY_BOOST


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkState:
    """Network state with relay topology and bandwidth allocation.

    Attributes:
        bandwidth_total: Earth anchor capacity, normalized to 1.0
        bandwidth_allocation: dict[str, float] body_id -> share, must sum to 1.0
        relay_graph: nx.DiGraph nodes=bodies, edges=relay paths, edge weights=efficiency
        congestion: dict[str, float] body_id -> congestion factor 0-1
    """
    bandwidth_total: float = 1.0
    bandwidth_allocation: dict = field(default_factory=dict)
    relay_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    congestion: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# RELAY GRAPH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_relay_graph(bodies: list) -> nx.DiGraph:
    """Initialize graph with Earth as anchor.

    Direct edges Earth<->each body.

    Initial State:
        Earth (anchor) --- Moon (1.3s, efficiency 0.95)
              |
              +----------- Mars (8m avg, efficiency 0.80)
              |
              +----------- Orbital (0s, efficiency 0.99)

    Args:
        bodies: List of body IDs (excluding earth)

    Returns:
        NetworkX DiGraph with relay topology
    """
    g = nx.DiGraph()

    # Add Earth as anchor node
    g.add_node("earth", delay_s=0, is_anchor=True)

    # Default efficiencies for direct Earth links
    default_efficiencies = {
        "moon": 0.95,    # 1.3s delay, high efficiency
        "mars": 0.80,    # 8min avg, lower efficiency due to distance
        "orbital": 0.99,  # LEO, nearly instantaneous
    }

    # Add nodes and edges for each body
    for body in bodies:
        if body == "earth":
            continue

        # Get default efficiency or use 0.85
        eff = default_efficiencies.get(body, 0.85)

        # Add node
        g.add_node(body)

        # Add bidirectional edges Earth <-> Body
        g.add_edge("earth", body, efficiency=eff, weight=1.0 / eff)
        g.add_edge(body, "earth", efficiency=eff, weight=1.0 / eff)

    return g


def add_relay(graph: nx.DiGraph, from_body: str, to_body: str, efficiency: float) -> nx.DiGraph:
    """Add relay edge between bodies.

    Moon->Mars relay has efficiency 0.85 (1.3s vs 8m).

    Args:
        graph: The relay graph
        from_body: Source body
        to_body: Target body
        efficiency: Relay efficiency (0-1)

    Returns:
        Updated graph
    """
    # Add bidirectional relay edges
    graph.add_edge(from_body, to_body, efficiency=efficiency, weight=1.0 / efficiency)
    graph.add_edge(to_body, from_body, efficiency=efficiency, weight=1.0 / efficiency)
    return graph


def shortest_relay_path(graph: nx.DiGraph, body: str, target: str = "earth") -> list:
    """Dijkstra shortest path weighted by efficiency.

    Args:
        graph: The relay graph
        body: Source body
        target: Target body (default "earth")

    Returns:
        List of body IDs in path (e.g., ["mars", "moon", "earth"])
    """
    try:
        return nx.shortest_path(graph, body, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [body, target]


def relay_efficiency(graph: nx.DiGraph, body: str, target: str = "earth") -> float:
    """Product of edge efficiencies along shortest path to target.

    Args:
        graph: The relay graph
        body: Source body
        target: Target body (default "earth")

    Returns:
        Combined efficiency (product of all edges)
    """
    path = shortest_relay_path(graph, body, target)
    if len(path) < 2:
        return 1.0

    total_efficiency = 1.0
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i + 1], {})
        total_efficiency *= edge_data.get("efficiency", 1.0)

    return total_efficiency


# ═══════════════════════════════════════════════════════════════════════════════
# BANDWIDTH ALLOCATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def allocate_bandwidth(bodies: list, demands: dict) -> dict:
    """Proportional allocation of bandwidth.

    If demand > supply, scale down all proportionally.

    Args:
        bodies: List of body IDs
        demands: dict[str, float] of requested bandwidth shares

    Returns:
        dict[str, float] of allocated shares (sums to 1.0)
    """
    total_demand = sum(demands.get(b, 0.0) for b in bodies)

    if total_demand <= 0:
        # Equal distribution
        return {b: 1.0 / len(bodies) for b in bodies if b != "earth"}

    if total_demand <= 1.0:
        # Demand fits, allocate as requested
        return {b: demands.get(b, 0.0) for b in bodies if b != "earth"}

    # Scale down proportionally
    scale = 1.0 / total_demand
    return {b: demands.get(b, 0.0) * scale for b in bodies if b != "earth"}


def compute_congestion(allocation: dict, demands: dict) -> dict:
    """Compute congestion factor for each body.

    congestion = 1 - (allocated / demanded) for each body.

    Args:
        allocation: Allocated bandwidth shares
        demands: Requested bandwidth shares

    Returns:
        dict[str, float] of congestion factors (0 = no congestion, 1 = fully congested)
    """
    congestion = {}
    for body, alloc in allocation.items():
        demand = demands.get(body, alloc)
        if demand > 0:
            congestion[body] = max(0.0, 1.0 - (alloc / demand))
        else:
            congestion[body] = 0.0
    return congestion


def external_rate_adjusted(base_rate: float, relay_eff: float, congestion: float,
                           solar_factor: float, conjunction: float,
                           moon_relay_active: bool = False) -> float:
    """Calculate adjusted external rate.

    base * relay_eff * (1 - congestion) * solar_factor * (1 - conjunction)

    Grok: "Moon relay upgrade -> Mars external_rate +40%"

    Args:
        base_rate: Base external compression rate
        relay_eff: Relay path efficiency
        congestion: Congestion factor (0-1)
        solar_factor: Solar activity factor (1 = nominal)
        conjunction: Conjunction factor (1 = full blackout)
        moon_relay_active: If True, apply MOON_RELAY_BOOST for Mars

    Returns:
        Adjusted external rate
    """
    adjusted = base_rate * relay_eff * (1 - congestion) * solar_factor * (1 - conjunction)

    # Apply Moon relay boost (Grok: +40% Mars external rate)
    if moon_relay_active:
        adjusted *= (1 + MOON_RELAY_BOOST)

    return adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def create_network_state(bodies: list) -> NetworkState:
    """Create initial network state.

    Args:
        bodies: List of body IDs

    Returns:
        Initialized NetworkState
    """
    graph = create_relay_graph(bodies)

    # Initial equal allocation
    non_earth = [b for b in bodies if b != "earth"]
    allocation = {b: 1.0 / len(non_earth) for b in non_earth} if non_earth else {}

    return NetworkState(
        bandwidth_total=1.0,
        bandwidth_allocation=allocation,
        relay_graph=graph,
        congestion={b: 0.0 for b in non_earth}
    )


def update_network_state(state: NetworkState, demands: dict) -> NetworkState:
    """Update network state with new demands.

    Args:
        state: Current network state
        demands: New bandwidth demands

    Returns:
        Updated NetworkState
    """
    bodies = list(state.bandwidth_allocation.keys())
    state.bandwidth_allocation = allocate_bandwidth(bodies + ["earth"], demands)
    state.congestion = compute_congestion(state.bandwidth_allocation, demands)
    return state


def enable_moon_relay(state: NetworkState) -> NetworkState:
    """Enable Moon relay to Mars.

    Adds Moon->Mars edge with efficiency 0.85.

    Args:
        state: Current network state

    Returns:
        Updated NetworkState with Moon relay enabled
    """
    state.relay_graph = add_relay(state.relay_graph, "moon", "mars", 0.85)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_network_receipt(state: NetworkState, trigger: str = "update") -> dict:
    """Emit network_state_receipt.

    Args:
        state: Current network state
        trigger: What triggered the emission

    Returns:
        Receipt dict
    """
    # Serialize relay paths
    relay_paths = {}
    for body in state.bandwidth_allocation.keys():
        relay_paths[body] = shortest_relay_path(state.relay_graph, body)

    data = {
        "bandwidth_total": state.bandwidth_total,
        "bandwidth_allocation": state.bandwidth_allocation,
        "congestion": state.congestion,
        "relay_paths": relay_paths,
        "trigger": trigger,
    }
    return emit_receipt("network_state", data)


def emit_relay_update_receipt(from_body: str, to_body: str, efficiency: float, action: str) -> dict:
    """Emit relay_update_receipt when topology changes.

    Args:
        from_body: Source body
        to_body: Target body
        efficiency: Relay efficiency
        action: "add" or "remove"

    Returns:
        Receipt dict
    """
    data = {
        "from_body": from_body,
        "to_body": to_body,
        "efficiency": efficiency,
        "action": action,
    }
    return emit_receipt("relay_update", data)
