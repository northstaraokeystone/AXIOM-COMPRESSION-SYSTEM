"""AXIOM-SYSTEM v2 Cascade Module - Multi-body event propagation.

Status: NEW
Purpose: Event propagation via relay graph, CME cascade, Kessler effects

Grok: "Cascade Risk as Propagation Graph" - Use Dijkstra shortest path
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import deque

import networkx as nx

from .core import emit_receipt
from .network import NetworkState, shortest_relay_path


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CascadeState:
    """Cascade propagation state.

    Attributes:
        active_events: List of events currently propagating
        affected_bodies: Bodies affected by current cascade
        entropy_delta: Total entropy change from cascade
        cascade_depth: How many hops the cascade has traveled
    """
    active_events: List[Any] = None
    affected_bodies: List[str] = None
    entropy_delta: float = 0.0
    cascade_depth: int = 0

    def __post_init__(self):
        if self.active_events is None:
            self.active_events = []
        if self.affected_bodies is None:
            self.affected_bodies = []


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_affected_bodies(relay_graph: nx.DiGraph, event) -> List[str]:
    """BFS from event origin to find all affected bodies.

    Args:
        relay_graph: Network relay graph
        event: Event with 'origin' attribute

    Returns:
        List of affected body IDs
    """
    origin = getattr(event, 'origin', 'earth')

    # BFS traversal
    affected = []
    visited = set()
    queue = deque([origin])

    while queue:
        body = queue.popleft()
        if body in visited:
            continue
        visited.add(body)
        affected.append(body)

        # Add neighbors
        for neighbor in relay_graph.neighbors(body):
            if neighbor not in visited:
                queue.append(neighbor)

    return affected


def propagation_order(relay_graph: nx.DiGraph, origin: str) -> List[tuple]:
    """Get bodies in propagation order with delay.

    Returns list of (body, delay) tuples sorted by propagation time.

    Args:
        relay_graph: Network relay graph
        origin: Event origin body

    Returns:
        List of (body_id, delay) tuples
    """
    result = [(origin, 0.0)]

    for body in relay_graph.nodes():
        if body == origin:
            continue

        try:
            path = nx.shortest_path(relay_graph, origin, body, weight="weight")
            # Calculate total delay (sum of edge weights)
            delay = 0.0
            for i in range(len(path) - 1):
                edge_data = relay_graph.get_edge_data(path[i], path[i + 1], {})
                delay += edge_data.get("weight", 1.0)
            result.append((body, delay))
        except nx.NetworkXNoPath:
            # Body not reachable, infinite delay
            result.append((body, float('inf')))

    # Sort by delay
    result.sort(key=lambda x: x[1])
    return result


def cascade_cme(system_state, cme_event) -> "SystemState":
    """Cascade CME event through the system.

    Light-speed propagation, apply to all bodies by arrival time.

    Args:
        system_state: Current system state
        cme_event: CME event to cascade

    Returns:
        Updated SystemState
    """
    from .bodies.base import BodyState

    # Get propagation order
    order = propagation_order(system_state.network.relay_graph, "earth")

    entropy_delta = 0.0
    affected = []

    for body_id, delay in order:
        if body_id == "earth":
            continue  # Earth is origin, not affected

        if body_id not in system_state.bodies:
            continue

        body_state = system_state.bodies[body_id]

        # Import and apply CME impact
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from events.solar import cme_impact

        prev_entropy = body_state.entropy
        body_state = cme_impact(body_state, cme_event, system_state.sol)
        entropy_delta += body_state.entropy - prev_entropy

        system_state.bodies[body_id] = body_state
        affected.append(body_id)

    # Emit cascade receipt
    emit_cascade_receipt(cme_event, affected, entropy_delta)

    return system_state


def cascade_kessler(system_state) -> "SystemState":
    """Cascade Kessler syndrome effects.

    If Kessler active, set launch_blackout for all, isolate bodies.

    Args:
        system_state: Current system state

    Returns:
        Updated SystemState
    """
    if not system_state.orbital.kessler_active:
        return system_state

    # All bodies experience reduced external rates due to orbital debris
    debris_factor = 1.0 - (system_state.orbital.debris_ratio * 0.3)  # Up to 30% reduction

    for body_id, body_state in system_state.bodies.items():
        if body_id == "earth":
            continue

        body_state.external_rate *= debris_factor

        # Update status
        if body_state.status == "nominal":
            body_state.status = "stressed"

        system_state.bodies[body_id] = body_state

    # Emit cascade receipt
    affected = [b for b in system_state.bodies.keys() if b != "earth"]
    emit_cascade_receipt(
        type('KesslerEvent', (), {'origin': 'orbital', 'sol': system_state.sol})(),
        affected,
        0.0  # No direct entropy change
    )

    return system_state


def cascade_network_failure(system_state, failure_event) -> "SystemState":
    """Cascade network failure effects.

    Args:
        system_state: Current system state
        failure_event: Network failure event

    Returns:
        Updated SystemState
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from events.network import failure_impact

    # Apply failure to network
    system_state.network = failure_impact(system_state.network, failure_event)

    # Recalculate relay paths for all bodies
    from .network import shortest_relay_path, relay_efficiency

    for body_id, body_state in system_state.bodies.items():
        if body_id == "earth":
            continue

        body_state.relay_path = shortest_relay_path(system_state.network.relay_graph, body_id)

        # Recalculate external rate with new relay efficiency
        eff = relay_efficiency(system_state.network.relay_graph, body_id)
        body_state.external_rate *= eff

        system_state.bodies[body_id] = body_state

    return system_state


def propagate_event(system_state, event) -> "SystemState":
    """General event propagation router.

    Args:
        system_state: Current system state
        event: Event to propagate

    Returns:
        Updated SystemState
    """
    event_type = type(event).__name__

    if event_type == "CMEEvent":
        return cascade_cme(system_state, event)
    elif event_type == "CollisionEvent":
        # Apply collision to orbital, then cascade Kessler if needed
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from events.debris import collision_impact
        system_state.orbital = collision_impact(system_state.orbital, event)
        return cascade_kessler(system_state)
    elif event_type == "FailureEvent":
        return cascade_network_failure(system_state, event)
    else:
        # Unknown event type, return unchanged
        return system_state


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assess_cascade_risk(system_state) -> Dict[str, float]:
    """Assess cascade risk across the system.

    Returns dict with risk factors for each event type.

    Args:
        system_state: Current system state

    Returns:
        Dict[str, float] of risk factors
    """
    from .entropy import CME_PROBABILITY_PER_DAY, KESSLER_THRESHOLD

    risks = {
        "solar": CME_PROBABILITY_PER_DAY,
        "kessler": system_state.orbital.kessler_risk,
        "network": 0.0,
    }

    # Network risk based on relay redundancy
    relay_edges = list(system_state.network.relay_graph.edges())
    if len(relay_edges) < 3:
        risks["network"] = 0.3  # High risk with few relays
    elif len(relay_edges) < 5:
        risks["network"] = 0.1  # Moderate risk
    else:
        risks["network"] = 0.05  # Low risk with redundancy

    return risks


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_cascade_receipt(event, affected: List[str], entropy_delta: float) -> dict:
    """Emit CLAUDEME-compliant cascade_event_receipt.

    Args:
        event: Triggering event
        affected: List of affected body IDs
        entropy_delta: Total entropy change

    Returns:
        Receipt dict
    """
    data = {
        "event_type": type(event).__name__,
        "origin": getattr(event, 'origin', getattr(event, 'sol', 'unknown')),
        "affected_bodies": affected,
        "entropy_delta": entropy_delta,
        "cascade_depth": len(affected),
    }
    return emit_receipt("cascade_event", data)
