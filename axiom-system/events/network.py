"""AXIOM-SYSTEM v2 Network Events Module - Relay failures, congestion.

Status: NEW
Purpose: Network failure event generation and recovery

Grok: Network topology affects all bodies via relay graph.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import emit_receipt
from src.network import NetworkState


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

RELAY_FAILURE_PROBABILITY = 0.005      # 0.5% daily failure probability per relay
CONGESTION_SPIKE_PROBABILITY = 0.02    # 2% daily congestion spike
FAILURE_RECOVERY_SOLS = 5              # Sols to recover from failure
CONGESTION_DURATION_SOLS = 2           # Sols of elevated congestion


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureEvent:
    """Relay failure event.

    Attributes:
        sol: Sol when failure occurred
        edge: Tuple of (from_body, to_body)
        recovery_sols: Sols until recovery
    """
    sol: int = 0
    edge: Tuple[str, str] = None
    recovery_sols: int = FAILURE_RECOVERY_SOLS


@dataclass
class CongestionEvent:
    """Network congestion spike event.

    Attributes:
        sol: Sol when congestion started
        affected_bodies: List of affected body IDs
        congestion_increase: Amount of congestion increase (0-1)
        duration_sols: How long congestion lasts
    """
    sol: int = 0
    affected_bodies: list = None
    congestion_increase: float = 0.3
    duration_sols: int = CONGESTION_DURATION_SOLS

    def __post_init__(self):
        if self.affected_bodies is None:
            self.affected_bodies = []


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK EVENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def roll_relay_failure(network_state: NetworkState, seed: int = None) -> Optional[Tuple[str, str]]:
    """Roll for relay failure event.

    Args:
        network_state: Current network state
        seed: Random seed

    Returns:
        Failed relay edge (from_body, to_body) or None
    """
    rng = random.Random(seed)

    # Get all relay edges (excluding Earth anchor edges)
    edges = []
    for u, v in network_state.relay_graph.edges():
        if u != "earth" and v != "earth":
            edges.append((u, v))

    if not edges:
        return None

    # Check each edge for failure
    for edge in edges:
        if rng.random() < RELAY_FAILURE_PROBABILITY:
            return edge

    return None


def roll_congestion_spike(network_state: NetworkState, seed: int = None) -> bool:
    """Roll for congestion spike event.

    Args:
        network_state: Current network state
        seed: Random seed

    Returns:
        True if congestion spike occurs
    """
    rng = random.Random(seed)
    return rng.random() < CONGESTION_SPIKE_PROBABILITY


def create_failure_event(sol: int, edge: Tuple[str, str]) -> FailureEvent:
    """Create relay failure event.

    Args:
        sol: Sol when failure occurred
        edge: Failed relay edge

    Returns:
        FailureEvent
    """
    return FailureEvent(
        sol=sol,
        edge=edge,
        recovery_sols=FAILURE_RECOVERY_SOLS,
    )


def create_congestion_event(sol: int, network_state: NetworkState,
                            seed: int = None) -> CongestionEvent:
    """Create congestion spike event.

    Args:
        sol: Sol when congestion started
        network_state: Current network state
        seed: Random seed

    Returns:
        CongestionEvent
    """
    rng = random.Random(seed)

    # Randomly select affected bodies
    bodies = list(network_state.bandwidth_allocation.keys())
    n_affected = rng.randint(1, len(bodies))
    affected = rng.sample(bodies, n_affected)

    # Random congestion increase
    congestion_increase = 0.2 + rng.random() * 0.3  # 20-50%

    return CongestionEvent(
        sol=sol,
        affected_bodies=affected,
        congestion_increase=congestion_increase,
        duration_sols=CONGESTION_DURATION_SOLS,
    )


def failure_impact(network_state: NetworkState, event: FailureEvent) -> NetworkState:
    """Apply relay failure impact.

    Removes failed edge from relay graph.

    Args:
        network_state: Current network state
        event: Failure event

    Returns:
        Updated NetworkState
    """
    if event.edge is None:
        return network_state

    u, v = event.edge

    # Remove bidirectional edges
    if network_state.relay_graph.has_edge(u, v):
        network_state.relay_graph.remove_edge(u, v)
    if network_state.relay_graph.has_edge(v, u):
        network_state.relay_graph.remove_edge(v, u)

    return network_state


def congestion_impact(network_state: NetworkState, event: CongestionEvent) -> NetworkState:
    """Apply congestion spike impact.

    Increases congestion for affected bodies.

    Args:
        network_state: Current network state
        event: Congestion event

    Returns:
        Updated NetworkState
    """
    for body in event.affected_bodies:
        if body in network_state.congestion:
            current = network_state.congestion[body]
            network_state.congestion[body] = min(1.0, current + event.congestion_increase)

    return network_state


def recover_relay(network_state: NetworkState, edge: Tuple[str, str],
                  efficiency: float = 0.85) -> NetworkState:
    """Recover failed relay edge.

    Args:
        network_state: Current network state
        edge: Edge to recover
        efficiency: Restored efficiency

    Returns:
        Updated NetworkState
    """
    from src.network import add_relay

    u, v = edge
    return add_relay(network_state.relay_graph, u, v, efficiency)


def recover_congestion(network_state: NetworkState, event: CongestionEvent) -> NetworkState:
    """Recover from congestion spike.

    Args:
        network_state: Current network state
        event: Congestion event to recover from

    Returns:
        Updated NetworkState
    """
    for body in event.affected_bodies:
        if body in network_state.congestion:
            current = network_state.congestion[body]
            network_state.congestion[body] = max(0.0, current - event.congestion_increase)

    return network_state


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_failure_receipt(event: FailureEvent) -> dict:
    """Emit relay failure receipt.

    Args:
        event: Failure event

    Returns:
        Receipt dict
    """
    data = {
        "event_type": "relay_failure",
        "sol": event.sol,
        "edge": event.edge,
        "recovery_sols": event.recovery_sols,
    }
    return emit_receipt("network_event", data)


def emit_congestion_receipt(event: CongestionEvent) -> dict:
    """Emit congestion spike receipt.

    Args:
        event: Congestion event

    Returns:
        Receipt dict
    """
    data = {
        "event_type": "congestion_spike",
        "sol": event.sol,
        "affected_bodies": event.affected_bodies,
        "congestion_increase": event.congestion_increase,
        "duration_sols": event.duration_sols,
    }
    return emit_receipt("network_event", data)
