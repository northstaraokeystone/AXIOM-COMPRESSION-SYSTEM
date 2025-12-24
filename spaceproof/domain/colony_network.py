"""colony_network.py - Multi-Colony Network Dynamics for 1M Colonists

THE NETWORK INSIGHT:
    1000 colonies = 1000 independent ledgers.
    Inter-colony receipts = distributed consensus protocol.
    Network sovereignty = when majority can validate without Earth.

Source: SpaceProof v3.0 Multi-Tier Autonomy Network Evolution
Grok: "1M colonists by 2050"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math
import numpy as np

from ..core import emit_receipt, dual_hash, merkle
from .colony import ColonyConfig, generate as generate_colony, DEFAULT_CREW_SIZE

# === CONSTANTS ===

TENANT_ID = "spaceproof-network"

# Target scale (Grok)
MARS_COLONIST_TARGET_2050 = 1_000_000  # Grok: "1M colonists by 2050"
COLONY_NETWORK_SIZE_TARGET = 1000  # 1M @ 1000/colony
COLONISTS_PER_COLONY_TARGET = 1000  # Optimal colony size

# Network parameters
INTER_COLONY_BANDWIDTH_MBPS = 10.0  # Mars surface network
MAX_INTER_COLONY_DISTANCE_KM = 5000  # Max practical relay distance
BANDWIDTH_DECAY_PER_KM = 0.001  # Mbps lost per km distance

# Entropy thresholds
NETWORK_ENTROPY_STABLE_THRESHOLD = 0.90  # dH/dt <= 0 for 90% of time
PARTITION_RECOVERY_MAX_HOURS = 48  # Maximum acceptable partition time


class NetworkStatus(Enum):
    """Network operational status."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    PARTITIONED = "partitioned"
    CRITICAL = "critical"


@dataclass
class ColonyNode:
    """A colony in the network.

    Attributes:
        colony_id: Unique identifier
        name: Colony name
        population: Current population
        position: (x, y) position in km from reference
        decision_capacity_bps: Internal decision rate
        bandwidth_to_earth_mbps: Earth communication bandwidth
        active: Whether colony is operational
    """

    colony_id: str
    name: str
    population: int
    position: Tuple[float, float]
    decision_capacity_bps: float
    bandwidth_to_earth_mbps: float
    active: bool = True


@dataclass
class ColonyNetwork:
    """Network of Mars colonies.

    Attributes:
        network_id: Unique network identifier
        colonies: List of colony nodes
        n_colonies: Number of colonies
        total_population: Total network population
        inter_colony_links: Bandwidth between colony pairs
    """

    network_id: str
    colonies: List[ColonyNode]
    n_colonies: int
    total_population: int
    inter_colony_links: Dict[Tuple[str, str], float]


@dataclass
class NetworkState:
    """State of colony network at a point in time.

    Attributes:
        ts: Timestamp (ISO8601)
        network: Network configuration
        total_entropy: Current network entropy
        entropy_rate: dH/dt for the network
        partitions: List of partition groups
        earth_input_bps: Bits/sec from Earth
        network_internal_bps: Internal network bits/sec
        sovereign: Whether network is sovereign
        status: Network status
    """

    ts: str
    network: ColonyNetwork
    total_entropy: float
    entropy_rate: float
    partitions: List[List[str]]
    earth_input_bps: float
    network_internal_bps: float
    sovereign: bool
    status: str


def calculate_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    """Calculate distance between two colonies in km.

    Args:
        pos_a: Position of colony A
        pos_b: Position of colony B

    Returns:
        Distance in km
    """
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return math.sqrt(dx * dx + dy * dy)


def inter_colony_bandwidth(
    colony_a: ColonyNode, colony_b: ColonyNode, base_bandwidth: float = INTER_COLONY_BANDWIDTH_MBPS
) -> float:
    """Calculate bandwidth between two colonies.

    Bandwidth degrades with distance.

    Args:
        colony_a: First colony
        colony_b: Second colony
        base_bandwidth: Maximum bandwidth in Mbps

    Returns:
        Effective bandwidth in Mbps
    """
    distance = calculate_distance(colony_a.position, colony_b.position)

    if distance > MAX_INTER_COLONY_DISTANCE_KM:
        return 0.0  # Too far for direct link

    # Linear decay with distance
    bandwidth = base_bandwidth * (1 - BANDWIDTH_DECAY_PER_KM * distance)
    return max(0.0, bandwidth)


def initialize_network(
    n_colonies: int,
    colonists_per_colony: int = COLONISTS_PER_COLONY_TARGET,
    seed: int = 42,
) -> ColonyNetwork:
    """Create network of n colonies.

    Args:
        n_colonies: Number of colonies to create
        colonists_per_colony: Population per colony
        seed: Random seed

    Returns:
        Initialized ColonyNetwork
    """
    rng = np.random.default_rng(seed)
    colonies = []

    # Human decision rate from sovereignty_core
    HUMAN_DECISION_RATE_BPS = 10

    # Distribute colonies in a grid-like pattern with noise
    grid_size = int(math.ceil(math.sqrt(n_colonies)))
    spacing = MAX_INTER_COLONY_DISTANCE_KM / (grid_size + 1)

    for i in range(n_colonies):
        row = i // grid_size
        col = i % grid_size

        # Add position noise
        x = col * spacing + rng.normal(0, spacing * 0.1)
        y = row * spacing + rng.normal(0, spacing * 0.1)

        # Calculate decision capacity
        decision_capacity = colonists_per_colony * HUMAN_DECISION_RATE_BPS

        colony = ColonyNode(
            colony_id=f"C{i:04d}",
            name=f"Colony {i + 1}",
            population=colonists_per_colony,
            position=(x, y),
            decision_capacity_bps=decision_capacity,
            bandwidth_to_earth_mbps=2.0,  # Shared Mars-Earth relay
            active=True,
        )
        colonies.append(colony)

    # Calculate inter-colony links
    links = {}
    for i, c_a in enumerate(colonies):
        for j, c_b in enumerate(colonies):
            if i < j:  # Avoid duplicates
                bw = inter_colony_bandwidth(c_a, c_b)
                if bw > 0:
                    links[(c_a.colony_id, c_b.colony_id)] = bw

    network = ColonyNetwork(
        network_id=f"NET-{seed}",
        colonies=colonies,
        n_colonies=n_colonies,
        total_population=n_colonies * colonists_per_colony,
        inter_colony_links=links,
    )

    emit_receipt(
        "network_init_receipt",
        {
            "tenant_id": TENANT_ID,
            "network_id": network.network_id,
            "n_colonies": n_colonies,
            "total_population": network.total_population,
            "n_links": len(links),
            "data_hash": dual_hash(str(network.network_id)),
        },
    )

    return network


def network_entropy_rate(network: ColonyNetwork, rng: np.random.Generator) -> float:
    """Calculate dH/dt for entire network.

    Args:
        network: Current network state
        rng: Random number generator

    Returns:
        Network entropy rate (negative = stable)
    """
    # Base entropy from population
    population_entropy = math.log2(1 + network.total_population)

    # Entropy reduction from inter-colony communication
    communication_reduction = sum(network.inter_colony_links.values()) * 0.001

    # Random noise (can push rate positive or negative)
    noise = rng.normal(0, 0.1)

    # Typical rate should be slightly negative (entropy export)
    rate = noise - communication_reduction / population_entropy

    return rate


def detect_partition(network: ColonyNetwork) -> List[List[str]]:
    """Detect network partitions (isolated colony groups).

    Uses union-find to identify connected components.

    Args:
        network: Network to analyze

    Returns:
        List of partition groups (each is list of colony_ids)
    """
    # Build adjacency from links
    parent = {c.colony_id: c.colony_id for c in network.colonies}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union connected colonies
    for (c_a, c_b), bw in network.inter_colony_links.items():
        if bw > 0:
            union(c_a, c_b)

    # Group by root
    groups: Dict[str, List[str]] = {}
    for c in network.colonies:
        root = find(c.colony_id)
        if root not in groups:
            groups[root] = []
        groups[root].append(c.colony_id)

    partitions = list(groups.values())

    if len(partitions) > 1:
        emit_receipt(
            "partition_receipt",
            {
                "tenant_id": TENANT_ID,
                "network_id": network.network_id,
                "n_partitions": len(partitions),
                "partition_sizes": [len(p) for p in partitions],
            },
        )

    return partitions


def simulate_network(
    network: ColonyNetwork,
    duration_days: int,
    fleet_entropy_per_day: float = 0.0,
    seed: int = 42,
) -> List[NetworkState]:
    """Run network simulation.

    Args:
        network: Initial network state
        duration_days: Number of days to simulate
        fleet_entropy_per_day: Daily entropy injection from Starship fleet
        seed: Random seed

    Returns:
        List of NetworkState, one per day
    """
    rng = np.random.default_rng(seed)
    states = []

    total_entropy = 0.0

    for day in range(duration_days):
        # Calculate entropy rate
        entropy_rate = network_entropy_rate(network, rng)

        # Add fleet entropy injection
        entropy_rate -= fleet_entropy_per_day / 1e9  # Normalize

        # Update total entropy
        total_entropy += entropy_rate

        # Detect partitions
        partitions = detect_partition(network)

        # Calculate earth input (shared relay)
        earth_input_bps = network.colonies[0].bandwidth_to_earth_mbps * 1e6 / len(partitions)

        # Calculate internal network capacity
        network_internal_bps = (
            sum(c.decision_capacity_bps for c in network.colonies if c.active)
            + sum(network.inter_colony_links.values()) * 1e6
        )

        # Sovereignty check
        sovereign = network_internal_bps > earth_input_bps

        # Determine status
        if len(partitions) > 1:
            status = NetworkStatus.PARTITIONED.value
        elif entropy_rate > 0.1:
            status = NetworkStatus.CRITICAL.value
        elif entropy_rate > 0:
            status = NetworkStatus.DEGRADED.value
        else:
            status = NetworkStatus.NOMINAL.value

        state = NetworkState(
            ts=f"2050-{(day // 30) + 1:02d}-{(day % 30) + 1:02d}T12:00:00Z",
            network=network,
            total_entropy=total_entropy,
            entropy_rate=entropy_rate,
            partitions=partitions,
            earth_input_bps=earth_input_bps,
            network_internal_bps=network_internal_bps,
            sovereign=sovereign,
            status=status,
        )
        states.append(state)

    # Emit summary receipt
    stable_count = sum(1 for s in states if s.entropy_rate <= 0)
    emit_receipt(
        "colony_network_receipt",
        {
            "tenant_id": TENANT_ID,
            "network_id": network.network_id,
            "n_colonies": network.n_colonies,
            "total_population": network.total_population,
            "duration_days": duration_days,
            "entropy_stable_ratio": stable_count / duration_days,
            "final_sovereignty": states[-1].sovereign if states else False,
            "final_status": states[-1].status if states else "unknown",
        },
    )

    return states


def merge_partitions(
    network: ColonyNetwork,
    partition_a: List[str],
    partition_b: List[str],
    bridge_bandwidth: float = INTER_COLONY_BANDWIDTH_MBPS,
) -> ColonyNetwork:
    """Merge two partitioned network segments.

    Creates a bridge link between closest colonies in each partition.

    Args:
        network: Current network
        partition_a: First partition colony IDs
        partition_b: Second partition colony IDs
        bridge_bandwidth: Bandwidth for bridge link

    Returns:
        Updated network with merged partitions
    """
    # Find closest colony pair
    colony_map = {c.colony_id: c for c in network.colonies}

    min_distance = float("inf")
    bridge_pair = None

    for cid_a in partition_a:
        for cid_b in partition_b:
            c_a = colony_map[cid_a]
            c_b = colony_map[cid_b]
            dist = calculate_distance(c_a.position, c_b.position)
            if dist < min_distance:
                min_distance = dist
                bridge_pair = (cid_a, cid_b)

    if bridge_pair:
        # Add bridge link
        network.inter_colony_links[bridge_pair] = bridge_bandwidth

        emit_receipt(
            "partition_merge_receipt",
            {
                "tenant_id": TENANT_ID,
                "network_id": network.network_id,
                "bridge_colonies": bridge_pair,
                "bridge_bandwidth": bridge_bandwidth,
                "distance_km": min_distance,
            },
        )

    return network


def scale_network(
    network: ColonyNetwork,
    new_colonies: int,
    new_colonists_per_colony: int = COLONISTS_PER_COLONY_TARGET,
    seed: int = 42,
) -> ColonyNetwork:
    """Scale network by adding new colonies.

    Args:
        network: Existing network
        new_colonies: Number of colonies to add
        new_colonists_per_colony: Population per new colony
        seed: Random seed

    Returns:
        Expanded network
    """
    # Generate additional colonies
    extension = initialize_network(
        new_colonies,
        new_colonists_per_colony,
        seed=seed + network.n_colonies,
    )

    # Offset positions to avoid overlap
    max_x = max(c.position[0] for c in network.colonies)
    for c in extension.colonies:
        c.colony_id = f"C{network.n_colonies + int(c.colony_id[1:]):04d}"
        c.position = (c.position[0] + max_x + 100, c.position[1])

    # Merge colonies
    all_colonies = network.colonies + extension.colonies

    # Calculate new links
    new_links = dict(network.inter_colony_links)
    for (c_a, c_b), bw in extension.inter_colony_links.items():
        new_links[(c_a, c_b)] = bw

    # Add bridge links between old and new
    for old in network.colonies[-5:]:  # Last 5 of old network
        for new in extension.colonies[:5]:  # First 5 of new network
            bw = inter_colony_bandwidth(old, new)
            if bw > 0:
                new_links[(old.colony_id, new.colony_id)] = bw

    expanded = ColonyNetwork(
        network_id=network.network_id,
        colonies=all_colonies,
        n_colonies=len(all_colonies),
        total_population=sum(c.population for c in all_colonies),
        inter_colony_links=new_links,
    )

    emit_receipt(
        "network_scale_receipt",
        {
            "tenant_id": TENANT_ID,
            "network_id": expanded.network_id,
            "colonies_added": new_colonies,
            "new_total_colonies": expanded.n_colonies,
            "new_total_population": expanded.total_population,
        },
    )

    return expanded
