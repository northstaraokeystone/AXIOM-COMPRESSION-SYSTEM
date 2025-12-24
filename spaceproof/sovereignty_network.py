"""sovereignty_network.py - Network Sovereignty Threshold Calculation

THE NETWORK SOVEREIGNTY INSIGHT:
    Single colony sovereignty ≠ network sovereignty.
    Network sovereignty = when majority of ledgers can validate without Earth.
    This is distributed consensus at light-speed scale.

Source: SpaceProof v3.0 Multi-Tier Autonomy Network Evolution
Grok: "autonomous proof infrastructure"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .core import emit_receipt, dual_hash
from .domain.colony_network import ColonyNetwork, ColonyNode

# === CONSTANTS ===

TENANT_ID = "spaceproof-network-sovereignty"

# Network sovereignty thresholds
MIN_SOVEREIGN_COLONIES = 3  # Minimum for distributed validation
MAJORITY_THRESHOLD = 0.51  # Majority required for consensus
EARTH_BANDWIDTH_TOTAL_MBPS = 100.0  # Total Mars-Earth relay capacity

# Light delay (from sovereignty_core.py)
MARS_LIGHT_DELAY_AVG_S = 480  # 8 min average
MARS_LIGHT_DELAY_MIN_S = 180  # 3 min opposition
MARS_LIGHT_DELAY_MAX_S = 1320  # 22 min conjunction


@dataclass
class NetworkSovereigntyResult:
    """Result of network sovereignty calculation.

    Attributes:
        n_colonies: Number of colonies in network
        total_population: Total network population
        network_decision_capacity_bps: Total internal decision rate
        earth_input_bps: Earth communication rate
        sovereign_ratio: internal / external rate
        sovereign: Whether network is sovereign
        threshold_colonies: Minimum colonies for sovereignty
        consensus_possible: Whether distributed consensus is possible
    """

    n_colonies: int
    total_population: int
    network_decision_capacity_bps: float
    earth_input_bps: float
    sovereign_ratio: float
    sovereign: bool
    threshold_colonies: int
    consensus_possible: bool


def network_decision_capacity(
    network: ColonyNetwork,
    augmentation: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate total decision capacity across all colonies.

    Args:
        network: Colony network
        augmentation: Optional dict of colony_id -> augmentation_factor

    Returns:
        Total network decision rate in bits/sec
    """
    augmentation = augmentation or {}

    total_capacity = 0.0
    for colony in network.colonies:
        if not colony.active:
            continue

        # Get augmentation factor for this colony (default 1.0)
        factor = augmentation.get(colony.colony_id, 1.0)

        # Colony decision capacity scaled by augmentation
        colony_capacity = colony.decision_capacity_bps * factor
        total_capacity += colony_capacity

    # Add inter-colony communication capacity
    # Each Mbps of inter-colony bandwidth adds decision coordination
    link_capacity = sum(network.inter_colony_links.values()) * 1e6 * 0.01  # 1% efficiency

    return total_capacity + link_capacity


def earth_input_rate(
    network: ColonyNetwork,
    earth_bandwidth_mbps: float = EARTH_BANDWIDTH_TOTAL_MBPS,
    light_delay_s: float = MARS_LIGHT_DELAY_AVG_S,
) -> float:
    """Calculate Earth input rate for the network.

    Earth bandwidth is shared across all colonies.
    Rate is penalized by round-trip light delay.

    Args:
        network: Colony network
        earth_bandwidth_mbps: Total Earth-Mars bandwidth
        light_delay_s: One-way light delay in seconds

    Returns:
        Earth input rate in bits/sec
    """
    # Per-colony share of Earth bandwidth
    per_colony_bps = (earth_bandwidth_mbps * 1e6) / network.n_colonies

    # Light delay penalty (round trip)
    delay_penalty = 2 * light_delay_s

    # Effective rate considering delay
    effective_rate = per_colony_bps / delay_penalty

    return effective_rate * network.n_colonies


def network_sovereignty_threshold(
    network: ColonyNetwork,
    earth_bandwidth_mbps: float = EARTH_BANDWIDTH_TOTAL_MBPS,
    augmentation: Optional[Dict[str, float]] = None,
) -> Dict:
    """Find N colonies where network becomes autonomous.

    Binary search for the threshold where network_internal > earth_input.

    Args:
        network: Colony network
        earth_bandwidth_mbps: Earth-Mars bandwidth
        augmentation: Optional augmentation factors per colony

    Returns:
        Dict with threshold analysis
    """
    # Calculate current capacities
    internal_bps = network_decision_capacity(network, augmentation)
    external_bps = earth_input_rate(network, earth_bandwidth_mbps)

    sovereign_ratio = internal_bps / external_bps if external_bps > 0 else float("inf")
    sovereign = internal_bps > external_bps

    # Binary search for threshold colonies
    # If already sovereign, find minimum colonies needed
    # If not sovereign, find how many more needed
    threshold = network.n_colonies

    if sovereign:
        # Find minimum colonies that maintain sovereignty
        for n in range(MIN_SOVEREIGN_COLONIES, network.n_colonies):
            # Simulate with n colonies
            test_internal = internal_bps * (n / network.n_colonies)
            test_external = external_bps * (n / network.n_colonies)
            if test_internal > test_external:
                threshold = n
                break
    else:
        # Find colonies needed to achieve sovereignty
        # Estimate: need internal to grow faster than external
        ratio_needed = external_bps / internal_bps
        threshold = int(math.ceil(network.n_colonies * ratio_needed))

    # Check if distributed consensus is possible
    consensus_possible = (
        network.n_colonies >= MIN_SOVEREIGN_COLONIES
        and len(set(c.colony_id for c in network.colonies if c.active)) >= MIN_SOVEREIGN_COLONIES
    )

    result = {
        "n_colonies": network.n_colonies,
        "total_population": network.total_population,
        "network_decision_capacity_bps": internal_bps,
        "earth_input_bps": external_bps,
        "sovereign_ratio": sovereign_ratio,
        "sovereign": sovereign,
        "threshold_colonies": threshold,
        "consensus_possible": consensus_possible,
    }

    emit_receipt(
        "network_sovereignty_receipt",
        {
            "tenant_id": TENANT_ID,
            "network_id": network.network_id,
            **result,
        },
    )

    return result


def validate_network_sovereignty(
    network: ColonyNetwork,
    earth_bandwidth_mbps: float = EARTH_BANDWIDTH_TOTAL_MBPS,
    augmentation: Optional[Dict[str, float]] = None,
) -> bool:
    """Check if network_internal_bps > earth_input_bps.

    Args:
        network: Colony network
        earth_bandwidth_mbps: Earth-Mars bandwidth
        augmentation: Optional augmentation factors

    Returns:
        True if network is sovereign
    """
    internal = network_decision_capacity(network, augmentation)
    external = earth_input_rate(network, earth_bandwidth_mbps)
    return internal > external


def sovereignty_by_partition(
    network: ColonyNetwork,
    partitions: List[List[str]],
    earth_bandwidth_mbps: float = EARTH_BANDWIDTH_TOTAL_MBPS,
) -> Dict[int, Dict]:
    """Calculate sovereignty for each partition independently.

    When network is partitioned, each partition must be evaluated
    for its own sovereignty status.

    Args:
        network: Colony network
        partitions: List of partition groups (colony ID lists)
        earth_bandwidth_mbps: Earth-Mars bandwidth

    Returns:
        Dict mapping partition index to sovereignty result
    """
    colony_map = {c.colony_id: c for c in network.colonies}
    results = {}

    for i, partition in enumerate(partitions):
        # Get colonies in this partition
        partition_colonies = [colony_map[cid] for cid in partition if cid in colony_map]

        # Calculate partition capacity
        partition_capacity = sum(c.decision_capacity_bps for c in partition_colonies)

        # Earth bandwidth split by number of partitions
        partition_earth_bw = earth_bandwidth_mbps / len(partitions)
        partition_external = earth_input_rate(
            ColonyNetwork(
                network_id=f"{network.network_id}-P{i}",
                colonies=partition_colonies,
                n_colonies=len(partition_colonies),
                total_population=sum(c.population for c in partition_colonies),
                inter_colony_links={},
            ),
            partition_earth_bw,
        )

        results[i] = {
            "partition_id": i,
            "n_colonies": len(partition_colonies),
            "population": sum(c.population for c in partition_colonies),
            "internal_capacity_bps": partition_capacity,
            "earth_input_bps": partition_external,
            "sovereign": partition_capacity > partition_external,
        }

    emit_receipt(
        "partition_sovereignty_receipt",
        {
            "tenant_id": TENANT_ID,
            "network_id": network.network_id,
            "n_partitions": len(partitions),
            "sovereign_partitions": sum(1 for r in results.values() if r["sovereign"]),
            "partition_results": list(results.values()),
        },
    )

    return results


def network_sovereignty_sensitivity(
    network: ColonyNetwork,
    bandwidth_range: tuple = (10.0, 200.0),
    delay_range: tuple = (180.0, 1320.0),
    steps: int = 10,
) -> List[Dict]:
    """Analyze network sovereignty sensitivity to bandwidth and delay.

    Args:
        network: Colony network
        bandwidth_range: (min, max) Earth bandwidth in Mbps
        delay_range: (min, max) light delay in seconds
        steps: Number of steps per dimension

    Returns:
        List of sensitivity analysis results
    """
    results = []

    bw_step = (bandwidth_range[1] - bandwidth_range[0]) / steps
    delay_step = (delay_range[1] - delay_range[0]) / steps

    internal = network_decision_capacity(network)

    for i in range(steps + 1):
        bw = bandwidth_range[0] + i * bw_step
        for j in range(steps + 1):
            delay = delay_range[0] + j * delay_step

            external = earth_input_rate(network, bw, delay)
            sovereign = internal > external

            results.append(
                {
                    "bandwidth_mbps": bw,
                    "delay_s": delay,
                    "internal_rate": internal,
                    "external_rate": external,
                    "sovereign_ratio": internal / external if external > 0 else float("inf"),
                    "sovereign": sovereign,
                }
            )

    return results


def calculate_network_consensus_time(
    network: ColonyNetwork,
    message_size_bytes: int = 1000,
) -> float:
    """Calculate time to achieve network-wide consensus.

    Consensus requires message propagation across all links.
    Time is bounded by the slowest path in the network.

    Args:
        network: Colony network
        message_size_bytes: Size of consensus message

    Returns:
        Consensus time in seconds
    """
    if network.n_colonies <= 1:
        return 0.0

    # Find minimum bandwidth (bottleneck)
    if not network.inter_colony_links:
        return float("inf")  # No links = no consensus

    min_bandwidth_mbps = min(network.inter_colony_links.values())
    min_bandwidth_bps = min_bandwidth_mbps * 1e6

    # Time to transmit message
    transmit_time = (message_size_bytes * 8) / min_bandwidth_bps

    # Assume worst case: message must traverse all links
    # (This is simplified; real consensus would use spanning tree)
    hops = network.n_colonies - 1
    total_time = transmit_time * hops

    return total_time
