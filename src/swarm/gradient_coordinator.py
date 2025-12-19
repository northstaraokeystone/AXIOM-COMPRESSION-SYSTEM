"""Gradient-based coordination for D19 swarm intelligence.

Entropy-based coordination without central authority.
Proposals propagate via gradients, votes weighted by entropy.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID
from .entropy_engine import EntropyEngine, GRADIENT_THRESHOLD


@dataclass
class Proposal:
    """Coordination proposal."""

    proposal_id: str
    action: Dict[str, Any]
    proposer: str
    votes: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


@dataclass
class GradientCoordinator:
    """Gradient-based coordinator for swarm actions."""

    coordinator_id: str
    engine: EntropyEngine
    proposals: Dict[str, Proposal] = field(default_factory=dict)
    partitions: List[List[str]] = field(default_factory=list)
    latency_ms: float = 0.0


def init_coordinator(engine: EntropyEngine) -> GradientCoordinator:
    """Initialize gradient coordinator.

    Args:
        engine: EntropyEngine instance

    Returns:
        GradientCoordinator instance
    """
    coordinator_id = str(uuid.uuid4())[:8]
    return GradientCoordinator(coordinator_id=coordinator_id, engine=engine)


def propose_action(coord: GradientCoordinator, action: Dict[str, Any]) -> Dict[str, Any]:
    """Propose action via gradient broadcast.

    Proposal propagates along entropy gradients.
    Nodes with lower entropy receive first.

    Args:
        coord: GradientCoordinator instance
        action: Action to propose

    Returns:
        Proposal result

    Receipt: proposal_receipt
    """
    proposal_id = str(uuid.uuid4())[:8]
    proposer = action.get("proposer", "system")

    proposal = Proposal(proposal_id=proposal_id, action=action, proposer=proposer)
    coord.proposals[proposal_id] = proposal

    # Simulate gradient-based propagation
    # Lower entropy nodes receive proposal first
    nodes_sorted = sorted(coord.engine.nodes.values(), key=lambda n: n.entropy)
    propagation_order = [n.node_id for n in nodes_sorted]

    result = {
        "proposal_id": proposal_id,
        "action": action,
        "proposer": proposer,
        "propagation_order": propagation_order[:5],  # First 5 for display
        "status": "proposed",
    }

    emit_receipt(
        "proposal",
        {
            "receipt_type": "proposal",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "coordinator_id": coord.coordinator_id,
            "proposal_id": proposal_id,
            "action_type": action.get("type", "unknown"),
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def collect_votes(coord: GradientCoordinator, proposal_id: str, timeout_ms: int = 1000) -> Dict[str, Any]:
    """Collect entropy-weighted votes for proposal.

    Votes are weighted by inverse entropy - low entropy nodes have more weight.

    Args:
        coord: GradientCoordinator instance
        proposal_id: Proposal identifier
        timeout_ms: Timeout in milliseconds

    Returns:
        Vote collection result

    Receipt: vote_receipt
    """
    proposal = coord.proposals.get(proposal_id)
    if not proposal:
        return {"error": "proposal_not_found", "proposal_id": proposal_id}

    start_time = time.time()

    # Simulate vote collection
    total_weight = 0.0
    approve_weight = 0.0

    for node_id, node in coord.engine.nodes.items():
        # Weight: inverse of entropy (low entropy = high weight)
        weight = 1.0 / (1.0 + node.entropy)

        # Simulate vote based on gradient alignment
        # Nodes aligned with entropy flow tend to approve
        avg_gradient = sum(node.gradients.values()) / len(node.gradients) if node.gradients else 0
        vote = 1 if avg_gradient >= -GRADIENT_THRESHOLD else 0

        proposal.votes[node_id] = vote * weight
        total_weight += weight
        if vote:
            approve_weight += weight

    elapsed_ms = (time.time() - start_time) * 1000
    coord.latency_ms = elapsed_ms

    approval_ratio = approve_weight / total_weight if total_weight > 0 else 0

    result = {
        "proposal_id": proposal_id,
        "votes_collected": len(proposal.votes),
        "total_weight": round(total_weight, 4),
        "approve_weight": round(approve_weight, 4),
        "approval_ratio": round(approval_ratio, 4),
        "elapsed_ms": round(elapsed_ms, 2),
    }

    emit_receipt(
        "vote",
        {
            "receipt_type": "vote",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "coordinator_id": coord.coordinator_id,
            "proposal_id": proposal_id,
            "approval_ratio": round(approval_ratio, 4),
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def achieve_consensus(coord: GradientCoordinator, proposal_id: str) -> bool:
    """Achieve consensus via gradient convergence.

    Consensus requires >66% weighted approval.

    Args:
        coord: GradientCoordinator instance
        proposal_id: Proposal identifier

    Returns:
        True if consensus achieved

    Receipt: consensus_receipt
    """
    proposal = coord.proposals.get(proposal_id)
    if not proposal:
        return False

    # Collect votes if not already done
    if not proposal.votes:
        collect_votes(coord, proposal_id)

    # Calculate approval
    total_weight = sum(1.0 / (1.0 + coord.engine.nodes[n].entropy) for n in coord.engine.nodes)
    approve_weight = sum(proposal.votes.values())
    approval_ratio = approve_weight / total_weight if total_weight > 0 else 0

    # Consensus threshold: 66%
    consensus_achieved = approval_ratio >= 0.66

    proposal.status = "approved" if consensus_achieved else "rejected"

    emit_receipt(
        "consensus",
        {
            "receipt_type": "consensus",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "coordinator_id": coord.coordinator_id,
            "proposal_id": proposal_id,
            "approval_ratio": round(approval_ratio, 4),
            "consensus_achieved": consensus_achieved,
            "status": proposal.status,
            "payload_hash": dual_hash(
                json.dumps({"proposal_id": proposal_id, "consensus_achieved": consensus_achieved}, sort_keys=True)
            ),
        },
    )

    return consensus_achieved


def execute_coordinated(coord: GradientCoordinator, action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute action if consensus achieved.

    Args:
        coord: GradientCoordinator instance
        action: Action to execute

    Returns:
        Execution result
    """
    # Propose and vote
    proposal = propose_action(coord, action)
    proposal_id = proposal["proposal_id"]

    votes = collect_votes(coord, proposal_id)
    consensus = achieve_consensus(coord, proposal_id)

    if consensus:
        status = "executed"
        result = {"action": action, "success": True}
    else:
        status = "rejected"
        result = {"action": action, "success": False, "reason": "consensus_not_achieved"}

    return {
        "proposal_id": proposal_id,
        "status": status,
        "approval_ratio": votes["approval_ratio"],
        "result": result,
    }


def detect_partition(coord: GradientCoordinator) -> List[List[str]]:
    """Detect network partitions via gradient discontinuity.

    Partitions occur when gradient flow is interrupted.
    Large gradient jumps indicate partition boundaries.

    Args:
        coord: GradientCoordinator instance

    Returns:
        List of partitions (each partition is list of node IDs)

    Receipt: partition_receipt
    """
    partitions = []
    visited = set()

    def dfs(node_id: str, partition: List[str]):
        if node_id in visited:
            return
        visited.add(node_id)
        partition.append(node_id)

        node = coord.engine.nodes.get(node_id)
        if not node:
            return

        for neighbor_id in node.neighbors:
            gradient = node.gradients.get(neighbor_id, 0)
            # Large gradient = potential partition boundary
            if abs(gradient) < 1.0 and neighbor_id not in visited:
                dfs(neighbor_id, partition)

    for node_id in coord.engine.nodes:
        if node_id not in visited:
            partition = []
            dfs(node_id, partition)
            if partition:
                partitions.append(partition)

    coord.partitions = partitions

    emit_receipt(
        "partition",
        {
            "receipt_type": "partition",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "coordinator_id": coord.coordinator_id,
            "partition_count": len(partitions),
            "partition_sizes": [len(p) for p in partitions],
            "is_partitioned": len(partitions) > 1,
            "payload_hash": dual_hash(json.dumps({"partition_count": len(partitions)}, sort_keys=True)),
        },
    )

    return partitions


def heal_partition(coord: GradientCoordinator, partitions: List[List[str]]) -> Dict[str, Any]:
    """Heal partition via gradient bridge.

    Creates gradient bridges between partitions.

    Args:
        coord: GradientCoordinator instance
        partitions: List of partitions to heal

    Returns:
        Healing result

    Receipt: heal_receipt
    """
    if len(partitions) <= 1:
        return {"status": "no_partition", "partitions": len(partitions)}

    bridges_created = 0

    # Connect partitions by creating gradient bridges
    for i in range(len(partitions) - 1):
        # Pick a node from each partition
        node_a_id = partitions[i][0]
        node_b_id = partitions[i + 1][0]

        node_a = coord.engine.nodes.get(node_a_id)
        node_b = coord.engine.nodes.get(node_b_id)

        if node_a and node_b:
            # Average entropies to create bridge
            avg_entropy = (node_a.entropy + node_b.entropy) / 2
            node_a.gradients[node_b_id] = node_a.entropy - avg_entropy
            node_b.gradients[node_a_id] = node_b.entropy - avg_entropy
            bridges_created += 1

    result = {
        "status": "healed" if bridges_created > 0 else "failed",
        "partitions_before": len(partitions),
        "bridges_created": bridges_created,
    }

    emit_receipt(
        "heal",
        {
            "receipt_type": "heal",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "coordinator_id": coord.coordinator_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def measure_coordination_latency(coord: GradientCoordinator) -> float:
    """Measure time to achieve coordination.

    Args:
        coord: GradientCoordinator instance

    Returns:
        Latency in milliseconds
    """
    return coord.latency_ms


def get_coordinator_status() -> Dict[str, Any]:
    """Get current coordinator status.

    Returns:
        Coordinator status dict
    """
    return {
        "module": "swarm.gradient_coordinator",
        "version": "19.0.0",
        "coordination_mode": "entropy_gradient",
        "consensus_threshold": 0.66,
        "central_coordinator": False,
    }
