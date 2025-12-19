"""Cross-planet dispute arbitration.

Implements arbitration algorithms for resolving disputes between
federated planets with lag-compensated voting.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core import TENANT_ID, dual_hash, emit_receipt


@dataclass
class Dispute:
    """Represents a cross-planet dispute."""

    dispute_id: str
    dispute_type: str  # "resource", "priority", "protocol", "boundary"
    parties: List[str]
    subject: str
    description: str
    status: str = "pending"  # "pending", "voting", "resolved", "escalated"
    resolution: Optional[str] = None
    votes: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    resolved_at: Optional[str] = None


@dataclass
class ArbitrationState:
    """Global arbitration state."""

    disputes: Dict[str, Dispute] = field(default_factory=dict)
    dispute_counter: int = 0
    resolutions: int = 0
    escalations: int = 0


# Global arbitration state
_arbitration_state = ArbitrationState()


def create_dispute(
    dispute_type: str,
    parties: List[str],
    subject: str,
    description: str = "",
) -> Dict[str, Any]:
    """Create a new cross-planet dispute.

    Args:
        dispute_type: Type of dispute.
        parties: Planets involved in dispute.
        subject: Subject of dispute.
        description: Detailed description.

    Returns:
        dict: Created dispute.
    """
    global _arbitration_state

    _arbitration_state.dispute_counter += 1
    dispute_id = f"dispute_{_arbitration_state.dispute_counter:06d}"

    dispute = Dispute(
        dispute_id=dispute_id,
        dispute_type=dispute_type,
        parties=parties,
        subject=subject,
        description=description,
        status="pending",
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    _arbitration_state.disputes[dispute_id] = dispute

    result = {
        "created": True,
        "dispute_id": dispute_id,
        "dispute_type": dispute_type,
        "parties": parties,
        "subject": subject,
    }

    emit_receipt(
        "arbitration_dispute_receipt",
        {
            "receipt_type": "arbitration_dispute_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "create",
            "dispute_id": dispute_id,
            "dispute_type": dispute_type,
            "parties": parties,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )

    return result


def submit_vote(
    dispute_id: str,
    voter: str,
    vote: str,
    reasoning: str = "",
) -> Dict[str, Any]:
    """Submit arbitration vote.

    Args:
        dispute_id: Dispute to vote on.
        voter: Voting planet.
        vote: Vote decision (one of the parties).
        reasoning: Reasoning for vote.

    Returns:
        dict: Vote submission result.
    """
    global _arbitration_state

    if dispute_id not in _arbitration_state.disputes:
        return {"error": f"Dispute {dispute_id} not found"}

    dispute = _arbitration_state.disputes[dispute_id]

    if voter in dispute.parties:
        return {"error": f"Party {voter} cannot vote on their own dispute"}

    if vote not in dispute.parties:
        return {"error": f"Vote must be for one of the parties: {dispute.parties}"}

    # Record vote
    vote_record = {
        "voter": voter,
        "vote": vote,
        "reasoning": reasoning,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    dispute.votes.append(vote_record)
    dispute.status = "voting"

    return {
        "submitted": True,
        "dispute_id": dispute_id,
        "voter": voter,
        "vote": vote,
        "total_votes": len(dispute.votes),
    }


def resolve_dispute(dispute_id: str, min_votes: int = 2) -> Dict[str, Any]:
    """Resolve dispute based on votes.

    Args:
        dispute_id: Dispute to resolve.
        min_votes: Minimum votes required.

    Returns:
        dict: Resolution result.
    """
    global _arbitration_state

    if dispute_id not in _arbitration_state.disputes:
        return {"error": f"Dispute {dispute_id} not found"}

    dispute = _arbitration_state.disputes[dispute_id]

    if len(dispute.votes) < min_votes:
        return {
            "resolved": False,
            "reason": "insufficient_votes",
            "votes": len(dispute.votes),
            "required": min_votes,
        }

    # Count votes
    vote_counts = {}
    for v in dispute.votes:
        vote_counts[v["vote"]] = vote_counts.get(v["vote"], 0) + 1

    # Determine winner by majority
    if not vote_counts:
        dispute.status = "escalated"
        _arbitration_state.escalations += 1
        return {"resolved": False, "reason": "no_votes"}

    winner = max(vote_counts, key=vote_counts.get)

    # Check for tie
    max_count = vote_counts[winner]
    tied = [p for p, c in vote_counts.items() if c == max_count]

    if len(tied) > 1:
        # Random tiebreaker
        winner = random.choice(tied)

    dispute.resolution = winner
    dispute.status = "resolved"
    dispute.resolved_at = datetime.utcnow().isoformat() + "Z"
    _arbitration_state.resolutions += 1

    result = {
        "resolved": True,
        "dispute_id": dispute_id,
        "winner": winner,
        "vote_counts": vote_counts,
        "total_votes": len(dispute.votes),
    }

    emit_receipt(
        "arbitration_resolution_receipt",
        {
            "receipt_type": "arbitration_resolution_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "resolved": True,
            "dispute_id": dispute_id,
            "winner": winner,
            "total_votes": len(dispute.votes),
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )

    return result


def escalate_dispute(dispute_id: str, reason: str = "") -> Dict[str, Any]:
    """Escalate dispute for higher-level resolution.

    Args:
        dispute_id: Dispute to escalate.
        reason: Reason for escalation.

    Returns:
        dict: Escalation result.
    """
    global _arbitration_state

    if dispute_id not in _arbitration_state.disputes:
        return {"error": f"Dispute {dispute_id} not found"}

    dispute = _arbitration_state.disputes[dispute_id]
    dispute.status = "escalated"
    _arbitration_state.escalations += 1

    return {
        "escalated": True,
        "dispute_id": dispute_id,
        "reason": reason,
        "total_escalations": _arbitration_state.escalations,
    }


def get_dispute_status(dispute_id: str) -> Dict[str, Any]:
    """Get status of a dispute.

    Args:
        dispute_id: Dispute ID.

    Returns:
        dict: Dispute status.
    """
    if dispute_id not in _arbitration_state.disputes:
        return {"error": f"Dispute {dispute_id} not found"}

    dispute = _arbitration_state.disputes[dispute_id]

    return {
        "dispute_id": dispute_id,
        "dispute_type": dispute.dispute_type,
        "parties": dispute.parties,
        "subject": dispute.subject,
        "status": dispute.status,
        "resolution": dispute.resolution,
        "votes": len(dispute.votes),
        "created_at": dispute.created_at,
        "resolved_at": dispute.resolved_at,
    }


def get_arbitration_stats() -> Dict[str, Any]:
    """Get overall arbitration statistics.

    Returns:
        dict: Arbitration statistics.
    """
    global _arbitration_state

    disputes = _arbitration_state.disputes

    status_counts = {
        "pending": 0,
        "voting": 0,
        "resolved": 0,
        "escalated": 0,
    }

    for d in disputes.values():
        if d.status in status_counts:
            status_counts[d.status] += 1

    return {
        "total_disputes": len(disputes),
        "resolutions": _arbitration_state.resolutions,
        "escalations": _arbitration_state.escalations,
        "status_counts": status_counts,
        "resolution_rate": _arbitration_state.resolutions / max(1, len(disputes)),
    }
