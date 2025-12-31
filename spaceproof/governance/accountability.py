"""accountability.py - Ownership chain tracking.

Track complete ownership chains for decisions and interventions.
Provides audit trail showing who was responsible for each action.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt, merkle

# === CONSTANTS ===

GOVERNANCE_TENANT = "spaceproof-governance"


@dataclass
class OwnershipRecord:
    """Single ownership record in chain."""

    owner_id: str
    owner_role: str
    action: str
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "owner_id": self.owner_id,
            "owner_role": self.owner_role,
            "action": self.action,
            "timestamp": self.timestamp,
            "context": self.context,
        }


@dataclass
class OwnershipChain:
    """Complete ownership chain for a decision."""

    chain_id: str
    decision_id: str
    records: List[OwnershipRecord]
    merkle_root: str
    created_at: str
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "decision_id": self.decision_id,
            "records": [r.to_dict() for r in self.records],
            "merkle_root": self.merkle_root,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "chain_length": len(self.records),
        }

    @property
    def current_owner(self) -> Optional[OwnershipRecord]:
        """Get current (latest) owner."""
        return self.records[-1] if self.records else None


# In-memory storage for ownership chains (would be ledger in production)
_ownership_chains: Dict[str, OwnershipChain] = {}


def assign_ownership(
    decision_id: str,
    owner_id: str,
    owner_role: str,
    action: str = "assigned",
    context: Optional[Dict[str, Any]] = None,
) -> OwnershipChain:
    """Assign ownership for a decision.

    Args:
        decision_id: Decision identifier
        owner_id: Owner identifier
        owner_role: Owner's role
        action: Action type (assigned, transferred, escalated, etc.)
        context: Optional context

    Returns:
        Updated OwnershipChain
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    record = OwnershipRecord(
        owner_id=owner_id,
        owner_role=owner_role,
        action=action,
        timestamp=timestamp,
        context=context or {},
    )

    # Check if chain exists
    if decision_id in _ownership_chains:
        chain = _ownership_chains[decision_id]
        chain.records.append(record)
        chain.last_updated = timestamp
        # Update merkle root
        chain.merkle_root = merkle([r.to_dict() for r in chain.records])
    else:
        # Create new chain
        chain_id = str(uuid.uuid4())
        records = [record]
        chain = OwnershipChain(
            chain_id=chain_id,
            decision_id=decision_id,
            records=records,
            merkle_root=merkle([record.to_dict()]),
            created_at=timestamp,
            last_updated=timestamp,
        )
        _ownership_chains[decision_id] = chain

    return chain


def track_decision_chain(decision_id: str) -> Optional[OwnershipChain]:
    """Get ownership chain for a decision.

    Args:
        decision_id: Decision identifier

    Returns:
        OwnershipChain or None if not found
    """
    return _ownership_chains.get(decision_id)


def transfer_ownership(
    decision_id: str,
    from_owner: str,
    to_owner: str,
    to_role: str,
    reason: str,
) -> OwnershipChain:
    """Transfer ownership from one party to another.

    Args:
        decision_id: Decision identifier
        from_owner: Current owner ID
        to_owner: New owner ID
        to_role: New owner's role
        reason: Reason for transfer

    Returns:
        Updated OwnershipChain
    """
    return assign_ownership(
        decision_id=decision_id,
        owner_id=to_owner,
        owner_role=to_role,
        action="transferred",
        context={"from_owner": from_owner, "reason": reason},
    )


def emit_accountability_receipt(
    chain: OwnershipChain,
    event_type: str = "ownership_update",
) -> Dict[str, Any]:
    """Emit accountability receipt for ownership chain.

    Args:
        chain: OwnershipChain to emit
        event_type: Type of accountability event

    Returns:
        Receipt dict with dual-hash
    """
    receipt_data = {
        "tenant_id": GOVERNANCE_TENANT,
        "event_type": event_type,
        **chain.to_dict(),
    }

    return emit_receipt("accountability", receipt_data)


def verify_chain_integrity(chain: OwnershipChain) -> bool:
    """Verify ownership chain Merkle integrity.

    Args:
        chain: OwnershipChain to verify

    Returns:
        True if chain is valid
    """
    expected_root = merkle([r.to_dict() for r in chain.records])
    return expected_root == chain.merkle_root


def get_all_chains() -> Dict[str, OwnershipChain]:
    """Get all ownership chains (for testing/reporting).

    Returns:
        Dict of decision_id -> OwnershipChain
    """
    return _ownership_chains.copy()


def clear_chains() -> None:
    """Clear all ownership chains (for testing)."""
    global _ownership_chains
    _ownership_chains = {}
