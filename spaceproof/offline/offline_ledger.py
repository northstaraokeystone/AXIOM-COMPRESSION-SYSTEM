"""offline_ledger.py - Local ledger with eventual consistency.

Maintain local ledger during network partitions with sync preparation.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import dual_hash, emit_receipt, merkle

# === CONSTANTS ===

OFFLINE_TENANT = "spaceproof-offline"


@dataclass
class OfflineEntry:
    """Entry in offline ledger."""

    entry_id: str
    receipt_type: str
    data: Dict[str, Any]
    local_hash: str
    sequence_num: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    synced: bool = False
    synced_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "receipt_type": self.receipt_type,
            "data": self.data,
            "local_hash": self.local_hash,
            "sequence_num": self.sequence_num,
            "created_at": self.created_at,
            "synced": self.synced,
            "synced_at": self.synced_at,
        }


@dataclass
class OfflineLedger:
    """Local offline ledger."""

    ledger_id: str
    node_id: str
    entries: List[OfflineEntry]
    sequence_counter: int
    merkle_root: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_modified: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ledger_id": self.ledger_id,
            "node_id": self.node_id,
            "entry_count": len(self.entries),
            "sequence_counter": self.sequence_counter,
            "merkle_root": self.merkle_root,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "unsynced_count": sum(1 for e in self.entries if not e.synced),
        }


# Storage for offline ledgers
_offline_ledgers: Dict[str, OfflineLedger] = {}


def create_offline_ledger(node_id: str) -> OfflineLedger:
    """Create a new offline ledger.

    Args:
        node_id: Node identifier

    Returns:
        New OfflineLedger
    """
    ledger = OfflineLedger(
        ledger_id=str(uuid.uuid4()),
        node_id=node_id,
        entries=[],
        sequence_counter=0,
        merkle_root="",
    )

    _offline_ledgers[node_id] = ledger

    # Emit creation receipt
    emit_receipt(
        "offline_ledger_created",
        {
            "tenant_id": OFFLINE_TENANT,
            "ledger_id": ledger.ledger_id,
            "node_id": node_id,
        },
    )

    return ledger


def get_offline_ledger(node_id: str) -> Optional[OfflineLedger]:
    """Get offline ledger for node.

    Args:
        node_id: Node identifier

    Returns:
        OfflineLedger or None
    """
    return _offline_ledgers.get(node_id)


def append_offline(
    node_id: str,
    receipt_type: str,
    data: Dict[str, Any],
) -> OfflineEntry:
    """Append entry to offline ledger.

    Args:
        node_id: Node identifier
        receipt_type: Type of receipt
        data: Receipt data

    Returns:
        Created OfflineEntry
    """
    # Get or create ledger
    ledger = _offline_ledgers.get(node_id)
    if not ledger:
        ledger = create_offline_ledger(node_id)

    # Create entry
    entry_id = str(uuid.uuid4())
    ledger.sequence_counter += 1

    entry_data = {
        "entry_id": entry_id,
        "receipt_type": receipt_type,
        "sequence_num": ledger.sequence_counter,
        **data,
    }

    entry = OfflineEntry(
        entry_id=entry_id,
        receipt_type=receipt_type,
        data=data,
        local_hash=dual_hash(str(entry_data)),
        sequence_num=ledger.sequence_counter,
    )

    ledger.entries.append(entry)
    ledger.last_modified = datetime.utcnow().isoformat() + "Z"

    # Update Merkle root
    ledger.merkle_root = merkle([e.to_dict() for e in ledger.entries])

    return entry


def get_offline_entries(
    node_id: str,
    synced_only: bool = False,
    unsynced_only: bool = False,
    limit: int = 1000,
) -> List[OfflineEntry]:
    """Get entries from offline ledger.

    Args:
        node_id: Node identifier
        synced_only: Only return synced entries
        unsynced_only: Only return unsynced entries
        limit: Maximum entries to return

    Returns:
        List of OfflineEntry objects
    """
    ledger = _offline_ledgers.get(node_id)
    if not ledger:
        return []

    entries = ledger.entries

    if synced_only:
        entries = [e for e in entries if e.synced]
    elif unsynced_only:
        entries = [e for e in entries if not e.synced]

    return entries[:limit]


def mark_entries_synced(
    node_id: str,
    entry_ids: List[str],
) -> int:
    """Mark entries as synced.

    Args:
        node_id: Node identifier
        entry_ids: Entry IDs to mark

    Returns:
        Number of entries marked
    """
    ledger = _offline_ledgers.get(node_id)
    if not ledger:
        return 0

    count = 0
    synced_at = datetime.utcnow().isoformat() + "Z"

    for entry in ledger.entries:
        if entry.entry_id in entry_ids and not entry.synced:
            entry.synced = True
            entry.synced_at = synced_at
            count += 1

    return count


def prepare_for_sync(node_id: str) -> Dict[str, Any]:
    """Prepare ledger data for synchronization.

    Args:
        node_id: Node identifier

    Returns:
        Dict with sync-ready data
    """
    ledger = _offline_ledgers.get(node_id)
    if not ledger:
        return {
            "node_id": node_id,
            "entries": [],
            "merkle_root": "",
            "entry_count": 0,
        }

    unsynced = [e for e in ledger.entries if not e.synced]

    return {
        "node_id": node_id,
        "ledger_id": ledger.ledger_id,
        "entries": [e.to_dict() for e in unsynced],
        "merkle_root": ledger.merkle_root,
        "entry_count": len(unsynced),
        "sequence_counter": ledger.sequence_counter,
    }


def verify_ledger_integrity(node_id: str) -> bool:
    """Verify offline ledger Merkle integrity.

    Args:
        node_id: Node identifier

    Returns:
        True if ledger is valid
    """
    ledger = _offline_ledgers.get(node_id)
    if not ledger:
        return True  # Empty is valid

    expected_root = merkle([e.to_dict() for e in ledger.entries])
    return expected_root == ledger.merkle_root


def clear_offline_ledgers() -> None:
    """Clear all offline ledgers (for testing)."""
    global _offline_ledgers
    _offline_ledgers = {}


def merge_offline_ledger(
    source_node_id: str,
    target_node_id: str,
) -> Dict[str, Any]:
    """Merge source ledger into target ledger.

    Args:
        source_node_id: Source node ID
        target_node_id: Target node ID

    Returns:
        Merge result dict
    """
    source = _offline_ledgers.get(source_node_id)
    target = _offline_ledgers.get(target_node_id)

    if not source:
        return {
            "merged": False,
            "reason": "source ledger not found",
            "entries_merged": 0,
        }

    if not target:
        target = create_offline_ledger(target_node_id)

    # Get unsynced entries from source
    unsynced = [e for e in source.entries if not e.synced]

    # Merge into target
    entries_merged = 0
    for entry in unsynced:
        # Check if already exists in target
        exists = any(e.entry_id == entry.entry_id for e in target.entries)
        if not exists:
            target.entries.append(entry)
            target.sequence_counter += 1
            entries_merged += 1
            entry.synced = True
            entry.synced_at = datetime.utcnow().isoformat() + "Z"

    # Update target Merkle root
    target.merkle_root = merkle([e.to_dict() for e in target.entries])
    target.last_modified = datetime.utcnow().isoformat() + "Z"

    return {
        "merged": True,
        "source_node": source_node_id,
        "target_node": target_node_id,
        "entries_merged": entries_merged,
        "target_merkle_root": target.merkle_root,
    }


def emit_offline_ledger_receipt(ledger: OfflineLedger) -> Dict[str, Any]:
    """Emit receipt for offline ledger state.

    Args:
        ledger: OfflineLedger to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "offline_ledger",
        {
            "tenant_id": OFFLINE_TENANT,
            **ledger.to_dict(),
        },
    )
