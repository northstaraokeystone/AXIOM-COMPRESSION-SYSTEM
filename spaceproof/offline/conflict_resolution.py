"""conflict_resolution.py - Byzantine-resilient merge for offline ledgers.

Deterministic conflict resolution for distributed ledger consistency.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

from spaceproof.core import dual_hash, emit_receipt, merkle

# === CONSTANTS ===

OFFLINE_TENANT = "spaceproof-offline"


class MergeStrategy(Enum):
    """Merge strategy for conflict resolution."""

    TIMESTAMP = "timestamp"  # Newest wins
    HASH_ORDER = "hash_order"  # Deterministic by hash
    PRIORITY = "priority"  # Higher priority wins
    CONSENSUS = "consensus"  # Multi-node agreement


@dataclass
class Conflict:
    """Detected conflict between entries."""

    conflict_id: str
    entry_a: Dict[str, Any]
    entry_b: Dict[str, Any]
    conflict_type: str  # duplicate, divergent, ordering
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "entry_a_hash": dual_hash(str(self.entry_a)),
            "entry_b_hash": dual_hash(str(self.entry_b)),
            "conflict_type": self.conflict_type,
            "detected_at": self.detected_at,
        }


@dataclass
class ConflictResult:
    """Result of conflict resolution."""

    resolution_id: str
    conflicts_detected: int
    conflicts_resolved: int
    strategy_used: str
    merged_entries: List[Dict[str, Any]]
    rejected_entries: List[Dict[str, Any]]
    merkle_root: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resolution_id": self.resolution_id,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "strategy_used": self.strategy_used,
            "merged_count": len(self.merged_entries),
            "rejected_count": len(self.rejected_entries),
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
        }


def _extract_entries(ledgers_or_entries) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract entries from ledgers or entry lists.

    Args:
        ledgers_or_entries: List containing OfflineLedger objects or entry dicts

    Returns:
        Tuple of (entries_a, entries_b)
    """
    from spaceproof.offline.offline_ledger import OfflineLedger

    if len(ledgers_or_entries) < 2:
        return [], []

    item_a = ledgers_or_entries[0]
    item_b = ledgers_or_entries[1]

    if isinstance(item_a, OfflineLedger):
        entries_a = [e.to_dict() for e in item_a.entries]
    elif isinstance(item_a, list):
        entries_a = item_a
    else:
        entries_a = [item_a] if item_a else []

    if isinstance(item_b, OfflineLedger):
        entries_b = [e.to_dict() for e in item_b.entries]
    elif isinstance(item_b, list):
        entries_b = item_b
    else:
        entries_b = [item_b] if item_b else []

    return entries_a, entries_b


def detect_conflicts(
    entries_a_or_ledgers,
    entries_b: List[Dict[str, Any]] = None,
) -> List[Conflict]:
    """Detect conflicts between two entry sets.

    Args:
        entries_a_or_ledgers: First entry set, or list of two OfflineLedger objects
        entries_b: Second entry set (optional if ledgers list is passed)

    Returns:
        List of detected Conflict objects
    """
    from spaceproof.offline.offline_ledger import OfflineLedger

    # Handle list of ledgers format
    if isinstance(entries_a_or_ledgers, list) and len(entries_a_or_ledgers) > 0:
        if isinstance(entries_a_or_ledgers[0], OfflineLedger):
            entries_a, entries_b = _extract_entries(entries_a_or_ledgers)
        elif entries_b is None:
            # List of entry lists
            entries_a = entries_a_or_ledgers
            entries_b = []
        else:
            entries_a = entries_a_or_ledgers
    else:
        entries_a = entries_a_or_ledgers if entries_a_or_ledgers else []
        entries_b = entries_b if entries_b else []

    conflicts = []

    # Build hash map for quick lookup
    {dual_hash(str(e)): e for e in entries_a}
    {dual_hash(str(e)): e for e in entries_b}

    # Check for same-ID different-content conflicts
    id_map_a = {e.get("id", e.get("receipt_id")): e for e in entries_a if e.get("id") or e.get("receipt_id")}
    id_map_b = {e.get("id", e.get("receipt_id")): e for e in entries_b if e.get("id") or e.get("receipt_id")}

    for entry_id, entry_a in id_map_a.items():
        if entry_id in id_map_b:
            entry_b = id_map_b[entry_id]
            hash_a = dual_hash(str(entry_a))
            hash_b = dual_hash(str(entry_b))

            if hash_a != hash_b:
                conflicts.append(
                    Conflict(
                        conflict_id=str(uuid.uuid4()),
                        entry_a=entry_a,
                        entry_b=entry_b,
                        conflict_type="divergent",
                    )
                )

    return conflicts


def resolve_by_timestamp(
    entry_a: Dict[str, Any],
    entry_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve conflict by timestamp (newest wins).

    Args:
        entry_a: First entry
        entry_b: Second entry

    Returns:
        Winning entry
    """
    ts_a = entry_a.get("ts", entry_a.get("timestamp", ""))
    ts_b = entry_b.get("ts", entry_b.get("timestamp", ""))

    return entry_a if ts_a >= ts_b else entry_b


def resolve_by_hash(
    entry_a: Dict[str, Any],
    entry_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve conflict by hash order (deterministic).

    Args:
        entry_a: First entry
        entry_b: Second entry

    Returns:
        Winning entry
    """
    hash_a = dual_hash(str(entry_a))
    hash_b = dual_hash(str(entry_b))

    return entry_a if hash_a <= hash_b else entry_b


def resolve_by_priority(
    entry_a: Dict[str, Any],
    entry_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve conflict by priority (higher wins).

    Args:
        entry_a: First entry
        entry_b: Second entry

    Returns:
        Winning entry
    """
    priority_a = entry_a.get("priority", 0)
    priority_b = entry_b.get("priority", 0)

    return entry_a if priority_a >= priority_b else entry_b


def resolve_conflicts(
    entries_a_or_ledgers,
    entries_b_or_strategy = None,
    strategy: MergeStrategy = MergeStrategy.HASH_ORDER,
) -> ConflictResult:
    """Resolve conflicts between entry sets.

    Args:
        entries_a_or_ledgers: First entry set, or list of two OfflineLedger objects
        entries_b_or_strategy: Second entry set, or strategy if ledgers list is passed
        strategy: Resolution strategy

    Returns:
        ConflictResult with merged entries
    """
    from spaceproof.offline.offline_ledger import OfflineLedger

    # Handle list of ledgers format
    if isinstance(entries_a_or_ledgers, list) and len(entries_a_or_ledgers) > 0:
        if isinstance(entries_a_or_ledgers[0], OfflineLedger):
            entries_a, entries_b = _extract_entries(entries_a_or_ledgers)
            # entries_b_or_strategy is actually the strategy in this case
            if isinstance(entries_b_or_strategy, MergeStrategy):
                strategy = entries_b_or_strategy
        else:
            entries_a = entries_a_or_ledgers
            entries_b = entries_b_or_strategy if entries_b_or_strategy is not None else []
    else:
        entries_a = entries_a_or_ledgers if entries_a_or_ledgers else []
        entries_b = entries_b_or_strategy if entries_b_or_strategy is not None else []

    conflicts = detect_conflicts(entries_a, entries_b)
    resolved = []
    rejected = []

    # Process conflicts
    for conflict in conflicts:
        if strategy == MergeStrategy.TIMESTAMP:
            winner = resolve_by_timestamp(conflict.entry_a, conflict.entry_b)
        elif strategy == MergeStrategy.PRIORITY:
            winner = resolve_by_priority(conflict.entry_a, conflict.entry_b)
        else:  # HASH_ORDER (default)
            winner = resolve_by_hash(conflict.entry_a, conflict.entry_b)

        loser = conflict.entry_b if winner == conflict.entry_a else conflict.entry_a
        resolved.append(winner)
        rejected.append(loser)

    # Merge non-conflicting entries
    conflict_ids = set()
    for c in conflicts:
        conflict_ids.add(c.entry_a.get("id", c.entry_a.get("receipt_id")))
        conflict_ids.add(c.entry_b.get("id", c.entry_b.get("receipt_id")))

    merged = resolved.copy()

    for entry in entries_a:
        entry_id = entry.get("id", entry.get("receipt_id"))
        if entry_id not in conflict_ids:
            merged.append(entry)

    for entry in entries_b:
        entry_id = entry.get("id", entry.get("receipt_id"))
        if entry_id not in conflict_ids:
            # Check if already added from entries_a
            if not any(e.get("id", e.get("receipt_id")) == entry_id for e in merged):
                merged.append(entry)

    # Compute Merkle root
    merkle_root = merkle(merged) if merged else ""

    return ConflictResult(
        resolution_id=str(uuid.uuid4()),
        conflicts_detected=len(conflicts),
        conflicts_resolved=len(resolved),
        strategy_used=strategy.value,
        merged_entries=merged,
        rejected_entries=rejected,
        merkle_root=merkle_root,
    )


def merge_receipts(
    receipts_a_or_ledgers,
    receipts_b: List[Dict[str, Any]] = None,
    strategy: MergeStrategy = MergeStrategy.HASH_ORDER,
) -> List[Dict[str, Any]]:
    """Merge two receipt lists.

    Args:
        receipts_a_or_ledgers: First receipt list, or list of two OfflineLedger objects
        receipts_b: Second receipt list (optional if ledgers list is passed)
        strategy: Resolution strategy

    Returns:
        List of merged receipts
    """
    from spaceproof.offline.offline_ledger import OfflineLedger

    # Handle list of ledgers format
    if isinstance(receipts_a_or_ledgers, list) and len(receipts_a_or_ledgers) > 0:
        if isinstance(receipts_a_or_ledgers[0], OfflineLedger):
            receipts_a, receipts_b = _extract_entries(receipts_a_or_ledgers)
        else:
            receipts_a = receipts_a_or_ledgers
            receipts_b = receipts_b if receipts_b is not None else []
    else:
        receipts_a = receipts_a_or_ledgers if receipts_a_or_ledgers else []
        receipts_b = receipts_b if receipts_b is not None else []

    result = resolve_conflicts(receipts_a, receipts_b, strategy)
    return result.merged_entries


def emit_conflict_receipt(result: ConflictResult) -> Dict[str, Any]:
    """Emit conflict resolution receipt.

    Args:
        result: ConflictResult to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "conflict_resolution",
        {
            "tenant_id": OFFLINE_TENANT,
            **result.to_dict(),
        },
    )
