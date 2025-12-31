"""sync.py - Light-delay tolerant ledger synchronization.

Handle Mars-scale communication delays (3-22 minutes) with eventual consistency.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt, MARS_LIGHT_DELAY_MIN_SEC

# === CONSTANTS ===

OFFLINE_TENANT = "spaceproof-offline"


@dataclass
class SyncResult:
    """Result of synchronization operation."""

    sync_id: str
    source_node: str
    target_node: str
    entries_synced: int
    conflicts_detected: int
    conflicts_resolved: int
    sync_latency_ms: float
    light_delay_sec: float
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sync_id": self.sync_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "entries_synced": self.entries_synced,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "sync_latency_ms": self.sync_latency_ms,
            "light_delay_sec": self.light_delay_sec,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class SyncQueue:
    """Queue of entries pending synchronization."""

    queue_id: str
    node_id: str
    entries: List[Dict[str, Any]]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_sync_attempt: Optional[str] = None
    sync_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_id": self.queue_id,
            "node_id": self.node_id,
            "entry_count": len(self.entries),
            "created_at": self.created_at,
            "last_sync_attempt": self.last_sync_attempt,
            "sync_attempts": self.sync_attempts,
        }


# Node sync queues
_sync_queues: Dict[str, SyncQueue] = {}

# Sync history
_sync_history: List[SyncResult] = []


def queue_for_sync(
    node_id: str,
    entry: Dict[str, Any],
) -> SyncQueue:
    """Queue an entry for synchronization.

    Args:
        node_id: Node identifier
        entry: Entry to queue

    Returns:
        Updated SyncQueue
    """
    if node_id not in _sync_queues:
        _sync_queues[node_id] = SyncQueue(
            queue_id=str(uuid.uuid4()),
            node_id=node_id,
            entries=[],
        )

    queue = _sync_queues[node_id]
    queue.entries.append(entry)

    return queue


def get_sync_status(node_id: str) -> Dict[str, Any]:
    """Get synchronization status for a node.

    Args:
        node_id: Node identifier

    Returns:
        Status dict
    """
    queue = _sync_queues.get(node_id)

    if not queue:
        return {
            "node_id": node_id,
            "pending_entries": 0,
            "last_sync": None,
            "is_synced": True,
        }

    # Find last successful sync
    last_sync = None
    for result in reversed(_sync_history):
        if result.source_node == node_id or result.target_node == node_id:
            if result.success:
                last_sync = result.timestamp
                break

    return {
        "node_id": node_id,
        "pending_entries": len(queue.entries),
        "last_sync": last_sync,
        "sync_attempts": queue.sync_attempts,
        "is_synced": len(queue.entries) == 0,
    }


def sync_ledger(
    source_node: str,
    target_node: str,
    light_delay_sec: float = MARS_LIGHT_DELAY_MIN_SEC,
) -> SyncResult:
    """Synchronize ledger between nodes.

    Args:
        source_node: Source node ID
        target_node: Target node ID
        light_delay_sec: Light delay in seconds

    Returns:
        SyncResult
    """
    import time

    start_time = time.time()

    source_queue = _sync_queues.get(source_node)

    if not source_queue or not source_queue.entries:
        # Nothing to sync
        return SyncResult(
            sync_id=str(uuid.uuid4()),
            source_node=source_node,
            target_node=target_node,
            entries_synced=0,
            conflicts_detected=0,
            conflicts_resolved=0,
            sync_latency_ms=0,
            light_delay_sec=light_delay_sec,
            success=True,
        )

    entries_to_sync = source_queue.entries.copy()
    source_queue.entries = []
    source_queue.sync_attempts += 1
    source_queue.last_sync_attempt = datetime.utcnow().isoformat() + "Z"

    # Simulate sync (in production, would actually transfer to target)
    # For now, assume success with no conflicts
    sync_latency = (time.time() - start_time) * 1000

    result = SyncResult(
        sync_id=str(uuid.uuid4()),
        source_node=source_node,
        target_node=target_node,
        entries_synced=len(entries_to_sync),
        conflicts_detected=0,
        conflicts_resolved=0,
        sync_latency_ms=sync_latency,
        light_delay_sec=light_delay_sec,
        success=True,
    )

    _sync_history.append(result)

    return result


def emit_sync_receipt(result: SyncResult) -> Dict[str, Any]:
    """Emit synchronization receipt.

    Args:
        result: SyncResult to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "sync",
        {
            "tenant_id": OFFLINE_TENANT,
            **result.to_dict(),
            "within_latency_bound": result.sync_latency_ms < result.light_delay_sec * 2000,
        },
    )


def get_sync_history(
    node_id: Optional[str] = None,
    limit: int = 100,
) -> List[SyncResult]:
    """Get synchronization history.

    Args:
        node_id: Optional node filter
        limit: Maximum results

    Returns:
        List of SyncResult objects
    """
    results = _sync_history

    if node_id:
        results = [r for r in results if r.source_node == node_id or r.target_node == node_id]

    return results[-limit:]


def clear_sync_state() -> None:
    """Clear all sync state (for testing)."""
    global _sync_queues, _sync_history
    _sync_queues = {}
    _sync_history = []
