"""Offline module - Light-delay tolerant synchronization.

Mars-scale synchronization with Byzantine-resilient conflict resolution.
"""

from .sync import (
    sync_ledger,
    queue_for_sync,
    get_sync_status,
    emit_sync_receipt,
    sync_offline_receipts,
    calculate_sync_delay,
    SyncResult,
    SyncQueue,
)

from .conflict_resolution import (
    resolve_conflicts,
    resolve_conflict,
    detect_conflicts,
    merge_receipts,
    emit_conflict_receipt,
    ConflictResult,
    MergeStrategy,
)

from .offline_ledger import (
    create_offline_ledger,
    append_offline,
    get_offline_entries,
    prepare_for_sync,
    merge_offline_ledger,
    emit_offline_ledger_receipt,
    OfflineLedger,
    OfflineEntry,
)

__all__ = [
    # Sync
    "sync_ledger",
    "queue_for_sync",
    "get_sync_status",
    "emit_sync_receipt",
    "sync_offline_receipts",
    "calculate_sync_delay",
    "SyncResult",
    "SyncQueue",
    # Conflict resolution
    "resolve_conflicts",
    "resolve_conflict",
    "detect_conflicts",
    "merge_receipts",
    "emit_conflict_receipt",
    "ConflictResult",
    "MergeStrategy",
    # Offline ledger
    "create_offline_ledger",
    "append_offline",
    "get_offline_entries",
    "prepare_for_sync",
    "merge_offline_ledger",
    "emit_offline_ledger_receipt",
    "OfflineLedger",
    "OfflineEntry",
]
