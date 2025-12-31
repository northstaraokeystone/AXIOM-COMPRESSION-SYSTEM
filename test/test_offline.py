"""Tests for spaceproof.offline module."""

from spaceproof.offline import (
    sync_ledger,
    queue_for_sync,
    get_sync_status,
    emit_sync_receipt,
    SyncResult,
    SyncQueue,
    resolve_conflicts,
    detect_conflicts,
    merge_receipts,
    ConflictResult,
    MergeStrategy,
    create_offline_ledger,
    append_offline,
    get_offline_entries,
    prepare_for_sync,
    OfflineLedger,
    OfflineEntry,
)


def test_create_offline_ledger():
    """create_offline_ledger creates local ledger."""
    ledger = create_offline_ledger(node_id="colony_1")
    assert isinstance(ledger, OfflineLedger)
    assert ledger.node_id == "colony_1"


def test_append_offline():
    """append_offline adds entry to offline ledger."""
    # Signature: append_offline(node_id, receipt_type, data)
    entry = append_offline("colony_append", "test", {"data": "value"})
    assert isinstance(entry, OfflineEntry)


def test_get_offline_entries():
    """get_offline_entries returns entries."""
    ledger = create_offline_ledger(node_id="colony_entries_2")
    append_offline("colony_entries_2", "test", {"data": "value"})
    # Signature: get_offline_entries(node_id, ...)
    entries = get_offline_entries(ledger.node_id)
    assert isinstance(entries, list)


def test_prepare_for_sync():
    """prepare_for_sync prepares ledger for sync."""
    ledger = create_offline_ledger(node_id="colony_prep_2")
    append_offline("colony_prep_2", "test", {"data": "value"})
    # Signature: prepare_for_sync(node_id)
    prepared = prepare_for_sync(ledger.node_id)
    assert prepared is not None


def test_sync_ledger():
    """sync_ledger syncs offline ledger."""
    ledger1 = create_offline_ledger(node_id="colony_sync_source")
    create_offline_ledger(node_id="colony_sync_target")
    append_offline("colony_sync_source", "test", {"data": "value"})
    # Signature: sync_ledger(source_node, target_node, light_delay_sec=...)
    result = sync_ledger(ledger1.node_id, "colony_sync_target")
    assert isinstance(result, SyncResult)


def test_queue_for_sync():
    """queue_for_sync adds to sync queue."""
    ledger = create_offline_ledger(node_id="colony_queue_2")
    entry_data = {"receipt_type": "test", "data": "value"}
    # Signature: queue_for_sync(node_id, entry)
    queue = queue_for_sync(ledger.node_id, entry_data)
    assert isinstance(queue, SyncQueue)


def test_get_sync_status():
    """get_sync_status returns status."""
    status = get_sync_status(node_id="colony_status")
    assert isinstance(status, dict)


def test_emit_sync_receipt():
    """emit_sync_receipt emits valid receipt."""
    ledger1 = create_offline_ledger(node_id="colony_emit_src")
    create_offline_ledger(node_id="colony_emit_tgt")
    append_offline("colony_emit_src", "test", {"data": "value"})
    result = sync_ledger(ledger1.node_id, "colony_emit_tgt")
    receipt = emit_sync_receipt(result)
    assert receipt["receipt_type"] == "sync"


def test_detect_conflicts():
    """detect_conflicts finds conflicting receipts."""
    create_offline_ledger(node_id="colony_a2")
    create_offline_ledger(node_id="colony_b2")
    append_offline("colony_a2", "test", {"receipt_id": "r1", "data": "v1"})
    append_offline("colony_b2", "test", {"receipt_id": "r1", "data": "v2"})
    entries_a = get_offline_entries("colony_a2")
    entries_b = get_offline_entries("colony_b2")
    # Signature: detect_conflicts(entries_a, entries_b)
    # Convert entries to dicts
    entries_a_dicts = [e.to_dict() if hasattr(e, "to_dict") else {"data": e.data} for e in entries_a]
    entries_b_dicts = [e.to_dict() if hasattr(e, "to_dict") else {"data": e.data} for e in entries_b]
    conflicts = detect_conflicts(entries_a_dicts, entries_b_dicts)
    assert isinstance(conflicts, list)


def test_resolve_conflicts():
    """resolve_conflicts resolves conflicting receipts."""
    create_offline_ledger(node_id="colony_c2")
    create_offline_ledger(node_id="colony_d2")
    append_offline("colony_c2", "test", {"receipt_id": "r2", "data": "v1"})
    append_offline("colony_d2", "test", {"receipt_id": "r2", "data": "v2"})
    entries_a = get_offline_entries("colony_c2")
    entries_b = get_offline_entries("colony_d2")
    # Convert entries to dicts
    entries_a_dicts = [e.to_dict() if hasattr(e, "to_dict") else {"data": e.data} for e in entries_a]
    entries_b_dicts = [e.to_dict() if hasattr(e, "to_dict") else {"data": e.data} for e in entries_b]
    # Signature: resolve_conflicts(entries_a, entries_b, strategy)
    result = resolve_conflicts(entries_a_dicts, entries_b_dicts, strategy=MergeStrategy.TIMESTAMP)
    assert isinstance(result, ConflictResult)


def test_merge_receipts():
    """merge_receipts combines receipts."""
    create_offline_ledger(node_id="colony_e2")
    create_offline_ledger(node_id="colony_f2")
    append_offline("colony_e2", "test", {"receipt_id": "r3", "data": "a"})
    append_offline("colony_f2", "test", {"receipt_id": "r4", "data": "b"})
    entries_a = get_offline_entries("colony_e2")
    entries_b = get_offline_entries("colony_f2")
    # Convert entries to dicts (receipts)
    receipts_a = [e.to_dict() if hasattr(e, "to_dict") else {"data": e.data} for e in entries_a]
    receipts_b = [e.to_dict() if hasattr(e, "to_dict") else {"data": e.data} for e in entries_b]
    # Signature: merge_receipts(receipts_a, receipts_b, strategy) -> Tuple[List, ConflictResult]
    merged, conflict_result = merge_receipts(receipts_a, receipts_b)
    assert isinstance(merged, list)


def test_merge_strategy_enum():
    """MergeStrategy enum has expected values."""
    assert MergeStrategy.TIMESTAMP is not None
    assert MergeStrategy.HASH_ORDER is not None
    assert MergeStrategy.PRIORITY is not None
