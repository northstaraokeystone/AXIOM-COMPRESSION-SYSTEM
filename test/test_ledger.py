"""Tests for spaceproof.ledger module."""

import tempfile
import os
from spaceproof.ledger import (
    Ledger,
    create_ledger,
    append_to_ledger,
    verify_ledger,
    save_ledger,
    load_ledger,
)


def test_create_ledger():
    """create_ledger returns empty ledger."""
    ledger = create_ledger()
    assert len(ledger.entries) == 0


def test_append_to_ledger():
    """append_to_ledger adds entry."""
    ledger = create_ledger()
    receipt = {"type": "test", "data": "value"}

    entry = append_to_ledger(ledger, receipt)
    assert len(ledger.entries) == 1
    assert entry.index == 0
    assert entry.hash != ""


def test_ledger_merkle_updates():
    """Ledger merkle root updates on append."""
    ledger = Ledger()
    root1 = ledger.merkle_root

    ledger.append({"a": 1})
    root2 = ledger.merkle_root
    assert root2 != root1

    ledger.append({"b": 2})
    root3 = ledger.merkle_root
    assert root3 != root2


def test_verify_ledger_valid():
    """verify_ledger returns True for valid ledger."""
    ledger = create_ledger()
    append_to_ledger(ledger, {"x": 1})
    append_to_ledger(ledger, {"y": 2})

    result = verify_ledger(ledger)
    assert result["valid"] is True
    assert result["entry_count"] == 2


def test_verify_ledger_tampered():
    """verify_ledger detects tampering."""
    ledger = create_ledger()
    append_to_ledger(ledger, {"original": True})

    # Tamper with entry
    ledger.entries[0].receipt["original"] = False

    result = verify_ledger(ledger)
    assert result["valid"] is False


def test_save_and_load_ledger():
    """Ledger can be saved and loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")

        # Create and save
        ledger = create_ledger()
        append_to_ledger(ledger, {"id": 1})
        append_to_ledger(ledger, {"id": 2})
        save_ledger(ledger, path)

        # Load and verify
        loaded = load_ledger(path)
        assert len(loaded.entries) == 2
        assert loaded.entries[0].receipt["id"] == 1
