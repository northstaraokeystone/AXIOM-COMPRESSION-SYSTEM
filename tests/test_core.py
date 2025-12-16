"""Tests for AXIOM core.py - SLO validation for core functions.

Purpose: Verify CLAUDEME compliance of foundation module.
"""
import json
import sys
from io import StringIO

import pytest

from src.core import (
    dual_hash,
    emit_receipt,
    merkle,
    StopRule,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    TENANT_ID,
    RECEIPT_SCHEMA,
    HAS_BLAKE3,
)


# === dual_hash tests ===

def test_dual_hash_format():
    """Result contains ':' separator, both halves are 64-char hex.

    SLO: Format compliance
    """
    result = dual_hash(b"test")
    parts = result.split(":")
    assert len(parts) == 2, "dual_hash should return two parts separated by ':'"
    assert len(parts[0]) == 64, "First hash should be 64 hex chars"
    assert len(parts[1]) == 64, "Second hash should be 64 hex chars"
    # Verify both are valid hex
    int(parts[0], 16)
    int(parts[1], 16)


def test_dual_hash_deterministic():
    """Same input → same output.

    SLO: Determinism
    """
    input_data = b"deterministic test"
    result1 = dual_hash(input_data)
    result2 = dual_hash(input_data)
    assert result1 == result2, "dual_hash must be deterministic"


def test_dual_hash_str_bytes_equiv():
    """dual_hash("x") == dual_hash(b"x").

    SLO: Type handling
    """
    assert dual_hash("x") == dual_hash(b"x"), "String and bytes should produce same hash"


def test_dual_hash_different_inputs():
    """Different inputs produce different outputs."""
    hash1 = dual_hash(b"input1")
    hash2 = dual_hash(b"input2")
    assert hash1 != hash2, "Different inputs should produce different hashes"


# === emit_receipt tests ===

def test_emit_receipt_has_required_fields():
    """Result has ts, tenant_id, payload_hash, receipt_type.

    SLO: Schema compliance
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    result = emit_receipt("test", {"key": "value"})

    # Restore stdout
    sys.stdout = old_stdout

    assert "ts" in result, "Receipt must have 'ts' field"
    assert "tenant_id" in result, "Receipt must have 'tenant_id' field"
    assert "payload_hash" in result, "Receipt must have 'payload_hash' field"
    assert "receipt_type" in result, "Receipt must have 'receipt_type' field"
    assert result["receipt_type"] == "test", "receipt_type should match input"


def test_emit_receipt_tenant_override():
    """Custom tenant_id in data → used in receipt.

    SLO: Override works
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    result = emit_receipt("test", {"tenant_id": "custom-tenant"})

    sys.stdout = old_stdout

    assert result["tenant_id"] == "custom-tenant", "Custom tenant_id should override default"


def test_emit_receipt_default_tenant():
    """No tenant_id in data → TENANT_ID constant used.

    SLO: Default works
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    result = emit_receipt("test", {"key": "value"})

    sys.stdout = old_stdout

    assert result["tenant_id"] == TENANT_ID, f"Should use TENANT_ID constant ({TENANT_ID})"


def test_emit_receipt_stdout_json():
    """emit_receipt prints valid JSON to stdout."""
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    emit_receipt("test", {"data": 123})

    sys.stdout = old_stdout
    output = captured.getvalue().strip()

    # Should be valid JSON
    parsed = json.loads(output)
    assert parsed["receipt_type"] == "test"
    assert parsed["data"] == 123


def test_emit_receipt_ts_format():
    """Timestamp should be ISO8601 with Z suffix."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    result = emit_receipt("test", {})

    sys.stdout = old_stdout

    ts = result["ts"]
    assert ts.endswith("Z"), "Timestamp should end with 'Z'"
    assert "T" in ts, "Timestamp should be ISO8601 format"


# === merkle tests ===

def test_merkle_empty():
    """merkle([]) returns dual_hash(b"empty").

    SLO: Edge case
    """
    assert merkle([]) == dual_hash(b"empty"), "Empty list should hash 'empty'"


def test_merkle_single():
    """merkle([x]) returns hash of x.

    SLO: Single item
    """
    item = {"key": "value"}
    expected = dual_hash(json.dumps(item, sort_keys=True))
    result = merkle([item])
    assert result == expected, "Single item merkle should be hash of that item"


def test_merkle_deterministic():
    """Same items → same root.

    SLO: Determinism
    """
    items = [{"a": 1}, {"b": 2}, {"c": 3}]
    result1 = merkle(items)
    result2 = merkle(items)
    assert result1 == result2, "merkle must be deterministic"


def test_merkle_order_matters():
    """[a,b] ≠ [b,a].

    SLO: Order sensitivity
    """
    a = {"a": 1}
    b = {"b": 2}
    assert merkle([a, b]) != merkle([b, a]), "Order should affect merkle root"


def test_merkle_multiple_items():
    """merkle handles multiple items correctly."""
    items = [{"i": i} for i in range(10)]
    result = merkle(items)
    parts = result.split(":")
    assert len(parts) == 2, "merkle should return dual_hash format"
    assert len(parts[0]) == 64, "First hash should be 64 hex chars"


def test_merkle_odd_count():
    """merkle handles odd item count (duplicates last)."""
    items = [{"i": i} for i in range(3)]  # 3 items = odd
    result = merkle(items)
    # Should still produce valid hash
    parts = result.split(":")
    assert len(parts) == 2


# === StopRule tests ===

def test_stoprule_is_exception():
    """StopRule should be an Exception subclass."""
    assert issubclass(StopRule, Exception)


def test_stoprule_message():
    """StopRule should preserve message."""
    msg = "Test error message"
    exc = StopRule(msg)
    assert str(exc) == msg


# === stoprule functions tests ===

def test_stoprule_hash_mismatch_emits():
    """Verify anomaly receipt emitted before raise.

    SLO: Receipt first
    """
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    with pytest.raises(StopRule):
        stoprule_hash_mismatch("expected_hash", "actual_hash")

    sys.stdout = old_stdout
    output = captured.getvalue().strip()

    # Should have emitted a receipt
    receipt = json.loads(output)
    assert receipt["receipt_type"] == "anomaly"
    assert receipt["metric"] == "hash_mismatch"
    assert receipt["classification"] == "violation"
    assert receipt["action"] == "halt"


def test_stoprule_invalid_receipt_emits():
    """Verify anomaly receipt emitted before raise.

    SLO: Receipt first
    """
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    with pytest.raises(StopRule):
        stoprule_invalid_receipt("test reason")

    sys.stdout = old_stdout
    output = captured.getvalue().strip()

    # Should have emitted a receipt
    receipt = json.loads(output)
    assert receipt["receipt_type"] == "anomaly"
    assert receipt["metric"] == "invalid_receipt"
    assert receipt["classification"] == "anti_pattern"
    assert receipt["action"] == "halt"


def test_stoprule_hash_mismatch_raises():
    """stoprule_hash_mismatch raises StopRule exception.

    SLO: Exception raised
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    with pytest.raises(StopRule) as exc_info:
        stoprule_hash_mismatch("a", "b")

    sys.stdout = old_stdout

    assert "Hash mismatch" in str(exc_info.value)
    assert "a" in str(exc_info.value)
    assert "b" in str(exc_info.value)


def test_stoprule_invalid_receipt_raises():
    """stoprule_invalid_receipt raises StopRule exception.

    SLO: Exception raised
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    with pytest.raises(StopRule) as exc_info:
        stoprule_invalid_receipt("missing field")

    sys.stdout = old_stdout

    assert "Invalid receipt" in str(exc_info.value)
    assert "missing field" in str(exc_info.value)


# === Constants tests ===

def test_tenant_id_constant():
    """TENANT_ID should be 'axiom-witness'."""
    assert TENANT_ID == "axiom-witness"


def test_receipt_schema_has_required_keys():
    """RECEIPT_SCHEMA should have required keys."""
    required_keys = ["receipt_type", "ts", "tenant_id", "payload_hash"]
    for key in required_keys:
        assert key in RECEIPT_SCHEMA, f"RECEIPT_SCHEMA missing key: {key}"


def test_has_blake3_is_bool():
    """HAS_BLAKE3 should be a boolean."""
    assert isinstance(HAS_BLAKE3, bool)
