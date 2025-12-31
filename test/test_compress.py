"""Tests for spaceproof.compress module."""

from spaceproof.compress import (
    compress,
    decompress,
    calculate_recall,
    validate_compression_slo,
    MIN_COMPRESSION_RATIO,
    MIN_RECALL,
)


def test_compress_returns_tuple():
    """compress returns (compressed_bytes, stats_dict)."""
    data = b"test data " * 100
    compressed, stats = compress(data)

    assert isinstance(compressed, bytes)
    assert isinstance(stats, dict)
    assert "compression_ratio" in stats
    assert "recall" in stats


def test_decompress_roundtrip():
    """decompress reverses compress."""
    original = b"hello world " * 50
    compressed, _ = compress(original)
    recovered = decompress(compressed)

    assert recovered == original


def test_calculate_recall_perfect():
    """calculate_recall returns 1.0 for identical data."""
    data = b"test"
    assert calculate_recall(data, data) == 1.0


def test_calculate_recall_empty():
    """calculate_recall handles empty data."""
    assert calculate_recall(b"", b"") == 1.0


def test_calculate_recall_partial():
    """calculate_recall handles partial matches."""
    recall = calculate_recall(b"abcd", b"abcx")
    assert 0 < recall < 1


def test_validate_compression_slo_pass():
    """validate_compression_slo passes when SLOs met."""
    stats = {
        "compression_ratio": MIN_COMPRESSION_RATIO + 1,
        "recall": MIN_RECALL + 0.001,
    }
    assert validate_compression_slo(stats) is True


def test_validate_compression_slo_fail():
    """validate_compression_slo fails when SLOs not met."""
    stats = {
        "compression_ratio": MIN_COMPRESSION_RATIO - 1,
        "recall": MIN_RECALL - 0.01,
    }
    assert validate_compression_slo(stats) is False
