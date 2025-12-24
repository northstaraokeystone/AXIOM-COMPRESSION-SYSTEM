"""compress.py - Telemetry Compression Engine

D20 Production Evolution: Stakeholder-intuitive name for QED compression.

THE COMPRESSION INSIGHT:
    Compression = Discovery.
    High compression ratio reveals underlying structure.
    The pattern that compresses is the pattern that matters.

Source: AXIOM D20 Production Evolution

SLOs:
    - Compression ratio: >= 10x minimum
    - Recall: >= 0.999 (three nines)

Stakeholder Value:
    - Elon/xAI: 10x bandwidth savings for FSD telemetry
    - DOT: FMCSA ELD compliance with reduced storage
    - Defense: Fire-control telemetry at 50x with 0.9999 recall
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import json
import zlib
import struct

from .core import emit_receipt, dual_hash

# === CONSTANTS ===

TENANT_ID = "axiom-compress"

# Compression SLOs
MIN_COMPRESSION_RATIO = 10  # 10x minimum
MIN_RECALL = 0.999  # Three nines

# Algorithm options
ALGO_ZLIB = "zlib"
ALGO_PATTERN = "pattern"
ALGO_HYBRID = "hybrid"


@dataclass
class CompressionConfig:
    """Configuration for compression."""

    algorithm: str = ALGO_HYBRID
    target_ratio: float = MIN_COMPRESSION_RATIO
    min_recall: float = MIN_RECALL
    level: int = 9  # zlib level


@dataclass
class CompressionResult:
    """Result of compression operation."""

    input_size: int
    output_size: int
    compression_ratio: float
    recall: float
    input_hash: str
    output_hash: str
    config_hash: str
    passed_slo: bool


def compress(data: bytes, config: Optional[Dict] = None) -> Tuple[bytes, Dict]:
    """Compress telemetry data.

    Args:
        data: Input bytes to compress
        config: Optional configuration dict

    Returns:
        Tuple of (compressed_data, stats_dict)

    Receipt: compress_receipt with input_hash, output_hash, compression_ratio, recall
    """
    if config is None:
        config = {}

    algorithm = config.get("algorithm", ALGO_HYBRID)
    level = config.get("level", 9)
    target_ratio = config.get("target_ratio", MIN_COMPRESSION_RATIO)
    min_recall = config.get("min_recall", MIN_RECALL)

    input_size = len(data)
    input_hash = dual_hash(data)
    config_hash = dual_hash(json.dumps(config, sort_keys=True))

    # Compression based on algorithm
    if algorithm == ALGO_ZLIB:
        compressed = _compress_zlib(data, level)
    elif algorithm == ALGO_PATTERN:
        compressed = _compress_pattern(data)
    else:  # ALGO_HYBRID
        compressed = _compress_hybrid(data, level)

    output_size = len(compressed)
    output_hash = dual_hash(compressed)
    compression_ratio = input_size / output_size if output_size > 0 else float("inf")

    # Verify recall by decompressing
    try:
        decompressed = decompress(compressed, config)
        recall = calculate_recall(data, decompressed)
    except Exception:
        recall = 0.0

    passed_slo = compression_ratio >= target_ratio and recall >= min_recall

    stats = {
        "input_size": input_size,
        "output_size": output_size,
        "compression_ratio": compression_ratio,
        "recall": recall,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "config_hash": config_hash,
        "passed_slo": passed_slo,
        "algorithm": algorithm,
    }

    # Emit compress receipt
    emit_receipt(
        "compress_receipt",
        {
            "tenant_id": TENANT_ID,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "compression_ratio": compression_ratio,
            "recall": recall,
            "config_hash": config_hash,
            "passed_slo": passed_slo,
        },
    )

    return compressed, stats


def decompress(compressed: bytes, config: Optional[Dict] = None) -> bytes:
    """Decompress data to original.

    Args:
        compressed: Compressed bytes
        config: Configuration dict (must match compression config)

    Returns:
        Decompressed bytes
    """
    if config is None:
        config = {}

    algorithm = config.get("algorithm", ALGO_HYBRID)

    if algorithm == ALGO_ZLIB:
        return _decompress_zlib(compressed)
    elif algorithm == ALGO_PATTERN:
        return _decompress_pattern(compressed)
    else:  # ALGO_HYBRID
        return _decompress_hybrid(compressed)


def calculate_recall(original: bytes, decompressed: bytes) -> float:
    """Measure recall (0-1). Must be >= 0.999.

    Recall = (matching bytes) / (total original bytes)

    Args:
        original: Original data
        decompressed: Decompressed data

    Returns:
        Recall score between 0 and 1
    """
    if len(original) == 0:
        return 1.0 if len(decompressed) == 0 else 0.0

    if len(original) != len(decompressed):
        # Length mismatch - partial credit based on overlap
        min_len = min(len(original), len(decompressed))
        matching = sum(1 for i in range(min_len) if original[i] == decompressed[i])
        return matching / len(original)

    matching = sum(1 for i in range(len(original)) if original[i] == decompressed[i])
    return matching / len(original)


# === COMPRESSION ALGORITHMS ===


def _compress_zlib(data: bytes, level: int = 9) -> bytes:
    """Standard zlib compression.

    Args:
        data: Input bytes
        level: Compression level (1-9)

    Returns:
        Compressed bytes with header
    """
    compressed = zlib.compress(data, level)
    # Add header: algorithm byte + original length
    header = struct.pack(">BI", 0x01, len(data))  # 0x01 = zlib
    return header + compressed


def _decompress_zlib(compressed: bytes) -> bytes:
    """Decompress zlib data.

    Args:
        compressed: Compressed bytes with header

    Returns:
        Decompressed bytes
    """
    # Parse header
    algo, orig_len = struct.unpack(">BI", compressed[:5])
    if algo != 0x01:
        raise ValueError(f"Unknown algorithm: {algo}")
    return zlib.decompress(compressed[5:])


def _compress_pattern(data: bytes) -> bytes:
    """Pattern-based compression for structured data.

    Identifies repeating patterns and encodes them efficiently.
    Best for telemetry with regular sampling.

    Args:
        data: Input bytes

    Returns:
        Compressed bytes with header
    """
    # Simple run-length encoding for demonstration
    # In production, this would use more sophisticated pattern detection
    if len(data) == 0:
        return struct.pack(">BI", 0x02, 0)

    result = []
    i = 0
    while i < len(data):
        # Find run length
        run_char = data[i]
        run_len = 1
        while (
            i + run_len < len(data) and data[i + run_len] == run_char and run_len < 255
        ):
            run_len += 1

        if run_len >= 3:
            # Encode run
            result.append(0xFF)  # Escape
            result.append(run_len)
            result.append(run_char)
            i += run_len
        else:
            # Literal
            if run_char == 0xFF:
                result.append(0xFF)
                result.append(0x00)  # Escaped literal 0xFF
            else:
                result.append(run_char)
            i += 1

    # Add header
    header = struct.pack(">BI", 0x02, len(data))  # 0x02 = pattern
    return header + bytes(result)


def _decompress_pattern(compressed: bytes) -> bytes:
    """Decompress pattern-encoded data.

    Args:
        compressed: Compressed bytes with header

    Returns:
        Decompressed bytes
    """
    algo, orig_len = struct.unpack(">BI", compressed[:5])
    if algo != 0x02:
        raise ValueError(f"Unknown algorithm: {algo}")

    if orig_len == 0:
        return b""

    data = compressed[5:]
    result = []
    i = 0

    while i < len(data):
        if data[i] == 0xFF:
            if i + 1 >= len(data):
                break
            if data[i + 1] == 0x00:
                # Escaped 0xFF literal
                result.append(0xFF)
                i += 2
            else:
                # Run
                run_len = data[i + 1]
                run_char = data[i + 2]
                result.extend([run_char] * run_len)
                i += 3
        else:
            result.append(data[i])
            i += 1

    return bytes(result)


def _compress_hybrid(data: bytes, level: int = 9) -> bytes:
    """Hybrid compression: try both, use best.

    Tries pattern compression first, then zlib.
    Uses whichever gives better ratio.

    Args:
        data: Input bytes
        level: zlib level

    Returns:
        Best compressed result with header
    """
    # Try zlib
    zlib_result = _compress_zlib(data, level)

    # Try pattern (only for data that might benefit)
    if len(data) > 100:
        pattern_result = _compress_pattern(data)
        if len(pattern_result) < len(zlib_result):
            return pattern_result

    return zlib_result


def _decompress_hybrid(compressed: bytes) -> bytes:
    """Decompress hybrid data.

    Reads algorithm from header and dispatches.

    Args:
        compressed: Compressed bytes with header

    Returns:
        Decompressed bytes
    """
    if len(compressed) < 5:
        return b""

    algo = compressed[0]
    if algo == 0x01:
        return _decompress_zlib(compressed)
    elif algo == 0x02:
        return _decompress_pattern(compressed)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# === VALIDATION ===


def validate_compression_slo(stats: Dict) -> bool:
    """Validate compression meets SLOs.

    Args:
        stats: Stats dict from compress()

    Returns:
        True if all SLOs pass
    """
    ratio_ok = stats.get("compression_ratio", 0) >= MIN_COMPRESSION_RATIO
    recall_ok = stats.get("recall", 0) >= MIN_RECALL
    return ratio_ok and recall_ok


def compress_stream(
    stream: bytes, chunk_size: int = 4096, config: Optional[Dict] = None
) -> Tuple[bytes, Dict]:
    """Compress a stream in chunks.

    Args:
        stream: Full stream bytes
        chunk_size: Bytes per chunk
        config: Compression config

    Returns:
        Tuple of (compressed_stream, aggregate_stats)
    """
    if config is None:
        config = {}

    chunks = [stream[i : i + chunk_size] for i in range(0, len(stream), chunk_size)]
    compressed_chunks = []
    total_input = 0
    total_output = 0
    recalls = []

    for chunk in chunks:
        compressed, stats = compress(chunk, config)
        compressed_chunks.append(compressed)
        total_input += stats["input_size"]
        total_output += stats["output_size"]
        recalls.append(stats["recall"])

    # Aggregate stats
    aggregate_stats = {
        "total_input_size": total_input,
        "total_output_size": total_output,
        "overall_compression_ratio": total_input / total_output
        if total_output > 0
        else float("inf"),
        "min_recall": min(recalls) if recalls else 1.0,
        "avg_recall": sum(recalls) / len(recalls) if recalls else 1.0,
        "n_chunks": len(chunks),
    }

    # Concatenate with length prefixes
    output = b""
    for chunk in compressed_chunks:
        output += struct.pack(">I", len(chunk)) + chunk

    return output, aggregate_stats
