"""CLAUDEME §8 core functions - required by all modules.

This IS the AXIOM genome - every receipt carries its DNA.
"""
import hashlib
import json
from datetime import datetime
from typing import NoReturn

# === CONSTANTS ===

TENANT_ID = "axiom-colony"
"""Default tenant for all AXIOM receipts. Per CLAUDEME §8."""

# Runtime detection of blake3 availability
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

RECEIPT_SCHEMA = {
    "type": "object",
    "required": ["receipt_type", "ts", "tenant_id", "payload_hash"],
    "properties": {
        "receipt_type": {"type": "string"},
        "ts": {"type": "string", "format": "date-time"},
        "tenant_id": {"type": "string"},
        "payload_hash": {"type": "string", "pattern": "^[a-f0-9]{64}:[a-f0-9]{64}$"}
    }
}
"""JSON Schema for receipt autodocumentation. Per CLAUDEME §8."""


# === CORE FUNCTIONS ===

def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 format. ALWAYS use this, never single hash.

    If str, encode to bytes first. If blake3 unavailable, duplicate SHA256.
    Return format: "{sha256_hex}:{blake3_hex}". Pure function, no side effects.

    Source: CLAUDEME §8 lines 472-478
    """
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Create receipt with required fields. Every function calls this.

    Creates receipt with:
    - ts: ISO8601+Z format
    - tenant_id: from data or TENANT_ID constant
    - payload_hash: dual_hash of JSON-serialized data

    Print JSON to stdout with flush=True. Return complete receipt dict.

    Source: CLAUDEME §8 lines 480-491
    """
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": data.get("tenant_id", TENANT_ID),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data
    }
    print(json.dumps(receipt), flush=True)
    return receipt


def merkle(items: list) -> str:
    """Compute Merkle root of items.

    Empty list → dual_hash(b"empty").
    Otherwise: hash each item (JSON serialize, sort_keys=True),
    pair-and-hash upward, duplicate last if odd count.
    Return root hash.

    Source: CLAUDEME §8 lines 497-507
    """
    if not items:
        return dual_hash(b"empty")
    hashes = [dual_hash(json.dumps(i, sort_keys=True)) for i in items]
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i + 1])
                  for i in range(0, len(hashes), 2)]
    return hashes[0]


# === EXCEPTION CLASS ===

class StopRule(Exception):
    """Raised when stoprule triggers. NEVER catch silently.

    Message should include context for debugging.
    """
    pass


# === STOPRULE FUNCTIONS ===

def stoprule_hash_mismatch(expected: str, actual: str) -> NoReturn:
    """Emit anomaly_receipt and raise StopRule for hash mismatch.

    Emits anomaly_receipt with:
    - metric="hash_mismatch"
    - classification="violation"
    - action="halt"

    Then raises StopRule with details.

    Source: CLAUDEME §4.7 pattern
    """
    emit_receipt("anomaly", {
        "metric": "hash_mismatch",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "expected": expected,
        "actual": actual
    })
    raise StopRule(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(reason: str) -> NoReturn:
    """Emit anomaly_receipt and raise StopRule for invalid receipt.

    Emits anomaly_receipt with:
    - metric="invalid_receipt"
    - classification="anti_pattern"
    - action="halt"

    Then raises StopRule with reason.

    Source: CLAUDEME §4.7 pattern
    """
    emit_receipt("anomaly", {
        "metric": "invalid_receipt",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "anti_pattern",
        "action": "halt",
        "reason": reason
    })
    raise StopRule(f"Invalid receipt: {reason}")
