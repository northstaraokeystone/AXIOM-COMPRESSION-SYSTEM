"""AXIOM-COLONY v3.1 Core Module - CLAUDEME-compliant foundation."""

import hashlib
import json
from datetime import datetime, timezone

# Constants
TENANT_ID = "axiom-colony"

# Try to import blake3, fallback to sha256
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


class StopRule(Exception):
    """Never catch silently."""
    pass


def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 format. Fallback to SHA256:SHA256 if blake3 unavailable."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    sha256_hash = hashlib.sha256(data).hexdigest()
    if HAS_BLAKE3:
        blake3_hash = blake3.blake3(data).hexdigest()
    else:
        blake3_hash = hashlib.sha256(data + b'_blake3_fallback').hexdigest()
    return f"{sha256_hash}:{blake3_hash}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Create receipt with ts, tenant_id, payload_hash. Print JSON to stdout. Return dict."""
    payload = json.dumps(data, sort_keys=True, default=str)
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenant_id": TENANT_ID,
        "payload": data,
        "payload_hash": dual_hash(payload)
    }
    print(json.dumps(receipt))
    return receipt


def merkle(items: list) -> str:
    """Merkle root. Empty -> dual_hash(b'empty'). Odd count -> duplicate last."""
    if not items:
        return dual_hash(b'empty')
    hashes = [dual_hash(json.dumps(item, sort_keys=True, default=str)) for item in items]
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i + 1]) for i in range(0, len(hashes), 2)]
    return hashes[0]


def stoprule_hash_mismatch(expected: str, actual: str) -> None:
    """Emit anomaly_receipt and raise StopRule on hash mismatch."""
    emit_receipt("anomaly_receipt", {"type": "hash_mismatch", "expected": expected, "actual": actual})
    raise StopRule(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(reason: str) -> None:
    """Emit anomaly_receipt and raise StopRule on invalid receipt."""
    emit_receipt("anomaly_receipt", {"type": "invalid_receipt", "reason": reason})
    raise StopRule(f"Invalid receipt: {reason}")
