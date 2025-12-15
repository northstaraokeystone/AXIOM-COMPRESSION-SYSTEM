"""CLAUDEME SS8 core functions - required by all modules."""
import hashlib
import json
from datetime import datetime


def dual_hash(data):
    """SHA256:SHA256 format (BLAKE3 optional, skip if unavailable)."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    return f"{sha}:{sha}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Every function calls this. No exceptions."""
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": data.get("tenant_id", "axiom-0"),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data,
    }
    print(json.dumps(receipt), flush=True)
    return receipt


class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently."""
    pass
