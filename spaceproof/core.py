"""SpaceProof Core - CLAUDEME S8 compliant primitives.

Space-grade proof infrastructure. No receipt, not real.

This IS the SpaceProof genome - every receipt carries its DNA.
Part of ProofChain: SpaceProof | SpendProof | ClaimProof | VoteProof | OriginProof | GreenProof
"""

import hashlib
import json
from datetime import datetime
from typing import NoReturn

# === IDENTITY ===

TENANT_ID = "spaceproof"
"""Default tenant for all SpaceProof receipts. Per CLAUDEME S8."""

VERSION = "4.0.0"
"""SpaceProof version."""

SERIES = "ProofChain"
"""Part of the ProofChain series."""

# === RUNTIME DETECTION ===

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# === PHYSICS CONSTANTS (verified sources) ===

# Mars communication latency
MARS_LIGHT_DELAY_MIN_SEC = 180  # 3 min opposition
MARS_LIGHT_DELAY_MAX_SEC = 1320  # 22 min conjunction

# Mars relay bandwidth
MARS_RELAY_MBPS = 2.0  # MRO measured

# Human physiology
HUMAN_METABOLIC_W = 100  # Physiology

# NASA verified hardware
KILOPOWER_KW = 10  # NASA KRUSTY reactor
MOXIE_O2_G_PER_HR = 5.5  # Perseverance MOXIE
ISS_WATER_RECOVERY = 0.98  # NASA ECLSS 2023

# === DOGE CONSTANTS (GAO/CMS verified) ===

IMPROPER_PAYMENTS_TOTAL_B = 162  # GAO FY2024
MEDICAID_IMPROPER_B = 31.1  # CMS FY2024
MEDICARE_FFS_IMPROPER_B = 31.7  # CMS FY2024

# === COMPRESSION CONSTANTS ===

MIN_COMPRESSION_RATIO = 10
MIN_RECALL = 0.999

# === KAN CONSTANTS ===

KAN_ARCHITECTURE = [1, 6, 1]
MDL_BETA = 0.10
SPLINE_DEGREE = 3

# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA = {
    "type": "object",
    "required": ["receipt_type", "ts", "tenant_id", "payload_hash"],
    "properties": {
        "receipt_type": {"type": "string"},
        "ts": {"type": "string", "format": "date-time"},
        "tenant_id": {"type": "string"},
        "payload_hash": {"type": "string", "pattern": "^[a-f0-9]{64}:[a-f0-9]{64}$"},
    },
}
"""JSON Schema for receipt autodocumentation. Per CLAUDEME S8."""


# === CORE FUNCTIONS ===


def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 format. ALWAYS use this, never single hash.

    If str, encode to bytes first. If blake3 unavailable, duplicate SHA256.
    Return format: "{sha256_hex}:{blake3_hex}". Pure function, no side effects.

    Source: CLAUDEME S8 lines 472-478
    """
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict, tenant_id: str = TENANT_ID) -> dict:
    """Create receipt with required fields. Every function calls this.

    Creates receipt with:
    - ts: ISO8601+Z format
    - tenant_id: from param or data or TENANT_ID constant
    - payload_hash: dual_hash of JSON-serialized data

    Print JSON to stdout with flush=True. Return complete receipt dict.

    Source: CLAUDEME S8 lines 480-491
    """
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": data.get("tenant_id", tenant_id),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data,
    }
    print(json.dumps(receipt), flush=True)
    return receipt


def merkle(items: list) -> str:
    """Compute Merkle root of items.

    Empty list -> dual_hash(b"empty").
    Otherwise: hash each item (JSON serialize, sort_keys=True),
    pair-and-hash upward, duplicate last if odd count.
    Return root hash.

    Source: CLAUDEME S8 lines 497-507
    """
    if not items:
        return dual_hash(b"empty")
    hashes = [dual_hash(json.dumps(i, sort_keys=True)) for i in items]
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i + 1]) for i in range(0, len(hashes), 2)]
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

    Source: CLAUDEME S4.7 pattern
    """
    emit_receipt(
        "anomaly",
        {
            "metric": "hash_mismatch",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "violation",
            "action": "halt",
            "expected": expected,
            "actual": actual,
        },
    )
    raise StopRule(f"Hash mismatch: expected {expected}, got {actual}")


def stoprule_invalid_receipt(reason: str) -> NoReturn:
    """Emit anomaly_receipt and raise StopRule for invalid receipt.

    Emits anomaly_receipt with:
    - metric="invalid_receipt"
    - classification="anti_pattern"
    - action="halt"

    Then raises StopRule with reason.

    Source: CLAUDEME S4.7 pattern
    """
    emit_receipt(
        "anomaly",
        {
            "metric": "invalid_receipt",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "anti_pattern",
            "action": "halt",
            "reason": reason,
        },
    )
    raise StopRule(f"Invalid receipt: {reason}")
