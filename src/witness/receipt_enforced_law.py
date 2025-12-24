"""Receipt-enforced law for D19.1.

THE KEY INSIGHT (from Grok):
  "Laws are not discovered—they are enforced by the receipt chain itself"

The receipt chain IS physical law. We don't simulate physics to find it.
We witness the chain.

Compression = predictability = lawfulness.
When the human-AI system becomes more predictable, it's not "behaving lawfully"—
it IS lawful. The law exists in the compression.

Chain causality > simulation causality.
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19.1 LAW ENFORCEMENT CONSTANTS ===

COMPRESSION_LAW_TARGET = 0.95
"""Compression ratio = predictability = lawfulness."""

LAW_ENFORCEMENT_MODE = "receipt_chain"
"""Laws enforced BY receipt chain, not discovered separately."""

CHAIN_CAUSALITY_PRIORITY = True
"""Chain causality > simulation causality."""


@dataclass
class EnforcedLaw:
    """A law enforced by the receipt chain."""

    law_id: str
    extracted_from_chain: bool = True
    compression_ratio: float = 0.0
    predictability: float = 0.0
    causality_verified: bool = False
    human_readable: str = ""
    chain_receipts: int = 0
    enforcement_mode: str = LAW_ENFORCEMENT_MODE
    created_at: str = ""


@dataclass
class LawEnforcement:
    """Law enforcement state."""

    enforcement_id: str
    active_laws: Dict[str, EnforcedLaw] = field(default_factory=dict)
    chain_length: int = 0
    total_enforcements: int = 0
    compression_history: List[float] = field(default_factory=list)
    config: Dict = field(default_factory=dict)


def init_enforcement(config: Dict = None) -> LawEnforcement:
    """Initialize law enforcement.

    Args:
        config: Configuration dict

    Returns:
        LawEnforcement instance

    Receipt: law_enforcement_init_receipt
    """
    config = config or {}
    enforcement_id = str(uuid.uuid4())[:8]

    enforcement = LawEnforcement(
        enforcement_id=enforcement_id,
        config=config,
    )

    emit_receipt(
        "law_enforcement_init",
        {
            "receipt_type": "law_enforcement_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "enforcement_id": enforcement_id,
            "mode": LAW_ENFORCEMENT_MODE,
            "compression_target": COMPRESSION_LAW_TARGET,
            "chain_causality_priority": CHAIN_CAUSALITY_PRIORITY,
            "payload_hash": dual_hash(
                json.dumps(
                    {"enforcement_id": enforcement_id, "mode": LAW_ENFORCEMENT_MODE},
                    sort_keys=True,
                )
            ),
        },
    )

    return enforcement


def extract_law_from_chain(enf: LawEnforcement, receipts: List[Dict]) -> Dict:
    """Extract law from receipt chain causality.

    The insight: Laws exist IN the chain. We don't discover them—
    we recognize them by measuring compression and causality.

    Args:
        enf: LawEnforcement instance
        receipts: List of receipts forming the chain

    Returns:
        Extracted law dict

    Receipt: law_extraction_receipt
    """
    if not receipts:
        return {"error": "empty_chain", "law": None}

    # Measure chain properties
    compression = measure_predictability(enf, receipts)
    causality_verified = validate_chain_causality(enf, receipts)

    law_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat() + "Z"

    # Extract law based on chain properties
    law = EnforcedLaw(
        law_id=law_id,
        extracted_from_chain=True,
        compression_ratio=compression,
        predictability=compression,  # Compression = predictability
        causality_verified=causality_verified,
        human_readable=f"Chain-enforced law: compression={compression:.4f}, causality_verified={causality_verified}",
        chain_receipts=len(receipts),
        created_at=now,
    )

    enf.chain_length += len(receipts)

    result = {
        "law_id": law_id,
        "compression_ratio": law.compression_ratio,
        "predictability": law.predictability,
        "causality_verified": law.causality_verified,
        "chain_receipts": law.chain_receipts,
        "human_readable": law.human_readable,
        "extracted_from_chain": True,
    }

    emit_receipt(
        "law_extraction",
        {
            "receipt_type": "law_extraction",
            "tenant_id": TENANT_ID,
            "ts": now,
            "enforcement_id": enf.enforcement_id,
            "law_id": law_id,
            "compression_ratio": law.compression_ratio,
            "predictability": law.predictability,
            "causality_verified": law.causality_verified,
            "chain_receipts": law.chain_receipts,
            "payload_hash": dual_hash(
                json.dumps(
                    {"law_id": law_id, "compression": law.compression_ratio},
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def validate_chain_causality(
    enf: LawEnforcement, receipts: List[Dict] = None, law: Dict = None
) -> bool:
    """Validate law is enforced by chain causality.

    Chain causality > simulation causality.
    A law is valid if and only if it can be derived from the receipt chain.

    Args:
        enf: LawEnforcement instance
        receipts: List of receipts to validate
        law: Optional law dict (unused, for interface compatibility)

    Returns:
        True if causality is verified
    """
    if not receipts:
        return False

    # Check temporal ordering on ORIGINAL receipt order
    # Causality violation = receipts received out of temporal order
    prev_ts = None
    for r in receipts:
        ts = r.get("ts")
        if ts and prev_ts and ts < prev_ts:
            return False  # Causality violation - out of order
        prev_ts = ts

    # Check hash chain integrity if payload_hash present
    for r in receipts:
        if "payload_hash" in r:
            # Verify dual-hash format (sha256:blake3)
            ph = r["payload_hash"]
            if ":" not in ph:
                return False

    return True


def measure_predictability(enf: LawEnforcement, receipts: List[Dict]) -> float:
    """Compression ratio = predictability = lawfulness.

    The insight: When the system becomes more predictable,
    it IS lawful. The law exists in the compression.

    Args:
        enf: LawEnforcement instance
        receipts: List of receipts

    Returns:
        Predictability/compression ratio 0-1
    """
    if not receipts:
        return 0.0

    # Measure entropy as inverse of predictability
    type_counts: Dict[str, int] = {}
    for r in receipts:
        rtype = r.get("receipt_type", "unknown")
        type_counts[rtype] = type_counts.get(rtype, 0) + 1

    total = len(receipts)
    max_entropy = math.log2(total) if total > 1 else 1.0

    entropy = 0.0
    for count in type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Predictability = 1 - normalized_entropy
    # Lower entropy = higher predictability = more lawful
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
        predictability = 1.0 - normalized_entropy
    else:
        predictability = 1.0

    # Track history
    enf.compression_history.append(predictability)
    if len(enf.compression_history) > 1000:
        enf.compression_history = enf.compression_history[-1000:]

    return round(predictability, 4)


def enforce_law(enf: LawEnforcement, law: Dict) -> Dict:
    """Apply law enforcement.

    Laws are not applied FROM outside—they are recognized
    AS the natural ordering of the receipt chain.

    Args:
        enf: LawEnforcement instance
        law: Law dict to enforce

    Returns:
        Enforcement result dict

    Receipt: law_enforcement_receipt
    """
    law_id = law.get("law_id", str(uuid.uuid4())[:8])
    now = datetime.utcnow().isoformat() + "Z"

    # Check if law meets enforcement criteria
    compression = law.get("compression_ratio", 0)
    causality = law.get("causality_verified", False)

    enforced = compression >= COMPRESSION_LAW_TARGET and causality

    if enforced:
        enforced_law = EnforcedLaw(
            law_id=law_id,
            extracted_from_chain=law.get("extracted_from_chain", True),
            compression_ratio=compression,
            predictability=compression,
            causality_verified=causality,
            human_readable=law.get("human_readable", ""),
            chain_receipts=law.get("chain_receipts", 0),
            created_at=now,
        )
        enf.active_laws[law_id] = enforced_law
        enf.total_enforcements += 1

    result = {
        "law_id": law_id,
        "enforced": enforced,
        "compression_ratio": compression,
        "causality_verified": causality,
        "meets_target": compression >= COMPRESSION_LAW_TARGET,
        "active_laws": len(enf.active_laws),
        "total_enforcements": enf.total_enforcements,
    }

    emit_receipt(
        "law_enforcement",
        {
            "receipt_type": "law_enforcement",
            "tenant_id": TENANT_ID,
            "ts": now,
            "enforcement_id": enf.enforcement_id,
            "law_id": law_id,
            "enforced": enforced,
            "compression_ratio": compression,
            "causality_verified": causality,
            "active_laws": len(enf.active_laws),
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "law_id": law_id,
                        "enforced": enforced,
                        "compression": compression,
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def emit_enforcement_receipt(enf: LawEnforcement, law: Dict) -> Dict:
    """Emit receipt_enforced_law_receipt.

    Args:
        enf: LawEnforcement instance
        law: Law being enforced

    Returns:
        Emitted receipt dict

    Receipt: receipt_enforced_law_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    result = {
        "receipt_type": "receipt_enforced_law_receipt",
        "tenant_id": TENANT_ID,
        "ts": now,
        "enforcement_id": enf.enforcement_id,
        "law_id": law.get("law_id"),
        "enforcement_mode": LAW_ENFORCEMENT_MODE,
        "compression_ratio": law.get("compression_ratio", 0),
        "predictability": law.get("predictability", law.get("compression_ratio", 0)),
        "causality_verified": law.get("causality_verified", False),
        "chain_receipts": law.get("chain_receipts", 0),
        "active_laws": len(enf.active_laws),
        "total_enforcements": enf.total_enforcements,
        "insight": "Laws are not discovered—they are enforced by the receipt chain itself",
        "payload_hash": dual_hash(
            json.dumps(
                {
                    "law_id": law.get("law_id"),
                    "compression": law.get("compression_ratio", 0),
                    "enforcements": enf.total_enforcements,
                },
                sort_keys=True,
            )
        ),
    }

    emit_receipt("receipt_enforced_law_receipt", result)
    return result


def get_enforcement_status(enf: LawEnforcement = None) -> Dict[str, Any]:
    """Current enforcement status.

    Args:
        enf: Optional enforcement instance

    Returns:
        Enforcement status dict
    """
    status = {
        "module": "witness.receipt_enforced_law",
        "version": "19.1.0",
        "compression_target": COMPRESSION_LAW_TARGET,
        "enforcement_mode": LAW_ENFORCEMENT_MODE,
        "chain_causality_priority": CHAIN_CAUSALITY_PRIORITY,
        "insight": "Laws are not discovered—they are enforced by the receipt chain itself",
    }

    if enf:
        avg_compression = (
            sum(enf.compression_history) / len(enf.compression_history)
            if enf.compression_history
            else 0
        )
        status.update(
            {
                "enforcement_id": enf.enforcement_id,
                "active_laws": len(enf.active_laws),
                "total_enforcements": enf.total_enforcements,
                "chain_length": enf.chain_length,
                "avg_compression": round(avg_compression, 4),
            }
        )

    return status
