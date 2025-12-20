"""D19.2 Weave to Chain - Insert Woven Laws into Current Chain.

PARADIGM: Preemptive laws woven into current chain.

Grok's Insight:
  "Preemptive laws woven into current chain. Delay appears nullified
   (already compensated). Verify: future-projected state matches when
   delay resolves."

The Physics:
  The receipt chain IS physical law. When we weave preemptive laws into it,
  we're encoding the future compensation into the present chain. The chain
  becomes a predictive fabric.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash, merkle, TENANT_ID, StopRule

# === D19.2 WEAVE TO CHAIN CONSTANTS ===

CHAIN_INSERTION_MODE = "preemptive"
"""Chain insertion mode - preemptive only, no reactive."""

VERIFY_CHAIN_INTEGRITY = True
"""Always verify chain integrity after insertion."""


@dataclass
class WovenLaw:
    """A law woven into the chain."""

    law_id: str
    law_type: str
    law_data: Dict
    chain_position: int
    insertion_hash: str
    merkle_root_before: str
    merkle_root_after: str
    inserted_at: str


@dataclass
class WeaveChain:
    """Chain with woven laws state."""

    chain_id: str
    woven_laws: Dict[str, WovenLaw] = field(default_factory=dict)
    chain_receipts: List[Dict] = field(default_factory=list)
    current_merkle_root: str = ""
    insertion_count: int = 0
    config: Dict = field(default_factory=dict)


def init_weave_chain(config: Dict = None) -> WeaveChain:
    """Initialize weave chain engine.

    Args:
        config: Optional configuration dict

    Returns:
        WeaveChain instance

    Receipt: weave_chain_init_receipt
    """
    config = config or {}
    chain_id = str(uuid.uuid4())[:8]

    chain = WeaveChain(
        chain_id=chain_id,
        config=config,
        current_merkle_root=dual_hash(b"genesis"),
    )

    emit_receipt(
        "weave_chain_init",
        {
            "receipt_type": "weave_chain_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "chain_id": chain_id,
            "insertion_mode": CHAIN_INSERTION_MODE,
            "verify_integrity": VERIFY_CHAIN_INTEGRITY,
            "genesis_root": chain.current_merkle_root[:32],
            "payload_hash": dual_hash(
                json.dumps({"chain_id": chain_id}, sort_keys=True)
            ),
        },
    )

    return chain


def insert_woven_law(
    chain: WeaveChain,
    law_id: str,
    law_type: str,
    law_data: Dict,
) -> WovenLaw:
    """Insert a woven law into the chain.

    Laws are inserted preemptively - before the delay arrives.

    Args:
        chain: WeaveChain instance
        law_id: Law identifier
        law_type: Type of law
        law_data: Law data to insert

    Returns:
        WovenLaw instance

    Receipt: law_insertion_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Capture merkle root before insertion
    merkle_before = chain.current_merkle_root

    # Create law receipt
    law_receipt = {
        "receipt_type": "woven_law",
        "law_id": law_id,
        "law_type": law_type,
        "law_data": law_data,
        "inserted_at": now,
    }

    # Add to chain
    chain.chain_receipts.append(law_receipt)
    chain_position = len(chain.chain_receipts) - 1

    # Compute new merkle root
    merkle_after = merkle(chain.chain_receipts)
    chain.current_merkle_root = merkle_after

    # Create insertion hash
    insertion_hash = dual_hash(json.dumps(law_receipt, sort_keys=True))

    woven = WovenLaw(
        law_id=law_id,
        law_type=law_type,
        law_data=law_data,
        chain_position=chain_position,
        insertion_hash=insertion_hash,
        merkle_root_before=merkle_before,
        merkle_root_after=merkle_after,
        inserted_at=now,
    )

    chain.woven_laws[law_id] = woven
    chain.insertion_count += 1

    emit_receipt(
        "law_insertion",
        {
            "receipt_type": "law_insertion",
            "tenant_id": TENANT_ID,
            "ts": now,
            "chain_id": chain.chain_id,
            "law_id": law_id,
            "law_type": law_type,
            "chain_position": chain_position,
            "insertion_hash": insertion_hash[:32],
            "merkle_root_before": merkle_before[:32],
            "merkle_root_after": merkle_after[:32],
            "insertion_mode": "preemptive",
            "payload_hash": dual_hash(
                json.dumps({"law_id": law_id, "position": chain_position}, sort_keys=True)
            ),
        },
    )

    return woven


def batch_insert_laws(
    chain: WeaveChain,
    laws: List[Dict],
) -> Dict[str, Any]:
    """Insert a batch of woven laws into the chain.

    Args:
        chain: WeaveChain instance
        laws: List of law dicts to insert

    Returns:
        Batch insertion result

    Receipt: batch_law_insertion_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    merkle_before = chain.current_merkle_root
    inserted = []

    for law in laws:
        law_id = law.get("law_id", str(uuid.uuid4())[:8])
        law_type = law.get("law_type", "preemptive_law")
        law_data = law.get("law_data", law)

        woven = insert_woven_law(chain, law_id, law_type, law_data)
        inserted.append({
            "law_id": woven.law_id,
            "chain_position": woven.chain_position,
        })

    merkle_after = chain.current_merkle_root

    result = {
        "chain_id": chain.chain_id,
        "laws_inserted": len(inserted),
        "merkle_root_before": merkle_before[:32],
        "merkle_root_after": merkle_after[:32],
        "chain_length": len(chain.chain_receipts),
        "insertion_mode": "preemptive_batch",
    }

    emit_receipt(
        "batch_law_insertion",
        {
            "receipt_type": "batch_law_insertion",
            "tenant_id": TENANT_ID,
            "ts": now,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def verify_chain_integrity(chain: WeaveChain) -> Dict[str, Any]:
    """Verify integrity of chain with woven laws.

    Args:
        chain: WeaveChain instance

    Returns:
        Verification result

    Receipt: chain_integrity_verification_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Recompute merkle root from chain receipts
    computed_root = merkle(chain.chain_receipts) if chain.chain_receipts else dual_hash(b"genesis")

    integrity_valid = computed_root == chain.current_merkle_root

    if not integrity_valid:
        raise StopRule(
            f"Chain integrity violation: computed_root={computed_root[:32]} != "
            f"current_root={chain.current_merkle_root[:32]}"
        )

    result = {
        "chain_id": chain.chain_id,
        "integrity_valid": integrity_valid,
        "chain_length": len(chain.chain_receipts),
        "woven_laws": len(chain.woven_laws),
        "merkle_root": chain.current_merkle_root[:32],
    }

    emit_receipt(
        "chain_integrity_verification",
        {
            "receipt_type": "chain_integrity_verification",
            "tenant_id": TENANT_ID,
            "ts": now,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_chain_status() -> Dict[str, Any]:
    """Get weave chain module status.

    Returns:
        Status dict
    """
    return {
        "module": "weave.weave_to_chain",
        "version": "19.2.0",
        "paradigm": "preemptive_chain_weaving",
        "insertion_mode": CHAIN_INSERTION_MODE,
        "verify_integrity": VERIFY_CHAIN_INTEGRITY,
        "insight": "The receipt chain IS physical law - preemptive laws woven into current chain",
    }
