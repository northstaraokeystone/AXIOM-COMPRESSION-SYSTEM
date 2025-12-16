"""prove.py - Receipt Chain & Merkle Proofs (Updated for axiom-core)

THE PROOF INFRASTRUCTURE:
    Every calculation is verifiable.
    The merkle root is the cryptographic summary.
    Anyone can verify any result without seeing all others.

Source: CLAUDEME.md (S8)
"""

import json
from typing import Tuple, Any

from .core import dual_hash, emit_receipt, merkle

# === CONSTANTS ===

TENANT_ID = "axiom-core"
"""Receipt tenant isolation."""


def build_merkle_tree(items: list) -> Tuple[str, list]:
    """Build full merkle tree, returning root and all levels.

    Args:
        items: List of receipt dicts

    Returns:
        Tuple of (root_hash: str, levels: list[list[str]])
        levels[0] = leaf hashes, levels[-1] = [root]
    """
    if not items:
        empty_hash = dual_hash(b"empty")
        return empty_hash, [[empty_hash]]

    level_0 = [dual_hash(json.dumps(item, sort_keys=True)) for item in items]
    levels = [level_0]

    current_level = level_0
    while len(current_level) > 1:
        if len(current_level) % 2:
            current_level = current_level + [current_level[-1]]

        next_level = []
        for i in range(0, len(current_level), 2):
            parent = dual_hash(current_level[i] + current_level[i + 1])
            next_level.append(parent)

        levels.append(next_level)
        current_level = next_level

    root = current_level[0] if current_level else dual_hash(b"empty")
    return root, levels


def chain_receipts(receipts: list) -> dict:
    """Chain receipts and emit chain_receipt.

    Args:
        receipts: List of receipt dicts

    Returns:
        The chain_receipt dict

    MUST emit receipt (chain_receipt).
    """
    if not receipts:
        root = dual_hash(b"empty")
        return emit_receipt("chain", {
            "tenant_id": TENANT_ID,
            "n_receipts": 0,
            "merkle_root": root
        })

    root = merkle(receipts)

    return emit_receipt("chain", {
        "tenant_id": TENANT_ID,
        "n_receipts": len(receipts),
        "merkle_root": root
    })


def verify_proof(receipt: dict, proof_path: list, root: str) -> bool:
    """Verify a receipt is in the chain using its proof path.

    Args:
        receipt: The receipt being verified
        proof_path: List of {"sibling": hash, "position": "left"|"right"}
        root: The merkle root to verify against

    Returns:
        True if computed root matches provided root
    """
    current_hash = dual_hash(json.dumps(receipt, sort_keys=True))

    for step in proof_path:
        sibling = step["sibling"]
        position = step["position"]

        if position == "left":
            combined = sibling + current_hash
        else:
            combined = current_hash + sibling

        current_hash = dual_hash(combined)

    return current_hash == root
