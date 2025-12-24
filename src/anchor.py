"""anchor.py - Cryptographic Proof Generation and Verification

D20 Production Evolution: Stakeholder-intuitive name for Merkle proofs.

THE ANCHOR INSIGHT:
    An anchor is a cryptographic commitment.
    Once anchored, data cannot be modified without detection.
    The Merkle root is the signature of truth.

Source: AXIOM D20 Production Evolution

Stakeholder Value:
    - Defense: Tamper-proof verification
    - NRO: Decision lineage anchoring
    - DOGE: Audit trail integrity
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

from .core import emit_receipt, dual_hash, merkle

# === CONSTANTS ===

TENANT_ID = "axiom-anchor"


@dataclass
class Proof:
    """Merkle proof for an item."""

    item_hash: str
    proof_path: List[Dict]  # [{"sibling": hash, "position": "left"|"right"}, ...]
    root: str
    index: int


@dataclass
class AnchorResult:
    """Result of anchoring a batch of items."""

    root: str
    item_count: int
    timestamp: str
    algorithm: str
    proofs: Dict[str, Proof]  # item_hash -> Proof


def create_proof(item: Dict, items: List[Dict]) -> Proof:
    """Generate Merkle proof path for an item.

    Args:
        item: The item to prove inclusion for
        items: Full list of items (item must be in this list)

    Returns:
        Proof object with path from item to root
    """
    if not items:
        raise ValueError("Cannot create proof for empty item list")

    # Find item index
    item_hash = dual_hash(json.dumps(item, sort_keys=True))
    item_index = -1

    for i, it in enumerate(items):
        if dual_hash(json.dumps(it, sort_keys=True)) == item_hash:
            item_index = i
            break

    if item_index == -1:
        raise ValueError("Item not found in items list")

    # Build Merkle tree levels
    root, levels = _build_merkle_tree(items)

    # Build proof path
    proof_path = []
    idx = item_index

    for level in levels[:-1]:  # Exclude root level
        if idx % 2 == 0:
            # We're on the left, sibling is on right
            sibling_idx = idx + 1
            position = "right"
        else:
            # We're on the right, sibling is on left
            sibling_idx = idx - 1
            position = "left"

        if sibling_idx < len(level):
            proof_path.append(
                {
                    "sibling": level[sibling_idx],
                    "position": position,
                }
            )

        # Move to parent level
        idx = idx // 2

    return Proof(
        item_hash=item_hash,
        proof_path=proof_path,
        root=root,
        index=item_index,
    )


def verify_proof(item: Dict, proof: Proof, root: str) -> bool:
    """Verify a proof against a Merkle root.

    Args:
        item: The item being verified
        proof: Proof object with path
        root: The Merkle root to verify against

    Returns:
        True if proof is valid
    """
    item_hash = dual_hash(json.dumps(item, sort_keys=True))

    if item_hash != proof.item_hash:
        return False

    current_hash = item_hash

    for step in proof.proof_path:
        sibling = step["sibling"]
        position = step["position"]

        if position == "left":
            combined = sibling + current_hash
        else:
            combined = current_hash + sibling

        current_hash = dual_hash(combined)

    return current_hash == root


def anchor_batch(items: List[Dict]) -> AnchorResult:
    """Anchor a batch of items, returning root and metadata.

    Args:
        items: List of items to anchor

    Returns:
        AnchorResult with root, proofs, and metadata
    """
    from datetime import datetime

    if not items:
        empty_root = dual_hash(b"empty")
        emit_receipt(
            "anchor_receipt",
            {
                "tenant_id": TENANT_ID,
                "root": empty_root,
                "item_count": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "algorithm": "dual_hash_merkle",
            },
        )
        return AnchorResult(
            root=empty_root,
            item_count=0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            algorithm="dual_hash_merkle",
            proofs={},
        )

    # Build tree and get root
    root, levels = _build_merkle_tree(items)
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Generate proofs for all items
    proofs = {}
    for item in items:
        proof = create_proof(item, items)
        proofs[proof.item_hash] = proof

    result = AnchorResult(
        root=root,
        item_count=len(items),
        timestamp=timestamp,
        algorithm="dual_hash_merkle",
        proofs=proofs,
    )

    # Emit anchor receipt
    emit_receipt(
        "anchor_receipt",
        {
            "tenant_id": TENANT_ID,
            "root": root,
            "item_count": len(items),
            "timestamp": timestamp,
            "algorithm": "dual_hash_merkle",
        },
    )

    return result


def _build_merkle_tree(items: List[Dict]) -> Tuple[str, List[List[str]]]:
    """Build full Merkle tree, returning root and all levels.

    Args:
        items: List of items

    Returns:
        Tuple of (root_hash, levels)
        levels[0] = leaf hashes, levels[-1] = [root]
    """
    if not items:
        empty_hash = dual_hash(b"empty")
        return empty_hash, [[empty_hash]]

    # Level 0: leaf hashes
    level_0 = [dual_hash(json.dumps(item, sort_keys=True)) for item in items]
    levels = [level_0]

    current_level = level_0
    while len(current_level) > 1:
        # Pad with last element if odd
        if len(current_level) % 2:
            current_level = current_level + [current_level[-1]]

        # Build next level
        next_level = []
        for i in range(0, len(current_level), 2):
            parent = dual_hash(current_level[i] + current_level[i + 1])
            next_level.append(parent)

        levels.append(next_level)
        current_level = next_level

    root = current_level[0] if current_level else dual_hash(b"empty")
    return root, levels


def verify_batch(items: List[Dict], root: str) -> Dict:
    """Verify all items against a root.

    Args:
        items: List of items
        root: Expected Merkle root

    Returns:
        Dict with verification results
    """
    computed_root = merkle(items)

    verified = computed_root == root
    verified_items = []
    failed_items = []

    if verified:
        for item in items:
            verified_items.append(dual_hash(json.dumps(item, sort_keys=True)))
    else:
        # Try to identify which items differ
        for item in items:
            item_hash = dual_hash(json.dumps(item, sort_keys=True))
            try:
                proof = create_proof(item, items)
                if verify_proof(item, proof, root):
                    verified_items.append(item_hash)
                else:
                    failed_items.append(item_hash)
            except Exception:
                failed_items.append(item_hash)

    result = {
        "verified": verified,
        "expected_root": root,
        "computed_root": computed_root,
        "verified_count": len(verified_items),
        "failed_count": len(failed_items),
        "verified_items": verified_items,
        "failed_items": failed_items,
    }

    emit_receipt(
        "anchor_verify",
        {
            "tenant_id": TENANT_ID,
            "verified": verified,
            "expected_root": root,
            "computed_root": computed_root,
            "item_count": len(items),
            "failed_count": len(failed_items),
        },
    )

    return result


# === CHAIN RECEIPTS (from prove.py) ===


def chain_receipts(receipts: List[Dict]) -> Dict:
    """Chain receipts and emit chain_receipt.

    Args:
        receipts: List of receipt dicts

    Returns:
        The chain_receipt dict
    """
    if not receipts:
        root = dual_hash(b"empty")
        return emit_receipt(
            "chain",
            {"tenant_id": TENANT_ID, "n_receipts": 0, "merkle_root": root},
        )

    root = merkle(receipts)

    return emit_receipt(
        "chain",
        {"tenant_id": TENANT_ID, "n_receipts": len(receipts), "merkle_root": root},
    )
