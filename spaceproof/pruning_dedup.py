"""pruning_dedup.py - Phase 1: Deterministic Duplicate Removal

Zero risk deduplication. Identifies exact duplicate subtrees and removes them.
This is Phase 1 of the two-phase Merkle entropy pruning architecture.

Functions:
    - compute_subtree_hash: Hash a subtree for duplicate detection
    - dedup_prune: Remove exact duplicates from Merkle tree
"""

import json
import hashlib
from typing import Dict, Any

from .core import emit_receipt, dual_hash


def compute_subtree_hash(node: Dict[str, Any]) -> str:
    """Compute hash of a subtree for duplicate detection.

    Args:
        node: Subtree node with potential children

    Returns:
        Hash string representing subtree content
    """
    # Serialize node without volatile fields
    stable_content = {
        k: v for k, v in node.items() if k not in ("ts", "timestamp", "created_at")
    }
    return hashlib.sha256(
        json.dumps(stable_content, sort_keys=True).encode()
    ).hexdigest()


def dedup_prune(merkle_tree: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 1: Deterministic duplicate removal.

    Zero risk. Identifies exact duplicate subtrees and removes them.

    Args:
        merkle_tree: Dict containing tree structure

    Returns:
        Dict with pruned_tree, duplicates_removed, space_saved_pct

    Receipt: dedup_prune_receipt
    """
    leaves = merkle_tree.get("leaves", [])
    original_count = len(leaves)

    if original_count == 0:
        result = {
            "pruned_tree": merkle_tree,
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "space_saved_pct": 0.0,
        }
        emit_receipt(
            "dedup_prune",
            {
                "tenant_id": "spaceproof-pruning",
                **result,
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )
        return result

    # Find duplicates by content hash
    seen_hashes = {}
    unique_leaves = []
    duplicates_found = 0

    for leaf in leaves:
        leaf_hash = compute_subtree_hash(leaf)
        if leaf_hash not in seen_hashes:
            seen_hashes[leaf_hash] = leaf
            unique_leaves.append(leaf)
        else:
            duplicates_found += 1

    duplicates_removed = duplicates_found
    space_saved_pct = (
        round(duplicates_removed / original_count, 4) if original_count > 0 else 0.0
    )

    # Create pruned tree
    pruned_tree = {
        **merkle_tree,
        "leaves": unique_leaves,
        "pruning_phase": "dedup",
        "original_leaf_count": original_count,
        "pruned_leaf_count": len(unique_leaves),
    }

    result = {
        "pruned_tree": pruned_tree,
        "duplicates_found": duplicates_found,
        "duplicates_removed": duplicates_removed,
        "space_saved_pct": space_saved_pct,
    }

    emit_receipt(
        "dedup_prune",
        {
            "tenant_id": "spaceproof-pruning",
            "duplicates_found": duplicates_found,
            "duplicates_removed": duplicates_removed,
            "space_saved_pct": space_saved_pct,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "duplicates_found": duplicates_found,
                        "duplicates_removed": duplicates_removed,
                        "space_saved_pct": space_saved_pct,
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return result
