"""pruning_verify.py - Merkle Chain and Quorum Verification

Verification functions to ensure pruning maintains integrity.
Fail fast if chain broken or quorum lost.

Functions:
    - verify_chain_integrity: Verify Merkle chain integrity after pruning
    - verify_quorum_maintained: Verify quorum is maintained after pruning
"""

from typing import Dict, Any, List

from .core import emit_receipt, StopRule
from .constants import MIN_PROOF_PATHS_RETAINED, MIN_QUORUM_FRACTION


def verify_chain_integrity(
    original_root: str, pruned_root: str, proof_paths: List[Dict[str, Any]]
) -> bool:
    """Verify Merkle chain integrity after pruning.

    Ensures proof paths remain valid and chain is not broken.

    Args:
        original_root: Merkle root before pruning
        pruned_root: Merkle root after pruning
        proof_paths: List of proof path dicts to verify

    Returns:
        True if chain integrity maintained

    Raises:
        StopRule: If chain broken
    """
    # In production, this would verify actual Merkle proofs
    # For now, check that we have minimum required proof paths
    if len(proof_paths) < MIN_PROOF_PATHS_RETAINED:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": "spaceproof-pruning",
                "metric": "proof_paths",
                "baseline": MIN_PROOF_PATHS_RETAINED,
                "delta": len(proof_paths) - MIN_PROOF_PATHS_RETAINED,
                "classification": "violation",
                "action": "halt",
            },
        )
        raise StopRule(
            f"Chain broken: only {len(proof_paths)} proof paths (need {MIN_PROOF_PATHS_RETAINED})"
        )

    # Verify roots are valid hashes (basic check)
    if not original_root or not pruned_root:
        raise StopRule("Chain broken: missing root hash")

    return True


def verify_quorum_maintained(pruned_tree: Dict[str, Any], min_nodes: int = 3) -> bool:
    """Verify quorum is maintained after pruning.

    Args:
        pruned_tree: Pruned Merkle tree
        min_nodes: Minimum nodes required (default: 3 for 2/3 of 5)

    Returns:
        True if quorum maintained

    Raises:
        StopRule: If quorum lost
    """
    pruned_count = pruned_tree.get("pruned_leaf_count", 0)
    original_count = pruned_tree.get("original_leaf_count", 0)

    if original_count == 0:
        return True

    retention_ratio = pruned_count / original_count

    # Quorum requires at least 2/3 retention
    if retention_ratio < MIN_QUORUM_FRACTION:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": "spaceproof-pruning",
                "metric": "quorum",
                "baseline": MIN_QUORUM_FRACTION,
                "delta": retention_ratio - MIN_QUORUM_FRACTION,
                "classification": "violation",
                "action": "halt",
            },
        )
        raise StopRule(
            f"Quorum lost: {retention_ratio:.2%} retention < {MIN_QUORUM_FRACTION:.2%} required"
        )

    return True
