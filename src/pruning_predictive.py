"""pruning_predictive.py - Phase 2: GNN-Predicted Pruning

Bounded risk pruning using GNN predictions to anticipate low-information branches.
This is Phase 2 of the two-phase Merkle entropy pruning architecture.

Functions:
    - predictive_prune: GNN-assisted low-entropy branch trimming
    - generate_gnn_predictions: Generate/simulate GNN predictions
"""

import json
import random
from typing import Dict, Any, List

from .core import emit_receipt, dual_hash, StopRule
from .constants import (
    ENTROPY_PRUNE_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_PROOF_PATHS_RETAINED,
)


def generate_gnn_predictions(
    merkle_tree: Dict[str, Any],
    entropy_classifications: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate simulated GNN predictions for low-entropy branches.

    In production, this would call an actual GNN model.
    For now, simulates predictions based on entropy classification.

    Args:
        merkle_tree: The Merkle tree to analyze
        entropy_classifications: Output from classify_leaf_entropy

    Returns:
        List of prediction dicts with branch_id, confidence, prune_recommended
    """
    classifications = entropy_classifications.get("classifications", {})
    predictions = []

    for leaf_id, classification in classifications.items():
        entropy_score = classification.get("entropy_score", 0.5)
        is_low = classification.get("classification") == "low"

        # Simulate GNN confidence (higher for clear cases)
        if is_low:
            # Low entropy - recommend pruning with high confidence
            confidence = 0.85 + random.uniform(0, 0.1)
            prune_recommended = True
        else:
            # High entropy - don't prune
            confidence = 0.90 + random.uniform(0, 0.08)
            prune_recommended = False

        predictions.append({
            "branch_id": leaf_id,
            "entropy_score": entropy_score,
            "confidence": round(confidence, 4),
            "prune_recommended": prune_recommended
        })

    return predictions


def predictive_prune(
    merkle_tree: Dict[str, Any],
    gnn_predictions: List[Dict[str, Any]],
    threshold: float = ENTROPY_PRUNE_THRESHOLD
) -> Dict[str, Any]:
    """Phase 2: GNN-predicted low-entropy branch trimming.

    Bounded risk. Uses GNN predictions to anticipate low-information branches.
    Raises StopRule if confidence < MIN_CONFIDENCE_THRESHOLD.

    Args:
        merkle_tree: Dict containing tree structure
        gnn_predictions: List of GNN predictions with branch_id, confidence
        threshold: Entropy threshold for pruning

    Returns:
        Dict with pruned_tree, branches_pruned, confidence_score

    Raises:
        StopRule: If GNN confidence < 0.7

    Receipt: predictive_prune_receipt
    """
    leaves = merkle_tree.get("leaves", [])
    original_count = len(leaves)

    if not gnn_predictions or original_count == 0:
        result = {
            "pruned_tree": merkle_tree,
            "predictions_made": 0,
            "branches_pruned": 0,
            "confidence_score": 0.0,
            "false_positive_rate": 0.0
        }
        emit_receipt("predictive_prune", {
            "tenant_id": "axiom-pruning",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
        })
        return result

    # Compute average confidence
    total_confidence = sum(p.get("confidence", 0.0) for p in gnn_predictions)
    avg_confidence = total_confidence / len(gnn_predictions)

    # StopRule if confidence too low
    if avg_confidence < MIN_CONFIDENCE_THRESHOLD:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-pruning",
            "metric": "predictive_confidence",
            "baseline": MIN_CONFIDENCE_THRESHOLD,
            "delta": avg_confidence - MIN_CONFIDENCE_THRESHOLD,
            "classification": "deviation",
            "action": "skip_predictive"
        })
        raise StopRule(f"Predictive confidence {avg_confidence:.3f} < {MIN_CONFIDENCE_THRESHOLD} threshold")

    # Get leaf IDs to prune based on predictions
    prune_ids = set()
    for pred in gnn_predictions:
        if pred.get("prune_recommended", False) and pred.get("confidence", 0) >= MIN_CONFIDENCE_THRESHOLD:
            prune_ids.add(pred.get("branch_id"))

    # Filter leaves, keeping proof paths
    pruned_leaves = []
    branches_pruned = 0
    proof_paths_seen = 0

    for i, leaf in enumerate(leaves):
        leaf_id = leaf.get("id", f"leaf_{i}")
        is_proof_path = leaf.get("is_proof_path", False) or leaf.get("audit_path", False)

        if is_proof_path:
            proof_paths_seen += 1
            pruned_leaves.append(leaf)
        elif leaf_id in prune_ids:
            branches_pruned += 1
        else:
            pruned_leaves.append(leaf)

    # Ensure minimum proof paths retained
    if proof_paths_seen < MIN_PROOF_PATHS_RETAINED:
        # Keep some additional leaves as safety
        additional_needed = MIN_PROOF_PATHS_RETAINED - proof_paths_seen
        for i, leaf in enumerate(leaves):
            if len(pruned_leaves) >= original_count - branches_pruned + additional_needed:
                break
            if leaf not in pruned_leaves:
                pruned_leaves.append(leaf)
                branches_pruned -= 1

    # Estimate false positive rate (leaves incorrectly pruned)
    # This would be validated against ground truth in production
    false_positive_rate = 1.0 - avg_confidence

    pruned_tree = {
        **merkle_tree,
        "leaves": pruned_leaves,
        "pruning_phase": "predictive",
        "original_leaf_count": original_count,
        "pruned_leaf_count": len(pruned_leaves)
    }

    result = {
        "pruned_tree": pruned_tree,
        "predictions_made": len(gnn_predictions),
        "branches_pruned": branches_pruned,
        "confidence_score": round(avg_confidence, 4),
        "false_positive_rate": round(false_positive_rate, 4)
    }

    emit_receipt("predictive_prune", {
        "tenant_id": "axiom-pruning",
        "predictions_made": len(gnn_predictions),
        "branches_pruned": branches_pruned,
        "confidence_score": round(avg_confidence, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "payload_hash": dual_hash(json.dumps({
            "predictions_made": len(gnn_predictions),
            "branches_pruned": branches_pruned,
            "confidence_score": avg_confidence
        }, sort_keys=True))
    })

    return result
