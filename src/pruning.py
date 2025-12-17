"""pruning.py - Two-Phase Merkle Entropy Pruning Module

THE PHYSICS (from Grok analysis):
    - Merkle batch entropy bounds as ~e*ln(n) (Shannon entropy bound)
    - e is physics (~2.71828), not parameter tuning
    - GNN caching surfaces this bound via nonlinear stabilization
    - Pruning compresses the ln(n) factor while e remains invariant

TWO-PHASE ARCHITECTURE:
    Phase 1: DEDUP (deterministic, zero risk) - see pruning_dedup.py
    Phase 2: PREDICTIVE (GNN-assisted, bounded risk) - see pruning_predictive.py

This module provides the main orchestration functions.
Split modules:
    - pruning_entropy.py: Shannon entropy computation
    - pruning_dedup.py: Phase 1 dedup functions
    - pruning_predictive.py: Phase 2 GNN-predicted pruning
    - pruning_verify.py: Chain/quorum verification

Source: Grok - "Not coincidence - Merkle batch entropy often bounds as ~e*ln(n)"
"""

import json
import math
import os
import random
from typing import Dict, Any, List

from .core import emit_receipt, dual_hash, StopRule
# Import all constants for use AND for backward-compatible re-export
from .constants import (
    ENTROPY_ASYMPTOTE_E,
    PRUNING_TARGET_ALPHA,
    BLACKOUT_PRUNING_TARGET_DAYS,
    OVERFLOW_THRESHOLD_DAYS_PRUNED,
    LN_N_TRIM_FACTOR_BASE,
    LN_N_TRIM_FACTOR_MAX,
    OVER_PRUNE_STOPRULE_THRESHOLD,
    ENTROPY_PRUNE_THRESHOLD,
    MIN_PROOF_PATHS_RETAINED,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_QUORUM_FRACTION,
    DEDUP_PRIORITY,
    PREDICTIVE_PRIORITY,
    DEDUP_RATIO_EXPECTED,
    PREDICTIVE_ACCURACY_TARGET,
    RETENTION_FACTOR_PRUNE_RANGE,
    ABLATION_MODES,
    ENTROPY_PRUNING_SPEC_PATH,
)

# Backward-compatible re-exports (other modules import these from here)
__all__ = [
    # Functions
    "load_entropy_pruning_spec",
    "entropy_prune",
    "dedup_prune",
    "predictive_prune",
    "classify_leaf_entropy",
    "compute_alpha_uplift",
    "compute_shannon_entropy",
    "compute_leaf_entropy",
    "compute_subtree_hash",
    "generate_sample_merkle_tree",
    "generate_gnn_predictions",
    "verify_chain_integrity",
    "verify_quorum_maintained",
    "get_retention_factor_prune_isolated",
    "extended_250d_projection",
    "get_pruning_info",
    "stoprule_over_prune",
    # Constants (re-exported for backward compatibility)
    "ENTROPY_ASYMPTOTE_E",
    "PRUNING_TARGET_ALPHA",
    "BLACKOUT_PRUNING_TARGET_DAYS",
    "OVERFLOW_THRESHOLD_DAYS_PRUNED",
    "LN_N_TRIM_FACTOR_BASE",
    "LN_N_TRIM_FACTOR_MAX",
    "OVER_PRUNE_STOPRULE_THRESHOLD",
    "ENTROPY_PRUNE_THRESHOLD",
    "MIN_PROOF_PATHS_RETAINED",
    "MIN_CONFIDENCE_THRESHOLD",
    "MIN_QUORUM_FRACTION",
    "DEDUP_PRIORITY",
    "PREDICTIVE_PRIORITY",
    "DEDUP_RATIO_EXPECTED",
    "PREDICTIVE_ACCURACY_TARGET",
    "RETENTION_FACTOR_PRUNE_RANGE",
    "ABLATION_MODES",
]

# Import split modules
from .pruning_entropy import (
    compute_shannon_entropy,
    compute_leaf_entropy,
    classify_leaf_entropy,
)
from .pruning_dedup import compute_subtree_hash, dedup_prune
from .pruning_predictive import predictive_prune, generate_gnn_predictions
from .pruning_verify import verify_chain_integrity, verify_quorum_maintained


def load_entropy_pruning_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify entropy pruning specification file.

    Args:
        path: Optional path override (default: ENTROPY_PRUNING_SPEC_PATH)

    Returns:
        Dict containing entropy pruning specification

    Receipt: entropy_pruning_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, ENTROPY_PRUNING_SPEC_PATH)

    with open(path, 'r') as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt("entropy_pruning_spec_ingest", {
        "tenant_id": "axiom-pruning",
        "file_path": path,
        "entropy_asymptote_e": data["entropy_asymptote_e"],
        "pruning_target_alpha": data["pruning_target_alpha"],
        "blackout_pruning_target_days": data["blackout_pruning_target_days"],
        "overflow_threshold_pruned_days": data["overflow_threshold_pruned_days"],
        "ln_n_trim_factor_base": data["ln_n_trim_factor_base"],
        "payload_hash": content_hash
    })

    return data


def compute_alpha_uplift(
    entropy_before: float,
    entropy_after: float,
    base_alpha: float = ENTROPY_ASYMPTOTE_E
) -> float:
    """Compute alpha uplift from entropy reduction.

    Formula: uplift = base_alpha * (1 + (entropy_before - entropy_after) / entropy_before)
    Capped to not exceed PRUNING_TARGET_ALPHA.

    Args:
        entropy_before: Total entropy before pruning
        entropy_after: Total entropy after pruning
        base_alpha: Base effective alpha (default: e)

    Returns:
        Alpha uplift value
    """
    if entropy_before <= 0:
        return base_alpha

    entropy_reduction = (entropy_before - entropy_after) / entropy_before
    uplift_factor = 1.0 + (entropy_reduction * 0.1)

    uplifted_alpha = base_alpha * uplift_factor
    return round(min(PRUNING_TARGET_ALPHA, uplifted_alpha), 4)


def stoprule_over_prune(trim_factor: float) -> None:
    """StopRule if trim_factor exceeds safe threshold.

    Args:
        trim_factor: The trim factor being applied

    Raises:
        StopRule: If trim_factor > 0.6
    """
    if trim_factor > OVER_PRUNE_STOPRULE_THRESHOLD:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-pruning",
            "metric": "trim_factor",
            "baseline": OVER_PRUNE_STOPRULE_THRESHOLD,
            "delta": trim_factor - OVER_PRUNE_STOPRULE_THRESHOLD,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(f"Over-prune: trim_factor {trim_factor} > {OVER_PRUNE_STOPRULE_THRESHOLD} threshold")


def get_retention_factor_prune_isolated(
    merkle_tree: Dict[str, Any],
    trim_factor: float = LN_N_TRIM_FACTOR_BASE
) -> Dict[str, Any]:
    """Get isolated pruning retention factor contribution.

    Args:
        merkle_tree: Dict containing tree structure
        trim_factor: ln(n) trim factor (0.3-0.5 range)

    Returns:
        Dict with retention_factor_prune, contribution_pct, range_expected

    Receipt: retention_prune_isolated
    """
    leaves = merkle_tree.get("leaves", [])
    n_leaves = len(leaves)

    if n_leaves == 0:
        retention_factor_prune = 1.0
    else:
        min_retention, max_retention = RETENTION_FACTOR_PRUNE_RANGE
        retention_range = max_retention - min_retention

        size_factor = min(1.0, math.log(max(1, n_leaves)) / 10)
        trim_factor_normalized = trim_factor / LN_N_TRIM_FACTOR_MAX

        retention_factor_prune = min_retention + (size_factor * trim_factor_normalized * retention_range)
        retention_factor_prune = round(min(max_retention, max(min_retention, retention_factor_prune)), 4)

    contribution_pct = round((retention_factor_prune - 1.0) * 100, 3)

    result = {
        "n_leaves": n_leaves,
        "trim_factor": trim_factor,
        "retention_factor_prune": retention_factor_prune,
        "contribution_pct": contribution_pct,
        "range_expected": RETENTION_FACTOR_PRUNE_RANGE,
        "layer": "pruning"
    }

    emit_receipt("retention_prune_isolated", {
        "tenant_id": "axiom-pruning",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def entropy_prune(
    merkle_tree: Dict[str, Any],
    trim_factor: float = LN_N_TRIM_FACTOR_BASE,
    hybrid: bool = True,
    ablation_mode: str = "full"
) -> Dict[str, Any]:
    """Orchestrate two-phase entropy pruning.

    Phase 1: Deterministic dedup (zero risk)
    Phase 2: GNN-predicted pruning (bounded risk, if hybrid=True)

    Args:
        merkle_tree: Dict containing tree structure
        trim_factor: ln(n) trim factor (0.3-0.5 range)
        hybrid: Whether to enable predictive pruning (default: True)
        ablation_mode: Ablation mode for testing (default: "full")

    Returns:
        Dict with pruned_tree, alpha_uplift, entropy_reduction, etc.

    Raises:
        StopRule: If over-prune or chain broken

    Receipt: entropy_pruning_receipt
    """
    # Handle ablation modes
    if ablation_mode == "baseline" or ablation_mode == "no_prune":
        original_root = merkle_tree.get("root", dual_hash(json.dumps(merkle_tree, sort_keys=True)))
        result = {
            "pruned_tree": merkle_tree,
            "merkle_root_before": original_root[:32],
            "merkle_root_after": original_root[:32],
            "branches_pruned": 0,
            "entropy_before": 0.0,
            "entropy_after": 0.0,
            "entropy_reduction_pct": 0.0,
            "alpha_uplift": ENTROPY_ASYMPTOTE_E,
            "trim_factor_used": 0.0,
            "hybrid_enabled": False,
            "dedup_removed": 0,
            "predictive_pruned": 0,
            "confidence_score": 0.0,
            "retention_factor_prune": 1.0,
            "ablation_mode": ablation_mode
        }
        emit_receipt("entropy_pruning", {
            "tenant_id": "axiom-pruning",
            "receipt_type": "entropy_pruning",
            **{k: v for k, v in result.items() if k != "pruned_tree"},
            "payload_hash": dual_hash(json.dumps({k: v for k, v in result.items() if k != "pruned_tree"}, sort_keys=True))
        })
        return result

    # Validate trim factor
    stoprule_over_prune(trim_factor)

    original_root = merkle_tree.get("root", dual_hash(json.dumps(merkle_tree, sort_keys=True)))
    original_leaves = merkle_tree.get("leaves", [])
    original_count = len(original_leaves)

    # Compute initial entropy
    entropy_before = sum(compute_leaf_entropy(leaf) for leaf in original_leaves)

    # Phase 1: Deterministic dedup
    dedup_result = dedup_prune(merkle_tree)
    phase1_tree = dedup_result["pruned_tree"]

    # Phase 2: Predictive pruning (if hybrid enabled)
    if hybrid:
        entropy_class = classify_leaf_entropy(phase1_tree, ENTROPY_PRUNE_THRESHOLD * trim_factor)
        predictions = generate_gnn_predictions(phase1_tree, entropy_class)

        try:
            pred_result = predictive_prune(phase1_tree, predictions, ENTROPY_PRUNE_THRESHOLD * trim_factor)
            final_tree = pred_result["pruned_tree"]
            predictive_branches = pred_result["branches_pruned"]
            confidence_score = pred_result["confidence_score"]
        except StopRule:
            final_tree = phase1_tree
            predictive_branches = 0
            confidence_score = 0.0
    else:
        final_tree = phase1_tree
        predictive_branches = 0
        confidence_score = 0.0

    # Compute final entropy
    entropy_after = sum(compute_leaf_entropy(leaf) for leaf in final_tree.get("leaves", []))

    # Compute pruned root
    pruned_root = dual_hash(json.dumps(final_tree, sort_keys=True))

    # Verify integrity
    proof_paths = [l for l in final_tree.get("leaves", []) if l.get("is_proof_path") or l.get("audit_path")]
    if not proof_paths:
        proof_paths = final_tree.get("leaves", [])[:MIN_PROOF_PATHS_RETAINED]

    verify_chain_integrity(original_root, pruned_root, proof_paths)
    verify_quorum_maintained(final_tree)

    # Compute metrics
    alpha_uplift = compute_alpha_uplift(entropy_before, entropy_after)
    prune_isolated = get_retention_factor_prune_isolated(merkle_tree, trim_factor)
    retention_factor_prune = prune_isolated["retention_factor_prune"]
    branches_pruned = original_count - len(final_tree.get("leaves", []))
    entropy_reduction_pct = round((entropy_before - entropy_after) / max(0.001, entropy_before) * 100, 2)

    result = {
        "pruned_tree": final_tree,
        "merkle_root_before": original_root[:32],
        "merkle_root_after": pruned_root[:32],
        "branches_pruned": branches_pruned,
        "entropy_before": round(entropy_before, 4),
        "entropy_after": round(entropy_after, 4),
        "entropy_reduction_pct": entropy_reduction_pct,
        "alpha_uplift": alpha_uplift,
        "trim_factor_used": trim_factor,
        "hybrid_enabled": hybrid,
        "dedup_removed": dedup_result["duplicates_removed"],
        "predictive_pruned": predictive_branches,
        "confidence_score": confidence_score,
        "retention_factor_prune": retention_factor_prune,
        "ablation_mode": ablation_mode
    }

    emit_receipt("entropy_pruning", {
        "tenant_id": "axiom-pruning",
        "receipt_type": "entropy_pruning",
        "merkle_root_before": original_root[:32],
        "merkle_root_after": pruned_root[:32],
        "branches_pruned": branches_pruned,
        "entropy_before": round(entropy_before, 4),
        "entropy_after": round(entropy_after, 4),
        "entropy_reduction_pct": entropy_reduction_pct,
        "alpha_uplift": alpha_uplift,
        "trim_factor_used": trim_factor,
        "hybrid_enabled": hybrid,
        "retention_factor_prune": retention_factor_prune,
        "ablation_mode": ablation_mode,
        "payload_hash": dual_hash(json.dumps({
            "merkle_root_before": original_root[:32],
            "merkle_root_after": pruned_root[:32],
            "branches_pruned": branches_pruned,
            "alpha_uplift": alpha_uplift,
            "retention_factor_prune": retention_factor_prune
        }, sort_keys=True))
    })

    return result


def extended_250d_projection(
    base_projection: Dict[str, Any],
    pruning_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Project sovereignty timeline to 250d with pruning.

    Args:
        base_projection: Base projection dict with effective_alpha
        pruning_result: Output from entropy_prune

    Returns:
        Dict with extended projection metrics

    Receipt: extended_250d_receipt
    """
    base_alpha = base_projection.get("effective_alpha", ENTROPY_ASYMPTOTE_E)
    alpha_uplift = pruning_result.get("alpha_uplift", base_alpha)
    enhanced_alpha = max(base_alpha, alpha_uplift)
    target_achieved = enhanced_alpha >= PRUNING_TARGET_ALPHA
    overflow_margin = OVERFLOW_THRESHOLD_DAYS_PRUNED - BLACKOUT_PRUNING_TARGET_DAYS

    result = {
        "base_alpha": base_alpha,
        "alpha_uplift": alpha_uplift,
        "enhanced_alpha": enhanced_alpha,
        "target_alpha": PRUNING_TARGET_ALPHA,
        "target_achieved": target_achieved,
        "blackout_days": BLACKOUT_PRUNING_TARGET_DAYS,
        "overflow_threshold": OVERFLOW_THRESHOLD_DAYS_PRUNED,
        "overflow_margin": overflow_margin,
        "pruning_enabled": True
    }

    emit_receipt("extended_250d", {
        "tenant_id": "axiom-pruning",
        "receipt_type": "extended_250d",
        "blackout_days": BLACKOUT_PRUNING_TARGET_DAYS,
        "eff_alpha": enhanced_alpha,
        "pruning_enabled": True,
        "overflow_margin": overflow_margin,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def generate_sample_merkle_tree(n_leaves: int = 100, duplicate_ratio: float = 0.2) -> Dict[str, Any]:
    """Generate a sample Merkle tree for testing.

    Args:
        n_leaves: Number of leaves to generate
        duplicate_ratio: Fraction of leaves that are duplicates

    Returns:
        Dict representing Merkle tree structure
    """
    leaves = []
    unique_leaves = int(n_leaves * (1 - duplicate_ratio))
    duplicate_leaves = n_leaves - unique_leaves

    for i in range(unique_leaves):
        if random.random() < 0.8:
            leaf = {
                "id": f"leaf_{i}",
                "type": "telemetry",
                "data": {"metric": f"metric_{i % 10}", "value": i},
                "is_proof_path": i < MIN_PROOF_PATHS_RETAINED
            }
        else:
            leaf = {
                "id": f"leaf_{i}",
                "type": "decision",
                "data": {
                    "decision_id": f"dec_{i}",
                    "payload": os.urandom(32).hex(),
                    "timestamp": f"2025-01-{(i % 28) + 1:02d}T00:00:00Z"
                },
                "is_proof_path": i < MIN_PROOF_PATHS_RETAINED
            }
        leaves.append(leaf)

    for i in range(duplicate_leaves):
        original = random.choice(leaves[:unique_leaves])
        duplicate = {**original, "id": f"leaf_{unique_leaves + i}"}
        leaves.append(duplicate)

    random.shuffle(leaves)

    for i in range(min(MIN_PROOF_PATHS_RETAINED, len(leaves))):
        leaves[i]["is_proof_path"] = True

    tree = {
        "root": dual_hash(json.dumps(leaves, sort_keys=True)),
        "leaves": leaves,
        "leaf_count": len(leaves),
        "created_at": "2025-01-01T00:00:00Z"
    }

    return tree


def get_pruning_info() -> Dict[str, Any]:
    """Get pruning module configuration info.

    Returns:
        Dict with all pruning constants and configuration

    Receipt: pruning_info
    """
    info = {
        "entropy_asymptote_e": ENTROPY_ASYMPTOTE_E,
        "pruning_target_alpha": PRUNING_TARGET_ALPHA,
        "blackout_pruning_target_days": BLACKOUT_PRUNING_TARGET_DAYS,
        "overflow_threshold_pruned_days": OVERFLOW_THRESHOLD_DAYS_PRUNED,
        "ln_n_trim_factor_base": LN_N_TRIM_FACTOR_BASE,
        "ln_n_trim_factor_max": LN_N_TRIM_FACTOR_MAX,
        "entropy_prune_threshold": ENTROPY_PRUNE_THRESHOLD,
        "min_proof_paths_retained": MIN_PROOF_PATHS_RETAINED,
        "description": "Two-phase Merkle entropy pruning: deterministic dedup + GNN-predicted trimming"
    }

    emit_receipt("pruning_info", {
        "tenant_id": "axiom-pruning",
        **info,
        "payload_hash": dual_hash(json.dumps(info, sort_keys=True))
    })

    return info
