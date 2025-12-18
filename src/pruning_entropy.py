"""pruning_entropy.py - Shannon Entropy Computation for Pruning

Entropy calculation and leaf classification for Merkle tree pruning.
Uses Shannon entropy: H = -sum(p_i * log2(p_i))

Functions:
    - compute_shannon_entropy: Raw entropy in bits per byte
    - compute_leaf_entropy: Normalized entropy for a Merkle leaf
    - classify_leaf_entropy: Classify all leaves as low/high entropy
"""

import json
import math
from typing import Dict, Any

from .core import emit_receipt, dual_hash
from .constants import ENTROPY_PRUNE_THRESHOLD


def compute_shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy of data in bits per byte.

    Formula: H = -sum(p_i * log2(p_i)) for all byte values

    Args:
        data: Bytes to compute entropy for

    Returns:
        Entropy in bits per byte (0-8 range)
    """
    if not data:
        return 0.0

    # Count byte frequencies
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1

    # Compute entropy
    length = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 4)


def compute_leaf_entropy(leaf_data: Dict[str, Any]) -> float:
    """Compute entropy score for a single Merkle leaf.

    Uses Shannon entropy normalized by ln(n) factor.

    Args:
        leaf_data: Dict containing leaf content and metadata

    Returns:
        Normalized entropy score (0-1 range)
    """
    # Serialize leaf data
    content = json.dumps(leaf_data, sort_keys=True).encode()

    # Compute Shannon entropy (bits per byte, 0-8 range)
    raw_entropy = compute_shannon_entropy(content)

    # Normalize by theoretical max (8 bits for uniform distribution)
    # Then scale by ln(n) factor where n is content length
    n = max(1, len(content))
    ln_n = math.log(n)

    # Normalized score: higher is more information-dense
    normalized = (raw_entropy / 8.0) * (ln_n / max(1.0, ln_n))

    return round(normalized, 4)


def classify_leaf_entropy(
    merkle_tree: Dict[str, Any], threshold: float = ENTROPY_PRUNE_THRESHOLD
) -> Dict[str, Any]:
    """Classify all leaves in Merkle tree by entropy.

    Pure function. Returns classification for each leaf.

    Args:
        merkle_tree: Dict containing tree structure with leaves
        threshold: Entropy threshold for "low" classification (default: 0.1)

    Returns:
        Dict with leaf_id -> {entropy_score, classification ("low"/"high")}

    Receipt: leaf_entropy_receipt
    """
    leaves = merkle_tree.get("leaves", [])
    n_leaves = len(leaves)

    if n_leaves == 0:
        result = {
            "total_leaves": 0,
            "low_entropy_count": 0,
            "high_entropy_count": 0,
            "classification_threshold": threshold,
            "classifications": {},
        }
        emit_receipt(
            "leaf_entropy",
            {
                "tenant_id": "axiom-pruning",
                **result,
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )
        return result

    # Compute entropy for each leaf
    ln_n = math.log(max(1, n_leaves))
    adjusted_threshold = threshold * ln_n

    classifications = {}
    low_count = 0
    high_count = 0
    entropy_sum = 0.0

    for i, leaf in enumerate(leaves):
        leaf_id = leaf.get("id", f"leaf_{i}")
        entropy_score = compute_leaf_entropy(leaf)
        entropy_sum += entropy_score

        if entropy_score < adjusted_threshold:
            classification = "low"
            low_count += 1
        else:
            classification = "high"
            high_count += 1

        classifications[leaf_id] = {
            "entropy_score": entropy_score,
            "classification": classification,
        }

    result = {
        "total_leaves": n_leaves,
        "low_entropy_count": low_count,
        "high_entropy_count": high_count,
        "classification_threshold": threshold,
        "adjusted_threshold": round(adjusted_threshold, 4),
        "avg_entropy": round(entropy_sum / n_leaves, 4),
        "entropy_distribution": {
            "low_pct": round(low_count / n_leaves, 4),
            "high_pct": round(high_count / n_leaves, 4),
        },
        "classifications": classifications,
    }

    emit_receipt(
        "leaf_entropy",
        {
            "tenant_id": "axiom-pruning",
            "total_leaves": n_leaves,
            "low_entropy_count": low_count,
            "high_entropy_count": high_count,
            "classification_threshold": threshold,
            "entropy_distribution": result["entropy_distribution"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result
