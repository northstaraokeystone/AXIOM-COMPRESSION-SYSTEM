"""prove.py - Receipt Chain & Merkle Proofs

THE PROOF INFRASTRUCTURE:
    Every witness is verifiable.
    The merkle root is the cryptographic summary.
    Anyone can verify any witness without seeing all others.

KEY INSIGHT:
    The merkle root is tweet-sized proof of all experiments.
    "100 galaxies witnessed. Proof: 9f2a...1b47"
    Anyone with the receipts can verify.

The chain_receipt isn't metadata. It's the signature of truth.
Every witness leaves a trace. Every trace joins the chain. The chain is the proof.

Source: CLAUDEME.md (§8)
"""

import json
import math
import statistics
from typing import Tuple

from .core import dual_hash, emit_receipt, merkle, StopRule


# === CONSTANTS (Module Top) ===

TENANT_ID = "axiom-witness"
"""Receipt tenant isolation."""

REGIMES = ["newtonian", "mond", "nfw", "pbh_fog"]
"""Valid regime names for counting."""

COMPRESSION_THRESHOLD = 0.84
"""Floor for 'correct' classification."""

SIGNIFICANCE_THRESHOLD = 0.05
"""p-value threshold for statistical claims."""


# === STOPRULES ===

def stoprule_empty_chain() -> None:
    """Trigger stoprule for empty receipt chain.

    Emits anomaly receipt and raises StopRule.
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "chain",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt"
    })
    raise StopRule("Cannot chain empty receipt list")


def stoprule_missing_payload_hash(receipt_index: int) -> None:
    """Trigger stoprule for receipt missing payload_hash.

    Emits anomaly receipt and raises StopRule.

    Args:
        receipt_index: Index of the invalid receipt in the list
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "receipt_integrity",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "receipt_index": receipt_index
    })
    raise StopRule("Receipt missing payload_hash")


# === MERKLE TREE FUNCTIONS ===

def build_merkle_tree(items: list) -> Tuple[str, list]:
    """Build full merkle tree, returning root and all levels.

    Needed for proof path extraction.

    Args:
        items: List of receipt dicts

    Returns:
        Tuple of (root_hash: str, levels: list[list[str]])
        levels[0] = leaf hashes, levels[-1] = [root]

    Note: Does NOT emit receipt (internal utility).
    """
    if not items:
        empty_hash = dual_hash(b"empty")
        return empty_hash, [[empty_hash]]

    # Hash each item: level_0 = [dual_hash(json.dumps(item, sort_keys=True)) for item]
    level_0 = [dual_hash(json.dumps(item, sort_keys=True)) for item in items]
    levels = [level_0]

    # Build tree bottom-up
    current_level = level_0
    while len(current_level) > 1:
        # If odd count, duplicate last hash
        if len(current_level) % 2:
            current_level = current_level + [current_level[-1]]

        # Pair and hash: parent = dual_hash(left + right)
        next_level = []
        for i in range(0, len(current_level), 2):
            parent = dual_hash(current_level[i] + current_level[i + 1])
            next_level.append(parent)

        levels.append(next_level)
        current_level = next_level

    root = current_level[0] if current_level else dual_hash(b"empty")
    return root, levels


def prove_witness(witness_receipt: dict, chain: dict, all_receipts: list) -> dict:
    """Generate merkle proof path for single witness receipt.

    Args:
        witness_receipt: The receipt to prove
        chain: The chain_receipt (for root)
        all_receipts: All receipts in the chain (needed to reconstruct tree)

    Returns:
        proof_receipt dict with proof_path for verification

    Note: Does NOT emit receipt (proof is returned, not logged).
    """
    # Rebuild merkle tree from all_receipts
    root, levels = build_merkle_tree(all_receipts)

    # Hash the witness receipt to find its position
    witness_hash = dual_hash(json.dumps(witness_receipt, sort_keys=True))

    # Find position of witness_receipt in leaf level
    leaf_level = levels[0]
    try:
        leaf_index = leaf_level.index(witness_hash)
    except ValueError:
        # Receipt not found in tree
        return {
            "receipt_type": "proof",
            "tenant_id": TENANT_ID,
            "witness_hash": witness_hash,
            "merkle_root": chain.get("merkle_root", root),
            "proof_path": [],
            "leaf_index": -1,
            "payload_hash": dual_hash(f"{witness_hash}:not_found")
        }

    # Extract sibling hashes at each level (the proof path)
    proof_path = []
    current_index = leaf_index

    for level_idx, level in enumerate(levels[:-1]):  # Skip root level
        # Handle case where level was extended with duplicate
        if current_index >= len(level):
            current_index = len(level) - 1

        # Determine sibling position
        if current_index % 2 == 0:
            # Current is left child, sibling is right
            sibling_index = current_index + 1
            position = "right"
        else:
            # Current is right child, sibling is left
            sibling_index = current_index - 1
            position = "left"

        # Handle odd levels (last element is duplicated)
        if sibling_index >= len(level):
            sibling_index = len(level) - 1

        sibling_hash = level[sibling_index]
        proof_path.append({"sibling": sibling_hash, "position": position})

        # Move to parent index
        current_index = current_index // 2

    return {
        "receipt_type": "proof",
        "tenant_id": TENANT_ID,
        "witness_hash": witness_hash,
        "merkle_root": chain.get("merkle_root", root),
        "proof_path": proof_path,
        "leaf_index": leaf_index,
        "payload_hash": dual_hash(f"{witness_hash}:{root}")
    }


def verify_proof(witness_receipt: dict, proof: dict, root: str) -> bool:
    """Verify a witness receipt is in the chain using its proof path.

    Args:
        witness_receipt: The receipt being verified
        proof: The proof_receipt from prove_witness()
        root: The merkle root to verify against

    Returns:
        bool: True if computed root matches provided root, False otherwise

    Note: Does NOT emit receipt (pure verification).
    """
    # Hash the witness_receipt
    current_hash = dual_hash(json.dumps(witness_receipt, sort_keys=True))

    # Walk up the proof path
    for step in proof.get("proof_path", []):
        sibling = step["sibling"]
        position = step["position"]

        # Order determined by "position" field
        if position == "left":
            # Sibling is on the left
            combined = sibling + current_hash
        else:
            # Sibling is on the right
            combined = current_hash + sibling

        # Hash the combination
        current_hash = dual_hash(combined)

    # Compare final hash to root
    return current_hash == root


# === STATISTICS FUNCTIONS ===

def compute_chi_squared(observed: list, expected: list) -> Tuple[float, float]:
    """Compute chi-squared statistic and p-value for goodness-of-fit.

    Args:
        observed: Observed counts per category
        expected: Expected counts per category (null hypothesis)

    Returns:
        Tuple of (chi_squared: float, p_value: float)

    Note: Does NOT emit receipt (pure math).
    """
    if len(observed) != len(expected) or len(observed) == 0:
        return 0.0, 1.0

    # chi-squared = sum((O_i - E_i)^2 / E_i)
    chi_sq = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi_sq += (o - e) ** 2 / e

    # Degrees of freedom
    df = len(observed) - 1
    if df <= 0:
        return chi_sq, 1.0

    # Approximate p-value using chi-squared CDF
    # Using Wilson-Hilferty approximation for chi-squared distribution
    if chi_sq <= 0:
        return 0.0, 1.0

    # Wilson-Hilferty transformation: ((chi_sq/df)^(1/3) - (1 - 2/(9*df))) / sqrt(2/(9*df))
    # gives approximately standard normal z-score
    try:
        h = 2.0 / (9.0 * df)
        z = ((chi_sq / df) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)

        # Convert z-score to p-value using error function approximation
        # P(Z > z) for standard normal
        # Using approximation: 1 - 0.5 * (1 + erf(z / sqrt(2)))
        p_value = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
        p_value = max(0.0, min(1.0, p_value))
    except (ValueError, ZeroDivisionError):
        p_value = 0.5

    return chi_sq, p_value


def summarize_batch(receipts: list) -> dict:
    """Aggregate statistics from witness receipts.

    Args:
        receipts: List of witness_receipt dicts

    Returns:
        dict with accuracy metrics, compression stats, and chi-squared

    Note: Does NOT emit receipt (intermediate computation).
    """
    # Initialize counters
    by_regime = {
        "newtonian": {"total": 0, "correct": 0, "compressions": []},
        "mond": {"total": 0, "correct": 0, "compressions": []},
        "nfw": {"total": 0, "correct": 0, "compressions": []},
        "pbh_fog": {"total": 0, "correct": 0, "compressions": []}
    }

    all_compressions = []
    all_mse = []

    # Process each receipt
    for receipt in receipts:
        regime = receipt.get("physics_regime", "")
        compression = receipt.get("compression_ratio", 0.0)
        mse = receipt.get("final_mse", 0.0)

        # Normalize regime name
        if regime == "pbh":
            regime = "pbh_fog"

        if regime in by_regime:
            by_regime[regime]["total"] += 1
            by_regime[regime]["compressions"].append(compression)

            # Count "correct" if compression >= threshold
            if compression >= COMPRESSION_THRESHOLD:
                by_regime[regime]["correct"] += 1

        all_compressions.append(compression)
        if mse > 0:
            all_mse.append(mse)

    # Compute accuracies
    def accuracy(data: dict) -> float:
        if data["total"] == 0:
            return 0.0
        return data["correct"] / data["total"]

    newton_correct = accuracy(by_regime["newtonian"])
    mond_correct = accuracy(by_regime["mond"])
    nfw_correct = accuracy(by_regime["nfw"])
    pbh_fog_correct = accuracy(by_regime["pbh_fog"])

    # Determine pbh_novel: True if any pbh_fog receipt has compression > best nfw compression
    pbh_compressions = by_regime["pbh_fog"]["compressions"]
    nfw_compressions = by_regime["nfw"]["compressions"]

    pbh_novel = False
    if pbh_compressions and nfw_compressions:
        max_pbh = max(pbh_compressions)
        max_nfw = max(nfw_compressions)
        pbh_novel = max_pbh > max_nfw
    elif pbh_compressions and not nfw_compressions:
        pbh_novel = True

    # Compression stats
    if all_compressions:
        compression_mean = statistics.mean(all_compressions)
        compression_std = statistics.stdev(all_compressions) if len(all_compressions) > 1 else 0.0
        compression_min = min(all_compressions)
        compression_max = max(all_compressions)
    else:
        compression_mean = 0.0
        compression_std = 0.0
        compression_min = 0.0
        compression_max = 0.0

    # MSE stats
    if all_mse:
        mse_mean = statistics.mean(all_mse)
        mse_std = statistics.stdev(all_mse) if len(all_mse) > 1 else 0.0
    else:
        mse_mean = 0.0
        mse_std = 0.0

    # Chi-squared for regime classification
    observed = [
        by_regime["newtonian"]["correct"],
        by_regime["mond"]["correct"],
        by_regime["nfw"]["correct"],
        by_regime["pbh_fog"]["correct"]
    ]

    total_count = sum(r["total"] for r in by_regime.values())
    if total_count > 0:
        expected_per = total_count / 4.0
        expected = [expected_per * COMPRESSION_THRESHOLD] * 4
    else:
        expected = [0.0, 0.0, 0.0, 0.0]

    chi_sq, p_value = compute_chi_squared(observed, expected)

    # Build by_regime output (without compressions list)
    by_regime_output = {}
    for regime in REGIMES:
        data = by_regime[regime]
        by_regime_output[regime] = {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": accuracy(data)
        }

    return {
        "newton_correct": newton_correct,
        "mond_correct": mond_correct,
        "nfw_correct": nfw_correct,
        "pbh_fog_correct": pbh_fog_correct,
        "pbh_novel": pbh_novel,
        "total_count": total_count,
        "by_regime": by_regime_output,
        "compression_stats": {
            "mean": compression_mean,
            "std": compression_std,
            "min": compression_min,
            "max": compression_max
        },
        "mse_stats": {
            "mean": mse_mean,
            "std": mse_std
        },
        "chi_squared": chi_sq,
        "p_value": p_value,
        "degrees_of_freedom": len(REGIMES) - 1
    }


# === MAIN ENTRY POINT ===

def chain_receipts(receipts: list) -> dict:
    """MAIN ENTRY POINT for batch proof.

    Compute merkle root of all receipts, generate summary statistics,
    emit chain_receipt.

    Args:
        receipts: List of receipt dicts (witness_receipts, topology_receipts, or mixed)

    Returns:
        The chain_receipt dict

    MUST emit receipt (chain_receipt).
    """
    # Stoprule: empty chain
    if not receipts:
        stoprule_empty_chain()

    # Stoprule: missing payload_hash
    for i, receipt in enumerate(receipts):
        if "payload_hash" not in receipt:
            stoprule_missing_payload_hash(i)

    # Extract payload_hash from each receipt and compute merkle root
    root = merkle(receipts)

    # Summarize batch statistics
    summary = summarize_batch(receipts)

    # Build chain_receipt
    chain_receipt = emit_receipt("chain", {
        "tenant_id": TENANT_ID,
        "n_witnesses": len(receipts),
        "merkle_root": root,
        "summary": {
            "newton_correct": summary["newton_correct"],
            "mond_correct": summary["mond_correct"],
            "nfw_correct": summary["nfw_correct"],
            "pbh_novel": summary["pbh_novel"]
        }
    })

    return chain_receipt


# === FORMATTING FUNCTIONS ===

def format_for_publication(summary: dict, chain: dict) -> str:
    """Generate academic-ready text for arXiv/publication.

    Args:
        summary: From summarize_batch()
        chain: The chain_receipt

    Returns:
        str: Multi-line formatted text

    Note: Does NOT emit receipt (formatting utility).
    """
    root = chain.get("merkle_root", "unknown")
    n_witnesses = chain.get("n_witnesses", summary.get("total_count", 0))

    by_regime = summary.get("by_regime", {})
    newton_n = by_regime.get("newtonian", {}).get("total", 0)
    mond_n = by_regime.get("mond", {}).get("total", 0)
    nfw_n = by_regime.get("nfw", {}).get("total", 0)
    pbh_n = by_regime.get("pbh_fog", {}).get("total", 0)

    compression_stats = summary.get("compression_stats", {})
    mean_comp = compression_stats.get("mean", 0.0)
    std_comp = compression_stats.get("std", 0.0)
    min_comp = compression_stats.get("min", 0.0)
    max_comp = compression_stats.get("max", 0.0)

    pbh_novel = summary.get("pbh_novel", False)
    chi_sq = summary.get("chi_squared", 0.0)
    p_value = summary.get("p_value", 1.0)
    df = summary.get("degrees_of_freedom", 3)

    # Compute delta between pbh and nfw
    pbh_correct = summary.get("pbh_fog_correct", 0.0)
    nfw_correct = summary.get("nfw_correct", 0.0)
    delta = (pbh_correct - nfw_correct) * 100  # percentage points

    significant = "Significant" if p_value < SIGNIFICANCE_THRESHOLD else "Not significant"
    novel_status = "NOVEL SIGNAL DETECTED" if pbh_novel else "No significant difference"

    # Truncate root for display
    root_short = f"{root[:16]}...{root[-8:]}" if len(root) > 24 else root

    return f"""AXIOM Witness Protocol Results
==============================

Dataset: {n_witnesses} synthetic galaxies
Merkle Root: {root_short}

Classification Accuracy by Regime:
- Newtonian: {summary.get('newton_correct', 0.0):.1%} (n={newton_n})
- MOND: {summary.get('mond_correct', 0.0):.1%} (n={mond_n})
- NFW Dark Matter: {summary.get('nfw_correct', 0.0):.1%} (n={nfw_n})
- PBH Fog: {summary.get('pbh_fog_correct', 0.0):.1%} (n={pbh_n})

Compression Statistics:
- Mean: {mean_comp:.3f} +/- {std_comp:.3f}
- Range: [{min_comp:.3f}, {max_comp:.3f}]

Novel Physics Detection:
- PBH fog compression vs NFW: {delta:+.1f}%
- Status: {novel_status}

Statistical Significance:
- chi^2({df}) = {chi_sq:.2f}, p = {p_value:.4f}
- Result: {significant} at alpha=0.05

Verification:
All results verifiable against Merkle root using prove.verify_proof()
"""


def format_for_tweet(summary: dict, chain: dict) -> str:
    """Generate tweet-sized summary for X/Grok.

    Args:
        summary: From summarize_batch()
        chain: The chain_receipt

    Returns:
        str: Tweet-length text (<=280 chars)

    Note: Does NOT emit receipt (formatting utility).
    """
    n = summary.get("total_count", chain.get("n_witnesses", 0))
    root = chain.get("merkle_root", "unknown")

    newton = summary.get("newton_correct", 0.0)
    mond = summary.get("mond_correct", 0.0)
    nfw = summary.get("nfw_correct", 0.0)

    pbh_novel = summary.get("pbh_novel", False)
    pbh_status = "NOVEL" if pbh_novel else "nominal"

    # Truncate root
    root_short = f"{root[:8]}...{root[-4:]}" if len(root) > 12 else root

    tweet = f"""AXIOM WITNESS: {n} galaxies
Newton {newton:.0%} | MOND {mond:.0%} | NFW {nfw:.0%}
PBH fog: {pbh_status}
Proof: {root_short}"""

    # Ensure <= 280 chars
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
