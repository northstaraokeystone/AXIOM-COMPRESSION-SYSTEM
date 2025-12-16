"""test_prove.py - Tests for receipt chain and merkle proof verification.

Source: CLAUDEME.md LAW_2 - No test -> not shipped
"""

import math
import pytest

from src.prove import (
    TENANT_ID,
    REGIMES,
    COMPRESSION_THRESHOLD,
    SIGNIFICANCE_THRESHOLD,
    build_merkle_tree,
    chain_receipts,
    prove_witness,
    verify_proof,
    summarize_batch,
    compute_chi_squared,
    format_for_publication,
    format_for_tweet,
    stoprule_empty_chain,
    stoprule_missing_payload_hash,
)
from src.core import StopRule, dual_hash


# === MERKLE TREE TESTS ===

def test_build_merkle_tree_single():
    """Tree with 1 item returns that item's hash as root."""
    items = [{"a": 1, "payload_hash": "test_hash"}]
    root, levels = build_merkle_tree(items)

    # Root should be the hash of the single item
    expected_leaf = dual_hash('{"a": 1, "payload_hash": "test_hash"}')
    assert root == expected_leaf
    assert len(levels) == 1
    assert levels[0] == [expected_leaf]


def test_build_merkle_tree_power_of_two():
    """4 items produces 3-level tree."""
    items = [
        {"id": 1, "payload_hash": "h1"},
        {"id": 2, "payload_hash": "h2"},
        {"id": 3, "payload_hash": "h3"},
        {"id": 4, "payload_hash": "h4"},
    ]
    root, levels = build_merkle_tree(items)

    # Should have 3 levels: 4 leaves -> 2 parents -> 1 root
    assert len(levels) == 3
    assert len(levels[0]) == 4  # Leaves
    assert len(levels[1]) == 2  # Parents
    assert len(levels[2]) == 1  # Root
    assert root == levels[2][0]


def test_build_merkle_tree_odd():
    """5 items handles duplication correctly."""
    items = [
        {"id": i, "payload_hash": f"h{i}"}
        for i in range(5)
    ]
    root, levels = build_merkle_tree(items)

    # 5 leaves -> (dup to 6) -> 3 -> (dup to 4) -> 2 -> 1
    # So we should have 4 levels
    assert len(levels) >= 3
    assert len(levels[0]) == 5  # Original 5 leaves
    # Tree should build correctly despite odd count
    assert root is not None
    assert len(root) > 0


def test_build_merkle_tree_empty():
    """Empty tree returns hash of 'empty'."""
    root, levels = build_merkle_tree([])
    expected = dual_hash(b"empty")
    assert root == expected


# === CHAIN RECEIPTS TESTS ===

def test_chain_receipts_emits():
    """'chain' in result and receipt_type correct (CLAUDEME LAW_1)."""
    receipts = [
        {"receipt_type": "witness", "physics_regime": "newtonian", "compression_ratio": 0.94, "payload_hash": "h1"},
        {"receipt_type": "witness", "physics_regime": "mond", "compression_ratio": 0.91, "payload_hash": "h2"},
    ]
    chain = chain_receipts(receipts)

    assert chain["receipt_type"] == "chain"
    assert "merkle_root" in chain
    assert "n_witnesses" in chain
    assert chain["n_witnesses"] == 2
    assert "summary" in chain
    assert chain["tenant_id"] == TENANT_ID


def test_chain_receipts_root_deterministic():
    """Same receipts -> same root (reproducibility)."""
    receipts = [
        {"receipt_type": "witness", "physics_regime": "newtonian", "compression_ratio": 0.90, "payload_hash": "h1"},
        {"receipt_type": "witness", "physics_regime": "mond", "compression_ratio": 0.88, "payload_hash": "h2"},
    ]

    chain1 = chain_receipts(receipts)
    chain2 = chain_receipts(receipts)

    assert chain1["merkle_root"] == chain2["merkle_root"]


# === PROOF TESTS ===

def test_prove_witness_path_length():
    """Path length = log2(n) rounded up (tree depth)."""
    # 4 items -> log2(4) = 2 levels of proof
    items = [
        {"payload_hash": "h1"},
        {"payload_hash": "h2"},
        {"payload_hash": "h3"},
        {"payload_hash": "h4"},
    ]
    root, _ = build_merkle_tree(items)
    chain = {"merkle_root": root}

    proof = prove_witness(items[1], chain, items)

    # For 4 items (2^2), we need 2 siblings to reach root
    assert len(proof["proof_path"]) == 2


def test_prove_witness_path_length_8_items():
    """Path length for 8 items should be 3."""
    items = [{"payload_hash": f"h{i}"} for i in range(8)]
    root, _ = build_merkle_tree(items)
    chain = {"merkle_root": root}

    proof = prove_witness(items[0], chain, items)

    # For 8 items (2^3), we need 3 siblings
    assert len(proof["proof_path"]) == 3


def test_verify_proof_valid():
    """verify_proof returns True for valid proof (core function)."""
    items = [
        {"payload_hash": "h1"},
        {"payload_hash": "h2"},
    ]
    root, _ = build_merkle_tree(items)
    chain = {"merkle_root": root}

    proof = prove_witness(items[0], chain, items)
    assert verify_proof(items[0], proof, root) is True


def test_verify_proof_tampered():
    """verify_proof returns False if witness altered (security)."""
    items = [
        {"payload_hash": "h1"},
        {"payload_hash": "h2"},
    ]
    root, _ = build_merkle_tree(items)
    chain = {"merkle_root": root}

    proof = prove_witness(items[0], chain, items)

    # Tamper with the witness
    tampered_witness = {"payload_hash": "h1_tampered"}
    assert verify_proof(tampered_witness, proof, root) is False


def test_verify_proof_wrong_root():
    """verify_proof returns False for wrong root (security)."""
    items = [
        {"payload_hash": "h1"},
        {"payload_hash": "h2"},
    ]
    root, _ = build_merkle_tree(items)
    chain = {"merkle_root": root}

    proof = prove_witness(items[0], chain, items)

    # Use wrong root
    wrong_root = dual_hash(b"wrong_root")
    assert verify_proof(items[0], proof, wrong_root) is False


def test_verify_proof_multiple_items():
    """verify_proof works for all items in a larger tree."""
    items = [{"payload_hash": f"h{i}", "id": i} for i in range(7)]
    root, _ = build_merkle_tree(items)
    chain = {"merkle_root": root}

    # Verify each item
    for item in items:
        proof = prove_witness(item, chain, items)
        assert verify_proof(item, proof, root) is True, f"Failed for item {item}"


# === SUMMARIZE BATCH TESTS ===

def test_summarize_batch_counts():
    """Correct counts per regime (statistics)."""
    receipts = [
        {"physics_regime": "newtonian", "compression_ratio": 0.90, "payload_hash": "h1"},
        {"physics_regime": "newtonian", "compression_ratio": 0.85, "payload_hash": "h2"},
        {"physics_regime": "mond", "compression_ratio": 0.80, "payload_hash": "h3"},
        {"physics_regime": "nfw", "compression_ratio": 0.86, "payload_hash": "h4"},
    ]
    summary = summarize_batch(receipts)

    assert summary["by_regime"]["newtonian"]["total"] == 2
    assert summary["by_regime"]["mond"]["total"] == 1
    assert summary["by_regime"]["nfw"]["total"] == 1
    assert summary["by_regime"]["pbh_fog"]["total"] == 0
    assert summary["total_count"] == 4


def test_summarize_batch_correct_counts():
    """Correct classification based on COMPRESSION_THRESHOLD."""
    receipts = [
        {"physics_regime": "newtonian", "compression_ratio": 0.90, "payload_hash": "h1"},  # Above threshold
        {"physics_regime": "newtonian", "compression_ratio": 0.80, "payload_hash": "h2"},  # Below threshold
        {"physics_regime": "mond", "compression_ratio": 0.84, "payload_hash": "h3"},  # At threshold
    ]
    summary = summarize_batch(receipts)

    # 0.90 >= 0.84 -> correct
    # 0.80 < 0.84 -> incorrect
    assert summary["by_regime"]["newtonian"]["correct"] == 1

    # 0.84 >= 0.84 -> correct
    assert summary["by_regime"]["mond"]["correct"] == 1


def test_summarize_batch_pbh_novel():
    """pbh_novel True when pbh > nfw compression (detection)."""
    receipts = [
        {"physics_regime": "nfw", "compression_ratio": 0.85, "payload_hash": "h1"},
        {"physics_regime": "pbh_fog", "compression_ratio": 0.90, "payload_hash": "h2"},  # Better than NFW
    ]
    summary = summarize_batch(receipts)

    assert summary["pbh_novel"] is True


def test_summarize_batch_pbh_not_novel():
    """pbh_novel False when pbh <= nfw compression."""
    receipts = [
        {"physics_regime": "nfw", "compression_ratio": 0.90, "payload_hash": "h1"},
        {"physics_regime": "pbh_fog", "compression_ratio": 0.85, "payload_hash": "h2"},  # Worse than NFW
    ]
    summary = summarize_batch(receipts)

    assert summary["pbh_novel"] is False


def test_summarize_batch_compression_stats():
    """Compression statistics are computed correctly."""
    receipts = [
        {"physics_regime": "newtonian", "compression_ratio": 0.80, "payload_hash": "h1"},
        {"physics_regime": "newtonian", "compression_ratio": 0.90, "payload_hash": "h2"},
        {"physics_regime": "newtonian", "compression_ratio": 1.00, "payload_hash": "h3"},
    ]
    summary = summarize_batch(receipts)

    stats = summary["compression_stats"]
    assert stats["min"] == 0.80
    assert stats["max"] == 1.00
    assert abs(stats["mean"] - 0.90) < 0.01


# === CHI-SQUARED TESTS ===

def test_compute_chi_squared_perfect_match():
    """Chi-squared is 0 when observed matches expected."""
    observed = [25, 25, 25, 25]
    expected = [25.0, 25.0, 25.0, 25.0]
    chi_sq, p_value = compute_chi_squared(observed, expected)

    assert chi_sq == 0.0
    # p-value should be high (close to 1) for perfect match


def test_compute_chi_squared_nonzero():
    """Chi-squared is positive when observed differs from expected."""
    observed = [30, 20, 30, 20]
    expected = [25.0, 25.0, 25.0, 25.0]
    chi_sq, p_value = compute_chi_squared(observed, expected)

    assert chi_sq > 0.0


def test_compute_chi_squared_empty():
    """Empty lists return 0, 1."""
    chi_sq, p_value = compute_chi_squared([], [])
    assert chi_sq == 0.0
    assert p_value == 1.0


# === FORMAT TESTS ===

def test_format_for_publication_contains_root():
    """Merkle root in output (formatting)."""
    summary = {
        "newton_correct": 0.96,
        "mond_correct": 0.92,
        "nfw_correct": 0.84,
        "pbh_fog_correct": 0.88,
        "pbh_novel": True,
        "total_count": 100,
        "by_regime": {
            "newtonian": {"total": 25, "correct": 24, "accuracy": 0.96},
            "mond": {"total": 25, "correct": 23, "accuracy": 0.92},
            "nfw": {"total": 25, "correct": 21, "accuracy": 0.84},
            "pbh_fog": {"total": 25, "correct": 22, "accuracy": 0.88},
        },
        "compression_stats": {"mean": 0.90, "std": 0.05, "min": 0.80, "max": 0.99},
        "mse_stats": {"mean": 0.01, "std": 0.005},
        "chi_squared": 5.2,
        "p_value": 0.16,
        "degrees_of_freedom": 3
    }
    chain = {"merkle_root": "abc123def456789012345678901234567890", "n_witnesses": 100}

    output = format_for_publication(summary, chain)

    assert "abc123def456789" in output  # Root prefix
    assert "100" in output  # Galaxy count
    assert "Newtonian" in output
    assert "MOND" in output
    assert "NFW" in output
    assert "PBH" in output
    assert "Merkle Root:" in output


def test_format_for_tweet_length():
    """len(output) <= 280 (Twitter constraint)."""
    summary = {
        "newton_correct": 0.96,
        "mond_correct": 0.92,
        "nfw_correct": 0.84,
        "pbh_novel": True,
        "total_count": 100
    }
    chain = {"merkle_root": "9f2a1b47c821", "n_witnesses": 100}

    tweet = format_for_tweet(summary, chain)

    assert len(tweet) <= 280


def test_format_for_tweet_content():
    """Tweet contains essential information."""
    summary = {
        "newton_correct": 0.96,
        "mond_correct": 0.92,
        "nfw_correct": 0.84,
        "pbh_novel": True,
        "total_count": 100
    }
    chain = {"merkle_root": "9f2a1b47c821deadbeef", "n_witnesses": 100}

    tweet = format_for_tweet(summary, chain)

    assert "AXIOM" in tweet
    assert "100" in tweet
    assert "Newton" in tweet
    assert "NOVEL" in tweet  # pbh_novel is True


# === STOPRULE TESTS ===

def test_empty_chain_stoprule():
    """StopRule raised for empty chain (safety)."""
    with pytest.raises(StopRule):
        chain_receipts([])


def test_missing_payload_hash_stoprule():
    """StopRule raised for receipt missing payload_hash."""
    receipts = [
        {"receipt_type": "witness", "physics_regime": "newtonian"}  # No payload_hash
    ]
    with pytest.raises(StopRule):
        chain_receipts(receipts)


# === INTEGRATION TESTS ===

def test_full_pipeline_witness_to_proof():
    """Full pipeline: generate receipts, chain, prove, verify."""
    # Create mock witness receipts
    receipts = [
        {"receipt_type": "witness", "physics_regime": "newtonian", "compression_ratio": 0.95, "payload_hash": "h1", "galaxy_id": "g1"},
        {"receipt_type": "witness", "physics_regime": "mond", "compression_ratio": 0.92, "payload_hash": "h2", "galaxy_id": "g2"},
        {"receipt_type": "witness", "physics_regime": "nfw", "compression_ratio": 0.88, "payload_hash": "h3", "galaxy_id": "g3"},
        {"receipt_type": "witness", "physics_regime": "pbh_fog", "compression_ratio": 0.91, "payload_hash": "h4", "galaxy_id": "g4"},
    ]

    # Chain receipts
    chain = chain_receipts(receipts)
    assert chain["receipt_type"] == "chain"
    assert chain["n_witnesses"] == 4

    # Get root from chain
    root = chain["merkle_root"]

    # Prove each witness and verify
    for receipt in receipts:
        proof = prove_witness(receipt, chain, receipts)
        assert verify_proof(receipt, proof, root) is True, f"Verification failed for {receipt['galaxy_id']}"


def test_constants_values():
    """Verify constants have expected values."""
    assert TENANT_ID == "axiom-witness"
    assert REGIMES == ["newtonian", "mond", "nfw", "pbh_fog"]
    assert COMPRESSION_THRESHOLD == 0.84
    assert SIGNIFICANCE_THRESHOLD == 0.05
