"""Tests for spaceproof.anchor module."""

from spaceproof.anchor import (
    create_proof,
    verify_proof,
    anchor_batch,
    verify_batch,
    chain_receipts,
)


def test_anchor_batch_empty():
    """anchor_batch handles empty list."""
    result = anchor_batch([])
    assert result.item_count == 0
    assert result.root != ""


def test_anchor_batch_single():
    """anchor_batch handles single item."""
    items = [{"id": 1, "data": "test"}]
    result = anchor_batch(items)
    assert result.item_count == 1
    assert result.root != ""


def test_anchor_batch_multiple():
    """anchor_batch handles multiple items."""
    items = [{"id": i} for i in range(5)]
    result = anchor_batch(items)
    assert result.item_count == 5
    assert len(result.proofs) == 5


def test_create_proof_valid():
    """create_proof returns valid proof."""
    items = [{"a": 1}, {"b": 2}, {"c": 3}]
    proof = create_proof(items[1], items)
    assert proof.index == 1
    assert proof.root != ""


def test_verify_proof_valid():
    """verify_proof returns True for valid proof."""
    items = [{"x": 1}, {"y": 2}]
    result = anchor_batch(items)
    proof = create_proof(items[0], items)

    assert verify_proof(items[0], proof, result.root) is True


def test_verify_proof_invalid():
    """verify_proof returns False for tampered item."""
    items = [{"x": 1}, {"y": 2}]
    result = anchor_batch(items)
    proof = create_proof(items[0], items)

    # Tamper with item
    tampered = {"x": 999}
    assert verify_proof(tampered, proof, result.root) is False


def test_verify_batch_valid():
    """verify_batch returns True for valid batch."""
    items = [{"id": i} for i in range(3)]
    result = anchor_batch(items)

    verification = verify_batch(items, result.root)
    assert verification["verified"] is True


def test_chain_receipts():
    """chain_receipts emits chain receipt."""
    receipts = [
        {"receipt_type": "test", "data": "a"},
        {"receipt_type": "test", "data": "b"},
    ]
    chain = chain_receipts(receipts)
    assert chain["n_receipts"] == 2
    assert "merkle_root" in chain
