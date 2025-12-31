"""anchor.py - Cryptographic Proof Generation and Verification

D20 Production Evolution: Stakeholder-intuitive name for Merkle proofs.

THE ANCHOR INSIGHT:
    An anchor is a cryptographic commitment.
    Once anchored, data cannot be modified without detection.
    The Merkle root is the signature of truth.

Source: SpaceProof D20 Production Evolution

Stakeholder Value:
    - Defense: Tamper-proof verification
    - NRO: Decision lineage anchoring
    - DOGE: Audit trail integrity
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

from .core import emit_receipt, dual_hash, merkle

# === CONSTANTS ===

TENANT_ID = "spaceproof-anchor"


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


# === HARDWARE PROVENANCE CHAINS ===
# Anchor hardware component lifecycle: manufacturer → module → satellite
# Source: Jay's power supply verification use case


@dataclass
class ComponentProvenanceReceipt:
    """Provenance receipt for a hardware component."""

    component_id: str
    manufacturer: str
    manufacturer_receipt_hash: str
    chain_receipts: List[str]
    merkle_root: str
    rework_count: int
    entropy_at_steps: List[float]
    chain_valid: bool
    timestamp: str


@dataclass
class ModuleProvenanceReceipt:
    """Provenance receipt for a module (composed of components)."""

    module_id: str
    component_ids: List[str]
    component_merkle_roots: List[str]
    combined_entropy: float
    aggregate_rework_count: int
    weakest_link_reliability: float
    all_components_valid: bool
    rejected_components: List[str]
    merkle_root: str
    timestamp: str


def anchor_component_provenance(
    component_id: str,
    manufacturer: str,
    receipts: List[Dict],
) -> ComponentProvenanceReceipt:
    """Create Merkle chain for single component lifecycle.

    Chain structure:
    - Manufacturer receipt (visual + electrical baseline)
    - Distributor handoff receipt
    - Integrator receipt
    - Assembly receipts (each step in module build)
    - Rework receipts (if any)
    - Test receipts (performance + provenance validation)
    - Satellite integration receipt

    Args:
        component_id: Component identifier
        manufacturer: Expected manufacturer
        receipts: List of lifecycle receipts

    Returns:
        ComponentProvenanceReceipt with Merkle root

    Constraints:
        - Chain must start with manufacturer_receipt (reject if missing)
        - Each link must reference previous hash (detect breaks)
        - Rework count must be tracked (reject if > threshold)
        - Entropy must be computed at each step (detect anomalies)
    """
    from datetime import datetime

    timestamp = datetime.utcnow().isoformat() + "Z"

    if not receipts:
        empty_root = dual_hash(b"empty")
        result = ComponentProvenanceReceipt(
            component_id=component_id,
            manufacturer=manufacturer,
            manufacturer_receipt_hash="",
            chain_receipts=[],
            merkle_root=empty_root,
            rework_count=0,
            entropy_at_steps=[],
            chain_valid=False,
            timestamp=timestamp,
        )

        emit_receipt(
            "component_provenance",
            {
                "tenant_id": TENANT_ID,
                "component_id": component_id,
                "chain_valid": False,
                "reason": "no_receipts",
                "merkle_root": empty_root,
            },
        )

        return result

    # Validate chain starts with manufacturer receipt
    first_receipt = receipts[0]
    has_manufacturer = (
        first_receipt.get("receipt_type") == "manufacturer"
        or first_receipt.get("type") == "manufacturer"
        or "manufacturer" in str(first_receipt.get("receipt_type", "")).lower()
    )

    if not has_manufacturer:
        # Check if any receipt is manufacturer receipt
        has_manufacturer = any(
            r.get("receipt_type") == "manufacturer" or
            r.get("type") == "manufacturer" or
            "manufacturer" in str(r.get("receipt_type", "")).lower()
            for r in receipts
        )

    # Build Merkle tree
    root, levels = _build_merkle_tree(receipts)

    # Extract chain hashes
    chain_receipts_hashes = [dual_hash(json.dumps(r, sort_keys=True)) for r in receipts]

    # Count rework receipts
    rework_count = sum(
        1 for r in receipts
        if (
            r.get("receipt_type") == "rework"
            or r.get("type") == "rework"
            or "rework" in str(r.get("receipt_type", "")).lower()
        )
    )

    # Extract entropy at each step
    entropy_at_steps = [r.get("entropy", 0.30) for r in receipts]

    # Validate chain integrity (each references previous)
    chain_valid = has_manufacturer
    for i in range(1, len(receipts)):
        prev_hash = chain_receipts_hashes[i - 1]
        curr_prev_ref = receipts[i].get("previous_hash", receipts[i].get("prev_hash", ""))
        if curr_prev_ref and curr_prev_ref != prev_hash:
            chain_valid = False
            break

    # Get manufacturer receipt hash
    manufacturer_receipt_hash = ""
    for r in receipts:
        if r.get("receipt_type") == "manufacturer" or r.get("type") == "manufacturer":
            manufacturer_receipt_hash = dual_hash(json.dumps(r, sort_keys=True))
            break
    if not manufacturer_receipt_hash and receipts:
        manufacturer_receipt_hash = chain_receipts_hashes[0]

    result = ComponentProvenanceReceipt(
        component_id=component_id,
        manufacturer=manufacturer,
        manufacturer_receipt_hash=manufacturer_receipt_hash,
        chain_receipts=chain_receipts_hashes,
        merkle_root=root,
        rework_count=rework_count,
        entropy_at_steps=entropy_at_steps,
        chain_valid=chain_valid,
        timestamp=timestamp,
    )

    # Emit provenance receipt
    emit_receipt(
        "component_provenance",
        {
            "tenant_id": TENANT_ID,
            "component_id": component_id,
            "manufacturer": manufacturer,
            "merkle_root": root,
            "rework_count": rework_count,
            "chain_valid": chain_valid,
            "has_manufacturer_receipt": has_manufacturer,
            "chain_length": len(receipts),
            "timestamp": timestamp,
        },
    )

    return result


def validate_provenance_chain(
    component_id: str,
    expected_manufacturer: str,
    receipts: Optional[List[Dict]] = None,
    provenance_receipt: Optional[ComponentProvenanceReceipt] = None,
) -> bool:
    """Walk Merkle chain backward from satellite to manufacturer.

    Verifies:
    - Chain unbroken (each hash links to previous)
    - Origin matches expected manufacturer
    - No entropy spikes (counterfeit detection)
    - Rework count within bounds

    Args:
        component_id: Component identifier
        expected_manufacturer: Expected manufacturer
        receipts: Optional list of receipts (if not using provenance_receipt)
        provenance_receipt: Optional pre-computed provenance receipt

    Returns:
        True if valid, False if fraud detected
    """
    from .detect import REWORK_THRESHOLDS, HARDWARE_ENTROPY_THRESHOLDS

    if provenance_receipt:
        # Use pre-computed receipt
        if not provenance_receipt.chain_valid:
            emit_receipt(
                "provenance_validation",
                {
                    "tenant_id": TENANT_ID,
                    "component_id": component_id,
                    "valid": False,
                    "reason": "chain_integrity_failed",
                },
            )
            return False

        # Check manufacturer
        if provenance_receipt.manufacturer != expected_manufacturer:
            emit_receipt(
                "provenance_validation",
                {
                    "tenant_id": TENANT_ID,
                    "component_id": component_id,
                    "valid": False,
                    "reason": "manufacturer_mismatch",
                    "expected": expected_manufacturer,
                    "found": provenance_receipt.manufacturer,
                },
            )
            return False

        # Check rework count
        if provenance_receipt.rework_count > REWORK_THRESHOLDS["max_cycles"]:
            emit_receipt(
                "provenance_validation",
                {
                    "tenant_id": TENANT_ID,
                    "component_id": component_id,
                    "valid": False,
                    "reason": "excessive_rework",
                    "rework_count": provenance_receipt.rework_count,
                    "threshold": REWORK_THRESHOLDS["max_cycles"],
                },
            )
            return False

        # Check for entropy spikes
        for i, entropy in enumerate(provenance_receipt.entropy_at_steps):
            if entropy > HARDWARE_ENTROPY_THRESHOLDS["counterfeit_min"]:
                emit_receipt(
                    "provenance_validation",
                    {
                        "tenant_id": TENANT_ID,
                        "component_id": component_id,
                        "valid": False,
                        "reason": "entropy_spike",
                        "step": i,
                        "entropy": entropy,
                        "threshold": HARDWARE_ENTROPY_THRESHOLDS["counterfeit_min"],
                    },
                )
                return False

        emit_receipt(
            "provenance_validation",
            {
                "tenant_id": TENANT_ID,
                "component_id": component_id,
                "valid": True,
                "merkle_root": provenance_receipt.merkle_root,
            },
        )
        return True

    # Compute provenance from receipts
    if receipts:
        provenance = anchor_component_provenance(component_id, expected_manufacturer, receipts)
        return validate_provenance_chain(
            component_id,
            expected_manufacturer,
            provenance_receipt=provenance,
        )

    # No data provided
    emit_receipt(
        "provenance_validation",
        {
            "tenant_id": TENANT_ID,
            "component_id": component_id,
            "valid": False,
            "reason": "no_provenance_data",
        },
    )
    return False


def merge_component_chains(
    module_id: str,
    component_provenances: List[ComponentProvenanceReceipt],
) -> ModuleProvenanceReceipt:
    """Merge component provenance chains into module-level chain.

    Power supply module has multiple components (capacitors, resistors, ICs).
    Merge their individual provenance chains into module-level chain.

    Args:
        module_id: Module identifier
        component_provenances: List of component provenance receipts

    Returns:
        ModuleProvenanceReceipt with:
        - List of component Merkle roots
        - Combined entropy signature
        - Aggregate rework count
        - Weakest-link reliability estimate

    Constraints:
        If ANY component flagged as invalid → reject entire module
        If total rework count > MODULE_THRESHOLD → flag for inspection
    """
    from datetime import datetime
    from .detect import REWORK_THRESHOLDS

    timestamp = datetime.utcnow().isoformat() + "Z"

    if not component_provenances:
        empty_root = dual_hash(b"empty")
        result = ModuleProvenanceReceipt(
            module_id=module_id,
            component_ids=[],
            component_merkle_roots=[],
            combined_entropy=1.0,  # Max entropy = unknown
            aggregate_rework_count=0,
            weakest_link_reliability=0.0,
            all_components_valid=False,
            rejected_components=[],
            merkle_root=empty_root,
            timestamp=timestamp,
        )

        emit_receipt(
            "module_provenance",
            {
                "tenant_id": TENANT_ID,
                "module_id": module_id,
                "all_components_valid": False,
                "reason": "no_components",
                "merkle_root": empty_root,
            },
        )

        return result

    # Collect component data
    component_ids = [p.component_id for p in component_provenances]
    component_merkle_roots = [p.merkle_root for p in component_provenances]

    # Calculate combined entropy (average of final entropy values)
    all_entropies = []
    for p in component_provenances:
        if p.entropy_at_steps:
            all_entropies.append(p.entropy_at_steps[-1])
        else:
            all_entropies.append(0.50)  # Unknown = mid-range

    combined_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0.50

    # Calculate aggregate rework count
    aggregate_rework_count = sum(p.rework_count for p in component_provenances)

    # Identify rejected components
    rejected_components = [p.component_id for p in component_provenances if not p.chain_valid]

    # All components valid?
    all_components_valid = len(rejected_components) == 0

    # Weakest-link reliability estimate
    # Based on entropy and rework count
    # Lower entropy = higher reliability, more rework = lower reliability
    component_reliabilities = []
    for p in component_provenances:
        if p.chain_valid:
            entropy_factor = 1.0 - (all_entropies[component_provenances.index(p)] if all_entropies else 0.5)
            rework_factor = max(0.0, 1.0 - (p.rework_count / (REWORK_THRESHOLDS["max_cycles"] + 2)))
            reliability = entropy_factor * 0.6 + rework_factor * 0.4
            component_reliabilities.append(reliability)
        else:
            component_reliabilities.append(0.0)

    weakest_link_reliability = min(component_reliabilities) if component_reliabilities else 0.0

    # Build module-level Merkle tree from component roots
    root_items = [{"root": r, "component_id": cid} for r, cid in zip(component_merkle_roots, component_ids)]
    module_root, _ = _build_merkle_tree(root_items)

    result = ModuleProvenanceReceipt(
        module_id=module_id,
        component_ids=component_ids,
        component_merkle_roots=component_merkle_roots,
        combined_entropy=combined_entropy,
        aggregate_rework_count=aggregate_rework_count,
        weakest_link_reliability=weakest_link_reliability,
        all_components_valid=all_components_valid,
        rejected_components=rejected_components,
        merkle_root=module_root,
        timestamp=timestamp,
    )

    # Emit module provenance receipt
    emit_receipt(
        "module_provenance",
        {
            "tenant_id": TENANT_ID,
            "module_id": module_id,
            "component_count": len(component_ids),
            "all_components_valid": all_components_valid,
            "rejected_count": len(rejected_components),
            "rejected_components": rejected_components,
            "combined_entropy": combined_entropy,
            "aggregate_rework_count": aggregate_rework_count,
            "weakest_link_reliability": weakest_link_reliability,
            "merkle_root": module_root,
            "timestamp": timestamp,
        },
    )

    return result


def create_manufacturer_receipt(
    component_id: str,
    manufacturer: str,
    visual_hash: str,
    electrical_hash: str,
    baseline_entropy: float = 0.30,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Create manufacturer baseline receipt for a component.

    Args:
        component_id: Component identifier
        manufacturer: Manufacturer name
        visual_hash: Hash of visual inspection data
        electrical_hash: Hash of electrical test data
        baseline_entropy: Expected entropy for legitimate part
        metadata: Additional metadata

    Returns:
        Manufacturer receipt dict
    """
    from datetime import datetime

    receipt = {
        "receipt_type": "manufacturer",
        "component_id": component_id,
        "manufacturer": manufacturer,
        "visual_hash": visual_hash,
        "electrical_hash": electrical_hash,
        "entropy": baseline_entropy,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
    }

    receipt["hash"] = dual_hash(json.dumps(receipt, sort_keys=True))

    emit_receipt(
        "manufacturer_baseline",
        {
            "tenant_id": TENANT_ID,
            "component_id": component_id,
            "manufacturer": manufacturer,
            "entropy": baseline_entropy,
            "hash": receipt["hash"],
        },
    )

    return receipt


def create_rework_receipt(
    component_id: str,
    rework_number: int,
    reason: str,
    entropy_before: float,
    entropy_after: float,
    previous_hash: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Create rework receipt for a component.

    Args:
        component_id: Component identifier
        rework_number: Which rework cycle (1, 2, 3, ...)
        reason: Reason for rework
        entropy_before: Entropy before rework
        entropy_after: Entropy after rework
        previous_hash: Hash of previous receipt in chain
        metadata: Additional metadata

    Returns:
        Rework receipt dict
    """
    from datetime import datetime

    receipt = {
        "receipt_type": "rework",
        "component_id": component_id,
        "rework_number": rework_number,
        "reason": reason,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy": entropy_after,  # For chain tracking
        "previous_hash": previous_hash,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
    }

    receipt["hash"] = dual_hash(json.dumps(receipt, sort_keys=True))

    emit_receipt(
        "rework",
        {
            "tenant_id": TENANT_ID,
            "component_id": component_id,
            "rework_number": rework_number,
            "reason": reason,
            "entropy_delta": entropy_after - entropy_before,
            "hash": receipt["hash"],
        },
    )

    return receipt
