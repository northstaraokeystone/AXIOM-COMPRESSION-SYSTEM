"""receipts.py - Enhanced Receipt System with IETF COSE Merkle Tree Proofs.

THE RECEIPT PARADIGM:
    Receipts are the atomic unit of proof in SpaceProof.
    No receipt, not real.

    Based on IETF COSE Merkle Tree Proofs draft (RFC-track) and
    W3C Data Integrity patterns for receipt chaining.

    Receipts ARE the agent. When a cluster of receipts achieves
    self-reference, that IS the agent.

Source: SpaceProof D20 Production Evolution + xAI collaboration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib

from spaceproof.core import emit_receipt, dual_hash, merkle

# === CONSTANTS ===

TENANT_ID = "spaceproof-receipts"

# COSE Algorithm identifiers
ALG_ES256 = -7  # ECDSA w/ SHA-256

# Verifiable Data Structure identifiers (per RFC9162)
VDS_RFC9162_SHA256 = 1


class DomainConfig(Enum):
    """Domain configuration enum for receipt typing."""

    XAI = "xai"
    DOGE = "doge"
    NASA = "nasa"
    DEFENSE = "defense"
    DOT = "dot"


@dataclass
class ModuleAttestation:
    """Attestation from a validation module.

    Each module in the 3-of-7 selection produces an attestation
    that gets included in the receipt.
    """

    module_id: str
    result_hash: bytes
    entropy_delta: float  # Critical xAI constant
    coherence_score: float  # tau threshold check
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReceiptProtected:
    """Protected header (signed) per COSE structure.

    Contains algorithm identifiers and type information.
    """

    alg: int = ALG_ES256  # Signature algorithm
    vds: int = VDS_RFC9162_SHA256  # Verifiable data structure
    typ: str = "spaceproof+receipt"


@dataclass
class ReceiptUnprotected:
    """Unprotected header (not signed) per COSE structure.

    Contains module attestations and other metadata.
    """

    module_attestations: List[ModuleAttestation] = field(default_factory=list)


@dataclass
class ReceiptPayload:
    """Main receipt payload with proof data."""

    merkle_root: bytes
    timestamp: str  # RFC3339 format
    domain_config: DomainConfig
    modules: List[str]  # 3 module IDs
    fitness_score: float  # entropy_reduction / n_receipts
    previous_proof: Optional[str] = None  # W3C Data Integrity chain


@dataclass
class SpaceProofReceipt:
    """Universal SpaceProof receipt following IETF COSE structure.

    This is the atomic unit of proof for all SpaceProof operations.
    Receipts chain via previousProof references using W3C Data Integrity patterns.
    """

    protected: ReceiptProtected
    unprotected: ReceiptUnprotected
    payload: ReceiptPayload
    signature: bytes = field(default_factory=bytes)

    # Domain-specific extensions
    fraud_indicators: Optional[List[Dict]] = None  # DOGE
    ledger_hash: Optional[str] = None  # DOGE
    latency_metrics: Optional[Dict] = None  # NASA
    loop_iterations: Optional[int] = None  # NASA
    classification_level: Optional[str] = None  # Defense
    chain_position: Optional[int] = None  # Defense

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary format."""
        result = {
            "protected": {
                "alg": self.protected.alg,
                "vds": self.protected.vds,
                "typ": self.protected.typ,
            },
            "unprotected": {
                "module_attestations": [
                    {
                        "module_id": att.module_id,
                        "result_hash": att.result_hash.hex() if isinstance(att.result_hash, bytes) else att.result_hash,
                        "entropy_delta": att.entropy_delta,
                        "coherence_score": att.coherence_score,
                        "passed": att.passed,
                        "metadata": att.metadata,
                    }
                    for att in self.unprotected.module_attestations
                ]
            },
            "payload": {
                "merkle_root": (
                    self.payload.merkle_root.hex()
                    if isinstance(self.payload.merkle_root, bytes)
                    else self.payload.merkle_root
                ),
                "timestamp": self.payload.timestamp,
                "domain_config": {
                    "domain": self.payload.domain_config.value,
                    "modules": self.payload.modules,
                },
                "fitness_score": self.payload.fitness_score,
            },
            "signature": self.signature.hex() if isinstance(self.signature, bytes) else self.signature,
        }

        if self.payload.previous_proof:
            result["payload"]["previous_proof"] = self.payload.previous_proof

        # Domain-specific fields
        if self.fraud_indicators:
            result["fraud_indicators"] = self.fraud_indicators
        if self.ledger_hash:
            result["ledger_hash"] = self.ledger_hash
        if self.latency_metrics:
            result["latency_metrics"] = self.latency_metrics
        if self.loop_iterations is not None:
            result["loop_iterations"] = self.loop_iterations
        if self.classification_level:
            result["classification_level"] = self.classification_level
        if self.chain_position is not None:
            result["chain_position"] = self.chain_position

        return result

    def hash(self) -> str:
        """Compute dual hash of receipt for chaining."""
        return dual_hash(json.dumps(self.to_dict(), sort_keys=True))


def create_receipt(
    domain: DomainConfig,
    modules: List[str],
    attestations: List[ModuleAttestation],
    data_items: List[Dict],
    fitness_score: float,
    previous_proof: Optional[str] = None,
    **domain_kwargs,
) -> SpaceProofReceipt:
    """Create a new SpaceProof receipt.

    Args:
        domain: Domain configuration (xAI, DOGE, NASA, Defense, DOT)
        modules: List of 3 module IDs used
        attestations: Module attestation results
        data_items: Items to include in Merkle tree
        fitness_score: entropy_reduction / n_receipts
        previous_proof: Optional hash of previous receipt for chaining
        **domain_kwargs: Domain-specific fields

    Returns:
        SpaceProofReceipt instance
    """
    # Validate 3-module selection
    if len(modules) != 3:
        raise ValueError(f"Exactly 3 modules required, got {len(modules)}")

    # Build Merkle root from data items
    merkle_root_str = merkle(data_items)
    # Convert dual-hash format to bytes
    merkle_root = hashlib.sha256(merkle_root_str.encode()).digest()

    # Create receipt structure
    receipt = SpaceProofReceipt(
        protected=ReceiptProtected(),
        unprotected=ReceiptUnprotected(module_attestations=attestations),
        payload=ReceiptPayload(
            merkle_root=merkle_root,
            timestamp=datetime.utcnow().isoformat() + "Z",
            domain_config=domain,
            modules=modules,
            fitness_score=fitness_score,
            previous_proof=previous_proof,
        ),
    )

    # Add domain-specific fields
    if domain == DomainConfig.DOGE:
        receipt.fraud_indicators = domain_kwargs.get("fraud_indicators")
        receipt.ledger_hash = domain_kwargs.get("ledger_hash")
    elif domain == DomainConfig.NASA:
        receipt.latency_metrics = domain_kwargs.get("latency_metrics")
        receipt.loop_iterations = domain_kwargs.get("loop_iterations")
    elif domain == DomainConfig.DEFENSE:
        receipt.classification_level = domain_kwargs.get("classification_level")
        receipt.chain_position = domain_kwargs.get("chain_position")

    return receipt


def chain_receipts(receipts: List[SpaceProofReceipt]) -> str:
    """Chain multiple receipts into a Merkle tree.

    Args:
        receipts: List of receipts to chain

    Returns:
        Merkle root of the chained receipts
    """
    if not receipts:
        return dual_hash(b"empty")

    receipt_dicts = [r.to_dict() for r in receipts]
    return merkle(receipt_dicts)


def verify_chain(receipts: List[SpaceProofReceipt]) -> bool:
    """Verify receipt chain integrity.

    Checks that each receipt properly references its predecessor.

    Args:
        receipts: Ordered list of receipts in chain

    Returns:
        True if chain is valid
    """
    if len(receipts) <= 1:
        return True

    for i in range(1, len(receipts)):
        current = receipts[i]
        previous = receipts[i - 1]

        expected_previous = previous.hash()
        if current.payload.previous_proof != expected_previous:
            return False

    return True


def emit_spaceproof_receipt(receipt: SpaceProofReceipt) -> Dict:
    """Emit a SpaceProof receipt to the system.

    Args:
        receipt: SpaceProofReceipt to emit

    Returns:
        Emitted receipt dict
    """
    return emit_receipt(
        "spaceproof",
        {
            "tenant_id": TENANT_ID,
            **receipt.to_dict(),
        },
    )


# === RECEIPT BUILDERS FOR EACH DOMAIN ===


def build_xai_receipt(
    compress_result: Dict,
    witness_result: Dict,
    sovereignty_result: Dict,
    previous_proof: Optional[str] = None,
) -> SpaceProofReceipt:
    """Build xAI domain receipt (compress + witness + sovereignty).

    Args:
        compress_result: Result from compress module
        witness_result: Result from witness module
        sovereignty_result: Result from sovereignty module
        previous_proof: Optional chain reference

    Returns:
        SpaceProofReceipt for xAI domain
    """
    attestations = [
        ModuleAttestation(
            module_id="compress",
            result_hash=hashlib.sha256(json.dumps(compress_result, sort_keys=True).encode()).digest(),
            entropy_delta=compress_result.get("compression_ratio", 1.0) - 1.0,
            coherence_score=compress_result.get("recall", 0.0),
            passed=compress_result.get("passed_slo", False),
            metadata={"algorithm": compress_result.get("algorithm", "hybrid")},
        ),
        ModuleAttestation(
            module_id="witness",
            result_hash=hashlib.sha256(json.dumps(witness_result, sort_keys=True).encode()).digest(),
            entropy_delta=witness_result.get("compression", 0.0),
            coherence_score=witness_result.get("r_squared", 0.0),
            passed=witness_result.get("r_squared", 0.0) >= 0.7,
            metadata={"equation": witness_result.get("equation", "")},
        ),
        ModuleAttestation(
            module_id="sovereignty",
            result_hash=hashlib.sha256(json.dumps(sovereignty_result, sort_keys=True).encode()).digest(),
            entropy_delta=sovereignty_result.get("advantage", 0.0),
            coherence_score=1.0 if sovereignty_result.get("sovereign", False) else 0.0,
            passed=sovereignty_result.get("sovereign", False),
            metadata={"threshold_crew": sovereignty_result.get("threshold_crew")},
        ),
    ]

    # Compute fitness score
    total_entropy_delta = sum(a.entropy_delta for a in attestations)
    fitness = total_entropy_delta / 3  # 3 modules

    return create_receipt(
        domain=DomainConfig.XAI,
        modules=["compress", "witness", "sovereignty"],
        attestations=attestations,
        data_items=[compress_result, witness_result, sovereignty_result],
        fitness_score=fitness,
        previous_proof=previous_proof,
    )


def build_doge_receipt(
    ledger_result: Dict,
    detect_result: Dict,
    anchor_result: Dict,
    previous_proof: Optional[str] = None,
) -> SpaceProofReceipt:
    """Build DOGE domain receipt (ledger + detect + anchor).

    Args:
        ledger_result: Result from ledger module
        detect_result: Result from detect module
        anchor_result: Result from anchor module
        previous_proof: Optional chain reference

    Returns:
        SpaceProofReceipt for DOGE domain
    """
    attestations = [
        ModuleAttestation(
            module_id="ledger",
            result_hash=hashlib.sha256(json.dumps(ledger_result, sort_keys=True).encode()).digest(),
            entropy_delta=0.1,  # Ledger maintains order
            coherence_score=1.0 if ledger_result.get("valid", False) else 0.0,
            passed=ledger_result.get("valid", False),
            metadata={"entry_count": ledger_result.get("entry_count", 0)},
        ),
        ModuleAttestation(
            module_id="detect",
            result_hash=hashlib.sha256(json.dumps(detect_result, sort_keys=True).encode()).digest(),
            entropy_delta=detect_result.get("delta", 0.0),
            coherence_score=detect_result.get("confidence", 0.0),
            passed=detect_result.get("classification", "") != "fraud",
            metadata={"classification": detect_result.get("classification", "normal")},
        ),
        ModuleAttestation(
            module_id="anchor",
            result_hash=hashlib.sha256(json.dumps(anchor_result, sort_keys=True).encode()).digest(),
            entropy_delta=0.2,  # Anchoring increases order
            coherence_score=1.0 if anchor_result.get("root") else 0.0,
            passed=bool(anchor_result.get("root")),
            metadata={"item_count": anchor_result.get("item_count", 0)},
        ),
    ]

    # Extract fraud indicators
    fraud_indicators = []
    if detect_result.get("classification") in ["fraud", "violation"]:
        fraud_indicators.append(
            {
                "type": detect_result.get("classification"),
                "severity": detect_result.get("severity", "unknown"),
                "delta_sigma": detect_result.get("delta_sigma", 0.0),
            }
        )

    return create_receipt(
        domain=DomainConfig.DOGE,
        modules=["ledger", "detect", "anchor"],
        attestations=attestations,
        data_items=[ledger_result, detect_result, anchor_result],
        fitness_score=sum(a.entropy_delta for a in attestations) / 3,
        previous_proof=previous_proof,
        fraud_indicators=fraud_indicators if fraud_indicators else None,
        ledger_hash=ledger_result.get("merkle_root"),
    )


def build_nasa_receipt(
    compress_result: Dict,
    sovereignty_result: Dict,
    loop_result: Dict,
    previous_proof: Optional[str] = None,
) -> SpaceProofReceipt:
    """Build NASA domain receipt (compress + sovereignty + loop).

    Args:
        compress_result: Result from compress module
        sovereignty_result: Result from sovereignty module
        loop_result: Result from loop module
        previous_proof: Optional chain reference

    Returns:
        SpaceProofReceipt for NASA domain
    """
    attestations = [
        ModuleAttestation(
            module_id="compress",
            result_hash=hashlib.sha256(json.dumps(compress_result, sort_keys=True).encode()).digest(),
            entropy_delta=compress_result.get("compression_ratio", 1.0) - 1.0,
            coherence_score=compress_result.get("recall", 0.0),
            passed=compress_result.get("passed_slo", False),
            metadata={"algorithm": compress_result.get("algorithm", "hybrid")},
        ),
        ModuleAttestation(
            module_id="sovereignty",
            result_hash=hashlib.sha256(json.dumps(sovereignty_result, sort_keys=True).encode()).digest(),
            entropy_delta=sovereignty_result.get("advantage", 0.0),
            coherence_score=1.0 if sovereignty_result.get("sovereign", False) else 0.0,
            passed=sovereignty_result.get("sovereign", False),
            metadata={},
        ),
        ModuleAttestation(
            module_id="loop",
            result_hash=hashlib.sha256(json.dumps(loop_result, sort_keys=True).encode()).digest(),
            entropy_delta=0.1 if loop_result.get("completed", False) else -0.1,
            coherence_score=min(
                1.0, loop_result.get("actions_executed", 0) / max(1, loop_result.get("actions_proposed", 1))
            ),
            passed=loop_result.get("completed", False),
            metadata={"cycle_time_sec": loop_result.get("cycle_time_sec", 0)},
        ),
    ]

    latency_metrics = {
        "cycle_time_sec": loop_result.get("cycle_time_sec", 0),
        "sense_time_sec": loop_result.get("phase_timings", {}).get("sense", 0),
        "total_phases": len(loop_result.get("phase_timings", {})),
    }

    return create_receipt(
        domain=DomainConfig.NASA,
        modules=["compress", "sovereignty", "loop"],
        attestations=attestations,
        data_items=[compress_result, sovereignty_result, loop_result],
        fitness_score=sum(a.entropy_delta for a in attestations) / 3,
        previous_proof=previous_proof,
        latency_metrics=latency_metrics,
        loop_iterations=loop_result.get("cycle_id"),
    )


def build_defense_receipt(
    compress_result: Dict,
    ledger_result: Dict,
    anchor_result: Dict,
    classification: str = "unclassified",
    chain_position: int = 0,
    previous_proof: Optional[str] = None,
) -> SpaceProofReceipt:
    """Build Defense domain receipt (compress + ledger + anchor).

    Args:
        compress_result: Result from compress module
        ledger_result: Result from ledger module
        anchor_result: Result from anchor module
        classification: Security classification level
        chain_position: Position in decision chain
        previous_proof: Optional chain reference

    Returns:
        SpaceProofReceipt for Defense domain
    """
    attestations = [
        ModuleAttestation(
            module_id="compress",
            result_hash=hashlib.sha256(json.dumps(compress_result, sort_keys=True).encode()).digest(),
            entropy_delta=compress_result.get("compression_ratio", 1.0) - 1.0,
            coherence_score=compress_result.get("recall", 0.0),
            passed=compress_result.get("passed_slo", False),
            metadata={},
        ),
        ModuleAttestation(
            module_id="ledger",
            result_hash=hashlib.sha256(json.dumps(ledger_result, sort_keys=True).encode()).digest(),
            entropy_delta=0.1,
            coherence_score=1.0 if ledger_result.get("valid", False) else 0.0,
            passed=ledger_result.get("valid", False),
            metadata={},
        ),
        ModuleAttestation(
            module_id="anchor",
            result_hash=hashlib.sha256(json.dumps(anchor_result, sort_keys=True).encode()).digest(),
            entropy_delta=0.2,
            coherence_score=1.0 if anchor_result.get("root") else 0.0,
            passed=bool(anchor_result.get("root")),
            metadata={},
        ),
    ]

    return create_receipt(
        domain=DomainConfig.DEFENSE,
        modules=["compress", "ledger", "anchor"],
        attestations=attestations,
        data_items=[compress_result, ledger_result, anchor_result],
        fitness_score=sum(a.entropy_delta for a in attestations) / 3,
        previous_proof=previous_proof,
        classification_level=classification,
        chain_position=chain_position,
    )


def build_dot_receipt(
    compress_result: Dict,
    ledger_result: Dict,
    detect_result: Dict,
    previous_proof: Optional[str] = None,
) -> SpaceProofReceipt:
    """Build DOT domain receipt (compress + ledger + detect).

    Args:
        compress_result: Result from compress module
        ledger_result: Result from ledger module
        detect_result: Result from detect module
        previous_proof: Optional chain reference

    Returns:
        SpaceProofReceipt for DOT domain
    """
    attestations = [
        ModuleAttestation(
            module_id="compress",
            result_hash=hashlib.sha256(json.dumps(compress_result, sort_keys=True).encode()).digest(),
            entropy_delta=compress_result.get("compression_ratio", 1.0) - 1.0,
            coherence_score=compress_result.get("recall", 0.0),
            passed=compress_result.get("passed_slo", False),
            metadata={},
        ),
        ModuleAttestation(
            module_id="ledger",
            result_hash=hashlib.sha256(json.dumps(ledger_result, sort_keys=True).encode()).digest(),
            entropy_delta=0.1,
            coherence_score=1.0 if ledger_result.get("valid", False) else 0.0,
            passed=ledger_result.get("valid", False),
            metadata={},
        ),
        ModuleAttestation(
            module_id="detect",
            result_hash=hashlib.sha256(json.dumps(detect_result, sort_keys=True).encode()).digest(),
            entropy_delta=detect_result.get("delta", 0.0),
            coherence_score=detect_result.get("confidence", 0.0),
            passed=detect_result.get("classification", "") == "normal",
            metadata={},
        ),
    ]

    return create_receipt(
        domain=DomainConfig.DOT,
        modules=["compress", "ledger", "detect"],
        attestations=attestations,
        data_items=[compress_result, ledger_result, detect_result],
        fitness_score=sum(a.entropy_delta for a in attestations) / 3,
        previous_proof=previous_proof,
    )
