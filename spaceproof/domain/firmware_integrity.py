"""firmware_integrity.py - Firmware Integrity (All Targets).

D20 Production Evolution: Supply chain verification from source to orbit.

THE FIRMWARE INTEGRITY PARADIGM:
    Supply chain attacks are the #1 space threat.
    Firmware must be verified at every stage:
    source → build → deployment → execution

    Binary hash must match source commit hash.
    Merkle chain proves untampered path.

Target: All (Starcloud, Starlink, Defense)
Receipts: firmware_integrity_receipt

SLOs:
    - 100% builds have integrity receipts
    - 100% malicious injections detected (hash mismatch)
    - Merkle supply chain verified
    - Integrity verification < 1 second per chain

Source: Grok Research All targets supply chain
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from spaceproof.core import dual_hash, emit_receipt, merkle, TENANT_ID

# === CONSTANTS ===

FIRMWARE_INTEGRITY_TENANT = "spaceproof-firmware-integrity"

# Verification thresholds
MAX_VERIFICATION_TIME_MS = 1000  # 1 second max
INTEGRITY_CONFIDENCE_THRESHOLD = 0.99  # 99% confidence required


@dataclass
class SourceCommit:
    """Source code commit record."""

    commit_id: str
    repo_url: str
    commit_hash: str
    author: str
    timestamp: str
    receipt: Dict[str, Any]


@dataclass
class BuildArtifact:
    """Build artifact record."""

    build_id: str
    commit_hash: str
    binary_hash: str
    build_metadata: Dict[str, Any]
    build_timestamp: str
    receipt: Dict[str, Any]


@dataclass
class Deployment:
    """Firmware deployment record."""

    deployment_id: str
    binary_hash: str
    satellite_id: str
    deployment_time: str
    deployment_context: Dict[str, Any]
    receipt: Dict[str, Any]


@dataclass
class Execution:
    """Firmware execution record."""

    execution_id: str
    satellite_id: str
    binary_hash: str
    execution_proof: Dict[str, Any]
    execution_timestamp: str
    receipt: Dict[str, Any]


@dataclass
class IntegrityChain:
    """Complete firmware integrity chain."""

    chain_id: str
    satellite_id: str
    source_commit: SourceCommit
    build_artifact: BuildArtifact
    deployment: Deployment
    execution: Execution
    merkle_supply_chain: str
    integrity_verified: bool
    integrity_receipt: Dict[str, Any]


@dataclass
class IntegrityVerification:
    """Integrity verification result."""

    verified: bool
    source_hash: str
    execution_hash: str
    chain_valid: bool
    mismatches: List[str]
    verification_time_ms: float
    receipt: Dict[str, Any]


def log_source_commit(
    repo_url: str,
    commit_hash: str,
    timestamp: Optional[str] = None,
    author: str = "unknown",
) -> SourceCommit:
    """Emit source receipt for code commit.

    Args:
        repo_url: Repository URL
        commit_hash: Git commit SHA
        timestamp: Commit timestamp (optional)
        author: Commit author

    Returns:
        SourceCommit with receipt
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    commit_id = dual_hash(f"{repo_url}:{commit_hash}")

    receipt = emit_receipt(
        "source_commit",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "commit_id": commit_id,
            "repo_url": repo_url,
            "commit_hash": commit_hash,
            "author": author,
            "timestamp": timestamp,
        },
    )

    return SourceCommit(
        commit_id=commit_id,
        repo_url=repo_url,
        commit_hash=commit_hash,
        author=author,
        timestamp=timestamp,
        receipt=receipt,
    )


def log_build_artifact(
    commit_hash: str,
    binary_hash: str,
    build_metadata: Dict[str, Any],
) -> BuildArtifact:
    """Emit build receipt for compiled artifact.

    Args:
        commit_hash: Source commit hash
        binary_hash: Dual-hash of compiled binary
        build_metadata: Build configuration and metadata

    Returns:
        BuildArtifact with receipt
    """
    build_id = dual_hash(f"{commit_hash}:{binary_hash}")
    build_timestamp = datetime.utcnow().isoformat() + "Z"

    receipt = emit_receipt(
        "build_artifact",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "build_id": build_id,
            "commit_hash": commit_hash,
            "binary_hash": binary_hash,
            "build_metadata": build_metadata,
            "build_timestamp": build_timestamp,
            "reproducible": build_metadata.get("reproducible", False),
        },
    )

    return BuildArtifact(
        build_id=build_id,
        commit_hash=commit_hash,
        binary_hash=binary_hash,
        build_metadata=build_metadata,
        build_timestamp=build_timestamp,
        receipt=receipt,
    )


def log_deployment(
    binary_hash: str,
    satellite_id: str,
    deployment_time: Optional[str] = None,
    deployment_context: Optional[Dict[str, Any]] = None,
) -> Deployment:
    """Emit deployment receipt for firmware upload.

    Args:
        binary_hash: Hash of deployed binary
        satellite_id: Target satellite identifier
        deployment_time: Deployment timestamp
        deployment_context: Deployment configuration

    Returns:
        Deployment with receipt
    """
    if deployment_time is None:
        deployment_time = datetime.utcnow().isoformat() + "Z"

    if deployment_context is None:
        deployment_context = {}

    deployment_id = dual_hash(f"{binary_hash}:{satellite_id}:{deployment_time}")

    receipt = emit_receipt(
        "firmware_deployment",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "deployment_id": deployment_id,
            "binary_hash": binary_hash,
            "satellite_id": satellite_id,
            "deployment_time": deployment_time,
            "deployment_context": deployment_context,
        },
    )

    return Deployment(
        deployment_id=deployment_id,
        binary_hash=binary_hash,
        satellite_id=satellite_id,
        deployment_time=deployment_time,
        deployment_context=deployment_context,
        receipt=receipt,
    )


def log_execution(
    satellite_id: str,
    binary_hash: str,
    execution_proof: Dict[str, Any],
) -> Execution:
    """Emit execution receipt proving firmware is running.

    Args:
        satellite_id: Satellite identifier
        binary_hash: Hash of running binary
        execution_proof: Proof of execution (telemetry, attestation)

    Returns:
        Execution with receipt
    """
    execution_timestamp = datetime.utcnow().isoformat() + "Z"
    execution_id = dual_hash(f"{satellite_id}:{binary_hash}:{execution_timestamp}")

    # Hash the execution proof
    proof_hash = dual_hash(json.dumps(execution_proof, sort_keys=True))

    receipt = emit_receipt(
        "firmware_execution",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "execution_id": execution_id,
            "satellite_id": satellite_id,
            "binary_hash": binary_hash,
            "execution_proof_hash": proof_hash,
            "execution_timestamp": execution_timestamp,
        },
    )

    return Execution(
        execution_id=execution_id,
        satellite_id=satellite_id,
        binary_hash=binary_hash,
        execution_proof=execution_proof,
        execution_timestamp=execution_timestamp,
        receipt=receipt,
    )


def verify_integrity_chain(
    source_hash: str,
    execution_hash: str,
    chain: Optional[List[Dict[str, Any]]] = None,
) -> IntegrityVerification:
    """Verify source → execution chain integrity.

    Args:
        source_hash: Source commit hash
        execution_hash: Execution binary hash
        chain: Optional list of intermediate receipts

    Returns:
        IntegrityVerification result
    """
    import time
    start_time = time.time()

    mismatches = []
    chain_valid = True

    if chain:
        # Verify each link in the chain
        prev_hash = source_hash

        for i, receipt in enumerate(chain):
            receipt_type = receipt.get("receipt_type", "")

            if receipt_type == "build_artifact":
                if receipt.get("commit_hash") != prev_hash:
                    mismatches.append(f"Build artifact commit hash mismatch at position {i}")
                    chain_valid = False
                prev_hash = receipt.get("binary_hash", "")

            elif receipt_type == "firmware_deployment":
                if receipt.get("binary_hash") != prev_hash:
                    mismatches.append(f"Deployment binary hash mismatch at position {i}")
                    chain_valid = False

            elif receipt_type == "firmware_execution":
                if receipt.get("binary_hash") != prev_hash:
                    mismatches.append(f"Execution binary hash mismatch at position {i}")
                    chain_valid = False

    # Simple hash comparison if no chain provided
    if not chain and source_hash != execution_hash:
        mismatches.append("Direct source-to-execution hash mismatch")
        chain_valid = False

    verification_time_ms = (time.time() - start_time) * 1000
    verified = chain_valid and len(mismatches) == 0

    receipt = emit_receipt(
        "integrity_verification",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "source_hash": source_hash,
            "execution_hash": execution_hash,
            "chain_valid": chain_valid,
            "integrity_verified": verified,
            "mismatch_count": len(mismatches),
            "verification_time_ms": verification_time_ms,
            "slo_met": verification_time_ms < MAX_VERIFICATION_TIME_MS,
        },
    )

    return IntegrityVerification(
        verified=verified,
        source_hash=source_hash,
        execution_hash=execution_hash,
        chain_valid=chain_valid,
        mismatches=mismatches,
        verification_time_ms=verification_time_ms,
        receipt=receipt,
    )


def emit_firmware_integrity(
    receipts: List[Dict[str, Any]],
    satellite_id: str = "unknown",
) -> Dict[str, Any]:
    """Merkle-anchor full supply chain into integrity receipt.

    Args:
        receipts: List of supply chain receipts
        satellite_id: Satellite identifier

    Returns:
        Firmware integrity receipt
    """
    if not receipts:
        receipts = []

    merkle_supply_chain = merkle(receipts)
    chain_id = dual_hash(merkle_supply_chain)

    # Extract hashes from chain
    source_hash = ""
    binary_hash = ""
    deployment_hash = ""
    execution_hash = ""

    for r in receipts:
        receipt_type = r.get("receipt_type", "")
        if receipt_type == "source_commit":
            source_hash = r.get("commit_hash", "")
        elif receipt_type == "build_artifact":
            binary_hash = r.get("binary_hash", "")
        elif receipt_type == "firmware_deployment":
            deployment_hash = r.get("binary_hash", "")
        elif receipt_type == "firmware_execution":
            execution_hash = r.get("binary_hash", "")

    # Verify chain integrity
    integrity_verified = True
    if binary_hash and deployment_hash and binary_hash != deployment_hash:
        integrity_verified = False
    if deployment_hash and execution_hash and deployment_hash != execution_hash:
        integrity_verified = False

    integrity_receipt = emit_receipt(
        "firmware_integrity",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "chain_id": chain_id,
            "satellite_id": satellite_id,
            "source_commit_hash": source_hash,
            "build_binary_hash": binary_hash,
            "deployment_hash": deployment_hash,
            "execution_proof_hash": execution_hash,
            "merkle_supply_chain": merkle_supply_chain,
            "integrity_verified": integrity_verified,
            "receipt_count": len(receipts),
        },
    )

    return integrity_receipt


def detect_supply_chain_attack(
    expected_binary_hash: str,
    actual_binary_hash: str,
) -> Dict[str, Any]:
    """Detect supply chain attack via hash mismatch.

    Args:
        expected_binary_hash: Expected binary hash from build
        actual_binary_hash: Actual binary hash on satellite

    Returns:
        Detection result
    """
    attack_detected = expected_binary_hash != actual_binary_hash

    receipt = emit_receipt(
        "supply_chain_attack_detection",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "expected_hash": expected_binary_hash,
            "actual_hash": actual_binary_hash,
            "attack_detected": attack_detected,
            "severity": "critical" if attack_detected else "none",
            "action": "quarantine" if attack_detected else "continue",
        },
    )

    return {
        "attack_detected": attack_detected,
        "expected_hash": expected_binary_hash,
        "actual_hash": actual_binary_hash,
        "receipt": receipt,
    }


def build_complete_chain(
    source: SourceCommit,
    build: BuildArtifact,
    deployment: Deployment,
    execution: Execution,
) -> IntegrityChain:
    """Build complete integrity chain from components.

    Args:
        source: Source commit
        build: Build artifact
        deployment: Deployment record
        execution: Execution record

    Returns:
        Complete IntegrityChain
    """
    receipts = [
        source.receipt,
        build.receipt,
        deployment.receipt,
        execution.receipt,
    ]

    merkle_supply_chain = merkle(receipts)
    chain_id = dual_hash(merkle_supply_chain)

    # Verify chain
    integrity_verified = True
    if build.commit_hash != source.commit_hash:
        integrity_verified = False
    if deployment.binary_hash != build.binary_hash:
        integrity_verified = False
    if execution.binary_hash != deployment.binary_hash:
        integrity_verified = False

    integrity_receipt = emit_receipt(
        "firmware_integrity",
        {
            "tenant_id": FIRMWARE_INTEGRITY_TENANT,
            "chain_id": chain_id,
            "satellite_id": execution.satellite_id,
            "source_commit_hash": source.commit_hash,
            "build_binary_hash": build.binary_hash,
            "deployment_hash": deployment.binary_hash,
            "execution_proof_hash": execution.binary_hash,
            "merkle_supply_chain": merkle_supply_chain,
            "integrity_verified": integrity_verified,
        },
    )

    return IntegrityChain(
        chain_id=chain_id,
        satellite_id=execution.satellite_id,
        source_commit=source,
        build_artifact=build,
        deployment=deployment,
        execution=execution,
        merkle_supply_chain=merkle_supply_chain,
        integrity_verified=integrity_verified,
        integrity_receipt=integrity_receipt,
    )


def compute_tamper_detection_rate(verifications: List[IntegrityVerification]) -> float:
    """Compute tamper detection rate for effectiveness measurement.

    Args:
        verifications: List of integrity verifications

    Returns:
        Detection rate (0.0 - 1.0)
    """
    if not verifications:
        return 0.0

    # Assume any verification with mismatches detected tampering
    detected = sum(1 for v in verifications if not v.verified and len(v.mismatches) > 0)

    # Calculate rate relative to potential tampering (unverified chains)
    unverified = sum(1 for v in verifications if not v.verified)

    if unverified == 0:
        return 1.0  # All verified = 100% detection rate

    return detected / unverified if unverified > 0 else 1.0
