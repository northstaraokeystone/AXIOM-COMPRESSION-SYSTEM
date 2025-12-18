"""provenance_mars.py - Mars Receipt Provenance with Merkle Batching

THE PARADIGM SHIFT:
    τ penalty used to be: "High latency → low trust → slow compounding"
    Now it's: "High latency → receipts required → trust compounds → fast compounding"

    The constraint became the forcing function.

Every autonomous decision during blackout emits a receipt.
Receipts batch into merkle roots during comm windows.
Roots sync on reconnect. Post-sync audit validates everything that happened offline.

THE MATH:
    - Without receipts (integrity=0): effective_α = 1.69 × 0.35 = 0.59
    - With 90% receipts: effective_α = 1.69 × 0.935 = 1.58

That's a 2.7× improvement in effective compounding from receipts alone.

Source: CLAUDEME v3.1, Grok hybrid autonomy analysis
Source: Grok - "emit_receipt on every simulated decision cycle"
Source: Grok - "merkle-root batching for sync windows"
Source: Grok - "disparity >0.5% triggers halt"
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .core import emit_receipt, dual_hash, merkle, StopRule


# === CONSTANTS ===

RECEIPT_INTEGRITY_BASELINE = 0.90
"""90% receipted decisions. Source: Grok - '90% receipted decisions'"""

DISPARITY_HALT_THRESHOLD = 0.005
"""Halt if >0.5% decisions unreceipted. Source: Grok - 'disparity >0.5% triggers halt'"""

SYNC_WINDOW_HOURS = 4
"""Typical Mars comm window duration in hours."""

MERKLE_BATCH_SIZE = 1000
"""Batch size for efficiency in merkle root computation."""

RECEIPT_MITIGATION_ACCELERATION_CYCLES = 12
"""Midpoint of Grok's '8-15 cycles' acceleration estimate."""

RECEIPT_PARAMS_PATH = "data/verified/receipt_params.json"
"""Path to receipt parameters file."""


# === DATACLASSES ===


@dataclass
class ProvenanceConfig:
    """Configuration for Mars provenance system.

    Attributes:
        sync_window_hours: Duration of Mars comm window (default 4)
        disparity_threshold: Halt threshold for missing receipts (default 0.005)
        integrity_target: Target receipt coverage (default 0.90)
    """

    sync_window_hours: int = SYNC_WINDOW_HOURS
    disparity_threshold: float = DISPARITY_HALT_THRESHOLD
    integrity_target: float = RECEIPT_INTEGRITY_BASELINE


@dataclass
class ProvenanceState:
    """State of Mars provenance system.

    Attributes:
        pending_receipts: Receipts awaiting batch (list of dicts)
        merkle_batches: Merkle roots from completed batches (list of str)
        receipt_count: Total receipts emitted
        decisions_total: Total decisions made
        integrity: Receipt coverage ratio (receipt_count / decisions_total)
        last_sync_ts: ISO8601 timestamp of last sync
        synced_batches: Set of batch IDs that have been synced
    """

    pending_receipts: List[Dict] = field(default_factory=list)
    merkle_batches: List[str] = field(default_factory=list)
    receipt_count: int = 0
    decisions_total: int = 0
    integrity: float = 1.0
    last_sync_ts: Optional[str] = None
    synced_batches: List[str] = field(default_factory=list)


# === FUNCTIONS ===


def load_receipt_params(path: str = None) -> Dict[str, Any]:
    """Load and verify receipt integrity parameters.

    Loads data/verified/receipt_params.json, verifies payload_hash,
    and emits receipt_params_ingest receipt.

    Args:
        path: Optional path override (default: RECEIPT_PARAMS_PATH)

    Returns:
        Dict containing verified receipt parameters

    Raises:
        StopRule: If payload_hash doesn't match computed hash
        FileNotFoundError: If data file doesn't exist

    Receipt: receipt_params_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, RECEIPT_PARAMS_PATH)

    with open(path, "r") as f:
        data = json.load(f)

    # Extract stored hash
    stored_hash = data.pop("payload_hash", None)
    if stored_hash is None:
        raise StopRule("receipt_params.json missing payload_hash field")

    # Compute expected hash from data (without payload_hash)
    computed_hash = dual_hash(json.dumps(data, sort_keys=True))

    # Verify hash
    hash_verified = stored_hash == computed_hash

    if not hash_verified:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": "axiom-autonomy",
                "metric": "hash_mismatch",
                "classification": "violation",
                "action": "halt",
                "expected": stored_hash,
                "actual": computed_hash,
                "file_path": path,
            },
        )
        raise StopRule(
            f"Receipt params hash mismatch: expected {stored_hash}, got {computed_hash}"
        )

    # Emit ingest receipt
    emit_receipt(
        "receipt_params_ingest",
        {
            "tenant_id": "axiom-autonomy",
            "file_path": path,
            "receipt_integrity_baseline": data["receipt_integrity_baseline"],
            "receipt_efficacy_factor": data["receipt_efficacy_factor"],
            "disparity_halt_threshold": data["disparity_halt_threshold"],
            "sync_window_hours": data["sync_window_hours"],
            "receipt_mitigation_acceleration_cycles": data[
                "receipt_mitigation_acceleration_cycles"
            ],
            "hash_verified": hash_verified,
            "payload_hash": stored_hash,
        },
    )

    # Restore hash to data for downstream use
    data["payload_hash"] = stored_hash

    return data


def emit_mars_receipt(decision: Dict, state: ProvenanceState) -> ProvenanceState:
    """Emit receipt for a Mars decision cycle.

    Add receipt to pending queue. Increment counts. Emit mars_provenance_receipt.

    Args:
        decision: Decision dict with at least 'decision_id', 'decision_type', 'cycle'
        state: Current ProvenanceState

    Returns:
        Updated ProvenanceState with receipt added

    Receipt: mars_provenance_receipt
    """
    state.decisions_total += 1
    state.receipt_count += 1

    # Compute updated integrity
    state.integrity = compute_integrity(state)

    # Create receipt record
    receipt_record = {
        "decision_id": decision.get("decision_id", f"d_{state.decisions_total}"),
        "decision_type": decision.get("decision_type", "autonomous"),
        "cycle": decision.get("cycle", state.decisions_total),
        "ts": datetime.utcnow().isoformat() + "Z",
        "integrity": state.integrity,
    }

    # Add to pending queue
    state.pending_receipts.append(receipt_record)

    # Emit mars_provenance_receipt
    emit_receipt(
        "mars_provenance_receipt",
        {
            "tenant_id": "axiom-autonomy",
            "decision_id": receipt_record["decision_id"],
            "decision_type": receipt_record["decision_type"],
            "cycle": receipt_record["cycle"],
            "integrity": state.integrity,
            "pending_count": len(state.pending_receipts),
        },
    )

    return state


def batch_pending(state: ProvenanceState) -> Tuple[str, ProvenanceState]:
    """Compute merkle root of pending receipts and clear queue.

    Compute merkle root of pending receipts. Clear pending. Add root to merkle_batches.
    Emit merkle_batch_receipt.

    Args:
        state: Current ProvenanceState

    Returns:
        Tuple of (merkle_root, updated ProvenanceState)

    Receipt: merkle_batch_receipt
    """
    if not state.pending_receipts:
        # Empty batch - still compute root for consistency
        root = dual_hash("empty_batch")
    else:
        root = merkle(state.pending_receipts)

    batch_id = f"batch_{len(state.merkle_batches)}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    receipt_count = len(state.pending_receipts)

    # Add root to batches
    state.merkle_batches.append(root)

    # Clear pending
    state.pending_receipts = []

    # Emit merkle_batch_receipt
    emit_receipt(
        "merkle_batch_receipt",
        {
            "tenant_id": "axiom-autonomy",
            "batch_id": batch_id,
            "receipt_count": receipt_count,
            "merkle_root": root,
            "pending_sync": True,
            "total_batches": len(state.merkle_batches),
        },
    )

    return root, state


def sync_batch(state: ProvenanceState, root: str) -> ProvenanceState:
    """Mark batch as synced.

    Mark batch as synced. Update last_sync_ts. Emit sync_receipt.

    Args:
        state: Current ProvenanceState
        root: Merkle root of batch to sync

    Returns:
        Updated ProvenanceState

    Receipt: sync_receipt
    """
    sync_ts = datetime.utcnow().isoformat() + "Z"
    state.last_sync_ts = sync_ts

    # Mark as synced
    state.synced_batches.append(root)

    # Emit sync_receipt
    emit_receipt(
        "sync_receipt",
        {
            "tenant_id": "axiom-autonomy",
            "batch_id": f"sync_{len(state.synced_batches)}",
            "merkle_root": root,
            "synced_ts": sync_ts,
            "total_synced": len(state.synced_batches),
        },
    )

    return state


def check_disparity(state: ProvenanceState, config: ProvenanceConfig = None) -> bool:
    """Check if decisions without receipts exceed threshold.

    Return True if disparity is within limits.
    If disparity exceeds threshold, emit disparity_halt_receipt and raise StopRule.

    Args:
        state: Current ProvenanceState
        config: ProvenanceConfig (uses defaults if None)

    Returns:
        True if disparity within limits

    Raises:
        StopRule: If disparity > threshold

    Receipt: disparity_halt_receipt (if halt triggered)
    """
    if config is None:
        config = ProvenanceConfig()

    if state.decisions_total == 0:
        return True  # No decisions yet, no disparity

    # Calculate disparity (unreceipted decisions / total)
    unreceipted = state.decisions_total - state.receipt_count
    disparity = unreceipted / state.decisions_total

    if disparity > config.disparity_threshold:
        # Emit disparity_halt_receipt
        emit_receipt(
            "disparity_halt_receipt",
            {
                "tenant_id": "axiom-autonomy",
                "integrity": state.integrity,
                "threshold": config.disparity_threshold,
                "disparity": disparity,
                "decisions_unreceipted": unreceipted,
                "decisions_total": state.decisions_total,
                "action": "HALT",
            },
        )
        raise StopRule(
            f"Disparity halt: {disparity:.4f} > {config.disparity_threshold} "
            f"({unreceipted}/{state.decisions_total} decisions unreceipted)"
        )

    return True


def compute_integrity(state: ProvenanceState) -> float:
    """Compute receipt integrity metric.

    Return receipt_count / decisions_total. This is the receipt_integrity metric.

    Args:
        state: Current ProvenanceState

    Returns:
        Receipt integrity (0-1). Returns 1.0 if no decisions yet.
    """
    if state.decisions_total == 0:
        return 1.0
    return state.receipt_count / state.decisions_total


def initialize_provenance_state() -> ProvenanceState:
    """Initialize fresh provenance state.

    Returns:
        Fresh ProvenanceState
    """
    return ProvenanceState(
        pending_receipts=[],
        merkle_batches=[],
        receipt_count=0,
        decisions_total=0,
        integrity=1.0,
        last_sync_ts=None,
        synced_batches=[],
    )


def register_decision_without_receipt(state: ProvenanceState) -> ProvenanceState:
    """Register a decision that was made without a receipt.

    This is used for testing disparity halt. In production, this would represent
    a decision made during a failure condition where receipt emission failed.

    Args:
        state: Current ProvenanceState

    Returns:
        Updated ProvenanceState with decisions_total incremented but not receipt_count
    """
    state.decisions_total += 1
    state.integrity = compute_integrity(state)
    return state
