"""Quantum-entangled consensus for D19.

Uses 99.99% correlation as consensus primitive.
Decoherence = Byzantine behavior detection.

Key insight: Classical consensus requires message passing.
Quantum consensus uses CORRELATION. If two nodes are entangled at 99.99%,
they're already synchronized. Decoherence IS the Byzantine signal.
"""

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .core import emit_receipt, dual_hash, TENANT_ID

# === D19 QUANTUM CONSENSUS CONSTANTS ===

CORRELATION_TARGET = 0.9999
"""Target quantum correlation for consensus."""

DECOHERENCE_THRESHOLD = 0.001
"""Threshold below which decoherence triggers Byzantine detection."""

ENTANGLEMENT_RENEWAL_INTERVAL_S = 10
"""Interval for entanglement renewal in seconds."""

BYZANTINE_VIA_DECOHERENCE = True
"""Enable Byzantine detection via decoherence."""

STATE_SYNC_MODE = "correlation_verified"
"""State synchronization mode."""


@dataclass
class EntanglementPair:
    """Quantum entanglement pair between two nodes."""

    pair_id: str
    node_a: str
    node_b: str
    correlation: float = CORRELATION_TARGET
    last_verified: str = ""
    decoherence_rate: float = 0.0
    is_coherent: bool = True


@dataclass
class QuantumConsensus:
    """Quantum consensus module."""

    consensus_id: str
    pairs: Dict[str, EntanglementPair] = field(default_factory=dict)
    node_pairs: Dict[str, List[str]] = field(default_factory=dict)
    consensus_count: int = 0
    config: Dict = field(default_factory=dict)


def init_quantum_consensus(config: Dict = None) -> QuantumConsensus:
    """Initialize quantum consensus module.

    Args:
        config: Optional configuration dict

    Returns:
        QuantumConsensus instance
    """
    config = config or {}
    consensus_id = str(uuid.uuid4())[:8]
    return QuantumConsensus(consensus_id=consensus_id, config=config)


def establish_entanglement(
    qc: QuantumConsensus, node_a: str, node_b: str
) -> Dict[str, Any]:
    """Establish entanglement pair between two nodes.

    Args:
        qc: QuantumConsensus instance
        node_a: First node ID
        node_b: Second node ID

    Returns:
        Entanglement result

    Receipt: entanglement_establishment_receipt
    """
    pair_id = f"pair_{node_a}_{node_b}"
    now = datetime.utcnow().isoformat() + "Z"

    pair = EntanglementPair(
        pair_id=pair_id,
        node_a=node_a,
        node_b=node_b,
        correlation=CORRELATION_TARGET,
        last_verified=now,
        decoherence_rate=0.0001,  # Natural decoherence
        is_coherent=True,
    )

    qc.pairs[pair_id] = pair

    # Track node pairs
    if node_a not in qc.node_pairs:
        qc.node_pairs[node_a] = []
    if node_b not in qc.node_pairs:
        qc.node_pairs[node_b] = []
    qc.node_pairs[node_a].append(pair_id)
    qc.node_pairs[node_b].append(pair_id)

    result = {
        "pair_id": pair_id,
        "node_a": node_a,
        "node_b": node_b,
        "correlation": pair.correlation,
        "status": "entangled",
    }

    emit_receipt(
        "entanglement_establishment",
        {
            "receipt_type": "entanglement_establishment",
            "tenant_id": TENANT_ID,
            "ts": now,
            "consensus_id": qc.consensus_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def verify_correlation(qc: QuantumConsensus, pair_id: str) -> float:
    """Verify current correlation of entanglement pair.

    Args:
        qc: QuantumConsensus instance
        pair_id: Pair identifier

    Returns:
        Current correlation 0-1

    Receipt: correlation_verification_receipt
    """
    pair = qc.pairs.get(pair_id)
    if not pair:
        return 0.0

    now = datetime.utcnow().isoformat() + "Z"

    # Simulate natural decoherence over time
    # Correlation decays slightly each verification
    noise = random.gauss(0, 0.0001)
    pair.correlation = max(0, min(1, pair.correlation - pair.decoherence_rate + noise))
    pair.last_verified = now

    # Check if still coherent
    pair.is_coherent = pair.correlation >= (CORRELATION_TARGET - DECOHERENCE_THRESHOLD)

    emit_receipt(
        "correlation_verification",
        {
            "receipt_type": "correlation_verification",
            "tenant_id": TENANT_ID,
            "ts": now,
            "consensus_id": qc.consensus_id,
            "pair_id": pair_id,
            "correlation": round(pair.correlation, 6),
            "is_coherent": pair.is_coherent,
            "payload_hash": dual_hash(
                json.dumps(
                    {"pair_id": pair_id, "corr": pair.correlation}, sort_keys=True
                )
            ),
        },
    )

    return pair.correlation


def detect_decoherence(qc: QuantumConsensus, pair_id: str) -> bool:
    """Detect if decoherence has occurred.

    Args:
        qc: QuantumConsensus instance
        pair_id: Pair identifier

    Returns:
        True if decoherence detected

    Receipt: decoherence_detection_receipt
    """
    pair = qc.pairs.get(pair_id)
    if not pair:
        return False

    # Verify current correlation
    current_corr = verify_correlation(qc, pair_id)

    # Decoherence = correlation dropped below threshold
    decoherence_detected = current_corr < (CORRELATION_TARGET - DECOHERENCE_THRESHOLD)

    if decoherence_detected:
        pair.is_coherent = False

    emit_receipt(
        "decoherence_detection",
        {
            "receipt_type": "decoherence_detection",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "consensus_id": qc.consensus_id,
            "pair_id": pair_id,
            "correlation": round(current_corr, 6),
            "decoherence_detected": decoherence_detected,
            "payload_hash": dual_hash(
                json.dumps(
                    {"pair_id": pair_id, "decoherence": decoherence_detected},
                    sort_keys=True,
                )
            ),
        },
    )

    return decoherence_detected


def interpret_decoherence(qc: QuantumConsensus, pair_id: str) -> str:
    """Interpret cause of decoherence.

    Args:
        qc: QuantumConsensus instance
        pair_id: Pair identifier

    Returns:
        Interpretation: "noise" | "byzantine" | "partition"
    """
    pair = qc.pairs.get(pair_id)
    if not pair:
        return "unknown"

    # Interpret based on decoherence rate
    if pair.decoherence_rate > 0.01:
        # Rapid decoherence = likely Byzantine
        return "byzantine"
    elif pair.decoherence_rate > 0.001:
        # Moderate decoherence = possible partition
        return "partition"
    else:
        # Slow decoherence = natural noise
        return "noise"


def achieve_quantum_consensus(qc: QuantumConsensus, proposal: Dict) -> Dict[str, Any]:
    """Achieve consensus via quantum correlation.

    Consensus is achieved when all pairs maintain high correlation.

    Args:
        qc: QuantumConsensus instance
        proposal: Proposal dict

    Returns:
        Consensus result

    Receipt: quantum_consensus_receipt
    """
    qc.consensus_count += 1
    start_time = time.time()

    # Check correlation across all pairs
    coherent_pairs = 0
    total_pairs = len(qc.pairs)
    total_correlation = 0.0

    for pair_id, pair in qc.pairs.items():
        corr = verify_correlation(qc, pair_id)
        total_correlation += corr
        if pair.is_coherent:
            coherent_pairs += 1

    avg_correlation = total_correlation / total_pairs if total_pairs > 0 else 0
    coherence_ratio = coherent_pairs / total_pairs if total_pairs > 0 else 0

    # Consensus achieved if >90% pairs are coherent
    consensus_achieved = coherence_ratio >= 0.90 and avg_correlation >= (
        CORRELATION_TARGET - DECOHERENCE_THRESHOLD
    )

    elapsed_ms = (time.time() - start_time) * 1000

    result = {
        "proposal_id": proposal.get("proposal_id", str(uuid.uuid4())[:8]),
        "consensus_achieved": consensus_achieved,
        "avg_correlation": round(avg_correlation, 6),
        "coherent_pairs": coherent_pairs,
        "total_pairs": total_pairs,
        "coherence_ratio": round(coherence_ratio, 4),
        "latency_ms": round(elapsed_ms, 2),
    }

    emit_receipt(
        "quantum_consensus",
        {
            "receipt_type": "quantum_consensus",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "consensus_id": qc.consensus_id,
            "consensus_count": qc.consensus_count,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def sync_state_via_correlation(qc: QuantumConsensus, state: Dict) -> Dict[str, Any]:
    """Synchronize state using quantum correlation.

    State is consistent when correlation is maintained.

    Args:
        qc: QuantumConsensus instance
        state: State dict to synchronize

    Returns:
        Sync result

    Receipt: state_sync_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Verify all pairs maintain correlation
    synced_nodes = set()
    for pair_id, pair in qc.pairs.items():
        if pair.is_coherent:
            synced_nodes.add(pair.node_a)
            synced_nodes.add(pair.node_b)

    result = {
        "state_hash": dual_hash(json.dumps(state, sort_keys=True))[:16],
        "synced_nodes": len(synced_nodes),
        "total_nodes": len(qc.node_pairs),
        "sync_mode": STATE_SYNC_MODE,
        "sync_ratio": round(len(synced_nodes) / len(qc.node_pairs), 4)
        if qc.node_pairs
        else 1.0,
    }

    emit_receipt(
        "state_sync",
        {
            "receipt_type": "state_sync",
            "tenant_id": TENANT_ID,
            "ts": now,
            "consensus_id": qc.consensus_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def renew_entanglement(qc: QuantumConsensus, pair_id: str) -> Dict[str, Any]:
    """Renew entanglement to combat decoherence.

    Args:
        qc: QuantumConsensus instance
        pair_id: Pair identifier

    Returns:
        Renewal result

    Receipt: entanglement_renewal_receipt
    """
    pair = qc.pairs.get(pair_id)
    if not pair:
        return {"error": "pair_not_found", "pair_id": pair_id}

    now = datetime.utcnow().isoformat() + "Z"

    # Renew to target correlation
    old_correlation = pair.correlation
    pair.correlation = CORRELATION_TARGET
    pair.last_verified = now
    pair.is_coherent = True

    result = {
        "pair_id": pair_id,
        "old_correlation": round(old_correlation, 6),
        "new_correlation": CORRELATION_TARGET,
        "renewed_at": now,
    }

    emit_receipt(
        "entanglement_renewal",
        {
            "receipt_type": "entanglement_renewal",
            "tenant_id": TENANT_ID,
            "ts": now,
            "consensus_id": qc.consensus_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_quantum_consensus_status() -> Dict[str, Any]:
    """Get current quantum consensus status.

    Returns:
        Quantum consensus status dict
    """
    return {
        "module": "quantum_entangled_consensus",
        "version": "19.0.0",
        "correlation_target": CORRELATION_TARGET,
        "decoherence_threshold": DECOHERENCE_THRESHOLD,
        "entanglement_renewal_interval_s": ENTANGLEMENT_RENEWAL_INTERVAL_S,
        "byzantine_via_decoherence": BYZANTINE_VIA_DECOHERENCE,
        "state_sync_mode": STATE_SYNC_MODE,
    }
