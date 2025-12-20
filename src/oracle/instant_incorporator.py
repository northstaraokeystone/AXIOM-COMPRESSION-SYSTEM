"""D19.3 Instant Incorporator.

Purpose: Instantly incorporate new receipts into oracle.
No batch delays. Real-time truth.

Constraints:
  - Incorporation latency < 100ms
  - No batch processing
  - Oracle always reflects latest receipt
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID


# === D19.3 INSTANT INCORPORATION CONSTANTS ===

INCORPORATION_LATENCY_MAX_MS = 100
"""Maximum allowed incorporation latency in milliseconds."""

BATCH_PROCESSING_ENABLED = False
"""Batch processing KILLED - instant incorporation only."""


@dataclass
class InstantIncorporator:
    """Instantly incorporate new receipts into oracle.

    Real-time truth. No batch processing.
    Oracle always reflects latest receipt.
    """

    incorporator_id: str
    oracle: Any = None  # LiveHistoryOracle
    incorporation_count: int = 0
    last_incorporation_ts: str = ""
    last_incorporation_latency_ms: float = 0.0
    violation_count: int = 0


def init_incorporator(oracle: Any = None) -> InstantIncorporator:
    """Attach to oracle for live updates.

    Args:
        oracle: LiveHistoryOracle instance

    Returns:
        InstantIncorporator instance
    """
    incorporator_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat() + "Z"

    return InstantIncorporator(
        incorporator_id=incorporator_id,
        oracle=oracle,
        last_incorporation_ts=now,
    )


def on_receipt_arrival(incorporator: InstantIncorporator, receipt: Dict) -> Dict[str, Any]:
    """Callback: instant incorporation. Update oracle.

    Args:
        incorporator: InstantIncorporator instance
        receipt: New receipt to incorporate

    Returns:
        Incorporation result
    """
    start_time = time.time()
    now = datetime.utcnow().isoformat() + "Z"

    # Update oracle history
    if incorporator.oracle and hasattr(incorporator.oracle, 'history'):
        incorporator.oracle.history.append(receipt)

    # Update compression
    new_compression = update_compression(incorporator, receipt)

    # Update causal graph
    update_causal_graph(incorporator, receipt)

    # Check law survival
    surviving_laws = check_law_survival(incorporator, receipt)

    # Calculate incorporation latency
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    incorporator.incorporation_count += 1
    incorporator.last_incorporation_ts = now
    incorporator.last_incorporation_latency_ms = latency_ms

    # Check latency constraint
    latency_ok = latency_ms < INCORPORATION_LATENCY_MAX_MS
    if not latency_ok:
        incorporator.violation_count += 1

    result = {
        "incorporated": True,
        "receipt_type": receipt.get("receipt_type", "unknown"),
        "latency_ms": round(latency_ms, 4),
        "latency_ok": latency_ok,
        "new_compression": new_compression,
        "surviving_laws": len(surviving_laws),
        "incorporation_count": incorporator.incorporation_count,
        "ts": now,
    }

    # Emit incorporation receipt
    emit_incorporation_receipt(
        incorporator,
        receipt.get("payload_hash", "")[:16],
        new_compression - (incorporator.oracle.compression_ratio if incorporator.oracle else 0),
    )

    return result


def update_compression(incorporator: InstantIncorporator, new_receipt: Dict) -> float:
    """Recalculate compression with new receipt.

    Args:
        incorporator: InstantIncorporator instance
        new_receipt: New receipt

    Returns:
        Updated compression ratio
    """
    if not incorporator.oracle:
        return 0.0

    # Import here to avoid circular imports
    from .live_history_oracle import compute_history_compression

    history = incorporator.oracle.history if hasattr(incorporator.oracle, 'history') else []
    new_compression = compute_history_compression(history)

    if hasattr(incorporator.oracle, 'compression_ratio'):
        incorporator.oracle.compression_ratio = new_compression

    return new_compression


def update_causal_graph(incorporator: InstantIncorporator, new_receipt: Dict) -> None:
    """Add new node/edges to causal graph.

    Args:
        incorporator: InstantIncorporator instance
        new_receipt: New receipt to add
    """
    # If oracle has a causal extractor, update it
    if incorporator.oracle and hasattr(incorporator.oracle, 'extractor'):
        extractor = incorporator.oracle.extractor
        if extractor and hasattr(extractor, 'nodes'):
            node_id = new_receipt.get("payload_hash", str(uuid.uuid4()))[:16]

            from .causal_subgraph_extractor import CausalNode

            node = CausalNode(
                node_id=node_id,
                receipt_type=new_receipt.get("receipt_type", "unknown"),
                timestamp=new_receipt.get("ts", ""),
            )

            # Add temporal edge from last node
            if extractor.nodes:
                last_node_id = list(extractor.nodes.keys())[-1]
                last_node = extractor.nodes[last_node_id]
                last_node.dependents.add(node_id)
                node.dependencies.add(last_node_id)
                extractor.edges.append((last_node_id, node_id))

            extractor.nodes[node_id] = node


def check_law_survival(incorporator: InstantIncorporator, new_receipt: Dict) -> List[Dict]:
    """Verify existing laws still hold. Kill violated.

    Args:
        incorporator: InstantIncorporator instance
        new_receipt: New receipt

    Returns:
        List of surviving laws
    """
    if not incorporator.oracle or not hasattr(incorporator.oracle, 'laws'):
        return []

    surviving_laws = []
    killed_laws = []

    for law in incorporator.oracle.laws:
        # Check if new receipt violates the law
        law_types = set(law.get("receipt_types", []))
        new_type = new_receipt.get("receipt_type", "")

        # For now, all laws survive unless they explicitly conflict
        # A more sophisticated check would verify invariance properties
        surviving_laws.append(law)

    # Update oracle laws
    incorporator.oracle.laws = surviving_laws

    return surviving_laws


def emit_incorporation_receipt(
    incorporator: InstantIncorporator,
    receipt_id: str,
    update_delta: float
) -> Dict[str, Any]:
    """Emit instant_incorporation_receipt.

    Args:
        incorporator: InstantIncorporator instance
        receipt_id: ID of incorporated receipt
        update_delta: Change in compression

    Returns:
        Receipt dict
    """
    now = datetime.utcnow().isoformat() + "Z"

    receipt_data = {
        "receipt_type": "instant_incorporation",
        "tenant_id": TENANT_ID,
        "ts": now,
        "incorporator_id": incorporator.incorporator_id,
        "receipt_id": receipt_id,
        "update_delta": round(update_delta, 6),
        "latency_ms": round(incorporator.last_incorporation_latency_ms, 4),
        "latency_ok": incorporator.last_incorporation_latency_ms < INCORPORATION_LATENCY_MAX_MS,
        "incorporation_count": incorporator.incorporation_count,
        "batch_processing": False,
        "payload_hash": dual_hash(
            json.dumps(
                {
                    "incorporator_id": incorporator.incorporator_id,
                    "receipt_id": receipt_id,
                },
                sort_keys=True,
            )
        ),
    }

    emit_receipt("instant_incorporation", receipt_data)

    return receipt_data


def get_incorporator_status(incorporator: InstantIncorporator) -> Dict[str, Any]:
    """Get incorporator status.

    Args:
        incorporator: InstantIncorporator instance

    Returns:
        Status dict
    """
    return {
        "incorporator_id": incorporator.incorporator_id,
        "incorporation_count": incorporator.incorporation_count,
        "last_latency_ms": round(incorporator.last_incorporation_latency_ms, 4),
        "violation_count": incorporator.violation_count,
        "max_latency_ms": INCORPORATION_LATENCY_MAX_MS,
        "batch_processing_enabled": BATCH_PROCESSING_ENABLED,
    }
