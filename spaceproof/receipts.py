"""receipts.py - Receipt Emission Helpers

DRY helpers for common receipt emission patterns.
All receipts comply with CLAUDEME S8 schema.

Usage:
    from .receipts import emit_with_hash, emit_anomaly, emit_spec_ingest

    # Instead of manual emit_receipt + dual_hash boilerplate:
    emit_with_hash("my_receipt", tenant_id, {
        "field1": value1,
        "field2": value2
    })
"""

import json
from typing import Dict, Any, Optional

from .core import emit_receipt, dual_hash


def emit_with_hash(
    receipt_type: str,
    tenant_id: str,
    data: Dict[str, Any],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit receipt with automatic payload_hash computation.

    Standard pattern for most receipts. Computes dual_hash of data
    and includes it as payload_hash.

    Args:
        receipt_type: Receipt type string (e.g., "entropy_pruning")
        tenant_id: Tenant identifier (e.g., "spaceproof-pruning")
        data: Receipt payload data
        extra_fields: Optional additional fields to include in receipt

    Returns:
        Complete receipt dict

    Example:
        emit_with_hash("cache_depth", "spaceproof-gnn-cache", {
            "blackout_days": 90,
            "utilization_pct": 0.45
        })
    """
    payload_hash = dual_hash(json.dumps(data, sort_keys=True))

    receipt_data = {"tenant_id": tenant_id, **data, "payload_hash": payload_hash}

    if extra_fields:
        receipt_data.update(extra_fields)

    return emit_receipt(receipt_type, receipt_data)


def emit_anomaly(
    tenant_id: str,
    metric: str,
    baseline: float,
    delta: float,
    classification: str,
    action: str,
    **extra,
) -> Dict[str, Any]:
    """Emit anomaly receipt for deviation detection.

    Standard pattern for anomaly/stoprule receipts.

    Args:
        tenant_id: Tenant identifier
        metric: Metric name (e.g., "hash_mismatch", "predictive_confidence")
        baseline: Expected/baseline value
        delta: Deviation from baseline
        classification: "deviation", "violation", "anti_pattern"
        action: "halt", "investigate", "skip_predictive"
        **extra: Additional context fields

    Returns:
        Complete anomaly receipt dict

    Example:
        emit_anomaly(
            "spaceproof-pruning",
            metric="trim_factor",
            baseline=0.6,
            delta=0.1,
            classification="violation",
            action="halt"
        )
    """
    data = {
        "metric": metric,
        "baseline": baseline,
        "delta": delta,
        "classification": classification,
        "action": action,
        **extra,
    }

    return emit_with_hash("anomaly", tenant_id, data)


def emit_spec_ingest(
    receipt_type: str,
    tenant_id: str,
    file_path: str,
    spec_data: Dict[str, Any],
    key_fields: Optional[list] = None,
) -> Dict[str, Any]:
    """Emit spec file ingest receipt.

    Standard pattern for loading JSON specification files.

    Args:
        receipt_type: Receipt type (e.g., "entropy_pruning_spec_ingest")
        tenant_id: Tenant identifier
        file_path: Path to spec file
        spec_data: Loaded specification data
        key_fields: List of key field names to include in receipt

    Returns:
        Complete ingest receipt dict

    Example:
        emit_spec_ingest(
            "gnn_cache_spec_ingest",
            "spaceproof-gnn-cache",
            "/path/to/spec.json",
            loaded_data,
            key_fields=["asymptote_alpha", "overflow_threshold_days"]
        )
    """
    content_hash = dual_hash(json.dumps(spec_data, sort_keys=True))

    receipt_data = {"file_path": file_path, "payload_hash": content_hash}

    # Include specified key fields from spec
    if key_fields:
        for field in key_fields:
            if field in spec_data:
                receipt_data[field] = spec_data[field]

    return emit_receipt(receipt_type, {"tenant_id": tenant_id, **receipt_data})


def emit_result(
    receipt_type: str,
    tenant_id: str,
    result: Dict[str, Any],
    hash_fields: Optional[list] = None,
) -> Dict[str, Any]:
    """Emit result receipt with selective hashing.

    For cases where you want to hash only specific fields
    (e.g., excluding large objects like pruned_tree).

    Args:
        receipt_type: Receipt type string
        tenant_id: Tenant identifier
        result: Full result dict
        hash_fields: Fields to include in hash (if None, uses all)

    Returns:
        Complete receipt dict

    Example:
        emit_result("entropy_pruning", "spaceproof-pruning", result,
            hash_fields=["merkle_root_before", "merkle_root_after", "branches_pruned"]
        )
    """
    if hash_fields:
        hash_data = {k: result[k] for k in hash_fields if k in result}
    else:
        # Exclude common non-hashable fields
        hash_data = {
            k: v
            for k, v in result.items()
            if k not in ("pruned_tree", "tree", "leaves")
        }

    payload_hash = dual_hash(json.dumps(hash_data, sort_keys=True))

    receipt_data = {
        "tenant_id": tenant_id,
        **{
            k: v
            for k, v in result.items()
            if k not in ("pruned_tree", "tree", "leaves")
        },
        "payload_hash": payload_hash,
    }

    return emit_receipt(receipt_type, receipt_data)
