"""D19.2 Path Compression Estimator - Estimate Future Path Compression.

PARADIGM: Estimate compression of projected paths for preemptive selection.

Grok's Insight:
  "High-future-compression paths are pre-amplified in today's selection.
   Low-future paths pre-starved before they waste cycles."

Selection is on PROJECTED future, not observed past.
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

from ..core import emit_receipt, dual_hash, TENANT_ID

# === COMPRESSION ESTIMATION CONSTANTS ===

HIGH_COMPRESSION_THRESHOLD = 0.85
"""Threshold for high-future-compression paths (pre-amplify)."""

LOW_COMPRESSION_THRESHOLD = 0.50
"""Threshold for low-future-compression paths (pre-starve)."""

MDL_ALPHA = 1.0
"""MDL alpha parameter for compression estimation."""

MDL_BETA = 0.10
"""MDL beta parameter for regularization."""


@dataclass
class CompressionEstimate:
    """Estimated compression for a projected path."""

    path_id: str
    current_compression: float
    projected_compression: float
    entropy_at_arrival: float
    classification: str  # "high", "medium", "low"
    recommendation: str  # "amplify", "neutral", "starve"


@dataclass
class PathCompressionEstimator:
    """Estimator for projected path compression."""

    estimator_id: str
    estimates: Dict[str, CompressionEstimate] = field(default_factory=dict)
    high_compression_paths: List[str] = field(default_factory=list)
    low_compression_paths: List[str] = field(default_factory=list)
    config: Dict = field(default_factory=dict)


def init_estimator(config: Dict = None) -> PathCompressionEstimator:
    """Initialize path compression estimator.

    Args:
        config: Optional configuration dict

    Returns:
        PathCompressionEstimator instance

    Receipt: estimator_init_receipt
    """
    config = config or {}
    estimator_id = str(uuid.uuid4())[:8]

    estimator = PathCompressionEstimator(
        estimator_id=estimator_id,
        config=config,
    )

    emit_receipt(
        "compression_estimator_init",
        {
            "receipt_type": "compression_estimator_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "estimator_id": estimator_id,
            "high_threshold": HIGH_COMPRESSION_THRESHOLD,
            "low_threshold": LOW_COMPRESSION_THRESHOLD,
            "payload_hash": dual_hash(
                json.dumps({"estimator_id": estimator_id}, sort_keys=True)
            ),
        },
    )

    return estimator


def estimate_path_compression(
    estimator: PathCompressionEstimator,
    path_id: str,
    current_entropy: float,
    projected_entropy: float,
    travel_time_years: float,
) -> CompressionEstimate:
    """Estimate compression for a single projected path.

    Uses MDL-based compression estimation projected into future.

    Args:
        estimator: PathCompressionEstimator instance
        path_id: Path identifier
        current_entropy: Current entropy at path origin
        projected_entropy: Projected entropy at arrival
        travel_time_years: Travel time in years

    Returns:
        CompressionEstimate instance

    Receipt: path_compression_estimate_receipt
    """
    # Current compression based on current entropy
    current_compression = math.exp(-current_entropy * MDL_BETA)
    current_compression = max(0.0, min(1.0, current_compression))

    # Projected compression based on future entropy
    # Account for entropy growth during travel
    entropy_growth_factor = 1 + (travel_time_years * 0.05)  # 5% per year
    adjusted_entropy = projected_entropy * entropy_growth_factor

    projected_compression = math.exp(-adjusted_entropy * MDL_BETA)
    projected_compression = max(0.0, min(1.0, projected_compression))

    # MDL score combining current and projected
    mdl_score = MDL_ALPHA * current_compression + (1 - MDL_ALPHA) * projected_compression

    # Classify path
    if projected_compression >= HIGH_COMPRESSION_THRESHOLD:
        classification = "high"
        recommendation = "amplify"
    elif projected_compression <= LOW_COMPRESSION_THRESHOLD:
        classification = "low"
        recommendation = "starve"
    else:
        classification = "medium"
        recommendation = "neutral"

    estimate = CompressionEstimate(
        path_id=path_id,
        current_compression=round(current_compression, 4),
        projected_compression=round(projected_compression, 4),
        entropy_at_arrival=round(adjusted_entropy, 6),
        classification=classification,
        recommendation=recommendation,
    )

    estimator.estimates[path_id] = estimate

    # Update classification lists
    if classification == "high":
        if path_id not in estimator.high_compression_paths:
            estimator.high_compression_paths.append(path_id)
    elif classification == "low":
        if path_id not in estimator.low_compression_paths:
            estimator.low_compression_paths.append(path_id)

    emit_receipt(
        "path_compression_estimate",
        {
            "receipt_type": "path_compression_estimate",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "estimator_id": estimator.estimator_id,
            "path_id": path_id,
            "current_compression": estimate.current_compression,
            "projected_compression": estimate.projected_compression,
            "classification": classification,
            "recommendation": recommendation,
            "payload_hash": dual_hash(
                json.dumps(
                    {"path_id": path_id, "projected": estimate.projected_compression},
                    sort_keys=True,
                )
            ),
        },
    )

    return estimate


def estimate_batch_compression(
    estimator: PathCompressionEstimator,
    paths: List[Dict],
) -> Dict[str, Any]:
    """Estimate compression for a batch of projected paths.

    Args:
        estimator: PathCompressionEstimator instance
        paths: List of path dicts with entropy and travel time

    Returns:
        Batch estimation result

    Receipt: batch_compression_estimate_receipt
    """
    estimates = []

    for path in paths:
        path_id = path.get("path_id", str(uuid.uuid4())[:8])
        current_entropy = path.get("current_entropy", 1.0)
        projected_entropy = path.get("projected_entropy", 1.5)
        travel_time = path.get("travel_time_years", 1.0)

        estimate = estimate_path_compression(
            estimator, path_id, current_entropy, projected_entropy, travel_time
        )
        estimates.append({
            "path_id": estimate.path_id,
            "projected_compression": estimate.projected_compression,
            "classification": estimate.classification,
            "recommendation": estimate.recommendation,
        })

    # Summary statistics
    high_count = sum(1 for e in estimates if e["classification"] == "high")
    low_count = sum(1 for e in estimates if e["classification"] == "low")
    avg_compression = sum(e["projected_compression"] for e in estimates) / len(estimates) if estimates else 0

    result = {
        "estimator_id": estimator.estimator_id,
        "paths_estimated": len(estimates),
        "high_compression_count": high_count,
        "low_compression_count": low_count,
        "avg_projected_compression": round(avg_compression, 4),
        "paths_to_amplify": high_count,
        "paths_to_starve": low_count,
    }

    emit_receipt(
        "batch_compression_estimate",
        {
            "receipt_type": "batch_compression_estimate",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def rank_by_projected_compression(
    estimator: PathCompressionEstimator,
) -> List[Tuple[str, float, str]]:
    """Rank all estimated paths by projected compression.

    Args:
        estimator: PathCompressionEstimator instance

    Returns:
        List of (path_id, projected_compression, recommendation) sorted descending

    Receipt: compression_ranking_receipt
    """
    rankings = [
        (e.path_id, e.projected_compression, e.recommendation)
        for e in estimator.estimates.values()
    ]

    # Sort by projected compression descending
    rankings.sort(key=lambda x: x[1], reverse=True)

    emit_receipt(
        "compression_ranking",
        {
            "receipt_type": "compression_ranking",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "estimator_id": estimator.estimator_id,
            "paths_ranked": len(rankings),
            "top_path": rankings[0][0] if rankings else None,
            "top_compression": rankings[0][1] if rankings else 0,
            "bottom_path": rankings[-1][0] if rankings else None,
            "bottom_compression": rankings[-1][1] if rankings else 0,
            "payload_hash": dual_hash(
                json.dumps({"count": len(rankings)}, sort_keys=True)
            ),
        },
    )

    return rankings


def get_estimator_status() -> Dict[str, Any]:
    """Get estimator module status.

    Returns:
        Status dict
    """
    return {
        "module": "projection.path_compression_estimator",
        "version": "19.2.0",
        "high_compression_threshold": HIGH_COMPRESSION_THRESHOLD,
        "low_compression_threshold": LOW_COMPRESSION_THRESHOLD,
        "mdl_alpha": MDL_ALPHA,
        "mdl_beta": MDL_BETA,
        "selection_mode": "projected_future",
        "selection_on_past": False,
    }
