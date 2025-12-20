"""D19.2 Preemptive Weave - Pre-Amplify/Starve Based on Projected Compression.

PARADIGM: Selection based on PROJECTED future, not observed past.

Grok's Insight:
  "High-future-compression paths are pre-amplified in today's selection.
   Low-future paths pre-starved before they waste cycles."

KILLED:
  - Reactive law enforcement (replaced by preemptive weaving)
  - Darwinian selection on OBSERVED fitness (replaced by PROJECTED future fitness)
  - Post-event pattern detection (replaced by pre-amplify/starve BEFORE)
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19.2 PREEMPTIVE WEAVE CONSTANTS ===

PREEMPTIVE_AMPLIFY_THRESHOLD = 0.85
"""Threshold for pre-amplification (high-future-compression)."""

PREEMPTIVE_STARVE_THRESHOLD = 0.50
"""Threshold for pre-starvation (low-future-compression)."""

AMPLIFICATION_FACTOR = 2.0
"""Factor to amplify high-future paths."""

STARVATION_FACTOR = 0.1
"""Factor to starve low-future paths."""

# KILLED: Reactive mode
REACTIVE_MODE_ENABLED = False
"""Reactive mode KILLED - preemptive only."""

# KILLED: Selection on observed past
SELECTION_ON_PAST = False
"""Selection on past KILLED - projected future only."""


@dataclass
class PreemptiveSelection:
    """A preemptive selection decision."""

    selection_id: str
    path_id: str
    projected_compression: float
    action: str  # "amplify" or "starve"
    factor: float
    reason: str
    woven_at: str


@dataclass
class PreemptiveWeave:
    """Preemptive weave engine state."""

    weave_id: str
    selections: Dict[str, PreemptiveSelection] = field(default_factory=dict)
    amplified_paths: List[str] = field(default_factory=list)
    starved_paths: List[str] = field(default_factory=list)
    total_amplified: int = 0
    total_starved: int = 0
    config: Dict = field(default_factory=dict)


def init_preemptive_weave(config: Dict = None) -> PreemptiveWeave:
    """Initialize preemptive weave engine.

    Args:
        config: Optional configuration dict

    Returns:
        PreemptiveWeave instance

    Receipt: preemptive_weave_init_receipt
    """
    config = config or {}
    weave_id = str(uuid.uuid4())[:8]

    weave = PreemptiveWeave(weave_id=weave_id, config=config)

    emit_receipt(
        "preemptive_weave_init",
        {
            "receipt_type": "preemptive_weave_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "weave_id": weave_id,
            "amplify_threshold": PREEMPTIVE_AMPLIFY_THRESHOLD,
            "starve_threshold": PREEMPTIVE_STARVE_THRESHOLD,
            "reactive_mode": REACTIVE_MODE_ENABLED,
            "selection_on_past": SELECTION_ON_PAST,
            "payload_hash": dual_hash(
                json.dumps({"weave_id": weave_id}, sort_keys=True)
            ),
        },
    )

    return weave


def amplify_high_future_paths(
    weave: PreemptiveWeave,
    paths: List[Dict],
) -> List[PreemptiveSelection]:
    """Pre-amplify paths with high projected compression.

    Selection based on PROJECTED future, not observed past.

    Args:
        weave: PreemptiveWeave instance
        paths: List of path dicts with projected_compression

    Returns:
        List of amplification selections

    Receipt: path_amplification_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"
    amplifications = []

    for path in paths:
        projected = path.get("projected_compression", 0)
        path_id = path.get("path_id", str(uuid.uuid4())[:8])

        if projected >= PREEMPTIVE_AMPLIFY_THRESHOLD:
            selection_id = str(uuid.uuid4())[:8]

            selection = PreemptiveSelection(
                selection_id=selection_id,
                path_id=path_id,
                projected_compression=projected,
                action="amplify",
                factor=AMPLIFICATION_FACTOR,
                reason=f"projected_compression_{projected:.4f}_>=_{PREEMPTIVE_AMPLIFY_THRESHOLD}",
                woven_at=now,
            )

            weave.selections[selection_id] = selection
            weave.amplified_paths.append(path_id)
            weave.total_amplified += 1
            amplifications.append(selection)

    if amplifications:
        emit_receipt(
            "path_amplification",
            {
                "receipt_type": "path_amplification",
                "tenant_id": TENANT_ID,
                "ts": now,
                "weave_id": weave.weave_id,
                "paths_amplified": len(amplifications),
                "amplification_factor": AMPLIFICATION_FACTOR,
                "threshold": PREEMPTIVE_AMPLIFY_THRESHOLD,
                "selection_basis": "projected_future",
                "reactive_mode": False,
                "payload_hash": dual_hash(
                    json.dumps({"count": len(amplifications)}, sort_keys=True)
                ),
            },
        )

    return amplifications


def starve_low_future_paths(
    weave: PreemptiveWeave,
    paths: List[Dict],
) -> List[PreemptiveSelection]:
    """Pre-starve paths with low projected compression.

    Low-future paths pre-starved BEFORE they waste cycles.

    Args:
        weave: PreemptiveWeave instance
        paths: List of path dicts with projected_compression

    Returns:
        List of starvation selections

    Receipt: path_starvation_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"
    starvations = []

    for path in paths:
        projected = path.get("projected_compression", 0)
        path_id = path.get("path_id", str(uuid.uuid4())[:8])

        if projected <= PREEMPTIVE_STARVE_THRESHOLD:
            selection_id = str(uuid.uuid4())[:8]

            selection = PreemptiveSelection(
                selection_id=selection_id,
                path_id=path_id,
                projected_compression=projected,
                action="starve",
                factor=STARVATION_FACTOR,
                reason=f"projected_compression_{projected:.4f}_<=_{PREEMPTIVE_STARVE_THRESHOLD}",
                woven_at=now,
            )

            weave.selections[selection_id] = selection
            weave.starved_paths.append(path_id)
            weave.total_starved += 1
            starvations.append(selection)

    if starvations:
        emit_receipt(
            "path_starvation",
            {
                "receipt_type": "path_starvation",
                "tenant_id": TENANT_ID,
                "ts": now,
                "weave_id": weave.weave_id,
                "paths_starved": len(starvations),
                "starvation_factor": STARVATION_FACTOR,
                "threshold": PREEMPTIVE_STARVE_THRESHOLD,
                "selection_basis": "projected_future",
                "reactive_mode": False,
                "payload_hash": dual_hash(
                    json.dumps({"count": len(starvations)}, sort_keys=True)
                ),
            },
        )

    return starvations


def apply_preemptive_selection(
    weave: PreemptiveWeave,
    paths: List[Dict],
) -> Dict[str, Any]:
    """Apply preemptive selection to all paths.

    Amplify high-future-compression, starve low-future-compression.
    Selection on PROJECTED future, not observed past.

    Args:
        weave: PreemptiveWeave instance
        paths: List of path dicts with projected_compression

    Returns:
        Selection result dict

    Receipt: preemptive_selection_receipt
    """
    now = datetime.utcnow()

    # Apply amplification
    amplifications = amplify_high_future_paths(weave, paths)

    # Apply starvation
    starvations = starve_low_future_paths(weave, paths)

    # Neutral paths (between thresholds)
    neutral_count = len(paths) - len(amplifications) - len(starvations)

    result = {
        "weave_id": weave.weave_id,
        "paths_evaluated": len(paths),
        "paths_amplified": len(amplifications),
        "paths_starved": len(starvations),
        "paths_neutral": neutral_count,
        "amplification_factor": AMPLIFICATION_FACTOR,
        "starvation_factor": STARVATION_FACTOR,
        "selection_basis": "projected_future",
        "reactive_mode": REACTIVE_MODE_ENABLED,
        "selection_on_past": SELECTION_ON_PAST,
    }

    emit_receipt(
        "preemptive_selection",
        {
            "receipt_type": "preemptive_selection",
            "tenant_id": TENANT_ID,
            "ts": now.isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_weave_status() -> Dict[str, Any]:
    """Get preemptive weave module status.

    Returns:
        Status dict
    """
    return {
        "module": "weave.preemptive_weave",
        "version": "19.2.0",
        "paradigm": "preemptive_selection",
        "amplify_threshold": PREEMPTIVE_AMPLIFY_THRESHOLD,
        "starve_threshold": PREEMPTIVE_STARVE_THRESHOLD,
        "amplification_factor": AMPLIFICATION_FACTOR,
        "starvation_factor": STARVATION_FACTOR,
        "reactive_mode_enabled": REACTIVE_MODE_ENABLED,
        "selection_on_past": SELECTION_ON_PAST,
        "killed": [
            "reactive_law_enforcement",
            "selection_on_observed_fitness",
            "post_event_pattern_detection",
        ],
        "insight": "High-future-compression paths pre-amplified, low-future paths pre-starved BEFORE",
    }
