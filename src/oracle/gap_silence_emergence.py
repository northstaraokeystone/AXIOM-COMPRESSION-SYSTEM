"""D19.3 Gap-Silence Law Emergence.

Purpose: Use interstellar latency gaps as selection pressure.
Long silence forces minimal-sync laws.

The Physics:
  Interstellar latency silence becomes selection pressure—long gaps force
  emergence of minimal-sync laws from prior chain alone. Laws that survive
  silence = invariants.

Gap Classification:
  - Normal: < 60 seconds (local network)
  - Extended: 60 seconds - 1 hour (planetary)
  - Interstellar: > 4 years (Proxima Centauri scale)
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash, TENANT_ID


# === D19.3 GAP-SILENCE CONSTANTS ===

LATENCY_SILENCE_THRESHOLD_SECONDS = 60
"""Normal gap threshold in seconds."""

EXTENDED_SILENCE_THRESHOLD_SECONDS = 3600
"""Extended gap threshold (1 hour)."""

INTERSTELLAR_SILENCE_THRESHOLD_YEARS = 4.0
"""Interstellar gap threshold in years."""

PROXIMA_SILENCE_YEARS = 8.48
"""Proxima Centauri round-trip time in years."""


@dataclass
class GapSilenceEmergence:
    """Use interstellar latency gaps as selection pressure.

    Long gaps force minimal-sync laws that require no state exchange.
    Silence doesn't break the system; silence purifies the laws.
    """

    detector_id: str
    latency_catalog: Dict[str, float] = field(default_factory=dict)
    normal_threshold_seconds: float = LATENCY_SILENCE_THRESHOLD_SECONDS
    extended_threshold_seconds: float = EXTENDED_SILENCE_THRESHOLD_SECONDS
    interstellar_threshold_years: float = INTERSTELLAR_SILENCE_THRESHOLD_YEARS
    gaps_detected: List[Dict] = field(default_factory=list)
    emerged_laws: List[Dict] = field(default_factory=list)


def init_gap_detector(latency_catalog: Dict[str, float] = None) -> GapSilenceEmergence:
    """Load known latencies. Set thresholds.

    Args:
        latency_catalog: Dict of destination -> latency in seconds

    Returns:
        GapSilenceEmergence instance
    """
    detector_id = str(uuid.uuid4())[:8]

    if latency_catalog is None:
        # Default catalog with known interstellar latencies
        latency_catalog = {
            "local": 0.001,
            "mars_min": 182.0,  # 3 minutes
            "mars_max": 1320.0,  # 22 minutes
            "jupiter": 2520.0,  # 42 minutes
            "saturn": 4800.0,  # 80 minutes
            "proxima_one_way_years": 4.24,
            "proxima_rtt_years": 8.48,
        }

    return GapSilenceEmergence(
        detector_id=detector_id,
        latency_catalog=latency_catalog,
    )


def detect_gap(
    detector: GapSilenceEmergence, last_receipt_ts: str, now: str = None
) -> Dict[str, Any]:
    """Identify gap duration and type.

    Args:
        detector: GapSilenceEmergence instance
        last_receipt_ts: Timestamp of last receipt
        now: Current timestamp (default: now)

    Returns:
        Gap dict with duration and type
    """
    if now is None:
        now = datetime.utcnow().isoformat() + "Z"

    # Parse timestamps
    try:
        last_ts = datetime.fromisoformat(last_receipt_ts.rstrip("Z"))
        current_ts = datetime.fromisoformat(now.rstrip("Z"))
        gap_seconds = (current_ts - last_ts).total_seconds()
    except (ValueError, TypeError):
        gap_seconds = 0.0

    gap_type = classify_gap(detector, gap_seconds)

    gap = {
        "gap_id": str(uuid.uuid4())[:8],
        "last_receipt_ts": last_receipt_ts,
        "current_ts": now,
        "gap_seconds": gap_seconds,
        "gap_years": gap_seconds / (365.25 * 24 * 3600),
        "gap_type": gap_type,
        "detected_at": now,
    }

    detector.gaps_detected.append(gap)

    return gap


def classify_gap(detector: GapSilenceEmergence, gap_seconds: float) -> str:
    """Return "normal", "extended", "interstellar".

    Args:
        detector: GapSilenceEmergence instance
        gap_seconds: Gap duration in seconds

    Returns:
        Gap classification string
    """
    gap_years = gap_seconds / (365.25 * 24 * 3600)

    if gap_years >= detector.interstellar_threshold_years:
        return "interstellar"
    elif gap_seconds >= detector.extended_threshold_seconds:
        return "extended"
    elif gap_seconds >= detector.normal_threshold_seconds:
        return "normal"
    else:
        return "negligible"


def trigger_minimal_law_selection(
    detector: GapSilenceEmergence, oracle: Any, gap_type: str
) -> List[Dict]:
    """Select laws that survive this gap duration.

    Args:
        detector: GapSilenceEmergence instance
        oracle: LiveHistoryOracle instance
        gap_type: Type of gap ("normal", "extended", "interstellar")

    Returns:
        List of laws that survive the gap
    """
    if not oracle or not hasattr(oracle, "laws"):
        return []

    all_laws = oracle.laws
    gap_duration = _gap_type_to_duration(detector, gap_type)

    surviving_laws = minimal_sync_law(all_laws, gap_duration)

    # Record emerged laws
    detector.emerged_laws = surviving_laws

    return surviving_laws


def minimal_sync_law(laws: List[Dict], gap_duration: float) -> List[Dict]:
    """Filter to laws that require minimum state.

    Longer gaps = stricter law selection.
    Interstellar gaps (years) = only fundamental invariants survive.

    Args:
        laws: List of law dicts
        gap_duration: Gap duration in seconds

    Returns:
        List of minimal-sync laws
    """
    if not laws:
        return []

    # Calculate strictness based on gap duration
    gap_years = gap_duration / (365.25 * 24 * 3600)

    if gap_years >= 4.0:  # Interstellar
        # Only fundamental invariants survive
        # Select laws with highest invariance scores
        min_invariance = 0.8
    elif gap_duration >= 3600:  # Extended (1 hour+)
        min_invariance = 0.6
    elif gap_duration >= 60:  # Normal (1 minute+)
        min_invariance = 0.4
    else:
        # All laws survive negligible gaps
        return laws

    surviving = []
    for law in laws:
        invariance = law.get(
            "invariance_score", law.get("compression_contribution", 0.5)
        )
        if invariance >= min_invariance:
            law["survived_gap_years"] = gap_years
            law["minimal_sync"] = True
            surviving.append(law)

    return surviving


def _gap_type_to_duration(detector: GapSilenceEmergence, gap_type: str) -> float:
    """Convert gap type to duration in seconds.

    Args:
        detector: GapSilenceEmergence instance
        gap_type: Gap type string

    Returns:
        Duration in seconds
    """
    if gap_type == "interstellar":
        return detector.interstellar_threshold_years * 365.25 * 24 * 3600
    elif gap_type == "extended":
        return detector.extended_threshold_seconds
    elif gap_type == "normal":
        return detector.normal_threshold_seconds
    else:
        return 0.0


def emit_gap_emergence_receipt(
    detector: GapSilenceEmergence, gap: Dict, emerged_laws: List[Dict]
) -> Dict[str, Any]:
    """Emit gap_silence_emergence_receipt.

    Args:
        detector: GapSilenceEmergence instance
        gap: Gap dict
        emerged_laws: Laws that emerged/survived

    Returns:
        Receipt dict
    """
    now = datetime.utcnow().isoformat() + "Z"

    receipt_data = {
        "receipt_type": "gap_silence_emergence",
        "tenant_id": TENANT_ID,
        "ts": now,
        "detector_id": detector.detector_id,
        "gap_id": gap.get("gap_id"),
        "gap_type": gap.get("gap_type"),
        "gap_seconds": gap.get("gap_seconds"),
        "gap_years": gap.get("gap_years"),
        "emerged_laws_count": len(emerged_laws),
        "law_ids": [law.get("law_id") for law in emerged_laws[:10]],
        "selection_pressure": "silence",
        "insight": "Long gaps force emergence of minimal-sync laws from prior chain alone",
        "payload_hash": dual_hash(
            json.dumps(
                {
                    "detector_id": detector.detector_id,
                    "gap_type": gap.get("gap_type"),
                    "emerged_count": len(emerged_laws),
                },
                sort_keys=True,
            )
        ),
    }

    emit_receipt("gap_silence_emergence", receipt_data)

    return receipt_data


def get_gap_detector_status(detector: GapSilenceEmergence) -> Dict[str, Any]:
    """Get gap detector status.

    Args:
        detector: GapSilenceEmergence instance

    Returns:
        Status dict
    """
    return {
        "detector_id": detector.detector_id,
        "normal_threshold_seconds": detector.normal_threshold_seconds,
        "extended_threshold_seconds": detector.extended_threshold_seconds,
        "interstellar_threshold_years": detector.interstellar_threshold_years,
        "proxima_silence_years": PROXIMA_SILENCE_YEARS,
        "gaps_detected": len(detector.gaps_detected),
        "emerged_laws": len(detector.emerged_laws),
        "insight": "Silence doesn't break the system; silence purifies the laws",
    }
