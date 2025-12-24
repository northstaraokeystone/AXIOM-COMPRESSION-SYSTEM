"""D19.2 Delay Nullification - Laws That Nullify Delay Before Arrival.

PARADIGM: Preemptive laws woven into current chain nullify delay.

Grok's Insight:
  "system weaves laws that nullify delay before it arrives"

The Physics (Block Universe):
  The delay appears to vanish because we've already accounted for it.
  We know the future (deterministic trajectory), so we weave compensation
  laws into the present chain. When the "delay" resolves, we're already there.

Verification:
  Future-projected state matches when delay resolves.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from src.core import emit_receipt, dual_hash, TENANT_ID

# === D19.2 DELAY NULLIFICATION CONSTANTS ===

PROXIMA_RTT_YEARS = 8.48
"""Proxima Centauri RTT for primary nullification target."""

NULLIFICATION_MODES = ["preemptive", "anticipatory", "compensatory"]
"""Available nullification modes."""


@dataclass
class PreemptiveLaw:
    """A preemptive law that nullifies delay."""

    law_id: str
    law_type: str
    delay_nullified_years: float
    delay_nullified_seconds: float
    target_destination: str
    woven_into_chain: bool
    verification_pending: bool
    created_at: str


@dataclass
class DelayNullification:
    """Delay nullification engine state."""

    nullification_id: str
    preemptive_laws: Dict[str, PreemptiveLaw] = field(default_factory=dict)
    delays_nullified: List[Dict] = field(default_factory=list)
    total_years_nullified: float = 0.0
    verifications_pending: int = 0
    verifications_passed: int = 0
    config: Dict = field(default_factory=dict)


def init_nullification(config: Dict = None) -> DelayNullification:
    """Initialize delay nullification engine.

    Args:
        config: Optional configuration dict

    Returns:
        DelayNullification instance

    Receipt: nullification_init_receipt
    """
    config = config or {}
    nullification_id = str(uuid.uuid4())[:8]

    nullification = DelayNullification(
        nullification_id=nullification_id,
        config=config,
    )

    emit_receipt(
        "nullification_init",
        {
            "receipt_type": "nullification_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "nullification_id": nullification_id,
            "mode": "preemptive",
            "simulation_enabled": False,
            "reactive_mode": False,
            "payload_hash": dual_hash(
                json.dumps({"nullification_id": nullification_id}, sort_keys=True)
            ),
        },
    )

    return nullification


def nullify_known_delay(
    nullification: DelayNullification,
    destination: str,
    delay_years: float,
) -> PreemptiveLaw:
    """Nullify a known delay by weaving preemptive law.

    The delay appears to vanish because we've already accounted for it.

    Args:
        nullification: DelayNullification instance
        destination: Target destination
        delay_years: Delay to nullify in years

    Returns:
        PreemptiveLaw instance

    Receipt: delay_nullification_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"
    law_id = str(uuid.uuid4())[:8]

    delay_seconds = delay_years * 365.25 * 24 * 3600

    law = PreemptiveLaw(
        law_id=law_id,
        law_type="delay_nullification",
        delay_nullified_years=delay_years,
        delay_nullified_seconds=delay_seconds,
        target_destination=destination,
        woven_into_chain=True,
        verification_pending=True,
        created_at=now,
    )

    nullification.preemptive_laws[law_id] = law
    nullification.delays_nullified.append(
        {
            "law_id": law_id,
            "destination": destination,
            "delay_years": delay_years,
            "nullified_at": now,
        }
    )
    nullification.total_years_nullified += delay_years
    nullification.verifications_pending += 1

    emit_receipt(
        "delay_nullification",
        {
            "receipt_type": "delay_nullification",
            "tenant_id": TENANT_ID,
            "ts": now,
            "nullification_id": nullification.nullification_id,
            "law_id": law_id,
            "destination": destination,
            "delay_nullified_years": round(delay_years, 4),
            "delay_nullified_seconds": round(delay_seconds, 2),
            "woven_into_chain": True,
            "verification_pending": True,
            "insight": "delay appears to vanish - already accounted for",
            "payload_hash": dual_hash(
                json.dumps(
                    {"law_id": law_id, "destination": destination}, sort_keys=True
                )
            ),
        },
    )

    return law


def generate_preemptive_law(
    nullification: DelayNullification,
    law_type: str,
    destination: str,
    delay_years: float,
    additional_params: Dict = None,
) -> PreemptiveLaw:
    """Generate a preemptive law for delay nullification.

    Args:
        nullification: DelayNullification instance
        law_type: Type of preemptive law
        destination: Target destination
        delay_years: Delay to nullify
        additional_params: Optional additional parameters

    Returns:
        PreemptiveLaw instance

    Receipt: preemptive_law_generation_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"
    law_id = str(uuid.uuid4())[:8]

    delay_seconds = delay_years * 365.25 * 24 * 3600

    law = PreemptiveLaw(
        law_id=law_id,
        law_type=law_type,
        delay_nullified_years=delay_years,
        delay_nullified_seconds=delay_seconds,
        target_destination=destination,
        woven_into_chain=False,
        verification_pending=True,
        created_at=now,
    )

    nullification.preemptive_laws[law_id] = law
    nullification.verifications_pending += 1

    emit_receipt(
        "preemptive_law_generation",
        {
            "receipt_type": "preemptive_law_generation",
            "tenant_id": TENANT_ID,
            "ts": now,
            "nullification_id": nullification.nullification_id,
            "law_id": law_id,
            "law_type": law_type,
            "destination": destination,
            "delay_years": round(delay_years, 4),
            "additional_params": additional_params or {},
            "payload_hash": dual_hash(
                json.dumps({"law_id": law_id, "type": law_type}, sort_keys=True)
            ),
        },
    )

    return law


def verify_nullification(
    nullification: DelayNullification,
    law_id: str,
    projected_state: Dict,
    resolved_state: Dict,
) -> Dict[str, Any]:
    """Verify nullification by comparing projected and resolved states.

    When the "delay" resolves, the state should match what we projected.
    If it matches, the delay was successfully nullified.

    Args:
        nullification: DelayNullification instance
        law_id: Law to verify
        projected_state: State we projected
        resolved_state: State when delay resolved

    Returns:
        Verification result dict

    Receipt: nullification_verification_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    if law_id not in nullification.preemptive_laws:
        return {"error": "law_not_found", "law_id": law_id}

    law = nullification.preemptive_laws[law_id]

    # Compare states
    projected_hash = dual_hash(json.dumps(projected_state, sort_keys=True))
    resolved_hash = dual_hash(json.dumps(resolved_state, sort_keys=True))

    states_match = projected_hash == resolved_hash

    # Calculate state divergence (simplified)
    divergence = 0.0
    if not states_match:
        # Calculate approximate divergence
        projected_keys = set(projected_state.keys())
        resolved_keys = set(resolved_state.keys())
        common_keys = projected_keys & resolved_keys
        if common_keys:
            matches = sum(
                1
                for k in common_keys
                if projected_state.get(k) == resolved_state.get(k)
            )
            divergence = 1.0 - (matches / len(common_keys))

    # Update verification status
    law.verification_pending = False
    nullification.verifications_pending -= 1

    verification_passed = states_match or divergence < 0.05

    if verification_passed:
        nullification.verifications_passed += 1

    result = {
        "law_id": law_id,
        "verification_passed": verification_passed,
        "states_match": states_match,
        "state_divergence": round(divergence, 4),
        "delay_nullified_years": law.delay_nullified_years,
        "destination": law.target_destination,
        "insight": "delay nullified - projected state matched"
        if verification_passed
        else "divergence detected",
    }

    emit_receipt(
        "nullification_verification",
        {
            "receipt_type": "nullification_verification",
            "tenant_id": TENANT_ID,
            "ts": now,
            "nullification_id": nullification.nullification_id,
            **result,
            "projected_hash": projected_hash[:32],
            "resolved_hash": resolved_hash[:32],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_nullification_status() -> Dict[str, Any]:
    """Get nullification module status.

    Returns:
        Status dict
    """
    return {
        "module": "weave.delay_nullification",
        "version": "19.2.0",
        "paradigm": "preemptive_nullification",
        "proxima_rtt_years": PROXIMA_RTT_YEARS,
        "nullification_modes": NULLIFICATION_MODES,
        "simulation_enabled": False,
        "reactive_mode": False,
        "insight": "system weaves laws that nullify delay before it arrives",
    }
