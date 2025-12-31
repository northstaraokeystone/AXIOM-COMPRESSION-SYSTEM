"""raci.py - RACI matrix assignment per event type.

RACI = Responsible, Accountable, Consulted, Informed.
Every event must have explicit accountability chain.
Assigns ownership roles for audit and compliance tracking.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

GOVERNANCE_TENANT = "spaceproof-governance"
CONFIG_DIR = Path(__file__).parent.parent / "config"
RACI_MATRIX_FILE = CONFIG_DIR / "raci_matrix.json"

# Cache for RACI matrix
_raci_cache: Optional[Dict[str, Any]] = None


@dataclass
class RACIAssignment:
    """RACI assignment for an event."""

    event_id: str
    event_type: str
    responsible: str
    accountable: str
    consulted: List[str]
    informed: List[str]
    escalation_path: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "responsible": self.responsible,
            "accountable": self.accountable,
            "consulted": self.consulted,
            "informed": self.informed,
            "escalation_path": self.escalation_path,
            "timestamp": self.timestamp,
        }


def load_raci_matrix() -> Dict[str, Any]:
    """Load RACI matrix from config file.

    Returns:
        RACI matrix configuration

    Caches result in memory for performance.
    """
    global _raci_cache

    if _raci_cache is not None:
        return _raci_cache

    if not RACI_MATRIX_FILE.exists():
        # Return default matrix if file not found
        _raci_cache = {
            "event_types": {
                "default": {
                    "responsible": "SYSTEM",
                    "accountable": "OPERATOR",
                    "consulted": [],
                    "informed": ["AUDIT_TEAM"],
                }
            },
            "escalation_chains": {
                "critical": ["SAFETY_OFFICER", "COMMANDING_OFFICER"],
                "high": ["TEAM_LEAD"],
                "medium": [],
                "low": [],
            },
        }
        return _raci_cache

    with open(RACI_MATRIX_FILE, "r") as f:
        _raci_cache = json.load(f)

    return _raci_cache


def get_raci_for_event(event_type: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get RACI assignment for an event type.

    Args:
        event_type: Type of event (e.g., "orbital_compute", "human_override")
        context: Optional context for dynamic RACI resolution

    Returns:
        Dict with responsible, accountable, consulted, informed fields
    """
    matrix = load_raci_matrix()
    event_types = matrix.get("event_types", {})

    # Look up event type, fall back to default
    raci = event_types.get(event_type, event_types.get("default", {}))

    if not raci:
        # Ultimate fallback
        raci = {
            "responsible": "SYSTEM",
            "accountable": "OPERATOR",
            "consulted": [],
            "informed": [],
        }

    # Apply context overrides if provided
    if context:
        if "override_responsible" in context:
            raci = {**raci, "responsible": context["override_responsible"]}
        if "override_accountable" in context:
            raci = {**raci, "accountable": context["override_accountable"]}

    return raci


def validate_raci(raci: Dict[str, Any]) -> bool:
    """Verify all RACI roles are present.

    Args:
        raci: RACI assignment dictionary

    Returns:
        True if valid (all 4 fields present)
    """
    required_fields = ["responsible", "accountable", "consulted", "informed"]
    return all(field in raci for field in required_fields)


def emit_raci_receipt(
    event_id: str,
    event_type: str,
    raci: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Document RACI accountability assignment.

    Args:
        event_id: Unique event identifier
        event_type: Type of event
        raci: RACI assignment
        context: Optional additional context

    Returns:
        Receipt dict with dual-hash
    """
    if not event_id:
        event_id = str(uuid.uuid4())

    # Get escalation path based on context severity
    matrix = load_raci_matrix()
    severity = (context or {}).get("severity", "low")
    escalation_chains = matrix.get("escalation_chains", {})
    escalation_path = escalation_chains.get(severity.lower(), [])

    assignment = RACIAssignment(
        event_id=event_id,
        event_type=event_type,
        responsible=raci.get("responsible", "UNKNOWN"),
        accountable=raci.get("accountable", "UNKNOWN"),
        consulted=raci.get("consulted", []),
        informed=raci.get("informed", []),
        escalation_path=escalation_path,
    )

    receipt_data = {
        "tenant_id": GOVERNANCE_TENANT,
        **assignment.to_dict(),
    }

    if context:
        receipt_data["context"] = context

    return emit_receipt("raci_assignment", receipt_data)


def create_raci_assignment(
    event_type: str,
    context: Optional[Dict[str, Any]] = None,
    event_id: Optional[str] = None,
) -> RACIAssignment:
    """Create and emit a complete RACI assignment.

    Args:
        event_type: Type of event
        context: Optional context for resolution
        event_id: Optional event ID (generated if not provided)

    Returns:
        RACIAssignment with receipt emitted
    """
    if not event_id:
        event_id = str(uuid.uuid4())

    raci = get_raci_for_event(event_type, context)

    # Validate before emitting
    if not validate_raci(raci):
        raise ValueError(f"Invalid RACI for event type: {event_type}")

    # Get escalation path
    matrix = load_raci_matrix()
    severity = (context or {}).get("severity", "low")
    escalation_path = matrix.get("escalation_chains", {}).get(severity.lower(), [])

    # Emit receipt
    emit_raci_receipt(event_id, event_type, raci, context)

    return RACIAssignment(
        event_id=event_id,
        event_type=event_type,
        responsible=raci["responsible"],
        accountable=raci["accountable"],
        consulted=raci.get("consulted", []),
        informed=raci.get("informed", []),
        escalation_path=escalation_path,
    )


def clear_raci_cache() -> None:
    """Clear the RACI matrix cache (for testing)."""
    global _raci_cache
    _raci_cache = None
