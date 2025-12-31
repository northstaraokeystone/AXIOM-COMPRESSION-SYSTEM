"""reason_codes.py - Structured codes for human interventions.

10 structured reason codes (RE001-RE010) for categorizing human corrections.
Every override/correction must include a valid reason code for training pipeline.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

GOVERNANCE_TENANT = "spaceproof-governance"
CONFIG_DIR = Path(__file__).parent.parent / "config"
REASON_CODES_FILE = CONFIG_DIR / "reason_codes.json"

# Cache for reason codes
_reason_codes_cache: Optional[Dict[str, Any]] = None


@dataclass
class ReasonCode:
    """Structured reason code for interventions."""

    code: str
    severity: str
    requires_retraining: bool
    requires_justification: bool
    description: str
    category: str
    retraining_priority: str

    @classmethod
    def from_dict(cls, code: str, data: Dict[str, Any]) -> "ReasonCode":
        """Create ReasonCode from config dict."""
        return cls(
            code=code,
            severity=data.get("severity", "MEDIUM"),
            requires_retraining=data.get("requires_retraining", False),
            requires_justification=data.get("requires_justification", False),
            description=data.get("description", ""),
            category=data.get("category", "other"),
            retraining_priority=data.get("retraining_priority", "LOW"),
        )


@dataclass
class Intervention:
    """Human intervention record."""

    intervention_id: str
    target_decision_id: str
    intervener_id: str
    intervener_role: str
    intervention_type: str  # OVERRIDE | CORRECTION | ANNOTATION | ABORT
    reason_code: str
    justification: Optional[str]
    original_action: Dict[str, Any]
    corrected_action: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "intervention_id": self.intervention_id,
            "target_decision_id": self.target_decision_id,
            "intervener_id": self.intervener_id,
            "intervener_role": self.intervener_role,
            "intervention_type": self.intervention_type,
            "reason_code": self.reason_code,
            "justification": self.justification,
            "original_action": self.original_action,
            "corrected_action": self.corrected_action,
            "timestamp": self.timestamp,
        }


def load_reason_codes() -> Dict[str, Any]:
    """Load reason codes from config file.

    Returns:
        Reason codes configuration

    Caches result in memory for performance.
    """
    global _reason_codes_cache

    if _reason_codes_cache is not None:
        return _reason_codes_cache

    if not REASON_CODES_FILE.exists():
        # Return default codes if file not found
        _reason_codes_cache = {
            "reason_codes": {
                "RE001_FACTUAL_ERROR": {
                    "severity": "HIGH",
                    "requires_retraining": True,
                    "requires_justification": True,
                    "description": "Agent output contained factually incorrect information",
                    "category": "accuracy",
                    "retraining_priority": "HIGH",
                },
                "RE005_USER_PREFERENCE": {
                    "severity": "LOW",
                    "requires_retraining": False,
                    "requires_justification": False,
                    "description": "User preferred different approach",
                    "category": "preference",
                    "retraining_priority": "LOW",
                },
            }
        }
        return _reason_codes_cache

    with open(REASON_CODES_FILE, "r") as f:
        _reason_codes_cache = json.load(f)

    return _reason_codes_cache


def validate_reason_code(code: str) -> bool:
    """Check if reason code exists in enum.

    Args:
        code: Reason code string (e.g., "RE001_FACTUAL_ERROR")

    Returns:
        True if code is valid
    """
    config = load_reason_codes()
    return code in config.get("reason_codes", {})


def get_reason_metadata(code: str) -> Optional[ReasonCode]:
    """Get metadata for a reason code.

    Args:
        code: Reason code string

    Returns:
        ReasonCode object or None if invalid
    """
    config = load_reason_codes()
    codes = config.get("reason_codes", {})

    if code not in codes:
        return None

    return ReasonCode.from_dict(code, codes[code])


def require_justification(code: str) -> bool:
    """Check if reason code requires free-text justification.

    Args:
        code: Reason code string

    Returns:
        True if justification required (CRITICAL codes always require)
    """
    metadata = get_reason_metadata(code)
    if metadata is None:
        return True  # Unknown codes require justification

    # CRITICAL severity always requires justification
    if metadata.severity == "CRITICAL":
        return True

    return metadata.requires_justification


def get_severity_weight(code: str) -> int:
    """Get severity weight for prioritization.

    Args:
        code: Reason code string

    Returns:
        Weight (4=CRITICAL, 3=HIGH, 2=MEDIUM, 1=LOW)
    """
    metadata = get_reason_metadata(code)
    if metadata is None:
        return 2  # Default to MEDIUM

    weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return weights.get(metadata.severity, 2)


def emit_intervention_receipt(
    intervention: Intervention,
    raci_accountable: Optional[str] = None,
) -> Dict[str, Any]:
    """Emit intervention receipt with dual-hash.

    Args:
        intervention: Intervention record
        raci_accountable: Accountable party from RACI system

    Returns:
        Receipt dict with dual-hash
    """
    metadata = get_reason_metadata(intervention.reason_code)

    receipt_data = {
        "tenant_id": GOVERNANCE_TENANT,
        **intervention.to_dict(),
        "reason_severity": metadata.severity if metadata else "UNKNOWN",
        "requires_retraining": metadata.requires_retraining if metadata else False,
        "retraining_priority": metadata.retraining_priority if metadata else "LOW",
        "raci_accountable": raci_accountable,
    }

    return emit_receipt("human_intervention", receipt_data)


def create_intervention(
    target_decision_id: str,
    intervener_id: str,
    intervener_role: str,
    intervention_type: str,
    reason_code: str,
    original_action: Dict[str, Any],
    corrected_action: Dict[str, Any],
    justification: Optional[str] = None,
    intervention_id: Optional[str] = None,
) -> Intervention:
    """Create and validate an intervention.

    Args:
        target_decision_id: ID of decision being corrected
        intervener_id: ID of human intervener
        intervener_role: Role of intervener
        intervention_type: Type (OVERRIDE|CORRECTION|ANNOTATION|ABORT)
        reason_code: Reason code (must be valid)
        original_action: Original agent action
        corrected_action: Corrected action
        justification: Free-text justification (required for some codes)
        intervention_id: Optional ID (generated if not provided)

    Returns:
        Validated Intervention object

    Raises:
        ValueError: If reason code invalid or justification missing when required
    """
    if not validate_reason_code(reason_code):
        raise ValueError(f"Invalid reason code: {reason_code}")

    if require_justification(reason_code) and not justification:
        raise ValueError(f"Reason code {reason_code} requires justification")

    valid_types = ["OVERRIDE", "CORRECTION", "ANNOTATION", "ABORT"]
    if intervention_type not in valid_types:
        raise ValueError(f"Invalid intervention type: {intervention_type}")

    if not intervention_id:
        intervention_id = str(uuid.uuid4())

    return Intervention(
        intervention_id=intervention_id,
        target_decision_id=target_decision_id,
        intervener_id=intervener_id,
        intervener_role=intervener_role,
        intervention_type=intervention_type,
        reason_code=reason_code,
        justification=justification,
        original_action=original_action,
        corrected_action=corrected_action,
    )


def clear_reason_codes_cache() -> None:
    """Clear the reason codes cache (for testing)."""
    global _reason_codes_cache
    _reason_codes_cache = None
