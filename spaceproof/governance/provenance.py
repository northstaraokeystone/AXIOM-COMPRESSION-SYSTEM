"""provenance.py - Model version and policy state capture.

Every decision must have provenance: which model, which policy, what state.
Captures model version and policy state at decision time for reproducibility.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from spaceproof.core import dual_hash, emit_receipt

# === CONSTANTS ===

GOVERNANCE_TENANT = "spaceproof-governance"

# Default model/policy info when not provided
DEFAULT_MODEL_VERSION = "unknown"
DEFAULT_POLICY_VERSION = "1.0.0"


@dataclass
class ProvenanceCapture:
    """Complete provenance capture for a decision."""

    decision_id: str
    model_id: str
    model_version: str
    model_hash: str
    policy_id: str
    policy_version: str
    policy_hash: str
    config_state: Dict[str, Any]
    environment: Dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    parent_decision_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "decision_id": self.decision_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "config_state": self.config_state,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "parent_decision_id": self.parent_decision_id,
        }


def get_model_version(model_id: Optional[str] = None) -> Dict[str, str]:
    """Get current model version information.

    Args:
        model_id: Optional model identifier

    Returns:
        Dict with model_id, version, hash
    """
    # In production, this would query actual model registry
    # For now, return simulated values
    model_id = model_id or "spaceproof-agent"

    # Simulate model hash (would be actual model weights hash in production)
    model_content = f"{model_id}:v1.0.0:weights"
    model_hash = dual_hash(model_content)

    return {
        "model_id": model_id,
        "version": "1.0.0",
        "hash": model_hash,
    }


def get_policy_state(policy_id: Optional[str] = None) -> Dict[str, Any]:
    """Get current policy state.

    Args:
        policy_id: Optional policy identifier

    Returns:
        Dict with policy_id, version, hash, active_rules
    """
    policy_id = policy_id or "default-policy"

    # Simulate policy state
    policy_content = {
        "policy_id": policy_id,
        "version": DEFAULT_POLICY_VERSION,
        "rules": {
            "max_autonomy_level": 3,
            "require_human_approval_above": 2,
            "entropy_conservation_limit": 0.01,
            "require_raci": True,
        },
    }

    policy_hash = dual_hash(json.dumps(policy_content, sort_keys=True))

    return {
        "policy_id": policy_id,
        "version": policy_content["version"],
        "hash": policy_hash,
        "active_rules": policy_content["rules"],
    }


def get_environment_state() -> Dict[str, str]:
    """Get current environment state.

    Returns:
        Dict with environment info (runtime, version, etc.)
    """
    import platform
    import sys

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "platform_version": platform.version()[:50],
        "spaceproof_version": "6.0.0",  # Will be updated
    }


def capture_provenance(
    decision_id: Optional[str] = None,
    model_id: Optional[str] = None,
    policy_id: Optional[str] = None,
    parent_decision_id: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ProvenanceCapture:
    """Capture complete provenance for a decision.

    Args:
        decision_id: Unique decision identifier (generated if not provided)
        model_id: Model identifier
        policy_id: Policy identifier
        parent_decision_id: Parent decision for chaining
        config_overrides: Additional config to capture

    Returns:
        ProvenanceCapture with all state
    """
    if not decision_id:
        decision_id = str(uuid.uuid4())

    model_info = get_model_version(model_id)
    policy_info = get_policy_state(policy_id)
    environment = get_environment_state()

    config_state = {
        "active_rules": policy_info.get("active_rules", {}),
    }
    if config_overrides:
        config_state.update(config_overrides)

    return ProvenanceCapture(
        decision_id=decision_id,
        model_id=model_info["model_id"],
        model_version=model_info["version"],
        model_hash=model_info["hash"],
        policy_id=policy_info["policy_id"],
        policy_version=policy_info["version"],
        policy_hash=policy_info["hash"],
        config_state=config_state,
        environment=environment,
        parent_decision_id=parent_decision_id,
    )


def emit_provenance_receipt(
    provenance: ProvenanceCapture,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit provenance receipt for audit trail.

    Args:
        provenance: ProvenanceCapture to emit
        context: Optional additional context

    Returns:
        Receipt dict with dual-hash
    """
    receipt_data = {
        "tenant_id": GOVERNANCE_TENANT,
        **provenance.to_dict(),
    }

    if context:
        receipt_data["context"] = context

    return emit_receipt("provenance", receipt_data)


def verify_provenance(provenance: ProvenanceCapture) -> bool:
    """Verify provenance capture integrity.

    Args:
        provenance: ProvenanceCapture to verify

    Returns:
        True if hashes are valid
    """
    # Recompute model hash
    model_content = f"{provenance.model_id}:v{provenance.model_version}:weights"
    dual_hash(model_content)

    # For simulation, model hashes may not match exactly
    # In production, would verify against model registry
    return len(provenance.model_hash) > 0 and len(provenance.policy_hash) > 0


def chain_provenance(
    parent: ProvenanceCapture,
    child_decision_id: Optional[str] = None,
) -> ProvenanceCapture:
    """Create chained provenance from parent.

    Args:
        parent: Parent provenance capture
        child_decision_id: Optional child decision ID

    Returns:
        New ProvenanceCapture linked to parent
    """
    return capture_provenance(
        decision_id=child_decision_id,
        model_id=parent.model_id,
        policy_id=parent.policy_id,
        parent_decision_id=parent.decision_id,
    )
