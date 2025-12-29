"""Governance module - RACI, provenance, reason codes.

Enterprise governance for AI/autonomous systems.
Provides structured accountability through RACI assignments,
model/policy provenance tracking, and standardized intervention codes.
"""

from .raci import (
    load_raci_matrix,
    get_raci_for_event,
    validate_raci,
    emit_raci_receipt,
    RACIAssignment,
)

from .provenance import (
    capture_provenance,
    get_model_version,
    get_policy_state,
    emit_provenance_receipt,
    ProvenanceCapture,
)

from .reason_codes import (
    load_reason_codes,
    validate_reason_code,
    get_reason_metadata,
    require_justification,
    emit_intervention_receipt,
    ReasonCode,
)

from .accountability import (
    assign_ownership,
    track_decision_chain,
    emit_accountability_receipt,
    OwnershipChain,
)

from .escalation import (
    get_escalation_path,
    should_escalate,
    evaluate_escalation,
    emit_escalation_receipt,
    EscalationResult,
)

__all__ = [
    # RACI
    "load_raci_matrix",
    "get_raci_for_event",
    "validate_raci",
    "emit_raci_receipt",
    "RACIAssignment",
    # Provenance
    "capture_provenance",
    "get_model_version",
    "get_policy_state",
    "emit_provenance_receipt",
    "ProvenanceCapture",
    # Reason codes
    "load_reason_codes",
    "validate_reason_code",
    "get_reason_metadata",
    "require_justification",
    "emit_intervention_receipt",
    "ReasonCode",
    # Accountability
    "assign_ownership",
    "track_decision_chain",
    "emit_accountability_receipt",
    "OwnershipChain",
    # Escalation
    "get_escalation_path",
    "should_escalate",
    "evaluate_escalation",
    "emit_escalation_receipt",
    "EscalationResult",
]
