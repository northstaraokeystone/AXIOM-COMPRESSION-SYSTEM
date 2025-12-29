"""Privacy module - PII redaction and differential privacy.

Privacy-preserving receipt handling with cryptographic proofs.
"""

from .redaction import (
    redact_pii,
    detect_pii,
    get_redaction_stats,
    get_pii_patterns,
    emit_redaction_receipt,
    RedactionResult,
    PIIMatch,
)

from .differential_privacy import (
    add_laplace_noise,
    add_gaussian_noise,
    compute_sensitivity,
    check_privacy_budget,
    get_privacy_budget,
    spend_privacy_budget,
    emit_dp_receipt,
    DPResult,
    PrivacyBudget,
)

from .privacy_receipt import (
    emit_privacy_receipt,
    track_privacy_operation,
    get_privacy_audit_log,
    emit_privacy_operation_receipt,
    get_privacy_operations,
    validate_privacy_compliance,
    PrivacyOperation,
)

__all__ = [
    # Redaction
    "redact_pii",
    "detect_pii",
    "get_redaction_stats",
    "get_pii_patterns",
    "emit_redaction_receipt",
    "RedactionResult",
    "PIIMatch",
    # Differential privacy
    "add_laplace_noise",
    "add_gaussian_noise",
    "compute_sensitivity",
    "check_privacy_budget",
    "get_privacy_budget",
    "spend_privacy_budget",
    "emit_dp_receipt",
    "DPResult",
    "PrivacyBudget",
    # Privacy receipts
    "emit_privacy_receipt",
    "track_privacy_operation",
    "get_privacy_audit_log",
    "emit_privacy_operation_receipt",
    "get_privacy_operations",
    "validate_privacy_compliance",
    "PrivacyOperation",
]
