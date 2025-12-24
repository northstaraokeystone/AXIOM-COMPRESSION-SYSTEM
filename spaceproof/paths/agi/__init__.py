"""Post-AGI ethics modeling via fractal policies.

Evolution path: stub -> policy -> evaluate -> autonomous_audit

KEY INSIGHT: "Audit trail IS alignment"
- If system cannot prove what it did, it did not do it
- Compression as alignment metric
- Fractal policies are self-similar at all scales

Source: SpaceProof scalable paths architecture - AGI ethics modeling
"""

from .core import (
    stub_status,
    fractal_policy,
    evaluate_ethics,
    compute_alignment,
    audit_decision,
    get_agi_info,
    AGI_TENANT_ID,
    POLICY_DEPTH_DEFAULT,
    ETHICS_DIMENSIONS,
    ALIGNMENT_METRIC,
    AUDIT_REQUIREMENT,
)

from .receipts import (
    emit_agi_status,
    emit_agi_policy,
    emit_agi_ethics,
    emit_agi_alignment,
    emit_agi_audit,
)

from . import cli

__all__ = [
    # Core functions
    "stub_status",
    "fractal_policy",
    "evaluate_ethics",
    "compute_alignment",
    "audit_decision",
    "get_agi_info",
    # Constants
    "AGI_TENANT_ID",
    "POLICY_DEPTH_DEFAULT",
    "ETHICS_DIMENSIONS",
    "ALIGNMENT_METRIC",
    "AUDIT_REQUIREMENT",
    # Receipt helpers
    "emit_agi_status",
    "emit_agi_policy",
    "emit_agi_ethics",
    "emit_agi_alignment",
    "emit_agi_audit",
    # CLI module
    "cli",
]
