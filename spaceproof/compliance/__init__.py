"""Compliance module - Audit trails and reports.

Generate compliance reports and audit trails.
Provides accountability documentation for enterprise governance.
"""

from .audit_trail import (
    generate_audit_trail,
    query_audit_trail,
    export_audit_trail,
    get_audit_trail,
    emit_audit_receipt,
    AuditTrailEntry,
    AuditTrailReport,
)

from .raci_report import (
    generate_raci_report,
    get_accountability_summary,
    get_raci_coverage,
    emit_raci_report_receipt,
    RACIReport,
)

from .intervention_report import (
    generate_intervention_report,
    get_intervention_metrics,
    emit_intervention_report_receipt,
    InterventionReport,
)

from .provenance_report import (
    generate_provenance_report,
    get_model_history,
    get_policy_history,
    get_provenance_history,
    emit_provenance_report_receipt,
    ProvenanceReport,
)

__all__ = [
    # Audit trail
    "generate_audit_trail",
    "query_audit_trail",
    "export_audit_trail",
    "get_audit_trail",
    "emit_audit_receipt",
    "AuditTrailEntry",
    "AuditTrailReport",
    # RACI report
    "generate_raci_report",
    "get_accountability_summary",
    "get_raci_coverage",
    "emit_raci_report_receipt",
    "RACIReport",
    # Intervention report
    "generate_intervention_report",
    "get_intervention_metrics",
    "emit_intervention_report_receipt",
    "InterventionReport",
    # Provenance report
    "generate_provenance_report",
    "get_model_history",
    "get_policy_history",
    "get_provenance_history",
    "emit_provenance_report_receipt",
    "ProvenanceReport",
]
