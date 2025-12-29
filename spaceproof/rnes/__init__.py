"""RNES module - Receipts-Native Execution Standard.

Standard for receipt-driven execution with cryptographic guarantees.
"""

from .execution_standard import (
    validate_rnes_compliance,
    check_receipt_coverage,
    check_receipt_emission,
    get_compliance_report,
    emit_rnes_receipt,
    execute_receipt_chain,
    get_execution_result,
    emit_interpretation_receipt,
    validate_sandbox_output,
    RNESCompliance,
    ComplianceIssue,
)

from .interpreter import (
    interpret_receipt,
    execute_from_receipt,
    build_execution_plan,
    emit_execution_receipt,
    ExecutionResult,
    ExecutionPlan,
)

from .sandbox import (
    create_sandbox,
    execute_in_sandbox,
    cleanup_sandbox,
    emit_sandbox_receipt,
    SandboxConfig,
    SandboxResult,
)

__all__ = [
    # Execution standard
    "validate_rnes_compliance",
    "check_receipt_coverage",
    "check_receipt_emission",
    "get_compliance_report",
    "emit_rnes_receipt",
    "execute_receipt_chain",
    "get_execution_result",
    "emit_interpretation_receipt",
    "validate_sandbox_output",
    "RNESCompliance",
    "ComplianceIssue",
    # Interpreter
    "interpret_receipt",
    "execute_from_receipt",
    "build_execution_plan",
    "emit_execution_receipt",
    "ExecutionResult",
    "ExecutionPlan",
    # Sandbox
    "create_sandbox",
    "execute_in_sandbox",
    "cleanup_sandbox",
    "emit_sandbox_receipt",
    "SandboxConfig",
    "SandboxResult",
]
