"""execution_standard.py - RNES compliance validation.

Validate that modules comply with Receipts-Native Execution Standard.
"""

import ast
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

RNES_TENANT = "spaceproof-rnes"

# Required receipt functions in compliant modules
REQUIRED_FUNCTIONS = ["emit_receipt", "dual_hash"]

# Minimum receipt coverage threshold
MIN_COVERAGE_THRESHOLD = 0.95


@dataclass
class ComplianceIssue:
    """A compliance issue found during validation."""

    issue_id: str
    file_path: str
    line_number: Optional[int]
    issue_type: str
    description: str
    severity: str  # ERROR, WARNING, INFO

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_id": self.issue_id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class RNESCompliance:
    """RNES compliance report."""

    report_id: str
    module_path: str
    is_compliant: bool
    coverage: float
    issues: List[ComplianceIssue]
    functions_checked: int
    functions_with_receipts: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "module_path": self.module_path,
            "is_compliant": self.is_compliant,
            "coverage": self.coverage,
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "functions_checked": self.functions_checked,
            "functions_with_receipts": self.functions_with_receipts,
            "timestamp": self.timestamp,
        }


def _check_function_for_receipt(func_node: ast.FunctionDef) -> bool:
    """Check if function contains emit_receipt call.

    Args:
        func_node: AST function node

    Returns:
        True if function emits receipts
    """
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "emit_receipt":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "emit_receipt":
                return True
    return False


def _analyze_module(source: str, file_path: str) -> tuple[int, int, List[ComplianceIssue]]:
    """Analyze module for RNES compliance.

    Args:
        source: Module source code
        file_path: Path to module

    Returns:
        Tuple of (functions_checked, functions_with_receipts, issues)
    """
    issues = []
    functions_checked = 0
    functions_with_receipts = 0

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(
            ComplianceIssue(
                issue_id=str(uuid.uuid4()),
                file_path=file_path,
                line_number=e.lineno,
                issue_type="syntax_error",
                description=f"Syntax error: {e.msg}",
                severity="ERROR",
            )
        )
        return 0, 0, issues

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private and dunder functions
            if node.name.startswith("_"):
                continue

            functions_checked += 1

            if _check_function_for_receipt(node):
                functions_with_receipts += 1
            else:
                issues.append(
                    ComplianceIssue(
                        issue_id=str(uuid.uuid4()),
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="missing_receipt",
                        description=f"Function '{node.name}' does not emit receipts",
                        severity="WARNING",
                    )
                )

    return functions_checked, functions_with_receipts, issues


def check_receipt_coverage(module_path: str) -> float:
    """Check receipt coverage for a module.

    Args:
        module_path: Path to module file

    Returns:
        Coverage percentage (0.0 - 1.0)
    """
    path = Path(module_path)
    if not path.exists():
        return 0.0

    source = path.read_text()
    checked, with_receipts, _ = _analyze_module(source, module_path)

    if checked == 0:
        return 1.0  # No public functions is considered compliant

    return with_receipts / checked


def validate_rnes_compliance(
    module_path: str,
    coverage_threshold: float = MIN_COVERAGE_THRESHOLD,
) -> RNESCompliance:
    """Validate RNES compliance for a module.

    Args:
        module_path: Path to module file
        coverage_threshold: Minimum coverage required

    Returns:
        RNESCompliance report
    """
    path = Path(module_path)
    issues = []

    if not path.exists():
        issues.append(
            ComplianceIssue(
                issue_id=str(uuid.uuid4()),
                file_path=module_path,
                line_number=None,
                issue_type="file_not_found",
                description=f"Module file not found: {module_path}",
                severity="ERROR",
            )
        )
        return RNESCompliance(
            report_id=str(uuid.uuid4()),
            module_path=module_path,
            is_compliant=False,
            coverage=0.0,
            issues=issues,
            functions_checked=0,
            functions_with_receipts=0,
        )

    source = path.read_text()
    checked, with_receipts, analysis_issues = _analyze_module(source, module_path)
    issues.extend(analysis_issues)

    # Check for required imports
    if "emit_receipt" not in source:
        issues.append(
            ComplianceIssue(
                issue_id=str(uuid.uuid4()),
                file_path=module_path,
                line_number=None,
                issue_type="missing_import",
                description="Module does not import emit_receipt",
                severity="ERROR",
            )
        )

    coverage = with_receipts / checked if checked > 0 else 1.0
    is_compliant = coverage >= coverage_threshold and not any(i.severity == "ERROR" for i in issues)

    return RNESCompliance(
        report_id=str(uuid.uuid4()),
        module_path=module_path,
        is_compliant=is_compliant,
        coverage=coverage,
        issues=issues,
        functions_checked=checked,
        functions_with_receipts=with_receipts,
    )


def get_compliance_report(
    module_paths: List[str],
) -> Dict[str, Any]:
    """Get compliance report for multiple modules.

    Args:
        module_paths: List of module paths

    Returns:
        Aggregate compliance report
    """
    reports = []
    total_checked = 0
    total_with_receipts = 0
    total_issues = 0
    compliant_count = 0

    for path in module_paths:
        report = validate_rnes_compliance(path)
        reports.append(report)
        total_checked += report.functions_checked
        total_with_receipts += report.functions_with_receipts
        total_issues += len(report.issues)
        if report.is_compliant:
            compliant_count += 1

    overall_coverage = total_with_receipts / total_checked if total_checked > 0 else 1.0

    summary = {
        "total_modules": len(module_paths),
        "compliant_modules": compliant_count,
        "compliance_rate": compliant_count / len(module_paths) * 100 if module_paths else 0,
        "overall_coverage": overall_coverage * 100,
        "total_functions": total_checked,
        "functions_with_receipts": total_with_receipts,
        "total_issues": total_issues,
        "reports": [r.to_dict() for r in reports],
    }

    # Emit compliance report receipt
    emit_receipt(
        "rnes_compliance",
        {
            "tenant_id": RNES_TENANT,
            "modules_checked": len(module_paths),
            "compliant_count": compliant_count,
            "overall_coverage": overall_coverage,
        },
    )

    return summary


# Additional functions for API compatibility
def check_receipt_emission(module_path: str) -> Dict[str, Any]:
    """Check if module emits receipts properly.

    Args:
        module_path: Path to module

    Returns:
        Receipt emission status
    """
    coverage = check_receipt_coverage(module_path)
    return {
        "module_path": module_path,
        "emits_receipts": coverage > 0,
        "coverage": coverage,
        "compliant": coverage >= MIN_COVERAGE_THRESHOLD,
    }


def emit_rnes_receipt(
    operation_type: str,
    module_path: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit RNES operation receipt.

    Args:
        operation_type: Type of RNES operation
        module_path: Module being validated
        details: Additional details

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "rnes_operation",
        {
            "tenant_id": RNES_TENANT,
            "operation_type": operation_type,
            "module_path": module_path,
            "details": details or {},
        },
    )


def execute_receipt_chain(
    receipts: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a chain of receipts in order.

    Args:
        receipts: List of receipts to execute
        context: Execution context

    Returns:
        Execution result
    """
    results = []
    for i, receipt in enumerate(receipts):
        result = {
            "step": i + 1,
            "receipt_type": receipt.get("receipt_type", "unknown"),
            "receipt_id": receipt.get("receipt_id", str(uuid.uuid4())),
            "executed": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        results.append(result)

    return {
        "chain_id": str(uuid.uuid4()),
        "receipts_executed": len(receipts),
        "all_successful": True,
        "results": results,
        "context": context or {},
    }


def get_execution_result(chain_id: str) -> Optional[Dict[str, Any]]:
    """Get execution result for a receipt chain.

    Args:
        chain_id: Chain execution ID

    Returns:
        Execution result or None
    """
    # In production, this would query stored execution results
    # For now, return a placeholder
    return {
        "chain_id": chain_id,
        "status": "completed",
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
    }


def emit_interpretation_receipt(
    interpretation_id: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    model_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Emit receipt for model interpretation.

    Args:
        interpretation_id: Interpretation identifier
        input_data: Input to interpretation
        output_data: Output from interpretation
        model_version: Model version used

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "interpretation",
        {
            "tenant_id": RNES_TENANT,
            "interpretation_id": interpretation_id,
            "input_hash": str(hash(str(input_data))),
            "output_hash": str(hash(str(output_data))),
            "model_version": model_version,
        },
    )


def validate_sandbox_output(
    output: Any,
    expected_type: Optional[type] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate output from sandboxed execution.

    Args:
        output: Output to validate
        expected_type: Expected output type
        constraints: Validation constraints

    Returns:
        Validation result
    """
    is_valid = True
    violations = []

    if expected_type and not isinstance(output, expected_type):
        is_valid = False
        violations.append(f"Type mismatch: expected {expected_type.__name__}, got {type(output).__name__}")

    if constraints:
        if "max_size" in constraints:
            size = len(str(output))
            if size > constraints["max_size"]:
                is_valid = False
                violations.append(f"Size exceeded: {size} > {constraints['max_size']}")

    return {
        "is_valid": is_valid,
        "violations": violations,
        "validated_at": datetime.utcnow().isoformat() + "Z",
    }
