"""Tests for spaceproof.rnes module."""

from spaceproof.rnes import (
    validate_rnes_compliance,
    check_receipt_coverage,
    get_compliance_report,
    RNESCompliance,
    interpret_receipt,
    execute_from_receipt,
    build_execution_plan,
    emit_execution_receipt,
    ExecutionResult,
    ExecutionPlan,
    create_sandbox,
    execute_in_sandbox,
    cleanup_sandbox,
    emit_sandbox_receipt,
    SandboxConfig,
    SandboxResult,
)


def test_validate_rnes_compliance_valid():
    """validate_rnes_compliance checks code."""
    code = '''
def process():
    emit_receipt("operation", {"data": "test"})
    return True
'''
    result = validate_rnes_compliance(code)
    assert isinstance(result, RNESCompliance)


def test_check_receipt_coverage():
    """check_receipt_coverage verifies coverage."""
    code = '''
def process():
    return True
'''
    coverage = check_receipt_coverage(code)
    assert coverage is not None


def test_get_compliance_report():
    """get_compliance_report returns report."""
    code = "def test(): pass"
    report = get_compliance_report(code)
    assert report is not None


def test_interpret_receipt():
    """interpret_receipt parses receipt for execution."""
    receipt = {
        "receipt_type": "command",
        "action": "navigate",
        "parameters": {"target": "waypoint_1"},
    }
    interpretation = interpret_receipt(receipt)
    assert interpretation is not None


def test_execute_from_receipt():
    """execute_from_receipt executes from receipt."""
    receipt = {
        "receipt_type": "command",
        "action": "noop",
    }
    result = execute_from_receipt(receipt)
    assert isinstance(result, ExecutionResult)


def test_build_execution_plan():
    """build_execution_plan creates plan."""
    receipts = [
        {"receipt_type": "init", "action": "setup"},
        {"receipt_type": "command", "action": "execute"},
    ]
    plan = build_execution_plan(receipts)
    assert isinstance(plan, ExecutionPlan)


def test_emit_execution_receipt():
    """emit_execution_receipt emits valid receipt."""
    receipt = {
        "receipt_type": "command",
        "action": "noop",
    }
    result = execute_from_receipt(receipt)
    exec_receipt = emit_execution_receipt(result)
    assert "receipt_type" in exec_receipt


def test_create_sandbox():
    """create_sandbox creates isolated environment."""
    sandbox = create_sandbox()
    assert isinstance(sandbox, SandboxConfig)


def test_execute_in_sandbox():
    """execute_in_sandbox runs function in isolation."""
    sandbox = create_sandbox(max_cpu_seconds=5.0)
    # Signature: execute_in_sandbox(sandbox_id, operation, func, args=(), kwargs=None)

    def simple_op():
        return 2 + 2

    result = execute_in_sandbox(sandbox.sandbox_id, "test_op", simple_op)
    assert isinstance(result, SandboxResult)


def test_cleanup_sandbox():
    """cleanup_sandbox cleans up sandbox."""
    sandbox = create_sandbox()
    # Signature: cleanup_sandbox(sandbox_id)
    cleanup_sandbox(sandbox.sandbox_id)
    # Should not raise


def test_emit_sandbox_receipt():
    """emit_sandbox_receipt emits valid receipt."""
    sandbox = create_sandbox(max_cpu_seconds=5.0)

    def simple_op():
        return 1

    result = execute_in_sandbox(sandbox.sandbox_id, "test_op", simple_op)
    receipt = emit_sandbox_receipt(result)
    assert receipt["receipt_type"] == "sandbox_execution"
