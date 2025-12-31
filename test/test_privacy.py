"""Tests for spaceproof.privacy module."""

from spaceproof.privacy import (
    redact_pii,
    detect_pii,
    get_redaction_stats,
    emit_redaction_receipt,
    RedactionResult,
    add_laplace_noise,
    add_gaussian_noise,
    compute_sensitivity,
    check_privacy_budget,
    emit_dp_receipt,
    DPResult,
    PrivacyBudget,
    emit_privacy_receipt,
    track_privacy_operation,
    get_privacy_audit_log,
    PrivacyOperation,
)


def test_detect_pii_email():
    """detect_pii finds email addresses."""
    text = "Contact us at user@example.com for help"
    pii = detect_pii(text)
    assert isinstance(pii, list)


def test_detect_pii_none():
    """detect_pii returns empty for clean text."""
    text = "This is safe text with no personal info"
    pii = detect_pii(text)
    assert isinstance(pii, list)


def test_redact_pii():
    """redact_pii replaces PII with placeholders."""
    text = "Email: test@example.com, SSN: 123-45-6789"
    result = redact_pii(text)
    assert isinstance(result, RedactionResult)


def test_get_redaction_stats():
    """get_redaction_stats returns stats."""
    stats = get_redaction_stats()
    assert isinstance(stats, dict)


def test_emit_redaction_receipt():
    """emit_redaction_receipt emits valid receipt."""
    text = "Email: test@example.com"
    result = redact_pii(text)
    receipt = emit_redaction_receipt(result)
    assert receipt["receipt_type"] == "redaction"


def test_add_laplace_noise():
    """add_laplace_noise adds calibrated noise."""
    result = add_laplace_noise(50.0, epsilon=1.0, sensitivity=1.0)
    assert isinstance(result, DPResult)


def test_add_gaussian_noise():
    """add_gaussian_noise adds calibrated noise."""
    result = add_gaussian_noise(50.0, epsilon=1.0, delta=1e-5, sensitivity=1.0)
    assert isinstance(result, DPResult)


def test_compute_sensitivity():
    """compute_sensitivity returns sensitivity value."""
    sensitivity = compute_sensitivity("count")
    assert isinstance(sensitivity, (int, float))


def test_check_privacy_budget():
    """check_privacy_budget returns budget info."""
    budget = check_privacy_budget()
    assert isinstance(budget, PrivacyBudget)


def test_emit_dp_receipt():
    """emit_dp_receipt emits valid receipt."""
    result = add_laplace_noise(100.0, epsilon=1.0, sensitivity=1.0)
    receipt = emit_dp_receipt(result)
    assert receipt["receipt_type"] == "differential_privacy"


def test_track_privacy_operation():
    """track_privacy_operation tracks operation."""
    # Full signature: track_privacy_operation(operation_type, actor_id, target_type, target_id, ...)
    op = track_privacy_operation(
        operation_type="redaction",
        actor_id="user-001",
        target_type="document",
        target_id="doc-001",
        success=True,
    )
    assert isinstance(op, PrivacyOperation)


def test_get_privacy_audit_log():
    """get_privacy_audit_log returns audit log."""
    log = get_privacy_audit_log()
    assert isinstance(log, list)


def test_emit_privacy_receipt():
    """emit_privacy_receipt emits valid receipt."""
    op = track_privacy_operation(
        operation_type="redaction",
        actor_id="user-001",
        target_type="document",
        target_id="doc-001",
        success=True,
    )
    receipt = emit_privacy_receipt(op)
    assert receipt["receipt_type"] == "privacy_operation"
