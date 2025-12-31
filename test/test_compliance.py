"""Tests for spaceproof.compliance module."""

from spaceproof.compliance import (
    generate_audit_trail,
    query_audit_trail,
    AuditTrailReport,
    generate_raci_report,
    get_accountability_summary,
    RACIReport,
    generate_intervention_report,
    get_intervention_metrics,
    InterventionReport,
    generate_provenance_report,
    get_model_history,
    get_policy_history,
    ProvenanceReport,
)


def test_generate_audit_trail():
    """generate_audit_trail creates trail within SLO."""
    import time
    start = time.time()
    trail = generate_audit_trail(
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-02T00:00:00Z",
    )
    elapsed_ms = (time.time() - start) * 1000

    assert isinstance(trail, AuditTrailReport)
    assert hasattr(trail, "entries")
    assert elapsed_ms < 5000  # 5 second SLO


def test_query_audit_trail():
    """query_audit_trail retrieves entries."""
    entries = query_audit_trail()
    assert isinstance(entries, list)


def test_generate_raci_report():
    """generate_raci_report creates accountability report."""
    report = generate_raci_report()
    assert isinstance(report, RACIReport)
    assert hasattr(report, "raci_coverage")


def test_get_accountability_summary():
    """get_accountability_summary returns coverage stats."""
    summary = get_accountability_summary()
    assert isinstance(summary, dict)
    assert "total_decisions" in summary


def test_generate_intervention_report():
    """generate_intervention_report creates intervention metrics."""
    report = generate_intervention_report()
    assert isinstance(report, InterventionReport)
    assert hasattr(report, "total_interventions")


def test_get_intervention_metrics():
    """get_intervention_metrics returns intervention stats."""
    metrics = get_intervention_metrics()
    assert isinstance(metrics, dict)


def test_generate_provenance_report():
    """generate_provenance_report creates version history."""
    report = generate_provenance_report()
    assert isinstance(report, ProvenanceReport)
    assert hasattr(report, "report_id")


def test_get_model_history():
    """get_model_history returns model version history."""
    history = get_model_history()
    assert isinstance(history, list)


def test_get_policy_history():
    """get_policy_history returns policy version history."""
    history = get_policy_history()
    assert isinstance(history, list)
