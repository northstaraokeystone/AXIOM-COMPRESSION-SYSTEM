"""Tests for spaceproof.loop module."""

from spaceproof.loop import (
    Loop,
    Action,
    run_loop_once,
)


def test_loop_init():
    """Loop initializes with defaults."""
    loop = Loop()
    assert loop.cycle_count == 0
    assert loop.sources == []


def test_loop_run_cycle():
    """Loop run_cycle completes."""
    loop = Loop()
    result = loop.run_cycle()

    assert result.cycle_id != ""
    assert result.cycle_time_sec >= 0
    assert result.completed is True


def test_loop_sense_empty():
    """Loop sense returns empty without sources."""
    loop = Loop()
    receipts = loop.sense()
    assert receipts == []


def test_loop_sense_with_source():
    """Loop sense collects from sources."""
    def source():
        return [{"type": "test", "value": 1}]

    loop = Loop(sources=[source])
    receipts = loop.sense()
    assert len(receipts) == 1


def test_loop_analyze():
    """Loop analyze processes receipts."""
    loop = Loop()
    receipts = [
        {"receipt_type": "test", "value": 1},
        {"receipt_type": "anomaly", "metric": "x", "classification": "drift"},
    ]
    analysis = loop.analyze(receipts)

    assert analysis["receipt_count"] == 2
    assert analysis["anomaly_count"] == 1


def test_loop_hypothesize():
    """Loop hypothesize generates proposals."""
    loop = Loop()
    analysis = {"anomalies": [{"classification": "drift", "metric": "test"}]}
    proposals = loop.hypothesize(analysis)

    assert len(proposals) > 0
    assert all(isinstance(p, Action) for p in proposals)


def test_loop_gate_low_risk():
    """Loop gate auto-approves low-risk actions."""
    loop = Loop()
    actions = [
        Action(id="1", action_type="monitor", description="Test", risk=0.3),
    ]
    approved = loop.gate(actions)
    assert len(approved) == 1
    assert approved[0].approved is True


def test_loop_gate_high_risk_no_hitl():
    """Loop gate skips high-risk without HITL."""
    loop = Loop()
    actions = [
        Action(id="1", action_type="investigate", description="Test", risk=0.8),
    ]
    approved = loop.gate(actions)
    assert len(approved) == 0


def test_run_loop_once():
    """run_loop_once convenience function works."""
    result = run_loop_once()
    assert result.cycle_id != ""
