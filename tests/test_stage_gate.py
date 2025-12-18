"""test_stage_gate.py - Tests for dynamic allocation with alpha trigger

Validates Grok's hedge: "30% now, +10% if alpha > 1.9 in 12 months"
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage_gate import (
    evaluate_gate,
    apply_escalation,
    get_allocation,
    reset_window,
    check_gate_slos,
    simulate_gate_progression,
    get_gate_recommendation,
    StageGateState,
    STAGE_GATE_INITIAL,
    STAGE_GATE_TRIGGER_ALPHA,
    STAGE_GATE_ESCALATION,
    STAGE_GATE_WINDOW_MONTHS,
    STAGE_GATE_MAX_AUTONOMY,
    STAGE_GATE_MIN_PROPULSION,
)


class TestEvaluateGate:
    """Tests for evaluate_gate function."""

    def test_trigger_when_alpha_high(self):
        """Alpha > 1.9 with high confidence should trigger."""
        state = evaluate_gate(
            alpha_measured=2.0, alpha_confidence=0.85, months_elapsed=6
        )
        assert state.trigger_met is True
        assert state.escalation_applied is True
        assert state.current_autonomy_fraction == STAGE_GATE_MAX_AUTONOMY

    def test_no_trigger_below_alpha(self):
        """Alpha < 1.9 should not trigger."""
        state = evaluate_gate(
            alpha_measured=1.7, alpha_confidence=0.85, months_elapsed=6
        )
        assert state.trigger_met is False
        assert state.escalation_applied is False
        assert state.current_autonomy_fraction == STAGE_GATE_INITIAL

    def test_no_trigger_low_confidence(self):
        """Low confidence should not trigger even with high alpha."""
        state = evaluate_gate(
            alpha_measured=2.5,
            alpha_confidence=0.50,  # Below threshold
            months_elapsed=6,
        )
        assert state.trigger_met is False
        assert state.escalation_applied is False

    def test_no_trigger_after_window(self):
        """After window expires, should not trigger."""
        state = evaluate_gate(
            alpha_measured=2.5,
            alpha_confidence=0.85,
            months_elapsed=15,  # Past 12 month window
        )
        assert state.trigger_met is False


class TestApplyEscalation:
    """Tests for apply_escalation function."""

    def test_basic_escalation(self):
        """30% + 10% = 40%."""
        result = apply_escalation(0.30, 0.10, 0.40)
        assert result == 0.40

    def test_respects_ceiling(self):
        """Should not exceed max fraction."""
        result = apply_escalation(0.35, 0.10, 0.40)
        assert result == 0.40

    def test_multiple_escalations_capped(self):
        """Multiple escalations should still cap at max."""
        result = apply_escalation(0.40, 0.10, 0.40)
        assert result == 0.40


class TestGetAllocation:
    """Tests for get_allocation function."""

    def test_allocation_sums_to_one(self):
        """Propulsion + autonomy should sum to 1.0."""
        state = StageGateState(
            current_autonomy_fraction=0.35,
            alpha_measured=1.8,
            alpha_confidence=0.80,
            trigger_met=False,
            months_elapsed=3,
            escalation_applied=False,
        )
        prop, auto = get_allocation(state)
        assert abs(prop + auto - 1.0) < 0.001

    def test_propulsion_floor_enforced(self):
        """Propulsion should not go below 60%."""
        state = StageGateState(
            current_autonomy_fraction=0.50,  # Would make prop=0.50
            alpha_measured=1.8,
            alpha_confidence=0.80,
            trigger_met=False,
            months_elapsed=3,
            escalation_applied=False,
        )
        prop, auto = get_allocation(state)
        assert prop >= STAGE_GATE_MIN_PROPULSION


class TestResetWindow:
    """Tests for reset_window function."""

    def test_resets_months(self):
        """Should reset months_elapsed to 0."""
        state = StageGateState(
            current_autonomy_fraction=0.40,
            alpha_measured=2.0,
            alpha_confidence=0.85,
            trigger_met=True,
            months_elapsed=8,
            escalation_applied=True,
        )
        new_state = reset_window(state)
        assert new_state.months_elapsed == 0
        assert new_state.trigger_met is False  # Reset for new window

    def test_preserves_allocation(self):
        """Should keep current allocation after reset."""
        state = StageGateState(
            current_autonomy_fraction=0.40,
            alpha_measured=2.0,
            alpha_confidence=0.85,
            trigger_met=True,
            months_elapsed=8,
            escalation_applied=True,
        )
        new_state = reset_window(state)
        assert new_state.current_autonomy_fraction == 0.40


class TestCheckGateSLOs:
    """Tests for check_gate_slos function."""

    def test_all_slos_pass(self):
        """Valid state should pass all SLOs."""
        state = StageGateState(
            current_autonomy_fraction=0.35,
            alpha_measured=1.8,
            alpha_confidence=0.80,
            trigger_met=False,
            months_elapsed=6,
            escalation_applied=False,
        )
        slos = check_gate_slos(state)
        assert slos["all_passed"] is True

    def test_autonomy_ceiling_violation(self):
        """Autonomy > max should fail SLO."""
        state = StageGateState(
            current_autonomy_fraction=0.50,  # Exceeds 0.40
            alpha_measured=1.8,
            alpha_confidence=0.80,
            trigger_met=False,
            months_elapsed=6,
            escalation_applied=False,
        )
        slos = check_gate_slos(state)
        assert slos["autonomy_ceiling"] is False


class TestSimulateGateProgression:
    """Tests for simulate_gate_progression function."""

    def test_escalation_persists(self):
        """Once escalated, should stay escalated."""
        # Alpha trajectory: starts low, exceeds trigger at month 4
        alpha_trajectory = [1.5, 1.6, 1.7, 1.8, 2.0, 2.1, 2.2, 2.3]
        states = simulate_gate_progression(alpha_trajectory)

        # Find first escalation
        first_escalation = None
        for i, s in enumerate(states):
            if s.escalation_applied:
                first_escalation = i
                break

        assert first_escalation is not None
        # All subsequent states should also be escalated
        for s in states[first_escalation:]:
            assert s.escalation_applied is True

    def test_no_escalation_below_trigger(self):
        """Alpha never exceeding trigger should not escalate."""
        alpha_trajectory = [1.5, 1.6, 1.7, 1.8, 1.85, 1.89, 1.88, 1.87]
        states = simulate_gate_progression(alpha_trajectory)

        assert all(not s.escalation_applied for s in states)


class TestGetGateRecommendation:
    """Tests for get_gate_recommendation function."""

    def test_escalate_recommendation(self):
        """High alpha + high confidence should recommend escalation."""
        rec = get_gate_recommendation(2.0, 0.85, 6)
        assert "ESCALATE" in rec

    def test_hold_low_confidence(self):
        """Low confidence should recommend hold."""
        rec = get_gate_recommendation(2.0, 0.50, 6)
        assert "HOLD" in rec
        assert "Confidence" in rec

    def test_monitor_recommendation(self):
        """Below trigger should recommend monitor."""
        rec = get_gate_recommendation(1.7, 0.85, 6)
        assert "MONITOR" in rec


class TestConstants:
    """Tests for stage gate constants."""

    def test_initial_allocation(self):
        """Initial allocation should be 30%."""
        assert STAGE_GATE_INITIAL == 0.30

    def test_trigger_alpha(self):
        """Trigger alpha should be 1.9."""
        assert STAGE_GATE_TRIGGER_ALPHA == 1.9

    def test_escalation_increment(self):
        """Escalation increment should be 10%."""
        assert STAGE_GATE_ESCALATION == 0.10

    def test_window_months(self):
        """Window should be 12 months."""
        assert STAGE_GATE_WINDOW_MONTHS == 12

    def test_max_autonomy(self):
        """Max autonomy should be 40%."""
        assert STAGE_GATE_MAX_AUTONOMY == 0.40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
