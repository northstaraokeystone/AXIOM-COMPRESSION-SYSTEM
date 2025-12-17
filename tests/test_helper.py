"""test_helper.py - Tests for self-improving helper layer

Validates ProofPack v3 §3 LOOP patterns:
- HARVEST finds recurring patterns
- HYPOTHESIZE creates blueprints
- GATE auto-approves low-risk
- ACTUATE deploys helpers
- Effectiveness tracking and retirement
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helper import (
    HelperConfig,
    HelperBlueprint,
    harvest,
    hypothesize,
    gate,
    actuate,
    measure_effectiveness,
    retire,
    check_retirement_candidates,
    get_active_helpers,
    create_gap_receipt,
)


class TestHarvest:
    """Tests for harvest function."""

    @staticmethod
    def _current_ts():
        """Get current timestamp for tests."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

    def test_harvest_finds_patterns(self, capsys):
        """5+ same-type gaps should trigger pattern detection."""
        config = HelperConfig(recurrence_threshold=5)
        ts = self._current_ts()

        # Create 6 gap receipts of same type with current timestamp
        gaps = [
            {"receipt_type": "gap", "type": "config_error", "ts": ts}
            for _ in range(6)
        ]

        patterns = harvest(gaps, config)

        assert len(patterns) >= 1
        assert any(p["problem_type"] == "config_error" for p in patterns)

        # Check harvest receipt emitted
        captured = capsys.readouterr()
        assert '"receipt_type": "harvest"' in captured.out

    def test_harvest_ignores_below_threshold(self):
        """Fewer than threshold gaps should not trigger pattern."""
        config = HelperConfig(recurrence_threshold=5)
        ts = self._current_ts()

        # Create only 3 gap receipts
        gaps = [
            {"receipt_type": "gap", "type": "config_error", "ts": ts}
            for _ in range(3)
        ]

        patterns = harvest(gaps, config)

        assert len(patterns) == 0

    def test_harvest_groups_by_type(self):
        """Should group gaps by problem type."""
        config = HelperConfig(recurrence_threshold=3)
        ts = self._current_ts()

        gaps = [
            {"receipt_type": "gap", "type": "config_error", "ts": ts},
            {"receipt_type": "gap", "type": "config_error", "ts": ts},
            {"receipt_type": "gap", "type": "config_error", "ts": ts},
            {"receipt_type": "gap", "type": "timeout", "ts": ts},
            {"receipt_type": "gap", "type": "timeout", "ts": ts},
            {"receipt_type": "gap", "type": "timeout", "ts": ts},
        ]

        patterns = harvest(gaps, config)

        assert len(patterns) == 2
        types_found = {p["problem_type"] for p in patterns}
        assert types_found == {"config_error", "timeout"}

    def test_harvest_filters_non_gap_receipts(self):
        """Should only process gap receipts."""
        config = HelperConfig(recurrence_threshold=3)
        ts = self._current_ts()

        receipts = [
            {"receipt_type": "gap", "type": "error", "ts": ts},
            {"receipt_type": "gap", "type": "error", "ts": ts},
            {"receipt_type": "gap", "type": "error", "ts": ts},
            {"receipt_type": "telemetry", "type": "error", "ts": ts},  # Not a gap
        ]

        patterns = harvest(receipts, config)

        # Should find pattern for gaps only
        assert len(patterns) == 1
        assert patterns[0]["count"] == 3


class TestHypothesize:
    """Tests for hypothesize function."""

    def test_hypothesize_creates_blueprint(self, capsys):
        """Pattern should generate blueprint with backtest stats."""
        patterns = [{
            "problem_type": "config_error",
            "count": 8,
            "gap_ids": ["gap1", "gap2", "gap3"],
            "sample_gap": {"type": "config_error"}
        }]

        blueprints = hypothesize(patterns)

        assert len(blueprints) == 1
        bp = blueprints[0]

        assert bp.id is not None
        assert len(bp.origin_gaps) > 0
        assert "trigger" in bp.pattern
        assert "action" in bp.pattern
        assert "backtest_success_rate" in bp.validation_stats
        assert 0 <= bp.risk_score <= 1
        assert bp.status == "proposed"

        # Check receipt emitted
        captured = capsys.readouterr()
        assert '"receipt_type": "helper_blueprint"' in captured.out

    def test_higher_count_yields_higher_confidence(self):
        """More gap occurrences should yield higher backtest success rate."""
        low_count = [{"problem_type": "error", "count": 5, "gap_ids": [], "sample_gap": {}}]
        high_count = [{"problem_type": "error", "count": 15, "gap_ids": [], "sample_gap": {}}]

        bp_low = hypothesize(low_count)[0]
        bp_high = hypothesize(high_count)[0]

        assert bp_high.validation_stats["backtest_success_rate"] >= bp_low.validation_stats["backtest_success_rate"]
        assert bp_high.risk_score <= bp_low.risk_score


class TestGate:
    """Tests for gate function."""

    def test_gate_auto_approves_low_risk(self, capsys):
        """Low risk + high confidence should auto-approve."""
        bp = HelperBlueprint(
            id="test-bp",
            origin_gaps=["gap1"],
            pattern={"trigger": "test", "action": "test"},
            validation_stats={"backtest_success_rate": 0.95},
            risk_score=0.1,  # Low risk
            status="proposed"
        )

        config = HelperConfig(auto_approve_confidence=0.9)
        decision = gate(bp, config)

        assert decision == "auto_approve"
        assert bp.status == "approved"

        captured = capsys.readouterr()
        assert '"receipt_type": "gate_decision"' in captured.out
        assert '"decision": "auto_approve"' in captured.out

    def test_gate_requires_hitl_high_risk(self):
        """High risk should require HITL."""
        bp = HelperBlueprint(
            id="test-bp",
            origin_gaps=["gap1"],
            pattern={"trigger": "test", "action": "test"},
            validation_stats={"backtest_success_rate": 0.95},
            risk_score=0.5,  # High risk
            status="proposed"
        )

        decision = gate(bp, HelperConfig())

        assert decision == "hitl_required"
        assert bp.status == "proposed"  # Not auto-approved

    def test_gate_requires_hitl_low_confidence(self):
        """Low backtest success rate should require HITL."""
        bp = HelperBlueprint(
            id="test-bp",
            origin_gaps=["gap1"],
            pattern={"trigger": "test", "action": "test"},
            validation_stats={"backtest_success_rate": 0.7},  # Below threshold
            risk_score=0.1,
            status="proposed"
        )

        decision = gate(bp, HelperConfig(auto_approve_confidence=0.9))

        assert decision == "hitl_required"


class TestActuate:
    """Tests for actuate function."""

    def test_actuate_deploys_helper(self, capsys):
        """Approved blueprint should be deployed as active."""
        bp = HelperBlueprint(
            id="test-bp",
            origin_gaps=["gap1", "gap2"],
            pattern={"trigger": "gap.type == 'error'", "action": "auto_fix"},
            validation_stats={"backtest_success_rate": 0.95},
            risk_score=0.1,
            status="approved"
        )

        result = actuate(bp)

        assert result.status == "active"

        captured = capsys.readouterr()
        assert '"receipt_type": "helper_deployment"' in captured.out

    def test_actuate_skips_retired(self):
        """Retired blueprint should not be redeployed."""
        bp = HelperBlueprint(
            id="test-bp",
            origin_gaps=[],
            pattern={},
            validation_stats={},
            risk_score=0.1,
            status="retired"
        )

        result = actuate(bp)

        assert result.status == "retired"


class TestMeasureEffectiveness:
    """Tests for measure_effectiveness function."""

    def test_effectiveness_positive(self, capsys):
        """Active helper with entropy reduction should have positive effectiveness."""
        helper = HelperBlueprint(
            id="test-helper",
            origin_gaps=[],
            pattern={},
            validation_stats={},
            risk_score=0.1,
            status="active"
        )

        # Simulate receipts showing this helper took action
        receipts = [
            {"helper_id": "test-helper", "entropy_before": 1.0, "entropy_after": 0.3},
            {"helper_id": "test-helper", "entropy_before": 0.8, "entropy_after": 0.2},
        ]

        effectiveness = measure_effectiveness(helper, receipts)

        assert effectiveness > 0
        assert helper.actions_taken == 2

        captured = capsys.readouterr()
        assert '"receipt_type": "helper_effectiveness"' in captured.out

    def test_effectiveness_zero_for_inactive(self):
        """Inactive helper should have zero effectiveness."""
        helper = HelperBlueprint(
            id="test-helper",
            origin_gaps=[],
            pattern={},
            validation_stats={},
            risk_score=0.1,
            status="proposed"  # Not active
        )

        effectiveness = measure_effectiveness(helper, [])

        assert effectiveness == 0.0


class TestRetire:
    """Tests for retire function."""

    def test_retirement_on_low_effectiveness(self, capsys):
        """Helper should be retired with reason."""
        helper = HelperBlueprint(
            id="test-helper",
            origin_gaps=["gap1"],
            pattern={},
            validation_stats={},
            risk_score=0.1,
            status="active",
            actions_taken=50,
            effectiveness_sum=0.1
        )

        result = retire(helper, "low_effectiveness")

        assert result.status == "retired"

        captured = capsys.readouterr()
        assert '"receipt_type": "helper_retirement"' in captured.out
        assert '"reason": "low_effectiveness"' in captured.out


class TestCheckRetirementCandidates:
    """Tests for check_retirement_candidates function."""

    def test_finds_low_effectiveness_helpers(self):
        """Should find helpers with effectiveness below threshold."""
        helpers = [
            HelperBlueprint(
                id="good",
                origin_gaps=[],
                pattern={},
                validation_stats={},
                risk_score=0.1,
                status="active",
                actions_taken=20,
                effectiveness_sum=5.0  # 0.25 avg - good
            ),
            HelperBlueprint(
                id="bad",
                origin_gaps=[],
                pattern={},
                validation_stats={},
                risk_score=0.1,
                status="active",
                actions_taken=20,
                effectiveness_sum=0.001  # ~0 avg - bad
            ),
        ]

        candidates = check_retirement_candidates(helpers)

        assert len(candidates) == 1
        assert candidates[0].id == "bad"

    def test_ignores_helpers_with_few_actions(self):
        """Should not retire helpers that haven't had enough actions."""
        helpers = [
            HelperBlueprint(
                id="new",
                origin_gaps=[],
                pattern={},
                validation_stats={},
                risk_score=0.1,
                status="active",
                actions_taken=5,  # Below min_actions threshold
                effectiveness_sum=0.0
            ),
        ]

        candidates = check_retirement_candidates(helpers, min_actions=10)

        assert len(candidates) == 0


class TestGetActiveHelpers:
    """Tests for get_active_helpers function."""

    def test_filters_to_active_only(self):
        """Should return only active helpers."""
        helpers = [
            HelperBlueprint(id="1", origin_gaps=[], pattern={}, validation_stats={}, risk_score=0.1, status="active"),
            HelperBlueprint(id="2", origin_gaps=[], pattern={}, validation_stats={}, risk_score=0.1, status="proposed"),
            HelperBlueprint(id="3", origin_gaps=[], pattern={}, validation_stats={}, risk_score=0.1, status="retired"),
            HelperBlueprint(id="4", origin_gaps=[], pattern={}, validation_stats={}, risk_score=0.1, status="active"),
        ]

        active = get_active_helpers(helpers)

        assert len(active) == 2
        assert all(h.status == "active" for h in active)


class TestCreateGapReceipt:
    """Tests for create_gap_receipt function."""

    def test_creates_valid_gap_receipt(self, capsys):
        """Should create gap receipt with correct fields."""
        receipt = create_gap_receipt("config_error", "Test description", 0.7)

        assert receipt["receipt_type"] == "gap"
        assert receipt["type"] == "config_error"
        assert "id" in receipt

        captured = capsys.readouterr()
        assert '"receipt_type": "gap"' in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
