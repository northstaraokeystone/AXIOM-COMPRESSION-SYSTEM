"""Tests for autonomous_decision.py - Defense DOD 3000.09 compliance."""

from spaceproof.domain.autonomous_decision import (
    log_sensor_inputs,
    log_decision,
    log_human_override,
    validate_accountability,
    emit_decision_lineage,
    get_criticality_for_decision_type,
    should_require_hitl,
    should_require_hotl,
    compute_transfer_score,
    generate_training_examples,
    DecisionCriticality,
    OverrideReasonCode,
)


class TestSensorInputs:
    """Tests for log_sensor_inputs function."""

    def test_sensor_input_returns_result(self):
        """Test that sensor input returns valid result."""
        sensor_data = {"radar": 5, "ir": 3}
        result = log_sensor_inputs(sensor_data)

        assert result.input_id is not None
        assert result.input_hash is not None
        assert ":" in result.input_hash  # Dual-hash format
        assert result.receipt is not None

    def test_sensor_input_captures_timestamp(self):
        """Test that sensor input captures timestamp."""
        result = log_sensor_inputs({"data": "test"})

        assert result.timestamp is not None
        assert "Z" in result.timestamp  # ISO8601 format


class TestDecision:
    """Tests for log_decision function."""

    def test_decision_returns_result(self):
        """Test that decision returns valid result."""
        decision = log_decision(
            inputs_hash="abc123:def456",
            algorithm_id="classifier-v1",
            output={"action": "track"},
            confidence=0.9,
        )

        assert decision.decision_id is not None
        assert decision.output_hash is not None
        assert decision.confidence == 0.9
        assert decision.receipt is not None

    def test_decision_override_always_available(self):
        """Test that override is always available per DOD 3000.09."""
        decision = log_decision(
            inputs_hash="abc:def",
            algorithm_id="test",
            output={},
            confidence=0.5,
        )

        assert decision.override_available is True

    def test_decision_criticality_levels(self):
        """Test decision criticality levels."""
        for crit in DecisionCriticality:
            decision = log_decision(
                inputs_hash="abc:def",
                algorithm_id="test",
                output={},
                confidence=0.5,
                criticality=crit,
            )
            assert decision.criticality == crit


class TestHumanOverride:
    """Tests for log_human_override function."""

    def test_override_returns_result(self):
        """Test that override returns valid result."""
        override = log_human_override(
            decision_id="decision-123",
            human_id="operator-001",
            override_reason=OverrideReasonCode.SAFETY_CONCERN,
            justification="Potential civilian presence",
        )

        assert override.override_id is not None
        assert override.reason_code == OverrideReasonCode.SAFETY_CONCERN
        assert override.receipt is not None

    def test_override_captures_corrected_output(self):
        """Test that override captures corrected output."""
        corrected = {"action": "abort", "reason": "safety"}
        override = log_human_override(
            decision_id="decision-123",
            human_id="operator-001",
            override_reason=OverrideReasonCode.POLICY_VIOLATION,
            corrected_output=corrected,
        )

        assert override.corrected_output == corrected

    def test_all_reason_codes(self):
        """Test all override reason codes."""
        for reason in OverrideReasonCode:
            override = log_human_override(
                decision_id="test",
                human_id="operator",
                override_reason=reason,
            )
            assert override.reason_code == reason


class TestAccountabilityValidation:
    """Tests for validate_accountability function."""

    def test_valid_chain(self):
        """Test valid accountability chain."""

        class MockDecision:
            def __init__(self):
                self.decision_id = "test"
                self.override_available = True
                self.inputs_hash = "abc:def"
                self.criticality = DecisionCriticality.LOW
                self.human_override_occurred = False

        decisions = [MockDecision(), MockDecision()]
        validation = validate_accountability(decisions)

        assert validation.valid is True
        assert validation.override_available_all is True
        assert validation.human_accountability_proven is True

    def test_missing_override_flag(self):
        """Test missing override flag is caught."""

        class MockDecision:
            def __init__(self):
                self.decision_id = "test"
                self.override_available = False
                self.inputs_hash = "abc:def"
                self.criticality = DecisionCriticality.LOW
                self.human_override_occurred = False

        decisions = [MockDecision()]
        validation = validate_accountability(decisions)

        assert validation.override_available_all is False
        assert len(validation.violations) > 0


class TestDecisionLineage:
    """Tests for emit_decision_lineage function."""

    def test_empty_lineage(self):
        """Test empty receipts list."""
        receipt = emit_decision_lineage([])

        assert receipt["receipt_type"] == "decision_lineage_chain"
        assert receipt["receipt_count"] == 0

    def test_lineage_with_receipts(self):
        """Test lineage with receipts."""
        receipts = [
            {"receipt_type": "sensor_input"},
            {"receipt_type": "decision_lineage", "override_available": True},
        ]

        receipt = emit_decision_lineage(receipts)

        assert receipt["receipt_count"] == 2
        assert receipt["override_available"] is True


class TestCriticalityMapping:
    """Tests for get_criticality_for_decision_type function."""

    def test_critical_types(self):
        """Test critical decision types."""
        assert get_criticality_for_decision_type("weapon_release") == DecisionCriticality.CRITICAL
        assert get_criticality_for_decision_type("lethal_force") == DecisionCriticality.CRITICAL

    def test_high_types(self):
        """Test high criticality types."""
        assert get_criticality_for_decision_type("intercept") == DecisionCriticality.HIGH
        assert get_criticality_for_decision_type("engage") == DecisionCriticality.HIGH

    def test_medium_types(self):
        """Test medium criticality types."""
        assert get_criticality_for_decision_type("track") == DecisionCriticality.MEDIUM
        assert get_criticality_for_decision_type("navigate") == DecisionCriticality.MEDIUM

    def test_low_types(self):
        """Test low criticality types."""
        assert get_criticality_for_decision_type("report") == DecisionCriticality.LOW


class TestHITLHOTL:
    """Tests for HITL/HOTL requirements."""

    def test_hitl_for_critical(self):
        """Test HITL required for critical decisions."""

        class MockDecision:
            criticality = DecisionCriticality.CRITICAL

        assert should_require_hitl(MockDecision()) is True

    def test_hotl_for_high(self):
        """Test HOTL required for high decisions."""

        class MockDecision:
            criticality = DecisionCriticality.HIGH

        assert should_require_hotl(MockDecision()) is True

    def test_no_hitl_for_low(self):
        """Test no HITL for low decisions."""

        class MockDecision:
            criticality = DecisionCriticality.LOW

        assert should_require_hitl(MockDecision()) is False


class TestTransferScore:
    """Tests for compute_transfer_score function."""

    def test_transfer_between_similar_domains(self):
        """Test transfer score between similar domains."""

        class MockDecision:
            algorithm_id = "shared-algo"
            confidence = 0.8

        source = [MockDecision()]
        target = [MockDecision()]

        score = compute_transfer_score(source, target)
        assert 0 <= score <= 1

    def test_empty_decisions(self):
        """Test empty decisions lists."""
        assert compute_transfer_score([], []) == 0.0


class TestTrainingExamples:
    """Tests for generate_training_examples function."""

    def test_generate_examples(self):
        """Test training example generation."""

        class MockOverride:
            decision_id = "test"
            reason_code = OverrideReasonCode.FACTUAL_ERROR
            corrected_output = {"action": "abort"}
            justification = "Test reason"

        overrides = [MockOverride()]
        examples = generate_training_examples(overrides)

        assert len(examples) == 1
        assert examples[0]["decision_id"] == "test"
        assert examples[0]["reason_code"] == "FACTUAL_ERROR"
