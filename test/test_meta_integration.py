"""Tests for meta_integration.py - Meta-Loop topology classification."""

from spaceproof.meta_integration import (
    classify_pattern,
    calculate_effectiveness,
    calculate_autonomy,
    calculate_transfer_score,
    emit_topology_receipt,
    trigger_cascade,
    transfer_pattern,
    compute_confidence,
    should_trigger_fallback,
    classify_all_patterns,
    process_graduated_patterns,
    get_domain_escape_velocity,
    validate_entropy_conservation,
    Topology,
    CASCADE_MULTIPLIER,
    CONFIDENCE_FALLBACK,
)


class TestClassifyPattern:
    """Tests for classify_pattern function."""

    def test_open_topology(self):
        """Test pattern classified as open (graduate)."""
        pattern = {
            "effectiveness": 0.95,  # >= 0.90 for orbital_compute
            "autonomy": 0.80,  # > 0.75
        }
        topology = classify_pattern(pattern, "orbital_compute")

        assert topology == "open"

    def test_closed_topology(self):
        """Test pattern classified as closed (optimize)."""
        pattern = {
            "effectiveness": 0.5,
            "autonomy": 0.5,
            "transfer_score": 0.5,
        }
        topology = classify_pattern(pattern, "orbital_compute")

        assert topology == "closed"

    def test_hybrid_topology(self):
        """Test pattern classified as hybrid (transfer)."""
        pattern = {
            "effectiveness": 0.5,  # Below escape velocity
            "autonomy": 0.5,  # Below autonomy threshold
            "transfer_score": 0.80,  # > 0.70
        }
        topology = classify_pattern(pattern, "orbital_compute")

        assert topology == "hybrid"


class TestCalculateEffectiveness:
    """Tests for calculate_effectiveness function."""

    def test_positive_reduction(self):
        """Test effectiveness with entropy reduction."""
        pattern = {
            "entropy_before": 0.8,
            "entropy_after": 0.3,
            "n_receipts": 10,
        }
        effectiveness = calculate_effectiveness(pattern)

        assert effectiveness > 0
        assert effectiveness <= 1.0

    def test_zero_receipts(self):
        """Test zero receipts returns zero."""
        pattern = {"n_receipts": 0}
        assert calculate_effectiveness(pattern) == 0.0


class TestCalculateAutonomy:
    """Tests for calculate_autonomy function."""

    def test_full_autonomy(self):
        """Test full autonomy."""
        pattern = {
            "auto_approved": 10,
            "total_actions": 10,
        }
        assert calculate_autonomy(pattern) == 1.0

    def test_no_autonomy(self):
        """Test no autonomy."""
        pattern = {
            "auto_approved": 0,
            "total_actions": 10,
        }
        assert calculate_autonomy(pattern) == 0.0

    def test_zero_actions(self):
        """Test zero actions."""
        pattern = {"total_actions": 0}
        assert calculate_autonomy(pattern) == 0.0


class TestCalculateTransferScore:
    """Tests for calculate_transfer_score function."""

    def test_high_compatibility(self):
        """Test high compatibility domains."""
        pattern = {
            "domain": "firmware_integrity",
            "effectiveness": 0.9,
        }
        score = calculate_transfer_score(pattern, "constellation_ops")

        assert 0 <= score <= 1

    def test_same_domain(self):
        """Test transfer to same domain."""
        pattern = {"domain": "orbital_compute", "effectiveness": 0.8}
        score = calculate_transfer_score(pattern, "orbital_compute")

        # Should still return a reasonable score
        assert 0 <= score <= 1


class TestEmitTopologyReceipt:
    """Tests for emit_topology_receipt function."""

    def test_emit_receipt(self):
        """Test topology receipt emission."""
        pattern = {"pattern_id": "test-123"}
        result = emit_topology_receipt(
            pattern=pattern,
            domain="orbital_compute",
            topology="open",
            effectiveness=0.95,
            autonomy=0.80,
            transfer=0.5,
            confidence=0.9,
        )

        assert result.topology == Topology.OPEN
        assert result.action == "cascade"
        assert result.receipt is not None


class TestTriggerCascade:
    """Tests for trigger_cascade function."""

    def test_cascade_spawns_five(self):
        """Test CASCADE spawns exactly 5 variants."""
        pattern = {"pattern_id": "parent-123", "effectiveness": 0.95}
        result = trigger_cascade(pattern)

        assert len(result.child_pattern_ids) == CASCADE_MULTIPLIER
        assert result.parent_pattern_id == "parent-123"

    def test_cascade_mutation_rate(self):
        """Test cascade mutation rate."""
        pattern = {"pattern_id": "test"}
        result = trigger_cascade(pattern, mutation_rate=0.10)

        assert result.mutation_rate == 0.10


class TestTransferPattern:
    """Tests for transfer_pattern function."""

    def test_transfer_result(self):
        """Test pattern transfer result."""
        pattern = {"pattern_id": "test-123", "effectiveness": 0.8}
        result = transfer_pattern(pattern, "orbital_compute", "constellation_ops")

        assert result.from_domain == "orbital_compute"
        assert result.to_domain == "constellation_ops"
        assert result.transfer_score > 0


class TestComputeConfidence:
    """Tests for compute_confidence function."""

    def test_high_confidence(self):
        """Test high confidence with many receipts."""
        pattern = {
            "n_receipts": 500,
            "effectiveness": 0.9,
        }
        confidence = compute_confidence(pattern)

        assert confidence > 0.5

    def test_low_confidence(self):
        """Test low confidence with few receipts."""
        pattern = {
            "n_receipts": 5,
            "effectiveness": 0.3,
        }
        confidence = compute_confidence(pattern)

        assert confidence < 0.8


class TestShouldTriggerFallback:
    """Tests for should_trigger_fallback function."""

    def test_trigger_below_threshold(self):
        """Test fallback triggered below threshold."""
        assert should_trigger_fallback(0.5) is True
        assert should_trigger_fallback(CONFIDENCE_FALLBACK - 0.01) is True

    def test_no_trigger_above_threshold(self):
        """Test no fallback above threshold."""
        assert should_trigger_fallback(0.95) is False
        assert should_trigger_fallback(CONFIDENCE_FALLBACK + 0.01) is False


class TestClassifyAllPatterns:
    """Tests for classify_all_patterns function."""

    def test_classify_multiple(self):
        """Test classifying multiple patterns."""
        patterns = [
            {"pattern_id": "1", "entropy_before": 0.8, "entropy_after": 0.1, "n_receipts": 10},
            {"pattern_id": "2", "entropy_before": 0.5, "entropy_after": 0.4, "n_receipts": 5},
        ]

        results = classify_all_patterns(patterns, "orbital_compute")

        assert len(results) == 2


class TestProcessGraduatedPatterns:
    """Tests for process_graduated_patterns function."""

    def test_process_open_patterns(self):
        """Test processing open patterns triggers cascade."""

        class MockResult:
            topology = Topology.OPEN
            pattern_id = "test"
            effectiveness = 0.95
            domain = "orbital_compute"

        results = [MockResult()]
        cascades, transfers = process_graduated_patterns(results)

        assert len(cascades) == 1
        assert len(transfers) == 0

    def test_process_hybrid_patterns(self):
        """Test processing hybrid patterns triggers transfer."""

        class MockResult:
            topology = Topology.HYBRID
            pattern_id = "test"
            effectiveness = 0.7
            domain = "orbital_compute"

        results = [MockResult()]
        cascades, transfers = process_graduated_patterns(results)

        assert len(cascades) == 0
        assert len(transfers) == 1


class TestEscapeVelocity:
    """Tests for get_domain_escape_velocity function."""

    def test_known_domains(self):
        """Test known domain escape velocities."""
        assert get_domain_escape_velocity("orbital_compute") == 0.90
        assert get_domain_escape_velocity("constellation_ops") == 0.85
        assert get_domain_escape_velocity("autonomous_decision") == 0.88
        assert get_domain_escape_velocity("firmware_integrity") == 0.80

    def test_unknown_domain(self):
        """Test unknown domain returns default."""
        assert get_domain_escape_velocity("unknown") == 0.85


class TestEntropyConservation:
    """Tests for validate_entropy_conservation function."""

    def test_valid_conservation(self):
        """Test valid entropy conservation."""
        patterns = [
            {"pattern_id": "1", "entropy_before": 0.5, "entropy_after": 0.505},
            {"pattern_id": "2", "entropy_before": 0.6, "entropy_after": 0.598},
        ]

        assert validate_entropy_conservation(patterns) is True

    def test_violation_detected(self):
        """Test entropy conservation violation."""
        patterns = [
            {"pattern_id": "1", "entropy_before": 0.5, "entropy_after": 0.6},  # 0.1 > 0.01
        ]

        assert validate_entropy_conservation(patterns) is False
