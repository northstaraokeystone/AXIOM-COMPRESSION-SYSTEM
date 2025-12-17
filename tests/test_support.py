"""test_support.py - Tests for L0-L4 receipt level infrastructure

Validates ProofPack v3 §3.1 patterns:
- Five receipt levels (L0-L4)
- Coverage measurement
- Self-verification via L4→L0 feedback
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.support import (
    SupportLevel,
    SupportCoverage,
    classify_receipt,
    measure_coverage,
    check_completeness,
    detect_gaps,
    l4_feedback,
    initialize_coverage,
    get_level_summary,
    LEVEL_RECEIPT_TYPES,
    EXPECTED_TYPES,
)


class TestClassifyReceipt:
    """Tests for classify_receipt function."""

    def test_classify_receipt_l0(self):
        """Telemetry receipt should classify as L0."""
        receipt = {"receipt_type": "autonomy_state", "state": "active"}
        level = classify_receipt(receipt)
        assert level == SupportLevel.L0_TELEMETRY

    def test_classify_receipt_l0_other_types(self):
        """Various L0 types should classify correctly."""
        l0_types = ["propulsion_state", "latency", "bandwidth", "telemetry", "heartbeat"]

        for rtype in l0_types:
            receipt = {"receipt_type": rtype}
            level = classify_receipt(receipt)
            assert level == SupportLevel.L0_TELEMETRY, f"{rtype} should be L0"

    def test_classify_receipt_l1(self):
        """Agent receipt should classify as L1."""
        receipt = {"receipt_type": "optimization", "patterns_selected": 3}
        level = classify_receipt(receipt)
        assert level == SupportLevel.L1_AGENTS

    def test_classify_receipt_l1_other_types(self):
        """Various L1 types should classify correctly."""
        l1_types = ["decision", "gate_decision", "stage_gate", "agent_action"]

        for rtype in l1_types:
            receipt = {"receipt_type": rtype}
            level = classify_receipt(receipt)
            assert level == SupportLevel.L1_AGENTS, f"{rtype} should be L1"

    def test_classify_receipt_l2(self):
        """Deployment receipt should classify as L2."""
        receipt = {"receipt_type": "helper_deployment", "helper_id": "123"}
        level = classify_receipt(receipt)
        assert level == SupportLevel.L2_CHANGES

    def test_classify_receipt_l3(self):
        """Quality receipt should classify as L3."""
        receipt = {"receipt_type": "helper_effectiveness", "score": 0.85}
        level = classify_receipt(receipt)
        assert level == SupportLevel.L3_QUALITY

    def test_classify_receipt_l4(self):
        """Meta receipt should classify as L4."""
        receipt = {"receipt_type": "support_level", "level": "L0"}
        level = classify_receipt(receipt)
        assert level == SupportLevel.L4_META

    def test_unknown_type_defaults_to_l0(self):
        """Unknown receipt type should default to L0."""
        receipt = {"receipt_type": "unknown_type_xyz"}
        level = classify_receipt(receipt)
        assert level == SupportLevel.L0_TELEMETRY


class TestMeasureCoverage:
    """Tests for measure_coverage function."""

    def test_coverage_all_levels(self, capsys):
        """After varied receipts, all levels should have some coverage."""
        receipts = [
            # L0
            {"receipt_type": "autonomy_state", "state": "active"},
            {"receipt_type": "propulsion_state", "state": "nominal"},
            {"receipt_type": "latency", "ms": 1320000},
            # L1
            {"receipt_type": "optimization", "selected": 3},
            {"receipt_type": "decision", "action": "proceed"},
            {"receipt_type": "gate_decision", "decision": "approve"},
            # L2
            {"receipt_type": "helper_deployment", "id": "h1"},
            {"receipt_type": "helper_blueprint", "id": "b1"},
            {"receipt_type": "config_change", "key": "rate"},
            # L3
            {"receipt_type": "helper_effectiveness", "score": 0.8},
            {"receipt_type": "validation", "passed": True},
            {"receipt_type": "chain", "n": 100},
            # L4
            {"receipt_type": "support_level", "level": "L0"},
            {"receipt_type": "coverage", "ratio": 0.9},
            {"receipt_type": "harvest", "patterns": 2},
        ]

        coverage = measure_coverage(receipts)

        # All levels should have receipts
        for level in SupportLevel:
            assert coverage[level].receipt_count > 0, f"{level} should have receipts"
            assert coverage[level].coverage_ratio > 0, f"{level} should have coverage"

        # Check support_level receipts emitted
        captured = capsys.readouterr()
        assert '"receipt_type": "support_level"' in captured.out

    def test_coverage_empty_receipts(self):
        """Empty receipts should yield zero coverage."""
        coverage = measure_coverage([])

        for level in SupportLevel:
            assert coverage[level].receipt_count == 0
            assert coverage[level].coverage_ratio == 0.0

    def test_coverage_tracks_gaps(self):
        """Should identify missing receipt types per level."""
        # Only provide some L0 types
        receipts = [
            {"receipt_type": "autonomy_state", "state": "active"},
            # Missing: propulsion_state, latency
        ]

        coverage = measure_coverage(receipts)

        l0_cov = coverage[SupportLevel.L0_TELEMETRY]
        assert "propulsion_state" in l0_cov.gaps or "latency" in l0_cov.gaps


class TestCheckCompleteness:
    """Tests for check_completeness function."""

    def test_completeness_at_threshold(self):
        """Coverage ≥0.95 for all should return True."""
        coverage = {
            level: SupportCoverage(
                level=level,
                receipt_count=100,
                coverage_ratio=0.95,
                gaps=[]
            )
            for level in SupportLevel
        }

        assert check_completeness(coverage) is True

    def test_completeness_below_threshold(self):
        """Coverage <0.95 for any should return False."""
        coverage = {
            level: SupportCoverage(
                level=level,
                receipt_count=100,
                coverage_ratio=0.95 if level != SupportLevel.L2_CHANGES else 0.8,
                gaps=[]
            )
            for level in SupportLevel
        }

        assert check_completeness(coverage) is False


class TestDetectGaps:
    """Tests for detect_gaps function."""

    def test_detect_gaps_returns_formatted(self):
        """Should return 'LEVEL: type' formatted gaps."""
        coverage = {
            SupportLevel.L0_TELEMETRY: SupportCoverage(
                level=SupportLevel.L0_TELEMETRY,
                receipt_count=10,
                coverage_ratio=0.67,
                gaps=["latency"]
            ),
            SupportLevel.L1_AGENTS: SupportCoverage(
                level=SupportLevel.L1_AGENTS,
                receipt_count=5,
                coverage_ratio=0.5,
                gaps=["decision", "gate_decision"]
            ),
            SupportLevel.L2_CHANGES: SupportCoverage(
                level=SupportLevel.L2_CHANGES,
                receipt_count=0,
                coverage_ratio=0.0,
                gaps=["helper_deployment"]
            ),
            SupportLevel.L3_QUALITY: SupportCoverage(
                level=SupportLevel.L3_QUALITY,
                receipt_count=10,
                coverage_ratio=1.0,
                gaps=[]
            ),
            SupportLevel.L4_META: SupportCoverage(
                level=SupportLevel.L4_META,
                receipt_count=10,
                coverage_ratio=1.0,
                gaps=[]
            ),
        }

        gaps = detect_gaps(coverage)

        assert "L0: latency" in gaps
        assert "L1: decision" in gaps
        assert "L2: helper_deployment" in gaps


class TestL4Feedback:
    """Tests for l4_feedback function."""

    def test_l4_feedback_works(self, capsys):
        """L4 insights should improve L0 params."""
        coverage = {
            SupportLevel.L0_TELEMETRY: SupportCoverage(
                level=SupportLevel.L0_TELEMETRY,
                receipt_count=5,
                coverage_ratio=0.5,  # Low coverage
                gaps=["latency", "propulsion_state"]
            ),
            SupportLevel.L1_AGENTS: SupportCoverage(
                level=SupportLevel.L1_AGENTS,
                receipt_count=10,
                coverage_ratio=0.9,
                gaps=[]
            ),
            SupportLevel.L2_CHANGES: SupportCoverage(
                level=SupportLevel.L2_CHANGES,
                receipt_count=10,
                coverage_ratio=0.9,
                gaps=[]
            ),
            SupportLevel.L3_QUALITY: SupportCoverage(
                level=SupportLevel.L3_QUALITY,
                receipt_count=3,
                coverage_ratio=0.6,  # Low quality coverage
                gaps=["validation"]
            ),
            SupportLevel.L4_META: SupportCoverage(
                level=SupportLevel.L4_META,
                receipt_count=10,
                coverage_ratio=0.9,
                gaps=[]
            ),
        }

        l0_params = {"sample_rate": 1.0, "telemetry_level": "normal"}

        improved = l4_feedback(coverage, l0_params)

        # Should suggest improvements
        assert improved["l4_feedback_applied"] is True
        assert "telemetry_level" in improved
        assert improved["telemetry_level"] == "verbose"  # Upgraded due to low L0 coverage
        assert "enable_types" in improved  # Specific gaps to enable
        assert "validation_frequency" in improved  # Due to low L3 coverage

        # Check meta receipt emitted
        captured = capsys.readouterr()
        assert '"self_verifying": true' in captured.out

    def test_l4_feedback_preserves_good_params(self):
        """L4 feedback should preserve params when coverage is good."""
        coverage = {
            level: SupportCoverage(
                level=level,
                receipt_count=100,
                coverage_ratio=0.95,
                gaps=[]
            )
            for level in SupportLevel
        }

        l0_params = {"sample_rate": 1.0, "telemetry_level": "normal"}

        improved = l4_feedback(coverage, l0_params)

        assert improved["l4_feedback_applied"] is True
        # Original params preserved when coverage is good
        assert improved["sample_rate"] == 1.0


class TestInitializeCoverage:
    """Tests for initialize_coverage function."""

    def test_creates_all_levels(self):
        """Should create coverage entry for all 5 levels."""
        coverage = initialize_coverage()

        assert len(coverage) == 5
        for level in SupportLevel:
            assert level in coverage
            assert coverage[level].receipt_count == 0
            assert coverage[level].coverage_ratio == 0.0


class TestGetLevelSummary:
    """Tests for get_level_summary function."""

    def test_formats_summary(self):
        """Should generate readable summary."""
        coverage = {
            SupportLevel.L0_TELEMETRY: SupportCoverage(
                SupportLevel.L0_TELEMETRY, 100, 1.0, []
            ),
            SupportLevel.L1_AGENTS: SupportCoverage(
                SupportLevel.L1_AGENTS, 50, 0.95, []
            ),
            SupportLevel.L2_CHANGES: SupportCoverage(
                SupportLevel.L2_CHANGES, 20, 0.8, ["config_change"]
            ),
            SupportLevel.L3_QUALITY: SupportCoverage(
                SupportLevel.L3_QUALITY, 30, 0.9, []
            ),
            SupportLevel.L4_META: SupportCoverage(
                SupportLevel.L4_META, 10, 1.0, []
            ),
        }

        summary = get_level_summary(coverage)

        assert "L0" in summary
        assert "L1" in summary
        assert "L2" in summary
        assert "L3" in summary
        assert "L4" in summary
        assert "GAP" in summary  # L2 has gap
        assert "config_change" in summary


class TestSupportLevelEnum:
    """Tests for SupportLevel enum."""

    def test_five_levels(self):
        """Should have exactly 5 levels."""
        assert len(SupportLevel) == 5

    def test_level_values(self):
        """Levels should have L0-L4 values."""
        assert SupportLevel.L0_TELEMETRY.value == "L0"
        assert SupportLevel.L1_AGENTS.value == "L1"
        assert SupportLevel.L2_CHANGES.value == "L2"
        assert SupportLevel.L3_QUALITY.value == "L3"
        assert SupportLevel.L4_META.value == "L4"


class TestLevelReceiptTypeMappings:
    """Tests for receipt type to level mappings."""

    def test_no_overlap_between_levels(self):
        """Each receipt type should only appear in one level."""
        all_types = set()

        for level, types in LEVEL_RECEIPT_TYPES.items():
            overlap = all_types & types
            assert len(overlap) == 0, f"Overlap found: {overlap}"
            all_types.update(types)

    def test_expected_types_subset_of_level_types(self):
        """Expected types should be subset of level types."""
        for level, expected in EXPECTED_TYPES.items():
            level_types = LEVEL_RECEIPT_TYPES.get(level, set())
            assert expected.issubset(level_types), f"Expected types for {level} not in level types"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
