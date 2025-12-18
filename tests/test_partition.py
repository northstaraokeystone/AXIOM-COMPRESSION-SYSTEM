"""test_partition.py - Partition Resilience Validation

Validates partition resilience across 1000 iterations per CLAUDEME spec.

SLOs:
    - 100% of 1000 iterations complete without violation
    - Receipt ledger contains 1000+ partition_stress_receipts after stress run
    - eff_alpha >= 2.63 at 40% partition
    - α drop < 0.05 from baseline

Source: Grok - "100-10,000 entries", "eff_α to 2.68 (+0.12)"
"""

import pytest
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.partition import (
    partition_sim,
    quorum_check,
    stress_sweep,
    load_partition_spec,
    get_rerouting_potential,
    NODE_BASELINE,
    QUORUM_THRESHOLD,
    PARTITION_MAX_TEST_PCT,
    BASE_ALPHA,
    ALPHA_DROP_FACTOR,
    GREENS_CURRENT,
    REROUTING_POTENTIAL_BOOST,
)
from src.ledger import (
    apply_ledger_boost,
    apply_quorum_factor,
    get_effective_alpha_with_partition,
    LEDGER_ALPHA_BOOST_VALIDATED,
)
from src.reasoning import partition_sweep, validate_resilience_slo, MIN_EFF_ALPHA_BOUND
from src.mitigation import compute_partition_tolerance, compute_mitigation_score
from src.core import StopRule


class TestQuorumSurvives:
    """Test quorum survival under various partition scenarios."""

    def test_quorum_survives_2_node_loss(self):
        """5-node baseline with 2 nodes lost maintains quorum (3 surviving)."""
        # 2 node loss = 40% partition on 5 nodes
        result = partition_sim(
            nodes_total=5,
            loss_pct=0.40,  # 40% = 2 nodes lost
            base_alpha=BASE_ALPHA,
            emit=False,
        )

        assert result["nodes_surviving"] == 3, (
            f"Expected 3 nodes surviving, got {result['nodes_surviving']}"
        )
        assert result["quorum_status"] is True, "Quorum should survive with 3 nodes"
        assert result["nodes_total"] - result["nodes_surviving"] == 2, (
            "Should have lost exactly 2 nodes"
        )

    def test_quorum_survives_1_node_loss(self):
        """5-node baseline with 1 node lost maintains quorum (4 surviving)."""
        result = partition_sim(
            nodes_total=5,
            loss_pct=0.20,  # 20% = 1 node lost
            base_alpha=BASE_ALPHA,
            emit=False,
        )

        assert result["nodes_surviving"] == 4, (
            f"Expected 4 nodes surviving, got {result['nodes_surviving']}"
        )
        assert result["quorum_status"] is True, "Quorum should survive with 4 nodes"

    def test_quorum_full_when_no_loss(self):
        """5-node baseline with 0 loss maintains full quorum."""
        result = partition_sim(
            nodes_total=5, loss_pct=0.0, base_alpha=BASE_ALPHA, emit=False
        )

        assert result["nodes_surviving"] == 5, (
            f"Expected 5 nodes surviving, got {result['nodes_surviving']}"
        )
        assert result["quorum_status"] is True
        assert result["eff_alpha_drop"] == 0.0


class TestAlphaDropWithinBounds:
    """Test that alpha drop stays within acceptable bounds."""

    def test_alpha_drop_at_40_percent(self):
        """At 40% partition, α drop <= 0.05 from baseline."""
        result = partition_sim(
            nodes_total=NODE_BASELINE,
            loss_pct=PARTITION_MAX_TEST_PCT,
            base_alpha=BASE_ALPHA,
            emit=False,
        )

        assert result["eff_alpha_drop"] <= 0.05, (
            f"Alpha drop {result['eff_alpha_drop']} > 0.05 at 40% partition"
        )

        # Verify the drop formula: loss_pct * factor (not multiplied by base)
        expected_drop = PARTITION_MAX_TEST_PCT * ALPHA_DROP_FACTOR
        assert abs(result["eff_alpha_drop"] - expected_drop) < 0.001, (
            f"Drop {result['eff_alpha_drop']} doesn't match formula {expected_drop}"
        )

    def test_alpha_drop_scales_linearly(self):
        """Alpha drop should scale linearly with partition loss."""
        drop_10 = partition_sim(5, 0.10, BASE_ALPHA, emit=False)["eff_alpha_drop"]
        drop_20 = partition_sim(5, 0.20, BASE_ALPHA, emit=False)["eff_alpha_drop"]
        drop_40 = partition_sim(5, 0.40, BASE_ALPHA, emit=False)["eff_alpha_drop"]

        # 20% should be ~2x of 10%
        assert abs(drop_20 / drop_10 - 2.0) < 0.1, "Drop should scale linearly"

        # 40% should be ~4x of 10%
        assert abs(drop_40 / drop_10 - 4.0) < 0.1, "Drop should scale linearly"

    def test_eff_alpha_above_minimum_bound(self):
        """Effective alpha >= 2.63 at max partition."""
        result = partition_sim(
            nodes_total=NODE_BASELINE,
            loss_pct=PARTITION_MAX_TEST_PCT,
            base_alpha=BASE_ALPHA,
            emit=False,
        )

        assert result["eff_alpha"] >= MIN_EFF_ALPHA_BOUND, (
            f"eff_alpha {result['eff_alpha']} < min bound {MIN_EFF_ALPHA_BOUND}"
        )


class TestStress1000Iterations:
    """Stress test with 1000 iterations."""

    def test_stress_1000_iterations(self):
        """Run 1000 random partitions (0-40%), verify SLOs."""
        results = stress_sweep(
            nodes_total=NODE_BASELINE,
            loss_range=(0.0, PARTITION_MAX_TEST_PCT),
            n_iterations=1000,
            base_alpha=BASE_ALPHA,
            seed=42,  # Reproducible
        )

        # (1) All quorum checks pass
        quorum_successes = [r for r in results if r["quorum_status"]]
        assert len(quorum_successes) == 1000, (
            f"Expected 1000 quorum successes, got {len(quorum_successes)}"
        )

        # (2) Avg α drop < 0.05
        avg_drop = sum(r["eff_alpha_drop"] for r in quorum_successes) / len(
            quorum_successes
        )
        assert avg_drop < 0.05, f"Average α drop {avg_drop} >= 0.05"

        # (3) All results are populated
        for i, r in enumerate(results):
            assert "nodes_total" in r, f"Result {i} missing nodes_total"
            assert "loss_pct" in r, f"Result {i} missing loss_pct"
            assert "eff_alpha_drop" in r, f"Result {i} missing eff_alpha_drop"
            assert "quorum_status" in r, f"Result {i} missing quorum_status"

    def test_stress_receipts_count(self):
        """Verify stress sweep emits summary receipt."""
        # Capture stdout for receipt
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            results = stress_sweep(
                nodes_total=NODE_BASELINE,
                loss_range=(0.0, PARTITION_MAX_TEST_PCT),
                n_iterations=100,
                base_alpha=BASE_ALPHA,
                seed=42,
            )

        output = f.getvalue()
        receipts = [json.loads(line) for line in output.strip().split("\n") if line]

        # Should have at least one quorum_resilience receipt
        resilience_receipts = [
            r for r in receipts if r.get("receipt_type") == "quorum_resilience"
        ]
        assert len(resilience_receipts) >= 1, (
            "Expected at least 1 quorum_resilience receipt"
        )

        # Verify receipt contents
        receipt = resilience_receipts[0]
        assert receipt["test_iterations"] == 100
        assert receipt["success_rate"] == 1.0
        assert receipt["avg_alpha_drop"] < 0.05


class TestQuorumFailureRaisesStopRule:
    """Test that losing 3+ nodes on 5-node baseline raises StopRule."""

    def test_quorum_failure_raises_stoprule(self):
        """Losing 3+ nodes on 5-node baseline raises StopRule."""
        # 60% loss = 3 nodes, leaving only 2 (below threshold of 3)
        with pytest.raises(StopRule) as exc_info:
            partition_sim(
                nodes_total=5, loss_pct=0.60, base_alpha=BASE_ALPHA, emit=False
            )

        assert "Quorum failed" in str(exc_info.value)
        assert "2 nodes surviving < 3 threshold" in str(exc_info.value)

    def test_quorum_check_direct_failure(self):
        """Direct quorum_check with 2 surviving raises StopRule."""
        with pytest.raises(StopRule) as exc_info:
            quorum_check(nodes_surviving=2, quorum_min=QUORUM_THRESHOLD)

        assert "Quorum failed" in str(exc_info.value)

    def test_quorum_check_at_threshold_passes(self):
        """Quorum check at exactly threshold passes."""
        result = quorum_check(nodes_surviving=3, quorum_min=QUORUM_THRESHOLD)
        assert result is True

    def test_quorum_check_above_threshold_passes(self):
        """Quorum check above threshold passes."""
        result = quorum_check(nodes_surviving=5, quorum_min=QUORUM_THRESHOLD)
        assert result is True


class TestReceiptsEmitted:
    """Test that all partition_sim calls produce valid receipts."""

    def test_partition_sim_emits_receipt(self):
        """Every partition_sim call produces valid partition_stress_receipt."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = partition_sim(
                nodes_total=5, loss_pct=0.20, base_alpha=BASE_ALPHA, emit=True
            )

        output = f.getvalue()
        # With reroute_enabled=True (default), multiple receipts may be emitted.
        # Parse the last line which should be the partition_stress receipt.
        lines = [l for l in output.strip().split("\n") if l]
        receipt = json.loads(lines[-1])

        # Validate receipt structure
        assert receipt["receipt_type"] == "partition_stress"
        assert receipt["tenant_id"] == "axiom-resilience"
        assert receipt["nodes_total"] == 5
        assert receipt["loss_pct"] == 0.20
        assert "eff_alpha_drop" in receipt
        assert "quorum_status" in receipt
        assert "payload_hash" in receipt
        assert "ts" in receipt

    def test_spec_ingest_emits_receipt(self):
        """load_partition_spec emits ingest receipt with dual_hash."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_partition_spec()

        output = f.getvalue()
        receipt = json.loads(output.strip())

        assert receipt["receipt_type"] == "partition_spec_ingest"
        assert receipt["node_baseline"] == NODE_BASELINE
        assert receipt["quorum_min"] == QUORUM_THRESHOLD
        assert "payload_hash" in receipt


class TestLedgerIntegration:
    """Test ledger module integration with partition."""

    def test_ledger_boost_value(self):
        """Verify ledger boost is 0.12."""
        assert LEDGER_ALPHA_BOOST_VALIDATED == 0.12
        assert apply_ledger_boost(2.56, 0.12) == 2.68

    def test_quorum_factor_full(self):
        """Full quorum applies no degradation."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = apply_quorum_factor(
                base_alpha=2.68, nodes_surviving=5, nodes_baseline=5
            )

        assert result == 2.68  # No degradation

    def test_quorum_factor_degraded(self):
        """Degraded quorum applies degradation per missing node."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = apply_quorum_factor(
                base_alpha=2.68, nodes_surviving=3, nodes_baseline=5
            )

        # 2 missing nodes × 0.02 degradation = 0.04
        expected = 2.68 - 0.04
        assert abs(result - expected) < 0.001

    def test_effective_alpha_with_partition(self):
        """get_effective_alpha_with_partition combines boost and partition."""
        result = get_effective_alpha_with_partition(loss_pct=0.40, base_alpha=2.56)

        assert result["ledger_boost"] == 0.12
        assert result["boosted_alpha"] == 2.68
        assert result["effective_alpha"] >= 2.63


class TestReasoningIntegration:
    """Test reasoning module integration."""

    def test_partition_sweep_results(self):
        """partition_sweep returns valid resilience report."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            report = partition_sweep(
                nodes=5, loss_range=(0.0, 0.40), iterations=50, base_alpha=BASE_ALPHA
            )

        assert report["nodes"] == 5
        assert report["iterations"] == 50
        assert report["quorum_success_rate"] == 1.0
        assert report["worst_case_drop"] <= 0.05  # At boundary at 40%
        assert report["min_eff_alpha"] >= 2.63

    def test_resilience_slo_validation(self):
        """validate_resilience_slo passes all checks."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = validate_resilience_slo(
                nodes=NODE_BASELINE,
                max_loss=PARTITION_MAX_TEST_PCT,
                min_alpha=MIN_EFF_ALPHA_BOUND,
                max_drop=0.05,
            )

        assert result["all_passed"] is True
        assert result["alpha_slo"] is True
        assert result["drop_slo"] is True
        assert result["quorum_slo"] is True


class TestMitigationIntegration:
    """Test mitigation module integration."""

    def test_partition_tolerance_full(self):
        """Full partition tolerance at 0% loss."""
        score = compute_partition_tolerance(0.0, BASE_ALPHA)
        assert score == 1.0

    def test_partition_tolerance_at_max(self):
        """Partition tolerance at max loss."""
        score = compute_partition_tolerance(PARTITION_MAX_TEST_PCT, BASE_ALPHA)
        # At max, drop is ~PARTITION_MITIGATION_FACTOR, so score ~0
        assert score >= 0.0

    def test_mitigation_score_combined(self):
        """Combined mitigation score includes all factors."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            score = compute_mitigation_score(
                loss_pct=0.20, nodes_surviving=4, receipt_integrity=0.9
            )

        assert 0.0 <= score.partition_score <= 1.0
        assert 0.0 <= score.quorum_score <= 1.0
        assert 0.0 <= score.tau_score <= 1.0
        assert 0.0 <= score.combined_score <= 1.0
        assert score.effective_alpha >= 2.60


class TestConstants:
    """Verify all constants are correctly defined."""

    def test_node_baseline(self):
        """NODE_BASELINE = 5."""
        assert NODE_BASELINE == 5

    def test_quorum_threshold(self):
        """QUORUM_THRESHOLD = 3."""
        assert QUORUM_THRESHOLD == 3

    def test_partition_max(self):
        """PARTITION_MAX_TEST_PCT = 0.40."""
        assert PARTITION_MAX_TEST_PCT == 0.40

    def test_alpha_drop_factor(self):
        """ALPHA_DROP_FACTOR = 0.125."""
        assert ALPHA_DROP_FACTOR == 0.125

    def test_greens_current(self):
        """GREENS_CURRENT = 81."""
        assert GREENS_CURRENT == 81

    def test_rerouting_potential(self):
        """REROUTING_POTENTIAL_BOOST = 0.07."""
        assert REROUTING_POTENTIAL_BOOST == 0.07

    def test_base_alpha(self):
        """BASE_ALPHA = 2.68."""
        assert BASE_ALPHA == 2.68

    def test_ledger_boost(self):
        """LEDGER_ALPHA_BOOST_VALIDATED = 0.12."""
        assert LEDGER_ALPHA_BOOST_VALIDATED == 0.12


class TestSpecFile:
    """Test partition spec file loading."""

    def test_spec_file_loads(self):
        """Partition spec file loads correctly."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_partition_spec()

        assert spec["node_baseline"] == 5
        assert spec["quorum_min"] == 3
        assert spec["partition_test_range"] == [0.0, 0.40]
        assert spec["greens_current"] == 81
        assert spec["ledger_alpha_boost"] == 0.12
        assert spec["rerouting_potential"] == 0.07


class TestReroutingStub:
    """Test adaptive rerouting stub."""

    def test_rerouting_potential_stub(self):
        """Rerouting potential stub returns expected values."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = get_rerouting_potential()

        assert result["boost_potential"] == 0.07
        assert result["target_alpha"] == BASE_ALPHA + 0.07
        assert result["status"] == "stub"
        assert result["next_gate"] is True
