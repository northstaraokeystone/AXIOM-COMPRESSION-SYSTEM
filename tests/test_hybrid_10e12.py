"""Test suite for 10^12 hybrid benchmark and fractal recursion.

FULL-SYSTEM TESTS ONLY - Per directive: "Stop: Single-module tests"

Tests cover:
- 10^12 benchmark (5 tests)
- Scale decay validation (3 tests)
- Release gate 3.1 (3 tests)
- Fractal recursion (5 tests)
- Recursion sweep (4 tests)
- Receipts (5 tests)
"""

import json
import pytest

from src.hybrid_benchmark import (
    benchmark_10e12,
    validate_scale_decay,
    check_release_gate_3_1,
    get_benchmark_info,
    get_hybrid_10e12_spec,
    TREE_10E12,
    ALPHA_10E12_FLOOR,
    ALPHA_10E12_TARGET,
    SCALE_DECAY_MAX,
    INSTABILITY_MAX,
)
from src.fractal_layers import (
    recursive_fractal,
    recursive_fractal_sweep,
    get_recursive_fractal_info,
    FRACTAL_RECURSION_MAX_DEPTH,
    FRACTAL_RECURSION_DEFAULT_DEPTH,
    FRACTAL_RECURSION_DECAY,
    FRACTAL_UPLIFT,
)


# === 10^12 BENCHMARK TESTS (5) ===


class TestBenchmark10E12:
    """Full-system tests for 10^12 hybrid benchmark."""

    def test_10e12_alpha_floor(self):
        """Test eff_alpha >= 3.05 at 10^12 scale."""
        result = benchmark_10e12()
        assert result["eff_alpha"] >= ALPHA_10E12_FLOOR, (
            f"eff_alpha {result['eff_alpha']} < floor {ALPHA_10E12_FLOOR}"
        )

    def test_10e12_instability_zero(self):
        """Test instability == 0.00 at 10^12 scale."""
        result = benchmark_10e12()
        assert result["instability"] == INSTABILITY_MAX, (
            f"instability {result['instability']} != 0.00"
        )

    def test_10e12_scale_decay(self):
        """Test scale decay <= 0.02 at 10^12 scale."""
        result = benchmark_10e12()
        assert result["scale_decay"] <= SCALE_DECAY_MAX, (
            f"scale_decay {result['scale_decay']} > max {SCALE_DECAY_MAX}"
        )

    def test_10e12_gate_pass(self):
        """Test full gate pass at 10^12 scale."""
        result = benchmark_10e12()
        assert result["gate_pass"] is True, (
            f"Gate failed: alpha_ok={result['validation']['alpha_ok']}, "
            f"instability_ok={result['validation']['instability_ok']}, "
            f"decay_ok={result['validation']['decay_ok']}"
        )

    def test_10e12_quantum_fractal_synergy(self):
        """Test quantum + fractal contributions are present."""
        result = benchmark_10e12()
        assert result["quantum_contrib"] > 0
        assert result["fractal_contrib"] > 0
        assert result["hybrid_total"] == pytest.approx(
            result["quantum_contrib"] + result["fractal_contrib"], abs=0.001
        )


# === SCALE DECAY VALIDATION TESTS (3) ===


class TestScaleDecay:
    """Full-system tests for scale decay validation."""

    def test_decay_within_slo(self):
        """Test decay validation passes with valid inputs."""
        result = validate_scale_decay(
            baseline_alpha=ALPHA_10E12_TARGET,
            scaled_alpha=3.065,
            max_decay=SCALE_DECAY_MAX
        )
        assert result["valid"] is True

    def test_decay_exceeds_slo(self):
        """Test decay validation fails when decay exceeds SLO."""
        result = validate_scale_decay(
            baseline_alpha=ALPHA_10E12_TARGET,
            scaled_alpha=3.0,  # 0.07 decay > 0.02
            max_decay=SCALE_DECAY_MAX
        )
        assert result["valid"] is False

    def test_decay_percentage_correct(self):
        """Test decay percentage is computed correctly."""
        result = validate_scale_decay(
            baseline_alpha=3.07,
            scaled_alpha=3.05,
            max_decay=0.05
        )
        expected_decay = 0.02
        expected_pct = (0.02 / 3.07) * 100
        assert result["decay"] == pytest.approx(expected_decay, abs=0.001)
        assert result["decay_pct"] == pytest.approx(expected_pct, abs=0.1)


# === RELEASE GATE 3.1 TESTS (3) ===


class TestReleaseGate31:
    """Full-system tests for release gate 3.1."""

    def test_release_gate_pass(self):
        """Test release gate passes with valid benchmark."""
        result = check_release_gate_3_1()
        assert result["gate_pass"] is True

    def test_release_version_unlocked(self):
        """Test version 3.1 is unlocked when gate passes."""
        result = check_release_gate_3_1()
        if result["gate_pass"]:
            assert result["version"] == "3.1"
        else:
            assert result["version"] is None

    def test_release_no_blockers(self):
        """Test no blockers when gate passes."""
        result = check_release_gate_3_1()
        if result["gate_pass"]:
            assert len(result["blockers"]) == 0


# === FRACTAL RECURSION TESTS (5) ===


class TestFractalRecursion:
    """Full-system tests for recursive fractal ceiling breach."""

    def test_recursion_depth_1(self):
        """Test single depth recursion provides base uplift."""
        result = recursive_fractal(10**9, 2.99, depth=1)
        assert result["total_uplift"] == pytest.approx(FRACTAL_UPLIFT, abs=0.001)

    def test_recursion_depth_3_ceiling_breach(self):
        """Test depth 3 recursion breaches ceiling."""
        result = recursive_fractal(10**9, 2.99, depth=3)
        assert result["ceiling_breached"] is True
        assert result["final_alpha"] > 3.0

    def test_recursion_depth_3_uplift_compound(self):
        """Test depth 3 uplift is compounded correctly."""
        result = recursive_fractal(10**9, 2.99, depth=3)
        # Expected: 0.05 + 0.04 + 0.032 = 0.122
        expected_uplift = FRACTAL_UPLIFT * (1 + 0.8 + 0.64)
        assert result["total_uplift"] == pytest.approx(expected_uplift, abs=0.01)

    def test_recursion_target_3_1_achievable(self):
        """Test target 3.1 is achievable with sufficient depth."""
        result = recursive_fractal(10**9, 2.99, depth=FRACTAL_RECURSION_MAX_DEPTH)
        # At max depth, should exceed 3.1
        assert result["target_3_1_reached"] is True

    def test_recursion_depth_contributions_present(self):
        """Test depth contributions are tracked."""
        result = recursive_fractal(10**9, 2.99, depth=3)
        assert len(result["depth_contributions"]) == 3
        for contrib in result["depth_contributions"]:
            assert "depth" in contrib
            assert "contribution" in contrib
            assert "decay_factor" in contrib


# === RECURSION SWEEP TESTS (4) ===


class TestRecursionSweep:
    """Full-system tests for recursion depth sweep."""

    def test_sweep_all_depths(self):
        """Test sweep covers all depths up to max."""
        result = recursive_fractal_sweep(10**9, 2.99)
        assert len(result["sweep_results"]) == FRACTAL_RECURSION_MAX_DEPTH

    def test_sweep_optimal_depth(self):
        """Test optimal depth is identified."""
        result = recursive_fractal_sweep(10**9, 2.99)
        # Optimal should be max depth (highest alpha)
        assert result["optimal_depth"] == FRACTAL_RECURSION_MAX_DEPTH

    def test_sweep_alpha_increases_with_depth(self):
        """Test alpha increases with each depth level."""
        result = recursive_fractal_sweep(10**9, 2.99)
        alphas = [r["final_alpha"] for r in result["sweep_results"]]
        for i in range(1, len(alphas)):
            assert alphas[i] >= alphas[i-1]

    def test_sweep_target_3_1_achievable(self):
        """Test 3.1 target achievability is detected."""
        result = recursive_fractal_sweep(10**9, 2.99)
        assert result["target_3_1_achievable"] is True


# === RECEIPT TESTS (5) ===


class TestReceipts:
    """Full-system tests for receipt emission."""

    def test_benchmark_receipt_emitted(self, capsys):
        """Test hybrid_10e12_benchmark receipt is emitted."""
        benchmark_10e12()
        captured = capsys.readouterr()
        assert "hybrid_10e12_benchmark" in captured.out

    def test_benchmark_receipt_has_payload_hash(self, capsys):
        """Test benchmark receipt has payload_hash."""
        benchmark_10e12()
        captured = capsys.readouterr()
        assert "payload_hash" in captured.out

    def test_release_gate_receipt_emitted(self, capsys):
        """Test release_gate_3_1 receipt is emitted."""
        check_release_gate_3_1()
        captured = capsys.readouterr()
        assert "release_gate_3_1" in captured.out

    def test_fractal_recursion_receipt_emitted(self, capsys):
        """Test fractal_recursion receipt is emitted."""
        recursive_fractal(10**9, 2.99, depth=3)
        captured = capsys.readouterr()
        assert "fractal_recursion" in captured.out

    def test_recursion_sweep_receipt_emitted(self, capsys):
        """Test fractal_recursion_sweep receipt is emitted."""
        recursive_fractal_sweep(10**9, 2.99)
        captured = capsys.readouterr()
        assert "fractal_recursion_sweep" in captured.out


# === SPEC FILE TESTS (3) ===


class TestSpecFile:
    """Full-system tests for spec file loading."""

    def test_spec_file_exists(self):
        """Test hybrid_10e12_spec.json can be loaded."""
        spec = get_hybrid_10e12_spec()
        assert spec is not None

    def test_spec_has_required_fields(self):
        """Test spec has required configuration fields."""
        spec = get_hybrid_10e12_spec()
        required = [
            "tree_target",
            "alpha_floor",
            "alpha_target",
            "scale_decay_max",
            "dual_hash"
        ]
        for field in required:
            assert field in spec, f"Missing field: {field}"

    def test_spec_values_match_constants(self):
        """Test spec values align with module constants."""
        spec = get_hybrid_10e12_spec()
        assert spec["tree_target"] == TREE_10E12
        assert spec["alpha_floor"] == ALPHA_10E12_FLOOR
        assert spec["alpha_target"] == ALPHA_10E12_TARGET
        assert spec["scale_decay_max"] == SCALE_DECAY_MAX


# === INFO FUNCTION TESTS (2) ===


class TestInfoFunctions:
    """Full-system tests for info functions."""

    def test_benchmark_info_returns_dict(self):
        """Test get_benchmark_info returns valid dict."""
        info = get_benchmark_info()
        assert isinstance(info, dict)
        assert "tree_10e12" in info
        assert "slo" in info

    def test_recursive_fractal_info_returns_dict(self):
        """Test get_recursive_fractal_info returns valid dict."""
        info = get_recursive_fractal_info()
        assert isinstance(info, dict)
        assert "max_depth" in info
        assert "expected_uplifts" in info
