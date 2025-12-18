"""Test suite for fractal ceiling breach and quantum-fractal hybrid.

30 tests covering:
- Spec loading (5 tests)
- Fractal dimension (5 tests)
- Multi-scale entropy (5 tests)
- Cross-scale correlation (5 tests)
- Hybrid combination (5 tests)
- Receipts (5 tests)
"""

import json

from src.fractal_layers import (
    multi_scale_fractal,
    compute_fractal_dimension,
    cross_scale_correlation,
    get_fractal_hybrid_spec,
    FRACTAL_SCALES,
    FRACTAL_DIM_MIN,
    FRACTAL_DIM_MAX,
    CROSS_SCALE_CORRELATION_MIN,
    CROSS_SCALE_CORRELATION_MAX,
)
from src.quantum_rl_hybrid import (
    quantum_fractal_hybrid,
    QUANTUM_RETENTION_BOOST,
)


# === SPEC LOADING TESTS (5) ===


class TestSpecLoading:
    """Tests for fractal hybrid spec loading."""

    def test_spec_file_exists(self):
        """Test that spec file can be loaded."""
        spec = get_fractal_hybrid_spec()
        assert spec is not None

    def test_spec_has_required_fields(self):
        """Test spec contains all required fields."""
        spec = get_fractal_hybrid_spec()
        required = [
            "fractal_uplift_target",
            "ceiling_break_target",
            "scales",
            "quantum_contribution",
            "hybrid_total",
        ]
        for field in required:
            assert field in spec, f"Missing field: {field}"

    def test_spec_fractal_uplift_value(self):
        """Test fractal uplift target is 0.05."""
        spec = get_fractal_hybrid_spec()
        assert spec["fractal_uplift_target"] == 0.05

    def test_spec_quantum_contribution_value(self):
        """Test quantum contribution is 0.03."""
        spec = get_fractal_hybrid_spec()
        assert spec["quantum_contribution"] == 0.03

    def test_spec_hybrid_total_value(self):
        """Test hybrid total is 0.08."""
        spec = get_fractal_hybrid_spec()
        assert spec["hybrid_total"] == 0.08


# === FRACTAL DIMENSION TESTS (5) ===


class TestFractalDimension:
    """Tests for fractal dimension computation."""

    def test_dimension_at_1e6(self):
        """Test fractal dimension at 10^6 tree size."""
        dim = compute_fractal_dimension(1_000_000)
        assert FRACTAL_DIM_MIN <= dim <= FRACTAL_DIM_MAX

    def test_dimension_at_1e9(self):
        """Test fractal dimension at 10^9 tree size."""
        dim = compute_fractal_dimension(1_000_000_000)
        assert FRACTAL_DIM_MIN <= dim <= FRACTAL_DIM_MAX
        # At 10^9, dimension should be closer to max
        assert dim >= 1.7

    def test_dimension_increases_with_size(self):
        """Test dimension increases with tree size."""
        dim_small = compute_fractal_dimension(1_000_000)
        dim_large = compute_fractal_dimension(1_000_000_000)
        assert dim_large >= dim_small

    def test_dimension_min_bound(self):
        """Test dimension never goes below minimum."""
        dim = compute_fractal_dimension(100)  # Very small
        assert dim >= FRACTAL_DIM_MIN

    def test_dimension_max_bound(self):
        """Test dimension never exceeds maximum."""
        dim = compute_fractal_dimension(10**12)  # Very large
        assert dim <= FRACTAL_DIM_MAX


# === MULTI-SCALE ENTROPY TESTS (5) ===


class TestMultiScaleEntropy:
    """Tests for multi-scale fractal entropy."""

    def test_fractal_alpha_exceeds_3(self):
        """Test that fractal_alpha > 3.0 at 10^6 with base 2.99."""
        result = multi_scale_fractal(1_000_000, 2.99)
        assert result["fractal_alpha"] > 3.0, f"fractal_alpha={result['fractal_alpha']}"

    def test_entropy_at_each_scale(self):
        """Test entropy computed at each scale."""
        result = multi_scale_fractal(1_000_000, 2.99)
        for scale in FRACTAL_SCALES:
            key = f"scale_{scale}"
            assert key in result["scale_entropies"]
            assert result["scale_entropies"][key] >= 0

    def test_uplift_positive(self):
        """Test uplift is positive."""
        result = multi_scale_fractal(1_000_000, 2.99)
        assert result["uplift_achieved"] > 0

    def test_scales_used_correct(self):
        """Test correct scales are used."""
        result = multi_scale_fractal(1_000_000, 2.99)
        assert result["scales_used"] == FRACTAL_SCALES

    def test_ceiling_breached_flag(self):
        """Test ceiling_breached flag is set correctly."""
        result = multi_scale_fractal(1_000_000, 2.99)
        expected = result["fractal_alpha"] > 3.0
        assert result["ceiling_breached"] == expected


# === CROSS-SCALE CORRELATION TESTS (5) ===


class TestCrossScaleCorrelation:
    """Tests for cross-scale correlation computation."""

    def test_correlation_in_range(self):
        """Test correlation is in valid range."""
        corr = cross_scale_correlation(FRACTAL_SCALES)
        assert CROSS_SCALE_CORRELATION_MIN <= corr <= CROSS_SCALE_CORRELATION_MAX

    def test_correlation_with_geometric_scales(self):
        """Test correlation with perfect geometric progression."""
        # [1, 2, 4, 8, 16] is geometric with ratio 2
        corr = cross_scale_correlation([1, 2, 4, 8, 16])
        assert corr > CROSS_SCALE_CORRELATION_MIN

    def test_correlation_with_single_scale(self):
        """Test correlation with single scale returns minimum."""
        corr = cross_scale_correlation([1])
        assert corr == CROSS_SCALE_CORRELATION_MIN

    def test_correlation_with_empty_scales(self):
        """Test correlation with empty scales returns minimum."""
        corr = cross_scale_correlation([])
        assert corr == CROSS_SCALE_CORRELATION_MIN

    def test_correlation_result_in_multi_scale(self):
        """Test correlation is included in multi_scale_fractal result."""
        result = multi_scale_fractal(1_000_000, 2.99)
        assert "cross_scale_corr" in result
        assert (
            CROSS_SCALE_CORRELATION_MIN
            <= result["cross_scale_corr"]
            <= CROSS_SCALE_CORRELATION_MAX
        )


# === HYBRID COMBINATION TESTS (5) ===


class TestHybridCombination:
    """Tests for quantum-fractal hybrid combination."""

    def test_hybrid_alpha(self):
        """Test hybrid alpha >= 3.07."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        state = {"alpha": 2.99}
        result = quantum_fractal_hybrid(state, fractal_result)
        # With base 2.99 + quantum 0.03 + fractal ~0.05 = ~3.07
        assert result["final_alpha"] >= 3.07, f"final_alpha={result['final_alpha']}"

    def test_instability_zero(self):
        """Test instability == 0.00."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        state = {"alpha": 2.99}
        result = quantum_fractal_hybrid(state, fractal_result)
        assert result["instability"] == 0.00

    def test_quantum_contribution(self):
        """Test quantum contribution == 0.03."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        state = {"alpha": 2.99}
        result = quantum_fractal_hybrid(state, fractal_result)
        assert result["quantum_contribution"] == QUANTUM_RETENTION_BOOST
        assert result["quantum_contribution"] == 0.03

    def test_fractal_contribution(self):
        """Test fractal contribution is positive."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        state = {"alpha": 2.99}
        result = quantum_fractal_hybrid(state, fractal_result)
        assert result["fractal_contribution"] > 0

    def test_ceiling_breached_in_hybrid(self):
        """Test ceiling is breached in hybrid mode."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        state = {"alpha": 2.99}
        result = quantum_fractal_hybrid(state, fractal_result)
        assert result["ceiling_breached"] is True


# === RECEIPT TESTS (5) ===


class TestReceipts:
    """Tests for receipt emission."""

    def test_fractal_layer_receipt_emitted(self, capsys):
        """Test fractal_layer receipt is emitted."""
        multi_scale_fractal(1_000_000, 2.99)
        captured = capsys.readouterr()
        assert "fractal_layer" in captured.out

    def test_fractal_layer_receipt_has_payload_hash(self, capsys):
        """Test fractal_layer receipt has payload_hash."""
        multi_scale_fractal(1_000_000, 2.99)
        captured = capsys.readouterr()
        assert "payload_hash" in captured.out

    def test_quantum_fractal_hybrid_receipt_emitted(self, capsys):
        """Test quantum_fractal_hybrid receipt is emitted."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        # Clear the previous output
        capsys.readouterr()
        state = {"alpha": 2.99}
        quantum_fractal_hybrid(state, fractal_result)
        captured = capsys.readouterr()
        assert "quantum_fractal_hybrid" in captured.out

    def test_hybrid_receipt_has_required_fields(self, capsys):
        """Test hybrid receipt has required fields."""
        fractal_result = multi_scale_fractal(1_000_000, 2.99)
        capsys.readouterr()
        state = {"alpha": 2.99}
        quantum_fractal_hybrid(state, fractal_result)
        captured = capsys.readouterr()
        # Parse the JSON receipt
        receipt_str = captured.out.strip().split("\n")[-1]
        receipt = json.loads(receipt_str)
        assert receipt["receipt_type"] == "quantum_fractal_hybrid"
        assert "quantum_contribution" in receipt
        assert "fractal_contribution" in receipt
        assert "final_alpha" in receipt
        assert "instability" in receipt
        assert "payload_hash" in receipt

    def test_spec_load_receipt_emitted(self, capsys):
        """Test fractal_hybrid_spec_load receipt is emitted."""
        get_fractal_hybrid_spec()
        captured = capsys.readouterr()
        assert "fractal_hybrid_spec_load" in captured.out
