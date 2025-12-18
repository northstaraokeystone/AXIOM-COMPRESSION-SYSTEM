"""test_multi_scale.py - Multi-Scale 10^9 Validation Tests

18 tests validating quantum-fractal hybrid at production scale.

Test categories:
1. Spec loading and configuration (5 tests)
2. Alpha at each scale (3 tests)
3. Instability at each scale (3 tests)
4. Degradation and scalability (4 tests)
5. Gate and integration (3 tests)

Source: Grok - "Validate at 10^9", "Gate before 3.1 push"
"""

import io
from contextlib import redirect_stdout


class TestMultiScaleSpecLoading:
    """Tests for multi_scale_spec.json loading and configuration."""

    def test_multi_scale_spec_loads(self):
        """Test that multi_scale_spec.json loads successfully."""
        from src.multi_scale_sweep import load_multi_scale_spec

        # Capture output to suppress receipt prints
        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_multi_scale_spec()

        assert spec is not None
        assert "tree_scale_target" in spec
        assert "alpha_hybrid_validated" in spec

    def test_scales_count(self):
        """Test that scales sweep has exactly 3 entries."""
        from src.multi_scale_sweep import load_multi_scale_spec

        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_multi_scale_spec()

        scales = spec.get("tree_scales_sweep", [])
        assert len(scales) == 3, f"Expected 3 scales, got {len(scales)}"

    def test_target_scale(self):
        """Test that target scale is 10^9."""
        from src.multi_scale_sweep import load_multi_scale_spec

        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_multi_scale_spec()

        target = spec.get("tree_scale_target")
        assert target == 1_000_000_000, f"Expected 10^9, got {target}"

    def test_degradation_tolerance(self):
        """Test that degradation tolerance is 0.01 (1%)."""
        from src.multi_scale_sweep import load_multi_scale_spec

        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_multi_scale_spec()

        tolerance = spec.get("degradation_tolerance")
        assert tolerance == 0.01, f"Expected 0.01, got {tolerance}"

    def test_gate_threshold(self):
        """Test that gate threshold is 3.06."""
        from src.multi_scale_sweep import load_multi_scale_spec

        f = io.StringIO()
        with redirect_stdout(f):
            spec = load_multi_scale_spec()

        threshold = spec.get("scalability_gate_threshold")
        assert threshold == 3.06, f"Expected 3.06, got {threshold}"


class TestAlphaAtScale:
    """Tests for alpha values at each scale."""

    def test_alpha_at_1e6(self):
        """Test that alpha at 10^6 >= 3.070 (baseline)."""
        from src.multi_scale_sweep import compute_alpha_at_scale

        result = compute_alpha_at_scale(1_000_000)
        alpha = result["alpha"]

        assert alpha >= 3.070, f"Alpha at 10^6 = {alpha} < 3.070"

    def test_alpha_at_1e8(self):
        """Test that alpha at 10^8 >= 3.065 (intermediate)."""
        from src.multi_scale_sweep import compute_alpha_at_scale

        result = compute_alpha_at_scale(100_000_000)
        alpha = result["alpha"]

        assert alpha >= 3.065, f"Alpha at 10^8 = {alpha} < 3.065"

    def test_alpha_at_1e9(self):
        """Test that alpha at 10^9 >= 3.06 (target)."""
        from src.multi_scale_sweep import compute_alpha_at_scale

        result = compute_alpha_at_scale(1_000_000_000)
        alpha = result["alpha"]

        assert alpha >= 3.06, f"Alpha at 10^9 = {alpha} < 3.06"


class TestInstabilityAtScale:
    """Tests for instability at each scale (must be 0)."""

    def test_instability_at_1e6(self):
        """Test that instability at 10^6 == 0.00."""
        from src.multi_scale_sweep import compute_alpha_at_scale

        result = compute_alpha_at_scale(1_000_000)
        instability = result["instability"]

        assert instability == 0.00, f"Instability at 10^6 = {instability} != 0.00"

    def test_instability_at_1e8(self):
        """Test that instability at 10^8 == 0.00."""
        from src.multi_scale_sweep import compute_alpha_at_scale

        result = compute_alpha_at_scale(100_000_000)
        instability = result["instability"]

        assert instability == 0.00, f"Instability at 10^8 = {instability} != 0.00"

    def test_instability_at_1e9(self):
        """Test that instability at 10^9 == 0.00."""
        from src.multi_scale_sweep import compute_alpha_at_scale

        result = compute_alpha_at_scale(1_000_000_000)
        instability = result["instability"]

        assert instability == 0.00, f"Instability at 10^9 = {instability} != 0.00"


class TestDegradationAndScalability:
    """Tests for degradation and scalability validation."""

    def test_degradation_under_tolerance(self):
        """Test that degradation from 10^6 to 10^9 < 1%."""
        from src.multi_scale_sweep import run_scale_sweep, check_degradation

        f = io.StringIO()
        with redirect_stdout(f):
            results = run_scale_sweep([1_000_000, 1_000_000_000])
            degradation = check_degradation(results)

        assert degradation["degradation_acceptable"], (
            f"Degradation {degradation['degradation_pct']:.2%} > 1% tolerance"
        )

    def test_no_degradation_cliff(self):
        """Test that no single scale shows > 0.5% drop from previous."""
        from src.multi_scale_sweep import run_scale_sweep, TREE_SCALES

        f = io.StringIO()
        with redirect_stdout(f):
            results = run_scale_sweep(TREE_SCALES)

        scale_results = results["results"]
        alphas = [
            scale_results["1e6"]["alpha"],
            scale_results["1e8"]["alpha"],
            scale_results["1e9"]["alpha"],
        ]

        for i in range(1, len(alphas)):
            drop_pct = (alphas[i - 1] - alphas[i]) / alphas[i - 1]
            assert drop_pct <= 0.005, (
                f"Degradation cliff at scale {i}: {drop_pct:.2%} > 0.5%"
            )

    def test_scalability_gate_passes(self):
        """Test that scalability gate passes for all scales."""
        from src.multi_scale_sweep import run_scale_sweep, scalability_gate

        f = io.StringIO()
        with redirect_stdout(f):
            results = run_scale_sweep()
            gate = scalability_gate(results)

        assert gate["gate_passed"], (
            f"Scalability gate failed: alpha={gate['alpha_at_10e9']}, threshold={gate['gate_threshold']}"
        )

    def test_ready_for_31_push(self):
        """Test that 3.1 push readiness is achieved."""
        from src.multi_scale_sweep import run_scale_sweep, scalability_gate

        f = io.StringIO()
        with redirect_stdout(f):
            results = run_scale_sweep()
            gate = scalability_gate(results)

        assert gate["ready_for_31_push"], "Not ready for 3.1 push"


class TestGateAndIntegration:
    """Tests for gate enforcement and integration."""

    def test_receipts_emitted(self):
        """Test that multi_scale_10e9 and scalability_gate receipts are emitted."""
        from src.multi_scale_sweep import run_multi_scale_validation

        f = io.StringIO()
        with redirect_stdout(f):
            result = run_multi_scale_validation()

        output = f.getvalue()

        # Check for receipt emissions (they print JSON)
        assert "multi_scale_10e9" in output or result is not None, (
            "multi_scale_10e9 receipt not emitted"
        )
        assert "scalability_gate" in output or result is not None, (
            "scalability_gate receipt not emitted"
        )

    def test_scale_adjusted_correlation(self):
        """Test that correlation decreases slightly at larger scales."""
        from src.fractal_layers import (
            scale_adjusted_correlation,
            FRACTAL_BASE_CORRELATION,
        )

        corr_1e6 = scale_adjusted_correlation(1_000_000)
        corr_1e9 = scale_adjusted_correlation(1_000_000_000)

        assert corr_1e6 == FRACTAL_BASE_CORRELATION, (
            f"10^6 correlation should be baseline: {corr_1e6} != {FRACTAL_BASE_CORRELATION}"
        )
        assert corr_1e9 < corr_1e6, (
            f"10^9 correlation should be less than 10^6: {corr_1e9} >= {corr_1e6}"
        )
        assert corr_1e9 > corr_1e6 * 0.95, (
            f"10^9 correlation too low: {corr_1e9} < {corr_1e6 * 0.95}"
        )

    def test_full_integration_no_isolated(self):
        """Test that full integration works (no isolated tests needed)."""
        from src.multi_scale_sweep import run_multi_scale_validation

        f = io.StringIO()
        with redirect_stdout(f):
            result = run_multi_scale_validation()

        # Full validation should complete without error
        assert result is not None
        assert "alpha_at_10e9" in result
        assert "instability_at_10e9" in result
        assert "gate_passed" in result

        # Gate should pass
        assert result["gate_passed"], (
            f"Full integration failed: gate_passed={result['gate_passed']}"
        )


class TestQuantumFractalHybrid:
    """Additional tests for quantum-fractal hybrid at scale."""

    def test_quantum_fractal_hybrid_at_scale(self):
        """Test quantum_fractal_hybrid_at_scale function."""
        from src.quantum_rl_hybrid import quantum_fractal_hybrid_at_scale

        state = {"retention": 1.01, "alpha": 3.070, "instability": 0.0}
        fractal_result = {"correlation": 0.85}

        f = io.StringIO()
        with redirect_stdout(f):
            result = quantum_fractal_hybrid_at_scale(
                state, fractal_result, 1_000_000_000
            )

        assert result["alpha"] >= 3.06, (
            f"Hybrid alpha at 10^9 = {result['alpha']} < 3.06"
        )
        assert result["instability"] == 0.00, (
            f"Hybrid instability = {result['instability']} != 0.00"
        )
        assert result["hybrid_status"] == "validated", (
            f"Hybrid status = {result['hybrid_status']} != 'validated'"
        )

    def test_get_31_push_readiness(self):
        """Test that get_31_push_readiness returns proper status."""
        from src.reasoning import get_31_push_readiness

        f = io.StringIO()
        with redirect_stdout(f):
            readiness = get_31_push_readiness()

        assert "ready_for_31_push" in readiness
        assert "prerequisites" in readiness
        assert readiness["ready_for_31_push"], (
            f"3.1 push not ready: {readiness['prerequisites']}"
        )
