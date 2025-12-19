"""Tests for quantum refinement module."""

import pytest
from src.quantum_refine import (
    load_refine_config,
    create_entangled_pairs,
    refine_correlation,
    mitigate_decoherence,
    apply_error_correction,
    iterative_refinement,
    measure_refined_correlation,
    compare_to_unrefined,
    verify_bell_violation,
    get_refine_status,
)


class TestRefineConfig:
    """Tests for refinement configuration."""

    def test_refine_config_loads(self):
        """Config loads successfully."""
        config = load_refine_config()
        assert config is not None
        assert "correlation_target" in config
        assert "decoherence_mitigation" in config

    def test_refine_target(self):
        """Target is 99%."""
        config = load_refine_config()
        assert config["correlation_target"] == 0.99


class TestRefineCorrelation:
    """Tests for correlation refinement."""

    def test_refine_correlation(self):
        """Correlation improved."""
        result = refine_correlation()
        assert result["correlation_after"] >= result["correlation_before"]

    def test_refine_target_met(self):
        """Target >= 0.99."""
        result = refine_correlation()
        assert result["correlation"] >= 0.99


class TestRefineDecoherence:
    """Tests for decoherence mitigation."""

    def test_refine_decoherence(self):
        """Decoherence mitigated."""
        pairs = create_entangled_pairs(50)
        result = mitigate_decoherence(pairs)
        assert result["pairs_processed"] == 50
        assert "mitigated_count" in result


class TestRefineErrorCorrection:
    """Tests for error correction."""

    def test_refine_error_correction(self):
        """Errors corrected."""
        pairs = create_entangled_pairs(50)
        result = apply_error_correction(pairs)
        assert result["pairs_processed"] == 50
        assert "errors_corrected" in result


class TestRefineIterative:
    """Tests for iterative refinement."""

    def test_refine_iterative(self):
        """Iterative passes work."""
        result = iterative_refinement(iterations=5)
        assert result["iterations_completed"] <= 5
        assert result["correlation_after"] >= result["correlation_before"]


class TestRefineMeasurement:
    """Tests for correlation measurement."""

    def test_refine_measure(self):
        """Measure correlation works."""
        pairs = create_entangled_pairs(50)
        correlation = measure_refined_correlation(pairs)
        assert 0.0 <= correlation <= 1.0


class TestRefineComparison:
    """Tests for before/after comparison."""

    def test_refine_compare(self):
        """Refined > unrefined."""
        result = compare_to_unrefined()
        assert result["refinement_effective"] is True
        assert result["refined_correlation"] >= result["unrefined_correlation"]


class TestRefineBell:
    """Tests for Bell violation."""

    def test_refine_bell_violation(self):
        """Bell violation detected."""
        pairs = create_entangled_pairs(100)
        result = verify_bell_violation(pairs)
        assert result["bell_violated"] is True
        assert result["s_parameter"] > result["classical_limit"]


class TestRefineReceipts:
    """Tests for receipt emission."""

    def test_refine_receipt(self, capsys):
        """Receipt emitted."""
        load_refine_config()
        captured = capsys.readouterr()
        assert "quantum_refine_config_receipt" in captured.out


class TestRefineStatus:
    """Tests for status queries."""

    def test_refine_status(self):
        """Status query works."""
        status = get_refine_status()
        assert "correlation_target" in status
        assert "decoherence_mitigation" in status
        assert "error_correction" in status
        assert "bell_limit_classical" in status
        assert "bell_limit_quantum" in status


class TestRefinePairs:
    """Tests for entangled pair creation."""

    def test_refine_pairs_create(self):
        """Pairs created."""
        pairs = create_entangled_pairs(100)
        assert len(pairs) == 100

    def test_refine_pairs_properties(self):
        """Pairs have expected properties."""
        pairs = create_entangled_pairs(10)
        for pair in pairs:
            assert hasattr(pair, "correlation")
            assert hasattr(pair, "fidelity")
            assert hasattr(pair, "state")
            assert 0.0 <= pair.correlation <= 1.0
