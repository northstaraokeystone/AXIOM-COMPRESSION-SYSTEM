"""Tests for witness.py - The Kolmogorov Lens SLO verification.

Purpose: Verify KAN law discovery implementation meets SLOs.
Source: CLAUDEME LAW_1 (No receipt -> not real), LAW_2 (No test -> not shipped)
"""

import json
import sys
import time
from io import StringIO

import pytest
import numpy as np

from src.witness import (
    # Constants
    KAN_ARCHITECTURE,
    MDL_BETA,
    L1_LAMBDA,
    COMPLEXITY_THRESHOLD,
    TENANT_ID,
    # Functions
    bspline_basis,
    classify_spline,
    mdl_loss,
    spline_to_law,
    train,
    witness,
    # Classes
    KAN,
    KANLayer,
    # Stoprules
    stoprule_nan_loss,
    stoprule_divergence,
    StopRule,
)


# === Structural Tests ===

class TestKANForwardShape:
    """Test KAN forward pass produces correct output shape."""

    def test_kan_forward_shape(self):
        """Output shape should be (batch, 1).

        SLO: Structural
        """
        np.random.seed(42)
        kan = KAN()
        x = np.random.randn(10, 1)
        y = kan(x)
        assert y.shape == (10, 1), f"Expected (10, 1), got {y.shape}"

    def test_kan_forward_single_sample(self):
        """Single sample input should work."""
        np.random.seed(42)
        kan = KAN()
        x = np.random.randn(1, 1)
        y = kan(x)
        assert y.shape == (1, 1), f"Expected (1, 1), got {y.shape}"

    def test_kan_forward_large_batch(self):
        """Large batch should work."""
        np.random.seed(42)
        kan = KAN()
        x = np.random.randn(1000, 1)
        y = kan(x)
        assert y.shape == (1000, 1), f"Expected (1000, 1), got {y.shape}"


class TestBSplineBasisShape:
    """Test B-spline basis function output shape."""

    def test_bspline_basis_shape(self):
        """Basis shape should match expected dimensions.

        SLO: Structural
        """
        n_knots = 10
        degree = 3
        n_samples = 50
        knots = np.linspace(0, 1, n_knots)
        x = np.linspace(0.1, 0.9, n_samples)

        basis = bspline_basis(x, knots, degree)

        expected_n_basis = n_knots - degree - 1  # 10 - 3 - 1 = 6
        assert basis.shape == (n_samples, expected_n_basis), \
            f"Expected ({n_samples}, {expected_n_basis}), got {basis.shape}"

    def test_bspline_basis_partition_of_unity(self):
        """Basis functions should sum to approximately 1 (partition of unity) in interior."""
        knots = np.linspace(0, 1, 10)
        # Use more interior points to avoid boundary effects
        x = np.linspace(0.35, 0.65, 30)
        basis = bspline_basis(x, knots, degree=3)

        # Sum across basis functions should be close to 1 in interior
        basis_sum = basis.sum(axis=1)
        assert np.allclose(basis_sum, np.ones_like(basis_sum), atol=0.05), \
            f"Basis sum should be ~1, got {basis_sum.mean():.3f}"


# === Numerical Tests ===

class TestMDLLoss:
    """Test MDL loss computation."""

    def test_mdl_loss_positive(self):
        """Loss should be non-negative.

        SLO: Numerical
        """
        np.random.seed(42)
        kan = KAN()
        x = np.random.randn(10, 1)
        pred = kan(x)
        obs = np.random.randn(10, 1)

        loss = mdl_loss(pred, obs, kan)
        assert loss >= 0, f"Loss should be >= 0, got {loss}"

    def test_mdl_loss_zero_error(self):
        """Loss should be small when prediction equals observation."""
        np.random.seed(42)
        kan = KAN()
        x = np.random.randn(10, 1)
        pred = kan(x)

        # Use prediction as observation (zero MSE)
        loss = mdl_loss(pred, pred.copy(), kan)
        # Loss should just be complexity term
        assert loss < 1.0, f"Loss with zero error should be small, got {loss}"


# === CLAUDEME LAW_1 Tests ===

class TestReceiptEmission:
    """Test receipt emission per CLAUDEME LAW_1."""

    def test_train_emits_receipt(self):
        """Training should emit a receipt with receipt_type='training'.

        SLO: CLAUDEME LAW_1
        """
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        np.random.seed(42)
        kan = KAN()
        x = np.linspace(0.1, 10, 50).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        receipt = train(kan, x, y, epochs=10)

        sys.stdout = old_stdout
        output = captured.getvalue()

        assert "training" in receipt["receipt_type"], \
            f"Receipt type should contain 'training', got {receipt['receipt_type']}"
        assert "tenant_id" in receipt, "Receipt should have tenant_id"
        assert receipt["tenant_id"] == TENANT_ID, f"Tenant should be {TENANT_ID}"

    def test_witness_emits_receipt(self):
        """Witness should emit a receipt with receipt_type='witness'.

        SLO: CLAUDEME LAW_1
        """
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        np.random.seed(42)
        x = np.linspace(0.1, 10, 50).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        receipt = witness("test_galaxy", x, y, "newtonian", epochs=10)

        sys.stdout = old_stdout
        output = captured.getvalue()

        assert "witness" in receipt["receipt_type"], \
            f"Receipt type should contain 'witness', got {receipt['receipt_type']}"
        assert receipt["galaxy_id"] == "test_galaxy"
        assert receipt["physics_regime"] == "newtonian"


# === SLO Floor Tests ===

class TestCompressionSLO:
    """Test compression ratio SLO."""

    def test_compression_slo(self):
        """Compression ratio should be >= 0.84 on 1/sqrt(r) curve.

        SLO: compression >= 0.84
        """
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        np.random.seed(42)
        x = np.linspace(0.1, 10, 100).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        receipt = witness("slo_test", x, y, "newtonian", epochs=100)

        sys.stdout = old_stdout

        assert receipt["compression_ratio"] >= 0.84, \
            f"Compression ratio {receipt['compression_ratio']:.3f} < 0.84 SLO floor"


class TestMSESLO:
    """Test MSE SLO ceiling."""

    def test_mse_slo(self):
        """Final MSE should be <= 10.0 on synthetic data.

        SLO: MSE ceiling
        """
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        np.random.seed(42)
        x = np.linspace(0.1, 10, 100).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        receipt = witness("mse_test", x, y, "newtonian", epochs=100)

        sys.stdout = old_stdout

        assert receipt["final_mse"] <= 10.0, \
            f"MSE {receipt['final_mse']:.2f} > 10.0 SLO ceiling"


# === Performance Tests ===

class TestTrainingTimeSLO:
    """Test training time SLO."""

    def test_training_time_slo(self):
        """Training should complete in <= 60s for 100 epochs.

        SLO: Performance
        """
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        np.random.seed(42)
        x = np.linspace(0.1, 10, 100).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        start_time = time.time()
        receipt = witness("timing_test", x, y, "newtonian", epochs=100)
        elapsed = time.time() - start_time

        sys.stdout = old_stdout

        assert elapsed <= 60, f"Training took {elapsed:.1f}s > 60s SLO"


# === Classification Tests ===

class TestClassifySpline:
    """Test spline classification."""

    def test_classify_spline_sqrt(self):
        """classify_spline should detect sqrt pattern.

        SLO: Classification
        """
        x = np.linspace(0.1, 10, 100)
        y = np.sqrt(x)

        classification = classify_spline(x, y)
        assert classification == "sqrt", f"Expected 'sqrt', got '{classification}'"

    def test_classify_spline_inverse(self):
        """classify_spline should detect inverse (1/x) pattern.

        SLO: Classification
        """
        x = np.linspace(0.1, 10, 100)
        y = 1 / x

        classification = classify_spline(x, y)
        assert classification == "inverse", f"Expected 'inverse', got '{classification}'"

    def test_classify_spline_linear(self):
        """classify_spline should detect linear pattern."""
        x = np.linspace(0.1, 10, 100)
        y = 2.5 * x + 1.0

        classification = classify_spline(x, y)
        assert classification == "linear", f"Expected 'linear', got '{classification}'"

    def test_classify_spline_log(self):
        """classify_spline should detect log pattern."""
        x = np.linspace(0.1, 10, 100)
        y = np.log(x)

        classification = classify_spline(x, y)
        assert classification == "log", f"Expected 'log', got '{classification}'"


# === Safety Tests ===

class TestStopruleNaN:
    """Test NaN stoprule."""

    def test_stoprule_nan(self):
        """StopRule should be raised on NaN injection.

        SLO: Safety
        """
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        with pytest.raises(StopRule) as exc_info:
            stoprule_nan_loss()

        sys.stdout = old_stdout

        assert "NaN" in str(exc_info.value), "StopRule message should mention NaN"


class TestStopruleDivergence:
    """Test gradient divergence stoprule."""

    def test_stoprule_divergence(self):
        """StopRule should be raised on divergence."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        with pytest.raises(StopRule) as exc_info:
            stoprule_divergence(5000.0)

        sys.stdout = old_stdout

        assert "divergence" in str(exc_info.value).lower() or "5000" in str(exc_info.value), \
            "StopRule message should mention divergence or value"


# === Reproducibility Tests ===

class TestDeterministic:
    """Test reproducibility."""

    def test_deterministic(self):
        """Same input with fixed seed should produce same output.

        SLO: Reproducibility
        """
        np.random.seed(42)
        kan1 = KAN()
        x = np.linspace(0.1, 10, 50).reshape(-1, 1)
        y1 = kan1(x)

        np.random.seed(42)
        kan2 = KAN()
        y2 = kan2(x)

        assert np.allclose(y1, y2), "Same seed should produce same output"


# === Constants Tests ===

class TestConstants:
    """Test module constants."""

    def test_kan_architecture(self):
        """KAN_ARCHITECTURE should be [1, 6, 1]."""
        assert KAN_ARCHITECTURE == [1, 6, 1], f"Expected [1, 6, 1], got {KAN_ARCHITECTURE}"

    def test_mdl_beta(self):
        """MDL_BETA should be 0.10."""
        assert MDL_BETA == 0.10, f"Expected 0.10, got {MDL_BETA}"

    def test_l1_lambda(self):
        """L1_LAMBDA should be 0.015."""
        assert L1_LAMBDA == 0.015, f"Expected 0.015, got {L1_LAMBDA}"

    def test_tenant_id(self):
        """TENANT_ID should be 'axiom-witness'."""
        assert TENANT_ID == "axiom-witness", f"Expected 'axiom-witness', got {TENANT_ID}"


# === Integration Tests ===

class TestWitnessIntegration:
    """Integration tests for witness pipeline."""

    def test_witness_receipt_fields(self):
        """Witness receipt should have all required fields."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        np.random.seed(42)
        x = np.linspace(0.1, 10, 50).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        receipt = witness("integration_test", x, y, "newtonian", epochs=10)

        sys.stdout = old_stdout

        required_fields = [
            "receipt_type", "tenant_id", "galaxy_id", "physics_regime",
            "kan_architecture", "epochs_trained", "final_mse",
            "compression_ratio", "discovered_law", "spline_classification",
            "payload_hash"
        ]

        for field in required_fields:
            assert field in receipt, f"Missing required field: {field}"

    def test_witness_kan_architecture_in_receipt(self):
        """Receipt should record correct KAN architecture."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        np.random.seed(42)
        x = np.linspace(0.1, 10, 50).reshape(-1, 1)
        y = 1 / np.sqrt(x)

        receipt = witness("arch_test", x, y, "newtonian", epochs=10)

        sys.stdout = old_stdout

        assert receipt["kan_architecture"] == [1, 6, 1], \
            f"Expected [1, 6, 1], got {receipt['kan_architecture']}"


# === KAN Architecture Tests ===

class TestKANArchitecture:
    """Test KAN model architecture."""

    def test_kan_get_architecture(self):
        """KAN.get_architecture() should return [1, 6, 1]."""
        np.random.seed(42)
        kan = KAN()
        arch = kan.get_architecture()
        assert arch == [1, 6, 1], f"Expected [1, 6, 1], got {arch}"

    def test_kan_layer_count(self):
        """KAN should have 2 layers (input->hidden, hidden->output)."""
        np.random.seed(42)
        kan = KAN()
        assert len(kan.layers) == 2, f"Expected 2 layers, got {len(kan.layers)}"

    def test_kan_input_layer_shape(self):
        """Input layer should be KANLayer(1, 6)."""
        np.random.seed(42)
        kan = KAN()
        input_layer = kan.layers[0]
        assert input_layer.in_features == 1, f"Input features: expected 1, got {input_layer.in_features}"
        assert input_layer.out_features == 6, f"Output features: expected 6, got {input_layer.out_features}"

    def test_kan_output_layer_shape(self):
        """Output layer should be KANLayer(6, 1)."""
        np.random.seed(42)
        kan = KAN()
        output_layer = kan.layers[1]
        assert output_layer.in_features == 6, f"Input features: expected 6, got {output_layer.in_features}"
        assert output_layer.out_features == 1, f"Output features: expected 1, got {output_layer.out_features}"
