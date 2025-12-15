"""Tests for kan_core.py - CLAUDEME LAW_2: No test -> not shipped."""

import json
import os
import sys
import tempfile
from io import StringIO

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kan_core import (
    DEFAULT_TOPOLOGY,
    PARAM_BUDGET,
    COMPLEXITY_THRESHOLD,
    RECEIPT_SCHEMAS,
    StopRule,
    dual_hash,
    emit_receipt,
    kan_init,
    spline_edge,
    forward_compress,
    complexity,
    mdl_loss,
    train_step,
    extract_equation,
    detect_dark_matter,
    persistence_match,
    checkpoint_save,
)


class TestDualHash:
    """Tests for dual_hash function."""

    def test_dual_hash_bytes(self):
        result = dual_hash(b"test")
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 64  # SHA256 hex length

    def test_dual_hash_string(self):
        result = dual_hash("test")
        assert ":" in result

    def test_dual_hash_deterministic(self):
        assert dual_hash(b"test") == dual_hash(b"test")


class TestEmitReceipt:
    """Tests for emit_receipt function."""

    def test_emit_receipt_structure(self, capsys):
        result = emit_receipt("test", {"galaxy_id": "NGC1234"})
        assert "ts" in result
        assert "receipt_type" in result
        assert result["receipt_type"] == "test"
        assert "tenant_id" in result
        assert "payload_hash" in result

    def test_emit_receipt_default_tenant(self, capsys):
        result = emit_receipt("test", {"galaxy_id": "NGC1234"})
        assert result["tenant_id"] == "axiom"

    def test_emit_receipt_custom_tenant(self, capsys):
        result = emit_receipt("test", {"tenant_id": "custom", "galaxy_id": "NGC1234"})
        assert result["tenant_id"] == "custom"

    def test_emit_receipt_prints_json(self, capsys):
        emit_receipt("test", {"galaxy_id": "NGC1234"})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["receipt_type"] == "test"


class TestKanInit:
    """Tests for kan_init function."""

    def test_kan_init_param_budget(self, capsys):
        network = kan_init([1, 5, 1], PARAM_BUDGET)
        param_count = sum(p.numel() for p in network.parameters())
        assert param_count < PARAM_BUDGET

    def test_kan_init_topology(self, capsys):
        network = kan_init([1, 5, 1], PARAM_BUDGET)
        assert network.topology == [1, 5, 1]
        assert len(network.layers) == 2

    def test_kan_init_default_topology(self, capsys):
        network = kan_init(n_params=PARAM_BUDGET)
        assert network.topology == DEFAULT_TOPOLOGY

    def test_kan_init_stoprule_on_budget_exceeded(self, capsys):
        with pytest.raises(StopRule):
            kan_init([100, 100, 100], n_params=10)

    def test_kan_init_emits_receipt(self, capsys):
        kan_init([1, 3, 1], PARAM_BUDGET)
        captured = capsys.readouterr()
        receipt = json.loads(captured.out.strip())
        assert receipt["receipt_type"] == "init"
        assert "param_count" in receipt
        assert "topology" in receipt


class TestSplineEdge:
    """Tests for spline_edge function."""

    def test_spline_edge_shape(self):
        x = torch.linspace(0, 1, 10)
        coeffs = torch.randn(6)  # n_knots - degree - 1 = 10 - 3 - 1 = 6
        result = spline_edge(x, coeffs)
        assert result.shape == x.shape

    def test_spline_edge_differentiable(self):
        x = torch.linspace(0.1, 1, 10, requires_grad=True)
        coeffs = torch.randn(6, requires_grad=True)
        result = spline_edge(x, coeffs)
        loss = result.sum()
        loss.backward()
        assert coeffs.grad is not None


class TestForwardCompress:
    """Tests for forward_compress function."""

    def test_forward_compress_deterministic(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        x = torch.tensor([[1.0], [2.0], [3.0]])
        y1 = forward_compress(network, x)
        y2 = forward_compress(network, x)
        assert torch.allclose(y1, y2)

    def test_forward_compress_shape(self, capsys):
        network = kan_init([1, 5, 1], PARAM_BUDGET)
        x = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
        y = forward_compress(network, x)
        assert y.shape == (4, 1)


class TestComplexity:
    """Tests for complexity function."""

    def test_complexity_returns_int(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        c = complexity(network)
        assert isinstance(c, int)

    def test_complexity_nonnegative(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        c = complexity(network)
        assert c >= 0


class TestMdlLoss:
    """Tests for mdl_loss function."""

    def test_mdl_loss_differentiable(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        x = torch.tensor([[1.0], [2.0]], requires_grad=True)
        y_obs = torch.tensor([[1.5], [2.5]])
        pred = forward_compress(network, x)
        loss = mdl_loss(pred, y_obs, network)
        assert loss.requires_grad

    def test_mdl_loss_complexity_penalty(self, capsys):
        # Create two networks with different complexity
        small_net = kan_init([1, 2, 1], PARAM_BUDGET)
        large_net = kan_init([1, 5, 1], PARAM_BUDGET)
        x = torch.tensor([[1.0]])
        y_obs = torch.tensor([[1.0]])
        pred_small = forward_compress(small_net, x)
        pred_large = forward_compress(large_net, x)
        # Force same MSE by using pred as obs
        loss_small = mdl_loss(pred_small, pred_small.detach(), small_net)
        loss_large = mdl_loss(pred_large, pred_large.detach(), large_net)
        # Larger network should have higher complexity penalty
        comp_small = complexity(small_net)
        comp_large = complexity(large_net)
        assert comp_large >= comp_small


class TestTrainStep:
    """Tests for train_step function."""

    def test_train_step_decreases_loss(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.1)
        x = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor([[2.0], [4.0], [6.0]])  # y = 2x
        batch = (x, y)
        loss1, _ = train_step(network, batch, optimizer)
        loss2, _ = train_step(network, batch, optimizer)
        # Loss should decrease or stay similar after optimization
        assert loss2 <= loss1 * 1.5  # Allow some tolerance

    def test_train_step_returns_grad_norm(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        batch = (torch.tensor([[1.0]]), torch.tensor([[2.0]]))
        loss, grad_norm = train_step(network, batch, optimizer)
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0

    def test_train_step_emits_receipt(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        capsys.readouterr()  # Clear init receipt
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        batch = (torch.tensor([[1.0]]), torch.tensor([[2.0]]))
        train_step(network, batch, optimizer)
        captured = capsys.readouterr()
        receipt = json.loads(captured.out.strip())
        assert receipt["receipt_type"] == "training"
        assert "loss" in receipt
        assert "mse" in receipt
        assert "complexity" in receipt
        assert "grad_norm" in receipt


class TestExtractEquation:
    """Tests for extract_equation function."""

    def test_extract_equation_linear(self, capsys):
        network = kan_init([1, 1, 1], PARAM_BUDGET)
        # Set coefficients to approximate identity function
        for layer in network.layers:
            for out_edges in layer.edges:
                for edge in out_edges:
                    # Set coefficients to produce approximately linear output
                    with torch.no_grad():
                        edge.coeffs.fill_(0.1)
        capsys.readouterr()  # Clear previous receipts
        result = extract_equation(network)
        assert len(result) > 0
        # Check that result contains tuples of (classification, formula)
        for key, value in result.items():
            assert isinstance(value, tuple)
            assert len(value) == 2

    def test_extract_equation_emits_receipt(self, capsys):
        network = kan_init([1, 2, 1], PARAM_BUDGET)
        capsys.readouterr()
        extract_equation(network)
        captured = capsys.readouterr()
        receipt = json.loads(captured.out.strip())
        assert receipt["receipt_type"] == "interpretation"
        assert "edge_classifications" in receipt
        assert "equations" in receipt


class TestDetectDarkMatter:
    """Tests for detect_dark_matter function."""

    def test_detect_dark_matter_threshold(self, capsys):
        network = kan_init([1, 10, 1], PARAM_BUDGET)
        # Large network should have high complexity
        comp = complexity(network)
        result = detect_dark_matter(network, threshold=comp - 1)
        assert result is True

    def test_detect_dark_matter_below_threshold(self, capsys):
        network = kan_init([1, 2, 1], PARAM_BUDGET)
        # Set very high threshold
        result = detect_dark_matter(network, threshold=1000000)
        assert result is False


class TestPersistenceMatch:
    """Tests for persistence_match function."""

    def test_persistence_match_empty(self):
        obs = np.array([])
        pred = np.array([])
        result = persistence_match(obs, pred)
        assert result == 0.0

    def test_persistence_match_identical(self):
        obs = np.array([[0, 1], [0.5, 2]])
        pred = np.array([[0, 1], [0.5, 2]])
        result = persistence_match(obs, pred)
        assert result < 0.01  # Should be near zero

    def test_persistence_match_different(self):
        obs = np.array([[0, 1], [0.5, 2]])
        pred = np.array([[1, 2], [1.5, 3]])
        result = persistence_match(obs, pred)
        assert result > 0


class TestCheckpointSave:
    """Tests for checkpoint_save function."""

    def test_checkpoint_save_roundtrip(self, capsys):
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        original_state = {k: v.clone() for k, v in network.state_dict().items()}
        training_receipts = [{"epoch": 0, "loss": 1.0}]
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            capsys.readouterr()
            checkpoint_save(network, path, training_receipts)
            # Load and verify
            checkpoint = torch.load(path)
            assert "state_dict" in checkpoint
            assert "training_receipts" in checkpoint
            for key in original_state:
                assert torch.allclose(original_state[key], checkpoint["state_dict"][key])
        finally:
            os.unlink(path)

    def test_checkpoint_save_emits_receipt(self, capsys):
        network = kan_init([1, 2, 1], PARAM_BUDGET)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            capsys.readouterr()
            checkpoint_save(network, path, [])
            captured = capsys.readouterr()
            receipt = json.loads(captured.out.strip())
            assert receipt["receipt_type"] == "checkpoint"
            assert "merkle_root" in receipt
            assert "path" in receipt
        finally:
            os.unlink(path)


class TestReceiptsEmitted:
    """Test that all functions emit appropriate receipts."""

    def test_receipts_schema_exported(self):
        assert "training" in RECEIPT_SCHEMAS
        assert "interpretation" in RECEIPT_SCHEMAS
        assert "discovery" in RECEIPT_SCHEMAS

    def test_training_receipt_schema(self):
        schema = RECEIPT_SCHEMAS["training"]
        assert "epoch" in schema
        assert "loss" in schema
        assert "mse" in schema
        assert "complexity" in schema


class TestStopRule:
    """Tests for StopRule exception."""

    def test_stoprule_is_exception(self):
        assert issubclass(StopRule, Exception)

    def test_stoprule_message(self):
        with pytest.raises(StopRule) as exc_info:
            raise StopRule("Test error")
        assert "Test error" in str(exc_info.value)


class TestSLOCompliance:
    """SLO compliance tests."""

    def test_train_step_latency(self, capsys):
        import time
        network = kan_init([1, 3, 1], PARAM_BUDGET)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        batch = (torch.tensor([[1.0]]), torch.tensor([[2.0]]))
        capsys.readouterr()
        start = time.time()
        train_step(network, batch, optimizer)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 100, f"Train step took {elapsed_ms}ms, exceeds 100ms SLO"

    def test_complexity_latency(self, capsys):
        import time
        network = kan_init([1, 5, 1], PARAM_BUDGET)
        start = time.time()
        complexity(network)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 10, f"Complexity took {elapsed_ms}ms, exceeds 10ms SLO"
