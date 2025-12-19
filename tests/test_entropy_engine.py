"""Tests for swarm entropy engine module."""

import pytest


class TestEntropyEngine:
    """Test entropy engine functionality."""

    def test_init_entropy_engine(self):
        """Test entropy engine initialization."""
        from src.swarm.entropy_engine import init_entropy_engine

        engine = init_entropy_engine({})

        assert engine is not None
        assert len(engine.nodes) == 100

    def test_measure_local_entropy(self):
        """Test local entropy measurement."""
        from src.swarm.entropy_engine import measure_local_entropy

        receipts = [
            {"receipt_type": "type_a"},
            {"receipt_type": "type_a"},
            {"receipt_type": "type_b"},
        ]

        entropy = measure_local_entropy("node_001", receipts)

        assert entropy >= 0

    def test_measure_swarm_coherence(self):
        """Test swarm coherence measurement."""
        from src.swarm.entropy_engine import init_entropy_engine, measure_swarm_coherence

        engine = init_entropy_engine({})
        coherence = measure_swarm_coherence(engine)

        assert 0 <= coherence <= 1

    def test_simulate_coordination(self):
        """Test coordination simulation."""
        from src.swarm.entropy_engine import init_entropy_engine, simulate_coordination

        engine = init_entropy_engine({})
        result = simulate_coordination(engine, "consensus")

        assert result is not None
        assert "scenario" in result
        assert result["scenario"] == "consensus"


class TestEntropyConstants:
    """Test entropy engine constants."""

    def test_node_count(self):
        """Test node count is 100."""
        from src.swarm.entropy_engine import NODE_COUNT

        assert NODE_COUNT == 100

    def test_convergence_target(self):
        """Test convergence target is 0.95."""
        from src.swarm.entropy_engine import CONVERGENCE_TARGET

        assert CONVERGENCE_TARGET == 0.95
