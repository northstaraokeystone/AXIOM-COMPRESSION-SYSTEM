"""Tests for quantum consensus module."""

import pytest


class TestQuantumConsensus:
    """Test quantum consensus functionality."""

    def test_init_quantum_consensus(self):
        """Test quantum consensus initialization."""
        from src.quantum_entangled_consensus import init_quantum_consensus

        qc = init_quantum_consensus({})

        assert qc is not None

    def test_establish_entanglement(self):
        """Test entanglement establishment."""
        from src.quantum_entangled_consensus import (
            init_quantum_consensus,
            establish_entanglement,
        )

        qc = init_quantum_consensus({})
        result = establish_entanglement(qc, "node_001", "node_002")

        assert result is not None
        assert "pair_id" in result
        assert "correlation" in result
        assert result["correlation"] >= 0.9999

    def test_achieve_consensus(self):
        """Test consensus achievement."""
        from src.quantum_entangled_consensus import (
            init_quantum_consensus,
            establish_entanglement,
            achieve_quantum_consensus,
        )

        qc = init_quantum_consensus({})

        # Establish some entanglement pairs
        for i in range(5):
            for j in range(i + 1, 5):
                establish_entanglement(qc, f"node_{i:03d}", f"node_{j:03d}")

        proposal = {"proposal_id": "test", "action": "test"}
        result = achieve_quantum_consensus(qc, proposal)

        assert result is not None
        assert "consensus_achieved" in result


class TestQuantumConsensusConstants:
    """Test quantum consensus constants."""

    def test_correlation_target(self):
        """Test correlation target is 0.9999."""
        from src.quantum_entangled_consensus import CORRELATION_TARGET

        assert CORRELATION_TARGET == 0.9999

    def test_decoherence_threshold(self):
        """Test decoherence threshold is 0.001."""
        from src.quantum_entangled_consensus import DECOHERENCE_THRESHOLD

        assert DECOHERENCE_THRESHOLD == 0.001


class TestByzantineDetection:
    """Test Byzantine detection via decoherence."""

    def test_init_byzantine_detector(self):
        """Test Byzantine detector initialization."""
        from src.quantum_entangled_consensus import init_quantum_consensus
        from src.quantum_decoherence_byzantine import init_byzantine_detector

        qc = init_quantum_consensus({})
        bd = init_byzantine_detector(qc)

        assert bd is not None

    def test_byzantine_types(self):
        """Test Byzantine behavior types."""
        from src.quantum_decoherence_byzantine import BYZANTINE_TYPES

        assert "crash" in BYZANTINE_TYPES
        assert "omission" in BYZANTINE_TYPES
        assert "commission" in BYZANTINE_TYPES
        assert "arbitrary" in BYZANTINE_TYPES
