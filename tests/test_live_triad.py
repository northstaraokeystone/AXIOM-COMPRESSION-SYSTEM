"""Tests for D19.1 live triad ingest module.

Tests for live AgentProof + NEURON receipt ingest.
Reality is the only valid scenario.
"""

import pytest


class TestLiveTriadIngestInit:
    """Test live triad ingest initialization."""

    def test_init_live_ingest(self):
        """Test live ingest initializes correctly."""
        from src.swarm.live_triad_ingest import init_live_ingest

        ingest = init_live_ingest({})

        assert ingest is not None
        assert ingest.source == "live_triad"
        assert "agentproof_ledger" in ingest.sources
        assert "neuron_ledger" in ingest.sources

    def test_entropy_source_is_live_triad(self):
        """Test entropy source is live_triad not synthetic."""
        from src.swarm.live_triad_ingest import ENTROPY_SOURCE

        assert ENTROPY_SOURCE == "live_triad"

    def test_synthetic_disabled(self):
        """Test synthetic scenarios are disabled."""
        from src.swarm.live_triad_ingest import SYNTHETIC_SCENARIOS_ENABLED

        assert SYNTHETIC_SCENARIOS_ENABLED is False


class TestLiveTriadConnection:
    """Test live source connections."""

    def test_connect_agentproof(self):
        """Test AgentProof ledger connection."""
        from src.swarm.live_triad_ingest import init_live_ingest, connect_agentproof

        ingest = init_live_ingest({})
        result = connect_agentproof(ingest)

        assert result is True
        assert ingest.agentproof_connected is True

    def test_connect_neuron(self):
        """Test NEURON ledger connection."""
        from src.swarm.live_triad_ingest import init_live_ingest, connect_neuron

        ingest = init_live_ingest({})
        result = connect_neuron(ingest)

        assert result is True
        assert ingest.neuron_connected is True

    def test_both_sources_connect(self):
        """Test both sources connect successfully."""
        from src.swarm.live_triad_ingest import (
            init_live_ingest,
            connect_agentproof,
            connect_neuron,
        )

        ingest = init_live_ingest({})
        agentproof_ok = connect_agentproof(ingest)
        neuron_ok = connect_neuron(ingest)

        assert agentproof_ok and neuron_ok
        assert ingest.agentproof_connected
        assert ingest.neuron_connected


class TestLiveTriadIngest:
    """Test live receipt ingestion."""

    def test_ingest_receipt(self):
        """Test single receipt ingestion."""
        from src.swarm.live_triad_ingest import init_live_ingest, ingest_receipt

        ingest = init_live_ingest({})
        receipt = ingest_receipt(ingest, "agentproof_ledger")

        assert receipt is not None
        assert "receipt_type" in receipt
        assert ingest.total_ingested == 1

    def test_batch_ingest(self):
        """Test batch receipt ingestion."""
        from src.swarm.live_triad_ingest import init_live_ingest, batch_ingest

        ingest = init_live_ingest({})
        receipts = batch_ingest(ingest, 50)

        assert len(receipts) == 50
        assert ingest.total_ingested == 50

    def test_buffer_maintains_size(self):
        """Test buffer respects size limit."""
        from src.swarm.live_triad_ingest import init_live_ingest, batch_ingest

        ingest = init_live_ingest({"buffer_size": 100})
        batch_ingest(ingest, 150)

        assert len(ingest.buffer) <= 100


class TestLiveEntropyCalculation:
    """Test live entropy calculation."""

    def test_calculate_live_entropy(self):
        """Test Shannon entropy calculation from live stream."""
        from src.swarm.live_triad_ingest import (
            init_live_ingest,
            batch_ingest,
            calculate_live_entropy,
        )

        ingest = init_live_ingest({})
        batch_ingest(ingest, 50)
        entropy = calculate_live_entropy(ingest)

        assert entropy >= 0
        assert isinstance(entropy, float)

    def test_entropy_from_empty_buffer(self):
        """Test entropy from empty buffer is zero."""
        from src.swarm.live_triad_ingest import init_live_ingest, calculate_live_entropy

        ingest = init_live_ingest({})
        entropy = calculate_live_entropy(ingest)

        assert entropy == 0.0


class TestAlphaTracking:
    """Test alpha value tracking from NEURON."""

    def test_get_current_alpha(self):
        """Test getting current alpha value."""
        from src.swarm.live_triad_ingest import init_live_ingest, get_current_alpha

        ingest = init_live_ingest({})
        alpha = get_current_alpha(ingest)

        assert alpha == 1.0  # Default value

    def test_set_alpha(self):
        """Test setting alpha value."""
        from src.swarm.live_triad_ingest import init_live_ingest, set_alpha, get_current_alpha

        ingest = init_live_ingest({})
        set_alpha(ingest, 1.25)
        alpha = get_current_alpha(ingest)

        assert alpha == 1.25


class TestLiveIngestReceipt:
    """Test live ingest receipt emission."""

    def test_emit_live_ingest_receipt(self):
        """Test live triad ingest receipt is emitted."""
        from src.swarm.live_triad_ingest import (
            init_live_ingest,
            connect_agentproof,
            connect_neuron,
            batch_ingest,
            emit_live_ingest_receipt,
        )

        ingest = init_live_ingest({})
        connect_agentproof(ingest)
        connect_neuron(ingest)
        batch_ingest(ingest, 50)
        receipt = emit_live_ingest_receipt(ingest)

        assert receipt is not None
        assert receipt["receipt_type"] == "live_triad_ingest_receipt"
        assert receipt["synthetic_enabled"] is False
        assert "live_entropy" in receipt


class TestIngestStatus:
    """Test ingest status reporting."""

    def test_get_ingest_status(self):
        """Test ingest status returns correct info."""
        from src.swarm.live_triad_ingest import get_ingest_status

        status = get_ingest_status()

        assert status["entropy_source"] == "live_triad"
        assert status["synthetic_enabled"] is False
        assert status["paradigm"] == "reality_only"
        assert "agentproof_ledger" in status["live_sources"]
        assert "neuron_ledger" in status["live_sources"]
