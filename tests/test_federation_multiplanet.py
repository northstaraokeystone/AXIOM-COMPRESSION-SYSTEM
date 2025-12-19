"""Tests for multi-planet federation module."""

import pytest
from src.federation_multiplanet import (
    load_federation_config,
    init_federation,
    add_planet,
    remove_planet,
    sync_federation,
    run_consensus,
    arbitrate_dispute,
    get_federation_status,
    measure_federation_health,
    simulate_partition,
    simulate_recovery,
    DEFAULT_PLANETS,
)


class TestFederationConfig:
    """Tests for federation configuration."""

    def test_federation_config_loads(self):
        """Config loads successfully."""
        config = load_federation_config()
        assert config is not None
        assert "planets" in config
        assert "consensus_lag_tolerance" in config

    def test_federation_default_planets(self):
        """Default planets configured."""
        assert "mars" in DEFAULT_PLANETS
        assert "venus" in DEFAULT_PLANETS


class TestFederationInit:
    """Tests for federation initialization."""

    def test_federation_init(self):
        """Federation initializes."""
        result = init_federation()
        assert result["initialized"] is True
        assert result["planet_count"] >= 4

    def test_federation_init_custom_planets(self):
        """Custom planets work."""
        result = init_federation(["earth", "mars", "venus"])
        assert result["planet_count"] == 3


class TestFederationPlanetManagement:
    """Tests for planet management."""

    def test_add_planet(self):
        """Planet can be added."""
        init_federation()
        result = add_planet("titan")
        assert result["added"] is True
        assert result["planet"] == "titan"

    def test_remove_planet(self):
        """Planet can be removed."""
        init_federation()
        add_planet("titan")
        result = remove_planet("titan")
        assert result["removed"] is True


class TestFederationSync:
    """Tests for federation sync."""

    def test_federation_sync(self):
        """Sync completes."""
        init_federation()
        result = sync_federation()
        assert result["sync_complete"] is True
        assert result["planets_synced"] > 0


class TestFederationConsensus:
    """Tests for federation consensus."""

    def test_federation_consensus(self):
        """Consensus reached."""
        init_federation()
        result = run_consensus()
        assert result["consensus_reached"] is True
        assert result["approval_rate"] >= 0.67

    def test_federation_quorum(self):
        """Quorum met."""
        init_federation()
        result = run_consensus()
        assert result["approval_rate"] >= result["quorum"]


class TestFederationArbitration:
    """Tests for dispute arbitration."""

    def test_federation_arbitrate(self):
        """Arbitration resolves."""
        init_federation()
        result = arbitrate_dispute()
        assert result["resolved"] is True
        assert "winner" in result


class TestFederationHealth:
    """Tests for health measurement."""

    def test_federation_health(self):
        """Health check works."""
        init_federation()
        health = measure_federation_health()
        assert "healthy" in health
        assert "availability" in health
        assert health["availability"] >= 0.99


class TestFederationPartition:
    """Tests for partition simulation."""

    def test_federation_partition(self):
        """Partition simulates."""
        init_federation()
        result = simulate_partition(["mars"])
        assert result["partitioned"] is True
        assert result["partition_count"] == 1

    def test_federation_recovery(self):
        """Recovery works."""
        init_federation()
        simulate_partition(["mars"])
        result = simulate_recovery(["mars"])
        assert result["recovered"] is True


class TestFederationStatus:
    """Tests for status queries."""

    def test_federation_status(self):
        """Status query works."""
        init_federation()
        status = get_federation_status()
        assert "initialized" in status
        assert "planet_count" in status
        assert "consensus_round" in status


class TestFederationReceipts:
    """Tests for receipt emission."""

    def test_federation_receipt(self, capsys):
        """Receipt emitted."""
        init_federation()
        captured = capsys.readouterr()
        assert "federation_receipt" in captured.out
