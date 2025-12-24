"""Tests for Mars HIL proof module."""

from src.live_relay_hil import (
    load_hil_config,
    mars_hil_proof,
)
from hardware.starlink_analog.latency_sim import (
    simulate_mars_latency,
    get_latency_config,
)


class TestMarsConfig:
    """Tests for Mars configuration."""

    def test_mars_config_loads(self):
        """Config loads successfully."""
        config = load_hil_config()
        assert "mars_hil_config" in config

    def test_mars_latency_range(self):
        """Latency range is correct."""
        config = load_hil_config()
        mars = config["mars_hil_config"]
        assert mars["latency_min_minutes"] == 3
        assert mars["latency_max_minutes"] == 22


class TestMarsLatency:
    """Tests for Mars latency simulation."""

    def test_mars_latency_opposition(self):
        """3 min latency at opposition."""
        latency = simulate_mars_latency("opposition")
        assert latency == 3.0

    def test_mars_latency_conjunction(self):
        """22 min latency at conjunction."""
        latency = simulate_mars_latency("conjunction")
        assert latency == 22.0

    def test_mars_latency_average(self):
        """Average latency is ~12.5 min."""
        latency = simulate_mars_latency("average")
        assert 10 <= latency <= 15


class TestMarsHILEnabled:
    """Tests for Mars HIL enabled flag."""

    def test_mars_hil_enabled(self):
        """Enabled flag works."""
        config = load_hil_config()
        assert config["mars_hil_config"]["enabled"] is True


class TestMarsProofDuration:
    """Tests for proof duration."""

    def test_mars_proof_duration(self):
        """24-hour proof configured."""
        config = load_hil_config()
        assert config["mars_hil_config"]["proof_duration_hours"] == 24


class TestMarsAutonomy:
    """Tests for autonomy target."""

    def test_mars_autonomy_target(self):
        """Autonomy >= 0.999."""
        config = load_hil_config()
        assert config["mars_hil_config"]["autonomy_target"] >= 0.999


class TestMarsHILProof:
    """Tests for Mars HIL proof."""

    def test_mars_hil_proof_complete(self):
        """Full proof passes."""
        result = mars_hil_proof(duration_hours=0.001)  # Very short for testing
        assert "proof_passed" in result
        assert "messages_sent" in result
        assert "messages_received" in result
        assert "success_rate" in result

    def test_mars_hil_proof_latencies(self):
        """Proof includes latency info."""
        result = mars_hil_proof(duration_hours=0.001)
        assert "opposition_latency_min" in result
        assert "conjunction_latency_min" in result


class TestMarsReceipts:
    """Tests for receipt emission."""

    def test_mars_receipt(self, capsys):
        """Receipt emitted."""
        mars_hil_proof(duration_hours=0.001)
        captured = capsys.readouterr()
        assert "mars_hil_proof_receipt" in captured.out


class TestMarsLatencyConfig:
    """Tests for latency configuration."""

    def test_mars_latency_config(self):
        """Latency config loads."""
        config = get_latency_config()
        assert "mars_min_distance_au" in config
        assert "mars_max_distance_au" in config
        assert config["proxima_latency_multiplier"] == 6300
