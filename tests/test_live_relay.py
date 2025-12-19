"""Tests for live relay hardware-in-loop module."""

import pytest
from src.live_relay_hil import (
    load_hil_config,
    connect_starlink_analog,
    disconnect_starlink_analog,
    run_hil_test,
    validate_relay_chain,
    mars_hil_proof,
    stress_test_hil,
    get_hil_status,
    send_packet,
    receive_packet,
    measure_latency,
    HILConnectionError,
    HILTimeoutError,
    HILPacketLossError,
)


class TestHILConfig:
    """Tests for HIL configuration."""

    def test_hil_config_loads(self):
        """Config loads successfully."""
        config = load_hil_config()
        assert config is not None
        assert "enabled" in config
        assert "mode" in config
        assert "starlink_analog_config" in config

    def test_hil_config_enabled(self):
        """Config has enabled flag."""
        config = load_hil_config()
        assert config["enabled"] is True

    def test_hil_config_mode(self):
        """Config has hardware_in_loop mode."""
        config = load_hil_config()
        assert config["mode"] == "hardware_in_loop"

    def test_hil_config_priority(self):
        """Config has correct priority."""
        config = load_hil_config()
        assert config["priority"] == "live_verifiability_before_recursion"


class TestHILConnection:
    """Tests for HIL connection."""

    def test_hil_connect(self):
        """Connection works."""
        result = connect_starlink_analog()
        assert result["connected"] is True
        assert "interface" in result

    def test_hil_disconnect(self):
        """Disconnect is clean."""
        result = disconnect_starlink_analog()
        assert result["disconnected"] is True


class TestHILPackets:
    """Tests for packet send/receive."""

    def test_hil_send_receive(self):
        """Packet exchange works."""
        conn = connect_starlink_analog()
        interface = conn["interface"]

        # Send packet
        send_result = send_packet(interface, b"test_data")
        assert send_result["success"] is True
        assert send_result["direction"] == "send"

        # Receive packet
        recv_result = receive_packet(interface)
        assert recv_result["direction"] == "receive"

        disconnect_starlink_analog(interface)


class TestHILLatency:
    """Tests for latency measurement."""

    def test_hil_latency(self):
        """Latency measured."""
        conn = connect_starlink_analog()
        interface = conn["interface"]

        result = measure_latency(interface, iterations=5)

        assert result["iterations"] == 5
        assert "min_ms" in result
        assert "max_ms" in result
        assert "avg_ms" in result

        disconnect_starlink_analog(interface)


class TestHILRelayChain:
    """Tests for relay chain validation."""

    def test_hil_relay_chain(self):
        """Multi-node relay works."""
        result = validate_relay_chain(nodes=3)
        assert result["validated"] is True
        assert result["nodes"] == 3
        assert "chain_results" in result


class TestHILTest:
    """Tests for full HIL test."""

    def test_hil_test_complete(self):
        """Full test passes."""
        result = run_hil_test(duration_s=2)
        assert result["test_passed"] is True
        assert "packets_sent" in result
        assert "packets_received" in result
        assert "loss_rate" in result


class TestMarsHIL:
    """Tests for Mars HIL proof."""

    def test_mars_hil_proof(self):
        """Mars proof works."""
        result = mars_hil_proof(duration_hours=0.001)  # Short test
        assert "proof_passed" in result
        assert "opposition_latency_min" in result
        assert "conjunction_latency_min" in result


class TestHILStress:
    """Tests for stress testing."""

    def test_hil_stress(self):
        """Stress test passes."""
        result = stress_test_hil(iterations=10)
        assert "stress_passed" in result
        assert result["iterations"] == 10
        assert "success_rate" in result


class TestHILStatus:
    """Tests for status queries."""

    def test_hil_status(self):
        """Status query works."""
        status = get_hil_status()
        assert "hil_enabled" in status
        assert "mode" in status
        assert "starlink_analog" in status
        assert "mars_hil" in status


class TestHILReceipts:
    """Tests for receipt emission."""

    def test_hil_receipt(self, capsys):
        """Receipt emitted."""
        load_hil_config()
        captured = capsys.readouterr()
        assert "live_relay_config_receipt" in captured.out


class TestHILStopRules:
    """Tests for StopRules."""

    def test_hil_stoprule_connection(self):
        """StopRule on connection failure."""
        from src.live_relay_hil import stoprule_hil_connection_failed

        with pytest.raises(HILConnectionError):
            stoprule_hil_connection_failed("test error")

    def test_hil_stoprule_timeout(self):
        """StopRule on timeout."""
        from src.live_relay_hil import stoprule_hil_timeout

        with pytest.raises(HILTimeoutError):
            stoprule_hil_timeout(10000)
