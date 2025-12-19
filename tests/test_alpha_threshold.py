"""Tests for D19.1 alpha threshold law discovery.

Tests for law discovery trigger when α > 1.20.
Laws are not discovered—they are enforced by the receipt chain itself.
"""

import pytest
import time


class TestAlphaThresholdConstants:
    """Test alpha threshold constants."""

    def test_alpha_law_threshold(self):
        """Test alpha law threshold is 1.20."""
        from src.witness.alpha_threshold import ALPHA_LAW_THRESHOLD

        assert ALPHA_LAW_THRESHOLD == 1.20

    def test_alpha_source(self):
        """Test alpha source is neuron_ledger."""
        from src.witness.alpha_threshold import ALPHA_SOURCE

        assert ALPHA_SOURCE == "neuron_ledger"

    def test_law_discovery_cooldown(self):
        """Test law discovery cooldown is 60s."""
        from src.witness.alpha_threshold import LAW_DISCOVERY_COOLDOWN_S

        assert LAW_DISCOVERY_COOLDOWN_S == 60

    def test_compression_law_target(self):
        """Test compression law target is 0.95."""
        from src.witness.alpha_threshold import COMPRESSION_LAW_TARGET

        assert COMPRESSION_LAW_TARGET == 0.95

    def test_law_enforcement_mode(self):
        """Test law enforcement mode is receipt_chain."""
        from src.witness.alpha_threshold import LAW_ENFORCEMENT_MODE

        assert LAW_ENFORCEMENT_MODE == "receipt_chain"


class TestAlphaThresholdMonitorInit:
    """Test alpha threshold monitor initialization."""

    def test_init_threshold_monitor(self):
        """Test threshold monitor initializes correctly."""
        from src.witness.alpha_threshold import init_threshold_monitor

        monitor = init_threshold_monitor({})

        assert monitor is not None
        assert monitor.threshold == 1.20
        assert monitor.source == "neuron_ledger"
        assert monitor.trigger_count == 0

    def test_init_with_custom_threshold(self):
        """Test monitor initializes with custom threshold."""
        from src.witness.alpha_threshold import init_threshold_monitor

        monitor = init_threshold_monitor({"alpha_law_threshold": 1.50})

        assert monitor.threshold == 1.50


class TestThresholdChecking:
    """Test threshold checking logic."""

    def test_check_below_threshold(self):
        """Test check returns False when α < threshold."""
        from src.witness.alpha_threshold import init_threshold_monitor, check_threshold

        monitor = init_threshold_monitor({})
        result = check_threshold(monitor, 1.15)

        assert result is False

    def test_check_above_threshold(self):
        """Test check returns True when α > threshold."""
        from src.witness.alpha_threshold import init_threshold_monitor, check_threshold

        monitor = init_threshold_monitor({})
        result = check_threshold(monitor, 1.25)

        assert result is True

    def test_check_at_threshold(self):
        """Test check returns False when α == threshold."""
        from src.witness.alpha_threshold import init_threshold_monitor, check_threshold

        monitor = init_threshold_monitor({})
        result = check_threshold(monitor, 1.20)

        assert result is False  # Must be > not >=


class TestAlphaUpdate:
    """Test alpha value updates."""

    def test_update_alpha(self):
        """Test alpha value is updated."""
        from src.witness.alpha_threshold import init_threshold_monitor, update_alpha

        monitor = init_threshold_monitor({})
        update_alpha(monitor, 1.35)

        assert monitor.current_alpha == 1.35
        assert 1.35 in monitor.alpha_history


class TestLawTrigger:
    """Test law discovery triggering."""

    def test_trigger_when_above_threshold(self):
        """Test law discovery triggers when α > threshold."""
        from src.witness.alpha_threshold import (
            init_threshold_monitor,
            update_alpha,
            trigger_law_discovery,
        )

        monitor = init_threshold_monitor({})
        update_alpha(monitor, 1.25)
        result = trigger_law_discovery(monitor)

        assert result["triggered"] is True
        assert "law" in result
        assert result["law"]["law_id"] is not None
        assert monitor.trigger_count == 1

    def test_no_trigger_when_below_threshold(self):
        """Test law discovery does not trigger when α < threshold."""
        from src.witness.alpha_threshold import (
            init_threshold_monitor,
            update_alpha,
            trigger_law_discovery,
        )

        monitor = init_threshold_monitor({})
        update_alpha(monitor, 1.15)
        result = trigger_law_discovery(monitor)

        assert result["triggered"] is False
        assert result["reason"] == "threshold_not_crossed"

    def test_law_contains_alpha_value(self):
        """Test triggered law contains alpha value."""
        from src.witness.alpha_threshold import (
            init_threshold_monitor,
            update_alpha,
            trigger_law_discovery,
        )

        monitor = init_threshold_monitor({})
        update_alpha(monitor, 1.30)
        result = trigger_law_discovery(monitor)

        assert result["law"]["alpha_at_trigger"] == 1.30


class TestCooldown:
    """Test cooldown behavior."""

    def test_is_in_cooldown_false_initially(self):
        """Test cooldown is False initially."""
        from src.witness.alpha_threshold import init_threshold_monitor, is_in_cooldown

        monitor = init_threshold_monitor({})

        assert is_in_cooldown(monitor) is False

    def test_cooldown_prevents_rapid_triggers(self):
        """Test cooldown prevents rapid-fire triggers."""
        from src.witness.alpha_threshold import (
            init_threshold_monitor,
            update_alpha,
            trigger_law_discovery,
        )

        monitor = init_threshold_monitor({"law_discovery_cooldown_s": 60})
        update_alpha(monitor, 1.25)

        # First trigger
        result1 = trigger_law_discovery(monitor)
        assert result1["triggered"] is True

        # Second trigger should be blocked by cooldown
        result2 = trigger_law_discovery(monitor)
        assert result2["triggered"] is False
        assert result2["reason"] == "cooldown_active"


class TestThresholdStatus:
    """Test threshold status reporting."""

    def test_get_threshold_status(self):
        """Test threshold status returns correct info."""
        from src.witness.alpha_threshold import init_threshold_monitor, get_threshold_status

        monitor = init_threshold_monitor({})
        status = get_threshold_status(monitor)

        assert status["threshold"] == 1.20
        assert status["enforcement_mode"] == "receipt_chain"
        assert "trigger_count" in status

    def test_get_threshold_status_without_monitor(self):
        """Test threshold status without monitor returns module info."""
        from src.witness.alpha_threshold import get_threshold_status

        status = get_threshold_status()

        assert status["alpha_law_threshold"] == 1.20
        assert status["enforcement_mode"] == "receipt_chain"
