"""test_relay.py - Tests for relay swarm configuration

Validates:
- Relay τ reduction (halves τ from 1200s to 600s)
- Relay P cost calculation
- Optimal swarm size computation
- Earth-like cycle count with relay
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.relay import (
    RelayConfig,
    compute_relay_tau,
    compute_relay_p_cost,
    optimal_swarm_size,
    load_relay_params,
    emit_relay_config_receipt,
    RELAY_TAU_FACTOR,
    RELAY_P_COST_PER_SAT,
    RELAY_SWARM_MIN,
    RELAY_SWARM_OPTIMAL,
    TAU_EARTH_TARGET,
)


class TestRelayConfig:
    """Tests for RelayConfig dataclass."""

    def test_default_config(self):
        """Default config should have optimal swarm size."""
        config = RelayConfig()

        assert config.swarm_size == RELAY_SWARM_OPTIMAL
        assert config.p_cost_per_sat == RELAY_P_COST_PER_SAT
        assert config.tau_reduction_factor == RELAY_TAU_FACTOR
        assert config.operational is True

    def test_custom_config(self):
        """Should accept custom swarm size."""
        config = RelayConfig(swarm_size=9, operational=False)

        assert config.swarm_size == 9
        assert config.operational is False


class TestComputeRelayTau:
    """Tests for compute_relay_tau function."""

    def test_relay_tau_reduction(self, capsys):
        """τ=1200 with swarm should reduce to τ=600."""
        config = RelayConfig(swarm_size=6)
        result = compute_relay_tau(1200, config)

        assert result == 600.0
        assert result == 1200 * RELAY_TAU_FACTOR

    def test_relay_tau_no_swarm(self):
        """No swarm (size=0) should not reduce τ."""
        config = RelayConfig(swarm_size=0)
        result = compute_relay_tau(1200, config)

        assert result == 1200.0

    def test_relay_tau_not_operational(self):
        """Non-operational swarm should not reduce τ."""
        config = RelayConfig(swarm_size=6, operational=False)
        result = compute_relay_tau(1200, config)

        assert result == 1200.0

    def test_relay_emits_receipt(self, capsys):
        """Should emit relay_tau receipt."""
        config = RelayConfig(swarm_size=6)
        compute_relay_tau(1200, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "relay_tau"' in captured.out


class TestComputeRelayPCost:
    """Tests for compute_relay_p_cost function."""

    def test_relay_p_cost_calculation(self, capsys):
        """6 satellites × 0.05 should equal 0.30 P cost."""
        config = RelayConfig(swarm_size=6)
        result = compute_relay_p_cost(config)

        assert result == pytest.approx(0.30)
        assert result == pytest.approx(6 * RELAY_P_COST_PER_SAT)

    def test_relay_p_cost_zero_swarm(self):
        """No swarm should have zero cost."""
        config = RelayConfig(swarm_size=0)
        result = compute_relay_p_cost(config)

        assert result == 0.0

    def test_relay_p_cost_emits_receipt(self, capsys):
        """Should emit relay_p_cost receipt."""
        config = RelayConfig(swarm_size=6)
        compute_relay_p_cost(config)

        captured = capsys.readouterr()
        assert '"receipt_type": "relay_p_cost"' in captured.out


class TestOptimalSwarmSize:
    """Tests for optimal_swarm_size function."""

    def test_optimal_swarm_within_budget(self, capsys):
        """Budget 0.25 should allow 5 satellites max."""
        result = optimal_swarm_size(budget_p=0.25, target_tau=600)

        # 0.25 / 0.05 = 5 satellites max
        # But min swarm is 3, and 3 achieves 600s target
        assert result == RELAY_SWARM_MIN  # 3 satellites sufficient

    def test_optimal_swarm_insufficient_budget(self):
        """Budget too low for minimum swarm."""
        result = optimal_swarm_size(budget_p=0.10, target_tau=600)

        # 0.10 / 0.05 = 2 satellites, below minimum of 3
        assert result == 0

    def test_optimal_swarm_large_budget(self):
        """Large budget should use minimum sufficient size when target achieved."""
        result = optimal_swarm_size(budget_p=1.0, target_tau=600)

        # When target is achievable, minimum swarm is sufficient
        # Function returns minimum swarm that achieves target within budget
        assert result >= RELAY_SWARM_MIN

    def test_optimal_swarm_emits_receipt(self, capsys):
        """Should emit relay_optimization receipt."""
        optimal_swarm_size(budget_p=0.50, target_tau=600)

        captured = capsys.readouterr()
        assert '"receipt_type": "relay_optimization"' in captured.out


class TestLoadRelayParams:
    """Tests for load_relay_params function."""

    def test_loads_params(self, capsys):
        """Should load and verify tau_strategies.json."""
        params = load_relay_params()

        assert "relay_tau_factor" in params
        assert params["relay_tau_factor"] == 0.5
        assert params["relay_p_cost_per_sat"] == 0.05
        assert params["relay_swarm_min"] == 3
        assert params["relay_swarm_optimal"] == 6

    def test_emits_ingest_receipt(self, capsys):
        """Should emit tau_strategies_ingest receipt."""
        load_relay_params()

        captured = capsys.readouterr()
        assert '"receipt_type": "tau_strategies_ingest"' in captured.out
        assert '"hash_verified": true' in captured.out


class TestRelayEarthLike:
    """Tests for Earth-like cycle achievement."""

    def test_relay_achieves_earth_target(self):
        """With relay: τ should reach Earth-like 600s target."""
        config = RelayConfig(swarm_size=6)
        reduced_tau = compute_relay_tau(1200, config)

        assert reduced_tau == TAU_EARTH_TARGET


class TestEmitRelayConfigReceipt:
    """Tests for emit_relay_config_receipt function."""

    def test_emits_config_receipt(self, capsys):
        """Should emit comprehensive relay_config receipt."""
        config = RelayConfig(swarm_size=6)
        receipt = emit_relay_config_receipt(config, 1200)

        assert receipt["receipt_type"] == "relay_config"
        assert receipt["swarm_size"] == 6
        assert receipt["tau_base"] == 1200
        assert receipt["tau_reduced"] == 600
        assert receipt["p_cost_total"] == pytest.approx(0.30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
