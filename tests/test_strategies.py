"""test_strategies.py - Tests for τ reduction strategy comparator

Validates:
- Onboard AI: effective α ≥ 1.2 regardless of τ
- Predictive: 30% τ reduction, c = 0.8
- Relay: τ halved, P cost
- Combined: best τ reduction (420s)
- Strategy comparison and ranking
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies import (
    Strategy,
    StrategyConfig,
    StrategyResult,
    apply_strategy,
    compare_strategies,
    recommend_strategy,
    compute_effective_tau,
    compute_effective_alpha,
    compute_c_factor,
    compute_p_cost,
    get_all_strategy_configs,
    ONBOARD_AI_EFF_ALPHA_FLOOR,
    PREDICTIVE_TAU_REDUCTION,
    PREDICTIVE_C_FACTOR,
)
from src.relay import RELAY_TAU_FACTOR, RELAY_SWARM_OPTIMAL


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_baseline_config(self):
        """Baseline should have no modifications."""
        config = StrategyConfig(strategy=Strategy.BASELINE)

        assert config.relay_swarm_size == 0
        assert config.predictive_enabled is False

    def test_relay_config_defaults(self):
        """Relay strategy should default to optimal swarm size."""
        config = StrategyConfig(strategy=Strategy.RELAY_SWARM)

        assert config.relay_swarm_size == RELAY_SWARM_OPTIMAL

    def test_combined_config(self):
        """Combined strategy should enable all features."""
        config = StrategyConfig(strategy=Strategy.COMBINED)

        assert config.relay_swarm_size == RELAY_SWARM_OPTIMAL
        assert config.predictive_enabled is True


class TestOnboardAI:
    """Tests for ONBOARD_AI strategy."""

    def test_onboard_ai_alpha_floor(self, capsys):
        """Effective α should be ≥ 1.2 regardless of τ."""
        config = StrategyConfig(strategy=Strategy.ONBOARD_AI)

        # At Mars max τ=1200s, standard α=1.69 would drop to ~0.59
        # But onboard AI should floor it at 1.2
        result = apply_strategy(1200, 1.69, config)

        assert result.effective_alpha >= ONBOARD_AI_EFF_ALPHA_FLOOR
        assert result.effective_alpha >= 1.2

    def test_onboard_ai_no_tau_change(self):
        """Onboard AI should not change τ."""
        config = StrategyConfig(strategy=Strategy.ONBOARD_AI)
        effective_tau = compute_effective_tau(1200, config)

        assert effective_tau == 1200

    def test_onboard_ai_no_p_cost(self):
        """Onboard AI should have no P cost."""
        config = StrategyConfig(strategy=Strategy.ONBOARD_AI)
        p_cost = compute_p_cost(config)

        assert p_cost == 0.0


class TestPredictive:
    """Tests for PREDICTIVE strategy."""

    def test_predictive_tau_reduction(self):
        """τ should be reduced by 30%."""
        config = StrategyConfig(strategy=Strategy.PREDICTIVE)
        effective_tau = compute_effective_tau(1200, config)

        expected = 1200 * (1.0 - PREDICTIVE_TAU_REDUCTION)  # 840s
        assert effective_tau == expected
        assert effective_tau == 840

    def test_predictive_c_factor(self):
        """c factor should be 0.8."""
        config = StrategyConfig(strategy=Strategy.PREDICTIVE)
        c_factor = compute_c_factor(config)

        assert c_factor == PREDICTIVE_C_FACTOR
        assert c_factor == 0.8

    def test_predictive_no_p_cost(self):
        """Predictive should have no P cost."""
        config = StrategyConfig(strategy=Strategy.PREDICTIVE)
        p_cost = compute_p_cost(config)

        assert p_cost == 0.0


class TestRelaySwarm:
    """Tests for RELAY_SWARM strategy."""

    def test_relay_physical_reduction(self, capsys):
        """τ should be halved to 600s."""
        config = StrategyConfig(strategy=Strategy.RELAY_SWARM, relay_swarm_size=6)
        effective_tau = compute_effective_tau(1200, config)

        assert effective_tau == 600
        assert effective_tau == 1200 * RELAY_TAU_FACTOR

    def test_relay_p_cost(self, capsys):
        """Should have P cost = swarm_size × 0.05."""
        config = StrategyConfig(strategy=Strategy.RELAY_SWARM, relay_swarm_size=6)
        p_cost = compute_p_cost(config)

        assert p_cost == pytest.approx(0.30)  # 6 × 0.05

    def test_relay_c_factor_unchanged(self):
        """Relay should not affect c factor."""
        config = StrategyConfig(strategy=Strategy.RELAY_SWARM, relay_swarm_size=6)
        c_factor = compute_c_factor(config)

        assert c_factor == 1.0


class TestCombined:
    """Tests for COMBINED strategy."""

    def test_combined_best_tau(self, capsys):
        """Combined should achieve τ = 420s (relay + predictive)."""
        config = StrategyConfig(strategy=Strategy.COMBINED, relay_swarm_size=6)
        effective_tau = compute_effective_tau(1200, config)

        # 1200 × 0.5 × 0.7 = 420s
        expected = 1200 * RELAY_TAU_FACTOR * (1.0 - PREDICTIVE_TAU_REDUCTION)
        assert effective_tau == expected
        assert effective_tau == 420

    def test_combined_alpha_floor(self):
        """Combined should have α floor of 1.2."""
        config = StrategyConfig(strategy=Strategy.COMBINED, relay_swarm_size=6)
        effective_alpha = compute_effective_alpha(1.69, config, 420)

        assert effective_alpha >= 1.2

    def test_combined_c_factor(self):
        """Combined should have c = 0.8 from predictive."""
        config = StrategyConfig(strategy=Strategy.COMBINED)
        c_factor = compute_c_factor(config)

        assert c_factor == PREDICTIVE_C_FACTOR

    def test_combined_p_cost(self):
        """Combined should have relay P cost."""
        config = StrategyConfig(strategy=Strategy.COMBINED, relay_swarm_size=6)
        p_cost = compute_p_cost(config)

        assert p_cost == pytest.approx(0.30)


class TestApplyStrategy:
    """Tests for apply_strategy function."""

    def test_apply_strategy_emits_receipt(self, capsys):
        """Should emit strategy_application receipt."""
        config = StrategyConfig(strategy=Strategy.RELAY_SWARM, relay_swarm_size=6)
        apply_strategy(1200, 1.69, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "strategy_application"' in captured.out

    def test_apply_strategy_returns_result(self):
        """Should return complete StrategyResult."""
        config = StrategyConfig(strategy=Strategy.RELAY_SWARM, relay_swarm_size=6)
        result = apply_strategy(1200, 1.69, config)

        assert isinstance(result, StrategyResult)
        assert result.strategy == Strategy.RELAY_SWARM
        assert result.effective_tau == 600
        assert result.p_cost == pytest.approx(0.30)


class TestCompareStrategies:
    """Tests for compare_strategies function."""

    def test_strategy_comparison_ranking(self, capsys):
        """Should return strategies sorted by cycles."""
        configs = get_all_strategy_configs()
        baseline = {"tau": 1200, "alpha": 1.69}

        results = compare_strategies(configs, baseline)

        # Results should be sorted by cycles_to_10k ascending
        for i in range(len(results) - 1):
            assert results[i].cycles_to_10k <= results[i + 1].cycles_to_10k

    def test_strategy_comparison_emits_receipt(self, capsys):
        """Should emit strategy_comparison receipt."""
        configs = get_all_strategy_configs()
        baseline = {"tau": 1200, "alpha": 1.69}

        compare_strategies(configs, baseline)

        captured = capsys.readouterr()
        assert '"receipt_type": "strategy_comparison"' in captured.out


class TestRecommendStrategy:
    """Tests for recommend_strategy function."""

    def test_recommend_within_constraints(self, capsys):
        """Should recommend best strategy within constraints."""
        configs = get_all_strategy_configs()
        baseline = {"tau": 1200, "alpha": 1.69}
        results = compare_strategies(configs, baseline)

        # Constrain to no P cost
        recommended = recommend_strategy(results, {"max_p_cost": 0.0})

        # Should recommend onboard AI or predictive (no P cost)
        assert recommended is not None
        assert recommended.p_cost <= 0.0

    def test_recommend_no_valid_returns_none(self):
        """Should return None if no valid strategies."""
        configs = get_all_strategy_configs()
        baseline = {"tau": 1200, "alpha": 1.69}
        results = compare_strategies(configs, baseline)

        # Impossible constraints
        recommended = recommend_strategy(results, {"max_p_cost": -1.0})

        assert recommended is None


class TestGetAllStrategyConfigs:
    """Tests for get_all_strategy_configs function."""

    def test_returns_all_strategies(self):
        """Should return config for each Strategy enum."""
        configs = get_all_strategy_configs()

        strategies = [c.strategy for c in configs]
        assert Strategy.BASELINE in strategies
        assert Strategy.ONBOARD_AI in strategies
        assert Strategy.PREDICTIVE in strategies
        assert Strategy.RELAY_SWARM in strategies
        assert Strategy.COMBINED in strategies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
