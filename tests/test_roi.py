"""test_roi.py - Tests for ROI reward/penalty system

Validates:
- ROI reward computation
- ROI penalty computation (P cost, c reduction)
- ROI computation formula
- ROI gate decisions (deploy/shadow/kill)
- Strategy ranking by ROI
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.roi import (
    ROIConfig,
    reward,
    penalty,
    compute_roi,
    roi_gate,
    rank_by_roi,
    update_result_roi,
    evaluate_strategy_roi,
    ROI_GATE_DEPLOY,
    ROI_GATE_SHADOW,
    DEFAULT_REWARD_PER_CYCLE,
    DEFAULT_PENALTY_PER_P_COST,
    DEFAULT_PENALTY_PER_C_REDUCTION,
)
from src.strategies import Strategy, StrategyResult


class TestROIConfig:
    """Tests for ROIConfig dataclass."""

    def test_default_config(self):
        """Should have expected default values."""
        config = ROIConfig()

        assert config.reward_per_cycle_saved == DEFAULT_REWARD_PER_CYCLE
        assert config.penalty_per_p_cost == DEFAULT_PENALTY_PER_P_COST
        assert config.penalty_per_c_reduction == DEFAULT_PENALTY_PER_C_REDUCTION

    def test_custom_config(self):
        """Should accept custom values."""
        config = ROIConfig(reward_per_cycle_saved=2.0, penalty_per_p_cost=1.0)

        assert config.reward_per_cycle_saved == 2.0
        assert config.penalty_per_p_cost == 1.0


class TestReward:
    """Tests for reward function."""

    def test_roi_reward(self, capsys):
        """3 cycles saved × 1.0 = 3.0 reward."""
        config = ROIConfig()
        result = reward(3, config)

        assert result == 3.0

    def test_roi_reward_zero_cycles(self):
        """Zero cycles saved should give zero reward."""
        config = ROIConfig()
        result = reward(0, config)

        assert result == 0.0

    def test_roi_reward_emits_receipt(self, capsys):
        """Should emit roi_reward receipt."""
        config = ROIConfig()
        reward(3, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "roi_reward"' in captured.out


class TestPenalty:
    """Tests for penalty function."""

    def test_roi_penalty_p_cost(self, capsys):
        """0.30 P × 0.5 = 0.15 penalty."""
        config = ROIConfig()
        result = penalty(0.30, 0.0, config)

        assert result == 0.15

    def test_roi_penalty_c_reduction(self):
        """0.2 c_reduction × 0.3 = 0.06 penalty."""
        config = ROIConfig()
        result = penalty(0.0, 0.2, config)

        assert result == pytest.approx(0.06)

    def test_roi_penalty_combined(self):
        """Combined P cost and c reduction penalty."""
        config = ROIConfig()
        result = penalty(0.30, 0.2, config)

        # 0.30 × 0.5 + 0.2 × 0.3 = 0.15 + 0.06 = 0.21
        assert result == pytest.approx(0.21)

    def test_roi_penalty_emits_receipt(self, capsys):
        """Should emit roi_penalty receipt."""
        config = ROIConfig()
        penalty(0.30, 0.2, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "roi_penalty"' in captured.out


class TestComputeROI:
    """Tests for compute_roi function."""

    def test_roi_computation(self, capsys):
        """ROI = reward - penalty."""
        config = ROIConfig()

        baseline = StrategyResult(
            strategy=Strategy.BASELINE,
            effective_tau=1200,
            effective_alpha=0.59,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=3,
            roi_score=0.0
        )

        result = StrategyResult(
            strategy=Strategy.RELAY_SWARM,
            effective_tau=600,
            effective_alpha=1.0,
            c_factor=1.0,
            p_cost=0.30,
            cycles_to_10k=2,
            roi_score=0.0
        )

        roi = compute_roi(result, baseline, config)

        # cycles_saved = 3 - 2 = 1
        # p_cost = 0.30
        # c_reduction = 0 (both c_factor = 1.0)
        # reward = 1 × 1.0 = 1.0
        # penalty = 0.30 × 0.5 = 0.15
        # ROI = 1.0 - 0.15 = 0.85
        assert roi == pytest.approx(0.85)

    def test_roi_computation_emits_receipt(self, capsys):
        """Should emit roi_computation receipt."""
        config = ROIConfig()

        baseline = StrategyResult(
            strategy=Strategy.BASELINE,
            effective_tau=1200,
            effective_alpha=0.59,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=3,
            roi_score=0.0
        )

        result = StrategyResult(
            strategy=Strategy.RELAY_SWARM,
            effective_tau=600,
            effective_alpha=1.0,
            c_factor=1.0,
            p_cost=0.30,
            cycles_to_10k=2,
            roi_score=0.0
        )

        compute_roi(result, baseline, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "roi_computation"' in captured.out


class TestROIGate:
    """Tests for roi_gate function."""

    def test_roi_gate_deploy(self, capsys):
        """ROI ≥ 0.5 should return 'deploy'."""
        config = ROIConfig()
        decision = roi_gate(0.5, config)

        assert decision == "deploy"

    def test_roi_gate_shadow(self):
        """0.1 ≤ ROI < 0.5 should return 'shadow'."""
        config = ROIConfig()

        assert roi_gate(0.1, config) == "shadow"
        assert roi_gate(0.3, config) == "shadow"
        assert roi_gate(0.49, config) == "shadow"

    def test_roi_gate_kill(self):
        """ROI < 0.1 should return 'kill'."""
        config = ROIConfig()

        assert roi_gate(0.09, config) == "kill"
        assert roi_gate(0.0, config) == "kill"
        assert roi_gate(-1.0, config) == "kill"

    def test_roi_gate_emits_receipt(self, capsys):
        """Should emit roi_gate receipt."""
        config = ROIConfig()
        roi_gate(0.5, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "roi_gate"' in captured.out


class TestRankByROI:
    """Tests for rank_by_roi function."""

    def test_rank_by_roi_descending(self, capsys):
        """Strategies should be sorted descending by ROI."""
        config = ROIConfig()

        baseline = StrategyResult(
            strategy=Strategy.BASELINE,
            effective_tau=1200,
            effective_alpha=0.59,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=5,
            roi_score=0.0
        )

        results = [
            StrategyResult(Strategy.ONBOARD_AI, 1200, 1.2, 1.0, 0.0, 3, 0.0),
            StrategyResult(Strategy.RELAY_SWARM, 600, 1.0, 1.0, 0.30, 2, 0.0),
            StrategyResult(Strategy.PREDICTIVE, 840, 0.8, 0.8, 0.0, 4, 0.0),
        ]

        ranked = rank_by_roi(results, baseline, config)

        # Check sorted descending
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]

    def test_rank_by_roi_emits_receipt(self, capsys):
        """Should emit roi_ranking receipt."""
        config = ROIConfig()

        baseline = StrategyResult(
            strategy=Strategy.BASELINE,
            effective_tau=1200,
            effective_alpha=0.59,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=5,
            roi_score=0.0
        )

        results = [
            StrategyResult(Strategy.ONBOARD_AI, 1200, 1.2, 1.0, 0.0, 3, 0.0),
        ]

        rank_by_roi(results, baseline, config)

        captured = capsys.readouterr()
        assert '"receipt_type": "roi_ranking"' in captured.out


class TestUpdateResultROI:
    """Tests for update_result_roi function."""

    def test_updates_roi_score(self):
        """Should return new result with roi_score populated."""
        config = ROIConfig()

        baseline = StrategyResult(
            strategy=Strategy.BASELINE,
            effective_tau=1200,
            effective_alpha=0.59,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=3,
            roi_score=0.0
        )

        result = StrategyResult(
            strategy=Strategy.ONBOARD_AI,
            effective_tau=1200,
            effective_alpha=1.2,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=2,
            roi_score=0.0
        )

        updated = update_result_roi(result, baseline, config)

        assert updated.roi_score != 0.0
        assert updated.strategy == Strategy.ONBOARD_AI


class TestEvaluateStrategyROI:
    """Tests for evaluate_strategy_roi function."""

    def test_full_evaluation(self):
        """Should return complete evaluation dict."""
        config = ROIConfig()

        baseline = StrategyResult(
            strategy=Strategy.BASELINE,
            effective_tau=1200,
            effective_alpha=0.59,
            c_factor=1.0,
            p_cost=0.0,
            cycles_to_10k=3,
            roi_score=0.0
        )

        result = StrategyResult(
            strategy=Strategy.RELAY_SWARM,
            effective_tau=600,
            effective_alpha=1.0,
            c_factor=1.0,
            p_cost=0.30,
            cycles_to_10k=2,
            roi_score=0.0
        )

        evaluation = evaluate_strategy_roi(result, baseline, config)

        assert "roi_score" in evaluation
        assert "decision" in evaluation
        assert "cycles_saved" in evaluation
        assert "p_cost" in evaluation
        assert "c_reduction" in evaluation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
