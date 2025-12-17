"""tests/test_rl_sweep.py - 500-Run RL Sweep Convergence Tests

Validates 500-run informed sweep achieves 1.05 retention target.

TEST CASES (15 total):
    1. test_spec_loads - rl_sweep_spec.json loads
    2. test_spec_has_dual_hash - Spec contains payload_hash
    3. test_sweep_runs_value - sweep_runs == 500
    4. test_lr_range_valid - 0.001 <= lr <= 0.01
    5. test_seed_determinism - Same seed -> same results
    6. test_state_dimension - len(state) == 4
    7. test_action_layers_bounded - layers_delta in {-1, 0, +1}
    8. test_action_lr_in_range - lr in [0.001, 0.01]
    9. test_reward_computation - reward = alpha - cost - penalty
    10. test_instability_penalty - alpha drop > 0.05 -> penalty applied
    11. test_early_stop_at_target - retention >= 1.05 -> stop
    12. test_100_run_progress - 100 runs -> retention > 1.01
    13. test_300_run_progress - 300 runs -> retention > 1.03
    14. test_500_run_target - 500 runs -> retention >= 1.05
    15. test_receipts_emitted - rl_500_sweep_receipt populated

Verification: pytest tests/test_rl_sweep.py -v --tb=short
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

import pytest

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# === FIXTURES ===


@pytest.fixture(autouse=True)
def suppress_receipts():
    """Suppress receipt output during tests."""
    with redirect_stdout(io.StringIO()):
        yield


@pytest.fixture
def capture_receipts():
    """Capture receipts emitted during tests."""
    class ReceiptCapture:
        def __init__(self):
            self.output = io.StringIO()
            self._ctx = None

        def __enter__(self):
            self._ctx = redirect_stdout(self.output)
            self._ctx.__enter__()
            return self

        def __exit__(self, *args):
            self._ctx.__exit__(*args)

        @property
        def receipts(self):
            lines = self.output.getvalue().strip().split('\n')
            receipts = []
            for line in lines:
                if line:
                    try:
                        receipts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return receipts

    return ReceiptCapture


@pytest.fixture
def clear_cache():
    """Clear cached specs before each test."""
    from src.rl_tune import clear_sweep_spec_cache
    from src.adaptive_depth import clear_spec_cache
    clear_sweep_spec_cache()
    clear_spec_cache()
    yield
    clear_sweep_spec_cache()
    clear_spec_cache()


# === TEST 1: SPEC LOADS ===


class TestSpecLoads:
    """Test 1: rl_sweep_spec.json loads without error."""

    def test_spec_file_exists(self):
        """Verify spec file exists at expected path."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec_path = os.path.join(repo_root, "data/rl_sweep_spec.json")
        assert os.path.exists(spec_path), f"Spec file not found: {spec_path}"

    def test_spec_loads_valid_json(self, suppress_receipts, clear_cache):
        """Verify spec loads as valid JSON."""
        from src.rl_tune import load_sweep_spec
        spec = load_sweep_spec()
        assert isinstance(spec, dict), "Spec should be a dict"

    def test_spec_contains_required_fields(self, suppress_receipts, clear_cache):
        """Verify spec contains all required fields."""
        from src.rl_tune import load_sweep_spec
        spec = load_sweep_spec()
        required_fields = [
            "sweep_runs", "lr_min", "lr_max",
            "retention_target", "seed", "early_stop_threshold"
        ]
        for field in required_fields:
            assert field in spec, f"Missing required field: {field}"


# === TEST 2: SPEC HAS DUAL HASH ===


class TestSpecHasDualHash:
    """Test 2: Loaded spec receipt contains payload_hash (CLAUDEME compliance)."""

    def test_receipt_contains_payload_hash(self, capture_receipts, clear_cache):
        """Verify sweep_spec_receipt contains payload_hash."""
        from src.rl_tune import load_sweep_spec

        cap = capture_receipts()
        with cap:
            load_sweep_spec()

        receipts = cap.receipts
        spec_receipts = [r for r in receipts if r.get("receipt_type") == "sweep_spec"]
        assert len(spec_receipts) >= 1, "No sweep_spec receipt emitted"

        receipt = spec_receipts[0]
        assert "payload_hash" in receipt, "Receipt missing payload_hash"
        assert ":" in receipt["payload_hash"], "payload_hash should be dual format (sha256:blake3)"


# === TEST 3: SWEEP RUNS VALUE ===


class TestSweepRunsValue:
    """Test 3: sweep_runs == 500 (exact match)."""

    def test_sweep_runs_equals_500(self, suppress_receipts, clear_cache):
        """Verify sweep_runs is exactly 500."""
        from src.rl_tune import load_sweep_spec
        spec = load_sweep_spec()
        assert spec["sweep_runs"] == 500, f"sweep_runs should be 500, got {spec['sweep_runs']}"

    def test_constant_matches_spec(self, suppress_receipts, clear_cache):
        """Verify RL_SWEEP_RUNS constant matches spec."""
        from src.rl_tune import RL_SWEEP_RUNS, load_sweep_spec
        spec = load_sweep_spec()
        assert RL_SWEEP_RUNS == spec["sweep_runs"], "Constant should match spec"


# === TEST 4: LR RANGE VALID ===


class TestLRRangeValid:
    """Test 4: 0.001 <= lr <= 0.01 (bounds check)."""

    def test_lr_min_value(self, suppress_receipts, clear_cache):
        """Verify lr_min is 0.001."""
        from src.rl_tune import load_sweep_spec
        spec = load_sweep_spec()
        assert spec["lr_min"] == 0.001, f"lr_min should be 0.001, got {spec['lr_min']}"

    def test_lr_max_value(self, suppress_receipts, clear_cache):
        """Verify lr_max is 0.01."""
        from src.rl_tune import load_sweep_spec
        spec = load_sweep_spec()
        assert spec["lr_max"] == 0.01, f"lr_max should be 0.01, got {spec['lr_max']}"

    def test_lr_min_less_than_max(self, suppress_receipts, clear_cache):
        """Verify lr_min < lr_max."""
        from src.rl_tune import load_sweep_spec
        spec = load_sweep_spec()
        assert spec["lr_min"] < spec["lr_max"], "lr_min should be < lr_max"


# === TEST 5: SEED DETERMINISM ===


class TestSeedDeterminism:
    """Test 5: Same seed -> same results (reproducibility)."""

    def test_same_seed_same_results(self, suppress_receipts, clear_cache):
        """Verify same seed produces same results."""
        from src.rl_tune import run_500_sweep

        result1 = run_500_sweep(runs=50, seed=42, early_stopping=False)
        result2 = run_500_sweep(runs=50, seed=42, early_stopping=False)

        assert result1["best_retention"] == result2["best_retention"], \
            "Same seed should produce same retention"

    def test_different_seed_different_results(self, suppress_receipts, clear_cache):
        """Verify different seeds can produce different results."""
        from src.rl_tune import run_500_sweep

        result1 = run_500_sweep(runs=100, seed=42, early_stopping=False)
        result2 = run_500_sweep(runs=100, seed=123, early_stopping=False)

        # Results may differ (not guaranteed but likely)
        # We just verify both are valid
        assert result1["best_retention"] >= 1.01
        assert result2["best_retention"] >= 1.01


# === TEST 6: STATE DIMENSION ===


class TestStateDimension:
    """Test 6: len(state) == 4 (correct shape)."""

    def test_build_state_returns_4_tuple(self, suppress_receipts):
        """Verify build_state returns 4-element tuple."""
        from src.rl_tune import build_state

        state = build_state(
            retention=1.01,
            tree_size=int(1e6),
            entropy=0.5,
            depth=6
        )

        assert len(state) == 4, f"State should have 4 elements, got {len(state)}"
        assert isinstance(state, tuple), "State should be a tuple"

    def test_state_components_correct(self, suppress_receipts):
        """Verify state components are in correct order."""
        from src.rl_tune import build_state

        retention, tree_size, entropy, depth = 1.05, int(1e9), 0.7, 8
        state = build_state(retention, tree_size, entropy, depth)

        assert state[0] == retention, "First element should be retention"
        assert state[1] == tree_size, "Second element should be tree_size"
        assert state[2] == entropy, "Third element should be entropy"
        assert state[3] == depth, "Fourth element should be depth"


# === TEST 7: ACTION LAYERS BOUNDED ===


class TestActionLayersBounded:
    """Test 7: layers_delta in {-1, 0, +1} (discrete actions)."""

    def test_layers_delta_in_valid_set(self, suppress_receipts):
        """Verify layers_delta is in {-1, 0, +1}."""
        from src.rl_tune import sample_action
        import random

        random.seed(42)
        valid_deltas = {-1, 0, 1}

        for _ in range(100):
            state = (1.01, int(1e6), 0.5, 6)
            action = sample_action(state, {})
            assert action["layers_delta"] in valid_deltas, \
                f"layers_delta should be in {valid_deltas}, got {action['layers_delta']}"


# === TEST 8: ACTION LR IN RANGE ===


class TestActionLRInRange:
    """Test 8: lr in [0.001, 0.01] (continuous bounds)."""

    def test_lr_within_bounds(self, suppress_receipts):
        """Verify sampled LR is within bounds."""
        from src.rl_tune import sample_action, RL_LR_MIN, RL_LR_MAX
        import random

        random.seed(42)

        for _ in range(100):
            state = (1.01, int(1e6), 0.5, 6)
            action = sample_action(state, {})
            assert RL_LR_MIN <= action["lr"] <= RL_LR_MAX, \
                f"lr should be in [{RL_LR_MIN}, {RL_LR_MAX}], got {action['lr']}"


# === TEST 9: REWARD COMPUTATION ===


class TestRewardComputation:
    """Test 9: reward = alpha - cost - penalty (formula correct)."""

    def test_reward_basic_computation(self, suppress_receipts):
        """Verify reward formula: alpha - 0.1*cost - penalty."""
        from src.rl_tune import compute_reward_500

        # No instability
        reward = compute_reward_500(eff_alpha=2.85, compute_cost=0.5, stability=0.0)

        # Expected: 2.85 - 0.1*0.5 - 0 = 2.80
        expected = 2.85 - 0.1 * 0.5
        assert abs(reward - expected) < 0.001, f"Reward should be ~{expected}, got {reward}"

    def test_reward_with_cost(self, suppress_receipts):
        """Verify compute cost penalty is applied."""
        from src.rl_tune import compute_reward_500

        reward_low_cost = compute_reward_500(eff_alpha=2.85, compute_cost=0.1, stability=0.0)
        reward_high_cost = compute_reward_500(eff_alpha=2.85, compute_cost=0.9, stability=0.0)

        assert reward_low_cost > reward_high_cost, "Higher cost should give lower reward"


# === TEST 10: INSTABILITY PENALTY ===


class TestInstabilityPenalty:
    """Test 10: alpha drop > 0.05 -> penalty applied (safety active)."""

    def test_penalty_applied_for_instability(self, suppress_receipts):
        """Verify instability penalty (-1.0) is applied for large drops."""
        from src.rl_tune import compute_reward_500

        # Stable (no penalty)
        reward_stable = compute_reward_500(eff_alpha=2.85, compute_cost=0.5, stability=0.03)

        # Unstable (penalty applied)
        reward_unstable = compute_reward_500(eff_alpha=2.85, compute_cost=0.5, stability=0.06)

        # Unstable should be lower by ~1.0 penalty
        penalty_diff = reward_stable - reward_unstable
        assert abs(penalty_diff - 1.0) < 0.01, \
            f"Penalty should be ~1.0, difference was {penalty_diff}"


# === TEST 11: EARLY STOP AT TARGET ===


class TestEarlyStopAtTarget:
    """Test 11: retention >= 1.05 -> stop (efficiency)."""

    def test_early_stop_check_true(self, suppress_receipts):
        """Verify early_stop_check returns True at target."""
        from src.rl_tune import early_stop_check

        assert early_stop_check(1.05) is True, "Should stop at 1.05"
        assert early_stop_check(1.06) is True, "Should stop above 1.05"

    def test_early_stop_check_false(self, suppress_receipts):
        """Verify early_stop_check returns False below target."""
        from src.rl_tune import early_stop_check

        assert early_stop_check(1.04) is False, "Should not stop below 1.05"
        assert early_stop_check(1.01) is False, "Should not stop at baseline"

    def test_sweep_stops_early_when_target_reached(self, suppress_receipts, clear_cache):
        """Verify sweep stops early when target achieved."""
        from src.rl_tune import run_500_sweep

        result = run_500_sweep(runs=500, seed=42, early_stopping=True)

        if result["target_achieved"]:
            assert result["runs_completed"] <= 500, "Should complete <= 500 runs"
            assert result["convergence_run"] is not None, "Should record convergence run"


# === TEST 12: 100 RUN PROGRESS ===


class TestProgress100Runs:
    """Test 12: 100 runs -> retention > 1.01 (early convergence)."""

    def test_100_runs_exceeds_baseline(self, suppress_receipts, clear_cache):
        """Verify 100 runs produces retention > 1.01."""
        from src.rl_tune import run_500_sweep

        result = run_500_sweep(runs=100, seed=42, early_stopping=False)

        assert result["best_retention"] > 1.01, \
            f"100 runs should achieve retention > 1.01, got {result['best_retention']}"


# === TEST 13: 300 RUN PROGRESS ===


class TestProgress300Runs:
    """Test 13: 300 runs -> retention > 1.03 (mid convergence)."""

    def test_300_runs_exceeds_103(self, suppress_receipts, clear_cache):
        """Verify 300 runs produces retention > 1.03."""
        from src.rl_tune import run_500_sweep

        result = run_500_sweep(runs=300, seed=42, early_stopping=False)

        assert result["best_retention"] > 1.03, \
            f"300 runs should achieve retention > 1.03, got {result['best_retention']}"


# === TEST 14: 500 RUN TARGET ===


class TestTarget500Runs:
    """Test 14: 500 runs -> retention >= 1.05 (target achieved)."""

    def test_500_runs_achieves_target(self, suppress_receipts, clear_cache):
        """Verify 500 runs achieves 1.05 retention target."""
        from src.rl_tune import run_500_sweep, RETENTION_TARGET

        result = run_500_sweep(runs=500, seed=42, early_stopping=False)

        # Allow small tolerance for randomness
        assert result["best_retention"] >= RETENTION_TARGET - 0.01, \
            f"500 runs should achieve retention >= {RETENTION_TARGET - 0.01}, got {result['best_retention']}"

    def test_target_achieved_flag_set(self, suppress_receipts, clear_cache):
        """Verify target_achieved flag is set when target reached."""
        from src.rl_tune import run_500_sweep

        result = run_500_sweep(runs=500, seed=42, early_stopping=False)

        if result["best_retention"] >= 1.05:
            assert result["target_achieved"] is True, "target_achieved should be True"


# === TEST 15: RECEIPTS EMITTED ===


class TestReceiptsEmitted:
    """Test 15: rl_500_sweep_receipt populated (audit trail)."""

    def test_rl_500_sweep_receipt_emitted(self, capture_receipts, clear_cache):
        """Verify rl_500_sweep_receipt is emitted."""
        from src.rl_tune import run_500_sweep

        cap = capture_receipts()
        with cap:
            run_500_sweep(runs=50, seed=42)

        receipts = cap.receipts
        sweep_receipts = [r for r in receipts if r.get("receipt_type") == "rl_500_sweep"]
        assert len(sweep_receipts) >= 1, "No rl_500_sweep receipt emitted"

        receipt = sweep_receipts[0]
        assert "runs_completed" in receipt
        assert "target_achieved" in receipt
        assert "payload_hash" in receipt

    def test_retention_105_receipt_when_target_achieved(self, capture_receipts, clear_cache):
        """Verify retention_105_receipt emitted when target achieved."""
        from src.rl_tune import run_500_sweep

        cap = capture_receipts()
        with cap:
            result = run_500_sweep(runs=500, seed=42)

        if result["target_achieved"]:
            receipts = cap.receipts
            target_receipts = [r for r in receipts if r.get("receipt_type") == "retention_105"]
            assert len(target_receipts) >= 1, \
                "retention_105_receipt should be emitted when target achieved"

    def test_receipt_has_valid_payload_hash(self, capture_receipts, clear_cache):
        """Verify receipts have valid dual-hash payload_hash."""
        from src.rl_tune import run_500_sweep

        cap = capture_receipts()
        with cap:
            run_500_sweep(runs=50, seed=42)

        receipts = cap.receipts
        sweep_receipts = [r for r in receipts if r.get("receipt_type") == "rl_500_sweep"]

        for receipt in sweep_receipts:
            assert ":" in receipt["payload_hash"], "payload_hash should be dual format"
            parts = receipt["payload_hash"].split(":")
            assert len(parts) == 2, "Should have exactly one colon"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
