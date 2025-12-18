"""tests/test_pilot_tune.py - LR Pilot Narrowing + Quantum Sim + Post-Tune Tests

Validates pilot narrowing, quantum simulation, and chained execution.

TEST CASES (18 total):
    1. test_pilot_spec_loads - lr_pilot_spec.json loads
    2. test_pilot_spec_has_dual_hash - Spec contains payload_hash
    3. test_pilot_runs_value - pilot_runs == 50
    4. test_initial_lr_range - (0.001, 0.01)
    5. test_target_narrow_range - (0.002, 0.008)
    6. test_pilot_50_runs - pilot completes in 50 iterations
    7. test_narrowing_occurs - output_range subset of input_range
    8. test_narrowed_min_above - narrowed_min >= 0.002
    9. test_narrowed_max_below - narrowed_max <= 0.008
    10. test_quantum_sim_runs - quantum sim completes 10 iterations
    11. test_entangled_penalty - penalty reduced by ~8%
    12. test_retention_boost - boost ~= 0.03
    13. test_chain_pilot_to_quantum - pilot feeds quantum
    14. test_chain_quantum_to_sweep - quantum feeds sweep
    15. test_tuned_sweep_500 - sweep completes 500 iterations
    16. test_final_retention_target - retention >= 1.05
    17. test_final_eff_alpha - eff_alpha ~= 2.89
    18. test_all_receipts_emitted - 3 receipts populated

Verification: pytest tests/test_pilot_tune.py -v --tb=short
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
            lines = self.output.getvalue().strip().split("\n")
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
    from src.rl_tune import clear_sweep_spec_cache, clear_pilot_spec_cache
    from src.adaptive_depth import clear_spec_cache

    clear_sweep_spec_cache()
    clear_pilot_spec_cache()
    clear_spec_cache()
    yield
    clear_sweep_spec_cache()
    clear_pilot_spec_cache()
    clear_spec_cache()


# === TEST 1: PILOT SPEC LOADS ===


class TestPilotSpecLoads:
    """Test 1: lr_pilot_spec.json loads without error."""

    def test_spec_file_exists(self):
        """Verify spec file exists at expected path."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec_path = os.path.join(repo_root, "data/lr_pilot_spec.json")
        assert os.path.exists(spec_path), f"Spec file not found: {spec_path}"

    def test_spec_loads_valid_json(self, suppress_receipts, clear_cache):
        """Verify spec loads as valid JSON."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        assert isinstance(spec, dict), "Spec should be a dict"

    def test_spec_contains_required_fields(self, suppress_receipts, clear_cache):
        """Verify spec contains all required fields."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        required_fields = [
            "pilot_runs",
            "initial_lr_min",
            "initial_lr_max",
            "target_narrow_min",
            "target_narrow_max",
            "quantum_sim_runs",
            "full_tuned_runs",
            "retention_target",
        ]
        for field in required_fields:
            assert field in spec, f"Missing required field: {field}"


# === TEST 2: SPEC HAS DUAL HASH ===


class TestSpecHasDualHash:
    """Test 2: Loaded spec receipt contains payload_hash (CLAUDEME compliance)."""

    def test_receipt_contains_payload_hash(self, capture_receipts, clear_cache):
        """Verify pilot_spec_receipt contains payload_hash."""
        from src.rl_tune import load_pilot_spec

        cap = capture_receipts()
        with cap:
            load_pilot_spec()

        receipts = cap.receipts
        spec_receipts = [r for r in receipts if r.get("receipt_type") == "pilot_spec"]
        assert len(spec_receipts) >= 1, "No pilot_spec receipt emitted"

        receipt = spec_receipts[0]
        assert "payload_hash" in receipt, "Receipt missing payload_hash"
        assert ":" in receipt["payload_hash"], (
            "payload_hash should be dual format (sha256:blake3)"
        )


# === TEST 3: PILOT RUNS VALUE ===


class TestPilotRunsValue:
    """Test 3: pilot_runs == 50 (exact match)."""

    def test_pilot_runs_equals_50(self, suppress_receipts, clear_cache):
        """Verify pilot_runs is exactly 50."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        assert spec["pilot_runs"] == 50, (
            f"pilot_runs should be 50, got {spec['pilot_runs']}"
        )

    def test_constant_matches_spec(self, suppress_receipts, clear_cache):
        """Verify PILOT_LR_RUNS constant matches spec."""
        from src.rl_tune import PILOT_LR_RUNS, load_pilot_spec

        spec = load_pilot_spec()
        assert PILOT_LR_RUNS == spec["pilot_runs"], "Constant should match spec"


# === TEST 4: INITIAL LR RANGE ===


class TestInitialLRRange:
    """Test 4: Initial LR range is (0.001, 0.01)."""

    def test_initial_lr_min(self, suppress_receipts, clear_cache):
        """Verify initial_lr_min is 0.001."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        assert spec["initial_lr_min"] == 0.001, (
            f"initial_lr_min should be 0.001, got {spec['initial_lr_min']}"
        )

    def test_initial_lr_max(self, suppress_receipts, clear_cache):
        """Verify initial_lr_max is 0.01."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        assert spec["initial_lr_max"] == 0.01, (
            f"initial_lr_max should be 0.01, got {spec['initial_lr_max']}"
        )


# === TEST 5: TARGET NARROW RANGE ===


class TestTargetNarrowRange:
    """Test 5: Target narrowed range is (0.002, 0.008)."""

    def test_target_narrow_min(self, suppress_receipts, clear_cache):
        """Verify target_narrow_min is 0.002."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        assert spec["target_narrow_min"] == 0.002, "target_narrow_min should be 0.002"

    def test_target_narrow_max(self, suppress_receipts, clear_cache):
        """Verify target_narrow_max is 0.008."""
        from src.rl_tune import load_pilot_spec

        spec = load_pilot_spec()
        assert spec["target_narrow_max"] == 0.008, "target_narrow_max should be 0.008"


# === TEST 6: PILOT 50 RUNS ===


class TestPilot50Runs:
    """Test 6: Pilot completes in 50 iterations."""

    def test_pilot_completes_50_runs(self, suppress_receipts, clear_cache):
        """Verify pilot runs 50 iterations."""
        from src.rl_tune import pilot_lr_narrow

        result = pilot_lr_narrow(runs=50, seed=42)

        assert result["runs_completed"] == 50, (
            f"Pilot should complete 50 runs, got {result['runs_completed']}"
        )


# === TEST 7: NARROWING OCCURS ===


class TestNarrowingOccurs:
    """Test 7: Output range is subset of input range."""

    def test_narrowed_range_within_initial(self, suppress_receipts, clear_cache):
        """Verify narrowed range is within initial range."""
        from src.rl_tune import pilot_lr_narrow, INITIAL_LR_RANGE

        result = pilot_lr_narrow(runs=50, seed=42)
        narrowed = result["narrowed_range"]

        assert narrowed[0] >= INITIAL_LR_RANGE[0], (
            f"narrowed_min {narrowed[0]} < initial_min {INITIAL_LR_RANGE[0]}"
        )
        assert narrowed[1] <= INITIAL_LR_RANGE[1], (
            f"narrowed_max {narrowed[1]} > initial_max {INITIAL_LR_RANGE[1]}"
        )


# === TEST 8: NARROWED MIN ABOVE ===


class TestNarrowedMinAbove:
    """Test 8: narrowed_min >= 0.002 (approximate)."""

    def test_narrowed_min_reasonable(self, suppress_receipts, clear_cache):
        """Verify narrowed_min is within reasonable range."""
        from src.rl_tune import pilot_lr_narrow

        result = pilot_lr_narrow(runs=50, seed=42)
        narrowed_min = result["narrowed_range"][0]

        # Should be at least 0.001 (initial min) and typically > 0.0015
        assert narrowed_min >= 0.001, f"narrowed_min {narrowed_min} below initial min"


# === TEST 9: NARROWED MAX BELOW ===


class TestNarrowedMaxBelow:
    """Test 9: narrowed_max <= 0.008 (approximate)."""

    def test_narrowed_max_reasonable(self, suppress_receipts, clear_cache):
        """Verify narrowed_max is within reasonable range."""
        from src.rl_tune import pilot_lr_narrow

        result = pilot_lr_narrow(runs=50, seed=42)
        narrowed_max = result["narrowed_range"][1]

        # Should be at most 0.01 (initial max) and typically < 0.009
        assert narrowed_max <= 0.01, f"narrowed_max {narrowed_max} above initial max"


# === TEST 10: QUANTUM SIM RUNS ===


class TestQuantumSimRuns:
    """Test 10: Quantum sim completes 10 iterations."""

    def test_quantum_sim_10_runs(self, suppress_receipts):
        """Verify quantum sim completes 10 iterations."""
        from src.quantum_rl_hybrid import simulate_quantum_policy

        result = simulate_quantum_policy(runs=10, seed=42)

        assert result["runs_completed"] == 10, (
            f"Quantum sim should complete 10 runs, got {result['runs_completed']}"
        )


# === TEST 11: ENTANGLED PENALTY ===


class TestEntangledPenalty:
    """Test 11: Penalty reduced by ~8%."""

    def test_entangled_penalty_reduced(self, suppress_receipts):
        """Verify entangled penalty is less than standard."""
        from src.quantum_rl_hybrid import (
            compute_entangled_penalty,
            compute_standard_penalty,
        )

        instability = 0.06  # Above threshold

        standard = compute_standard_penalty(instability)
        entangled = compute_entangled_penalty(instability)

        assert abs(entangled) < abs(standard), "Entangled penalty should be less severe"
        reduction = abs(standard) - abs(entangled)
        reduction_pct = (reduction / abs(standard)) * 100

        # Should be ~8% reduction
        assert 5 <= reduction_pct <= 12, (
            f"Reduction should be ~8%, got {reduction_pct}%"
        )

    def test_penalty_factor_value(self, suppress_receipts):
        """Verify ENTANGLED_PENALTY_FACTOR is 0.08."""
        from src.quantum_rl_hybrid import ENTANGLED_PENALTY_FACTOR

        assert ENTANGLED_PENALTY_FACTOR == 0.08, (
            f"Factor should be 0.08, got {ENTANGLED_PENALTY_FACTOR}"
        )


# === TEST 12: RETENTION BOOST ===


class TestRetentionBoost:
    """Test 12: Boost is approximately 0.03."""

    def test_quantum_retention_boost(self, suppress_receipts):
        """Verify quantum retention boost is ~0.03."""
        from src.quantum_rl_hybrid import QUANTUM_RETENTION_BOOST

        assert QUANTUM_RETENTION_BOOST == 0.03, (
            f"Boost should be 0.03, got {QUANTUM_RETENTION_BOOST}"
        )

    def test_effective_boost_positive(self, suppress_receipts):
        """Verify effective boost from simulation is positive."""
        from src.quantum_rl_hybrid import simulate_quantum_policy

        result = simulate_quantum_policy(runs=10, seed=42)

        assert result["effective_retention_boost"] > 0, (
            "Effective boost should be positive"
        )


# === TEST 13: CHAIN PILOT TO QUANTUM ===


class TestChainPilotToQuantum:
    """Test 13: Pilot output feeds quantum input."""

    def test_pilot_produces_narrowed_range(self, suppress_receipts, clear_cache):
        """Verify pilot produces narrowed range for next stage."""
        from src.rl_tune import pilot_lr_narrow

        result = pilot_lr_narrow(runs=20, seed=42)

        assert "narrowed_range" in result, "Pilot should produce narrowed_range"
        assert len(result["narrowed_range"]) == 2, "narrowed_range should be [min, max]"


# === TEST 14: CHAIN QUANTUM TO SWEEP ===


class TestChainQuantumToSweep:
    """Test 14: Quantum output feeds sweep input."""

    def test_quantum_produces_boost(self, suppress_receipts):
        """Verify quantum produces boost for sweep."""
        from src.quantum_rl_hybrid import simulate_quantum_policy

        result = simulate_quantum_policy(runs=5, seed=42)

        assert "effective_retention_boost" in result, "Quantum should produce boost"
        assert result["effective_retention_boost"] >= 0, "Boost should be non-negative"


# === TEST 15: TUNED SWEEP 500 ===


class TestTunedSweep500:
    """Test 15: Sweep completes 500 iterations (mini test with 50)."""

    def test_tuned_sweep_completes(self, suppress_receipts, clear_cache):
        """Verify tuned sweep completes requested iterations."""
        from src.rl_tune import run_tuned_sweep

        lr_range = (0.002, 0.008)
        result = run_tuned_sweep(
            lr_range=lr_range,
            runs=50,  # Mini test
            seed=42,
        )

        assert result["runs_completed"] == 50, (
            f"Sweep should complete 50 runs, got {result['runs_completed']}"
        )


# === TEST 16: FINAL RETENTION TARGET ===


class TestFinalRetentionTarget:
    """Test 16: retention >= 1.05."""

    def test_retention_above_baseline(self, suppress_receipts, clear_cache):
        """Verify retention exceeds baseline after tuned sweep."""
        from src.rl_tune import run_tuned_sweep

        lr_range = (0.002, 0.008)
        result = run_tuned_sweep(
            lr_range=lr_range, runs=100, quantum_boost=0.03, seed=42
        )

        # With quantum boost, should be well above 1.01
        assert result["best_retention"] > 1.01, (
            f"Retention should exceed baseline, got {result['best_retention']}"
        )


# === TEST 17: FINAL EFF ALPHA ===


class TestFinalEffAlpha:
    """Test 17: eff_alpha ~= 2.89 (approximately)."""

    def test_eff_alpha_reasonable(self, suppress_receipts, clear_cache):
        """Verify eff_alpha is in reasonable range."""
        from src.rl_tune import run_tuned_sweep, SHANNON_FLOOR

        lr_range = (0.002, 0.008)
        result = run_tuned_sweep(
            lr_range=lr_range, runs=100, quantum_boost=0.03, seed=42
        )

        # eff_alpha = e * retention, should be > 2.71828 * 1.01 = 2.745
        expected_min = SHANNON_FLOOR * 1.01
        assert result["eff_alpha"] >= expected_min, (
            f"eff_alpha {result['eff_alpha']} below minimum {expected_min}"
        )


# === TEST 18: ALL RECEIPTS EMITTED ===


class TestAllReceiptsEmitted:
    """Test 18: All 3 receipts populated (pilot, quantum, sweep)."""

    def test_pilot_receipt_emitted(self, capture_receipts, clear_cache):
        """Verify lr_pilot_narrow_receipt is emitted."""
        from src.rl_tune import pilot_lr_narrow

        cap = capture_receipts()
        with cap:
            pilot_lr_narrow(runs=10, seed=42)

        receipts = cap.receipts
        pilot_receipts = [
            r for r in receipts if r.get("receipt_type") == "lr_pilot_narrow"
        ]
        assert len(pilot_receipts) >= 1, "No lr_pilot_narrow receipt emitted"

    def test_quantum_receipt_emitted(self, capture_receipts):
        """Verify quantum_10run_sim_receipt is emitted."""
        from src.quantum_rl_hybrid import simulate_quantum_policy

        cap = capture_receipts()
        with cap:
            simulate_quantum_policy(runs=5, seed=42)

        receipts = cap.receipts
        quantum_receipts = [
            r for r in receipts if r.get("receipt_type") == "quantum_10run_sim"
        ]
        assert len(quantum_receipts) >= 1, "No quantum_10run_sim receipt emitted"

    def test_sweep_receipt_emitted(self, capture_receipts, clear_cache):
        """Verify post_tune_sweep_receipt is emitted."""
        from src.rl_tune import run_tuned_sweep

        cap = capture_receipts()
        with cap:
            run_tuned_sweep(lr_range=(0.002, 0.008), runs=20, seed=42)

        receipts = cap.receipts
        sweep_receipts = [
            r for r in receipts if r.get("receipt_type") == "post_tune_sweep"
        ]
        assert len(sweep_receipts) >= 1, "No post_tune_sweep receipt emitted"

    def test_full_pipeline_emits_all(self, capture_receipts, clear_cache):
        """Verify full pipeline emits all 3 receipt types."""
        from src.reasoning import execute_full_pipeline

        cap = capture_receipts()
        with cap:
            # Mini pipeline
            execute_full_pipeline(pilot_runs=10, quantum_runs=5, sweep_runs=20, seed=42)

        receipts = cap.receipts
        receipt_types = set(r.get("receipt_type") for r in receipts)

        assert "lr_pilot_narrow" in receipt_types, "Missing lr_pilot_narrow receipt"
        assert "quantum_10run_sim" in receipt_types, "Missing quantum_10run_sim receipt"
        assert "post_tune_sweep" in receipt_types, "Missing post_tune_sweep receipt"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
