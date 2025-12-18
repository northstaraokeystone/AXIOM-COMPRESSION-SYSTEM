"""Tests for quantum-resistant Spectre defense.

Test coverage:
- Quantum-resistant config loading
- Key generation
- Spectre variant defense (v1, v2, v4)
- Cache timing defense
- Overall quantum resilience
"""


class TestQuantumResistConfig:
    """Tests for quantum-resistant configuration."""

    def test_quantum_config_loads(self):
        """Test quantum-resistant config loads."""
        from src.quantum_resist_random import load_quantum_resist_config

        config = load_quantum_resist_config()
        assert config is not None
        assert "key_size_bits" in config
        assert "spectre_variants" in config

    def test_quantum_key_size(self):
        """Test key size is 256 bits."""
        from src.quantum_resist_random import (
            load_quantum_resist_config,
            QUANTUM_KEY_SIZE_BITS,
        )

        config = load_quantum_resist_config()
        assert config["key_size_bits"] == 256
        assert QUANTUM_KEY_SIZE_BITS == 256

    def test_spectre_variants_present(self):
        """Test all 3 Spectre variants present."""
        from src.quantum_resist_random import (
            load_quantum_resist_config,
            SPECTRE_VARIANTS,
        )

        config = load_quantum_resist_config()
        assert "v1" in config["spectre_variants"]
        assert "v2" in config["spectre_variants"]
        assert "v4" in config["spectre_variants"]
        assert SPECTRE_VARIANTS == ["v1", "v2", "v4"]

    def test_cache_randomization_enabled(self):
        """Test cache randomization is enabled."""
        from src.quantum_resist_random import (
            load_quantum_resist_config,
            CACHE_RANDOMIZATION_ENABLED,
        )

        config = load_quantum_resist_config()
        assert config["cache_randomization"] is True
        assert CACHE_RANDOMIZATION_ENABLED is True

    def test_branch_defense_enabled(self):
        """Test branch prediction defense is enabled."""
        from src.quantum_resist_random import (
            load_quantum_resist_config,
            BRANCH_PREDICTION_DEFENSE,
        )

        config = load_quantum_resist_config()
        assert config["branch_prediction_defense"] is True
        assert BRANCH_PREDICTION_DEFENSE is True

    def test_defense_mechanisms_present(self):
        """Test all 4 defense mechanisms present."""
        from src.quantum_resist_random import (
            load_quantum_resist_config,
            DEFENSE_MECHANISMS,
        )

        config = load_quantum_resist_config()
        assert "cache_partition" in config["defense_mechanisms"]
        assert "speculative_barrier" in config["defense_mechanisms"]
        assert "branch_hardening" in config["defense_mechanisms"]
        assert "timing_isolation" in config["defense_mechanisms"]
        assert len(DEFENSE_MECHANISMS) == 4


class TestKeyGeneration:
    """Tests for quantum-resistant key generation."""

    def test_key_generation(self):
        """Test key generation produces correct size."""
        from src.quantum_resist_random import generate_quantum_key

        key = generate_quantum_key(256)
        assert key is not None
        assert len(key) == 32  # 256 bits = 32 bytes

    def test_key_randomness(self):
        """Test keys are different each time."""
        from src.quantum_resist_random import generate_quantum_key

        key1 = generate_quantum_key(256)
        key2 = generate_quantum_key(256)
        assert key1 != key2


class TestSpectreDefense:
    """Tests for Spectre variant defense."""

    def test_spectre_v1_resilience(self):
        """Test Spectre v1 resilience is 1.0."""
        from src.quantum_resist_random import test_spectre_v1

        result = test_spectre_v1(100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_spectre_v2_resilience(self):
        """Test Spectre v2 resilience is 1.0."""
        from src.quantum_resist_random import test_spectre_v2

        result = test_spectre_v2(100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_spectre_v4_resilience(self):
        """Test Spectre v4 resilience is 1.0."""
        from src.quantum_resist_random import test_spectre_v4

        result = test_spectre_v4(100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_spectre_defense_combined(self):
        """Test combined Spectre defense."""
        from src.quantum_resist_random import test_spectre_defense

        result = test_spectre_defense(100)
        assert result["all_passed"] is True
        assert result["avg_resilience"] == 1.0


class TestCacheDefense:
    """Tests for cache timing defense."""

    def test_cache_timing_resilience(self):
        """Test cache timing resilience is 1.0."""
        from src.quantum_resist_random import test_cache_timing

        result = test_cache_timing(100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_cache_partition(self):
        """Test cache partitioning."""
        from src.quantum_resist_random import partition_cache

        result = partition_cache(4)
        assert result["partitions"] == 4
        assert result["effectiveness"] == 1.0


class TestDefenseMechanisms:
    """Tests for defense mechanisms."""

    def test_speculative_barrier(self):
        """Test speculative barrier insertion."""
        from src.quantum_resist_random import add_speculative_barrier

        code = ["if x == secret:", "  return data[x]"]
        result = add_speculative_barrier(code)
        assert len(result) > len(code)  # Barriers added

    def test_branch_hardening(self):
        """Test branch prediction hardening."""
        from src.quantum_resist_random import harden_branch_prediction

        code = ["if x == secret:", "  return data[x]"]
        result = harden_branch_prediction(code)
        assert len(result) >= len(code)  # Hardenings applied

    def test_timing_isolation(self):
        """Test timing isolation wrapper."""
        from src.quantum_resist_random import isolate_timing

        def dummy_op():
            return 42

        wrapped = isolate_timing(dummy_op)
        result = wrapped()
        assert result == 42


class TestQuantumResistAudit:
    """Tests for full quantum-resistant audit."""

    def test_quantum_resist_receipt(self):
        """Test quantum resist receipt emitted."""
        from src.quantum_resist_random import run_quantum_resist_audit

        result = run_quantum_resist_audit(iterations=50)
        assert result is not None
        assert result["audit_complete"] is True

    def test_overall_resilience(self):
        """Test overall resilience is 1.0."""
        from src.quantum_resist_random import (
            run_quantum_resist_audit,
            QUANTUM_RESILIENCE_TARGET,
        )

        result = run_quantum_resist_audit(iterations=50)
        assert result["overall_resilience"] >= QUANTUM_RESILIENCE_TARGET
        assert result["target_met"] is True

    def test_all_spectre_passed(self):
        """Test all Spectre variants passed."""
        from src.quantum_resist_random import run_quantum_resist_audit

        result = run_quantum_resist_audit(iterations=50)
        assert result["spectre_results"]["all_passed"] is True


class TestQuantumResistInfo:
    """Tests for quantum-resistant info."""

    def test_quantum_info(self):
        """Test quantum-resistant info retrieval."""
        from src.quantum_resist_random import get_quantum_resist_info

        info = get_quantum_resist_info()
        assert info is not None
        assert info["key_size_bits"] == 256
        assert info["resilience_target"] == 1.0
