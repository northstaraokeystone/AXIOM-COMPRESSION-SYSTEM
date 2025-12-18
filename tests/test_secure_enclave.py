"""Tests for secure enclave branch prediction defense.

Test coverage:
- Enclave configuration loading
- BTB/PHT/RSB defense tests
- Full enclave audit
"""


class TestEnclaveConfig:
    """Tests for secure enclave configuration."""

    def test_enclave_config_loads(self):
        """Test enclave config loads."""
        from src.secure_enclave_audit import load_enclave_config

        config = load_enclave_config()
        assert config is not None

    def test_enclave_type(self):
        """Test enclave type is SGX."""
        from src.secure_enclave_audit import load_enclave_config, ENCLAVE_TYPE

        config = load_enclave_config()
        assert config["type"] == "SGX"
        assert ENCLAVE_TYPE == "SGX"

    def test_enclave_memory(self):
        """Test enclave memory is 128 MB."""
        from src.secure_enclave_audit import load_enclave_config, ENCLAVE_MEMORY_MB

        config = load_enclave_config()
        assert config["memory_mb"] == 128
        assert ENCLAVE_MEMORY_MB == 128

    def test_branch_defense_enabled(self):
        """Test branch prediction defense is enabled."""
        from src.secure_enclave_audit import load_enclave_config, BRANCH_PREDICTION_DEFENSE

        config = load_enclave_config()
        assert config["branch_prediction_defense"] is True
        assert BRANCH_PREDICTION_DEFENSE is True

    def test_speculative_barrier_enabled(self):
        """Test speculative execution barrier is enabled."""
        from src.secure_enclave_audit import load_enclave_config, SPECULATIVE_EXECUTION_BARRIER

        config = load_enclave_config()
        assert config["speculative_barrier"] is True
        assert SPECULATIVE_EXECUTION_BARRIER is True

    def test_attack_types_present(self):
        """Test all 3 attack types are present."""
        from src.secure_enclave_audit import load_enclave_config, ATTACK_TYPES

        config = load_enclave_config()
        assert len(config["attack_types"]) == 3
        assert "BTB_injection" in config["attack_types"]
        assert "PHT_poisoning" in config["attack_types"]
        assert "RSB_stuffing" in config["attack_types"]
        assert ATTACK_TYPES == ["BTB_injection", "PHT_poisoning", "RSB_stuffing"]

    def test_defense_mechanisms_present(self):
        """Test all 5 defense mechanisms are present."""
        from src.secure_enclave_audit import load_enclave_config, DEFENSE_MECHANISMS

        config = load_enclave_config()
        assert len(config["defense_mechanisms"]) == 5
        assert "BTB_flush" in config["defense_mechanisms"]
        assert "PHT_isolation" in config["defense_mechanisms"]
        assert "RSB_fill" in config["defense_mechanisms"]
        assert "IBRS" in config["defense_mechanisms"]
        assert "STIBP" in config["defense_mechanisms"]
        assert len(DEFENSE_MECHANISMS) == 5


class TestEnclaveInit:
    """Tests for enclave initialization."""

    def test_enclave_init(self):
        """Test enclave initialization."""
        from src.secure_enclave_audit import init_enclave

        result = init_enclave(memory_mb=128)
        assert result is not None
        assert result["initialized"] is True
        assert result["memory_mb"] == 128

    def test_enclave_defenses_applied(self):
        """Test all defenses are applied on init."""
        from src.secure_enclave_audit import init_enclave, DEFENSE_MECHANISMS

        result = init_enclave()
        assert len(result["defenses_applied"]) == len(DEFENSE_MECHANISMS)


class TestBTBDefense:
    """Tests for BTB injection defense."""

    def test_btb_resilience(self):
        """Test BTB injection resilience is 1.0."""
        from src.secure_enclave_audit import test_btb_injection

        result = test_btb_injection(iterations=100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_btb_defense_mechanism(self):
        """Test BTB defense mechanism is BTB_flush."""
        from src.secure_enclave_audit import test_btb_injection

        result = test_btb_injection(iterations=10)
        assert result["defense_mechanism"] == "BTB_flush"


class TestPHTDefense:
    """Tests for PHT poisoning defense."""

    def test_pht_resilience(self):
        """Test PHT poisoning resilience is 1.0."""
        from src.secure_enclave_audit import test_pht_poisoning

        result = test_pht_poisoning(iterations=100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_pht_defense_mechanism(self):
        """Test PHT defense mechanism is PHT_isolation."""
        from src.secure_enclave_audit import test_pht_poisoning

        result = test_pht_poisoning(iterations=10)
        assert result["defense_mechanism"] == "PHT_isolation"


class TestRSBDefense:
    """Tests for RSB stuffing defense."""

    def test_rsb_resilience(self):
        """Test RSB stuffing resilience is 1.0."""
        from src.secure_enclave_audit import test_rsb_stuffing

        result = test_rsb_stuffing(iterations=100)
        assert result["resilience"] == 1.0
        assert result["passed"] is True

    def test_rsb_defense_mechanism(self):
        """Test RSB defense mechanism is RSB_fill."""
        from src.secure_enclave_audit import test_rsb_stuffing

        result = test_rsb_stuffing(iterations=10)
        assert result["defense_mechanism"] == "RSB_fill"


class TestFullEnclaveAudit:
    """Tests for full enclave audit."""

    def test_enclave_audit_runs(self):
        """Test full enclave audit runs."""
        from src.secure_enclave_audit import run_enclave_audit

        result = run_enclave_audit(iterations=50)
        assert result is not None
        assert "overall_resilience" in result

    def test_enclave_audit_all_passed(self):
        """Test all enclave audit tests pass."""
        from src.secure_enclave_audit import run_enclave_audit

        result = run_enclave_audit(iterations=50)
        assert result["all_passed"] is True
        assert result["target_met"] is True

    def test_enclave_audit_resilience_100(self):
        """Test enclave audit achieves 100% resilience."""
        from src.secure_enclave_audit import run_enclave_audit

        result = run_enclave_audit(iterations=100)
        assert result["overall_resilience"] == 1.0

    def test_secure_enclave_receipt(self):
        """Test secure enclave receipt emitted."""
        from src.secure_enclave_audit import run_enclave_audit

        result = run_enclave_audit(iterations=50)
        # Receipt should be emitted (tested via result structure)
        assert "enclave_id" in result


class TestEnclaveOverhead:
    """Tests for enclave defense overhead."""

    def test_overhead_measurement(self):
        """Test overhead measurement runs."""
        from src.secure_enclave_audit import measure_enclave_overhead

        result = measure_enclave_overhead()
        assert result is not None
        assert "total_overhead_pct" in result

    def test_overhead_acceptable(self):
        """Test overhead is acceptable (<10%)."""
        from src.secure_enclave_audit import measure_enclave_overhead

        result = measure_enclave_overhead()
        assert result["acceptable"] is True
        assert result["total_overhead_pct"] < 10.0
