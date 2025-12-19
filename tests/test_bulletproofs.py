"""Tests for Bulletproofs infinite proof chain stress testing.

Tests:
- Config loading
- Proof generation
- Proof verification
- Proof aggregation
- Stress testing
- Infinite chain generation
"""

from src.bulletproofs_infinite import (
    load_bulletproofs_config,
    generate_bulletproof_circuit,
    generate_bulletproof,
    verify_bulletproof,
    aggregate_bulletproofs,
    stress_test,
    generate_infinite_chain,
    run_bulletproofs_audit,
    BULLETPROOFS_PROOF_SIZE,
    BULLETPROOFS_VERIFY_TIME_MS,
    BULLETPROOFS_STRESS_DEPTH,
    BULLETPROOFS_NO_TRUSTED_SETUP,
)


class TestBulletproofsConfig:
    """Tests for Bulletproofs configuration."""

    def test_config_loads(self):
        """Config loads successfully."""
        config = load_bulletproofs_config()
        assert config is not None
        assert "proof_size_bytes" in config

    def test_proof_size(self):
        """Proof size is 672 bytes."""
        config = load_bulletproofs_config()
        assert config["proof_size_bytes"] == 672
        assert BULLETPROOFS_PROOF_SIZE == 672

    def test_verify_time(self):
        """Verify time is 2 ms."""
        config = load_bulletproofs_config()
        assert config["verify_time_ms"] == 2
        assert BULLETPROOFS_VERIFY_TIME_MS == 2

    def test_stress_depth(self):
        """Stress depth is 1000."""
        config = load_bulletproofs_config()
        assert config["stress_depth"] == 1000
        assert BULLETPROOFS_STRESS_DEPTH == 1000

    def test_no_trusted_setup(self):
        """No trusted setup required."""
        config = load_bulletproofs_config()
        assert config["no_trusted_setup"] is True
        assert BULLETPROOFS_NO_TRUSTED_SETUP is True


class TestBulletproofCircuit:
    """Tests for circuit generation."""

    def test_circuit_generates(self):
        """Circuit generates successfully."""
        circuit = generate_bulletproof_circuit()
        assert circuit is not None
        assert "gates" in circuit
        assert "constraints" in circuit

    def test_circuit_properties(self):
        """Circuit has correct properties."""
        circuit = generate_bulletproof_circuit()
        assert circuit["gates"] > 0
        assert circuit["constraints"] > 0


class TestBulletproofGeneration:
    """Tests for proof generation."""

    def test_proof_generates(self):
        """Proof generates successfully."""
        proof = generate_bulletproof(value=42)
        assert proof is not None
        assert "proof" in proof

    def test_proof_size(self):
        """Proof has correct size."""
        proof = generate_bulletproof(value=42)
        # Proof should be around 672 bytes (logarithmic)
        assert "proof_size" in proof
        assert proof["proof_size"] <= BULLETPROOFS_PROOF_SIZE * 2

    def test_proof_commitment(self):
        """Proof includes commitment."""
        proof = generate_bulletproof(value=42)
        assert "commitment" in proof

    def test_proof_range(self):
        """Proof works for range proof."""
        # Bulletproofs excel at range proofs
        proof = generate_bulletproof(value=100, range_bits=64)
        assert proof is not None
        assert "range_bits" in proof
        assert proof["range_bits"] == 64


class TestBulletproofVerification:
    """Tests for proof verification."""

    def test_valid_proof_verifies(self):
        """Valid proof verifies."""
        proof = generate_bulletproof(value=42)
        result = verify_bulletproof(proof)

        assert result["valid"] is True

    def test_verification_time(self):
        """Verification is fast."""
        proof = generate_bulletproof(value=42)
        result = verify_bulletproof(proof)

        assert "verify_time_ms" in result
        # Should be close to target
        assert result["verify_time_ms"] < BULLETPROOFS_VERIFY_TIME_MS * 10

    def test_invalid_proof_fails(self):
        """Tampered proof fails verification."""
        proof = generate_bulletproof(value=42)
        # Tamper with proof
        proof["proof"] = "tampered"
        result = verify_bulletproof(proof)

        assert result["valid"] is False


class TestBulletproofAggregation:
    """Tests for proof aggregation."""

    def test_aggregate_proofs(self):
        """Proofs aggregate successfully."""
        proofs = [generate_bulletproof(value=i) for i in range(5)]
        aggregated = aggregate_bulletproofs(proofs)

        assert aggregated is not None
        assert "aggregated_proof" in aggregated

    def test_aggregation_saves_space(self):
        """Aggregation reduces total size."""
        proofs = [generate_bulletproof(value=i) for i in range(10)]
        aggregated = aggregate_bulletproofs(proofs)

        individual_size = sum(p["proof_size"] for p in proofs)
        aggregated_size = aggregated["total_size"]

        # Aggregation should save space (logarithmic)
        assert aggregated_size < individual_size

    def test_aggregated_verifies(self):
        """Aggregated proof verifies."""
        proofs = [generate_bulletproof(value=i) for i in range(5)]
        aggregated = aggregate_bulletproofs(proofs)

        result = verify_bulletproof(aggregated["aggregated_proof"])
        assert result["valid"] is True


class TestBulletproofStressTest:
    """Tests for stress testing."""

    def test_stress_test_runs(self):
        """Stress test runs."""
        result = stress_test(depth=10)

        assert "depth" in result
        assert "proofs_generated" in result

    def test_stress_all_verified(self):
        """All stress proofs verify."""
        result = stress_test(depth=10)

        assert result["all_valid"] is True

    def test_stress_performance(self):
        """Stress test reports performance."""
        result = stress_test(depth=10)

        assert "avg_verify_time_ms" in result
        assert "total_time_ms" in result

    def test_stress_aggregation(self):
        """Stress test includes aggregation."""
        result = stress_test(depth=10)

        assert "aggregation_tested" in result
        assert result["aggregation_tested"] is True


class TestBulletproofInfiniteChain:
    """Tests for infinite proof chain."""

    def test_chain_generates(self):
        """Infinite chain generates."""
        result = generate_infinite_chain(depth=5)

        assert "chain_depth" in result
        assert "proofs_in_chain" in result

    def test_chain_valid(self):
        """Chain is valid."""
        result = generate_infinite_chain(depth=5)

        assert result["chain_valid"] is True

    def test_chain_size(self):
        """Chain has expected size."""
        result = generate_infinite_chain(depth=5)

        assert "total_size_bytes" in result
        # Logarithmic scaling
        assert result["total_size_bytes"] < 5 * BULLETPROOFS_PROOF_SIZE * 10

    def test_chain_verify_time(self):
        """Chain verify time is reasonable."""
        result = generate_infinite_chain(depth=5)

        assert "total_verify_time_ms" in result


class TestBulletproofAudit:
    """Tests for audit functionality."""

    def test_audit_runs(self):
        """Audit runs successfully."""
        result = run_bulletproofs_audit()

        assert "audit_complete" in result
        assert result["audit_complete"] is True

    def test_audit_checks(self):
        """Audit performs required checks."""
        result = run_bulletproofs_audit()

        assert "proof_generation" in result
        assert "proof_verification" in result
        assert "aggregation" in result
        assert "stress_test" in result

    def test_audit_receipt(self):
        """Audit emits receipt."""
        result = run_bulletproofs_audit()

        assert "receipt" in result
        receipt = result["receipt"]
        assert "timestamp" in receipt
        assert "payload_hash" in receipt
