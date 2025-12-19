"""Tests for PLONK ZK proof system upgrade."""


class TestPLONKConstants:
    """Tests for PLONK ZK constants."""

    def test_plonk_circuit_size(self) -> None:
        """Test PLONK circuit size constant."""
        from src.plonk_zk_upgrade import PLONK_CIRCUIT_SIZE

        assert PLONK_CIRCUIT_SIZE == 2**22

    def test_plonk_timing_constants(self) -> None:
        """Test PLONK timing constants."""
        from src.plonk_zk_upgrade import (
            PLONK_PROOF_TIME_MS,
            PLONK_VERIFY_TIME_MS,
        )

        assert PLONK_PROOF_TIME_MS == 200
        assert PLONK_VERIFY_TIME_MS == 5

    def test_plonk_feature_constants(self) -> None:
        """Test PLONK feature constants."""
        from src.plonk_zk_upgrade import (
            PLONK_UNIVERSAL_SETUP,
            PLONK_RECURSION_CAPABLE,
            PLONK_PRIVACY_PRESERVING,
        )

        assert PLONK_UNIVERSAL_SETUP is True
        assert PLONK_RECURSION_CAPABLE is True
        assert PLONK_PRIVACY_PRESERVING is True

    def test_plonk_resilience_target(self) -> None:
        """Test PLONK resilience target."""
        from src.plonk_zk_upgrade import PLONK_RESILIENCE_TARGET

        assert PLONK_RESILIENCE_TARGET == 1.0


class TestPLONKConfig:
    """Tests for PLONK configuration loading."""

    def test_load_plonk_config(self) -> None:
        """Test loading PLONK config from spec."""
        from src.plonk_zk_upgrade import load_plonk_config

        config = load_plonk_config()
        assert config is not None
        assert "proof_system" in config
        assert config["proof_system"] == "plonk"
        assert "universal_setup" in config
        assert config["universal_setup"] is True

    def test_get_plonk_info(self) -> None:
        """Test PLONK info retrieval."""
        from src.plonk_zk_upgrade import get_plonk_info

        info = get_plonk_info()
        assert info is not None
        assert "proof_system" in info
        assert info["proof_system"] == "plonk"
        assert "circuit_size" in info
        assert "proof_time_ms" in info
        assert "verify_time_ms" in info
        assert "universal_setup" in info
        assert "recursion_capable" in info


class TestUniversalSetup:
    """Tests for PLONK universal trusted setup."""

    def test_universal_setup_basic(self) -> None:
        """Test basic universal setup."""
        from src.plonk_zk_upgrade import universal_setup

        result = universal_setup(participants=10)

        assert result is not None
        assert "setup_type" in result
        assert result["setup_type"] == "universal"
        assert "max_circuit_size" in result
        assert "participants" in result
        assert result["participants"] == 10

    def test_universal_setup_keys(self) -> None:
        """Test universal setup key generation."""
        from src.plonk_zk_upgrade import universal_setup

        result = universal_setup(participants=10)

        assert "urs_hash" in result
        assert "verification_key_hash" in result
        assert len(result["urs_hash"]) == 64  # SHA256 hex
        assert len(result["verification_key_hash"]) == 64

    def test_universal_setup_toxic_waste(self) -> None:
        """Test toxic waste destruction."""
        from src.plonk_zk_upgrade import universal_setup

        result = universal_setup(participants=10)

        assert "toxic_waste_destroyed" in result
        assert result["toxic_waste_destroyed"] is True

    def test_universal_setup_complete(self) -> None:
        """Test universal setup completion status."""
        from src.plonk_zk_upgrade import universal_setup

        result = universal_setup(participants=10)

        assert "universal_setup_complete" in result
        assert result["universal_setup_complete"] is True
        assert "any_circuit_supported" in result
        assert result["any_circuit_supported"] is True


class TestPLONKCircuit:
    """Tests for PLONK circuit generation."""

    def test_generate_plonk_circuit(self) -> None:
        """Test PLONK circuit generation."""
        from src.plonk_zk_upgrade import generate_plonk_circuit

        circuit = generate_plonk_circuit()

        assert circuit is not None
        assert "proof_system" in circuit
        assert circuit["proof_system"] == "plonk"
        assert "constraints" in circuit
        assert "gates" in circuit

    def test_circuit_claims(self) -> None:
        """Test circuit attestation claims."""
        from src.plonk_zk_upgrade import generate_plonk_circuit

        circuit = generate_plonk_circuit()

        assert "claims" in circuit
        claims = circuit["claims"]
        assert "enclave_id" in claims
        assert "code_hash" in claims
        assert "config_hash" in claims


class TestPLONKProof:
    """Tests for PLONK proof generation and verification."""

    def test_generate_plonk_proof(self) -> None:
        """Test PLONK proof generation."""
        from src.plonk_zk_upgrade import generate_plonk_circuit, generate_plonk_proof

        circuit = generate_plonk_circuit()
        witness = {
            "enclave_id_private": "test_enclave",
            "enclave_id_commitment_public": "abc123",
            "code_hash_private": "code_hash",
            "code_hash_commitment_public": "def456",
            "config_hash_private": "config_hash",
            "config_hash_commitment_public": "ghi789",
            "timestamp_public": "2024-01-01T00:00:00Z",
            "recursion_depth_public": "0",
        }

        proof = generate_plonk_proof(circuit, witness)

        assert proof is not None
        assert "proof_system" in proof
        assert proof["proof_system"] == "plonk"
        assert "proof_a" in proof
        assert "proof_b" in proof
        assert "proof_c" in proof
        assert "proof_z" in proof

    def test_plonk_proof_size(self) -> None:
        """Test PLONK proof size."""
        from src.plonk_zk_upgrade import generate_plonk_circuit, generate_plonk_proof

        circuit = generate_plonk_circuit()
        witness = {
            "enclave_id_private": "test",
            "enclave_id_commitment_public": "abc",
            "code_hash_private": "hash",
            "code_hash_commitment_public": "def",
            "config_hash_private": "conf",
            "config_hash_commitment_public": "ghi",
            "timestamp_public": "2024-01-01T00:00:00Z",
            "recursion_depth_public": "0",
        }

        proof = generate_plonk_proof(circuit, witness)

        assert "proof_size_bytes" in proof
        # PLONK proofs are compact
        assert proof["proof_size_bytes"] < 1000

    def test_verify_plonk_proof(self) -> None:
        """Test PLONK proof verification."""
        from src.plonk_zk_upgrade import verify_plonk

        result = verify_plonk()

        assert result is not None
        assert "valid" in result
        assert result["valid"] is True
        assert "proof_system" in result
        assert result["proof_system"] == "plonk"

    def test_verify_plonk_timing(self) -> None:
        """Test PLONK verification timing."""
        from src.plonk_zk_upgrade import verify_plonk

        result = verify_plonk()

        assert "verification_time_ms" in result
        # Verification should be fast
        assert result["verification_time_ms"] <= 10


class TestRecursiveProof:
    """Tests for PLONK recursive proofs."""

    def test_recursive_proof_basic(self) -> None:
        """Test basic recursive proof generation."""
        from src.plonk_zk_upgrade import recursive_proof

        proofs = [
            {"proof_a": "a1", "proof_b": "b1"},
            {"proof_a": "a2", "proof_b": "b2"},
            {"proof_a": "a3", "proof_b": "b3"},
        ]

        result = recursive_proof(proofs)

        assert result is not None
        assert "proofs_aggregated" in result
        assert result["proofs_aggregated"] == 3
        assert "merkle_root" in result
        assert "valid" in result

    def test_recursive_proof_compression(self) -> None:
        """Test recursive proof compression ratio."""
        from src.plonk_zk_upgrade import recursive_proof

        proofs = [{"proof_a": f"a{i}", "proof_b": f"b{i}"} for i in range(5)]

        result = recursive_proof(proofs)

        assert "compression_ratio" in result
        # Should achieve significant compression
        assert result["compression_ratio"] >= 2

    def test_recursive_proof_components(self) -> None:
        """Test recursive proof components."""
        from src.plonk_zk_upgrade import recursive_proof

        proofs = [{"proof_a": "a1", "proof_b": "b1"}]

        result = recursive_proof(proofs)

        assert "recursive_proof_a" in result
        assert "recursive_proof_z" in result


class TestPLONKAttestation:
    """Tests for PLONK attestation creation."""

    def test_create_plonk_attestation(self) -> None:
        """Test PLONK attestation creation."""
        from src.plonk_zk_upgrade import create_plonk_attestation

        attestation = create_plonk_attestation(
            enclave_id="test_enclave",
            code_hash="test_code",
            config_hash="test_config",
        )

        assert attestation is not None
        assert "version" in attestation
        assert "proof_system" in attestation
        assert attestation["proof_system"] == "plonk"
        assert "circuit" in attestation
        assert "public_inputs" in attestation
        assert "proof" in attestation

    def test_attestation_public_inputs(self) -> None:
        """Test attestation public inputs."""
        from src.plonk_zk_upgrade import create_plonk_attestation

        attestation = create_plonk_attestation(
            enclave_id="test_enclave",
            code_hash="test_code",
            config_hash="test_config",
        )

        public_inputs = attestation["public_inputs"]
        assert "enclave_id_commitment" in public_inputs
        assert "code_hash_commitment" in public_inputs
        assert "config_hash_commitment" in public_inputs
        assert "timestamp" in public_inputs

    def test_attestation_metadata(self) -> None:
        """Test attestation metadata."""
        from src.plonk_zk_upgrade import create_plonk_attestation

        attestation = create_plonk_attestation(
            enclave_id="test_enclave",
            code_hash="test_code",
            config_hash="test_config",
        )

        assert "metadata" in attestation
        metadata = attestation["metadata"]
        assert "created_at" in metadata
        assert "proof_time_ms" in metadata
        assert "universal_setup" in metadata
        assert "recursion_capable" in metadata

    def test_verify_plonk_attestation(self) -> None:
        """Test PLONK attestation verification."""
        from src.plonk_zk_upgrade import (
            create_plonk_attestation,
            verify_plonk_attestation,
        )

        attestation = create_plonk_attestation(
            enclave_id="test_enclave",
            code_hash="test_code",
            config_hash="test_config",
        )

        result = verify_plonk_attestation(attestation)

        assert result is not None
        assert "valid" in result
        assert result["valid"] is True


class TestPLONKBenchmark:
    """Tests for PLONK benchmarking."""

    def test_benchmark_plonk(self) -> None:
        """Test PLONK benchmark."""
        from src.plonk_zk_upgrade import benchmark_plonk

        result = benchmark_plonk(iterations=5)

        assert result is not None
        assert "proof_system" in result
        assert result["proof_system"] == "plonk"
        assert "iterations" in result
        assert result["iterations"] == 5

    def test_benchmark_proof_time(self) -> None:
        """Test benchmark proof time stats."""
        from src.plonk_zk_upgrade import benchmark_plonk

        result = benchmark_plonk(iterations=5)

        assert "proof_time_ms" in result
        proof_time = result["proof_time_ms"]
        assert "min" in proof_time
        assert "max" in proof_time
        assert "avg" in proof_time

    def test_benchmark_verify_time(self) -> None:
        """Test benchmark verify time stats."""
        from src.plonk_zk_upgrade import benchmark_plonk

        result = benchmark_plonk(iterations=5)

        assert "verify_time_ms" in result
        verify_time = result["verify_time_ms"]
        assert "min" in verify_time
        assert "max" in verify_time
        assert "avg" in verify_time

    def test_benchmark_throughput(self) -> None:
        """Test benchmark throughput metrics."""
        from src.plonk_zk_upgrade import benchmark_plonk

        result = benchmark_plonk(iterations=5)

        assert "throughput_proofs_per_sec" in result
        assert "throughput_verifies_per_sec" in result


class TestPLONKComparison:
    """Tests for PLONK vs Groth16 comparison."""

    def test_compare_to_groth16(self) -> None:
        """Test PLONK vs Groth16 comparison."""
        from src.plonk_zk_upgrade import compare_to_groth16

        result = compare_to_groth16()

        assert result is not None
        assert "plonk" in result
        assert "groth16" in result
        assert "comparison" in result
        assert "recommendation" in result

    def test_comparison_plonk_section(self) -> None:
        """Test comparison PLONK section."""
        from src.plonk_zk_upgrade import compare_to_groth16

        result = compare_to_groth16()

        plonk = result["plonk"]
        assert "circuit_size" in plonk
        assert "proof_time_ms" in plonk
        assert "verify_time_ms" in plonk
        assert "universal_setup" in plonk
        assert plonk["universal_setup"] is True

    def test_comparison_groth16_section(self) -> None:
        """Test comparison Groth16 section."""
        from src.plonk_zk_upgrade import compare_to_groth16

        result = compare_to_groth16()

        groth16 = result["groth16"]
        assert "circuit_size" in groth16
        assert "proof_time_ms" in groth16
        assert "verify_time_ms" in groth16
        assert "per_circuit_setup" in groth16

    def test_comparison_speedup(self) -> None:
        """Test comparison speedup metrics."""
        from src.plonk_zk_upgrade import compare_to_groth16

        result = compare_to_groth16()

        comparison = result["comparison"]
        assert "proof_speedup" in comparison
        assert "verify_speedup" in comparison
        assert "plonk_advantages" in comparison


class TestPLONKAudit:
    """Tests for PLONK audit functionality."""

    def test_run_plonk_audit(self) -> None:
        """Test full PLONK audit."""
        from src.plonk_zk_upgrade import run_plonk_audit

        result = run_plonk_audit(attestation_count=3)

        assert result is not None
        assert "proof_system" in result
        assert result["proof_system"] == "plonk"
        assert "attestations_created" in result
        assert result["attestations_created"] == 3

    def test_audit_verification_rate(self) -> None:
        """Test audit verification rate."""
        from src.plonk_zk_upgrade import run_plonk_audit

        result = run_plonk_audit(attestation_count=5)

        assert "verifications_passed" in result
        assert "verification_rate" in result
        # Should have 100% verification rate
        assert result["verification_rate"] == 1.0

    def test_audit_resilience(self) -> None:
        """Test audit resilience metrics."""
        from src.plonk_zk_upgrade import run_plonk_audit

        result = run_plonk_audit(attestation_count=5)

        assert "resilience" in result
        assert "resilience_target" in result
        assert "resilience_target_met" in result

    def test_audit_recursive_proof(self) -> None:
        """Test audit includes recursive proof."""
        from src.plonk_zk_upgrade import run_plonk_audit

        result = run_plonk_audit(attestation_count=3)

        assert "recursive_proof_valid" in result
        assert "recursive_compression_ratio" in result

    def test_audit_overall_validation(self) -> None:
        """Test audit overall validation."""
        from src.plonk_zk_upgrade import run_plonk_audit

        result = run_plonk_audit(attestation_count=5)

        assert "overall_validated" in result
        assert result["overall_validated"] is True


class TestAGIPLONKIntegration:
    """Tests for AGI path PLONK integration."""

    def test_integrate_plonk(self) -> None:
        """Test PLONK integration into AGI path."""
        from src.paths.agi.core import integrate_plonk

        result = integrate_plonk()

        assert result is not None
        assert "proof_system" in result
        assert result["proof_system"] == "plonk"
        assert "integrated" in result

    def test_run_plonk_stress_test(self) -> None:
        """Test PLONK stress test."""
        from src.paths.agi.core import run_plonk_stress_test

        result = run_plonk_stress_test(iterations=5)

        assert result is not None
        assert "iterations" in result
        assert "stress_passed" in result

    def test_compare_zk_systems(self) -> None:
        """Test ZK systems comparison."""
        from src.paths.agi.core import compare_zk_systems

        result = compare_zk_systems()

        assert result is not None
        assert "plonk" in result
        assert "groth16" in result
        assert "comparison" in result
        assert "recommendation" in result

    def test_measure_plonk_overhead(self) -> None:
        """Test PLONK overhead measurement."""
        from src.paths.agi.core import measure_plonk_overhead

        result = measure_plonk_overhead()

        assert result is not None
        assert "plonk_actual" in result
        assert "speedup" in result
        assert "overall_improvement" in result

    def test_compute_plonk_alignment(self) -> None:
        """Test PLONK alignment score computation."""
        from src.paths.agi.core import compute_plonk_alignment

        result = compute_plonk_alignment()

        assert result is not None
        assert "plonk_resilience" in result
        assert "enhanced_alignment" in result
        assert "is_aligned" in result
