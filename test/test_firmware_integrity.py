"""Tests for firmware_integrity.py - Supply chain verification."""

from spaceproof.domain.firmware_integrity import (
    log_source_commit,
    log_build_artifact,
    log_deployment,
    log_execution,
    verify_integrity_chain,
    emit_firmware_integrity,
    detect_supply_chain_attack,
    build_complete_chain,
    compute_tamper_detection_rate,
)
from spaceproof.core import dual_hash


class TestSourceCommit:
    """Tests for log_source_commit function."""

    def test_commit_returns_result(self):
        """Test that commit returns valid result."""
        commit = log_source_commit(
            repo_url="https://github.com/test/repo",
            commit_hash="abc123def456",
            author="developer",
        )

        assert commit.commit_id is not None
        assert commit.repo_url == "https://github.com/test/repo"
        assert commit.commit_hash == "abc123def456"
        assert commit.receipt is not None

    def test_commit_captures_timestamp(self):
        """Test that commit captures timestamp."""
        commit = log_source_commit(
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
        )

        assert commit.timestamp is not None


class TestBuildArtifact:
    """Tests for log_build_artifact function."""

    def test_build_returns_result(self):
        """Test that build returns valid result."""
        binary_hash = dual_hash(b"compiled binary")
        build = log_build_artifact(
            commit_hash="abc123",
            binary_hash=binary_hash,
            build_metadata={"version": "1.0.0", "reproducible": True},
        )

        assert build.build_id is not None
        assert build.commit_hash == "abc123"
        assert build.binary_hash == binary_hash
        assert build.receipt is not None

    def test_build_captures_metadata(self):
        """Test that build captures metadata."""
        metadata = {"compiler": "gcc", "target": "arm64"}
        build = log_build_artifact(
            commit_hash="abc",
            binary_hash="def:ghi",
            build_metadata=metadata,
        )

        assert build.build_metadata == metadata


class TestDeployment:
    """Tests for log_deployment function."""

    def test_deployment_returns_result(self):
        """Test that deployment returns valid result."""
        deployment = log_deployment(
            binary_hash="abc:def",
            satellite_id="sat-001",
        )

        assert deployment.deployment_id is not None
        assert deployment.satellite_id == "sat-001"
        assert deployment.receipt is not None

    def test_deployment_context(self):
        """Test deployment context."""
        context = {"method": "ota", "partition": "A"}
        deployment = log_deployment(
            binary_hash="abc:def",
            satellite_id="sat-001",
            deployment_context=context,
        )

        assert deployment.deployment_context == context


class TestExecution:
    """Tests for log_execution function."""

    def test_execution_returns_result(self):
        """Test that execution returns valid result."""
        execution = log_execution(
            satellite_id="sat-001",
            binary_hash="abc:def",
            execution_proof={"attestation": "tpm"},
        )

        assert execution.execution_id is not None
        assert execution.satellite_id == "sat-001"
        assert execution.receipt is not None


class TestIntegrityVerification:
    """Tests for verify_integrity_chain function."""

    def test_matching_hashes_verify(self):
        """Test that matching hashes verify."""
        result = verify_integrity_chain(
            source_hash="abc123",
            execution_hash="abc123",
        )

        assert result.verified is True
        assert result.chain_valid is True

    def test_mismatched_hashes_fail(self):
        """Test that mismatched hashes fail."""
        result = verify_integrity_chain(
            source_hash="abc123",
            execution_hash="xyz789",
        )

        assert result.verified is False
        assert len(result.mismatches) > 0

    def test_verification_time_slo(self):
        """Test verification time SLO."""
        result = verify_integrity_chain(
            source_hash="abc",
            execution_hash="abc",
        )

        assert result.verification_time_ms < 1000  # < 1 second


class TestFirmwareIntegrity:
    """Tests for emit_firmware_integrity function."""

    def test_empty_receipts(self):
        """Test empty receipts list."""
        receipt = emit_firmware_integrity([])

        assert receipt["receipt_type"] == "firmware_integrity"
        assert receipt["receipt_count"] == 0

    def test_receipts_chain(self):
        """Test receipts chain."""
        receipts = [
            {"receipt_type": "source_commit", "commit_hash": "abc"},
            {"receipt_type": "build_artifact", "binary_hash": "def"},
            {"receipt_type": "firmware_deployment", "binary_hash": "def"},
            {"receipt_type": "firmware_execution", "binary_hash": "def"},
        ]

        receipt = emit_firmware_integrity(receipts)

        assert receipt["receipt_count"] == 4
        assert receipt["integrity_verified"] is True


class TestSupplyChainAttack:
    """Tests for detect_supply_chain_attack function."""

    def test_attack_detected(self):
        """Test that attack is detected."""
        result = detect_supply_chain_attack(
            expected_binary_hash="abc:def",
            actual_binary_hash="xyz:uvw",
        )

        assert result["attack_detected"] is True
        assert result["receipt"]["severity"] == "critical"

    def test_no_attack(self):
        """Test no attack when hashes match."""
        result = detect_supply_chain_attack(
            expected_binary_hash="abc:def",
            actual_binary_hash="abc:def",
        )

        assert result["attack_detected"] is False


class TestBuildCompleteChain:
    """Tests for build_complete_chain function."""

    def test_valid_chain(self):
        """Test building valid chain."""
        source = log_source_commit(
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
        )

        binary_hash = dual_hash(b"binary")
        build = log_build_artifact(
            commit_hash="abc123",
            binary_hash=binary_hash,
            build_metadata={},
        )

        deployment = log_deployment(
            binary_hash=binary_hash,
            satellite_id="sat-001",
        )

        execution = log_execution(
            satellite_id="sat-001",
            binary_hash=binary_hash,
            execution_proof={},
        )

        chain = build_complete_chain(source, build, deployment, execution)

        assert chain.integrity_verified is True
        assert chain.merkle_supply_chain is not None

    def test_invalid_chain(self):
        """Test invalid chain with mismatched hashes."""
        source = log_source_commit(
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
        )

        build = log_build_artifact(
            commit_hash="different",  # Mismatch
            binary_hash="def:ghi",
            build_metadata={},
        )

        deployment = log_deployment(
            binary_hash="def:ghi",
            satellite_id="sat-001",
        )

        execution = log_execution(
            satellite_id="sat-001",
            binary_hash="def:ghi",
            execution_proof={},
        )

        chain = build_complete_chain(source, build, deployment, execution)

        assert chain.integrity_verified is False


class TestTamperDetectionRate:
    """Tests for compute_tamper_detection_rate function."""

    def test_all_detected(self):
        """Test 100% detection rate."""

        class MockVerification:
            verified = False
            mismatches = ["mismatch"]

        verifications = [MockVerification(), MockVerification()]
        rate = compute_tamper_detection_rate(verifications)

        assert rate == 1.0

    def test_all_verified(self):
        """Test all verified (no tampering)."""

        class MockVerification:
            verified = True
            mismatches = []

        verifications = [MockVerification(), MockVerification()]
        rate = compute_tamper_detection_rate(verifications)

        assert rate == 1.0

    def test_empty_list(self):
        """Test empty verifications list."""
        assert compute_tamper_detection_rate([]) == 0.0
