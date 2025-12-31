"""Tests for orbital_compute.py - Starcloud orbital compute provenance."""

from spaceproof.domain.orbital_compute import (
    ingest_raw_data,
    execute_inference,
    detect_radiation_anomaly,
    emit_provenance_chain,
    verify_provenance,
    compute_effectiveness,
    RADIATION_ENTROPY_THRESHOLD,
)


class TestIngestRawData:
    """Tests for ingest_raw_data function."""

    def test_ingest_returns_result(self):
        """Test that ingest returns a valid IngestResult."""
        sensor_data = b"test sensor data bytes"
        result = ingest_raw_data(sensor_data, "sat-001")

        assert result.satellite_id == "sat-001"
        assert result.input_hash is not None
        assert ":" in result.input_hash  # Dual-hash format
        assert result.data_size_bytes == len(sensor_data)
        assert result.receipt is not None

    def test_ingest_emits_receipt(self):
        """Test that ingest emits proper receipt."""
        result = ingest_raw_data(b"data", "sat-002")

        assert result.receipt["receipt_type"] == "data_ingest"
        assert "payload_hash" in result.receipt
        assert result.receipt["satellite_id"] == "sat-002"


class TestExecuteInference:
    """Tests for execute_inference function."""

    def test_execute_inference_returns_result(self):
        """Test that inference returns valid result."""
        result = execute_inference(
            input_hash="abc123:def456",
            model_id="vision-v1",
            inference_result={"classification": "normal"},
            satellite_id="sat-001",
        )

        assert result.model_id == "vision-v1"
        assert result.output_hash is not None
        assert result.receipt is not None

    def test_execute_inference_captures_entropy(self):
        """Test that inference captures entropy metrics."""
        result = execute_inference(
            input_hash="abc123:def456",
            model_id="vision-v1",
            inference_result={"data": [1, 2, 3]},
            input_entropy=0.5,
        )

        assert result.entropy_input == 0.5
        assert result.entropy_output is not None
        assert result.entropy_delta is not None


class TestRadiationDetection:
    """Tests for detect_radiation_anomaly function."""

    def test_detects_radiation_spike(self):
        """Test that radiation spike is detected."""
        # Large deviation should trigger detection
        result = detect_radiation_anomaly(
            expected_entropy=0.5,
            actual_entropy=0.75,  # 50% deviation > 15% threshold
            threshold=0.15,
        )

        assert result.detected is True
        assert result.deviation > RADIATION_ENTROPY_THRESHOLD

    def test_normal_entropy_not_flagged(self):
        """Test that normal entropy is not flagged as radiation."""
        result = detect_radiation_anomaly(
            expected_entropy=0.5,
            actual_entropy=0.52,  # 4% deviation < 15% threshold
            threshold=0.15,
        )

        assert result.detected is False

    def test_custom_threshold(self):
        """Test custom threshold works."""
        result = detect_radiation_anomaly(
            expected_entropy=0.5,
            actual_entropy=0.55,
            threshold=0.05,  # Stricter threshold
        )

        assert result.detected is True


class TestProvenanceChain:
    """Tests for emit_provenance_chain function."""

    def test_empty_chain(self):
        """Test empty receipts list."""
        chain = emit_provenance_chain([], satellite_id="sat-001")

        assert chain.satellite_id == "sat-001"
        assert chain.merkle_anchor is not None
        assert chain.receipt_count == 0

    def test_chain_with_receipts(self):
        """Test chain with receipts."""
        receipts = [
            {"receipt_type": "ingest", "data": "test1"},
            {"receipt_type": "inference", "data": "test2"},
        ]
        chain = emit_provenance_chain(receipts, satellite_id="sat-001")

        assert chain.receipt_count == 2
        assert chain.merkle_anchor is not None

    def test_chain_tracks_radiation_events(self):
        """Test that chain tracks radiation events."""
        receipts = [
            {"receipt_type": "radiation", "radiation_detected": True},
            {"receipt_type": "inference", "radiation_detected": False},
        ]
        chain = emit_provenance_chain(receipts)

        assert chain.radiation_events == 1


class TestVerifyProvenance:
    """Tests for verify_provenance function."""

    def test_valid_chain_verifies(self):
        """Test that valid chain verifies."""
        receipts = [{"test": "data"}]
        chain = emit_provenance_chain(receipts)

        assert verify_provenance(chain) is True


class TestComputeEffectiveness:
    """Tests for compute_effectiveness function."""

    def test_effectiveness_calculation(self):
        """Test effectiveness calculation."""
        receipts = [
            {"entropy_input": 0.8, "entropy_output": 0.3},
            {"entropy_input": 0.7, "entropy_output": 0.2},
        ]
        effectiveness = compute_effectiveness(receipts)

        assert 0 <= effectiveness <= 1
        assert effectiveness > 0  # Should have positive reduction

    def test_empty_receipts(self):
        """Test empty receipts returns zero."""
        assert compute_effectiveness([]) == 0.0

    def test_no_entropy_data(self):
        """Test receipts without entropy data."""
        receipts = [{"other": "data"}]
        assert compute_effectiveness(receipts) == 0.0
