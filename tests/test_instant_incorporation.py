"""Tests for D19.3 Instant Incorporator.

Instant incorporation: Real-time oracle update on receipt arrival.
Incorporation latency must be < 100ms. No batch processing.
"""


class TestInstantIncorporator:
    """Tests for InstantIncorporator functionality."""

    def test_init_incorporator(self):
        """Test incorporator initialization."""
        from src.oracle import init_incorporator

        incorporator = init_incorporator()

        assert incorporator is not None
        assert incorporator.incorporator_id is not None
        assert len(incorporator.incorporator_id) == 8
        assert incorporator.incorporation_count == 0

    def test_on_receipt_arrival(self):
        """Test receipt incorporation."""
        from src.oracle import init_oracle, init_incorporator, on_receipt_arrival

        oracle = init_oracle({})
        oracle.history = []
        incorporator = init_incorporator(oracle)

        test_receipt = {
            "receipt_type": "test",
            "ts": "2024-01-01T00:00:00Z",
            "payload_hash": "test_hash",
        }

        result = on_receipt_arrival(incorporator, test_receipt)

        assert result["incorporated"] is True
        assert "latency_ms" in result
        assert "latency_ok" in result
        assert incorporator.incorporation_count == 1

    def test_incorporation_latency(self):
        """Test incorporation latency constraint (<100ms)."""
        from src.oracle import init_oracle, init_incorporator, on_receipt_arrival
        from src.oracle.instant_incorporator import INCORPORATION_LATENCY_MAX_MS

        assert INCORPORATION_LATENCY_MAX_MS == 100

        oracle = init_oracle({})
        oracle.history = []
        incorporator = init_incorporator(oracle)

        test_receipt = {
            "receipt_type": "test",
            "ts": "2024-01-01T00:00:00Z",
        }

        result = on_receipt_arrival(incorporator, test_receipt)

        # Latency should be under 100ms
        assert result["latency_ms"] < INCORPORATION_LATENCY_MAX_MS
        assert result["latency_ok"] is True

    def test_update_compression(self):
        """Test compression update on new receipt."""
        from src.oracle import init_oracle, init_incorporator
        from src.oracle.instant_incorporator import update_compression

        oracle = init_oracle({})
        oracle.history = [{"receipt_type": "a"} for _ in range(10)]
        incorporator = init_incorporator(oracle)

        new_receipt = {"receipt_type": "b"}
        new_compression = update_compression(incorporator, new_receipt)

        assert isinstance(new_compression, float)
        assert 0.0 <= new_compression <= 1.0

    def test_check_law_survival(self):
        """Test law survival check on new receipt."""
        from src.oracle import init_oracle, init_incorporator
        from src.oracle.instant_incorporator import check_law_survival

        oracle = init_oracle({})
        oracle.laws = [
            {"law_id": "law1", "receipt_types": ["a"]},
            {"law_id": "law2", "receipt_types": ["b"]},
        ]
        incorporator = init_incorporator(oracle)

        new_receipt = {"receipt_type": "a"}
        surviving = check_law_survival(incorporator, new_receipt)

        assert isinstance(surviving, list)

    def test_emit_incorporation_receipt(self):
        """Test incorporation receipt emission."""
        from src.oracle import init_incorporator, emit_incorporation_receipt

        incorporator = init_incorporator()
        incorporator.last_incorporation_latency_ms = 50.0

        receipt = emit_incorporation_receipt(incorporator, "test_receipt_id", 0.01)

        assert receipt["receipt_type"] == "instant_incorporation"
        assert receipt["receipt_id"] == "test_receipt_id"
        assert receipt["latency_ok"] is True
        assert receipt["batch_processing"] is False

    def test_no_batch_processing(self):
        """Verify batch processing is disabled."""
        from src.oracle.instant_incorporator import BATCH_PROCESSING_ENABLED

        assert BATCH_PROCESSING_ENABLED is False


class TestIncorporatorConstants:
    """Test D19.3 incorporator constants."""

    def test_max_latency(self):
        """Verify max latency is 100ms."""
        from src.oracle.instant_incorporator import INCORPORATION_LATENCY_MAX_MS

        assert INCORPORATION_LATENCY_MAX_MS == 100

    def test_batch_processing_disabled(self):
        """Verify batch processing is disabled."""
        from src.oracle.instant_incorporator import BATCH_PROCESSING_ENABLED

        assert BATCH_PROCESSING_ENABLED is False
