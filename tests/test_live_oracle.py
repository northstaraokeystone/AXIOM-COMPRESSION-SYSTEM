"""Tests for D19.3 Live History Oracle.

D19.3: Laws are oracled directly from the live chain's emergent causality.
Projection KILLED. Simulation KILLED. History is the only truth.
"""


class TestLiveHistoryOracle:
    """Tests for LiveHistoryOracle functionality."""

    def test_init_oracle(self):
        """Test oracle initialization."""
        from src.oracle import init_oracle

        oracle = init_oracle({})

        assert oracle is not None
        assert oracle.oracle_id is not None
        assert len(oracle.oracle_id) == 8

    def test_load_chain_history(self):
        """Test loading chain history."""
        from src.oracle import load_chain_history

        history = load_chain_history()

        assert isinstance(history, list)
        # History should be sorted by timestamp
        if len(history) > 1:
            for i in range(1, len(history)):
                assert history[i].get("ts", "") >= history[i - 1].get("ts", "")

    def test_compute_history_compression(self):
        """Test compression calculation on history."""
        from src.oracle import compute_history_compression

        # Empty history
        assert compute_history_compression([]) == 0.0

        # Single receipt
        history = [{"receipt_type": "test"}]
        compression = compute_history_compression(history)
        assert 0.0 <= compression <= 1.0

        # Multiple receipts with patterns
        history = [{"receipt_type": "a"} for _ in range(50)]
        history.extend([{"receipt_type": "b"} for _ in range(50)])
        compression = compute_history_compression(history)
        assert compression > 0.0

    def test_extract_laws_from_history(self):
        """Test law extraction from history."""
        from src.oracle import extract_laws_from_history

        # Empty history
        laws = extract_laws_from_history([])
        assert laws == []

        # History with patterns
        history = [{"receipt_type": f"type_{i % 3}"} for i in range(100)]
        laws = extract_laws_from_history(history)

        assert isinstance(laws, list)
        # Should find at least one pattern
        if history:
            assert len(laws) > 0

    def test_oracle_query(self):
        """Test oracle query functionality."""
        from src.oracle import init_oracle, oracle_query, extract_laws_from_history

        oracle = init_oracle({})
        oracle.history = [{"receipt_type": "test"}]
        oracle.laws = extract_laws_from_history(oracle.history)

        result = oracle_query(oracle, "test")

        assert "query" in result
        assert "projection_used" in result
        assert result["projection_used"] is False
        assert result["simulation_used"] is False

    def test_emit_oracle_receipt(self):
        """Test oracle receipt emission."""
        from src.oracle import init_oracle, emit_oracle_receipt

        oracle = init_oracle({})
        laws = [{"law_id": "test_law"}]
        compression = 0.5

        receipt = emit_oracle_receipt(oracle, laws, compression)

        assert receipt["receipt_type"] == "live_history_oracle"
        assert receipt["laws_discovered"] == 1
        assert receipt["compression_ratio"] == compression
        assert receipt["projection_enabled"] is False
        assert receipt["simulation_enabled"] is False

    def test_get_oracle_status(self):
        """Test oracle status."""
        from src.oracle import get_oracle_status

        status = get_oracle_status()

        assert status["oracle_mode"] == "live_history_only"
        assert status["projection_enabled"] is False
        assert status["simulation_enabled"] is False
        assert status["compression_source"] == "chain_history_only"


class TestOracleConstants:
    """Test D19.3 oracle constants."""

    def test_projection_disabled(self):
        """Verify projection is disabled."""
        from src.oracle.live_history_oracle import PROJECTION_ENABLED

        assert PROJECTION_ENABLED is False

    def test_simulation_disabled(self):
        """Verify simulation is disabled."""
        from src.oracle.live_history_oracle import SIMULATION_ENABLED

        assert SIMULATION_ENABLED is False

    def test_oracle_mode(self):
        """Verify oracle mode is live history only."""
        from src.oracle.live_history_oracle import ORACLE_MODE

        assert ORACLE_MODE == "live_history_only"

    def test_compression_source(self):
        """Verify compression source is chain history only."""
        from src.oracle.live_history_oracle import COMPRESSION_SOURCE

        assert COMPRESSION_SOURCE == "chain_history_only"
