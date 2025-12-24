"""Tests for D19.3 Causal Subgraph Extractor.

Laws ARE the maximal invariant subgraphs in chain history.
"""


class TestCausalSubgraphExtractor:
    """Tests for CausalSubgraphExtractor functionality."""

    def test_init_extractor(self):
        """Test extractor initialization."""
        from src.oracle import init_extractor

        history = [{"receipt_type": "test", "ts": "2024-01-01T00:00:00Z"}]
        extractor = init_extractor(history)

        assert extractor is not None
        assert extractor.extractor_id is not None
        assert len(extractor.extractor_id) == 8

    def test_build_causal_graph(self):
        """Test causal graph construction."""
        from src.oracle import build_causal_graph

        receipts = [
            {
                "receipt_type": "a",
                "ts": "2024-01-01T00:00:00Z",
                "payload_hash": "hash1",
            },
            {
                "receipt_type": "b",
                "ts": "2024-01-01T00:01:00Z",
                "payload_hash": "hash2",
            },
            {
                "receipt_type": "c",
                "ts": "2024-01-01T00:02:00Z",
                "payload_hash": "hash3",
            },
        ]

        graph = build_causal_graph(receipts)

        assert "nodes" in graph
        assert "edges" in graph
        assert graph["node_count"] == 3
        assert graph["edge_count"] == 2
        assert graph["is_dag"] is True

    def test_find_maximal_subgraphs(self):
        """Test maximal subgraph finding."""
        from src.oracle import init_extractor, find_maximal_subgraphs

        history = [
            {
                "receipt_type": "a",
                "ts": "2024-01-01T00:00:00Z",
                "payload_hash": "hash1",
            },
            {
                "receipt_type": "b",
                "ts": "2024-01-01T00:01:00Z",
                "payload_hash": "hash2",
            },
            {
                "receipt_type": "c",
                "ts": "2024-01-01T00:02:00Z",
                "payload_hash": "hash3",
            },
        ]

        extractor = init_extractor(history)
        subgraphs = find_maximal_subgraphs(extractor)

        assert isinstance(subgraphs, list)
        # Should find at least one connected component
        if history:
            assert len(subgraphs) >= 1

    def test_subgraph_to_law(self):
        """Test subgraph to law conversion."""
        from src.oracle import init_extractor, find_maximal_subgraphs, subgraph_to_law

        history = [
            {
                "receipt_type": "a",
                "ts": "2024-01-01T00:00:00Z",
                "payload_hash": "hash1",
            },
            {
                "receipt_type": "b",
                "ts": "2024-01-01T00:01:00Z",
                "payload_hash": "hash2",
            },
        ]

        extractor = init_extractor(history)
        subgraphs = find_maximal_subgraphs(extractor)

        if subgraphs:
            law = subgraph_to_law(subgraphs[0], extractor)

            assert "law_id" in law
            assert "law_type" in law
            assert law["law_type"] == "causal_subgraph"
            assert "node_count" in law
            assert "source" in law
            assert law["source"] == "causal_subgraph_extraction"

    def test_validate_causal_invariance(self):
        """Test causal invariance validation."""
        from src.oracle import validate_causal_invariance

        history = [
            {"receipt_type": "a", "ts": "2024-01-01T00:00:00Z"},
            {"receipt_type": "b", "ts": "2024-01-01T00:01:00Z"},
        ]

        # Valid law
        law = {
            "law_id": "test",
            "receipt_types": ["a", "b"],
            "invariance_score": 0.8,
        }

        is_valid = validate_causal_invariance(law, history)
        assert is_valid is True

        # Invalid law (missing types in history)
        invalid_law = {
            "law_id": "test",
            "receipt_types": ["a", "b", "missing_type"],
            "invariance_score": 0.8,
        }

        is_valid = validate_causal_invariance(invalid_law, history)
        assert is_valid is False

    def test_emit_subgraph_receipt(self):
        """Test subgraph receipt emission."""
        from src.oracle import init_extractor, emit_subgraph_receipt

        history = [{"receipt_type": "test", "ts": "2024-01-01T00:00:00Z"}]
        extractor = init_extractor(history)
        laws = [{"law_id": "test_law"}]

        receipt = emit_subgraph_receipt(extractor, laws)

        assert receipt["receipt_type"] == "causal_subgraph_law"
        assert receipt["laws_discovered"] == 1
        assert receipt["source"] == "maximal_causal_subgraph"
