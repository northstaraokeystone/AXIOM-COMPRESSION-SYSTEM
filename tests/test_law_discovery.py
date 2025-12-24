"""Tests for law discovery module."""


class TestKANSwarm:
    """Test KAN swarm functionality."""

    def test_init_swarm_kan(self):
        """Test KAN initialization."""
        from src.witness.kan_swarm import init_swarm_kan

        kan = init_swarm_kan({})

        assert kan is not None
        assert kan.architecture == [100, 20, 5, 1]

    def test_extract_law(self):
        """Test law extraction."""
        from src.witness.kan_swarm import init_swarm_kan, extract_law

        kan = init_swarm_kan({})
        law = extract_law(kan)

        assert law is not None
        assert "law_id" in law
        assert "compression_ratio" in law


class TestLawDiscovery:
    """Test law discovery functionality."""

    def test_init_law_discovery(self):
        """Test law discovery initialization."""
        from src.witness.kan_swarm import init_swarm_kan
        from src.witness.law_discovery import init_law_discovery

        kan = init_swarm_kan({})
        ld = init_law_discovery(kan)

        assert ld is not None

    def test_law_discovery_threshold(self):
        """Test law discovery threshold."""
        from src.witness.law_discovery import LAW_DISCOVERY_THRESHOLD

        assert LAW_DISCOVERY_THRESHOLD == 0.85


class TestGovernanceSynthesis:
    """Test governance synthesis functionality."""

    def test_synthesize_protocol(self):
        """Test protocol synthesis from law."""
        from src.witness.governance_synthesis import synthesize_protocol

        law = {
            "law_id": "test_law",
            "pattern_source": "high_coherence",
            "human_readable": "Test law description",
        }

        protocol = synthesize_protocol(law)

        assert protocol is not None
        assert protocol.law_id == "test_law"
