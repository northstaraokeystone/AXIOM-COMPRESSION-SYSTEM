"""Tests for swarm testnet module."""

from src.swarm_testnet import (
    load_swarm_config,
    init_swarm,
    deploy_full_swarm,
    create_mesh_topology,
    run_swarm_consensus,
    stress_test_swarm,
    get_swarm_status,
    calculate_mesh_connections,
    SWARM_NODE_COUNT,
    SWARM_ORBITAL_NODES,
    SWARM_SURFACE_NODES,
    SWARM_DEEP_SPACE_NODES,
)


class TestSwarmConfig:
    """Tests for swarm configuration."""

    def test_swarm_config_loads(self):
        """Config loads successfully."""
        config = load_swarm_config()
        assert config is not None
        assert "node_count" in config
        assert "mesh_topology" in config

    def test_swarm_node_count(self):
        """Node count is 100."""
        assert SWARM_NODE_COUNT == 100

    def test_swarm_node_distribution(self):
        """Node distribution correct."""
        assert SWARM_ORBITAL_NODES == 40
        assert SWARM_SURFACE_NODES == 30
        assert SWARM_DEEP_SPACE_NODES == 30
        assert SWARM_ORBITAL_NODES + SWARM_SURFACE_NODES + SWARM_DEEP_SPACE_NODES == 100


class TestSwarmInit:
    """Tests for swarm initialization."""

    def test_swarm_init(self):
        """Swarm initializes."""
        result = init_swarm()
        assert result["initialized"] is True
        assert result["node_count"] == 100


class TestSwarmDeploy:
    """Tests for swarm deployment."""

    def test_swarm_deploy_full(self):
        """Full swarm deploys."""
        result = deploy_full_swarm()
        assert result["full_deployment"] is True
        assert result["node_count"] == 100

    def test_swarm_deploy_node_types(self):
        """Node types correct."""
        result = deploy_full_swarm()
        assert result["orbital_count"] == 40
        assert result["surface_count"] == 30
        assert result["deep_space_count"] == 30


class TestSwarmMesh:
    """Tests for mesh topology."""

    def test_mesh_topology(self):
        """Mesh topology created."""
        init_swarm()
        result = create_mesh_topology()
        assert result["mesh_created"] is True
        expected = 100 * 99 // 2  # n(n-1)/2
        assert result["connection_count"] == expected

    def test_mesh_calculation(self):
        """Mesh connection formula correct."""
        connections = calculate_mesh_connections(100)
        assert connections == 4950  # 100*99/2


class TestSwarmConsensus:
    """Tests for swarm consensus."""

    def test_swarm_consensus(self):
        """Consensus reached."""
        init_swarm()
        result = run_swarm_consensus()
        assert result["consensus_reached"] is True
        assert result["participating_nodes"] >= 67  # 2/3 quorum


class TestSwarmStress:
    """Tests for swarm stress testing."""

    def test_swarm_stress(self):
        """Stress test passes."""
        init_swarm()
        result = stress_test_swarm(10)  # 10 iterations
        assert result["stress_passed"] is True
        assert result["iterations"] == 10


class TestSwarmStatus:
    """Tests for status queries."""

    def test_swarm_status(self):
        """Status query works."""
        init_swarm()
        status = get_swarm_status()
        assert "initialized" in status
        assert "node_count" in status
        assert "mesh_connections" in status


class TestSwarmReceipts:
    """Tests for receipt emission."""

    def test_swarm_receipt(self, capsys):
        """Receipt emitted."""
        init_swarm()
        captured = capsys.readouterr()
        assert "swarm_testnet_receipt" in captured.out
