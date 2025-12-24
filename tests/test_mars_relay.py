"""Tests for Mars relay node module."""

from src.mars_relay_node import (
    load_mars_config,
    deploy_node,
    deploy_mesh,
    run_mars_proof,
    measure_mars_latency,
    get_mars_status,
    simulate_opposition,
    simulate_conjunction,
    stress_test_mars,
    validate_autonomy,
    MARS_AUTONOMY_TARGET,
    MARS_LATENCY_OPPOSITION_MIN,
    MARS_LATENCY_CONJUNCTION_MIN,
)


class TestMarsConfig:
    """Tests for Mars relay configuration."""

    def test_mars_config_loads(self):
        """Config loads successfully."""
        config = load_mars_config()
        assert config is not None
        assert "mars_relay_enabled" in config
        assert "node_count" in config

    def test_mars_autonomy_target(self):
        """Autonomy target is 99.95%."""
        assert MARS_AUTONOMY_TARGET == 0.9995


class TestMarsNodeDeployment:
    """Tests for Mars node deployment."""

    def test_deploy_single_node(self):
        """Single node deploys."""
        result = deploy_node()
        assert result["deployed"] is True
        assert "node_id" in result
        assert result["node_type"] in ["orbital", "surface"]

    def test_deploy_mesh(self):
        """Mesh deploys with 5 nodes."""
        result = deploy_mesh(5)
        assert result["mesh_deployed"] is True
        assert result["total_nodes"] == 5


class TestMarsProof:
    """Tests for Mars relay proof."""

    def test_mars_proof_passes(self):
        """Proof passes autonomy target."""
        result = run_mars_proof(0.5)  # 0.5 hours
        assert result["proof_passed"] is True
        assert result["success_rate"] >= 0.9995

    def test_mars_autonomy_achieved(self):
        """Autonomy target achieved."""
        result = run_mars_proof(0.5)
        assert result["autonomy_achieved"] is True


class TestMarsLatency:
    """Tests for Mars latency measurement."""

    def test_mars_latency_measure(self):
        """Latency measured correctly."""
        result = measure_mars_latency()
        assert "measured_latency_min" in result
        assert result["measured_latency_min"] >= MARS_LATENCY_OPPOSITION_MIN
        assert result["measured_latency_min"] <= MARS_LATENCY_CONJUNCTION_MIN

    def test_mars_latency_within_spec(self):
        """Latency within spec."""
        result = measure_mars_latency()
        assert result["within_spec"] is True


class TestMarsOpposition:
    """Tests for Mars opposition simulation."""

    def test_mars_opposition_latency(self):
        """Opposition latency is ~3 minutes."""
        result = simulate_opposition()
        assert result["phase"] == "opposition"
        assert result["latency_min"] == MARS_LATENCY_OPPOSITION_MIN

    def test_mars_opposition_passes(self):
        """Opposition simulation passes."""
        result = simulate_opposition()
        assert result["simulation_passed"] is True


class TestMarsConjunction:
    """Tests for Mars conjunction simulation."""

    def test_mars_conjunction_latency(self):
        """Conjunction latency is ~22 minutes."""
        result = simulate_conjunction()
        assert result["phase"] == "conjunction"
        assert result["latency_min"] == MARS_LATENCY_CONJUNCTION_MIN

    def test_mars_conjunction_passes(self):
        """Conjunction simulation passes."""
        result = simulate_conjunction()
        assert result["simulation_passed"] is True


class TestMarsStress:
    """Tests for Mars stress testing."""

    def test_mars_stress_passes(self):
        """Stress test passes."""
        result = stress_test_mars(10)  # 10 cycles
        assert result["stress_passed"] is True
        assert result["cycles"] == 10

    def test_mars_stress_success_rate(self):
        """Success rate maintained."""
        result = stress_test_mars(10)
        assert result["avg_success_rate"] >= 0.9995


class TestMarsAutonomy:
    """Tests for autonomy validation."""

    def test_mars_autonomy_valid(self):
        """Autonomy validation passes."""
        result = validate_autonomy(0.9995)
        assert result["valid"] is True

    def test_mars_autonomy_below_target(self):
        """Below target fails."""
        result = validate_autonomy(0.99)
        assert result["valid"] is False


class TestMarsStatus:
    """Tests for status queries."""

    def test_mars_status(self):
        """Status query works."""
        status = get_mars_status()
        assert "mars_relay_enabled" in status
        assert "node_count" in status
        assert "autonomy_target" in status
        assert "latency_range_min" in status


class TestMarsReceipts:
    """Tests for receipt emission."""

    def test_mars_receipt(self, capsys):
        """Receipt emitted."""
        deploy_node()
        captured = capsys.readouterr()
        assert "mars_relay_receipt" in captured.out
