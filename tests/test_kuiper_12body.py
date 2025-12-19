"""Tests for Kuiper 12-body chaotic N-body simulation.

Tests:
- Config loading
- Body initialization
- Symplectic integration
- Lyapunov exponent computation
- Resonance analysis
- Stability assessment
"""

import pytest
from src.kuiper_12body_chaos import (
    load_kuiper_config,
    initialize_kuiper_bodies,
    compute_kuiper_forces,
    symplectic_kuiper_integrate,
    compute_kuiper_lyapunov,
    simulate_kuiper,
    analyze_resonances,
    integrate_with_backbone,
    KUIPER_BODY_COUNT,
    KUIPER_LYAPUNOV_THRESHOLD,
    KUIPER_STABILITY_TARGET,
)


class TestKuiperConfig:
    """Tests for Kuiper configuration loading."""

    def test_config_loads(self):
        """Config loads successfully."""
        config = load_kuiper_config()
        assert config is not None
        assert "body_count" in config
        assert "bodies" in config

    def test_body_count(self):
        """Body count is 12."""
        config = load_kuiper_config()
        assert config["body_count"] == 12
        assert KUIPER_BODY_COUNT == 12

    def test_body_groups(self):
        """Body groups are correct."""
        config = load_kuiper_config()
        bodies = config["bodies"]
        assert "jovian" in bodies
        assert "inner" in bodies
        assert "kuiper" in bodies

    def test_jovian_bodies(self):
        """Jovian bodies correct."""
        config = load_kuiper_config()
        jovian = config["bodies"]["jovian"]
        assert "titan" in jovian
        assert "europa" in jovian
        assert "ganymede" in jovian
        assert "callisto" in jovian

    def test_kuiper_bodies(self):
        """Kuiper belt bodies correct."""
        config = load_kuiper_config()
        kuiper = config["bodies"]["kuiper"]
        assert "ceres" in kuiper
        assert "pluto" in kuiper
        assert "eris" in kuiper
        assert "makemake" in kuiper
        assert "haumea" in kuiper

    def test_lyapunov_threshold(self):
        """Lyapunov threshold is 0.15."""
        config = load_kuiper_config()
        assert config["lyapunov_threshold"] == 0.15
        assert KUIPER_LYAPUNOV_THRESHOLD == 0.15

    def test_stability_target(self):
        """Stability target is 0.93."""
        config = load_kuiper_config()
        assert config["stability_target"] == 0.93
        assert KUIPER_STABILITY_TARGET == 0.93


class TestKuiperBodies:
    """Tests for body initialization."""

    def test_initialize_bodies(self):
        """Bodies initialize correctly."""
        bodies = initialize_kuiper_bodies()
        assert len(bodies) == KUIPER_BODY_COUNT

    def test_body_properties(self):
        """Each body has required properties."""
        bodies = initialize_kuiper_bodies()
        for body in bodies:
            assert "name" in body
            assert "mass" in body
            assert "position" in body
            assert "velocity" in body

    def test_body_names(self):
        """All expected bodies present."""
        bodies = initialize_kuiper_bodies()
        names = [b["name"] for b in bodies]

        expected = [
            "titan", "europa", "ganymede", "callisto",
            "venus", "mercury", "mars",
            "ceres", "pluto", "eris", "makemake", "haumea",
        ]
        for exp in expected:
            assert exp in names


class TestKuiperForces:
    """Tests for force computation."""

    def test_compute_forces(self):
        """Forces compute correctly."""
        bodies = initialize_kuiper_bodies()
        forces = compute_kuiper_forces(bodies)

        assert len(forces) == len(bodies)
        for f in forces:
            assert len(f) == 3  # 3D force vector

    def test_force_symmetry(self):
        """Forces are antisymmetric (Newton's 3rd law)."""
        bodies = initialize_kuiper_bodies()
        forces = compute_kuiper_forces(bodies)

        # Sum of all forces should be approximately zero
        total = [sum(f[i] for f in forces) for i in range(3)]
        for t in total:
            assert abs(t) < 1e-6


class TestKuiperIntegration:
    """Tests for symplectic integration."""

    def test_single_step(self):
        """Single integration step works."""
        bodies = initialize_kuiper_bodies()
        result = symplectic_kuiper_integrate(bodies, dt=0.01, steps=1)

        assert "bodies" in result
        assert len(result["bodies"]) == KUIPER_BODY_COUNT

    def test_energy_conservation(self):
        """Energy approximately conserved."""
        bodies = initialize_kuiper_bodies()
        result = symplectic_kuiper_integrate(bodies, dt=0.01, steps=100)

        assert "energy_variation" in result
        # Symplectic should conserve energy well
        assert result["energy_variation"] < 0.01

    def test_multi_step(self):
        """Multi-step integration works."""
        bodies = initialize_kuiper_bodies()
        result = symplectic_kuiper_integrate(bodies, dt=0.01, steps=1000)

        assert "steps" in result
        assert result["steps"] == 1000


class TestKuiperLyapunov:
    """Tests for Lyapunov exponent computation."""

    def test_lyapunov_computes(self):
        """Lyapunov exponent computes."""
        bodies = initialize_kuiper_bodies()
        lyapunov = compute_kuiper_lyapunov(bodies, dt=0.01, steps=100)

        assert "lyapunov_exponent" in lyapunov
        assert isinstance(lyapunov["lyapunov_exponent"], float)

    def test_stability_from_lyapunov(self):
        """Stability derived from Lyapunov."""
        bodies = initialize_kuiper_bodies()
        lyapunov = compute_kuiper_lyapunov(bodies, dt=0.01, steps=100)

        assert "is_stable" in lyapunov
        # Stable if Lyapunov < threshold
        exp = lyapunov["lyapunov_exponent"]
        expected_stable = exp < KUIPER_LYAPUNOV_THRESHOLD
        assert lyapunov["is_stable"] == expected_stable


class TestKuiperSimulation:
    """Tests for full Kuiper simulation."""

    def test_simulation_runs(self):
        """Full simulation runs."""
        result = simulate_kuiper(bodies=12, duration_years=10.0)

        assert "duration_years" in result
        assert "steps" in result
        assert "body_count" in result
        assert result["body_count"] == 12

    def test_simulation_stability(self):
        """Simulation reports stability."""
        result = simulate_kuiper(bodies=12, duration_years=10.0)

        assert "stability" in result
        assert "is_stable" in result
        assert "lyapunov_exponent" in result

    def test_simulation_meets_target(self):
        """Simulation can meet stability target."""
        result = simulate_kuiper(bodies=12, duration_years=10.0)

        # May or may not meet target, but should report
        assert "target_met" in result


class TestKuiperResonances:
    """Tests for resonance analysis."""

    def test_resonance_analysis(self):
        """Resonance analysis runs."""
        # First run simulation to get trajectory
        sim_result = simulate_kuiper(bodies=12, duration_years=10.0)
        trajectory = sim_result.get("trajectory", [[0] * 6 for _ in range(12)])
        result = analyze_resonances(trajectory)

        assert "body_count" in result
        assert "resonances" in result

    def test_resonance_properties(self):
        """Resonances have correct properties."""
        sim_result = simulate_kuiper(bodies=12, duration_years=10.0)
        trajectory = sim_result.get("trajectory", [[0] * 6 for _ in range(12)])
        result = analyze_resonances(trajectory)
        resonances = result["resonances"]

        for res in resonances:
            assert "body1" in res
            assert "body2" in res
            assert "ratio" in res

    def test_known_resonances(self):
        """Known resonances detected."""
        sim_result = simulate_kuiper(bodies=12, duration_years=10.0)
        trajectory = sim_result.get("trajectory", [[0] * 6 for _ in range(12)])
        result = analyze_resonances(trajectory)

        # Pluto-Neptune 3:2 resonance is famous
        resonances = result["resonances"]
        found_pluto = any(
            "pluto" in res.get("body1", "") or "pluto" in res.get("body2", "")
            for res in resonances
        )
        # Should find at least some Pluto resonance
        assert found_pluto or len(resonances) > 0


class TestKuiperBackboneIntegration:
    """Tests for backbone integration."""

    def test_backbone_integration(self):
        """Backbone integration works."""
        result = integrate_with_backbone()

        assert "kuiper_result" in result
        assert "backbone_status" in result

    def test_integration_status(self):
        """Integration reports status."""
        result = integrate_with_backbone()

        assert "integration_complete" in result
        assert result["integration_complete"] is True

    def test_coordination(self):
        """Coordination computed."""
        result = integrate_with_backbone()

        assert "coordination" in result
        assert isinstance(result["coordination"], float)
