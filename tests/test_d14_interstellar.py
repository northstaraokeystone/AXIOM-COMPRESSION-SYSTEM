"""Tests for D14 fractal recursion + interstellar backbone integration."""


class TestD14FractalRecursion:
    """Tests for D14 depth-14 fractal recursion."""

    def test_d14_spec_load(self) -> None:
        """Test loading D14 spec from JSON."""
        from src.fractal_layers import get_d14_spec

        spec = get_d14_spec()
        assert spec is not None
        assert "d14_config" in spec
        assert spec["d14_config"]["recursion_depth"] == 14
        assert spec["d14_config"]["alpha_floor"] == 3.73
        assert spec["d14_config"]["alpha_target"] == 3.75
        assert spec["d14_config"]["alpha_ceiling"] == 3.77

    def test_d14_uplift(self) -> None:
        """Test D14 uplift value at depth 14."""
        from src.fractal_layers import get_d14_uplift, D14_UPLIFT

        uplift = get_d14_uplift(14)
        assert uplift == D14_UPLIFT
        assert uplift == 0.34

    def test_d14_info(self) -> None:
        """Test D14 info retrieval."""
        from src.fractal_layers import get_d14_info

        info = get_d14_info()
        assert info is not None
        assert "version" in info
        assert "d14_config" in info
        assert "interstellar_config" in info
        assert "plonk_config" in info

    def test_d14_adaptive_termination(self) -> None:
        """Test adaptive termination check."""
        from src.fractal_layers import adaptive_termination_check

        # Should not terminate with high delta (current=3.5, previous=3.4 -> delta=0.1)
        result = adaptive_termination_check(3.5, 3.4, threshold=0.001)
        assert result is False

        # Should terminate with low delta (current=3.5, previous=3.5001 -> delta=0.0001)
        result = adaptive_termination_check(3.5, 3.5001, threshold=0.001)
        assert result is True

    def test_d14_recursive_fractal(self) -> None:
        """Test D14 recursive fractal computation."""
        from src.fractal_layers import d14_recursive_fractal

        result = d14_recursive_fractal(
            tree_size=10**6,
            base_alpha=3.41,
            depth=14,
            adaptive=True,
        )

        assert result is not None
        assert "eff_alpha" in result
        assert "actual_depth" in result
        assert "target_met" in result
        assert result["actual_depth"] <= 14

    def test_d14_push_simulate(self) -> None:
        """Test D14 push in simulation mode."""
        from src.fractal_layers import d14_push

        result = d14_push(
            tree_size=10**6,
            base_alpha=3.41,
            simulate=True,
            adaptive=True,
        )

        assert result is not None
        assert result["mode"] == "simulate"
        assert "eff_alpha" in result
        assert "floor_met" in result
        assert "target_met" in result
        assert "ceiling_met" in result

    def test_d14_push_execute(self) -> None:
        """Test D14 push in execute mode."""
        from src.fractal_layers import d14_push

        result = d14_push(
            tree_size=10**6,
            base_alpha=3.41,
            simulate=False,
            adaptive=True,
        )

        assert result is not None
        assert result["mode"] == "execute"
        assert result["eff_alpha"] >= 3.73  # Floor should be met

    def test_d14_constants(self) -> None:
        """Test D14 constants are correctly defined."""
        from src.fractal_layers import (
            D14_ALPHA_FLOOR,
            D14_ALPHA_TARGET,
            D14_ALPHA_CEILING,
            D14_UPLIFT,
            D14_TREE_MIN,
            D14_ADAPTIVE_TERMINATION,
            D14_TERMINATION_THRESHOLD,
        )

        assert D14_ALPHA_FLOOR == 3.73
        assert D14_ALPHA_TARGET == 3.75
        assert D14_ALPHA_CEILING == 3.77
        assert D14_UPLIFT == 0.34
        assert D14_TREE_MIN == 10**12
        assert D14_ADAPTIVE_TERMINATION is True
        assert D14_TERMINATION_THRESHOLD == 0.001


class TestInterstellarBackbone:
    """Tests for 7-body interstellar backbone coordination."""

    def test_interstellar_body_count(self) -> None:
        """Test that we have 7 bodies in the backbone."""
        from src.interstellar_backbone import (
            INTERSTELLAR_BODY_COUNT,
            INTERSTELLAR_JOVIAN_BODIES,
            INTERSTELLAR_INNER_BODIES,
            get_all_bodies,
        )

        assert INTERSTELLAR_BODY_COUNT == 7
        assert len(INTERSTELLAR_JOVIAN_BODIES) == 4
        assert len(INTERSTELLAR_INNER_BODIES) == 3

        bodies = get_all_bodies()
        assert len(bodies) == 7

    def test_interstellar_jovian_bodies(self) -> None:
        """Test Jovian bodies are correctly defined."""
        from src.interstellar_backbone import INTERSTELLAR_JOVIAN_BODIES

        assert "titan" in INTERSTELLAR_JOVIAN_BODIES
        assert "europa" in INTERSTELLAR_JOVIAN_BODIES
        assert "ganymede" in INTERSTELLAR_JOVIAN_BODIES
        assert "callisto" in INTERSTELLAR_JOVIAN_BODIES

    def test_interstellar_inner_bodies(self) -> None:
        """Test inner bodies are correctly defined."""
        from src.interstellar_backbone import INTERSTELLAR_INNER_BODIES

        assert "venus" in INTERSTELLAR_INNER_BODIES
        assert "mercury" in INTERSTELLAR_INNER_BODIES
        assert "mars" in INTERSTELLAR_INNER_BODIES

    def test_interstellar_info(self) -> None:
        """Test interstellar info retrieval."""
        from src.interstellar_backbone import get_interstellar_info

        info = get_interstellar_info()
        assert info is not None
        assert "bodies" in info
        assert "body_count" in info
        assert info["body_count"] == 7
        assert "sync_interval_days" in info
        assert "autonomy_target" in info
        assert info["autonomy_target"] == 0.98

    def test_body_positions(self) -> None:
        """Test body position computation."""
        from src.interstellar_backbone import compute_body_positions

        positions = compute_body_positions(timestamp=0.0)
        assert positions is not None
        assert len(positions) == 7

        # Check inner planets have positions
        assert "mars" in positions
        assert "venus" in positions
        assert "mercury" in positions

        # Check Jovian moons
        assert "titan" in positions
        assert "europa" in positions

    def test_communication_windows(self) -> None:
        """Test communication window computation."""
        from src.interstellar_backbone import compute_interstellar_windows

        windows = compute_interstellar_windows(timestamp=0.0)
        assert windows is not None
        assert len(windows) > 0

        # Each window should have distance, light time, quality
        for key, window in windows.items():
            assert "distance_au" in window
            assert "light_time_min" in window
            assert "window_quality" in window

    def test_backbone_autonomy(self) -> None:
        """Test backbone autonomy computation."""
        from src.interstellar_backbone import compute_backbone_autonomy

        result = compute_backbone_autonomy()
        assert result is not None
        assert "autonomy" in result
        assert "target" in result
        assert "target_met" in result
        assert "body_count" in result
        assert result["body_count"] == 7

    def test_backbone_operations_simulation(self) -> None:
        """Test backbone operations simulation."""
        from src.interstellar_backbone import simulate_backbone_operations

        result = simulate_backbone_operations(duration_days=60)
        assert result is not None
        assert "duration_days" in result
        assert "sync_cycles" in result
        assert "final_autonomy" in result
        assert "target_met" in result
        assert "simulation_complete" in result
        assert result["simulation_complete"] is True

    def test_emergency_failover(self) -> None:
        """Test emergency failover for a body."""
        from src.interstellar_backbone import emergency_failover

        result = emergency_failover("europa")
        assert result is not None
        assert "failed_body" in result
        assert result["failed_body"] == "europa"
        assert "primary_backup" in result
        assert "all_backups" in result
        assert "failover_success" in result

    def test_jovian_inner_handoff(self) -> None:
        """Test handoff between Jovian and inner systems."""
        from src.interstellar_backbone import jovian_inner_handoff

        result = jovian_inner_handoff({"test": "data"}, "jovian_to_inner")
        assert result is not None
        assert "direction" in result
        assert "source_bodies" in result
        assert "dest_bodies" in result
        assert "handoff_success" in result


class TestD14InterstellarHybrid:
    """Tests for D14 + interstellar backbone integration."""

    def test_d14_interstellar_hybrid_simulate(self) -> None:
        """Test D14+interstellar hybrid in simulation mode."""
        from src.interstellar_backbone import d14_interstellar_hybrid

        result = d14_interstellar_hybrid(
            tree_size=10**6,
            base_alpha=3.41,
            simulate=True,
        )

        assert result is not None
        assert result["mode"] == "simulate"
        assert "d14_result" in result
        assert "backbone_result" in result
        assert "combined_alpha" in result
        assert "combined_autonomy" in result
        assert "integration_status" in result

    def test_d14_interstellar_hybrid_execute(self) -> None:
        """Test D14+interstellar hybrid in execute mode."""
        from src.interstellar_backbone import d14_interstellar_hybrid

        result = d14_interstellar_hybrid(
            tree_size=10**6,
            base_alpha=3.41,
            simulate=False,
        )

        assert result is not None
        assert result["mode"] == "execute"
        assert result["combined_alpha"] >= 3.73  # Floor should be met

    def test_integration_status_values(self) -> None:
        """Test that integration status is correctly set."""
        from src.interstellar_backbone import d14_interstellar_hybrid

        # With small tree size, may not meet all targets
        result = d14_interstellar_hybrid(
            tree_size=10**6,
            base_alpha=3.41,
            simulate=False,
        )

        assert result["integration_status"] in [
            "operational",
            "partial",
        ]


class TestMultiplanetIntegration:
    """Tests for multiplanet path integration."""

    def test_integrate_interstellar_backbone(self) -> None:
        """Test multiplanet interstellar backbone integration."""
        from src.paths.multiplanet.core import integrate_interstellar_backbone

        result = integrate_interstellar_backbone()
        assert result is not None
        assert "bodies" in result
        assert "autonomy_achieved" in result
        assert "integrated" in result

    def test_compute_interstellar_autonomy(self) -> None:
        """Test interstellar autonomy computation."""
        from src.paths.multiplanet.core import compute_interstellar_autonomy

        result = compute_interstellar_autonomy()
        # Returns float, not dict
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_coordinate_full_system(self) -> None:
        """Test full system coordination."""
        from src.paths.multiplanet.core import coordinate_full_system

        result = coordinate_full_system()
        assert result is not None
        assert "subsystem" in result
        assert "bodies" in result
        assert "body_count" in result

    def test_get_backbone_status(self) -> None:
        """Test backbone status retrieval."""
        from src.paths.multiplanet.core import get_backbone_status

        result = get_backbone_status()
        assert result is not None
        assert "subsystem" in result
        assert "bodies" in result
        assert "body_count" in result
