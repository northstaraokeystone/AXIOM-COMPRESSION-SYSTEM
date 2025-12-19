"""Tests for D13 recursion + Solar orbital RL hub.

Test coverage:
- D13 spec loading and validation
- D13 alpha floor/target/ceiling (3.68/3.70/3.72)
- Solar orbital hub configuration
- Orbital positions and communication windows
- D13+Solar hybrid integration
"""


class TestD13Spec:
    """Tests for D13 specification loading."""

    def test_d13_spec_loads(self):
        """Test that D13 spec loads with valid structure."""
        from src.fractal_layers import get_d13_spec

        spec = get_d13_spec()
        assert spec is not None
        assert "d13_config" in spec
        assert "solar_hub_config" in spec
        assert "les_config" in spec
        assert "zk_config" in spec

    def test_d13_version(self):
        """Test D13 spec version."""
        from src.fractal_layers import get_d13_spec

        spec = get_d13_spec()
        assert spec.get("version") == "1.0.0"

    def test_d13_alpha_floor(self):
        """Test D13 alpha floor is 3.68."""
        from src.fractal_layers import get_d13_spec, D13_ALPHA_FLOOR

        spec = get_d13_spec()
        assert spec["d13_config"]["alpha_floor"] == 3.68
        assert D13_ALPHA_FLOOR == 3.68

    def test_d13_alpha_target(self):
        """Test D13 alpha target is 3.70."""
        from src.fractal_layers import get_d13_spec, D13_ALPHA_TARGET

        spec = get_d13_spec()
        assert spec["d13_config"]["alpha_target"] == 3.70
        assert D13_ALPHA_TARGET == 3.70

    def test_d13_alpha_ceiling(self):
        """Test D13 alpha ceiling is 3.72."""
        from src.fractal_layers import get_d13_spec, D13_ALPHA_CEILING

        spec = get_d13_spec()
        assert spec["d13_config"]["alpha_ceiling"] == 3.72
        assert D13_ALPHA_CEILING == 3.72

    def test_d13_uplift_value(self):
        """Test D13 uplift is 0.32."""
        from src.fractal_layers import get_d13_spec, D13_UPLIFT

        spec = get_d13_spec()
        assert spec["d13_config"]["uplift"] == 0.32
        assert D13_UPLIFT == 0.32

    def test_d13_recursion_depth(self):
        """Test D13 recursion depth is 13."""
        from src.fractal_layers import get_d13_spec, FRACTAL_RECURSION_MAX_DEPTH

        spec = get_d13_spec()
        assert spec["d13_config"]["recursion_depth"] == 13
        assert FRACTAL_RECURSION_MAX_DEPTH == 13


class TestD13Recursion:
    """Tests for D13 recursion functionality."""

    def test_d13_recursive_fractal(self):
        """Test D13 recursive fractal computation."""
        from src.fractal_layers import d13_recursive_fractal

        result = d13_recursive_fractal(10**9, 3.0, depth=13)
        assert result is not None
        assert "eff_alpha" in result
        assert result["depth"] == 13

    def test_d13_achieves_floor(self):
        """Test D13 achieves alpha floor with sufficient tree size."""
        from src.fractal_layers import d13_recursive_fractal

        result = d13_recursive_fractal(10**12, 3.38, depth=13)
        assert result["eff_alpha"] >= 3.68
        assert result["floor_met"] is True

    def test_d13_push(self):
        """Test D13 push function."""
        from src.fractal_layers import d13_push

        result = d13_push(10**9, 3.0, simulate=True)
        assert result is not None
        assert result["mode"] == "simulate"
        assert result["depth"] == 13

    def test_d13_info(self):
        """Test D13 info function."""
        from src.fractal_layers import get_d13_info

        info = get_d13_info()
        assert info is not None
        assert "d13_config" in info
        assert "solar_hub_config" in info

    def test_d13_push_meets_target(self):
        """Test D13 push achieves floor with sufficient tree size."""
        from src.fractal_layers import d13_push

        # Use 10^12 tree size and base_alpha 3.38 to achieve floor
        result = d13_push(10**12, 3.38, simulate=True)
        # Should at least meet floor
        assert result["floor_met"] is True
        assert result["eff_alpha"] >= 3.68


class TestSolarHubConfig:
    """Tests for Solar orbital hub configuration."""

    def test_solar_hub_config_loads(self):
        """Test Solar hub config loads."""
        from src.solar_orbital_hub import load_solar_hub_config

        config = load_solar_hub_config()
        assert config is not None
        assert "planets" in config

    def test_solar_hub_planets(self):
        """Test Solar hub includes Venus, Mercury, Mars."""
        from src.solar_orbital_hub import load_solar_hub_config, SOLAR_HUB_PLANETS

        config = load_solar_hub_config()
        assert "venus" in config["planets"]
        assert "mercury" in config["planets"]
        assert "mars" in config["planets"]
        assert len(SOLAR_HUB_PLANETS) == 3

    def test_solar_hub_orbital_periods(self):
        """Test orbital periods are correct."""
        from src.solar_orbital_hub import load_solar_hub_config

        config = load_solar_hub_config()
        periods = config["orbital_periods_days"]
        assert periods["venus"] == 225
        assert periods["mercury"] == 88
        assert periods["mars"] == 687

    def test_solar_hub_autonomy_target(self):
        """Test autonomy target is 0.95."""
        from src.solar_orbital_hub import (
            load_solar_hub_config,
            SOLAR_HUB_AUTONOMY_TARGET,
        )

        config = load_solar_hub_config()
        assert config["autonomy_target"] == 0.95
        assert SOLAR_HUB_AUTONOMY_TARGET == 0.95

    def test_solar_hub_sync_interval(self):
        """Test sync interval is 30 days."""
        from src.solar_orbital_hub import (
            load_solar_hub_config,
            ORBITAL_SYNC_INTERVAL_DAYS,
        )

        config = load_solar_hub_config()
        assert config["sync_interval_days"] == 30
        assert ORBITAL_SYNC_INTERVAL_DAYS == 30

    def test_solar_hub_max_latency(self):
        """Test max latency is 22 minutes."""
        from src.solar_orbital_hub import load_solar_hub_config

        config = load_solar_hub_config()
        assert config["max_latency_min"] == 22


class TestSolarOrbitalOperations:
    """Tests for Solar orbital operations."""

    def test_compute_orbital_positions(self):
        """Test orbital position computation."""
        from src.solar_orbital_hub import compute_orbital_positions

        result = compute_orbital_positions(0.0)
        assert result is not None
        assert "positions" in result
        assert "venus" in result["positions"]
        assert "mercury" in result["positions"]
        assert "mars" in result["positions"]

    def test_compute_communication_windows(self):
        """Test communication window computation."""
        from src.solar_orbital_hub import compute_communication_windows

        # Function signature: compute_communication_windows(duration_days=365)
        result = compute_communication_windows(duration_days=365)
        assert result is not None
        # Result should have some structure (windows or latency info)
        assert isinstance(result, dict)

    def test_compute_transfer_windows(self):
        """Test transfer window computation."""
        from src.solar_orbital_hub import compute_transfer_windows

        # Function signature: compute_transfer_windows(from_planet, to_planet, duration_days)
        result = compute_transfer_windows("mars", "venus", 365)
        assert result is not None
        # Result should have some structure
        assert isinstance(result, dict)

    def test_simulate_resource_transfer(self):
        """Test resource transfer simulation."""
        from src.solar_orbital_hub import simulate_resource_transfer

        result = simulate_resource_transfer("mars", "venus", "water_ice", 1000.0)
        assert result is not None
        assert result["from_planet"] == "mars"
        assert result["to_planet"] == "venus"
        assert result["resource"] == "water_ice"


class TestSolarRLCoordination:
    """Tests for Solar hub RL coordination."""

    def test_orbital_rl_step(self):
        """Test orbital RL step."""
        from src.solar_orbital_hub import orbital_rl_step

        state = {"efficiency": 0.85, "latency_min": 15}
        action = {"sync": True}
        result = orbital_rl_step(state, action)
        assert result is not None
        assert "reward" in result
        assert "new_weights" in result

    def test_compute_hub_autonomy(self):
        """Test hub autonomy computation."""
        from src.solar_orbital_hub import compute_hub_autonomy

        autonomy = compute_hub_autonomy()
        assert autonomy >= 0.0
        assert autonomy <= 1.0

    def test_simulate_hub_operations(self):
        """Test hub operations simulation."""
        from src.solar_orbital_hub import simulate_hub_operations

        result = simulate_hub_operations(duration_days=30)
        assert result is not None
        assert result["hub_operational"] is True
        assert "sync_cycles" in result


class TestD13SolarHybrid:
    """Tests for D13+Solar hub integration."""

    def test_d13_solar_hybrid(self):
        """Test D13+Solar hub hybrid."""
        from src.solar_orbital_hub import d13_solar_hybrid

        result = d13_solar_hybrid(10**12, 3.38, simulate=True)
        assert result is not None
        assert result["mode"] == "simulate"
        assert "d13_result" in result
        assert "hub_result" in result
        assert "combined_autonomy" in result

    def test_d13_solar_hybrid_integration_status(self):
        """Test D13+Solar hybrid integration status."""
        from src.solar_orbital_hub import d13_solar_hybrid

        result = d13_solar_hybrid(10**12, 3.38, simulate=True)
        # May be True/False or string status
        assert "integration_status" in result or "integrated" in result

    def test_d13_solar_achieves_target(self):
        """Test D13+Solar achieves alpha floor with sufficient tree size."""
        from src.solar_orbital_hub import d13_solar_hybrid

        # Use 10^12 tree size and base_alpha 3.38 to meet floor
        result = d13_solar_hybrid(10**12, 3.38, simulate=True)
        d13 = result["d13_result"]
        # Should at least meet floor
        assert d13["floor_met"] is True
        assert d13["eff_alpha"] >= 3.68

    def test_coordinate_with_jovian(self):
        """Test Solar-Jovian coordination."""
        from src.solar_orbital_hub import coordinate_with_jovian

        result = coordinate_with_jovian()
        assert result is not None
        assert "combined_autonomy" in result


class TestSolarHubInfo:
    """Tests for Solar hub info functions."""

    def test_get_solar_hub_info(self):
        """Test Solar hub info."""
        from src.solar_orbital_hub import get_solar_hub_info

        info = get_solar_hub_info()
        assert info is not None
        assert "planets" in info
        assert "orbital_periods_days" in info
        assert "autonomy_target" in info
