"""Tests for D9 recursion + Ganymede magnetic field + Atacama drone integration.

Coverage:
- D9 spec loading and validation
- D9 recursion and alpha targets
- Ganymede magnetic field navigation
- Atacama drone array validation
- D9+Ganymede hybrid integration
"""


class TestD9Spec:
    """Tests for D9 specification loading."""

    def test_d9_spec_loads(self):
        """Spec loads with valid structure."""
        from src.fractal_layers import get_d9_spec

        spec = get_d9_spec()
        assert "version" in spec
        assert "d9_config" in spec
        assert "ganymede_config" in spec
        assert "atacama_drone_config" in spec
        assert "randomized_paths_config" in spec

    def test_d9_alpha_floor(self):
        """D9 alpha floor is 3.48."""
        from src.fractal_layers import get_d9_spec, D9_ALPHA_FLOOR

        spec = get_d9_spec()
        assert spec["d9_config"]["alpha_floor"] == D9_ALPHA_FLOOR
        assert spec["d9_config"]["alpha_floor"] == 3.48

    def test_d9_alpha_target(self):
        """D9 alpha target is 3.50."""
        from src.fractal_layers import get_d9_spec, D9_ALPHA_TARGET

        spec = get_d9_spec()
        assert spec["d9_config"]["alpha_target"] == D9_ALPHA_TARGET
        assert spec["d9_config"]["alpha_target"] == 3.50

    def test_d9_uplift_value(self):
        """D9 uplift is 0.24."""
        from src.fractal_layers import get_d9_spec, D9_UPLIFT

        spec = get_d9_spec()
        assert spec["d9_config"]["uplift"] == D9_UPLIFT
        assert spec["d9_config"]["uplift"] == 0.24


class TestD9Recursion:
    """Tests for D9 recursion functionality."""

    def test_d9_recursive_fractal_depth(self):
        """D9 recursion uses depth 9."""
        from src.fractal_layers import d9_recursive_fractal

        result = d9_recursive_fractal(10**9, 3.0, depth=9)
        assert result["depth"] == 9

    def test_d9_alpha_meets_floor(self):
        """D9 achieves alpha >= 3.48."""
        from src.fractal_layers import d9_recursive_fractal, D9_ALPHA_FLOOR

        result = d9_recursive_fractal(10**12, 3.26, depth=9)
        assert result["eff_alpha"] >= D9_ALPHA_FLOOR

    def test_d9_instability_zero(self):
        """D9 maintains zero instability."""
        from src.fractal_layers import d9_recursive_fractal

        result = d9_recursive_fractal(10**12, 3.26, depth=9)
        assert result["instability"] == 0.00

    def test_d9_push_slo_passed(self):
        """D9 push passes SLO."""
        from src.fractal_layers import d9_push

        result = d9_push(10**12, 3.26, simulate=True)
        assert result["slo_passed"] is True


class TestGanymedeConfig:
    """Tests for Ganymede configuration."""

    def test_ganymede_config_loads(self):
        """Ganymede config loads from spec."""
        from src.ganymede_mag_hybrid import load_ganymede_config

        config = load_ganymede_config()
        assert config["body"] == "ganymede"
        assert config["resource"] == "magnetic_shielding"

    def test_ganymede_field_strength(self):
        """Ganymede surface field is 719 nT."""
        from src.ganymede_mag_hybrid import (
            load_ganymede_config,
            GANYMEDE_SURFACE_FIELD_NT,
        )

        config = load_ganymede_config()
        assert config["surface_field_nT"] == GANYMEDE_SURFACE_FIELD_NT
        assert config["surface_field_nT"] == 719

    def test_ganymede_magnetopause(self):
        """Ganymede magnetopause is 2600 km."""
        from src.ganymede_mag_hybrid import (
            load_ganymede_config,
            GANYMEDE_MAGNETOPAUSE_KM,
        )

        config = load_ganymede_config()
        assert config["magnetopause_km"] == GANYMEDE_MAGNETOPAUSE_KM
        assert config["magnetopause_km"] == 2600

    def test_ganymede_autonomy_requirement(self):
        """Ganymede autonomy requirement is 97%."""
        from src.ganymede_mag_hybrid import (
            load_ganymede_config,
            GANYMEDE_AUTONOMY_REQUIREMENT,
        )

        config = load_ganymede_config()
        assert config["autonomy_requirement"] == GANYMEDE_AUTONOMY_REQUIREMENT
        assert config["autonomy_requirement"] == 0.97

    def test_ganymede_navigation_modes(self):
        """All 3 navigation modes are present."""
        from src.ganymede_mag_hybrid import load_ganymede_config

        config = load_ganymede_config()
        assert len(config["navigation_modes"]) == 3
        assert "field_following" in config["navigation_modes"]
        assert "magnetopause_crossing" in config["navigation_modes"]
        assert "polar_transit" in config["navigation_modes"]

    def test_ganymede_radiation_shielding(self):
        """Ganymede radiation shielding is 3.5x."""
        from src.ganymede_mag_hybrid import (
            load_ganymede_config,
            GANYMEDE_RADIATION_SHIELD_FACTOR,
        )

        config = load_ganymede_config()
        assert config["radiation_shield_factor"] == GANYMEDE_RADIATION_SHIELD_FACTOR
        assert config["radiation_shield_factor"] == 3.5


class TestGanymedeNavigation:
    """Tests for Ganymede navigation simulation."""

    def test_field_following_nav(self):
        """Field following navigation works."""
        from src.ganymede_mag_hybrid import field_following_nav

        waypoints = [(3134, 0, 0), (3234, 500, 0)]
        result = field_following_nav(waypoints, duration_hrs=24)
        assert result["mode"] == "field_following"
        assert "autonomy_achieved" in result

    def test_magnetopause_crossing(self):
        """Magnetopause crossing navigation works."""
        from src.ganymede_mag_hybrid import magnetopause_crossing

        entry = (3634, 0, 0)
        exit = (5234, 0, 0)
        result = magnetopause_crossing(entry, exit, duration_hrs=4)
        assert result["mode"] == "magnetopause_crossing"
        assert "crosses_magnetopause" in result

    def test_polar_transit(self):
        """Polar transit navigation works."""
        from src.ganymede_mag_hybrid import polar_transit

        result = polar_transit("north", "south", 500)
        assert result["mode"] == "polar_transit"
        assert "start_field_nT" in result
        assert "end_field_nT" in result

    def test_simulate_navigation(self):
        """Navigation simulation works."""
        from src.ganymede_mag_hybrid import simulate_navigation

        result = simulate_navigation("field_following", 24)
        assert result["simulation"] is True
        assert "autonomy" in result


class TestD9GanymedeHybrid:
    """Tests for D9+Ganymede hybrid integration."""

    def test_d9_ganymede_hybrid_alpha(self):
        """Hybrid achieves alpha >= 3.48."""
        from src.ganymede_mag_hybrid import d9_ganymede_hybrid
        from src.fractal_layers import D9_ALPHA_FLOOR

        result = d9_ganymede_hybrid(10**12, 3.26)
        assert result["d9_result"]["eff_alpha"] >= D9_ALPHA_FLOOR

    def test_d9_ganymede_hybrid_autonomy(self):
        """Hybrid achieves autonomy >= 0.97."""
        from src.ganymede_mag_hybrid import d9_ganymede_hybrid
        from src.ganymede_mag_hybrid import GANYMEDE_AUTONOMY_REQUIREMENT

        result = d9_ganymede_hybrid(10**12, 3.26)
        assert result["ganymede_result"]["autonomy"] >= GANYMEDE_AUTONOMY_REQUIREMENT

    def test_d9_ganymede_hybrid_combined_slo(self):
        """Hybrid combined SLO is present."""
        from src.ganymede_mag_hybrid import d9_ganymede_hybrid

        result = d9_ganymede_hybrid(10**12, 3.26)
        assert "combined_slo" in result
        assert "all_targets_met" in result["combined_slo"]


class TestDroneConfig:
    """Tests for Atacama drone configuration."""

    def test_drone_config_loads(self):
        """Drone config loads from spec."""
        from src.atacama_drone import load_drone_config

        config = load_drone_config()
        assert "coverage_km2" in config
        assert "sample_rate_hz" in config

    def test_drone_coverage(self):
        """Drone coverage is 100 km2."""
        from src.atacama_drone import load_drone_config, ATACAMA_DRONE_COVERAGE_KM2

        config = load_drone_config()
        assert config["coverage_km2"] == ATACAMA_DRONE_COVERAGE_KM2
        assert config["coverage_km2"] == 100

    def test_drone_mars_correlation(self):
        """Drone Mars correlation is 92%."""
        from src.atacama_drone import load_drone_config, ATACAMA_MARS_CORRELATION

        config = load_drone_config()
        assert config["mars_correlation"] == ATACAMA_MARS_CORRELATION
        assert config["mars_correlation"] == 0.92


class TestDroneOperations:
    """Tests for Atacama drone operations."""

    def test_swarm_coverage(self):
        """Swarm coverage simulation works."""
        from src.atacama_drone import simulate_swarm_coverage

        result = simulate_swarm_coverage(10, 1000)
        assert "coverage_ratio" in result
        assert "total_coverage_km2" in result

    def test_dust_sampling(self):
        """Dust sampling works."""
        from src.atacama_drone import sample_dust_metrics

        result = sample_dust_metrics(10, 60)
        assert "metrics" in result
        assert "particle_size" in result["metrics"]

    def test_mars_correlation(self):
        """Mars correlation computation works."""
        from src.atacama_drone import sample_dust_metrics, compute_mars_correlation

        sampling = sample_dust_metrics(10, 60)
        correlation = compute_mars_correlation(sampling)
        assert correlation >= 0.0
        assert correlation <= 1.0

    def test_drone_validation(self):
        """Full drone validation works."""
        from src.atacama_drone import run_drone_validation

        result = run_drone_validation(10, 1000, 60)
        assert "validation_passed" in result
        assert "mars_correlation" in result
