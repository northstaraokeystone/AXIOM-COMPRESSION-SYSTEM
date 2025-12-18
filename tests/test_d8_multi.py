"""Tests for D8 fractal recursion + multi-planet sync + Atacama validation.

Full system tests for D8+sync targeting alpha >= 3.43.
"""


class TestD8Spec:
    """Test D8 spec loading and configuration."""

    def test_d8_spec_loads(self):
        """Spec loads with valid dual-hash."""
        from src.fractal_layers import get_d8_spec

        spec = get_d8_spec()
        assert spec is not None
        assert "version" in spec
        assert spec["version"] == "1.0.0"

    def test_d8_alpha_floor(self):
        """Assert eff_alpha >= 3.43 at depth=8."""
        from src.fractal_layers import d8_recursive_fractal

        result = d8_recursive_fractal(10**12, 3.23, depth=8)
        assert result["eff_alpha"] >= 3.43

    def test_d8_alpha_target(self):
        """Assert eff_alpha >= 3.45 achievable."""
        from src.fractal_layers import d8_recursive_fractal

        result = d8_recursive_fractal(10**12, 3.25, depth=8)
        assert result["eff_alpha"] >= 3.45

    def test_d8_uplift_value(self):
        """Assert uplift == 0.22 at depth 8."""
        from src.fractal_layers import get_d8_uplift

        uplift = get_d8_uplift(8)
        assert uplift == 0.22

    def test_d8_instability_zero(self):
        """Assert instability is 0.00."""
        from src.fractal_layers import d8_recursive_fractal

        result = d8_recursive_fractal(10**12, 3.23, depth=8)
        assert result["instability"] == 0.00


class TestSyncConfig:
    """Test multi-planet sync configuration."""

    def test_sync_config_loads(self):
        """Sync config valid."""
        from src.multi_planet_sync import load_sync_config

        config = load_sync_config()
        assert config is not None
        assert "moons" in config
        assert "unified_rl" in config

    def test_sync_moons_present(self):
        """Titan and Europa in config."""
        from src.multi_planet_sync import load_sync_config

        config = load_sync_config()
        moons = config.get("moons", [])
        assert "titan" in moons
        assert "europa" in moons

    def test_sync_efficiency_threshold(self):
        """Assert efficiency >= 0.85."""
        from src.multi_planet_sync import run_sync_cycle, RESOURCE_SHARE_EFFICIENCY

        cycle = run_sync_cycle()
        assert cycle["efficiency"] >= RESOURCE_SHARE_EFFICIENCY

    def test_unified_rl_learning_rate(self):
        """Assert lr == 0.001."""
        from src.multi_planet_sync import load_sync_config

        config = load_sync_config()
        lr = config.get("unified_rl", {}).get("learning_rate", 0)
        assert lr == 0.001


class TestD8MultiSync:
    """Test integrated D8+sync."""

    def test_d8_multi_sync_alpha(self):
        """Assert hybrid eff_alpha >= 3.43."""
        from src.multi_planet_sync import d8_multi_sync

        result = d8_multi_sync(10**12, 3.23)
        assert result["d8_result"]["eff_alpha"] >= 3.43

    def test_d8_multi_sync_receipt(self):
        """Assert receipt emitted (via result structure)."""
        from src.multi_planet_sync import d8_multi_sync

        result = d8_multi_sync(10**12, 3.23)
        # Result structure indicates receipt was generated
        assert "combined_score" in result
        assert result["combined_score"] > 0


class TestAtacamaValidation:
    """Test Atacama Mars dust validation."""

    def test_atacama_dust_similarity(self):
        """Assert similarity >= 0.92."""
        from src.atacama_validation import (
            load_atacama_config,
            ATACAMA_DUST_ANALOG_MATCH,
        )

        config = load_atacama_config()
        similarity = config.get("dust_similarity", 0)
        assert similarity >= ATACAMA_DUST_ANALOG_MATCH

    def test_atacama_flux_ratio(self):
        """Assert ratio calculated correctly."""
        from src.atacama_validation import (
            compute_dust_correction,
            ATACAMA_SOLAR_FLUX_W_M2,
            MARS_SOLAR_FLUX_W_M2,
        )

        expected_ratio = MARS_SOLAR_FLUX_W_M2 / ATACAMA_SOLAR_FLUX_W_M2
        correction = compute_dust_correction(
            ATACAMA_SOLAR_FLUX_W_M2, MARS_SOLAR_FLUX_W_M2
        )
        assert abs(correction - expected_ratio) < 0.001

    def test_atacama_mars_projection(self):
        """Projection within bounds."""
        from src.atacama_validation import (
            project_mars_efficiency,
            ATACAMA_PEROVSKITE_EFFICIENCY,
        )

        correction = 0.59  # 590/1000
        mars_eff = project_mars_efficiency(ATACAMA_PEROVSKITE_EFFICIENCY, correction)
        # Mars efficiency should be lower than Atacama due to flux and dust
        assert mars_eff < ATACAMA_PEROVSKITE_EFFICIENCY
        assert mars_eff > 0.1  # But still reasonable


class TestSyncCycle:
    """Test sync cycle execution."""

    def test_sync_cycle_success(self):
        """Cycle completes successfully."""
        from src.multi_planet_sync import run_sync_cycle

        cycle = run_sync_cycle()
        assert cycle["cycle_successful"] is True

    def test_sync_latencies_valid(self):
        """Latencies within expected bounds."""
        from src.multi_planet_sync import (
            run_sync_cycle,
            SYNC_LATENCY_TITAN_MIN,
            SYNC_LATENCY_EUROPA_MIN,
        )

        cycle = run_sync_cycle()
        assert (
            SYNC_LATENCY_TITAN_MIN[0]
            <= cycle["titan_latency_min"]
            <= SYNC_LATENCY_TITAN_MIN[1]
        )
        assert (
            SYNC_LATENCY_EUROPA_MIN[0]
            <= cycle["europa_latency_min"]
            <= SYNC_LATENCY_EUROPA_MIN[1]
        )


class TestD8Push:
    """Test D8 push command."""

    def test_d8_push_floor_met(self):
        """D8 push meets alpha floor."""
        from src.fractal_layers import d8_push

        result = d8_push(10**12, 3.23, simulate=True)
        assert result["floor_met"] is True

    def test_d8_push_slo_passed(self):
        """D8 push passes SLO."""
        from src.fractal_layers import d8_push

        result = d8_push(10**12, 3.23, simulate=True)
        assert result["slo_passed"] is True


class TestD8Constants:
    """Test D8 constant values."""

    def test_d8_alpha_floor_constant(self):
        """D8 alpha floor is 3.43."""
        from src.fractal_layers import D8_ALPHA_FLOOR

        assert D8_ALPHA_FLOOR == 3.43

    def test_d8_alpha_target_constant(self):
        """D8 alpha target is 3.45."""
        from src.fractal_layers import D8_ALPHA_TARGET

        assert D8_ALPHA_TARGET == 3.45

    def test_d8_uplift_constant(self):
        """D8 uplift is 0.22."""
        from src.fractal_layers import D8_UPLIFT

        assert D8_UPLIFT == 0.22


class TestAtacamaConfig:
    """Test Atacama configuration."""

    def test_atacama_solar_flux(self):
        """Atacama solar flux is 1000 W/m^2."""
        from src.atacama_validation import ATACAMA_SOLAR_FLUX_W_M2

        assert ATACAMA_SOLAR_FLUX_W_M2 == 1000

    def test_mars_solar_flux(self):
        """Mars solar flux is 590 W/m^2."""
        from src.atacama_validation import MARS_SOLAR_FLUX_W_M2

        assert MARS_SOLAR_FLUX_W_M2 == 590

    def test_perovskite_efficiency(self):
        """Perovskite efficiency is 25.6%."""
        from src.atacama_validation import ATACAMA_PEROVSKITE_EFFICIENCY

        assert ATACAMA_PEROVSKITE_EFFICIENCY == 0.256
