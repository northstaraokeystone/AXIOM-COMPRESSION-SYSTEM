"""Tests for D11 recursion + Venus acid-cloud autonomy.

Test coverage:
- D11 spec loading and validation
- D11 alpha floor/target/ceiling
- Venus configuration and operations
- D11+Venus hybrid integration
"""


class TestD11Spec:
    """Tests for D11 specification loading."""

    def test_d11_spec_loads(self):
        """Test that D11 spec loads with valid structure."""
        from src.fractal_layers import get_d11_spec

        spec = get_d11_spec()
        assert spec is not None
        assert "d11_config" in spec
        assert "venus_config" in spec
        assert "cfd_config" in spec
        assert "secure_enclave_config" in spec

    def test_d11_version(self):
        """Test D11 spec version."""
        from src.fractal_layers import get_d11_spec

        spec = get_d11_spec()
        assert spec.get("version") == "1.0.0"

    def test_d11_alpha_floor(self):
        """Test D11 alpha floor is 3.58."""
        from src.fractal_layers import get_d11_spec, D11_ALPHA_FLOOR

        spec = get_d11_spec()
        assert spec["d11_config"]["alpha_floor"] == 3.58
        assert D11_ALPHA_FLOOR == 3.58

    def test_d11_alpha_target(self):
        """Test D11 alpha target is 3.60."""
        from src.fractal_layers import get_d11_spec, D11_ALPHA_TARGET

        spec = get_d11_spec()
        assert spec["d11_config"]["alpha_target"] == 3.60
        assert D11_ALPHA_TARGET == 3.60

    def test_d11_alpha_ceiling(self):
        """Test D11 alpha ceiling is 3.62."""
        from src.fractal_layers import get_d11_spec, D11_ALPHA_CEILING

        spec = get_d11_spec()
        assert spec["d11_config"]["alpha_ceiling"] == 3.62
        assert D11_ALPHA_CEILING == 3.62

    def test_d11_uplift_value(self):
        """Test D11 uplift is 0.28."""
        from src.fractal_layers import get_d11_spec, D11_UPLIFT

        spec = get_d11_spec()
        assert spec["d11_config"]["uplift"] == 0.28
        assert D11_UPLIFT == 0.28


class TestD11Recursion:
    """Tests for D11 recursion functionality."""

    def test_d11_recursive_fractal(self):
        """Test D11 recursive fractal computation."""
        from src.fractal_layers import d11_recursive_fractal

        result = d11_recursive_fractal(10**9, 3.0, depth=11)
        assert result is not None
        assert "eff_alpha" in result
        assert result["depth"] == 11

    def test_d11_achieves_floor(self):
        """Test D11 achieves alpha floor with sufficient tree size."""
        from src.fractal_layers import d11_recursive_fractal

        result = d11_recursive_fractal(10**12, 3.32, depth=11)
        assert result["eff_alpha"] >= 3.58
        assert result["floor_met"] is True

    def test_d11_push(self):
        """Test D11 push function."""
        from src.fractal_layers import d11_push

        result = d11_push(10**9, 3.0, simulate=True)
        assert result is not None
        assert result["mode"] == "simulate"
        assert result["depth"] == 11

    def test_d11_info(self):
        """Test D11 info function."""
        from src.fractal_layers import get_d11_info

        info = get_d11_info()
        assert info is not None
        assert "d11_config" in info
        assert "venus_config" in info


class TestVenusConfig:
    """Tests for Venus acid-cloud configuration."""

    def test_venus_config_loads(self):
        """Test Venus config loads."""
        from src.venus_acid_hybrid import load_venus_config

        config = load_venus_config()
        assert config is not None
        assert config["body"] == "venus"

    def test_venus_surface_temp(self):
        """Test Venus surface temperature is 465°C."""
        from src.venus_acid_hybrid import load_venus_config, VENUS_SURFACE_TEMP_C

        config = load_venus_config()
        assert config["surface_temp_c"] == 465
        assert VENUS_SURFACE_TEMP_C == 465

    def test_venus_surface_pressure(self):
        """Test Venus surface pressure is 92 atm."""
        from src.venus_acid_hybrid import load_venus_config, VENUS_SURFACE_PRESSURE_ATM

        config = load_venus_config()
        assert config["surface_pressure_atm"] == 92
        assert VENUS_SURFACE_PRESSURE_ATM == 92

    def test_venus_cloud_zone(self):
        """Test Venus cloud zone altitude range."""
        from src.venus_acid_hybrid import load_venus_config, VENUS_CLOUD_ALTITUDE_KM

        config = load_venus_config()
        assert config["cloud_altitude_km"] == [48, 70]
        assert VENUS_CLOUD_ALTITUDE_KM == (48, 70)

    def test_venus_acid_concentration(self):
        """Test Venus acid concentration is 0.85."""
        from src.venus_acid_hybrid import load_venus_config, VENUS_ACID_CONCENTRATION

        config = load_venus_config()
        assert config["acid_concentration"] == 0.85
        assert VENUS_ACID_CONCENTRATION == 0.85

    def test_venus_autonomy_requirement(self):
        """Test Venus autonomy requirement is 0.99."""
        from src.venus_acid_hybrid import load_venus_config, VENUS_AUTONOMY_REQUIREMENT

        config = load_venus_config()
        assert config["autonomy_requirement"] == 0.99
        assert VENUS_AUTONOMY_REQUIREMENT == 0.99

    def test_venus_hazards_present(self):
        """Test all 3 Venus hazards are listed."""
        from src.venus_acid_hybrid import load_venus_config, VENUS_HAZARDS

        config = load_venus_config()
        assert len(config["hazards"]) == 3
        assert "sulfuric_acid" in config["hazards"]
        assert "pressure" in config["hazards"]
        assert "temperature" in config["hazards"]
        assert VENUS_HAZARDS == ["sulfuric_acid", "pressure", "temperature"]


class TestVenusOperations:
    """Tests for Venus cloud operations."""

    def test_cloud_zone_in_habitable(self):
        """Test cloud zone at habitable altitude."""
        from src.venus_acid_hybrid import compute_cloud_zone

        result = compute_cloud_zone(55.0)
        assert result is not None
        assert result["in_habitable_zone"] is True
        assert result["altitude_km"] == 55.0

    def test_cloud_zone_outside_habitable(self):
        """Test cloud zone outside habitable altitude."""
        from src.venus_acid_hybrid import compute_cloud_zone

        result = compute_cloud_zone(30.0)
        assert result["in_habitable_zone"] is False

    def test_acid_resistance_ptfe(self):
        """Test acid resistance for PTFE (excellent)."""
        from src.venus_acid_hybrid import simulate_acid_resistance

        result = simulate_acid_resistance("ptfe")
        assert result["resistance_coefficient"] == 0.99
        assert result["suitable_for_venus"] is True

    def test_acid_resistance_aluminum(self):
        """Test acid resistance for aluminum (poor)."""
        from src.venus_acid_hybrid import simulate_acid_resistance

        result = simulate_acid_resistance("aluminum")
        assert result["resistance_coefficient"] == 0.30
        assert result["suitable_for_venus"] is False

    def test_cloud_ops_simulation(self):
        """Test cloud operations simulation."""
        from src.venus_acid_hybrid import simulate_cloud_ops

        result = simulate_cloud_ops(duration_days=30, altitude_km=55.0)
        assert result is not None
        assert result["duration_days"] == 30
        assert result["operations_viable"] is True
        assert "autonomy" in result

    def test_cloud_ops_autonomy_met(self):
        """Test cloud operations achieve autonomy requirement."""
        from src.venus_acid_hybrid import simulate_cloud_ops

        result = simulate_cloud_ops(duration_days=30, altitude_km=55.0)
        assert result["autonomy"] >= 0.99
        assert result["autonomy_met"] is True


class TestD11VenusHybrid:
    """Tests for D11+Venus hybrid integration."""

    def test_d11_venus_hybrid_runs(self):
        """Test D11+Venus hybrid runs."""
        from src.venus_acid_hybrid import d11_venus_hybrid

        result = d11_venus_hybrid(10**9, 3.0, simulate=True)
        assert result is not None
        assert result["integrated"] is True

    def test_d11_venus_hybrid_alpha(self):
        """Test D11+Venus hybrid achieves alpha floor."""
        from src.venus_acid_hybrid import d11_venus_hybrid

        result = d11_venus_hybrid(10**12, 3.32, simulate=True)
        assert result["d11_result"]["floor_met"] is True

    def test_d11_venus_hybrid_receipt(self):
        """Test D11+Venus hybrid emits combined receipt."""
        from src.venus_acid_hybrid import d11_venus_hybrid

        result = d11_venus_hybrid(10**9, 3.0, simulate=True)
        assert "d11_result" in result
        assert "venus_result" in result
        assert "combined_success" in result
