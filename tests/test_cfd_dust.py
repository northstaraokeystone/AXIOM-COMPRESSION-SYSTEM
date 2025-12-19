"""Tests for CFD Navier-Stokes dust dynamics.

Test coverage:
- CFD configuration loading
- Reynolds number calculation
- Stokes settling velocity
- Atacama validation
"""


class TestCFDConfig:
    """Tests for CFD configuration loading."""

    def test_cfd_config_loads(self):
        """Test CFD config loads."""
        from src.cfd_dust_dynamics import load_cfd_config

        config = load_cfd_config()
        assert config is not None
        assert config["model"] == "navier_stokes"

    def test_cfd_reynolds_mars(self):
        """Test Mars Reynolds number is 50."""
        from src.cfd_dust_dynamics import load_cfd_config, CFD_REYNOLDS_NUMBER_MARS

        config = load_cfd_config()
        assert config["reynolds_number_mars"] == 50
        assert CFD_REYNOLDS_NUMBER_MARS == 50

    def test_cfd_gravity_mars(self):
        """Test Mars gravity is 3.71 m/s^2."""
        from src.cfd_dust_dynamics import load_cfd_config, CFD_GRAVITY_MARS_M_S2

        config = load_cfd_config()
        assert config["gravity_mars_m_s2"] == 3.71
        assert CFD_GRAVITY_MARS_M_S2 == 3.71

    def test_cfd_settling_model(self):
        """Test settling model is Stokes."""
        from src.cfd_dust_dynamics import load_cfd_config, CFD_SETTLING_MODEL

        config = load_cfd_config()
        assert config["settling_model"] == "stokes"
        assert CFD_SETTLING_MODEL == "stokes"

    def test_cfd_turbulence_model(self):
        """Test turbulence model is laminar."""
        from src.cfd_dust_dynamics import load_cfd_config, CFD_TURBULENCE_MODEL

        config = load_cfd_config()
        assert config["turbulence_model"] == "laminar"
        assert CFD_TURBULENCE_MODEL == "laminar"

    def test_cfd_particle_size_range(self):
        """Test particle size range is [1, 100] um."""
        from src.cfd_dust_dynamics import load_cfd_config, CFD_PARTICLE_SIZE_UM

        config = load_cfd_config()
        assert config["particle_size_um"] == [1, 100]
        assert CFD_PARTICLE_SIZE_UM == (1, 100)

    def test_cfd_validated(self):
        """Test CFD is validated."""
        from src.cfd_dust_dynamics import load_cfd_config

        config = load_cfd_config()
        assert config["validated"] is True


class TestReynoldsNumber:
    """Tests for Reynolds number calculation."""

    def test_reynolds_calculation(self):
        """Test Reynolds number calculation."""
        from src.cfd_dust_dynamics import compute_reynolds_number

        # Low velocity, small length = low Re
        re = compute_reynolds_number(velocity=1.0, length=0.001)
        assert re > 0

    def test_reynolds_laminar_regime(self):
        """Test Reynolds number indicates laminar regime."""
        from src.cfd_dust_dynamics import compute_reynolds_number

        re = compute_reynolds_number(velocity=1.0, length=0.001)
        # Should be well below turbulent transition (2300)
        assert re < 2300


class TestStokesSettling:
    """Tests for Stokes settling velocity."""

    def test_stokes_settling_calculation(self):
        """Test Stokes settling velocity calculation."""
        from src.cfd_dust_dynamics import stokes_settling

        v_s = stokes_settling(particle_size_um=10.0)
        assert v_s > 0

    def test_stokes_settling_larger_faster(self):
        """Test larger particles settle faster."""
        from src.cfd_dust_dynamics import stokes_settling

        v_s_small = stokes_settling(particle_size_um=1.0)
        v_s_large = stokes_settling(particle_size_um=100.0)
        assert v_s_large > v_s_small

    def test_stokes_settling_physics_correct(self):
        """Test Stokes settling follows r^2 relationship."""
        from src.cfd_dust_dynamics import stokes_settling

        v_s_1 = stokes_settling(particle_size_um=10.0)
        v_s_2 = stokes_settling(particle_size_um=20.0)
        # v_s scales with r^2, so 2x diameter = 4x velocity
        ratio = v_s_2 / v_s_1
        assert 3.5 < ratio < 4.5  # Allow some tolerance


class TestDustStorm:
    """Tests for dust storm simulation."""

    def test_dust_storm_simulation(self):
        """Test dust storm simulation runs."""
        from src.cfd_dust_dynamics import simulate_dust_storm

        result = simulate_dust_storm(intensity=0.5, duration_hrs=24.0)
        assert result is not None
        assert result["intensity"] == 0.5
        assert result["duration_hrs"] == 24.0

    def test_dust_storm_intensity_effect(self):
        """Test storm intensity affects wind speed."""
        from src.cfd_dust_dynamics import simulate_dust_storm

        result_low = simulate_dust_storm(intensity=0.1)
        result_high = simulate_dust_storm(intensity=0.9)
        assert result_high["wind_speed_m_s"] > result_low["wind_speed_m_s"]


class TestAtacamaValidation:
    """Tests for Atacama validation."""

    def test_atacama_validation(self):
        """Test Atacama validation runs."""
        from src.cfd_dust_dynamics import validate_against_atacama, stokes_settling

        settling = stokes_settling(10.0)
        result = validate_against_atacama({"settling_velocity_m_s": settling})
        assert result is not None
        assert "correlation" in result

    def test_full_cfd_validation(self):
        """Test full CFD validation."""
        from src.cfd_dust_dynamics import run_cfd_validation

        result = run_cfd_validation()
        assert result is not None
        assert result["cfd_model"] == "navier_stokes"

    def test_cfd_dust_receipt(self):
        """Test CFD dust receipt emitted."""
        from src.cfd_dust_dynamics import run_cfd_validation

        result = run_cfd_validation()
        # Receipt should be emitted (tested via result structure)
        assert "overall_validated" in result
