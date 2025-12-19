"""Tests for Atacama real-time LES validation mode."""


class TestAtacamaRealtimeConfig:
    """Tests for Atacama real-time configuration."""

    def test_atacama_constants(self) -> None:
        """Test Atacama real-time constants are correctly defined."""
        from src.cfd_dust_dynamics import (
            ATACAMA_DRONE_SAMPLING_HZ,
            ATACAMA_LES_CORRELATION_TARGET,
            ATACAMA_REYNOLDS_NUMBER,
            ATACAMA_DRONE_ALTITUDE_M,
            ATACAMA_GRID_SIZE_M,
        )

        assert ATACAMA_DRONE_SAMPLING_HZ == 100
        assert ATACAMA_LES_CORRELATION_TARGET == 0.95
        assert ATACAMA_REYNOLDS_NUMBER == 1_090_000
        assert ATACAMA_DRONE_ALTITUDE_M == 50
        assert ATACAMA_GRID_SIZE_M == 1000

    def test_load_atacama_config(self) -> None:
        """Test loading Atacama real-time config from spec."""
        from src.cfd_dust_dynamics import load_atacama_realtime_config

        config = load_atacama_realtime_config()
        assert config is not None
        assert "drone_sampling_hz" in config
        assert config["drone_sampling_hz"] == 100
        assert "les_correlation_target" in config
        assert config["les_correlation_target"] == 0.95


class TestAtacamaRealtimeLES:
    """Tests for Atacama real-time LES simulation."""

    def test_atacama_les_realtime_basic(self) -> None:
        """Test basic Atacama real-time LES execution."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=10.0,
            sampling_hz=100,
        )

        assert result is not None
        assert "duration_sec" in result
        assert result["duration_sec"] == 10.0
        assert "sampling_hz" in result
        assert result["sampling_hz"] == 100
        assert "samples_collected" in result
        assert result["samples_collected"] == 1000  # 10 sec * 100 Hz

    def test_atacama_les_realtime_correlation(self) -> None:
        """Test Atacama real-time correlation metric."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=60.0,
            sampling_hz=100,
        )

        assert "correlation" in result
        assert 0.0 <= result["correlation"] <= 1.0
        assert "target_met" in result

    def test_atacama_les_realtime_velocity_stats(self) -> None:
        """Test Atacama real-time velocity statistics."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=30.0,
            sampling_hz=100,
        )

        assert "velocity_stats" in result
        stats = result["velocity_stats"]
        assert "mean_ms" in stats
        assert "max_ms" in stats
        assert "min_ms" in stats


class TestDustDevilTracking:
    """Tests for dust devil tracking functionality."""

    def test_track_dust_devil_basic(self) -> None:
        """Test basic dust devil tracking."""
        from src.cfd_dust_dynamics import track_dust_devil

        result = track_dust_devil(duration_sec=10.0)

        assert result is not None
        assert "duration_sec" in result
        assert "samples_collected" in result
        assert "max_velocity_ms" in result
        assert "max_diameter_m" in result
        assert "max_height_m" in result

    def test_track_dust_devil_trajectory(self) -> None:
        """Test dust devil trajectory computation."""
        from src.cfd_dust_dynamics import track_dust_devil

        result = track_dust_devil(duration_sec=30.0)

        assert "trajectory" in result
        trajectory = result["trajectory"]
        assert "start_position" in trajectory
        assert "end_position" in trajectory
        assert "path_length_m" in trajectory

    def test_track_dust_devil_vorticity(self) -> None:
        """Test dust devil vorticity measurement."""
        from src.cfd_dust_dynamics import track_dust_devil

        result = track_dust_devil(duration_sec=20.0)

        assert "vorticity" in result
        vorticity = result["vorticity"]
        assert "max_vorticity" in vorticity
        assert "mean_vorticity" in vorticity


class TestRealtimeFeedbackLoop:
    """Tests for real-time feedback loop."""

    def test_realtime_feedback_loop_basic(self) -> None:
        """Test basic real-time feedback loop."""
        from src.cfd_dust_dynamics import realtime_feedback_loop

        result = realtime_feedback_loop(
            duration_sec=5.0,
            sampling_hz=100,
        )

        assert result is not None
        assert "duration_sec" in result
        assert "feedback_cycles" in result
        assert "final_correlation" in result

    def test_realtime_feedback_loop_convergence(self) -> None:
        """Test feedback loop convergence."""
        from src.cfd_dust_dynamics import realtime_feedback_loop

        result = realtime_feedback_loop(
            duration_sec=30.0,
            sampling_hz=100,
        )

        assert "convergence_achieved" in result
        assert "convergence_time_sec" in result


class TestRealtimeCorrelation:
    """Tests for real-time correlation computation."""

    def test_compute_realtime_correlation(self) -> None:
        """Test real-time correlation computation."""
        from src.cfd_dust_dynamics import compute_realtime_correlation

        result = compute_realtime_correlation()

        assert result is not None
        assert "correlation" in result
        assert "target" in result
        assert result["target"] == 0.95
        assert "target_met" in result

    def test_correlation_components(self) -> None:
        """Test correlation component breakdown."""
        from src.cfd_dust_dynamics import compute_realtime_correlation

        result = compute_realtime_correlation()

        assert "components" in result
        components = result["components"]
        assert "velocity_correlation" in components
        assert "temperature_correlation" in components
        assert "pressure_correlation" in components


class TestAtacamaValidation:
    """Tests for full Atacama validation suite."""

    def test_run_atacama_validation(self) -> None:
        """Test full Atacama validation."""
        from src.cfd_dust_dynamics import run_atacama_validation

        result = run_atacama_validation()

        assert result is not None
        assert "validation_type" in result
        assert result["validation_type"] == "atacama_realtime"
        assert "tests_passed" in result
        assert "tests_total" in result
        assert "overall_pass" in result

    def test_atacama_validation_components(self) -> None:
        """Test Atacama validation includes all components."""
        from src.cfd_dust_dynamics import run_atacama_validation

        result = run_atacama_validation()

        assert "realtime_les" in result
        assert "dust_devil_tracking" in result
        assert "correlation_check" in result
        assert "feedback_loop" in result

    def test_atacama_validation_metrics(self) -> None:
        """Test Atacama validation metrics."""
        from src.cfd_dust_dynamics import run_atacama_validation

        result = run_atacama_validation()

        assert "metrics" in result
        metrics = result["metrics"]
        assert "sampling_rate_hz" in metrics
        assert "correlation_achieved" in metrics
        assert "target_correlation" in metrics


class TestLESIntegration:
    """Tests for LES integration with Atacama real-time mode."""

    def test_les_atacama_comparison(self) -> None:
        """Test LES comparison with Atacama data."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=30.0,
            sampling_hz=100,
        )

        assert "les_comparison" in result
        comparison = result["les_comparison"]
        assert "model_type" in comparison
        assert comparison["model_type"] == "LES"
        assert "reynolds_number" in comparison

    def test_les_subgrid_model(self) -> None:
        """Test LES subgrid model in real-time mode."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=20.0,
            sampling_hz=100,
        )

        assert "subgrid_model" in result
        subgrid = result["subgrid_model"]
        assert "type" in subgrid
        assert subgrid["type"] == "smagorinsky"
        assert "coefficient" in subgrid


class TestDroneArrayIntegration:
    """Tests for drone array integration with real-time mode."""

    def test_drone_sampling_rate(self) -> None:
        """Test drone sampling rate configuration."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=10.0,
            sampling_hz=100,
        )

        assert result["sampling_hz"] == 100
        assert result["samples_collected"] == 1000

    def test_drone_coverage_area(self) -> None:
        """Test drone coverage area in real-time mode."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=30.0,
            sampling_hz=100,
        )

        assert "drone_config" in result
        config = result["drone_config"]
        assert "coverage_area_km2" in config
        assert "altitude_m" in config

    def test_drone_grid_resolution(self) -> None:
        """Test drone grid resolution."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(
            duration_sec=20.0,
            sampling_hz=100,
        )

        assert "grid" in result
        grid = result["grid"]
        assert "resolution_m" in grid
        assert "size_m" in grid
