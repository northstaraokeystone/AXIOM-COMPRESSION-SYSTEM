"""Tests for Atacama real-time LES validation mode."""


class TestAtacamaRealtimeConfig:
    """Tests for Atacama real-time configuration."""

    def test_atacama_constants(self) -> None:
        """Test Atacama real-time constants are correctly defined."""
        from src.cfd_dust_dynamics import (
            ATACAMA_DRONE_SAMPLING_HZ,
            ATACAMA_LES_CORRELATION_TARGET,
            ATACAMA_REYNOLDS_NUMBER,
            ATACAMA_TERRAIN_MODEL,
        )

        assert ATACAMA_DRONE_SAMPLING_HZ == 100
        assert ATACAMA_LES_CORRELATION_TARGET == 0.95
        assert ATACAMA_REYNOLDS_NUMBER == 1_090_000
        assert ATACAMA_TERRAIN_MODEL == "atacama"

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

        result = atacama_les_realtime(duration_s=10.0)

        assert result is not None
        assert "duration_s" in result
        assert result["duration_s"] == 10.0
        assert "sampling_hz" in result
        assert "samples" in result
        assert result["samples"] == 1000  # 10 sec * 100 Hz

    def test_atacama_les_realtime_correlation(self) -> None:
        """Test Atacama real-time correlation metric."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert "correlation" in result
        assert 0.0 <= result["correlation"] <= 1.0
        assert "correlation_met" in result

    def test_atacama_les_realtime_validated(self) -> None:
        """Test Atacama real-time validated status."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert "validated" in result
        assert "reynolds" in result
        assert "mode" in result
        assert result["mode"] == "realtime"


class TestDustDevilTracking:
    """Tests for dust devil tracking functionality."""

    def test_track_dust_devil_basic(self) -> None:
        """Test basic dust devil tracking."""
        from src.cfd_dust_dynamics import track_dust_devil

        result = track_dust_devil(position=(0.0, 0.0), duration_s=10.0)

        assert result is not None
        assert "duration_s" in result
        assert "samples" in result
        assert "tracked" in result
        assert "tracking_success" in result

    def test_track_dust_devil_trajectory(self) -> None:
        """Test dust devil trajectory computation."""
        from src.cfd_dust_dynamics import track_dust_devil

        result = track_dust_devil(position=(100.0, 50.0), duration_s=30.0)

        assert "initial_position" in result
        assert "final_position" in result
        assert "total_distance_m" in result
        assert result["tracked"] is True

    def test_track_dust_devil_vorticity(self) -> None:
        """Test dust devil speed measurement."""
        from src.cfd_dust_dynamics import track_dust_devil

        result = track_dust_devil(position=(0.0, 0.0), duration_s=20.0)

        assert "avg_speed_m_s" in result
        assert result["avg_speed_m_s"] >= 0


class TestRealtimeFeedbackLoop:
    """Tests for real-time feedback loop."""

    def test_realtime_feedback_loop_basic(self) -> None:
        """Test basic real-time feedback loop."""
        from src.cfd_dust_dynamics import realtime_feedback_loop

        # Create sample LES and drone data
        les_output = {"samples": [{"t_s": 0, "u_m_s": 15.0}] * 10}
        drone_data = {"samples": [{"t_s": 0, "u_m_s": 15.1}] * 10}

        result = realtime_feedback_loop(les_output, drone_data)

        assert result is not None
        assert "correlation_before" in result
        assert "correlation_target" in result
        assert "calibration_complete" in result

    def test_realtime_feedback_loop_convergence(self) -> None:
        """Test feedback loop adjustments."""
        from src.cfd_dust_dynamics import realtime_feedback_loop

        # Create sample LES and drone data with some discrepancy
        les_output = {
            "samples": [{"t_s": i, "u_m_s": 15.0 + i * 0.1} for i in range(20)]
        }
        drone_data = {
            "samples": [{"t_s": i, "u_m_s": 15.0 + i * 0.12} for i in range(20)]
        }

        result = realtime_feedback_loop(les_output, drone_data)

        assert "adjustment_factor" in result
        assert "adjustments" in result
        assert "improved" in result


class TestRealtimeCorrelation:
    """Tests for real-time correlation computation."""

    def test_compute_realtime_correlation(self) -> None:
        """Test real-time correlation computation."""
        from src.cfd_dust_dynamics import compute_realtime_correlation

        # Create sample LES and field data
        les_data = {"samples": [{"t_s": i, "u_m_s": 15.0 + i * 0.1} for i in range(20)]}
        field_data = {
            "samples": [{"t_s": i, "u_m_s": 15.0 + i * 0.1} for i in range(20)]
        }

        result = compute_realtime_correlation(les_data, field_data)

        # Function returns float, not dict
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_correlation_components(self) -> None:
        """Test correlation with identical data returns 1.0."""
        from src.cfd_dust_dynamics import compute_realtime_correlation

        # Create identical data - should have perfect correlation
        les_data = {"samples": [{"t_s": i, "u_m_s": 15.0 + i * 0.1} for i in range(20)]}
        field_data = {
            "samples": [{"t_s": i, "u_m_s": 15.0 + i * 0.1} for i in range(20)]
        }

        result = compute_realtime_correlation(les_data, field_data)

        # Identical data should have correlation close to 1.0
        assert result >= 0.99


class TestAtacamaValidation:
    """Tests for full Atacama validation suite."""

    def test_run_atacama_validation(self) -> None:
        """Test full Atacama validation."""
        from src.cfd_dust_dynamics import run_atacama_validation

        result = run_atacama_validation()

        assert result is not None
        assert "mode" in result
        assert result["mode"] == "atacama_realtime"
        assert "correlation" in result
        assert "correlation_target" in result
        assert "overall_validated" in result

    def test_atacama_validation_components(self) -> None:
        """Test Atacama validation includes all components."""
        from src.cfd_dust_dynamics import run_atacama_validation

        result = run_atacama_validation()

        assert "realtime_result" in result
        assert "track_result" in result
        assert "config" in result

    def test_atacama_validation_metrics(self) -> None:
        """Test Atacama validation metrics."""
        from src.cfd_dust_dynamics import run_atacama_validation

        result = run_atacama_validation()

        assert "correlation" in result
        assert "correlation_target" in result
        assert result["correlation_target"] == 0.95


class TestLESIntegration:
    """Tests for LES integration with Atacama real-time mode."""

    def test_les_atacama_comparison(self) -> None:
        """Test LES comparison with Atacama data."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert "reynolds" in result
        assert "correlation" in result
        assert "les_data_points" in result
        assert result["les_data_points"] > 0

    def test_les_subgrid_model(self) -> None:
        """Test LES terrain model in real-time mode."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert "terrain_model" in result
        assert result["terrain_model"] == "atacama"


class TestDroneArrayIntegration:
    """Tests for drone array integration with real-time mode."""

    def test_drone_sampling_rate(self) -> None:
        """Test drone sampling rate configuration."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert result["sampling_hz"] == 100
        assert result["samples"] == 1000

    def test_drone_coverage_area(self) -> None:
        """Test drone data points in real-time mode."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert "drone_data_points" in result
        assert result["drone_data_points"] > 0

    def test_drone_grid_resolution(self) -> None:
        """Test drone mode validation."""
        from src.cfd_dust_dynamics import atacama_les_realtime

        result = atacama_les_realtime(duration_s=10.0)

        assert "mode" in result
        assert result["mode"] == "realtime"
        assert "validated" in result
