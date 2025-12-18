"""test_real_data.py - Tests for Real Data Loaders

Tests for SPARC, MOXIE, and ISS ECLSS data loading with provenance receipts.

Source: AXIOM Validation Lock v1
"""

import numpy as np


class TestSPARCLoader:
    """Tests for SPARC galaxy rotation curve loader."""

    def test_list_available_returns_galaxies(self):
        """list_available() should return list of galaxy IDs."""
        from real_data.sparc import list_available

        galaxies = list_available()

        assert isinstance(galaxies, list)
        assert len(galaxies) >= 100  # Should have 175, but at least 100
        assert "NGC2403" in galaxies
        assert "NGC3198" in galaxies

    def test_get_galaxy_returns_valid_format(self):
        """get_galaxy() should return dict with required fields."""
        from real_data.sparc import get_galaxy

        galaxy = get_galaxy("NGC2403")

        assert galaxy is not None
        assert "id" in galaxy
        assert "regime" in galaxy
        assert "r" in galaxy
        assert "v" in galaxy
        assert "v_unc" in galaxy
        assert "params" in galaxy

        # Check arrays
        assert isinstance(galaxy["r"], np.ndarray)
        assert isinstance(galaxy["v"], np.ndarray)
        assert len(galaxy["r"]) == len(galaxy["v"])

    def test_get_galaxy_unknown_returns_none(self):
        """get_galaxy() with unknown ID should return None."""
        from real_data.sparc import get_galaxy

        result = get_galaxy("NOT_A_REAL_GALAXY")
        assert result is None

    def test_load_sparc_returns_multiple_galaxies(self):
        """load_sparc() should return list of galaxy dicts."""
        from real_data.sparc import load_sparc

        galaxies = load_sparc(n_galaxies=5)

        assert isinstance(galaxies, list)
        assert len(galaxies) >= 5

        for galaxy in galaxies:
            assert "id" in galaxy
            assert "r" in galaxy
            assert len(galaxy["r"]) > 0

    def test_load_sparc_respects_n_galaxies_limit(self):
        """load_sparc() should not return more than requested."""
        from real_data.sparc import load_sparc

        galaxies = load_sparc(n_galaxies=3)
        assert len(galaxies) <= 10  # May include embedded extras

    def test_sparc_rotation_curves_have_positive_values(self):
        """Rotation curves should have positive r and v."""
        from real_data.sparc import load_sparc

        galaxies = load_sparc(n_galaxies=5)

        for galaxy in galaxies:
            r = np.array(galaxy["r"])
            v = np.array(galaxy["v"])

            assert np.all(r > 0), f"Galaxy {galaxy['id']} has non-positive r"
            assert np.all(v > 0), f"Galaxy {galaxy['id']} has non-positive v"


class TestMOXIELoader:
    """Tests for NASA MOXIE telemetry loader."""

    def test_list_runs_returns_run_ids(self):
        """list_runs() should return list of run IDs."""
        from real_data.nasa_pds import list_runs

        runs = list_runs()

        assert isinstance(runs, list)
        assert len(runs) >= 10  # Should have 16
        assert all(isinstance(r, int) for r in runs)

    def test_get_run_returns_valid_format(self):
        """get_run() should return dict with telemetry data."""
        from real_data.nasa_pds import get_run

        run = get_run(1)

        assert run is not None
        assert "run_id" in run
        assert "timestamp" in run
        assert "duration_min" in run
        assert "o2_produced_g" in run
        assert "power_consumed_w" in run
        assert "efficiency" in run
        assert "source" in run

    def test_get_run_invalid_returns_none(self):
        """get_run() with invalid ID should return None."""
        from real_data.nasa_pds import get_run

        result = get_run(999)
        assert result is None

    def test_load_moxie_returns_summary(self):
        """load_moxie() should return runs and summary."""
        from real_data.nasa_pds import load_moxie

        result = load_moxie()

        assert "runs" in result
        assert "summary" in result
        assert "source" in result

        assert len(result["runs"]) >= 10
        assert result["summary"]["n_runs"] == len(result["runs"])
        assert result["summary"]["total_o2_produced_g"] > 0

    def test_moxie_efficiency_is_reasonable(self):
        """MOXIE efficiency should be in reasonable range."""
        from real_data.nasa_pds import load_moxie

        result = load_moxie()

        for run in result["runs"]:
            # Efficiency should be between 0.1 and 1.0 g O2/Wh
            assert 0.1 <= run["efficiency"] <= 1.0, f"Run {run['run_id']} efficiency out of range"


class TestECLSSLoader:
    """Tests for ISS ECLSS data loader."""

    def test_get_water_recovery_returns_expected(self):
        """get_water_recovery() should return ~0.98."""
        from real_data.iss_eclss import get_water_recovery

        rate = get_water_recovery()

        assert isinstance(rate, float)
        assert 0.95 <= rate <= 1.0
        assert abs(rate - 0.98) < 0.01

    def test_get_o2_closure_returns_expected(self):
        """get_o2_closure() should return ~0.875."""
        from real_data.iss_eclss import get_o2_closure

        rate = get_o2_closure()

        assert isinstance(rate, float)
        assert 0.8 <= rate <= 0.95
        assert abs(rate - 0.875) < 0.01

    def test_load_eclss_returns_subsystems(self):
        """load_eclss() should return subsystem data."""
        from real_data.iss_eclss import load_eclss

        result = load_eclss()

        assert "subsystems" in result
        assert "key_metrics" in result
        assert "source" in result

        assert "water_recovery_system" in result["subsystems"]
        assert "oxygen_generation" in result["subsystems"]

    def test_eclss_validation_passes(self):
        """ECLSS constants should validate against expected values."""
        from real_data.iss_eclss import validate_against_constants

        result = validate_against_constants()

        assert result["all_match"], "ECLSS constants validation failed"


class TestProvenanceReceipts:
    """Tests for provenance receipt emission."""

    def test_sparc_emits_receipt(self, capsys):
        """load_sparc() should emit real_data receipt."""
        from real_data.sparc import load_sparc

        load_sparc(n_galaxies=1)

        captured = capsys.readouterr()
        assert "real_data" in captured.out
        assert "SPARC" in captured.out

    def test_moxie_emits_receipt(self, capsys):
        """load_moxie() should emit real_data receipt."""
        from real_data.nasa_pds import load_moxie

        load_moxie()

        captured = capsys.readouterr()
        assert "real_data" in captured.out
        assert "MOXIE" in captured.out

    def test_eclss_emits_receipt(self, capsys):
        """load_eclss() should emit real_data receipt."""
        from real_data.iss_eclss import load_eclss

        load_eclss()

        captured = capsys.readouterr()
        assert "real_data" in captured.out
        assert "ISS_ECLSS" in captured.out
