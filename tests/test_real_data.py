"""test_real_data.py - Tests for real data loaders.

Tests:
- SPARC galaxy loader
- NASA PDS MOXIE loader
- ISS ECLSS data loader
- Landauer mass equivalent with uncertainty
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_data.sparc import (
    SPARC_RANDOM_SEED,
    SPARC_TOTAL_GALAXIES,
    load_sparc,
    get_galaxy,
    list_available,
)
from real_data.nasa_pds import (
    PDS_RELEASE_VERSION,
    MOXIE_RUN_COUNT,
    MOXIE_EFFICIENCY_MIN,
    MOXIE_EFFICIENCY_MAX,
    MOXIE_EFFICIENCY_MEAN,
    MOXIE_EFFICIENCY_VARIANCE_PCT,
    load_moxie,
    get_run,
    list_runs,
    get_efficiency_stats,
)
from real_data.iss_eclss import (
    ISS_WATER_RECOVERY,
    ISS_O2_CLOSURE,
    load_eclss,
    get_water_recovery,
    get_o2_closure,
)
from src.entropy import (
    landauer_mass_equivalent,
    BASELINE_MASS_KG,
)


class TestSPARCLoader:
    """Tests for SPARC galaxy rotation curve loader."""

    def test_load_sparc_returns_list(self):
        """load_sparc should return a list of dicts."""
        galaxies = load_sparc(n_galaxies=5, seed=42)
        assert isinstance(galaxies, list)
        assert len(galaxies) == 5

    def test_galaxy_structure(self):
        """Each galaxy should have required fields."""
        galaxies = load_sparc(n_galaxies=3, seed=42)

        for g in galaxies:
            assert 'id' in g
            assert 'r' in g
            assert 'v' in g
            assert 'v_unc' in g
            assert 'params' in g
            assert g['params']['source'] == 'SPARC'

    def test_seed_constant(self):
        """Default seed should be 42."""
        assert SPARC_RANDOM_SEED == 42

    def test_total_galaxies(self):
        """Total should be 175."""
        assert SPARC_TOTAL_GALAXIES == 175

    def test_list_available(self):
        """Should list all 175 galaxy IDs."""
        ids = list_available()
        assert len(ids) == SPARC_TOTAL_GALAXIES

    def test_get_galaxy(self):
        """Should get single galaxy by ID."""
        galaxy = get_galaxy("NGC2403")
        assert galaxy['id'] == "NGC2403"

    def test_invalid_galaxy_id(self):
        """Should raise error for invalid ID."""
        with pytest.raises(ValueError):
            get_galaxy("INVALID_ID")

    def test_receipt_emission(self, capsys):
        """Should emit real_data receipt."""
        load_sparc(n_galaxies=3, seed=42)

        captured = capsys.readouterr()
        assert '"receipt_type": "real_data"' in captured.out
        assert '"dataset_id": "SPARC"' in captured.out
        assert '"random_seed": 42' in captured.out


class TestMOXIELoader:
    """Tests for NASA PDS MOXIE loader."""

    def test_pds_version_pin(self):
        """PDS release version should be pinned to 14."""
        assert PDS_RELEASE_VERSION == "14"

    def test_moxie_run_count(self):
        """Should have 16 runs."""
        assert MOXIE_RUN_COUNT == 16

    def test_efficiency_constants(self):
        """Efficiency constants should match spec."""
        assert MOXIE_EFFICIENCY_MIN == 5.0
        assert MOXIE_EFFICIENCY_MAX == 6.1
        assert MOXIE_EFFICIENCY_MEAN == 5.5
        assert abs(MOXIE_EFFICIENCY_VARIANCE_PCT - 0.12) < 0.01

    def test_load_moxie(self):
        """load_moxie should return dict with runs and stats."""
        data = load_moxie()

        assert 'runs' in data
        assert 'stats' in data
        assert len(data['runs']) == MOXIE_RUN_COUNT
        assert data['pds_release'] == PDS_RELEASE_VERSION

    def test_run_structure(self):
        """Each run should have required fields."""
        data = load_moxie()

        for run in data['runs']:
            assert 'run_id' in run
            assert 'timestamp' in run
            assert 'efficiency' in run
            assert 'efficiency_uncertainty_pct' in run
            assert 'pds_release' in run
            assert run['source'] == 'NASA_PDS'

    def test_get_run(self):
        """Should get single run by ID."""
        run = get_run(1)
        assert run['run_id'] == 1

    def test_invalid_run_id(self):
        """Should raise error for invalid run ID."""
        with pytest.raises(ValueError):
            get_run(0)
        with pytest.raises(ValueError):
            get_run(17)

    def test_list_runs(self):
        """Should list all 16 run IDs."""
        runs = list_runs()
        assert runs == list(range(1, 17))

    def test_efficiency_stats(self):
        """get_efficiency_stats should return aggregate stats."""
        stats = get_efficiency_stats()

        assert stats['mean'] == MOXIE_EFFICIENCY_MEAN
        assert stats['min'] == MOXIE_EFFICIENCY_MIN
        assert stats['max'] == MOXIE_EFFICIENCY_MAX
        assert stats['variance_pct'] == MOXIE_EFFICIENCY_VARIANCE_PCT
        assert stats['n_runs'] == MOXIE_RUN_COUNT

    def test_receipt_emission(self, capsys):
        """Should emit real_data receipt with PDS version."""
        load_moxie()

        captured = capsys.readouterr()
        assert '"receipt_type": "real_data"' in captured.out
        assert '"dataset_id": "MOXIE"' in captured.out
        assert '"pds_release_version": "14"' in captured.out


class TestISSECLSSLoader:
    """Tests for ISS ECLSS data loader."""

    def test_water_recovery_constant(self):
        """Water recovery should be 0.98."""
        assert ISS_WATER_RECOVERY == 0.98

    def test_o2_closure_constant(self):
        """O2 closure should be 0.875."""
        assert ISS_O2_CLOSURE == 0.875

    def test_load_eclss(self):
        """load_eclss should return dict with ECLSS data."""
        data = load_eclss()

        assert data['water_recovery_rate'] == ISS_WATER_RECOVERY
        assert data['o2_closure_rate'] == ISS_O2_CLOSURE
        assert data['source'] == 'NASA_ECLSS_2023'

    def test_get_water_recovery(self):
        """get_water_recovery should match constant."""
        assert get_water_recovery() == ISS_WATER_RECOVERY

    def test_get_o2_closure(self):
        """get_o2_closure should match constant."""
        assert get_o2_closure() == ISS_O2_CLOSURE

    def test_receipt_emission(self, capsys):
        """Should emit real_data receipt."""
        load_eclss()

        captured = capsys.readouterr()
        assert '"receipt_type": "real_data"' in captured.out
        assert '"dataset_id": "ISS_ECLSS"' in captured.out


class TestLandauerUncertainty:
    """Tests for Landauer mass equivalent with uncertainty (v2 FIX #2)."""

    def test_returns_dict_with_uncertainty(self):
        """Should return dict with uncertainty fields."""
        result = landauer_mass_equivalent(1e6)

        assert 'value' in result
        assert 'uncertainty_pct' in result
        assert 'confidence_interval_lower' in result
        assert 'confidence_interval_upper' in result
        assert 'calibration_source' in result

    def test_uncertainty_from_moxie(self):
        """Uncertainty should be derived from MOXIE variance."""
        result = landauer_mass_equivalent(1e6)
        assert result['uncertainty_pct'] == MOXIE_EFFICIENCY_VARIANCE_PCT

    def test_uncertainty_within_15_percent(self):
        """Uncertainty should be <= 15%."""
        result = landauer_mass_equivalent(1e6)
        assert result['uncertainty_pct'] <= 0.15

    def test_ci_contains_baseline(self):
        """CI must contain 60,000 kg baseline."""
        result = landauer_mass_equivalent(1e6)

        ci_lower = result['confidence_interval_lower']
        ci_upper = result['confidence_interval_upper']

        # CI should be [51000, 69000] approximately
        assert ci_lower < BASELINE_MASS_KG < ci_upper, \
            f"CI [{ci_lower:.0f}, {ci_upper:.0f}] must contain {BASELINE_MASS_KG:.0f}"

    def test_ci_calculation(self):
        """CI should be value * (1 +/- uncertainty)."""
        result = landauer_mass_equivalent(1e6)

        value = result['value']
        pct = result['uncertainty_pct']

        expected_lower = value * (1 - pct)
        expected_upper = value * (1 + pct)

        assert abs(result['confidence_interval_lower'] - expected_lower) < 1
        assert abs(result['confidence_interval_upper'] - expected_upper) < 1

    def test_calibration_source(self):
        """Calibration source should be MOXIE_2025_PDS14."""
        result = landauer_mass_equivalent(1e6)
        assert result['calibration_source'] == 'MOXIE_2025_PDS14'

    def test_no_uncertainty_mode(self):
        """Should work without uncertainty if disabled."""
        result = landauer_mass_equivalent(1e6, include_uncertainty=False)

        assert 'value' in result
        # Uncertainty fields should not be present or be zero
        assert result.get('uncertainty_pct', 0) == 0 or 'uncertainty_pct' not in result


class TestZenodoMetadata:
    """Tests for Zenodo metadata template (v2 FIX #3)."""

    def test_metadata_file_exists(self):
        """zenodo.json should exist."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "zenodo", "zenodo.json"
        )
        assert os.path.exists(path)

    def test_required_fields_present(self):
        """Zenodo metadata should have all required fields."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "zenodo", "zenodo.json"
        )

        with open(path, 'r') as f:
            metadata = json.load(f)

        required = ['title', 'upload_type', 'description', 'creators', 'license', 'access_right']
        for field in required:
            assert field in metadata, f"Missing required field: {field}"

    def test_upload_type_is_software(self):
        """upload_type should be 'software'."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "zenodo", "zenodo.json"
        )

        with open(path, 'r') as f:
            metadata = json.load(f)

        assert metadata['upload_type'] == 'software'

    def test_has_keywords(self):
        """Should have keywords for discoverability."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "zenodo", "zenodo.json"
        )

        with open(path, 'r') as f:
            metadata = json.load(f)

        assert 'keywords' in metadata
        assert len(metadata['keywords']) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
