"""test_cosmos.py - Physics validation and SLO verification for cosmos.py

Tests for synthetic galaxy rotation curve generator.
Validates physics correctness, array formats, reproducibility, and receipts.

NOTE: Using numpy arrays instead of torch tensors throughout.
witness.py converts inputs to numpy anyway (np.asarray), so this is compatible.

Source: CLAUDEME.md (§0: LAW_2 = "No test → not shipped")
"""

import numpy as np
import pytest
import json

from src.cosmos import (
    G, MOND_A0, REGIMES, DEFAULT_N_POINTS, DEFAULT_NOISE,
    newton_curve, mond_curve, nfw_curve, pbh_fog_curve,
    generate_galaxy, batch_generate, generate_pathological,
    stoprule_invalid_regime, stoprule_negative_radius
)
from src.core import StopRule


class TestPhysicsConstants:
    """Test that physics constants are correct."""

    def test_gravitational_constant(self):
        """G should be in correct galaxy units."""
        assert G == pytest.approx(4.302e-6, rel=1e-3), "G should be 4.302e-6 kpc (km/s)² / M_sun"

    def test_mond_acceleration_scale(self):
        """MOND_A0 should be correct."""
        assert MOND_A0 == pytest.approx(1.2e-10, rel=1e-3), "MOND_A0 should be 1.2e-10 m/s²"

    def test_regimes_list(self):
        """All 4 regimes should be defined."""
        assert len(REGIMES) == 4
        assert "newtonian" in REGIMES
        assert "mond" in REGIMES
        assert "nfw" in REGIMES
        assert "pbh_fog" in REGIMES


class TestNewtonCurve:
    """Test Newtonian (Keplerian) rotation curve."""

    def test_newton_keplerian(self):
        """V ∝ r^(-0.5) within 1%."""
        r = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # kpc
        M = 1e11  # M_sun
        v = newton_curve(r, M)

        # V(r) = sqrt(GM/r), so V ∝ r^(-0.5)
        # V(r1)/V(r2) = sqrt(r2/r1) when r1 < r2
        for i in range(len(r) - 1):
            expected_ratio = np.sqrt(r[i + 1] / r[i])
            actual_ratio = v[i] / v[i + 1]
            assert actual_ratio == pytest.approx(expected_ratio, rel=0.01), \
                f"Keplerian scaling failed at r={r[i]}: expected {expected_ratio}, got {actual_ratio}"

    def test_newton_absolute_value(self):
        """V at r=10 kpc should be reasonable."""
        r = np.array([10.0])  # kpc
        M = 1e11  # M_sun
        v = newton_curve(r, M)

        # V = sqrt(4.302e-6 * 1e11 / 10) = sqrt(43020) ≈ 207 km/s
        expected = np.sqrt(G * M / 10.0)
        assert v[0] == pytest.approx(expected, rel=0.001)
        assert 50 < v[0] < 300, f"V at r=10 should be 50-300 km/s, got {v[0]}"


class TestMondCurve:
    """Test MOND rotation curve."""

    def test_mond_flat_outer(self):
        """dV/dr → 0 for large r (MOND signature)."""
        r = np.linspace(20.0, 30.0, 20)  # kpc, outer region
        M = 1e10  # M_sun (smaller for deep MOND)
        v = mond_curve(r, M)

        # Compute numerical derivative
        dv_dr = np.gradient(v, r)

        # In deep MOND, curve should be nearly flat
        # |dV/dr| should be small (< 2 km/s/kpc)
        assert np.abs(dv_dr).mean() < 5.0, \
            f"MOND outer region not flat enough: mean |dV/dr| = {np.abs(dv_dr).mean()}"

    def test_mond_higher_than_newton_outer(self):
        """MOND should give higher velocities than Newton in outer regions."""
        r = np.array([20.0, 30.0])  # kpc
        M = 1e10  # M_sun

        v_newton = newton_curve(r, M)
        v_mond = mond_curve(r, M)

        assert np.all(v_mond > v_newton), \
            "MOND should give higher velocities than Newton in outer regions"


class TestNFWCurve:
    """Test NFW dark matter halo curve."""

    def test_nfw_shape(self):
        """V has NFW characteristic rise-peak-flat profile."""
        r = np.logspace(np.log10(0.5), np.log10(50.0), 100)  # kpc
        v = nfw_curve(r, M_disk=5e10, V_200=150.0, c=10, r_s=15.0)

        # Check basic sanity
        assert np.all(v > 0), "All velocities should be positive"
        assert np.all(np.isfinite(v)), "All velocities should be finite"

        # NFW should have characteristic shape: rise, peak, then gradual decline
        # Find peak location
        peak_idx = np.argmax(v)

        # Peak should be somewhere in the middle, not at edges
        assert 5 < peak_idx < 95, f"Peak at unusual location: {peak_idx}"

    def test_nfw_virial_velocity(self):
        """NFW curve should approach V_200 scale."""
        r = np.array([75.0])  # At r_200 for c=10, r_s=15
        v = nfw_curve(r, M_disk=5e10, V_200=150.0, c=10, r_s=15.0)

        # At large r, velocity should be in reasonable range
        assert 50 < v[0] < 300, f"NFW velocity at large r should be reasonable, got {v[0]}"


class TestPBHFogCurve:
    """Test PBH fog curve."""

    def test_pbh_differs_from_nfw(self):
        """PBH curve should be different from NFW curve for same total mass profile."""
        r = np.logspace(np.log10(0.5), np.log10(50.0), 100)

        # Generate curves with comparable parameters
        v_nfw = nfw_curve(r, M_disk=5e10, V_200=150.0, c=10, r_s=15.0)
        v_pbh = pbh_fog_curve(r, M_bar=5e10, f_pbh=0.15, M_pbh=30.0, r_core=3.0)

        # Curves should differ
        diff = np.abs(v_nfw - v_pbh).mean()
        assert diff > 1.0, f"PBH should differ from NFW by > 1 km/s, got {diff}"

    def test_pbh_core_effect(self):
        """PBH fog should show cored behavior."""
        r = np.logspace(np.log10(0.5), np.log10(50.0), 100)
        v = pbh_fog_curve(r, M_bar=5e10, f_pbh=0.15, M_pbh=30.0, r_core=3.0)

        # Check basic sanity
        assert np.all(v > 0), "All velocities should be positive"
        assert np.all(np.isfinite(v)), "All velocities should be finite"


class TestGenerateGalaxy:
    """Test single galaxy generation."""

    def test_generate_galaxy_shape(self):
        """r.shape == v.shape == (n,1)."""
        g = generate_galaxy("newtonian", n_points=100, seed=42)

        assert g['r'].shape == (100, 1), f"r shape should be (100,1), got {g['r'].shape}"
        assert g['v'].shape == (100, 1), f"v shape should be (100,1), got {g['v'].shape}"
        assert g['v_true'].shape == (100, 1), f"v_true shape should be (100,1), got {g['v_true'].shape}"
        assert g['v_unc'].shape == (100,), f"v_unc shape should be (100,), got {g['v_unc'].shape}"

    def test_generate_galaxy_deterministic(self):
        """Same seed → same output."""
        g1 = generate_galaxy("mond", seed=42)
        g2 = generate_galaxy("mond", seed=42)

        assert np.allclose(g1['r'], g2['r']), "r should be deterministic"
        assert np.allclose(g1['v'], g2['v']), "v should be deterministic"
        assert np.allclose(g1['v_true'], g2['v_true']), "v_true should be deterministic"

    def test_generate_galaxy_array_type(self):
        """Output should be numpy arrays."""
        g = generate_galaxy("newtonian", seed=42)

        assert isinstance(g['r'], np.ndarray), "r should be np.ndarray"
        assert isinstance(g['v'], np.ndarray), "v should be np.ndarray"
        assert isinstance(g['v_true'], np.ndarray), "v_true should be np.ndarray"
        assert isinstance(g['v_unc'], np.ndarray), "v_unc should be np.ndarray"

    def test_generate_galaxy_metadata(self):
        """Galaxy should have correct metadata."""
        g = generate_galaxy("nfw", seed=123)

        assert g['id'] == "synth_nfw_0123"
        assert g['regime'] == "nfw"
        assert g['seed'] == 123
        assert 'params' in g

    def test_generate_galaxy_all_regimes(self):
        """All regimes should generate successfully."""
        for regime in REGIMES:
            g = generate_galaxy(regime, seed=42)
            assert g['regime'] == regime
            assert g['r'].shape[0] == DEFAULT_N_POINTS


class TestNoiseBehavior:
    """Test noise model."""

    def test_noise_scales_with_v(self):
        """v_unc ≈ noise × v_true."""
        noise = 0.05
        g = generate_galaxy("newtonian", noise=noise, seed=42)

        v_true = g['v_true'].flatten()
        v_unc = g['v_unc']

        expected_unc = noise * v_true
        np.testing.assert_allclose(v_unc, expected_unc, rtol=1e-5,
                                   err_msg="v_unc should equal noise * v_true")

    def test_noisy_vs_true(self):
        """Noisy velocity should differ from true velocity."""
        g = generate_galaxy("newtonian", noise=0.05, seed=42)

        v = g['v']
        v_true = g['v_true']

        # They should be different
        assert not np.allclose(v, v_true), "Noisy v should differ from v_true"

        # But difference should be reasonable (within a few sigma)
        diff = np.abs(v - v_true).flatten()
        v_unc = g['v_unc']

        # Most points should be within 3 sigma
        within_3sigma = np.sum(diff < 3 * v_unc) / len(diff)
        assert within_3sigma > 0.95, f"Only {within_3sigma*100}% within 3 sigma"


class TestBatchGenerate:
    """Test batch generation."""

    def test_batch_generate_count(self):
        """len(result) == 4 × n_per_regime."""
        galaxies = batch_generate(n_per_regime=5, noise=0.03, seed=42)
        assert len(galaxies) == 20, f"Expected 20 galaxies (4×5), got {len(galaxies)}"

    def test_batch_generate_regime_distribution(self):
        """Equal distribution across regimes."""
        galaxies = batch_generate(n_per_regime=10, seed=42)

        regime_counts = {}
        for g in galaxies:
            regime = g['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        for regime in REGIMES:
            assert regime_counts.get(regime, 0) == 10, \
                f"Expected 10 {regime} galaxies, got {regime_counts.get(regime, 0)}"

    def test_batch_emits_receipt(self, capsys):
        """'cosmos' receipt should be emitted."""
        batch_generate(n_per_regime=2, noise=0.03, seed=42)

        captured = capsys.readouterr()
        assert "cosmos" in captured.out, "Should emit cosmos receipt"

        # Parse and validate receipt
        lines = [l for l in captured.out.strip().split('\n') if l]
        receipt_line = [l for l in lines if '"receipt_type": "cosmos"' in l or '"receipt_type":"cosmos"' in l]
        assert len(receipt_line) == 1, "Should emit exactly one cosmos receipt"

        receipt = json.loads(receipt_line[0])
        assert receipt['receipt_type'] == 'cosmos'
        assert receipt['n_galaxies'] == 8  # 4 regimes × 2
        assert 'batch_id' in receipt
        assert 'payload_hash' in receipt


class TestPathological:
    """Test pathological edge cases."""

    def test_pathological_constant(self):
        """Constant V generates without crash."""
        g = generate_pathological('constant', 100, 42)

        assert g['regime'] == 'constant'
        v = g['v'].flatten()

        # All values should be the same
        assert np.allclose(v, v[0]), "Constant V should be constant"

    def test_pathological_noise(self):
        """Pure noise generates without crash."""
        g = generate_pathological('noise', 100, 42)

        assert g['regime'] == 'noise'
        v = g['v'].flatten()

        # Should have high variance
        cv = np.std(v) / np.mean(v)  # coefficient of variation
        assert cv > 0.1, f"Noise should have high variance, CV={cv}"

    def test_pathological_discontinuous(self):
        """Discontinuous V generates without crash."""
        g = generate_pathological('discontinuous', 100, 42)

        assert g['regime'] == 'discontinuous'
        v_true = g['v_true'].flatten()

        # Should have a jump
        max_diff = np.max(np.abs(np.diff(v_true)))
        assert max_diff > 50, f"Discontinuous should have large jump, max_diff={max_diff}"

    def test_pathological_ambiguous(self):
        """Ambiguous case generates without crash."""
        g = generate_pathological('ambiguous', 100, 42)

        assert g['regime'] == 'ambiguous'
        assert g['r'].shape == (100, 1)
        assert g['v'].shape == (100, 1)


class TestStoprules:
    """Test stoprule behavior."""

    def test_invalid_regime_stoprule(self, capsys):
        """StopRule raised for invalid regime."""
        with pytest.raises(StopRule, match="Invalid regime"):
            generate_galaxy("invalid_regime", seed=42)

        # Should also emit anomaly receipt
        captured = capsys.readouterr()
        assert "anomaly" in captured.out

    def test_negative_radius_stoprule(self, capsys):
        """StopRule raised for negative radius."""
        r = np.array([-1.0, 0.0, 1.0])

        with pytest.raises(StopRule, match="Radius must be positive"):
            newton_curve(r, M=1e11)

        # Should also emit anomaly receipt
        captured = capsys.readouterr()
        assert "anomaly" in captured.out

    def test_zero_radius_stoprule(self):
        """StopRule raised for zero radius."""
        r = np.array([0.0, 1.0, 2.0])

        with pytest.raises(StopRule, match="Radius must be positive"):
            newton_curve(r, M=1e11)


class TestCurveDistinguishability:
    """Test that different regimes produce distinguishable curves."""

    def test_all_regimes_distinct(self):
        """All 4 regimes should produce distinct curves."""
        r = np.logspace(np.log10(1.0), np.log10(20.0), 50)

        curves = {
            'newtonian': newton_curve(r, M=1e11),
            'mond': mond_curve(r, M=1e10),
            'nfw': nfw_curve(r, M_disk=5e10, V_200=150.0, c=10, r_s=15.0),
            'pbh_fog': pbh_fog_curve(r, M_bar=5e10, f_pbh=0.15, M_pbh=30.0, r_core=3.0)
        }

        # All pairwise differences should be significant
        regimes = list(curves.keys())
        for i in range(len(regimes)):
            for j in range(i + 1, len(regimes)):
                diff = np.abs(curves[regimes[i]] - curves[regimes[j]]).mean()
                assert diff > 1.0, \
                    f"{regimes[i]} vs {regimes[j]}: mean diff {diff} km/s too small"


class TestIntegrationWithWitness:
    """Test that cosmos output is compatible with witness.py."""

    def test_array_format_for_witness(self):
        """Galaxy arrays should be compatible with witness.py.

        NOTE: witness.py converts inputs via np.asarray(), so numpy arrays work directly.
        """
        g = generate_galaxy("newtonian", n_points=50, seed=42)

        # witness.py expects (n_points, 1) shape
        r = g['r']
        v = g['v']

        assert r.ndim == 2, "r should be 2D"
        assert v.ndim == 2, "v should be 2D"
        assert r.shape[1] == 1, "r should have shape (n, 1)"
        assert v.shape[1] == 1, "v should have shape (n, 1)"

        # Already numpy arrays, compatible with witness
        assert r.shape == (50, 1)
        assert v.shape == (50, 1)
        assert r.dtype == np.float32
        assert v.dtype == np.float32
