"""test_benchmarks.py - Tests for Benchmark Comparison Suite

Tests for AXIOM vs pySR comparison and symbolic baselines.

Source: AXIOM Validation Lock v1
"""

import pytest
import numpy as np


class TestSimpleKAN:
    """Tests for SimpleKAN implementation."""

    def test_kan_initialization(self):
        """SimpleKAN should initialize with default config."""
        from benchmarks.pysr_comparison import SimpleKAN

        kan = SimpleKAN(n_basis=10, degree=3)

        assert kan.n_basis == 10
        assert kan.degree == 3
        assert not kan.fitted

    def test_kan_fit_produces_results(self):
        """KAN.fit() should produce training results."""
        from benchmarks.pysr_comparison import SimpleKAN

        kan = SimpleKAN(n_basis=8)

        # Create simple test data
        r = np.linspace(1, 10, 20)
        v = 100 * np.sqrt(1 / r)  # Simple Newtonian curve

        result = kan.fit(r, v, epochs=50)

        assert "epochs" in result
        assert "final_mse" in result
        assert "r_squared" in result
        assert result["r_squared"] > 0.5  # Should fit reasonably well

    def test_kan_predict_after_fit(self):
        """KAN.predict() should work after fitting."""
        from benchmarks.pysr_comparison import SimpleKAN

        kan = SimpleKAN()
        r = np.linspace(1, 10, 20)
        v = np.sqrt(100 / r)

        kan.fit(r, v, epochs=50)
        pred = kan.predict(r)

        assert len(pred) == len(r)
        assert np.all(np.isfinite(pred))

    def test_kan_predict_raises_before_fit(self):
        """KAN.predict() should raise if not fitted."""
        from benchmarks.pysr_comparison import SimpleKAN

        kan = SimpleKAN()
        r = np.linspace(1, 10, 20)

        with pytest.raises(ValueError, match="not fitted"):
            kan.predict(r)

    def test_kan_compression_is_reasonable(self):
        """KAN compression ratio should be reasonable after fitting."""
        from benchmarks.pysr_comparison import SimpleKAN

        kan = SimpleKAN()
        r = np.linspace(1, 10, 20)
        v = np.sqrt(100 / r)

        kan.fit(r, v, epochs=50)

        # Compute compression manually
        data_bits = len(r) * 2 * 64  # 20 points * 2 values * 64 bits
        compression = kan.compute_compression(data_bits)

        # Compression can be positive (save bits) or near 0
        # For 20 data points vs 10 basis functions, expect ~0.5
        assert compression >= 0
        assert compression <= 1


class TestAXIOMRunner:
    """Tests for run_axiom function."""

    def test_run_axiom_returns_results(self):
        """run_axiom() should return benchmark results."""
        from benchmarks.pysr_comparison import run_axiom

        data = {
            "id": "TEST001",
            "r": np.linspace(1, 10, 20),
            "v": 100 * np.sqrt(1 / np.linspace(1, 10, 20)),
        }

        result = run_axiom(data, epochs=20)

        assert "equation" in result
        assert "mse" in result
        assert "compression" in result
        assert "r_squared" in result
        assert "time_ms" in result
        assert "tool" in result
        assert result["tool"] == "AXIOM_KAN"
        assert result["success"]

    def test_run_axiom_r_squared_reasonable(self):
        """run_axiom() should achieve reasonable R² on simple data."""
        from benchmarks.pysr_comparison import run_axiom

        r = np.linspace(1, 10, 20)
        data = {
            "id": "SIMPLE",
            "r": r,
            "v": 100 * np.sqrt(1 / r),
        }

        result = run_axiom(data, epochs=100)

        # Should fit simple Newtonian curve well
        assert result["r_squared"] > 0.9


class TestPySRRunner:
    """Tests for run_pysr function."""

    def test_run_pysr_returns_results(self):
        """run_pysr() should return results (with or without pySR)."""
        from benchmarks.pysr_comparison import run_pysr

        data = {
            "id": "TEST001",
            "r": np.linspace(1, 10, 20),
            "v": 100 * np.sqrt(1 / np.linspace(1, 10, 20)),
        }

        result = run_pysr(data)

        assert "equation" in result
        assert "mse" in result
        assert "complexity" in result
        assert "time_ms" in result
        assert "tool" in result
        assert result["success"]

    def test_run_pysr_fallback_works(self):
        """run_pysr() should work even without pySR installed."""
        from benchmarks.pysr_comparison import run_pysr

        data = {
            "id": "FALLBACK",
            "r": np.linspace(1, 10, 20),
            "v": 100 * np.sqrt(1 / np.linspace(1, 10, 20)),
        }

        result = run_pysr(data)

        # Should return results regardless of pySR availability
        assert result["success"]
        assert result["mse"] < 1000  # Should be reasonable


class TestComparison:
    """Tests for comparison functions."""

    def test_compare_returns_both_results(self):
        """compare() should return results for both methods."""
        from benchmarks.pysr_comparison import compare

        galaxy = {
            "id": "NGC_TEST",
            "r": np.linspace(0.5, 10, 20),
            "v": 100 * np.ones(20),  # Flat rotation curve
        }

        result = compare(galaxy)

        assert "galaxy_id" in result
        assert "pysr" in result
        assert "axiom" in result
        assert "comparison" in result

        assert result["pysr"]["success"]
        assert result["axiom"]["success"]

    def test_compare_identifies_winner(self):
        """compare() should identify MSE winner."""
        from benchmarks.pysr_comparison import compare

        galaxy = {
            "id": "NGC_WINNER",
            "r": np.linspace(0.5, 10, 20),
            "v": 100 * np.sqrt(1 / np.linspace(0.5, 10, 20)),
        }

        result = compare(galaxy)

        assert "winner_mse" in result["comparison"]
        assert result["comparison"]["winner_mse"] in ["axiom", "pysr"]

    def test_batch_compare_aggregates(self):
        """batch_compare() should aggregate multiple galaxies."""
        from benchmarks.pysr_comparison import batch_compare

        galaxies = [
            {
                "id": f"NGC_{i:04d}",
                "r": np.linspace(0.5, 10, 15),
                "v": 100 * np.sqrt(1 / np.linspace(0.5, 10, 15))
                + np.random.randn(15) * 5,
            }
            for i in range(3)
        ]

        result = batch_compare(galaxies)

        assert "individual_results" in result
        assert "summary" in result
        assert len(result["individual_results"]) == 3
        assert result["summary"]["n_galaxies"] == 3

    def test_generate_table_produces_markdown(self):
        """generate_table() should produce valid markdown."""
        from benchmarks.pysr_comparison import generate_table, compare

        galaxy = {
            "id": "NGC_TABLE",
            "r": np.linspace(0.5, 10, 20),
            "v": np.ones(20) * 100,
        }

        comparison = compare(galaxy)
        table = generate_table([comparison])

        assert "| Galaxy |" in table
        assert "|--------|" in table
        assert "NGC_TABLE" in table


class TestSymbolicBaselines:
    """Tests for symbolic baseline comparisons."""

    def test_ai_feynman_returns_results(self):
        """run_ai_feynman() should return physics-informed results."""
        from benchmarks.symbolic_baselines import run_ai_feynman

        data = {
            "id": "NGC_FEYNMAN",
            "r": np.linspace(0.5, 10, 20),
            "v": 100 * np.sqrt(1 / np.linspace(0.5, 10, 20)),
        }

        result = run_ai_feynman(data)

        assert "equation" in result
        assert "mse" in result
        assert "physics_form" in result
        assert "tool" in result
        assert result["tool"] == "AI_Feynman"
        assert result["success"]

    def test_ai_feynman_identifies_physics(self):
        """run_ai_feynman() should identify physics form."""
        from benchmarks.symbolic_baselines import run_ai_feynman

        data = {
            "id": "NGC_PHYS",
            "r": np.linspace(0.5, 10, 20),
            "v": 100 * np.sqrt(1 / np.linspace(0.5, 10, 20)),
        }

        result = run_ai_feynman(data)

        # Should identify as Newtonian or similar
        assert result["physics_form"] in [
            "newtonian",
            "mond_like",
            "isothermal_halo",
            "exponential_disk",
        ]

    def test_eureqa_stub_returns_unavailable(self):
        """run_eureqa_stub() should indicate unavailability."""
        from benchmarks.symbolic_baselines import run_eureqa_stub

        result = run_eureqa_stub({})

        assert not result["success"]
        assert "unavailable" in result["equation"]


class TestReport:
    """Tests for report generation."""

    def test_format_comparison_table_produces_markdown(self):
        """format_comparison_table() should produce markdown."""
        from benchmarks.report import format_comparison_table

        results = {
            "individual_results": [
                {
                    "galaxy_id": "NGC_RPT",
                    "axiom": {"compression": 0.85, "r_squared": 0.95, "mse": 10.0},
                    "pysr": {"mse": 12.0},
                    "comparison": {"winner_mse": "axiom"},
                }
            ],
            "summary": {
                "n_galaxies": 1,
                "axiom": {"mean_compression": 0.85, "mean_r_squared": 0.95},
                "axiom_wins_mse": 1,
                "axiom_wins_time": 1,
            },
        }

        table = format_comparison_table(results)

        assert "# AXIOM Benchmark Results" in table
        assert "NGC_RPT" in table
        assert "85" in table or "0.85" in table  # Compression

    def test_emit_benchmark_summary_returns_receipt(self, capsys):
        """emit_benchmark_summary() should emit receipt."""
        from benchmarks.report import emit_benchmark_summary

        results = {
            "summary": {
                "n_galaxies": 5,
                "axiom": {"mean_compression": 0.90, "mean_r_squared": 0.97},
                "axiom_wins_mse": 3,
            }
        }

        receipt = emit_benchmark_summary(results)

        assert receipt is not None
        assert "validation_lock" in str(receipt)
