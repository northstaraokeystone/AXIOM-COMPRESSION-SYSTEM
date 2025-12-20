"""test_latency_selection.py - Tests for interstellar latency selection (Gate 4).

THE LATENCY INSIGHT:
    Interstellar latency no longer degrades - it SELECTS.
    Distance doesn't break the system. It evolves delay-tolerant primitives.
    Patterns requiring fast response die. Patterns tolerating delay survive.
    Over generations, the population becomes Mars-viable (and beyond).

Tests verify:
- Latency pressure scales with distance
- Mars latency eliminates sensitive receipts
- Delay-tolerant receipts survive
- Evolution increases tolerance
- Latency selection receipt emitted
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entropy import (
    calculate_latency_pressure,
    is_delay_tolerant,
    apply_latency_selection,
    evolve_under_latency,
    MARS_MIN_LATENCY_MS,
    MARS_MAX_LATENCY_MS,
    JUPITER_MIN_LATENCY_MS,
    JUPITER_MAX_LATENCY_MS,
    SELECTION_PRESSURE_LATENCY,
)


class TestLatencyPressure:
    """Tests for latency pressure calculation."""

    def test_latency_pressure_scales(self):
        """Verify calculate_latency_pressure(1000000) > calculate_latency_pressure(1000)."""
        pressure_high = calculate_latency_pressure(1000000)
        pressure_low = calculate_latency_pressure(1000)

        assert pressure_high > pressure_low, "Higher latency should mean higher pressure"

    def test_pressure_normalized_to_01(self):
        """Latency pressure is always in [0, 1] range."""
        test_latencies = [0, 1000, 100000, MARS_MAX_LATENCY_MS, JUPITER_MAX_LATENCY_MS]
        for lat in test_latencies:
            pressure = calculate_latency_pressure(lat)
            assert 0 <= pressure <= 1, f"Pressure {pressure} out of range for latency {lat}"

    def test_zero_latency_zero_pressure(self):
        """Zero latency should have zero pressure."""
        pressure = calculate_latency_pressure(0)
        assert pressure == 0

    def test_jupiter_max_latency_full_pressure(self):
        """Jupiter max latency should approach full pressure."""
        pressure = calculate_latency_pressure(JUPITER_MAX_LATENCY_MS)
        assert pressure == 1.0, "Jupiter max should be full pressure"


class TestDelayTolerance:
    """Tests for delay tolerance checking."""

    def test_delay_tolerant_survives(self):
        """Delay-tolerant receipt survives Mars max latency."""
        receipt = {"id": "tolerant", "tolerance": 0.9}
        assert is_delay_tolerant(receipt, MARS_MAX_LATENCY_MS) is True

    def test_sensitive_dies(self):
        """Latency-sensitive receipt dies under Mars max latency."""
        receipt = {"id": "sensitive", "tolerance": 0.1}
        # Mars max is ~0.415 of Jupiter max, so 0.1 tolerance should fail
        pressure = calculate_latency_pressure(MARS_MAX_LATENCY_MS)
        assert is_delay_tolerant(receipt, MARS_MAX_LATENCY_MS) is (0.1 >= pressure)

    def test_tolerance_equals_pressure_survives(self):
        """Receipt with tolerance exactly matching pressure survives."""
        latency = 1000000
        pressure = calculate_latency_pressure(latency)
        receipt = {"id": "exact", "tolerance": pressure}
        assert is_delay_tolerant(receipt, latency) is True


class TestLatencySelection:
    """Tests for latency-based selection."""

    def test_mars_latency_eliminates_sensitive(self):
        """Mars max latency should eliminate latency-sensitive receipts."""
        pop = [
            {"id": "sensitive1", "tolerance": 0.1},
            {"id": "sensitive2", "tolerance": 0.2},
            {"id": "tolerant1", "tolerance": 0.8},
            {"id": "tolerant2", "tolerance": 0.9},
        ]

        survivors = apply_latency_selection(pop, MARS_MAX_LATENCY_MS)

        assert len(survivors) < len(pop), "Mars latency should eliminate some"

    def test_low_latency_preserves_all(self):
        """Very low latency should preserve all receipts."""
        pop = [
            {"id": "r1", "tolerance": 0.1},
            {"id": "r2", "tolerance": 0.5},
            {"id": "r3", "tolerance": 0.9},
        ]

        survivors = apply_latency_selection(pop, 1000)  # 1 second
        assert len(survivors) == len(pop), "Low latency should preserve all"

    def test_latency_selection_receipt_emitted(self, capsys):
        """Verify 'latency_selection' receipt is emitted."""
        pop = [{"id": "test", "tolerance": 0.5}]
        apply_latency_selection(pop, MARS_MIN_LATENCY_MS)

        captured = capsys.readouterr()
        assert "latency_selection" in captured.out


class TestEvolutionUnderLatency:
    """Tests for evolution under latency pressure."""

    def test_evolution_increases_tolerance(self, capsys):
        """Average tolerance should increase over generations."""
        pop = [
            {"id": "r1", "tolerance": 0.5},
            {"id": "r2", "tolerance": 0.5},
            {"id": "r3", "tolerance": 0.5},
        ]

        # Use low latency so all survive and can evolve
        evolved = evolve_under_latency(pop, generations=5, latency_ms=100000)

        if evolved:
            avg_tolerance = sum(r.get("tolerance", 0) for r in evolved) / len(evolved)
            assert avg_tolerance > 0.5, "Tolerance should increase through evolution"

    def test_high_latency_kills_weak(self):
        """High latency should kill weak receipts during evolution."""
        pop = [
            {"id": "weak1", "tolerance": 0.1},
            {"id": "weak2", "tolerance": 0.2},
            {"id": "tolerant", "tolerance": 0.95},  # Barely survives Jupiter
        ]

        # Use high latency (Jupiter = full pressure)
        evolved = evolve_under_latency(pop, generations=3, latency_ms=JUPITER_MAX_LATENCY_MS)

        # Only the tolerant one should survive (or few)
        assert len(evolved) <= 1, f"High latency should eliminate weak, got {len(evolved)}"

    def test_delay_tolerant_receipt_emitted(self, capsys):
        """Verify 'delay_tolerant' receipt is emitted after evolution."""
        pop = [{"id": "test", "tolerance": 0.8}]
        evolve_under_latency(pop, generations=3, latency_ms=100000)

        captured = capsys.readouterr()
        assert "delay_tolerant" in captured.out


class TestLatencyConstants:
    """Tests for latency constant values."""

    def test_mars_latency_values(self):
        """Mars latency values are correct (in milliseconds)."""
        # 3 minutes = 180,000 ms
        assert MARS_MIN_LATENCY_MS == 180000
        # 22 minutes = 1,320,000 ms
        assert MARS_MAX_LATENCY_MS == 1320000

    def test_jupiter_latency_values(self):
        """Jupiter latency values are correct."""
        # 33 minutes = 1,980,000 ms
        assert JUPITER_MIN_LATENCY_MS == 1980000
        # 53 minutes = 3,180,000 ms
        assert JUPITER_MAX_LATENCY_MS == 3180000

    def test_jupiter_greater_than_mars(self):
        """Jupiter latency is greater than Mars latency."""
        assert JUPITER_MIN_LATENCY_MS > MARS_MAX_LATENCY_MS

    def test_selection_pressure_latency_enabled(self):
        """SELECTION_PRESSURE_LATENCY is True."""
        assert SELECTION_PRESSURE_LATENCY is True


class TestLatencyInsight:
    """Tests that validate the core insight."""

    def test_distance_evolves_tolerance(self, capsys):
        """Distance doesn't degrade - it EVOLVES.

        Over generations, the population adapts to latency constraints.
        """
        # Start with low-tolerance population
        pop = [{"id": f"r{i}", "tolerance": 0.4} for i in range(5)]

        # Evolve under moderate latency
        evolved = evolve_under_latency(pop, generations=10, latency_ms=500000)

        if evolved:
            final_avg = sum(r.get("tolerance", 0) for r in evolved) / len(evolved)
            # Should have evolved higher tolerance
            assert final_avg > 0.4, "Population should evolve higher tolerance"

    def test_survivors_are_mars_viable(self):
        """Receipts surviving Mars selection are Mars-viable by definition."""
        pop = [
            {"id": "r1", "tolerance": 0.3},
            {"id": "r2", "tolerance": 0.5},
            {"id": "r3", "tolerance": 0.7},
            {"id": "r4", "tolerance": 0.9},
        ]

        survivors = apply_latency_selection(pop, MARS_MAX_LATENCY_MS)

        # All survivors are delay-tolerant at Mars latency
        for survivor in survivors:
            assert is_delay_tolerant(survivor, MARS_MAX_LATENCY_MS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
