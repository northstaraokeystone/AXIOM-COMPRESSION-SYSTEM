"""test_selection_pressure.py - Tests for real entropy as selection pressure.

THE DARWINIAN INSIGHT:
    Real entropy events become SELECTION PRESSURE.
    Synthetic entropy is REJECTED (stoprule).
    Weak receipts die under pressure; strong ones survive.

Tests verify:
- Real entropy is accepted
- Synthetic entropy is rejected with stoprule
- Pressure scales with magnitude
- Weak populations are eliminated under pressure
- Selection pressure receipt is emitted
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entropy import (
    validate_entropy_source,
    calculate_selection_pressure,
    apply_selection_pressure,
    ingest_real_entropy,
    calculate_latency_pressure,
    apply_latency_selection,
    is_delay_tolerant,
    evolve_under_latency,
    SELECTION_PRESSURE_LATENCY,
    REAL_ENTROPY_ONLY,
    DISRUPTION_WEIGHT,
    MARS_MIN_LATENCY_MS,
    MARS_MAX_LATENCY_MS,
    JUPITER_MAX_LATENCY_MS,
)
from src.core import StopRule


class TestRealEntropyAcceptance:
    """Tests for real entropy source validation."""

    def test_real_entropy_accepted(self):
        """Verify validate_entropy_source({'source': 'real'}) == True."""
        event = {"source": "real", "id": "test"}
        assert validate_entropy_source(event) is True

    def test_real_entropy_with_magnitude(self):
        """Real entropy with magnitude is accepted and processed."""
        event = {"source": "real", "magnitude": 0.8, "id": "test"}
        result = validate_entropy_source(event)
        assert result is True

    def test_ingest_real_entropy_works(self, capsys):
        """ingest_real_entropy processes valid events."""
        event = {"source": "real", "magnitude": 0.7, "id": "test123"}
        processed = ingest_real_entropy(event)

        assert processed["_validated"] is True
        assert "_selection_pressure" in processed

        # Check receipt was emitted
        captured = capsys.readouterr()
        assert "entropy_event" in captured.out


class TestSyntheticEntropyRejection:
    """Tests for synthetic entropy rejection."""

    def test_synthetic_entropy_rejected(self):
        """Verify validate_entropy_source({'source': 'synthetic'}) triggers stoprule."""
        event = {"source": "synthetic", "id": "test"}
        with pytest.raises(StopRule) as exc_info:
            validate_entropy_source(event)
        assert "stoprule" in str(exc_info.value).lower() or "Synthetic" in str(
            exc_info.value
        )

    def test_mock_entropy_rejected(self):
        """Mock source is also rejected."""
        event = {"source": "mock", "id": "test"}
        with pytest.raises(StopRule):
            validate_entropy_source(event)

    def test_unknown_entropy_rejected(self):
        """Unknown source is rejected."""
        event = {"source": "unknown", "id": "test"}
        with pytest.raises(StopRule):
            validate_entropy_source(event)

    def test_missing_source_rejected(self):
        """Missing source field is rejected."""
        event = {"id": "test"}
        with pytest.raises(StopRule):
            validate_entropy_source(event)

    def test_stoprule_emits_anomaly_receipt(self, capsys):
        """Synthetic rejection emits anomaly receipt."""
        event = {"source": "synthetic", "id": "test"}
        try:
            validate_entropy_source(event)
        except StopRule:
            pass

        captured = capsys.readouterr()
        assert "anomaly" in captured.out
        assert "synthetic" in captured.out.lower()


class TestPressureCalculation:
    """Tests for selection pressure calculation."""

    def test_pressure_scales_with_magnitude(self):
        """Verify pressure(mag=0.9) > pressure(mag=0.1)."""
        event_high = {"magnitude": 0.9}
        event_low = {"magnitude": 0.1}

        pressure_high = calculate_selection_pressure(event_high)
        pressure_low = calculate_selection_pressure(event_low)

        assert pressure_high > pressure_low, (
            f"High {pressure_high} should > Low {pressure_low}"
        )

    def test_pressure_normalized_to_01(self):
        """Pressure is always in [0, 1] range."""
        test_magnitudes = [0, 0.5, 1, 5, 10, 100]
        for mag in test_magnitudes:
            event = {"magnitude": mag}
            pressure = calculate_selection_pressure(event)
            assert 0 <= pressure <= 1, (
                f"Pressure {pressure} out of range for magnitude {mag}"
            )

    def test_disruption_increases_pressure(self):
        """Disruption events have higher pressure."""
        event_normal = {"magnitude": 0.5}
        event_disruption = {"magnitude": 0.5, "disruption": True}

        pressure_normal = calculate_selection_pressure(event_normal)
        pressure_disruption = calculate_selection_pressure(event_disruption)

        assert pressure_disruption > pressure_normal

    def test_default_pressure_when_missing(self):
        """Default to medium pressure when magnitude missing."""
        event = {"id": "test"}
        pressure = calculate_selection_pressure(event)
        assert 0.4 <= pressure <= 0.6, "Default pressure should be around 0.5"


class TestPressureElimination:
    """Tests for selection pressure elimination."""

    def test_pressure_eliminates_weak(self):
        """Verify len(apply_pressure(0.9, weak_pop)) < len(weak_pop)."""
        weak_pop = [
            {"id": "weak1", "compression": 0.3},
            {"id": "weak2", "compression": 0.4},
            {"id": "weak3", "compression": 0.5},
        ]

        survivors = apply_selection_pressure(0.9, weak_pop)

        assert len(survivors) < len(weak_pop), (
            "High pressure should eliminate weak receipts"
        )

    def test_strong_survive_pressure(self):
        """Strong receipts survive even high pressure."""
        strong_pop = [
            {"id": "strong1", "compression": 0.95},
            {"id": "strong2", "compression": 0.98},
        ]

        survivors = apply_selection_pressure(0.9, strong_pop)

        assert len(survivors) == len(strong_pop), "Strong receipts should survive"

    def test_zero_pressure_no_elimination(self):
        """Zero pressure should not eliminate anyone."""
        pop = [
            {"id": "r1", "compression": 0.5},
            {"id": "r2", "compression": 0.6},
        ]

        survivors = apply_selection_pressure(0.0, pop)
        assert len(survivors) == len(pop)

    def test_selection_pressure_receipt_emitted(self, capsys):
        """Verify 'selection_pressure' in receipt_types after apply_pressure."""
        pop = [{"id": "test", "compression": 0.7}]
        apply_selection_pressure(0.5, pop)

        captured = capsys.readouterr()
        assert "selection_pressure" in captured.out


class TestNoSyntheticCodeRemains:
    """Tests to ensure synthetic code is properly handled."""

    def test_real_entropy_only_constant(self):
        """REAL_ENTROPY_ONLY constant is True."""
        assert REAL_ENTROPY_ONLY is True

    def test_selection_pressure_latency_constant(self):
        """SELECTION_PRESSURE_LATENCY constant is True."""
        assert SELECTION_PRESSURE_LATENCY is True

    def test_disruption_weight_positive(self):
        """DISRUPTION_WEIGHT is positive."""
        assert DISRUPTION_WEIGHT > 1.0


class TestLatencySelection:
    """Tests for latency-based selection (Gate 4 preview)."""

    def test_latency_pressure_scales(self):
        """Verify calculate_latency_pressure(1000000) > calculate_latency_pressure(1000)."""
        pressure_high = calculate_latency_pressure(1000000)
        pressure_low = calculate_latency_pressure(1000)

        assert pressure_high > pressure_low

    def test_mars_latency_eliminates_sensitive(self):
        """Mars max latency should eliminate latency-sensitive receipts."""
        # Mix of tolerant and sensitive receipts
        pop = [
            {"id": "sensitive1", "tolerance": 0.1},
            {"id": "sensitive2", "tolerance": 0.2},
            {"id": "tolerant1", "tolerance": 0.8},
            {"id": "tolerant2", "tolerance": 0.9},
        ]

        survivors = apply_latency_selection(pop, MARS_MAX_LATENCY_MS)

        assert len(survivors) < len(pop), "Mars latency should eliminate some receipts"

    def test_delay_tolerant_survives(self):
        """Delay-tolerant receipts survive Mars max latency."""
        receipt = {"id": "tolerant", "tolerance": 0.9}
        assert is_delay_tolerant(receipt, MARS_MAX_LATENCY_MS) is True

    def test_latency_selection_receipt_emitted(self, capsys):
        """Verify 'latency_selection' in receipt_types."""
        pop = [{"id": "test", "tolerance": 0.5}]
        apply_latency_selection(pop, MARS_MIN_LATENCY_MS)

        captured = capsys.readouterr()
        assert "latency_selection" in captured.out


class TestEvolutionUnderLatency:
    """Tests for evolution under latency pressure."""

    def test_evolution_increases_tolerance(self, capsys):
        """Average tolerance should increase over generations."""
        # Start with medium tolerance
        pop = [
            {"id": "r1", "tolerance": 0.5},
            {"id": "r2", "tolerance": 0.5},
            {"id": "r3", "tolerance": 0.5},
        ]

        # Use low latency so all survive and can evolve
        evolved = evolve_under_latency(pop, generations=5, latency_ms=100000)

        if evolved:
            avg_tolerance_evolved = sum(r.get("tolerance", 0) for r in evolved) / len(
                evolved
            )
            assert avg_tolerance_evolved > 0.5, (
                "Tolerance should increase through evolution"
            )

    def test_delay_tolerant_receipt_emitted(self, capsys):
        """Verify 'delay_tolerant' receipt is emitted after evolution."""
        pop = [{"id": "test", "tolerance": 0.8}]
        evolve_under_latency(pop, generations=3, latency_ms=100000)

        captured = capsys.readouterr()
        assert "delay_tolerant" in captured.out


class TestLatencyConstants:
    """Tests for latency constants."""

    def test_mars_latency_range(self):
        """Mars latency min < max."""
        assert MARS_MIN_LATENCY_MS < MARS_MAX_LATENCY_MS

    def test_mars_latency_values(self):
        """Mars latency values are correct."""
        assert MARS_MIN_LATENCY_MS == 180000  # 3 minutes
        assert MARS_MAX_LATENCY_MS == 1320000  # 22 minutes

    def test_jupiter_latency_greater_than_mars(self):
        """Jupiter latency > Mars latency."""
        assert JUPITER_MAX_LATENCY_MS > MARS_MAX_LATENCY_MS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
