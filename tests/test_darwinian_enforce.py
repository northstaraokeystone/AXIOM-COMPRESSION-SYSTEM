"""test_darwinian_enforce.py - Tests for Darwinian selection and enforcement.

Tests the core Darwinian selection mechanism:
- Spec loading with dual-hash validation
- Compression scoring and fitness classification
- Path amplification (high-compression replication)
- Path starvation (low-compression death)
- Selection cycles
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.darwinian_enforce import (
    load_spec,
    score_compression,
    classify_fitness,
    amplify_path,
    starve_path,
    get_generation,
    run_selection_cycle,
    is_law_candidate,
    impose_law,
    reset_darwinian_state,
    get_darwinian_info,
    AMPLIFY_FACTOR_HIGH_COMPRESSION,
    STARVE_THRESHOLD_LOW_COMPRESSION,
    HIGH_COMPRESSION_THRESHOLD,
    DARWINIAN_GENERATIONS_PER_CYCLE,
    MAX_SURVIVAL_CYCLES,
)
from src.core import StopRule


class TestSpecLoading:
    """Tests for darwinian_spec.json loading."""

    def test_spec_loads_with_dual_hash(self, capsys):
        """Verify spec loads and darwinian_spec_load_receipt contains payload_hash with ':'."""
        reset_darwinian_state()
        spec = load_spec()

        # Check spec has _hash field with dual-hash format (contains ':')
        assert "_hash" in spec, "Spec should contain _hash field"
        assert ":" in spec["_hash"], (
            "Spec hash should be dual-hash format (contains ':')"
        )

        # Check receipt was emitted
        captured = capsys.readouterr()
        assert "darwinian_spec_load" in captured.out, (
            "Should emit darwinian_spec_load receipt"
        )
        assert "payload_hash" in captured.out, "Receipt should contain payload_hash"

    def test_spec_contains_required_fields(self):
        """Verify spec contains all required configuration fields."""
        spec = load_spec()

        required_fields = [
            "version",
            "amplify_factor",
            "starve_threshold",
            "high_compression_threshold",
            "latency_primary",
            "generations_per_cycle",
            "max_survival_cycles",
            "real_entropy_only",
        ]

        for field in required_fields:
            assert field in spec, f"Spec should contain {field}"

    def test_spec_values_match_constants(self):
        """Verify spec values match module constants."""
        spec = load_spec()

        assert spec["amplify_factor"] == AMPLIFY_FACTOR_HIGH_COMPRESSION
        assert spec["starve_threshold"] == STARVE_THRESHOLD_LOW_COMPRESSION
        assert spec["high_compression_threshold"] == HIGH_COMPRESSION_THRESHOLD
        assert spec["generations_per_cycle"] == DARWINIAN_GENERATIONS_PER_CYCLE
        assert spec["max_survival_cycles"] == MAX_SURVIVAL_CYCLES


class TestCompressionScoring:
    """Tests for compression score extraction."""

    def test_score_from_compression_field(self):
        """Extract score from 'compression' field."""
        receipt = {"compression": 0.95}
        assert score_compression(receipt) == 0.95

    def test_score_from_r_squared_field(self):
        """Extract score from 'r_squared' field."""
        receipt = {"r_squared": 0.88}
        assert score_compression(receipt) == 0.88

    def test_default_score_when_missing(self):
        """Default to neutral score when no compression field."""
        receipt = {"id": "test"}
        score = score_compression(receipt)
        assert 0.8 <= score <= 0.9, "Default score should be neutral"

    def test_invalid_score_triggers_stoprule(self):
        """Scores outside [0,1] should trigger stoprule."""
        receipt = {"compression": 1.5}
        with pytest.raises(StopRule) as exc_info:
            score_compression(receipt)
        assert "Invalid compression" in str(exc_info.value)


class TestFitnessClassification:
    """Tests for fitness classification."""

    def test_high_compression_classified_amplify(self):
        """Verify classify_fitness(0.95) == 'amplify'."""
        assert classify_fitness(0.95) == "amplify"
        assert classify_fitness(0.9) == "amplify"
        assert classify_fitness(0.99) == "amplify"

    def test_low_compression_classified_starve(self):
        """Verify classify_fitness(0.75) == 'starve'."""
        assert classify_fitness(0.75) == "starve"
        assert classify_fitness(0.5) == "starve"
        assert classify_fitness(0.0) == "starve"

    def test_neutral_compression_classified(self):
        """Verify classify_fitness(0.85) == 'neutral'."""
        assert classify_fitness(0.85) == "neutral"
        assert classify_fitness(0.8) == "neutral"
        assert classify_fitness(0.89) == "neutral"

    def test_boundary_values(self):
        """Test exact boundary values."""
        # At 0.9 -> amplify
        assert classify_fitness(0.9) == "amplify"
        # Just below 0.9 -> neutral
        assert classify_fitness(0.899) == "neutral"
        # At 0.8 -> neutral
        assert classify_fitness(0.8) == "neutral"
        # Just below 0.8 -> starve
        assert classify_fitness(0.799) == "starve"


class TestPathAmplification:
    """Tests for path amplification (high-compression replication)."""

    def test_amplify_path_doubles(self):
        """Verify len(amplify_path(receipt, 2.0)) == 2."""
        receipt = {"id": "test", "compression": 0.95}
        amplified = amplify_path(receipt, 2.0)
        assert len(amplified) == 2, (
            "Amplification with factor 2.0 should produce 2 copies"
        )

    def test_amplification_preserves_data(self):
        """Amplified copies preserve original data."""
        receipt = {"id": "test", "compression": 0.95, "data": "important"}
        amplified = amplify_path(receipt, 2.0)

        for copy in amplified:
            assert copy["id"] == "test"
            assert copy["compression"] == 0.95
            assert copy["data"] == "important"

    def test_amplification_marks_copies(self):
        """Amplified copies are marked as amplified."""
        receipt = {"id": "test", "compression": 0.95}
        amplified = amplify_path(receipt, 2.0)

        for copy in amplified:
            assert copy["_amplified"] is True
            assert copy["_amplification_factor"] == 2.0

    def test_amplification_receipt_emitted(self, capsys):
        """Verify 'path_amplification' in receipt_types after amplify."""
        reset_darwinian_state()
        receipt = {"id": "test", "compression": 0.95}
        amplify_path(receipt, 2.0)

        captured = capsys.readouterr()
        assert "path_amplification" in captured.out


class TestPathStarvation:
    """Tests for path starvation (low-compression death)."""

    def test_starve_path_returns_receipt_when_alive(self):
        """Starved path returns receipt if generation <= MAX_SURVIVAL_CYCLES."""
        receipt = {"id": "test", "compression": 0.5}
        result = starve_path(receipt, 3)
        assert result is not None
        assert result["_starved"] is True

    def test_starve_path_kills_old(self):
        """Verify starve_path(receipt, 6) is None (generation > 5)."""
        receipt = {"id": "test", "compression": 0.5}
        result = starve_path(receipt, 6)
        assert result is None, "Receipt should die after MAX_SURVIVAL_CYCLES (5)"

    def test_starvation_tracks_remaining(self):
        """Starved receipt tracks remaining survival cycles."""
        receipt = {"id": "test", "compression": 0.5}
        result = starve_path(receipt, 2)
        assert result["_survival_remaining"] == 3  # 5 - 2 = 3

    def test_starvation_receipt_emitted(self, capsys):
        """Verify 'path_starvation' in receipt_types after starve."""
        reset_darwinian_state()
        receipt = {"id": "test", "compression": 0.5}
        starve_path(receipt, 3)

        captured = capsys.readouterr()
        assert "path_starvation" in captured.out


class TestSelectionCycle:
    """Tests for full selection cycle."""

    def test_selection_cycle_culls_weak(self, capsys):
        """Survivors should have avg score > input avg after selection."""
        reset_darwinian_state()

        # Create population with mix of fitness levels
        population = [
            {"id": "high1", "compression": 0.95},
            {"id": "high2", "compression": 0.92},
            {"id": "neutral", "compression": 0.85},
            {"id": "low1", "compression": 0.5, "_generation": 4},
            {"id": "low2", "compression": 0.3, "_generation": 6},  # Should die
        ]

        # Calculate input average
        input_avg = sum(p["compression"] for p in population) / len(population)

        # Run selection
        survivors = run_selection_cycle(population)

        # Calculate survivor average (only for original data, not amplified)
        original_scores = []
        for s in survivors:
            if "compression" in s:
                original_scores.append(s["compression"])

        survivor_avg = (
            sum(original_scores) / len(original_scores) if original_scores else 0
        )

        # Survivors should have higher average (weak culled, strong amplified)
        assert survivor_avg >= input_avg, "Survivor avg should be >= input avg"

        # Check receipt was emitted
        captured = capsys.readouterr()
        assert "selection_cycle" in captured.out

    def test_selection_cycle_amplifies_winners(self):
        """High-compression receipts should be amplified in selection cycle."""
        reset_darwinian_state()

        population = [
            {"id": "winner", "compression": 0.95},
        ]

        survivors = run_selection_cycle(population)

        # Winner should be amplified (2 copies)
        assert len(survivors) >= 2, "High-compression should be amplified"

    def test_empty_population(self):
        """Empty population returns empty list."""
        result = run_selection_cycle([])
        assert result == []


class TestGenerationTracking:
    """Tests for generation tracking."""

    def test_generation_increments(self):
        """Generation should increment through selection cycles."""
        reset_darwinian_state()

        receipt = {"id": "tracked", "compression": 0.95}
        amplified = amplify_path(receipt, 1.0)

        assert amplified[0]["_generation"] >= 1

    def test_get_generation_returns_tracked(self):
        """get_generation returns tracked generation."""
        receipt = {"_generation": 5}
        assert get_generation(receipt) == 5


class TestDarwinianInfo:
    """Tests for status info function."""

    def test_get_darwinian_info(self):
        """get_darwinian_info returns correct configuration."""
        info = get_darwinian_info()

        assert info["amplify_factor"] == 2.0
        assert info["starve_threshold"] == 0.8
        assert info["high_compression_threshold"] == 0.9
        assert info["generations_per_cycle"] == 10
        assert info["max_survival_cycles"] == 5
        assert "causally enforced" in info["paradigm"]


# Additional tests for Gate 3 (law imposition) - will be in test_law_imposition.py
class TestLawImpositionPreview:
    """Preview tests for law imposition (full tests in test_law_imposition.py)."""

    def test_law_requires_10_generations(self):
        """Verify is_law_candidate(pattern, 9) == False."""
        pattern = {"key": "value"}
        assert is_law_candidate(pattern, 9) is False
        assert is_law_candidate(pattern, 10) is True
        assert is_law_candidate(pattern, 11) is True

    def test_law_imposed_after_threshold(self, capsys):
        """Patterns surviving 10+ generations become laws."""
        reset_darwinian_state()

        pattern = {"key": "value"}
        if is_law_candidate(pattern, 10):
            law_id = impose_law(pattern)
            assert law_id.startswith("law_")

            captured = capsys.readouterr()
            assert "darwinian_law" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
