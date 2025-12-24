"""test_law_imposition.py - Tests for causal law imposition.

THE LAW IMPOSITION INSIGHT:
    Surviving patterns after Darwinian selection become IMPOSED laws.
    Laws are not discovered - they are the survivors of evolution.
    Non-conforming receipts are REJECTED (not flagged - rejected).

Tests verify:
- Pattern extraction from survivors
- Law candidate threshold (10 generations)
- Law imposition and crystallization
- Receipt validation against active laws
- Non-conforming receipt rejection
- Law receipt emission
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.darwinian_enforce import (
    extract_surviving_pattern,
    is_law_candidate,
    impose_law,
    validate_against_laws,
    enforce_laws,
    get_active_laws,
    reset_darwinian_state,
    run_selection_cycle,
    DARWINIAN_GENERATIONS_PER_CYCLE,
)


class TestPatternExtraction:
    """Tests for surviving pattern extraction."""

    def test_pattern_extraction(self):
        """Verify extract_surviving_pattern([...]) returns dict with common keys."""
        reset_darwinian_state()

        survivors = [
            {"id": "s1", "pattern_type": "A", "value": 10, "_amplified": True},
            {"id": "s2", "pattern_type": "A", "value": 10, "_amplified": True},
            {"id": "s3", "pattern_type": "A", "value": 10, "_amplified": True},
        ]

        pattern = extract_surviving_pattern(survivors)

        assert isinstance(pattern, dict), "Pattern should be a dict"
        # Common values should be extracted
        assert "pattern_type" in pattern or "value" in pattern, (
            "Should extract common keys"
        )

    def test_pattern_extraction_empty(self):
        """Empty survivors list returns empty pattern."""
        reset_darwinian_state()
        pattern = extract_surviving_pattern([])
        assert pattern == {}

    def test_pattern_extraction_no_common(self):
        """Survivors with no common values return empty or partial pattern."""
        reset_darwinian_state()

        survivors = [
            {"id": "s1", "value": 1, "_amplified": True},
            {"id": "s2", "value": 2, "_amplified": True},
            {"id": "s3", "value": 3, "_amplified": True},
        ]

        pattern = extract_surviving_pattern(survivors)
        # Value differs, so may not be in pattern
        assert isinstance(pattern, dict)


class TestLawCandidacy:
    """Tests for law candidate threshold."""

    def test_law_requires_10_generations(self):
        """Verify is_law_candidate(pattern, 9) == False."""
        pattern = {"key": "value"}

        assert is_law_candidate(pattern, 9) is False
        assert is_law_candidate(pattern, 8) is False
        assert is_law_candidate(pattern, 1) is False

    def test_law_imposed_after_threshold(self):
        """Verify is_law_candidate(pattern, 10) == True."""
        pattern = {"key": "value"}

        assert is_law_candidate(pattern, 10) is True
        assert is_law_candidate(pattern, 11) is True
        assert is_law_candidate(pattern, 100) is True

    def test_generations_per_cycle_constant(self):
        """Verify DARWINIAN_GENERATIONS_PER_CYCLE == 10."""
        assert DARWINIAN_GENERATIONS_PER_CYCLE == 10


class TestLawImposition:
    """Tests for law imposition and crystallization."""

    def test_impose_law_returns_id(self, capsys):
        """impose_law returns law_id starting with 'law_'."""
        reset_darwinian_state()

        pattern = {"compression_type": "high", "threshold": 0.9}
        law_id = impose_law(pattern)

        assert law_id.startswith("law_"), "Law ID should start with 'law_'"

    def test_imposed_law_added_to_active(self):
        """Imposed law appears in get_active_laws()."""
        reset_darwinian_state()

        pattern = {"test_key": "test_value"}
        law_id = impose_law(pattern)

        active_laws = get_active_laws()
        law_ids = [law["id"] for law in active_laws]

        assert law_id in law_ids, "Imposed law should be in active laws"

    def test_darwinian_law_receipt_emitted(self, capsys):
        """Verify 'darwinian_law' in receipt_types after impose."""
        reset_darwinian_state()

        pattern = {"key": "value"}
        impose_law(pattern)

        captured = capsys.readouterr()
        assert "darwinian_law" in captured.out

    def test_law_imposition_receipt_emitted(self, capsys):
        """Verify 'law_imposition' in receipt_types after impose."""
        reset_darwinian_state()

        pattern = {"key": "value"}
        impose_law(pattern)

        captured = capsys.readouterr()
        assert "law_imposition" in captured.out


class TestLawValidation:
    """Tests for receipt validation against laws."""

    def test_receipt_validated_against_law(self):
        """Verify validate_against_laws(good, [law]) == True."""
        reset_darwinian_state()

        # Impose a law
        pattern = {"required_field": "expected_value"}
        impose_law(pattern)

        # Receipt that conforms
        good_receipt = {"id": "good", "required_field": "expected_value"}
        assert validate_against_laws(good_receipt) is True

    def test_receipt_violates_law(self):
        """Receipt with wrong value fails validation."""
        reset_darwinian_state()

        pattern = {"required_field": "expected_value"}
        impose_law(pattern)

        bad_receipt = {"id": "bad", "required_field": "wrong_value"}
        assert validate_against_laws(bad_receipt) is False

    def test_receipt_without_law_field_passes(self):
        """Receipt without law field passes (field not present)."""
        reset_darwinian_state()

        pattern = {"required_field": "expected_value"}
        impose_law(pattern)

        neutral_receipt = {"id": "neutral", "other_field": "value"}
        # No required_field, so it doesn't violate
        assert validate_against_laws(neutral_receipt) is True

    def test_no_laws_all_pass(self):
        """With no active laws, all receipts pass."""
        reset_darwinian_state()

        receipt = {"id": "any", "field": "value"}
        assert validate_against_laws(receipt) is True


class TestLawEnforcement:
    """Tests for law enforcement (reject non-conforming)."""

    def test_non_conforming_rejected(self, capsys):
        """Verify enforce_laws(bad_receipt) is None."""
        reset_darwinian_state()

        pattern = {"must_be": "correct"}
        impose_law(pattern)

        bad_receipt = {"id": "violator", "must_be": "wrong"}
        result = enforce_laws(bad_receipt)

        assert result is None, "Non-conforming receipt should be rejected (None)"

    def test_conforming_passes(self):
        """Conforming receipts pass enforcement."""
        reset_darwinian_state()

        pattern = {"must_be": "correct"}
        impose_law(pattern)

        good_receipt = {"id": "good", "must_be": "correct"}
        result = enforce_laws(good_receipt)

        assert result is not None
        assert result["id"] == "good"

    def test_law_violation_receipt_emitted(self, capsys):
        """Verify 'law_violation' receipt is emitted on rejection."""
        reset_darwinian_state()

        pattern = {"key": "value"}
        impose_law(pattern)

        bad_receipt = {"id": "violator", "key": "wrong"}
        enforce_laws(bad_receipt)

        captured = capsys.readouterr()
        assert "law_violation" in captured.out


class TestMultipleLaws:
    """Tests for multiple active laws."""

    def test_multiple_laws_all_checked(self):
        """Receipt must conform to ALL active laws."""
        reset_darwinian_state()

        impose_law({"field_a": "value_a"})
        impose_law({"field_b": "value_b"})

        # Passes both
        good = {"field_a": "value_a", "field_b": "value_b"}
        assert validate_against_laws(good) is True

        # Fails one
        partial = {"field_a": "value_a", "field_b": "wrong"}
        assert validate_against_laws(partial) is False

    def test_active_laws_count(self):
        """get_active_laws returns correct count."""
        reset_darwinian_state()

        assert len(get_active_laws()) == 0

        impose_law({"a": 1})
        assert len(get_active_laws()) == 1

        impose_law({"b": 2})
        assert len(get_active_laws()) == 2


class TestIntegrationWithSelection:
    """Integration tests with selection cycle."""

    def test_high_compression_patterns_become_law_candidates(self):
        """High-compression survivors are potential law candidates."""
        reset_darwinian_state()

        # Population with clear pattern in high-compression receipts
        population = [
            {"id": "h1", "compression": 0.95, "pattern": "winner"},
            {"id": "h2", "compression": 0.92, "pattern": "winner"},
            {"id": "l1", "compression": 0.5, "pattern": "loser"},
        ]

        # Run selection
        survivors = run_selection_cycle(population)

        # Extract pattern from high-compression survivors
        pattern = extract_surviving_pattern(survivors)

        # If pattern survived 10 generations, it would become law
        if is_law_candidate(pattern, 10):
            law_id = impose_law(pattern)
            assert law_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
