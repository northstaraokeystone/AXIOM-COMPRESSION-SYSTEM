"""Tests for autocatalytic pattern module."""

import pytest


class TestPatternDetector:
    """Test pattern detector functionality."""

    def test_init_detector(self):
        """Test detector initialization."""
        from src.autocatalytic.pattern_detector import init_detector

        detector = init_detector({})

        assert detector is not None

    def test_scan_receipt_stream(self):
        """Test receipt stream scanning."""
        from src.autocatalytic.pattern_detector import init_detector, scan_receipt_stream

        detector = init_detector({})
        receipts = [
            {"receipt_type": "type_a"},
            {"receipt_type": "type_b"},
            {"receipt_type": "type_a"},
        ]

        detected = scan_receipt_stream(detector, receipts)

        assert len(detected) > 0


class TestPatternLifecycle:
    """Test pattern lifecycle functionality."""

    def test_init_lifecycle(self):
        """Test lifecycle initialization."""
        from src.autocatalytic.pattern_detector import init_detector
        from src.autocatalytic.pattern_lifecycle import init_lifecycle

        detector = init_detector({})
        lifecycle = init_lifecycle(detector)

        assert lifecycle is not None

    def test_birth_pattern(self):
        """Test pattern birth."""
        from src.autocatalytic.pattern_detector import init_detector
        from src.autocatalytic.pattern_lifecycle import init_lifecycle, birth_pattern

        detector = init_detector({})
        lifecycle = init_lifecycle(detector)

        pattern = {"pattern_id": "test_pattern", "fitness": 0.8}
        result = birth_pattern(lifecycle, pattern)

        assert result is not None
        assert result["pattern_id"] == "test_pattern"
        assert result["status"] == "alive"


class TestCrossPlanetMigration:
    """Test cross-planet migration functionality."""

    def test_init_migration(self):
        """Test migration manager initialization."""
        from src.autocatalytic.cross_planet_migration import init_migration

        manager = init_migration({})

        assert manager is not None

    def test_migration_latency_tolerance(self):
        """Test migration latency tolerance."""
        from src.autocatalytic.cross_planet_migration import MIGRATION_LATENCY_TOLERANCE_MS

        assert MIGRATION_LATENCY_TOLERANCE_MS == 5000


class TestFitnessEvaluator:
    """Test fitness evaluator functionality."""

    def test_compute_pattern_fitness(self):
        """Test pattern fitness computation."""
        from src.autocatalytic.fitness_evaluator import compute_pattern_fitness

        fitness = compute_pattern_fitness(
            entropy_reduction=0.8,
            coordination_success=0.9,
            stability=0.7,
            diversity_contribution=0.5,
            recency_bonus=1.0,
        )

        assert 0 <= fitness <= 1

    def test_thompson_sampling_select(self):
        """Test Thompson sampling selection."""
        from src.autocatalytic.fitness_evaluator import thompson_sampling_select

        patterns = [
            {"pattern_id": "a", "fitness": 0.9},
            {"pattern_id": "b", "fitness": 0.5},
            {"pattern_id": "c", "fitness": 0.3},
        ]

        selected = thompson_sampling_select(patterns, k=2)

        assert len(selected) == 2
