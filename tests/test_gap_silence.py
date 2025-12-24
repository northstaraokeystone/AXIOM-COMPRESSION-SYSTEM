"""Tests for D19.3 Gap-Silence Law Emergence.

Interstellar latency gaps as selection pressure.
Long silence forces minimal-sync laws.
"""

from datetime import datetime, timedelta


class TestGapSilenceEmergence:
    """Tests for GapSilenceEmergence functionality."""

    def test_init_gap_detector(self):
        """Test gap detector initialization."""
        from src.oracle import init_gap_detector

        detector = init_gap_detector()

        assert detector is not None
        assert detector.detector_id is not None
        assert len(detector.detector_id) == 8
        assert detector.normal_threshold_seconds == 60
        assert detector.interstellar_threshold_years == 4.0

    def test_detect_gap_negligible(self):
        """Test negligible gap detection (< 60 seconds)."""
        from src.oracle import init_gap_detector, detect_gap

        detector = init_gap_detector()

        now = datetime.utcnow()
        recent = (now - timedelta(seconds=30)).isoformat() + "Z"

        gap = detect_gap(detector, recent, now.isoformat() + "Z")

        assert gap["gap_type"] == "negligible"
        assert gap["gap_seconds"] < 60

    def test_detect_gap_normal(self):
        """Test normal gap detection (60 seconds - 1 hour)."""
        from src.oracle import init_gap_detector, detect_gap

        detector = init_gap_detector()

        now = datetime.utcnow()
        past = (now - timedelta(minutes=30)).isoformat() + "Z"

        gap = detect_gap(detector, past, now.isoformat() + "Z")

        assert gap["gap_type"] == "normal"
        assert 60 <= gap["gap_seconds"] < 3600

    def test_detect_gap_extended(self):
        """Test extended gap detection (1+ hour)."""
        from src.oracle import init_gap_detector, detect_gap

        detector = init_gap_detector()

        now = datetime.utcnow()
        past = (now - timedelta(hours=2)).isoformat() + "Z"

        gap = detect_gap(detector, past, now.isoformat() + "Z")

        assert gap["gap_type"] == "extended"
        assert gap["gap_seconds"] >= 3600

    def test_classify_gap(self):
        """Test gap classification."""
        from src.oracle import init_gap_detector
        from src.oracle.gap_silence_emergence import classify_gap

        detector = init_gap_detector()

        # Negligible (< 60 seconds)
        assert classify_gap(detector, 30) == "negligible"

        # Normal (60 seconds - 1 hour)
        assert classify_gap(detector, 120) == "normal"

        # Extended (1+ hour)
        assert classify_gap(detector, 7200) == "extended"

        # Interstellar (4+ years)
        four_years_seconds = 4.0 * 365.25 * 24 * 3600
        assert classify_gap(detector, four_years_seconds) == "interstellar"

    def test_minimal_sync_law(self):
        """Test minimal-sync law selection."""
        from src.oracle import minimal_sync_law

        laws = [
            {"law_id": "law1", "invariance_score": 0.9},
            {"law_id": "law2", "invariance_score": 0.6},
            {"law_id": "law3", "invariance_score": 0.3},
        ]

        # Normal gap - less strict
        normal_survivors = minimal_sync_law(laws, 120)

        # Extended gap - more strict
        extended_survivors = minimal_sync_law(laws, 7200)

        # Interstellar gap - most strict
        interstellar_survivors = minimal_sync_law(laws, 4.0 * 365.25 * 24 * 3600)

        # Longer gaps should have fewer survivors
        assert len(extended_survivors) <= len(normal_survivors)
        assert len(interstellar_survivors) <= len(extended_survivors)

    def test_trigger_minimal_law_selection(self):
        """Test minimal law selection trigger."""
        from src.oracle import (
            init_gap_detector,
            init_oracle,
            trigger_minimal_law_selection,
        )

        detector = init_gap_detector()
        oracle = init_oracle({})
        oracle.laws = [
            {"law_id": "law1", "invariance_score": 0.9},
            {"law_id": "law2", "invariance_score": 0.5},
        ]

        survivors = trigger_minimal_law_selection(detector, oracle, "extended")

        assert isinstance(survivors, list)

    def test_emit_gap_emergence_receipt(self):
        """Test gap emergence receipt emission."""
        from src.oracle import init_gap_detector, emit_gap_emergence_receipt

        detector = init_gap_detector()
        gap = {
            "gap_id": "test_gap",
            "gap_type": "extended",
            "gap_seconds": 7200,
            "gap_years": 0.0,
        }
        emerged_laws = [{"law_id": "law1"}]

        receipt = emit_gap_emergence_receipt(detector, gap, emerged_laws)

        assert receipt["receipt_type"] == "gap_silence_emergence"
        assert receipt["gap_type"] == "extended"
        assert receipt["emerged_laws_count"] == 1
        assert receipt["selection_pressure"] == "silence"


class TestGapSilenceConstants:
    """Test D19.3 gap-silence constants."""

    def test_normal_threshold(self):
        """Verify normal gap threshold is 60 seconds."""
        from src.oracle.gap_silence_emergence import LATENCY_SILENCE_THRESHOLD_SECONDS

        assert LATENCY_SILENCE_THRESHOLD_SECONDS == 60

    def test_extended_threshold(self):
        """Verify extended gap threshold is 1 hour."""
        from src.oracle.gap_silence_emergence import EXTENDED_SILENCE_THRESHOLD_SECONDS

        assert EXTENDED_SILENCE_THRESHOLD_SECONDS == 3600

    def test_interstellar_threshold(self):
        """Verify interstellar gap threshold is 4 years."""
        from src.oracle.gap_silence_emergence import (
            INTERSTELLAR_SILENCE_THRESHOLD_YEARS,
        )

        assert INTERSTELLAR_SILENCE_THRESHOLD_YEARS == 4.0

    def test_proxima_silence(self):
        """Verify Proxima silence is 8.48 years."""
        from src.oracle.gap_silence_emergence import PROXIMA_SILENCE_YEARS

        assert PROXIMA_SILENCE_YEARS == 8.48
