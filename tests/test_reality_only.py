"""Tests for D19.1 reality-only mode.

Tests verifying synthetic scenarios are KILLED.
Reality is the only valid scenario.
"""


class TestSyntheticKilled:
    """Test synthetic scenarios are killed."""

    def test_synthetic_scenarios_disabled_in_d19(self):
        """Test SYNTHETIC_SCENARIOS_ENABLED is False in D19."""
        from src.depths.d19_swarm_intelligence import SYNTHETIC_SCENARIOS_ENABLED

        assert SYNTHETIC_SCENARIOS_ENABLED is False

    def test_synthetic_disabled_in_live_ingest(self):
        """Test synthetic is disabled in live ingest module."""
        from src.swarm.live_triad_ingest import SYNTHETIC_SCENARIOS_ENABLED

        assert SYNTHETIC_SCENARIOS_ENABLED is False

    def test_entropy_source_is_live_triad(self):
        """Test entropy source is live_triad not synthetic."""
        from src.depths.d19_swarm_intelligence import ENTROPY_SOURCE

        assert ENTROPY_SOURCE == "live_triad"

    def test_baseline_scenario_killed(self):
        """Test BASELINE scenario is killed."""
        from src.depths.d19_swarm_intelligence import BASELINE_SCENARIO

        assert BASELINE_SCENARIO is None

    def test_stress_scenario_killed(self):
        """Test STRESS scenario is killed."""
        from src.depths.d19_swarm_intelligence import STRESS_SCENARIO

        assert STRESS_SCENARIO is None

    def test_genesis_scenario_killed(self):
        """Test GENESIS scenario is killed."""
        from src.depths.d19_swarm_intelligence import GENESIS_SCENARIO

        assert GENESIS_SCENARIO is None

    def test_singularity_scenario_killed(self):
        """Test SINGULARITY scenario is killed."""
        from src.depths.d19_swarm_intelligence import SINGULARITY_SCENARIO

        assert SINGULARITY_SCENARIO is None

    def test_thermodynamic_scenario_killed(self):
        """Test THERMODYNAMIC scenario is killed."""
        from src.depths.d19_swarm_intelligence import THERMODYNAMIC_SCENARIO

        assert THERMODYNAMIC_SCENARIO is None

    def test_godel_scenario_killed(self):
        """Test GODEL scenario is killed."""
        from src.depths.d19_swarm_intelligence import GODEL_SCENARIO

        assert GODEL_SCENARIO is None


class TestLiveOnlyMode:
    """Test live-only mode execution."""

    def test_run_d19_live_only_returns_result(self):
        """Test run_d19_live_only returns valid result."""
        from src.depths.d19_swarm_intelligence import run_d19_live_only

        result = run_d19_live_only()

        assert result is not None
        assert result["mode"] == "live_only"
        assert result["synthetic_enabled"] is False

    def test_live_only_mode_has_four_gates(self):
        """Test live-only mode has 4 gates."""
        from src.depths.d19_swarm_intelligence import run_d19_live_only

        result = run_d19_live_only()
        gates = result.get("gates", {})

        assert "gate_1" in gates
        assert "gate_2" in gates
        assert "gate_3" in gates
        assert "gate_4" in gates

    def test_live_only_entropy_source(self):
        """Test live-only uses live_triad entropy source."""
        from src.depths.d19_swarm_intelligence import run_d19_live_only

        result = run_d19_live_only()

        assert result["entropy_source"] == "live_triad"

    def test_gate_4_validates_reality_only(self):
        """Test Gate 4 validates reality-only mode."""
        from src.depths.d19_swarm_intelligence import run_d19_live_only

        result = run_d19_live_only()
        gate_4 = result["gates"]["gate_4"]

        assert gate_4["name"] == "reality_only_validation"
        assert gate_4["reality_only"] is True
        assert gate_4["synthetic_enabled"] is False


class TestReceiptEnforcedLaw:
    """Test receipt-enforced law functionality."""

    def test_enforcement_mode_is_receipt_chain(self):
        """Test enforcement mode is receipt_chain."""
        from src.witness.receipt_enforced_law import LAW_ENFORCEMENT_MODE

        assert LAW_ENFORCEMENT_MODE == "receipt_chain"

    def test_chain_causality_priority(self):
        """Test chain causality has priority."""
        from src.witness.receipt_enforced_law import CHAIN_CAUSALITY_PRIORITY

        assert CHAIN_CAUSALITY_PRIORITY is True

    def test_init_enforcement(self):
        """Test law enforcement initializes."""
        from src.witness.receipt_enforced_law import init_enforcement

        enforcement = init_enforcement({})

        assert enforcement is not None
        assert enforcement.enforcement_id is not None

    def test_extract_law_from_chain(self):
        """Test law extraction from receipt chain."""
        from src.witness.receipt_enforced_law import (
            init_enforcement,
            extract_law_from_chain,
        )

        enforcement = init_enforcement({})
        receipts = [
            {
                "receipt_type": "test",
                "ts": "2024-01-01T00:00:00Z",
                "payload_hash": "a:b",
            },
            {
                "receipt_type": "test",
                "ts": "2024-01-01T00:00:01Z",
                "payload_hash": "c:d",
            },
        ]
        law = extract_law_from_chain(enforcement, receipts)

        assert law is not None
        assert law.get("law_id") is not None
        assert law.get("extracted_from_chain") is True

    def test_validate_chain_causality(self):
        """Test chain causality validation."""
        from src.witness.receipt_enforced_law import (
            init_enforcement,
            validate_chain_causality,
        )

        enforcement = init_enforcement({})
        receipts = [
            {
                "receipt_type": "test",
                "ts": "2024-01-01T00:00:00Z",
                "payload_hash": "a:b",
            },
            {
                "receipt_type": "test",
                "ts": "2024-01-01T00:00:01Z",
                "payload_hash": "c:d",
            },
        ]
        valid = validate_chain_causality(enforcement, receipts)

        assert valid is True

    def test_causality_violation_detected(self):
        """Test causality violation is detected."""
        from src.witness.receipt_enforced_law import (
            init_enforcement,
            validate_chain_causality,
        )

        enforcement = init_enforcement({})
        # Out of order timestamps
        receipts = [
            {"receipt_type": "test", "ts": "2024-01-01T00:00:02Z"},
            {"receipt_type": "test", "ts": "2024-01-01T00:00:01Z"},
        ]
        valid = validate_chain_causality(enforcement, receipts)

        assert valid is False


class TestNoSyntheticReceipts:
    """Test that no synthetic receipts are generated."""

    def test_live_ingest_no_synthetic(self):
        """Test live ingest does not generate synthetic."""
        from src.swarm.live_triad_ingest import (
            init_live_ingest,
            batch_ingest,
            emit_live_ingest_receipt,
        )

        ingest = init_live_ingest({})
        batch_ingest(ingest, 10)
        receipt = emit_live_ingest_receipt(ingest)

        assert receipt["synthetic_enabled"] is False

    def test_live_only_result_no_synthetic(self):
        """Test live-only result indicates no synthetic."""
        from src.depths.d19_swarm_intelligence import run_d19_live_only

        result = run_d19_live_only()

        assert result["synthetic_enabled"] is False
        assert result["gates"]["gate_1"]["synthetic"] is False


class TestD19Status:
    """Test D19 status reflects live-only mode."""

    def test_d19_status_shows_live_only(self):
        """Test D19 status shows live-only mode enabled."""
        from src.depths.d19_swarm_intelligence import get_d19_status

        status = get_d19_status()

        assert status["live_only_mode"] is True
        assert status["synthetic_enabled"] is False
        assert status["entropy_source"] == "live_triad"

    def test_d19_version_is_19_1(self):
        """Test D19 version is 19.1.0."""
        from src.depths.d19_swarm_intelligence import get_d19_status

        status = get_d19_status()

        assert status["version"] == "19.1.0"
