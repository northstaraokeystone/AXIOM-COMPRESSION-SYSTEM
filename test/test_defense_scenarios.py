"""Tests for defense expansion scenarios."""

import pytest
from spaceproof.sim.scenarios.orbital_compute import (
    OrbitalComputeScenario,
    OrbitalComputeConfig,
)
from spaceproof.sim.scenarios.constellation_scale import (
    ConstellationScaleScenario,
    ConstellationScaleConfig,
)
from spaceproof.sim.scenarios.autonomous_accountability import (
    AutonomousAccountabilityScenario,
    AutonomousAccountabilityConfig,
)
from spaceproof.sim.scenarios.firmware_supply_chain import (
    FirmwareSupplyChainScenario,
    FirmwareSupplyChainConfig,
)


class TestOrbitalComputeScenario:
    """Tests for SCENARIO_ORBITAL_COMPUTE."""

    def test_scenario_runs(self):
        """Test that scenario runs successfully."""
        config = OrbitalComputeConfig(
            inference_tasks=10,
            radiation_events=2,
            seed=42,
        )
        scenario = OrbitalComputeScenario(config)
        result = scenario.run()

        assert result.cycles_completed == 10
        assert result.inferences_with_receipts == 10

    def test_radiation_detection(self):
        """Test radiation events are detected."""
        config = OrbitalComputeConfig(
            inference_tasks=20,
            radiation_events=5,
            seed=42,
        )
        scenario = OrbitalComputeScenario(config)
        result = scenario.run()

        assert result.radiation_events_injected == 5
        # Should detect all or most radiation events
        assert result.radiation_events_detected >= 4

    def test_entropy_conservation(self):
        """Test entropy conservation is validated."""
        config = OrbitalComputeConfig(
            inference_tasks=10,
            radiation_events=0,  # No radiation
        )
        scenario = OrbitalComputeScenario(config)
        result = scenario.run()

        # Without radiation, entropy should be conserved
        assert result.entropy_conservation_violations == 0


class TestConstellationScaleScenario:
    """Tests for SCENARIO_CONSTELLATION_SCALE."""

    def test_scenario_runs(self):
        """Test that scenario runs successfully."""
        config = ConstellationScaleConfig(
            conjunction_alerts=5,
            deorbit_events=3,
            seed=42,
        )
        scenario = ConstellationScaleScenario(config)
        result = scenario.run()

        assert result.maneuvers_with_complete_chains == 5
        assert result.deorbits_with_receipts == 3

    def test_merkle_integrity(self):
        """Test Merkle chain integrity."""
        config = ConstellationScaleConfig(
            conjunction_alerts=5,
            deorbit_events=2,
        )
        scenario = ConstellationScaleScenario(config)
        result = scenario.run()

        assert result.merkle_chain_integrity_pct == 100.0

    def test_fcc_report_time(self):
        """Test FCC report generation time."""
        config = ConstellationScaleConfig(
            deorbit_events=10,
        )
        scenario = ConstellationScaleScenario(config)
        result = scenario.run()

        # Should complete in less than 5 seconds
        assert result.fcc_report_time_sec < 5.0


class TestAutonomousAccountabilityScenario:
    """Tests for SCENARIO_AUTONOMOUS_ACCOUNTABILITY."""

    def test_scenario_runs(self):
        """Test that scenario runs successfully."""
        config = AutonomousAccountabilityConfig(
            autonomous_decisions=20,
            human_overrides=3,
            adversarial_attacks=1,
            seed=42,
        )
        scenario = AutonomousAccountabilityScenario(config)
        result = scenario.run()

        assert result.decisions_with_lineage == 20
        assert result.decisions_with_override_flag == 20

    def test_override_reason_codes(self):
        """Test all overrides have reason codes."""
        config = AutonomousAccountabilityConfig(
            autonomous_decisions=10,
            human_overrides=5,
            adversarial_attacks=0,
        )
        scenario = AutonomousAccountabilityScenario(config)
        result = scenario.run()

        assert result.overrides_with_reason_code == 5
        assert result.overrides_total == 5

    def test_adversarial_detection(self):
        """Test adversarial attacks are detected."""
        config = AutonomousAccountabilityConfig(
            autonomous_decisions=20,
            human_overrides=2,
            adversarial_attacks=3,
        )
        scenario = AutonomousAccountabilityScenario(config)
        result = scenario.run()

        assert result.adversarial_attacks_injected == 3
        # Should detect all attacks via hash mismatch
        assert result.adversarial_attacks_detected == 3


class TestFirmwareSupplyChainScenario:
    """Tests for SCENARIO_FIRMWARE_SUPPLY_CHAIN."""

    def test_scenario_runs(self):
        """Test that scenario runs successfully."""
        config = FirmwareSupplyChainConfig(
            firmware_builds=10,
            malicious_injections=1,
            seed=42,
        )
        scenario = FirmwareSupplyChainScenario(config)
        result = scenario.run()

        assert result.builds_with_integrity_receipts == 10

    def test_malicious_detection(self):
        """Test malicious injections are detected."""
        config = FirmwareSupplyChainConfig(
            firmware_builds=20,
            malicious_injections=2,
        )
        scenario = FirmwareSupplyChainScenario(config)
        result = scenario.run()

        assert result.malicious_injections_total == 2
        # Should detect all injections via hash mismatch
        assert result.malicious_injections_detected == 2

    def test_verification_slo(self):
        """Test verification time SLO."""
        config = FirmwareSupplyChainConfig(
            firmware_builds=10,
        )
        scenario = FirmwareSupplyChainScenario(config)
        result = scenario.run()

        # Average verification should be < 1 second
        assert result.avg_verification_time_ms < 1000
        assert result.verification_slo_met is True

    def test_merkle_chain_verification(self):
        """Test Merkle chain is verified."""
        config = FirmwareSupplyChainConfig(
            firmware_builds=5,
        )
        scenario = FirmwareSupplyChainScenario(config)
        result = scenario.run()

        assert result.merkle_chain_verified is True


class TestAllCriteriaPassed:
    """Tests that scenarios can pass all criteria."""

    def test_orbital_compute_all_pass(self):
        """Test orbital compute can pass all criteria."""
        config = OrbitalComputeConfig(
            inference_tasks=50,
            radiation_events=5,
            seed=42,
        )
        scenario = OrbitalComputeScenario(config)
        result = scenario.run()

        # Check individual criteria
        assert result.inferences_with_receipts == result.inferences_total
        assert result.radiation_events_detected >= result.radiation_events_injected - 1

    def test_constellation_scale_all_pass(self):
        """Test constellation scale can pass all criteria."""
        config = ConstellationScaleConfig(
            conjunction_alerts=10,
            deorbit_events=5,
            seed=42,
        )
        scenario = ConstellationScaleScenario(config)
        result = scenario.run()

        assert result.maneuvers_with_complete_chains == result.maneuvers_total
        assert result.merkle_chain_integrity_pct == 100.0
        assert result.fcc_report_time_sec < 5.0

    def test_autonomous_accountability_all_pass(self):
        """Test autonomous accountability can pass all criteria."""
        config = AutonomousAccountabilityConfig(
            autonomous_decisions=30,
            human_overrides=5,
            adversarial_attacks=2,
            seed=42,
        )
        scenario = AutonomousAccountabilityScenario(config)
        result = scenario.run()

        assert result.decisions_with_lineage == result.decisions_total
        assert result.overrides_with_reason_code == result.overrides_total
        assert result.adversarial_attacks_detected == result.adversarial_attacks_injected

    def test_firmware_supply_chain_all_pass(self):
        """Test firmware supply chain can pass all criteria."""
        config = FirmwareSupplyChainConfig(
            firmware_builds=20,
            malicious_injections=2,
            seed=42,
        )
        scenario = FirmwareSupplyChainScenario(config)
        result = scenario.run()

        assert result.builds_with_integrity_receipts == result.builds_total
        assert result.malicious_injections_detected == result.malicious_injections_total
        assert result.merkle_chain_verified is True
        assert result.verification_slo_met is True
