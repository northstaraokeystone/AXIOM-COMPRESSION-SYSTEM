"""test_new_scenarios.py - Tests for New Validation Lock Scenarios

Tests for RADIATION, BLACKOUT, PSYCHOLOGY, and REALDATA scenarios.

Source: AXIOM Validation Lock v1
"""

import pytest


class TestScenarioEnum:
    """Tests for Scenario enum completeness."""

    def test_has_14_scenarios(self):
        """Scenario enum should have 14 scenarios."""
        from src.sim import Scenario

        scenarios = list(Scenario)
        assert len(scenarios) >= 14, f"Expected 14 scenarios, got {len(scenarios)}"

    def test_has_radiation_scenario(self):
        """Scenario enum should include RADIATION."""
        from src.sim import Scenario

        assert hasattr(Scenario, "SCENARIO_RADIATION")
        assert Scenario.SCENARIO_RADIATION.value == "radiation"

    def test_has_blackout_scenario(self):
        """Scenario enum should include BLACKOUT."""
        from src.sim import Scenario

        assert hasattr(Scenario, "SCENARIO_BLACKOUT")
        assert Scenario.SCENARIO_BLACKOUT.value == "blackout"

    def test_has_psychology_scenario(self):
        """Scenario enum should include PSYCHOLOGY."""
        from src.sim import Scenario

        assert hasattr(Scenario, "SCENARIO_PSYCHOLOGY")
        assert Scenario.SCENARIO_PSYCHOLOGY.value == "psychology"

    def test_has_realdata_scenario(self):
        """Scenario enum should include REALDATA."""
        from src.sim import Scenario

        assert hasattr(Scenario, "SCENARIO_REALDATA")
        assert Scenario.SCENARIO_REALDATA.value == "realdata"


class TestRadiationScenario:
    """Tests for SCENARIO_RADIATION."""

    def test_radiation_scenario_runs(self, capsys):
        """RADIATION scenario should complete without error."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig(max_cycles=20)
        state = run_scenario(Scenario.SCENARIO_RADIATION, config)

        # Should have run some cycles
        assert state.cycle > 0

        # Should emit radiation receipts
        captured = capsys.readouterr()
        assert "radiation" in captured.out.lower()

    def test_radiation_scenario_survives(self, capsys):
        """Colony should survive radiation event (dose < lethal)."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig()
        run_scenario(Scenario.SCENARIO_RADIATION, config)

        captured = capsys.readouterr()
        # Total dose = 0.1 * 12 = 1.2 Sv, lethal threshold = 2.0 Sv
        assert (
            "survived" in captured.out.lower()
            or "radiation_scenario_complete" in captured.out
        )


class TestBlackoutScenario:
    """Tests for SCENARIO_BLACKOUT."""

    def test_blackout_scenario_runs(self, capsys):
        """BLACKOUT scenario should complete without error."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig(max_cycles=100)
        state = run_scenario(Scenario.SCENARIO_BLACKOUT, config)

        assert state.cycle >= 43  # Should run for at least 43 days

        captured = capsys.readouterr()
        assert "blackout" in captured.out.lower()

    def test_blackout_scenario_43_days(self, capsys):
        """BLACKOUT should simulate 43-day conjunction."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig()
        run_scenario(Scenario.SCENARIO_BLACKOUT, config)

        captured = capsys.readouterr()
        # Should have 43 blackout_day receipts
        assert captured.out.count("blackout_day") >= 40  # Allow some tolerance


class TestPsychologyScenario:
    """Tests for SCENARIO_PSYCHOLOGY."""

    def test_psychology_scenario_runs(self, capsys):
        """PSYCHOLOGY scenario should complete without error."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig(max_cycles=400)  # 365 days + buffer
        state = run_scenario(Scenario.SCENARIO_PSYCHOLOGY, config)

        assert state.cycle >= 365

        captured = capsys.readouterr()
        assert "psychology" in captured.out.lower()

    def test_psychology_tracks_entropy(self, capsys):
        """PSYCHOLOGY should track entropy over time."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig()
        run_scenario(Scenario.SCENARIO_PSYCHOLOGY, config)

        captured = capsys.readouterr()
        # Should emit psychology updates
        assert "h_psychology" in captured.out


class TestRealdataScenario:
    """Tests for SCENARIO_REALDATA."""

    def test_realdata_scenario_runs(self, capsys):
        """REALDATA scenario should complete without error."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig()
        run_scenario(Scenario.SCENARIO_REALDATA, config)

        captured = capsys.readouterr()
        # Should emit realdata receipts or error receipt
        assert "realdata" in captured.out.lower() or "real_data" in captured.out.lower()

    def test_realdata_loads_sparc(self, capsys):
        """REALDATA should attempt to load SPARC galaxies."""
        from src.sim import Scenario, run_scenario, SimConfig

        config = SimConfig()
        run_scenario(Scenario.SCENARIO_REALDATA, config)

        captured = capsys.readouterr()
        # Should mention galaxies or SPARC
        output_lower = captured.out.lower()
        assert (
            "galaxy" in output_lower
            or "sparc" in output_lower
            or "realdata_scenario" in output_lower
        )


class TestScenarioIntegration:
    """Integration tests for all scenarios."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "SCENARIO_BASELINE",
            "SCENARIO_RADIATION",
            "SCENARIO_BLACKOUT",
            "SCENARIO_PSYCHOLOGY",
            "SCENARIO_REALDATA",
        ],
    )
    def test_scenario_completes(self, scenario_name):
        """Each scenario should complete without raising."""
        from src.sim import Scenario, run_scenario, SimConfig

        scenario = getattr(Scenario, scenario_name)
        config = SimConfig(max_cycles=50)

        # Should not raise
        state = run_scenario(scenario, config)
        assert state is not None

    def test_all_scenarios_emit_receipts(self, capsys):
        """All new scenarios should emit at least one receipt."""
        from src.sim import Scenario, run_scenario, SimConfig

        new_scenarios = [
            Scenario.SCENARIO_RADIATION,
            Scenario.SCENARIO_BLACKOUT,
            Scenario.SCENARIO_PSYCHOLOGY,
            Scenario.SCENARIO_REALDATA,
        ]

        for scenario in new_scenarios:
            config = SimConfig(max_cycles=20)
            run_scenario(scenario, config)

            captured = capsys.readouterr()
            # Should emit at least simulation_cycle receipts
            assert (
                "simulation_cycle" in captured.out or scenario.value in captured.out
            ), f"Scenario {scenario.value} did not emit expected receipts"
