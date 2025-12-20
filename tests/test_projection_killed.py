"""Tests to verify projection code is KILLED in D19.3.

D19.3: Projection KILLED. Preemptive weave KILLED. Simulation KILLED.
History is the only truth.
"""

import pytest
import os


class TestProjectionKilled:
    """Verify projection code is killed."""

    def test_projection_enabled_false(self):
        """Verify PROJECTION_ENABLED is False."""
        from src.depths.d19_swarm_intelligence import PROJECTION_ENABLED

        assert PROJECTION_ENABLED is False

    def test_simulation_enabled_false(self):
        """Verify SIMULATION_ENABLED is False."""
        from src.depths.d19_swarm_intelligence import SIMULATION_ENABLED

        assert SIMULATION_ENABLED is False

    def test_preemptive_weave_enabled_false(self):
        """Verify PREEMPTIVE_WEAVE_ENABLED is False."""
        from src.depths.d19_swarm_intelligence import PREEMPTIVE_WEAVE_ENABLED

        assert PREEMPTIVE_WEAVE_ENABLED is False

    def test_oracle_mode_live_history(self):
        """Verify ORACLE_MODE is live_history_only."""
        from src.depths.d19_swarm_intelligence import ORACLE_MODE

        assert ORACLE_MODE == "live_history_only"

    def test_projection_directory_killed(self):
        """Verify src/projection/ directory is killed (moved/deleted)."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        projection_path = os.path.join(base_path, "src", "projection")

        # Directory should not exist (moved to _killed_d19_3)
        assert not os.path.isdir(projection_path)

    def test_weave_directory_killed(self):
        """Verify src/weave/ directory is killed (moved/deleted)."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        weave_path = os.path.join(base_path, "src", "weave")

        # Directory should not exist (moved to _killed_d19_3)
        assert not os.path.isdir(weave_path)

    def test_killed_directory_exists(self):
        """Verify _killed_d19_3 directory exists with killed code."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        killed_path = os.path.join(base_path, "src", "_killed_d19_3")

        # Killed directory should exist
        assert os.path.isdir(killed_path)

        # Should contain projection and weave
        assert os.path.isdir(os.path.join(killed_path, "projection"))
        assert os.path.isdir(os.path.join(killed_path, "weave"))

    def test_verify_projection_killed_function(self):
        """Test the verify_projection_killed function."""
        from src.depths.d19_swarm_intelligence import verify_projection_killed

        result = verify_projection_killed()

        assert result["projection_killed"] is True
        assert result["simulation_killed"] is True
        assert result["preemptive_weave_killed"] is True
        assert result["projection_dir_exists"] is False
        assert result["weave_dir_exists"] is False
        assert result["killed_dir_exists"] is True
        assert result["passed"] is True


class TestKilledConstants:
    """Verify D19.2 constants are killed in D19.3."""

    def test_future_projection_mode_killed(self):
        """Verify FUTURE_PROJECTION_MODE is killed (None)."""
        from src.depths.d19_swarm_intelligence import FUTURE_PROJECTION_MODE_KILLED

        assert FUTURE_PROJECTION_MODE_KILLED is None

    def test_preemptive_amplify_threshold_killed(self):
        """Verify PREEMPTIVE_AMPLIFY_THRESHOLD is killed (None)."""
        from src.depths.d19_swarm_intelligence import PREEMPTIVE_AMPLIFY_THRESHOLD_KILLED

        assert PREEMPTIVE_AMPLIFY_THRESHOLD_KILLED is None

    def test_preemptive_starve_threshold_killed(self):
        """Verify PREEMPTIVE_STARVE_THRESHOLD is killed (None)."""
        from src.depths.d19_swarm_intelligence import PREEMPTIVE_STARVE_THRESHOLD_KILLED

        assert PREEMPTIVE_STARVE_THRESHOLD_KILLED is None

    def test_weave_horizon_killed(self):
        """Verify WEAVE_HORIZON is killed (None)."""
        from src.depths.d19_swarm_intelligence import WEAVE_HORIZON_KILLED

        assert WEAVE_HORIZON_KILLED is None


class TestOraclePackageExists:
    """Verify oracle package replaces killed packages."""

    def test_oracle_package_exists(self):
        """Verify src/oracle/ package exists."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        oracle_path = os.path.join(base_path, "src", "oracle")

        assert os.path.isdir(oracle_path)

    def test_oracle_modules_exist(self):
        """Verify oracle modules exist."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        oracle_path = os.path.join(base_path, "src", "oracle")

        assert os.path.isfile(os.path.join(oracle_path, "__init__.py"))
        assert os.path.isfile(os.path.join(oracle_path, "live_history_oracle.py"))
        assert os.path.isfile(os.path.join(oracle_path, "causal_subgraph_extractor.py"))
        assert os.path.isfile(os.path.join(oracle_path, "instant_incorporator.py"))
        assert os.path.isfile(os.path.join(oracle_path, "gap_silence_emergence.py"))

    def test_oracle_imports(self):
        """Verify oracle package imports work."""
        from src.oracle import (
            LiveHistoryOracle,
            CausalSubgraphExtractor,
            InstantIncorporator,
            GapSilenceEmergence,
        )

        # Just verify the imports succeed
        assert LiveHistoryOracle is not None
        assert CausalSubgraphExtractor is not None
        assert InstantIncorporator is not None
        assert GapSilenceEmergence is not None
