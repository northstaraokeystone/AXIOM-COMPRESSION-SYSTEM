"""Tests for D19.2 Preemptive Law Weaver.

D19.2: Laws are woven preemptively from projected future entropy trajectories.
Simulation KILLED. Reactive mode KILLED. Future projection only.

Grok's Core Insight:
  "Laws are not enforced reactively—they are woven preemptively
   from projected future entropy trajectories"
"""

import pytest
import json
import os


class TestD19_2SpecLoading:
    """Test D19.2 spec file loading."""

    def test_preemptive_weave_spec_exists(self):
        """Verify preemptive_weave_spec.json exists."""
        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "preemptive_weave_spec.json",
        )
        assert os.path.exists(spec_path), "preemptive_weave_spec.json must exist"

    def test_latency_catalog_exists(self):
        """Verify latency_catalog.json exists."""
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "latency_catalog.json",
        )
        assert os.path.exists(catalog_path), "latency_catalog.json must exist"

    def test_projection_horizon_config_exists(self):
        """Verify projection_horizon_config.json exists."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "projection_horizon_config.json",
        )
        assert os.path.exists(config_path), "projection_horizon_config.json must exist"

    def test_d19_spec_version_is_19_2(self):
        """Verify D19 spec version is 19.2.0."""
        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "d19_swarm_intelligence_spec.json",
        )
        with open(spec_path) as f:
            spec = json.load(f)
        assert spec.get("version") == "19.2.0", "D19 spec must be version 19.2.0"

    def test_d19_2_config_present(self):
        """Verify d19_2_config is present in spec."""
        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "d19_swarm_intelligence_spec.json",
        )
        with open(spec_path) as f:
            spec = json.load(f)
        assert "d19_2_config" in spec, "d19_2_config must be in spec"

    def test_simulation_disabled_in_d19_2_config(self):
        """Verify simulation is disabled in D19.2 config."""
        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "d19_swarm_intelligence_spec.json",
        )
        with open(spec_path) as f:
            spec = json.load(f)
        d19_2 = spec.get("d19_2_config", {})
        assert d19_2.get("simulation_enabled") is False, "Simulation must be disabled"

    def test_reactive_mode_disabled_in_d19_2_config(self):
        """Verify reactive mode is disabled in D19.2 config."""
        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "d19_swarm_intelligence_spec.json",
        )
        with open(spec_path) as f:
            spec = json.load(f)
        d19_2 = spec.get("d19_2_config", {})
        assert d19_2.get("reactive_mode_enabled") is False, "Reactive mode must be disabled"


class TestD19_2Constants:
    """Test D19.2 constants."""

    def test_simulation_enabled_is_false(self):
        """Verify SIMULATION_ENABLED is False."""
        from src.depths.d19_swarm_intelligence import SIMULATION_ENABLED
        assert SIMULATION_ENABLED is False, "SIMULATION_ENABLED must be False"

    def test_reactive_mode_enabled_is_false(self):
        """Verify REACTIVE_MODE_ENABLED is False."""
        from src.depths.d19_swarm_intelligence import REACTIVE_MODE_ENABLED
        assert REACTIVE_MODE_ENABLED is False, "REACTIVE_MODE_ENABLED must be False"

    def test_future_projection_mode_is_true(self):
        """Verify FUTURE_PROJECTION_MODE is True."""
        from src.depths.d19_swarm_intelligence import FUTURE_PROJECTION_MODE
        assert FUTURE_PROJECTION_MODE is True, "FUTURE_PROJECTION_MODE must be True"

    def test_proxima_rtt_years(self):
        """Verify PROXIMA_RTT_YEARS is 8.48."""
        from src.depths.d19_swarm_intelligence import PROXIMA_RTT_YEARS
        assert PROXIMA_RTT_YEARS == 8.48, "PROXIMA_RTT_YEARS must be 8.48"

    def test_projection_horizon_years(self):
        """Verify PROJECTION_HORIZON_YEARS is 10."""
        from src.depths.d19_swarm_intelligence import PROJECTION_HORIZON_YEARS
        assert PROJECTION_HORIZON_YEARS == 10, "PROJECTION_HORIZON_YEARS must be 10"

    def test_preemptive_amplify_threshold(self):
        """Verify PREEMPTIVE_AMPLIFY_THRESHOLD is 0.85."""
        from src.depths.d19_swarm_intelligence import PREEMPTIVE_AMPLIFY_THRESHOLD
        assert PREEMPTIVE_AMPLIFY_THRESHOLD == 0.85, "PREEMPTIVE_AMPLIFY_THRESHOLD must be 0.85"

    def test_preemptive_starve_threshold(self):
        """Verify PREEMPTIVE_STARVE_THRESHOLD is 0.50."""
        from src.depths.d19_swarm_intelligence import PREEMPTIVE_STARVE_THRESHOLD
        assert PREEMPTIVE_STARVE_THRESHOLD == 0.50, "PREEMPTIVE_STARVE_THRESHOLD must be 0.50"


class TestProjectionPackage:
    """Test projection package."""

    def test_projection_package_imports(self):
        """Verify projection package imports."""
        from src.projection import (
            init_projection,
            project_single_path,
            project_all_paths,
            get_projection_status,
        )
        assert init_projection is not None
        assert project_single_path is not None
        assert project_all_paths is not None
        assert get_projection_status is not None

    def test_latency_bound_model_imports(self):
        """Verify latency bound model imports."""
        from src.projection import (
            init_model,
            calculate_geodesic,
            validate_light_speed,
            get_model_status,
        )
        assert init_model is not None
        assert calculate_geodesic is not None
        assert validate_light_speed is not None
        assert get_model_status is not None

    def test_path_compression_estimator_imports(self):
        """Verify path compression estimator imports."""
        from src.projection import (
            init_estimator,
            estimate_path_compression,
            estimate_batch_compression,
            get_estimator_status,
        )
        assert init_estimator is not None
        assert estimate_path_compression is not None
        assert estimate_batch_compression is not None
        assert get_estimator_status is not None


class TestWeavePackage:
    """Test weave package."""

    def test_preemptive_weave_imports(self):
        """Verify preemptive weave imports."""
        from src.weave import (
            init_preemptive_weave,
            amplify_high_future_paths,
            starve_low_future_paths,
            apply_preemptive_selection,
            get_weave_status,
        )
        assert init_preemptive_weave is not None
        assert amplify_high_future_paths is not None
        assert starve_low_future_paths is not None
        assert apply_preemptive_selection is not None
        assert get_weave_status is not None

    def test_impending_entropy_weave_imports(self):
        """Verify impending entropy weave imports."""
        from src.weave import (
            init_entropy_weave,
            load_weave_template,
            weave_from_known_latency,
            get_entropy_weave_status,
        )
        assert init_entropy_weave is not None
        assert load_weave_template is not None
        assert weave_from_known_latency is not None
        assert get_entropy_weave_status is not None

    def test_delay_nullification_imports(self):
        """Verify delay nullification imports."""
        from src.weave import (
            init_nullification,
            nullify_known_delay,
            generate_preemptive_law,
            get_nullification_status,
        )
        assert init_nullification is not None
        assert nullify_known_delay is not None
        assert generate_preemptive_law is not None
        assert get_nullification_status is not None

    def test_weave_to_chain_imports(self):
        """Verify weave to chain imports."""
        from src.weave import (
            init_weave_chain,
            insert_woven_law,
            batch_insert_laws,
            verify_chain_integrity,
            get_chain_status,
        )
        assert init_weave_chain is not None
        assert insert_woven_law is not None
        assert batch_insert_laws is not None
        assert verify_chain_integrity is not None
        assert get_chain_status is not None


class TestFuturePathProjection:
    """Test future path projection functionality."""

    def test_init_projection(self):
        """Test projection initialization."""
        from src.projection import init_projection
        proj = init_projection({})
        assert proj is not None
        assert proj.projection_id is not None
        assert proj.horizon_years > 0

    def test_project_single_path_respects_light_speed(self):
        """Verify single path projection respects light speed."""
        from src.projection import init_projection, project_single_path
        proj = init_projection({})
        receipt = {"receipt_type": "test", "payload_hash": "test"}
        path = project_single_path(proj, receipt, "proxima_centauri")

        # Path must be light-speed valid
        assert path.light_speed_valid is True
        # Travel time must equal distance at light speed
        assert path.travel_time_years == path.distance_ly

    def test_projection_status_simulation_disabled(self):
        """Verify projection status shows simulation disabled."""
        from src.projection import get_projection_status
        status = get_projection_status()
        assert status.get("simulation_enabled") is False
        assert status.get("reactive_mode") is False


class TestPreemptiveWeave:
    """Test preemptive weave functionality."""

    def test_init_preemptive_weave(self):
        """Test preemptive weave initialization."""
        from src.weave import init_preemptive_weave
        weave = init_preemptive_weave({})
        assert weave is not None
        assert weave.weave_id is not None

    def test_amplify_high_future_paths(self):
        """Test amplification of high-future-compression paths."""
        from src.weave import init_preemptive_weave, amplify_high_future_paths
        weave = init_preemptive_weave({})
        paths = [
            {"path_id": "high_1", "projected_compression": 0.90},
            {"path_id": "high_2", "projected_compression": 0.88},
        ]
        amplified = amplify_high_future_paths(weave, paths)
        assert len(amplified) == 2
        for sel in amplified:
            assert sel.action == "amplify"

    def test_starve_low_future_paths(self):
        """Test starvation of low-future-compression paths."""
        from src.weave import init_preemptive_weave, starve_low_future_paths
        weave = init_preemptive_weave({})
        paths = [
            {"path_id": "low_1", "projected_compression": 0.40},
            {"path_id": "low_2", "projected_compression": 0.30},
        ]
        starved = starve_low_future_paths(weave, paths)
        assert len(starved) == 2
        for sel in starved:
            assert sel.action == "starve"

    def test_weave_status_reactive_disabled(self):
        """Verify weave status shows reactive mode disabled."""
        from src.weave import get_weave_status
        status = get_weave_status()
        assert status.get("reactive_mode_enabled") is False
        assert status.get("selection_on_past") is False


class TestImpendingEntropyWeave:
    """Test impending entropy weave functionality."""

    def test_init_entropy_weave(self):
        """Test entropy weave initialization."""
        from src.weave import init_entropy_weave
        weave = init_entropy_weave({})
        assert weave is not None
        assert weave.latency_catalog is not None

    def test_load_proxima_weave_template(self):
        """Test loading Proxima Centauri weave template."""
        from src.weave import init_entropy_weave, load_weave_template
        weave = init_entropy_weave({})
        template = load_weave_template(weave, "proxima_centauri")
        assert template is not None
        assert abs(template.latency_years - 8.48) < 0.1

    def test_weave_from_known_latency(self):
        """Test weaving from known latency."""
        from src.weave import init_entropy_weave, load_weave_template, weave_from_known_latency
        weave = init_entropy_weave({})
        template = load_weave_template(weave, "proxima_centauri")
        result = weave_from_known_latency(weave, template)
        assert result.get("laws_generated") > 0
        assert result.get("latency_is_input") is True

    def test_entropy_weave_status_latency_not_obstacle(self):
        """Verify entropy weave status shows latency is not obstacle."""
        from src.weave import get_entropy_weave_status
        status = get_entropy_weave_status()
        assert status.get("latency_as_obstacle") is False
        assert status.get("latency_is_input") is True


class TestDelayNullification:
    """Test delay nullification functionality."""

    def test_init_nullification(self):
        """Test nullification initialization."""
        from src.weave import init_nullification
        nullification = init_nullification({})
        assert nullification is not None
        assert nullification.nullification_id is not None

    def test_nullify_proxima_delay(self):
        """Test nullifying Proxima Centauri delay."""
        from src.weave import init_nullification, nullify_known_delay
        nullification = init_nullification({})
        law = nullify_known_delay(nullification, "proxima_centauri", 8.48)
        assert law is not None
        assert abs(law.delay_nullified_years - 8.48) < 0.01
        assert law.woven_into_chain is True


class TestWeaveToChain:
    """Test weave to chain functionality."""

    def test_init_weave_chain(self):
        """Test weave chain initialization."""
        from src.weave import init_weave_chain
        chain = init_weave_chain({})
        assert chain is not None
        assert chain.chain_id is not None

    def test_insert_woven_law(self):
        """Test inserting woven law into chain."""
        from src.weave import init_weave_chain, insert_woven_law
        chain = init_weave_chain({})
        woven = insert_woven_law(chain, "test_law", "delay_nullification", {"test": True})
        assert woven is not None
        assert woven.law_id == "test_law"
        assert chain.current_merkle_root is not None

    def test_verify_chain_integrity(self):
        """Test chain integrity verification."""
        from src.weave import init_weave_chain, insert_woven_law, verify_chain_integrity
        chain = init_weave_chain({})
        insert_woven_law(chain, "test_law", "delay_nullification", {"test": True})
        result = verify_chain_integrity(chain)
        assert result.get("integrity_valid") is True


class TestProjectedFutureFitness:
    """Test projected future fitness functionality."""

    def test_compute_projected_fitness(self):
        """Test computing projected fitness."""
        from src.autocatalytic.fitness_evaluator import compute_projected_fitness
        pattern = {"fitness": 0.8, "entropy": 1.0, "stability": 0.9}
        result = compute_projected_fitness(pattern)
        assert result.get("projected_fitness") is not None
        assert result.get("selection_on_past") is False

    def test_high_future_fitness_classified_amplify(self):
        """Test high future fitness classified for amplification."""
        from src.autocatalytic.fitness_evaluator import compute_projected_fitness
        pattern = {"fitness": 0.95, "entropy": 0.5, "stability": 0.95}
        result = compute_projected_fitness(pattern)
        assert result.get("classification") == "high_future"
        assert result.get("recommendation") == "amplify"

    def test_low_future_fitness_classified_starve(self):
        """Test low future fitness classified for starvation."""
        from src.autocatalytic.fitness_evaluator import compute_projected_fitness
        pattern = {"fitness": 0.2, "entropy": 5.0, "stability": 0.1}
        result = compute_projected_fitness(pattern)
        assert result.get("classification") == "low_future"
        assert result.get("recommendation") == "starve"


class TestD19_2Integration:
    """Integration tests for D19.2."""

    def test_run_d19_preemptive_executes(self):
        """Test that run_d19_preemptive executes successfully."""
        from src.depths.d19_swarm_intelligence import run_d19_preemptive
        result = run_d19_preemptive()
        assert result is not None
        assert result.get("mode") == "preemptive_weave"

    def test_run_d19_preemptive_simulation_killed(self):
        """Verify simulation is killed in D19.2 run."""
        from src.depths.d19_swarm_intelligence import run_d19_preemptive
        result = run_d19_preemptive()
        assert result.get("simulation_enabled") is False
        gates = result.get("gates", {})
        gate_5 = gates.get("gate_5", {})
        assert gate_5.get("simulation_killed") is True

    def test_run_d19_preemptive_reactive_killed(self):
        """Verify reactive mode is killed in D19.2 run."""
        from src.depths.d19_swarm_intelligence import run_d19_preemptive
        result = run_d19_preemptive()
        assert result.get("reactive_mode_enabled") is False
        gates = result.get("gates", {})
        gate_5 = gates.get("gate_5", {})
        assert gate_5.get("reactive_killed") is True

    def test_run_d19_preemptive_all_gates_pass(self):
        """Verify all gates pass in D19.2 run."""
        from src.depths.d19_swarm_intelligence import run_d19_preemptive
        result = run_d19_preemptive()
        assert result.get("all_gates_passed") is True

    def test_run_d19_preemptive_slo_passed(self):
        """Verify SLO passes in D19.2 run."""
        from src.depths.d19_swarm_intelligence import run_d19_preemptive
        result = run_d19_preemptive()
        assert result.get("slo_passed") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
