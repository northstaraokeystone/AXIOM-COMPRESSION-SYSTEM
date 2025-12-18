"""Tests for randomized execution paths defense.

Coverage:
- Randomized config loading
- Execution tree generation
- Timing/power/cache resilience testing
- Full randomized audit
"""


class TestRandomizedConfig:
    """Tests for randomized paths configuration."""

    def test_randomized_config_loads(self):
        """Config loads with valid structure."""
        from src.randomized_paths_audit import load_randomized_config

        config = load_randomized_config()
        assert "path_depth" in config
        assert "timing_jitter_ns" in config
        assert "shuffle_factor" in config
        assert "resilience_target" in config

    def test_path_depth(self):
        """Path depth is 8."""
        from src.randomized_paths_audit import (
            load_randomized_config,
            RANDOMIZED_PATH_DEPTH,
        )

        config = load_randomized_config()
        assert config["path_depth"] == RANDOMIZED_PATH_DEPTH
        assert config["path_depth"] == 8

    def test_timing_jitter_range(self):
        """Timing jitter is in [10, 100] ns."""
        from src.randomized_paths_audit import (
            load_randomized_config,
            TIMING_JITTER_NS_MIN,
            TIMING_JITTER_NS_MAX,
        )

        config = load_randomized_config()
        assert config["timing_jitter_ns"][0] == TIMING_JITTER_NS_MIN
        assert config["timing_jitter_ns"][1] == TIMING_JITTER_NS_MAX
        assert config["timing_jitter_ns"] == [10, 100]

    def test_shuffle_factor(self):
        """Shuffle factor is 0.3."""
        from src.randomized_paths_audit import (
            load_randomized_config,
            EXECUTION_SHUFFLE_FACTOR,
        )

        config = load_randomized_config()
        assert config["shuffle_factor"] == EXECUTION_SHUFFLE_FACTOR
        assert config["shuffle_factor"] == 0.3

    def test_resilience_target(self):
        """Resilience target is 0.95."""
        from src.randomized_paths_audit import (
            load_randomized_config,
            TIMING_LEAK_RESILIENCE,
        )

        config = load_randomized_config()
        assert config["resilience_target"] == TIMING_LEAK_RESILIENCE
        assert config["resilience_target"] == 0.95


class TestExecutionTree:
    """Tests for execution tree generation."""

    def test_generate_tree(self):
        """Tree generation works."""
        from src.randomized_paths_audit import generate_execution_tree

        result = generate_execution_tree(8)
        assert result["depth"] == 8
        assert "tree" in result
        assert "total_nodes" in result

    def test_tree_has_nodes(self):
        """Tree has multiple nodes."""
        from src.randomized_paths_audit import generate_execution_tree

        result = generate_execution_tree(4)
        assert result["total_nodes"] > 1


class TestDefenseMechanisms:
    """Tests for defense mechanism functions."""

    def test_shuffle_instructions(self):
        """Instruction shuffling works."""
        from src.randomized_paths_audit import shuffle_instructions

        code_block = ["op1", "op2", "op3", "op4", "op5"]
        result = shuffle_instructions(code_block, 0.3)
        assert len(result) == len(code_block)
        # All original elements still present
        assert set(result) == set(code_block)

    def test_add_dummy_operations(self):
        """Dummy operation insertion works."""
        from src.randomized_paths_audit import add_dummy_operations

        code_block = ["op1", "op2", "op3"]
        result = add_dummy_operations(code_block, 0.5)
        # Result should have more elements (dummies added)
        assert len(result) >= len(code_block)
        # Check that dummies are present
        dummy_count = sum(1 for op in result if op.startswith("DUMMY_"))
        assert dummy_count > 0


class TestResilienceTesting:
    """Tests for resilience testing functions."""

    def test_timing_resilience(self):
        """Timing resilience test works."""
        from src.randomized_paths_audit import test_timing_resilience

        result = test_timing_resilience(50)
        assert "attack_type" in result
        assert result["attack_type"] == "timing_analysis"
        assert "resilience" in result
        assert 0 <= result["resilience"] <= 1

    def test_power_resilience(self):
        """Power resilience test works."""
        from src.randomized_paths_audit import test_power_resilience

        result = test_power_resilience(50)
        assert "attack_type" in result
        assert result["attack_type"] == "power_analysis"
        assert "resilience" in result
        assert 0 <= result["resilience"] <= 1

    def test_cache_resilience(self):
        """Cache resilience test works."""
        from src.randomized_paths_audit import test_cache_resilience

        result = test_cache_resilience(50)
        assert "attack_type" in result
        assert result["attack_type"] == "cache_timing"
        assert "resilience" in result
        assert 0 <= result["resilience"] <= 1


class TestFullAudit:
    """Tests for full randomized audit."""

    def test_attack_types_present(self):
        """All 3 attack types are tested."""
        from src.randomized_paths_audit import run_randomized_audit

        result = run_randomized_audit(iterations=50)
        assert len(result["attack_types_tested"]) == 3
        assert "timing_analysis" in result["attack_types_tested"]
        assert "power_analysis" in result["attack_types_tested"]
        assert "cache_timing" in result["attack_types_tested"]

    def test_defense_mechanisms_present(self):
        """All 3 defense mechanisms are configured."""
        from src.randomized_paths_audit import load_randomized_config

        config = load_randomized_config()
        assert len(config["defense_mechanisms"]) == 3
        assert "instruction_shuffle" in config["defense_mechanisms"]
        assert "dummy_operations" in config["defense_mechanisms"]
        assert "random_delays" in config["defense_mechanisms"]

    def test_avg_resilience_computed(self):
        """Average resilience is computed."""
        from src.randomized_paths_audit import run_randomized_audit

        result = run_randomized_audit(iterations=50)
        assert "avg_resilience" in result
        assert 0 <= result["avg_resilience"] <= 1

    def test_randomized_paths_audit_complete(self):
        """Full audit completes successfully."""
        from src.randomized_paths_audit import run_randomized_audit

        result = run_randomized_audit(iterations=50)
        assert "all_passed" in result
        assert "results" in result


class TestRecommendation:
    """Tests for path depth recommendation."""

    def test_recommend_low(self):
        """Low threat level recommends depth 4."""
        from src.randomized_paths_audit import recommend_path_depth

        depth = recommend_path_depth("low")
        assert depth == 4

    def test_recommend_medium(self):
        """Medium threat level recommends depth 6."""
        from src.randomized_paths_audit import recommend_path_depth

        depth = recommend_path_depth("medium")
        assert depth == 6

    def test_recommend_high(self):
        """High threat level recommends depth 8."""
        from src.randomized_paths_audit import recommend_path_depth

        depth = recommend_path_depth("high")
        assert depth == 8

    def test_recommend_critical(self):
        """Critical threat level recommends depth 10."""
        from src.randomized_paths_audit import recommend_path_depth

        depth = recommend_path_depth("critical")
        assert depth == 10


class TestModuleInfo:
    """Tests for module info function."""

    def test_get_randomized_info(self):
        """Module info returns complete structure."""
        from src.randomized_paths_audit import get_randomized_info

        info = get_randomized_info()
        assert info["module"] == "randomized_paths_audit"
        assert "version" in info
        assert "config" in info
        assert "capabilities" in info
        assert "attack_types" in info
        assert "defense_mechanisms" in info
