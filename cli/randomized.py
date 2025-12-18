"""Randomized execution paths CLI commands.

Commands for:
- Randomized paths configuration display
- Execution tree generation
- Resilience testing and auditing
"""

import json


def cmd_randomized_info():
    """Show randomized paths configuration."""
    from src.randomized_paths_audit import get_randomized_info

    info = get_randomized_info()
    print(json.dumps(info, indent=2))


def cmd_randomized_config():
    """Show randomized paths configuration from spec."""
    from src.randomized_paths_audit import load_randomized_config

    config = load_randomized_config()
    print(json.dumps(config, indent=2))


def cmd_randomized_generate(depth: int = 8):
    """Generate randomized execution tree.

    Args:
        depth: Tree depth
    """
    from src.randomized_paths_audit import generate_execution_tree

    result = generate_execution_tree(depth)
    # Don't print full tree, just summary
    summary = {
        "depth": result["depth"],
        "total_nodes": result["total_nodes"],
        "randomization_factor": result["randomization_factor"],
    }
    print(json.dumps(summary, indent=2))


def cmd_randomized_audit(iterations: int = 100, simulate: bool = False):
    """Run full randomized paths audit.

    Args:
        iterations: Test iterations per attack type
        simulate: Whether to run in simulation mode
    """
    from src.randomized_paths_audit import run_randomized_audit

    result = run_randomized_audit(iterations=iterations)
    # Simplify results for display
    summary = {
        "attack_types_tested": result["attack_types_tested"],
        "iterations_per_type": result["iterations_per_type"],
        "avg_resilience": result["avg_resilience"],
        "target": result["target"],
        "all_passed": result["all_passed"],
        "results": {
            k: {
                "resilience": v["resilience"],
                "passed": v["passed"],
            }
            for k, v in result["results"].items()
        },
    }
    print(json.dumps(summary, indent=2))


def cmd_randomized_timing(iterations: int = 100, simulate: bool = False):
    """Test timing leak resilience.

    Args:
        iterations: Number of test iterations
        simulate: Whether to run in simulation mode
    """
    from src.randomized_paths_audit import test_timing_resilience

    result = test_timing_resilience(iterations)
    print(json.dumps(result, indent=2))


def cmd_randomized_power(iterations: int = 100, simulate: bool = False):
    """Test power analysis resilience.

    Args:
        iterations: Number of test iterations
        simulate: Whether to run in simulation mode
    """
    from src.randomized_paths_audit import test_power_resilience

    result = test_power_resilience(iterations)
    print(json.dumps(result, indent=2))


def cmd_randomized_cache(iterations: int = 100, simulate: bool = False):
    """Test cache timing resilience.

    Args:
        iterations: Number of test iterations
        simulate: Whether to run in simulation mode
    """
    from src.randomized_paths_audit import test_cache_resilience

    result = test_cache_resilience(iterations)
    print(json.dumps(result, indent=2))


def cmd_randomized_recommend(threat_level: str = "medium"):
    """Recommend path depth based on threat level.

    Args:
        threat_level: Threat level (low, medium, high, critical)
    """
    from src.randomized_paths_audit import recommend_path_depth

    depth = recommend_path_depth(threat_level)
    result = {
        "threat_level": threat_level,
        "recommended_depth": depth,
    }
    print(json.dumps(result, indent=2))
