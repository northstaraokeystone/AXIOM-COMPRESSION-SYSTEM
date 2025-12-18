"""autonomy.py - Common autonomy computation patterns.

Consolidates the autonomy computation logic used across all moon modules.
"""

from typing import Any, Dict, List


def compute_autonomy_from_latency(
    duration_hours: float,
    latency_min: List[float],
    earth_callback_max_pct: float,
) -> Dict[str, Any]:
    """Compute autonomy metrics from latency constraints.

    This is the common pattern used across Titan, Europa, Ganymede, Callisto modules.

    Args:
        duration_hours: Duration of operation in hours
        latency_min: [min_latency, max_latency] in minutes
        earth_callback_max_pct: Maximum allowed Earth callback percentage

    Returns:
        Dict with autonomy metrics:
            - earth_queries_possible: Total possible Earth round-trips
            - earth_queries_budget: Allowed Earth queries
            - local_decisions: Required local decisions
            - autonomy_achieved: Computed autonomy ratio (0-1)
    """
    # Compute possible Earth round-trips in the duration
    min_latency = (
        latency_min[0] if isinstance(latency_min, (list, tuple)) else latency_min
    )
    earth_queries_possible = (duration_hours * 60) / (min_latency * 2)  # Round-trip

    # Compute allowed queries and required local decisions
    earth_queries_budget = earth_queries_possible * earth_callback_max_pct
    local_decisions = earth_queries_possible - earth_queries_budget

    # Compute autonomy ratio
    autonomy_achieved = (
        round(local_decisions / earth_queries_possible, 4)
        if earth_queries_possible > 0
        else 1.0
    )

    return {
        "earth_queries_possible": round(earth_queries_possible, 2),
        "earth_queries_budget": round(earth_queries_budget, 2),
        "local_decisions": round(local_decisions, 2),
        "autonomy_achieved": autonomy_achieved,
    }


def check_autonomy_requirement(
    autonomy_achieved: float,
    autonomy_requirement: float,
) -> Dict[str, Any]:
    """Check if autonomy requirement is met.

    Args:
        autonomy_achieved: Computed autonomy ratio
        autonomy_requirement: Required autonomy level

    Returns:
        Dict with check results
    """
    return {
        "autonomy_achieved": autonomy_achieved,
        "autonomy_requirement": autonomy_requirement,
        "autonomy_met": autonomy_achieved >= autonomy_requirement,
        "margin": round(autonomy_achieved - autonomy_requirement, 4),
    }


def compute_combined_slo(
    alpha_result: Dict[str, Any],
    autonomy_result: Dict[str, Any],
    alpha_floor: float,
    autonomy_requirement: float,
) -> Dict[str, Any]:
    """Compute combined SLO for hybrid runs.

    This is the pattern used in d*_*_hybrid() functions.

    Args:
        alpha_result: Result from fractal recursion (must have 'eff_alpha', 'floor_met')
        autonomy_result: Result with 'autonomy_achieved'
        alpha_floor: Required alpha floor
        autonomy_requirement: Required autonomy level

    Returns:
        Dict with combined SLO results
    """
    autonomy_achieved = autonomy_result.get("autonomy_achieved", 0.0)
    autonomy_met = autonomy_achieved >= autonomy_requirement

    return {
        "alpha_target": alpha_floor,
        "alpha_achieved": alpha_result.get("eff_alpha", 0.0),
        "alpha_met": alpha_result.get("floor_met", False),
        "autonomy_target": autonomy_requirement,
        "autonomy_achieved": autonomy_achieved,
        "autonomy_met": autonomy_met,
        "all_targets_met": alpha_result.get("floor_met", False) and autonomy_met,
    }
