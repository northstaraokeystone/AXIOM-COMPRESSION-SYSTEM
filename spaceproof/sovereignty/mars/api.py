"""Mars Sovereignty High-Level API.

Purpose: Provide clean interface for CLI and external callers.

Usage:
    from spaceproof.sovereignty.mars import calculate_mars_sovereignty

    result = calculate_mars_sovereignty(
        config_path="config/mars/mars_nominal.yaml",
        monte_carlo=True,
        iterations=1000
    )
"""

import json
from pathlib import Path
from typing import Any

import yaml

from spaceproof.core import emit_receipt

from .constants import (
    BUFFER_DAYS_MINIMUM,
    CREW_MIN_GEORGE_MASON,
    CREW_MIN_SALOTTI,
    DEFAULT_WEIGHTS,
    HUMAN_O2_KG_PER_DAY,
    HUMAN_WATER_KG_PER_DAY,
    ISS_ECLSS_MTBF_HOURS,
    ISS_H2O_RECOVERY_RATIO,
    ISS_O2_CLOSURE_RATIO,
    MARS_LIGHT_DELAY_AVG_SEC,
    TENANT_ID,
)
from .crew_matrix import (
    calculate_coverage,
    calculate_redundancy,
    compute_crew_entropy,
    define_skill_matrix,
    identify_gaps,
)
from .decision_capacity import (
    calculate_conjunction_survival,
    calculate_earth_input_rate,
    calculate_internal_capacity,
    compute_sovereignty_threshold,
)
from .integrator import (
    calculate_comprehensive_sovereignty,
    calculate_sovereignty_score,
    emit_mars_sovereignty_receipt,
    validate_against_research,
)
from .life_support import (
    calculate_eclss_reliability,
    calculate_h2o_balance,
    calculate_life_support_entropy_rate,
    calculate_o2_balance,
    calculate_thermal_entropy,
)
from .monte_carlo import emit_monte_carlo_result_receipt, run_simulation
from .resources import (
    calculate_isru_closure,
    calculate_reserve_buffer,
    calculate_resource_score,
    calculate_starship_manifest,
)


def load_config(config_path: str) -> dict:
    """Load colony configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Colony configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def build_crew_from_config(config: dict) -> list[dict]:
    """Build crew list from config.

    Args:
        config: Colony configuration

    Returns:
        list: Crew member dicts with skills.
    """
    crew = []
    crew_skills = config.get("crew_skills", {})
    crew_count = config.get("colony", {}).get("crew_count", 22)

    # Build crew from skill definitions
    skill_matrix = define_skill_matrix()
    assigned = 0

    for skill_name, members in crew_skills.items():
        if isinstance(members, list):
            for member in members:
                if assigned < crew_count:
                    crew.append(
                        {
                            "name": member.get("name", f"Crew_{assigned}"),
                            "skills": {skill_name: member.get("proficiency", 0.7)},
                            "expertise_level": member.get("proficiency", 0.7),
                            "workload_hours": 40,
                        }
                    )
                    assigned += 1

    # Fill remaining crew with generalists
    while len(crew) < crew_count:
        crew.append(
            {
                "name": f"Generalist_{len(crew)}",
                "skills": {k: 0.3 for k in skill_matrix.keys()},
                "expertise_level": 0.5,
                "workload_hours": 40,
            }
        )

    return crew


def calculate_mars_sovereignty(
    config_path: str | None = None,
    config: dict | None = None,
    monte_carlo: bool = False,
    iterations: int = 1000,
    scenario: str | None = None,
) -> dict:
    """Calculate Mars sovereignty score for a colony configuration.

    This is the main entry point for Mars sovereignty calculations.

    Args:
        config_path: Path to YAML config file
        config: Direct config dict (alternative to config_path)
        monte_carlo: Run Monte Carlo validation
        iterations: Number of MC iterations
        scenario: Specific scenario to run

    Returns:
        dict: Comprehensive sovereignty result with score, components, validation.
    """
    # Load config
    if config is None:
        if config_path is None:
            raise ValueError("Must provide config_path or config")
        config = load_config(config_path)

    colony = config.get("colony", {})
    crew_count = colony.get("crew_count", 22)
    mission_days = colony.get("mission_duration_days", 500)

    # Build crew
    crew = build_crew_from_config(config)

    # Calculate crew metrics
    skills = define_skill_matrix()
    coverage = calculate_coverage(crew, skills)
    redundancy = calculate_redundancy(crew, skills)
    gaps = identify_gaps(crew, skills)
    workload_entropy = compute_crew_entropy(crew)

    crew_metrics = {
        "crew_count": crew_count,
        "coverage": coverage,
        "redundancy": redundancy,
        "gaps": gaps,
        "workload_entropy": workload_entropy,
    }

    # Calculate life support metrics
    life_support = config.get("life_support", {})
    o2_closure = life_support.get("eclss_closure_o2", ISS_O2_CLOSURE_RATIO)
    h2o_closure = life_support.get("eclss_closure_h2o", ISS_H2O_RECOVERY_RATIO)
    moxie_units = life_support.get("moxie_units", 1)
    mtbf = life_support.get("eclss_mtbf_hours", ISS_ECLSS_MTBF_HOURS)
    redundancy_factor = life_support.get("redundancy_factor", 1.0)

    power = config.get("power", {})
    power_available = (
        power.get("kilopower_units", 0) * 10000
        + power.get("solar_array_m2", 100) * 150
    )

    o2_balance = calculate_o2_balance(crew_count, moxie_units, o2_closure, power_available)
    h2o_balance = calculate_h2o_balance(
        crew_count,
        h2o_closure,
        config.get("isru", {}).get("water_extraction_kg_day", 0),
    )
    thermal = calculate_thermal_entropy(
        crew_count, power_available * 0.3, power.get("radiator_area_m2", 50)
    )
    reliability = calculate_eclss_reliability(mtbf, redundancy_factor, 0.8, mission_days)

    eclss_config = {
        "o2_closure": o2_closure,
        "h2o_closure": h2o_closure,
        "mtbf_hours": mtbf,
        "redundancy_factor": redundancy_factor,
    }
    entropy_rate = calculate_life_support_entropy_rate(crew_count, eclss_config)

    life_support_metrics = {
        "o2_balance": o2_balance,
        "h2o_balance": h2o_balance,
        "thermal": thermal,
        "reliability": reliability,
        "entropy_rate": entropy_rate,
    }

    # Calculate decision capacity metrics
    comm = config.get("communication", {})
    bandwidth = comm.get("bandwidth_mbps", 2.0)
    latency = comm.get("latency_one_way_sec", MARS_LIGHT_DELAY_AVG_SEC)

    internal_capacity = calculate_internal_capacity(crew)
    earth_capacity = calculate_earth_input_rate(bandwidth, latency)
    sovereign = compute_sovereignty_threshold(internal_capacity, earth_capacity)
    conjunction_survival = calculate_conjunction_survival(internal_capacity)

    decision_metrics = {
        "internal_capacity_bps": internal_capacity,
        "earth_capacity_bps": earth_capacity,
        "sovereign": sovereign,
        "advantage_ratio": internal_capacity / earth_capacity if earth_capacity > 0 else float("inf"),
        "conjunction_survival_probability": conjunction_survival,
    }

    # Calculate resource metrics
    resources = config.get("resources", {})
    isru = config.get("isru", {})

    production = {
        "o2": isru.get("o2_production_moxie_kg_day", 0.264) + crew_count * HUMAN_O2_KG_PER_DAY * o2_closure,
        "h2o": isru.get("water_extraction_kg_day", 0) + crew_count * HUMAN_WATER_KG_PER_DAY * h2o_closure,
        "food": resources.get("food_production_kg_day", 0),
    }

    consumption = {
        "o2": crew_count * HUMAN_O2_KG_PER_DAY,
        "h2o": crew_count * HUMAN_WATER_KG_PER_DAY,
        "food": crew_count * 1.8,
    }

    reserves = {
        "o2": resources.get("o2_reserve_kg", 500),
        "h2o": resources.get("water_reserve_kg", 5000),
        "food": resources.get("food_reserve_kcal", 15000000) / 2500 * 1.8,
    }

    resource_result = calculate_resource_score(crew_count, production, consumption, reserves)

    resource_metrics = {
        "closure_ratio": resource_result["closure_ratio"],
        "buffer_status": resource_result["buffer_status"],
        "binding_resource": resource_result["binding_resource"],
    }

    # Calculate comprehensive sovereignty
    sovereignty_result = calculate_comprehensive_sovereignty(
        crew_metrics, life_support_metrics, decision_metrics, resource_metrics
    )

    # Emit receipt
    emit_mars_sovereignty_receipt(crew_count, sovereignty_result, config)

    # Monte Carlo validation
    mc_results = None
    if monte_carlo:
        mc_config = {
            "crew_count": crew_count,
            "mission_duration_days": mission_days,
            "o2_reserve_days": reserves["o2"] / consumption["o2"],
            "h2o_reserve_days": reserves["h2o"] / consumption["h2o"],
            "food_reserve_days": reserves["food"] / consumption["food"],
        }
        mc_results = run_simulation(
            mc_config,
            n_iterations=iterations,
            scenarios=[scenario] if scenario else None,
        )
        emit_monte_carlo_result_receipt(mc_results)

    # Build result
    result = {
        "crew_count": crew_count,
        "sovereignty_score": sovereignty_result["sovereignty_score"],
        "is_sovereign": sovereignty_result["is_sovereign"],
        "can_survive_conjunction": conjunction_survival > 0.9,
        "binding_constraint": sovereignty_result["binding_constraint"],
        "research_benchmark_match": sovereignty_result["research_validated"],
        "component_scores": sovereignty_result["component_scores"],
        "crew_coverage": coverage,
        "life_support_entropy": entropy_rate,
        "decision_capacity_internal_bps": internal_capacity,
        "decision_capacity_earth_bps": earth_capacity,
        "resource_closure_ratio": resource_result["closure_ratio"],
        "warnings": [],
    }

    # Add warnings
    if gaps:
        result["warnings"].append(f"Skill gaps detected: {len(gaps)}")
    if entropy_rate > 0:
        result["warnings"].append("Life support entropy positive (unstable)")
    if not sovereign:
        result["warnings"].append("Colony not computationally sovereign")

    if mc_results:
        result["monte_carlo_survival_rate"] = mc_results["overall_survival_rate"]
        result["monte_carlo_confidence_95"] = mc_results["confidence_interval_95"]
        result["monte_carlo_failure_modes"] = mc_results["failure_modes"]

    return result


def find_crew_threshold(
    target_score: float = 95.0,
    base_config: dict | None = None,
) -> dict:
    """Find minimum crew size for target sovereignty score.

    Args:
        target_score: Target sovereignty score (0-100)
        base_config: Base configuration (default used if None)

    Returns:
        dict: Result with crew threshold and details.
    """
    if base_config is None:
        base_config = get_default_config()

    low, high = 4, 200
    results = []

    while low < high:
        mid = (low + high) // 2
        score = evaluate_config_with_crew(base_config, mid)
        results.append({"crew": mid, "score": score})

        if score >= target_score:
            high = mid
        else:
            low = mid + 1

    return {
        "target_score": target_score,
        "threshold_crew": low,
        "achieved_score": evaluate_config_with_crew(base_config, low),
        "search_results": results,
    }


def evaluate_config_with_crew(config: dict, crew_count: int) -> float:
    """Evaluate sovereignty score for given crew count.

    Args:
        config: Base configuration
        crew_count: Crew size to evaluate

    Returns:
        float: Sovereignty score.
    """
    test_config = config.copy()
    if "colony" not in test_config:
        test_config["colony"] = {}
    test_config["colony"]["crew_count"] = crew_count

    result = calculate_mars_sovereignty(config=test_config)
    return result["sovereignty_score"]


def compare_configs(config_path1: str, config_path2: str) -> dict:
    """Compare two colony configurations.

    Args:
        config_path1: Path to first config
        config_path2: Path to second config

    Returns:
        dict: Comparison result.
    """
    result1 = calculate_mars_sovereignty(config_path=config_path1)
    result2 = calculate_mars_sovereignty(config_path=config_path2)

    return {
        "config1": {
            "path": config_path1,
            "crew": result1["crew_count"],
            "score": result1["sovereignty_score"],
            "sovereign": result1["is_sovereign"],
        },
        "config2": {
            "path": config_path2,
            "crew": result2["crew_count"],
            "score": result2["sovereignty_score"],
            "sovereign": result2["is_sovereign"],
        },
        "difference": {
            "score_delta": result2["sovereignty_score"] - result1["sovereignty_score"],
            "crew_delta": result2["crew_count"] - result1["crew_count"],
        },
    }


def generate_report(result: dict, output_path: str) -> None:
    """Generate markdown report from sovereignty result.

    Args:
        result: Sovereignty calculation result
        output_path: Path to write markdown file
    """
    report = f"""# Mars Sovereignty Analysis Report

## Summary

| Metric | Value |
|--------|-------|
| Crew Size | {result['crew_count']} |
| Sovereignty Score | {result['sovereignty_score']:.1f}% |
| Is Sovereign | {'Yes' if result['is_sovereign'] else 'No'} |
| Can Survive Conjunction | {'Yes' if result['can_survive_conjunction'] else 'No'} |
| Binding Constraint | {result['binding_constraint']} |
| Research Validated | {'Yes' if result['research_benchmark_match'] else 'No'} |

## Component Scores

| Component | Score |
|-----------|-------|
| Crew Coverage | {result['crew_coverage']:.2f} |
| Life Support Entropy | {result['life_support_entropy']:.3f} |
| Decision Capacity (Internal) | {result['decision_capacity_internal_bps']:.2f} bps |
| Decision Capacity (Earth) | {result['decision_capacity_earth_bps']:.2f} bps |
| Resource Closure | {result['resource_closure_ratio']:.2f} |

## Warnings

"""
    if result['warnings']:
        for warning in result['warnings']:
            report += f"- {warning}\n"
    else:
        report += "No warnings.\n"

    if 'monte_carlo_survival_rate' in result:
        report += f"""
## Monte Carlo Validation

| Metric | Value |
|--------|-------|
| Survival Rate | {result['monte_carlo_survival_rate']:.1%} |
| 95% CI | [{result['monte_carlo_confidence_95'][0]:.1%}, {result['monte_carlo_confidence_95'][1]:.1%}] |

### Top Failure Modes

"""
        for mode, count in list(result.get('monte_carlo_failure_modes', {}).items())[:5]:
            report += f"- {mode}: {count}\n"

    report += """
## Research References

- George Mason 2023: Minimum viable crew 22 via agent-based modeling
- Salotti Nature 2020: Minimum viable crew 110 via work-capacity analysis
- NASA ECLSS 2019: ISS MTBF 1752h (5.6x lower than design)
- Perseverance MOXIE 2021-2025: 5.5 g/hr O2 production measured

---
*Generated by SpaceProof Mars Sovereignty Simulator*
"""

    with open(output_path, "w") as f:
        f.write(report)


def get_default_config() -> dict:
    """Get default colony configuration.

    Returns:
        dict: Default config for 22-person colony.
    """
    return {
        "colony": {
            "name": "Mars Default Colony",
            "crew_count": 22,
            "mission_duration_days": 500,
        },
        "life_support": {
            "eclss_closure_o2": 0.85,
            "eclss_closure_h2o": 0.95,
            "moxie_units": 2,
            "eclss_mtbf_hours": 1752,
            "redundancy_factor": 1.5,
        },
        "power": {
            "solar_array_m2": 200,
            "kilopower_units": 2,
            "radiator_area_m2": 50,
        },
        "resources": {
            "water_reserve_kg": 5000,
            "food_reserve_kcal": 15000000,
            "o2_reserve_kg": 500,
        },
        "isru": {
            "water_extraction_kg_day": 50,
            "o2_production_moxie_kg_day": 0.264,
        },
        "communication": {
            "bandwidth_mbps": 2.0,
            "latency_one_way_sec": 660,
        },
        "crew_skills": {
            "medical": [
                {"name": "Doctor A", "proficiency": 1.0},
                {"name": "Doctor B", "proficiency": 0.7},
            ],
            "engineering": [
                {"name": "Engineer A", "proficiency": 1.0},
                {"name": "Engineer B", "proficiency": 1.0},
                {"name": "Engineer C", "proficiency": 0.7},
            ],
            "systems": [
                {"name": "Systems A", "proficiency": 1.0},
                {"name": "Systems B", "proficiency": 1.0},
            ],
            "life_support": [
                {"name": "LifeSupport A", "proficiency": 1.0},
                {"name": "LifeSupport B", "proficiency": 0.7},
            ],
            "agriculture": [
                {"name": "Botanist A", "proficiency": 1.0},
                {"name": "Botanist B", "proficiency": 0.7},
            ],
        },
    }
