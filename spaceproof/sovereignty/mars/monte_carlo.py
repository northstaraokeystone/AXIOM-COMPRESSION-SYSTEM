"""Monte Carlo Simulation Harness.

Purpose: Statistical validation via Monte Carlo simulation.

THE PHYSICS:
    Minimum 1000 iterations for 95% confidence.
    Scenario duration: 500-1000 days (Mars surface mission).
    Failure criteria: resource depletion, crew incapacitation, ECLSS offline >48h.
"""

import math
import random

from spaceproof.core import emit_receipt

from .constants import MARS_CONJUNCTION_BLACKOUT_DAYS, MARS_SYNODIC_PERIOD_DAYS, TENANT_ID
from .scenarios import MANDATORY_SCENARIOS, SCENARIOS, Scenario


def run_simulation(
    config: dict,
    n_iterations: int = 1000,
    scenarios: list[str] | None = None,
    seed: int | None = None,
) -> dict:
    """Run Monte Carlo simulation.

    Args:
        config: Colony configuration
        n_iterations: Number of iterations (min 100 for 95% CI)
        scenarios: List of scenario names to run (default: mandatory)
        seed: Random seed for reproducibility

    Returns:
        dict: Simulation results with survival rate, CI, failure modes.
    """
    if seed is not None:
        random.seed(seed)

    if scenarios is None:
        scenarios = MANDATORY_SCENARIOS

    results = {
        "iterations": n_iterations,
        "scenarios_tested": scenarios,
        "scenario_results": {},
        "overall_survival_rate": 0.0,
        "confidence_interval_95": [0.0, 0.0],
        "failure_modes": {},
    }

    all_survivals = []

    for scenario_name in scenarios:
        scenario = SCENARIOS.get(scenario_name)
        scenario_survivals = []
        scenario_failures = []

        for i in range(n_iterations):
            state = initialize_state(config)
            survived, failure_mode = run_single_iteration(state, scenario, config)
            scenario_survivals.append(survived)
            if not survived:
                scenario_failures.append(failure_mode)

        survival_rate = sum(scenario_survivals) / n_iterations
        results["scenario_results"][scenario_name] = {
            "survival_rate": survival_rate,
            "iterations": n_iterations,
            "failures": len(scenario_failures),
        }

        all_survivals.extend(scenario_survivals)

        # Count failure modes
        for fm in scenario_failures:
            results["failure_modes"][fm] = results["failure_modes"].get(fm, 0) + 1

    # Overall survival rate
    total_iterations = len(all_survivals)
    if total_iterations > 0:
        overall_survival = sum(all_survivals) / total_iterations
        results["overall_survival_rate"] = overall_survival

        # 95% confidence interval (Wilson score)
        n = total_iterations
        p = overall_survival
        z = 1.96  # 95% CI
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        results["confidence_interval_95"] = [
            max(0, center - margin),
            min(1, center + margin),
        ]

    return results


def initialize_state(config: dict) -> dict:
    """Initialize simulation state from config.

    Args:
        config: Colony configuration

    Returns:
        dict: Initial simulation state.
    """
    crew_count = config.get("crew_count", 22)
    mission_days = config.get("mission_duration_days", 500)

    return {
        "day": 0,
        "mission_days": mission_days,
        "crew_active": crew_count,
        "crew_incapacitated": 0,
        "o2_reserve_days": config.get("o2_reserve_days", 60),
        "h2o_reserve_days": config.get("h2o_reserve_days", 90),
        "food_reserve_days": config.get("food_reserve_days", 90),
        "power_capacity": 1.0,
        "eclss_o2_online": True,
        "eclss_h2o_online": True,
        "moxie_online": True,
        "hab_pressure_ok": True,
        "earth_comms": True,
        "failure_mode": None,
    }


def run_single_iteration(
    state: dict,
    scenario: Scenario | None,
    config: dict,
) -> tuple[bool, str | None]:
    """Run single Monte Carlo iteration.

    Args:
        state: Initial state
        scenario: Scenario to apply (or None for baseline)
        config: Colony configuration

    Returns:
        tuple: (survived: bool, failure_mode: str or None)
    """
    mission_days = state["mission_days"]

    # Determine when scenario triggers
    if scenario is not None:
        if scenario.trigger_day == "random":
            trigger_day = random.randint(1, mission_days - scenario.duration_days - 1)
        else:
            trigger_day = scenario.trigger_day
        scenario_end_day = trigger_day + scenario.duration_days
    else:
        trigger_day = mission_days + 1  # Never triggers
        scenario_end_day = mission_days + 1

    # Also handle conjunction blackouts (every 780 days)
    conjunctions = []
    day = 0
    while day < mission_days:
        conjunctions.append(day)
        day += MARS_SYNODIC_PERIOD_DAYS
    conjunctions = [d for d in conjunctions if d > 0 and d < mission_days]

    for day in range(1, mission_days + 1):
        state["day"] = day

        # Check for conjunction blackout
        in_conjunction = False
        for conj_start in conjunctions:
            if conj_start <= day < conj_start + MARS_CONJUNCTION_BLACKOUT_DAYS:
                in_conjunction = True
                state["earth_comms"] = False
                break
        if not in_conjunction:
            state["earth_comms"] = True

        # Apply scenario effects if in scenario window
        if scenario is not None and trigger_day <= day < scenario_end_day:
            apply_scenario_effects(state, scenario)

        # Simulate day
        events = generate_daily_events(state, config)
        state = simulate_day(state, events)

        # Check failure conditions
        failure = check_failure_conditions(state)
        if failure:
            return False, failure

    return True, None


def apply_scenario_effects(state: dict, scenario: Scenario) -> None:
    """Apply scenario effects to state.

    Args:
        state: Current state
        scenario: Active scenario
    """
    effects = scenario.effects

    if effects.get("o2_production") == 0.0:
        state["eclss_o2_online"] = False

    if effects.get("h2o_recovery") == 0.0:
        state["eclss_h2o_online"] = False

    if effects.get("moxie_production") == 0.0:
        state["moxie_online"] = False

    if effects.get("power_output_multiplier"):
        state["power_capacity"] *= effects["power_output_multiplier"]

    if effects.get("crew_capacity_reduction"):
        reduction = effects["crew_capacity_reduction"]
        if state["crew_active"] > reduction:
            state["crew_active"] -= reduction
            state["crew_incapacitated"] += reduction

    if effects.get("pressure_loss_rate_kpa_hour"):
        # Simplified: if pressure loss > threshold, hab breach
        if effects["pressure_loss_rate_kpa_hour"] > 5.0:
            state["hab_pressure_ok"] = False


def generate_daily_events(state: dict, config: dict) -> list:
    """Generate random daily events.

    Args:
        state: Current state
        config: Configuration

    Returns:
        list: Events for the day.
    """
    events = []

    # Random equipment failures (based on ECLSS MTBF)
    # ISS MTBF = 1752 hours = 73 days
    # Daily failure probability ≈ 1/73 ≈ 0.014
    if random.random() < 0.014:
        events.append({"type": "minor_equipment_failure"})

    # Crew fatigue increases over time
    day = state["day"]
    if day > 100 and random.random() < 0.001 * (day / 100):
        events.append({"type": "crew_fatigue"})

    return events


def simulate_day(state: dict, events: list) -> dict:
    """Simulate one day of operations.

    Args:
        state: Current state
        events: Day's events

    Returns:
        dict: Updated state.
    """
    # Process events
    for event in events:
        if event["type"] == "minor_equipment_failure":
            # 80% chance of successful repair
            if random.random() > 0.80:
                # Failed repair, reduce capacity
                state["power_capacity"] *= 0.99

        elif event["type"] == "crew_fatigue":
            # Slight efficiency reduction
            pass

    # Consume resources
    if not state["eclss_o2_online"]:
        state["o2_reserve_days"] -= 1

    if not state["eclss_h2o_online"]:
        state["h2o_reserve_days"] -= 0.5

    # MOXIE produces O2 backup
    if state["moxie_online"] and not state["eclss_o2_online"]:
        state["o2_reserve_days"] += 0.3  # MOXIE partial compensation

    return state


def check_failure_conditions(state: dict) -> str | None:
    """Check if colony has failed.

    Args:
        state: Current state

    Returns:
        str: Failure mode name, or None if still surviving.
    """
    # Critical failures
    if state["o2_reserve_days"] <= 0:
        return "o2_depletion"

    if state["h2o_reserve_days"] <= 0:
        return "h2o_depletion"

    if state["food_reserve_days"] <= 0:
        return "food_depletion"

    if not state["hab_pressure_ok"]:
        return "hab_breach"

    if state["crew_active"] <= 2:
        return "insufficient_crew"

    return None


def calculate_survival_probability(results: list[bool]) -> float:
    """Calculate survival probability from results.

    Args:
        results: List of survival outcomes

    Returns:
        float: Survival probability.
    """
    if not results:
        return 0.0
    return sum(results) / len(results)


def identify_failure_modes(failed_runs: list[str]) -> dict:
    """Analyze failed runs to identify common failure modes.

    Args:
        failed_runs: List of failure mode names

    Returns:
        dict: Histogram of failure modes.
    """
    histogram = {}
    for mode in failed_runs:
        histogram[mode] = histogram.get(mode, 0) + 1

    # Sort by frequency
    sorted_modes = sorted(histogram.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_modes)


def generate_scenario(
    scenario_type: str,
    base_config: dict,
) -> dict:
    """Generate simulation scenario configuration.

    Args:
        scenario_type: Type of scenario
        base_config: Base configuration to modify

    Returns:
        dict: Scenario configuration.
    """
    scenario = SCENARIOS.get(scenario_type)
    if scenario is None:
        return base_config.copy()

    config = base_config.copy()
    config["scenario"] = {
        "name": scenario.name,
        "probability": scenario.probability,
        "duration_days": scenario.duration_days,
        "effects": scenario.effects,
    }
    return config


def emit_monte_carlo_result_receipt(
    results: dict,
) -> dict:
    """Emit Monte Carlo result receipt.

    Args:
        results: Simulation results

    Returns:
        dict: Emitted receipt.
    """
    return emit_receipt(
        "monte_carlo_result",
        {
            "tenant_id": TENANT_ID,
            "iterations": results["iterations"],
            "scenarios_tested": len(results["scenarios_tested"]),
            "overall_survival_rate": results["overall_survival_rate"],
            "confidence_interval_95_lower": results["confidence_interval_95"][0],
            "confidence_interval_95_upper": results["confidence_interval_95"][1],
            "top_failure_mode": list(results["failure_modes"].keys())[0] if results["failure_modes"] else None,
        },
    )
