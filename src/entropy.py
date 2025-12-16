"""BUILD C2: Information-theoretic entropy calculations.

THE NOVEL DIMENSION: Colony survival = entropy export capacity.
Shannon 1948, not metaphor. H = -Σ p(x) log₂ p(x)

Source: CLAUDEME §8, AXIOM_Colony_Build_Strategy_v2.md §2.4
"""
import math
from datetime import datetime
from typing import Dict, List

import numpy as np

from src.core import emit_receipt, TENANT_ID

# === VERIFIED CONSTANTS (Grok Research) ===

HUMAN_METABOLIC_W = 100
"""Human metabolic rate in watts. Physiology (measured), 80-120 range midpoint."""

MOXIE_O2_G_PER_HR = 5.5
"""MOXIE oxygen production g/hr. NASA Perseverance 2021-25, 5-6 range midpoint."""

ISS_WATER_RECOVERY = 0.98
"""ISS water recovery ratio. NASA ECLSS 2023, measured."""

ISS_O2_CLOSURE = 0.875
"""ISS oxygen closure ratio. NASA, 85-90% range midpoint."""

MARS_RELAY_MBPS = 2.0
"""Mars relay bandwidth Mbps. NASA MRO, measured."""

LIGHT_DELAY_MIN = 3
"""Minimum light delay Earth-Mars in minutes. Physics, opposition."""

LIGHT_DELAY_MAX = 22
"""Maximum light delay Earth-Mars in minutes. Physics, conjunction."""

SOLAR_FLUX_MAX = 590
"""Maximum solar flux on Mars W/m². NASA Viking, equator."""

SOLAR_FLUX_DUST = 6
"""Solar flux during global dust storm W/m². NASA, ~1% of max."""

KILOPOWER_KW = 10
"""Kilopower unit output in kW. NASA KRUSTY, per unit."""

# === PLACEHOLDER CONSTANTS (Derived - Need Research) ===

DECISION_BITS_PER_PERSON_PER_SEC = 0.1
"""RESEARCH_PLACEHOLDER: ~8640 decisions/day at 1 bit each. Needs R1.1."""

EXPERTISE_MULTIPLIER = 2.0
"""RESEARCH_PLACEHOLDER: Expert decides 2x faster. Needs R1.1."""

LATENCY_COST_FACTOR = 0.8
"""RESEARCH_PLACEHOLDER: Usable fraction per round-trip. Needs R1.2."""

SUBSYSTEM_WEIGHTS = {
    "atmosphere": 1.0,
    "thermal": 1.0,
    "resource": 1.0,
    "decision": 2.0  # Decision weighted 2x (binding constraint)
}
"""Subsystem weights for total entropy. Decision is binding constraint (novel claim)."""


# === CORE ENTROPY FUNCTIONS ===

def shannon_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy H = -Σ p(x) log₂ p(x).

    Args:
        distribution: Probability distribution as numpy array.

    Returns:
        Entropy in bits. 0.0 for empty or all-zeros input.

    Notes:
        - Add epsilon=1e-12 to avoid log(0)
        - Normalize if sum != 1.0
        - Use numpy for vectorized operations
    """
    if distribution.size == 0:
        return 0.0

    dist_sum = np.sum(distribution)
    if dist_sum == 0:
        return 0.0

    # Normalize if needed
    if dist_sum != 1.0:
        distribution = distribution / dist_sum

    # Add epsilon to avoid log(0)
    epsilon = 1e-12
    distribution = distribution + epsilon
    distribution = distribution / np.sum(distribution)  # Renormalize after epsilon

    # H = -Σ p(x) log₂ p(x)
    return float(-np.sum(distribution * np.log2(distribution)))


def subsystem_entropy(state: dict, subsystem: str) -> float:
    """Extract subsystem state and compute its entropy.

    Args:
        state: Colony state dictionary.
        subsystem: One of "atmosphere", "thermal", "resource", "decision".

    Returns:
        Shannon entropy of the subsystem's state variables.

    Subsystem extractors:
        - atmosphere: O2, CO2, N2 percentages as distribution
        - thermal: Normalized temperature deviation from bounds
        - resource: Water, food, power ratios to requirements
        - decision: Expertise coverage, latency normalized
    """
    if subsystem == "atmosphere":
        # Extract gas percentages
        o2 = state.get("O2_pct", 0.21)
        co2 = state.get("CO2_pct", 0.0004)
        n2 = state.get("N2_pct", 0.78)
        other = max(0, 1.0 - o2 - co2 - n2)
        dist = np.array([o2, co2, n2, other])

    elif subsystem == "thermal":
        # Normalized temperature deviation from bounds
        temp = state.get("temperature_C", 20.0)
        temp_min = state.get("temp_min_C", 18.0)
        temp_max = state.get("temp_max_C", 24.0)
        temp_range = temp_max - temp_min if temp_max > temp_min else 1.0
        deviation = abs(temp - (temp_min + temp_max) / 2) / temp_range
        # Create distribution from deviation
        stable = max(0, 1.0 - deviation)
        dist = np.array([stable, deviation])

    elif subsystem == "resource":
        # Water, food, power ratios to requirements
        water_ratio = state.get("water_ratio", 1.0)
        food_ratio = state.get("food_ratio", 1.0)
        power_ratio = state.get("power_ratio", 1.0)
        # Clamp to [0, 1]
        water_ratio = max(0, min(1, water_ratio))
        food_ratio = max(0, min(1, food_ratio))
        power_ratio = max(0, min(1, power_ratio))
        dist = np.array([water_ratio, food_ratio, power_ratio])

    elif subsystem == "decision":
        # Expertise coverage, latency normalized
        expertise = state.get("expertise", {})
        latency = state.get("latency_min", LIGHT_DELAY_MIN)
        # Expertise coverage: mean proficiency
        if expertise:
            coverage = sum(expertise.values()) / len(expertise)
        else:
            coverage = 0.5  # Baseline
        # Latency normalized: 0 at min, 1 at max
        latency_norm = (latency - LIGHT_DELAY_MIN) / max(1, LIGHT_DELAY_MAX - LIGHT_DELAY_MIN)
        latency_norm = max(0, min(1, latency_norm))
        dist = np.array([coverage, 1 - coverage, latency_norm, 1 - latency_norm])

    else:
        # Unknown subsystem, return 0
        return 0.0

    return shannon_entropy(dist)


def total_colony_entropy(state: dict) -> float:
    """Sum of subsystem entropies weighted by SUBSYSTEM_WEIGHTS.

    Args:
        state: Colony state dictionary.

    Returns:
        Total weighted entropy across all 4 subsystems.
    """
    total = 0.0
    for subsystem, weight in SUBSYSTEM_WEIGHTS.items():
        total += weight * subsystem_entropy(state, subsystem)
    return total


def entropy_rate(states: List[dict]) -> float:
    """Compute entropy rate over state history.

    Args:
        states: List of colony state dictionaries (time-ordered).

    Returns:
        (H_final - H_initial) / len(states).
        Positive = accumulating (bad), Negative = shedding (good).
        Returns 0.0 if len < 2.
    """
    if len(states) < 2:
        return 0.0

    h_initial = total_colony_entropy(states[0])
    h_final = total_colony_entropy(states[-1])

    return (h_final - h_initial) / len(states)


# === DECISION CAPACITY FUNCTIONS (THE NOVEL CONTRIBUTION) ===

def decision_capacity(crew: int, expertise: dict, bandwidth: float, latency: float) -> float:
    """Compute internal decision-making capacity in bits/second.

    Args:
        crew: Number of crew members.
        expertise: Dict mapping skill_domain -> proficiency (0-1).
        bandwidth: Communication bandwidth (unused in internal calculation).
        latency: Communication latency (unused in internal calculation).

    Returns:
        Bits/second colony can process internally.
        Formula: crew × DECISION_BITS_PER_PERSON_PER_SEC × expertise_factor
        If crew <= 0, returns 0.0.
    """
    if crew <= 0:
        return 0.0

    # Expertise factor: mean of expertise values, or 1.0 if empty
    if expertise:
        expertise_factor = sum(expertise.values()) / len(expertise)
    else:
        expertise_factor = 1.0

    return crew * DECISION_BITS_PER_PERSON_PER_SEC * expertise_factor


def earth_input_rate(bandwidth: float, latency: float) -> float:
    """Compute bits/second usable from Earth.

    Args:
        bandwidth: Communication bandwidth in Mbps.
        latency: One-way light delay in seconds.

    Returns:
        Bits/second of usable decisions from Earth.
        Formula: (bandwidth × 1e6) × LATENCY_COST_FACTOR / max(1, latency)
        At max latency, severe penalty applied.
    """
    # Convert Mbps to bits/sec
    bits_per_sec = bandwidth * 1e6

    # Apply latency degradation
    effective_latency = max(1, latency)

    # Severe penalty at max latency (22 min = 1320 sec)
    latency_seconds_max = LIGHT_DELAY_MAX * 60
    if latency >= latency_seconds_max:
        # Severe degradation at conjunction
        penalty = 0.1
    else:
        penalty = 1.0

    return (bits_per_sec * LATENCY_COST_FACTOR * penalty) / effective_latency


def sovereignty_threshold(internal: float, external: float) -> bool:
    """Determine if colony has achieved decision sovereignty.

    THE KEY METRIC: When internal > external, colony can think for itself.

    Args:
        internal: Internal decision capacity (bits/sec).
        external: External input rate from Earth (bits/sec).

    Returns:
        True when internal > external (sovereign).
        False otherwise (dependent on Earth).
    """
    return internal > external


# === SURVIVAL BOUND FUNCTIONS ===

def survival_bound(crew: int, volume: float, power: float) -> float:
    """Compute maximum sustainable entropy before cascade failure.

    Args:
        crew: Number of crew members.
        volume: Habitable volume in cubic meters.
        power: Available power in kW.

    Returns:
        Upper bound on colony entropy.
        Formula: log₂(crew + 1) + log₂(volume + 1) + log₂(power + 1)
    """
    return (
        math.log2(crew + 1) +
        math.log2(volume + 1) +
        math.log2(power + 1)
    )


def entropy_status(rate: float, bound: float, current: float) -> str:
    """Classify colony entropy status.

    Args:
        rate: Entropy rate (from entropy_rate function).
        bound: Survival bound (from survival_bound function).
        current: Current total entropy.

    Returns:
        "stable" if rate <= 0 (shedding entropy).
        "accumulating" if rate > 0 and current < bound.
        "critical" if current >= bound.
    """
    if current >= bound:
        return "critical"
    if rate <= 0:
        return "stable"
    return "accumulating"


# === RECEIPT EMISSION ===

def emit_entropy_receipt(colony_id: str, state: dict, states: list) -> dict:
    """Emit complete entropy receipt with all metrics.

    Args:
        colony_id: Colony identifier.
        state: Current colony state.
        states: Historical state list for rate calculation.

    Returns:
        Complete entropy_receipt dict.

    Receipt schema includes:
        - All 4 subsystem entropies
        - Total entropy
        - Entropy rate
        - Decision capacity and earth input rate
        - Sovereignty threshold (THE KEY OUTPUT)
        - Survival bound and status
    """
    # Extract parameters from state
    crew = state.get("crew", 1)
    expertise = state.get("expertise", {})
    bandwidth = state.get("bandwidth", MARS_RELAY_MBPS)
    latency = state.get("latency_sec", LIGHT_DELAY_MIN * 60)
    volume = state.get("volume_m3", 100.0)
    power = state.get("power_kw", KILOPOWER_KW)

    # Compute all metrics
    h_atmosphere = subsystem_entropy(state, "atmosphere")
    h_thermal = subsystem_entropy(state, "thermal")
    h_resource = subsystem_entropy(state, "resource")
    h_decision = subsystem_entropy(state, "decision")
    h_total = total_colony_entropy(state)

    rate = entropy_rate(states) if states else 0.0
    dc = decision_capacity(crew, expertise, bandwidth, latency)
    ei = earth_input_rate(bandwidth, latency)
    sov = sovereignty_threshold(dc, ei)
    bound = survival_bound(crew, volume, power)
    status = entropy_status(rate, bound, h_total)

    # Build receipt data
    data = {
        "colony_id": colony_id,
        "H_atmosphere": h_atmosphere,
        "H_thermal": h_thermal,
        "H_resource": h_resource,
        "H_decision": h_decision,
        "H_total": h_total,
        "entropy_rate": rate,
        "decision_capacity_bps": dc,
        "earth_input_bps": ei,
        "sovereignty": sov,
        "survival_bound": bound,
        "status": status,
    }

    return emit_receipt("entropy", data)
