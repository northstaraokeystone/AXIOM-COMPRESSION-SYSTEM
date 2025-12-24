"""Life Support Entropy Model.

Purpose: Model life support entropy generation and export capacity.

THE PHYSICS:
    A Mars colony is an INFORMATION SYSTEM fighting thermal death.
    Entropy generation rate must be less than entropy export capacity.
    Positive net entropy = accumulating disorder = approaching failure.
    Negative net entropy = shedding disorder = stable system.

Sources:
    - NASA ECLSS 2019: ISS reliability data (actual MTBF 1752h, not design 10000h)
    - NASA ECLSS 2023: O2 87.5%, H2O 98% closure ratios
    - Perseverance MOXIE: 5.5 g/hr O2 production (measured 2021-2025)
"""

import math
from typing import Any

from spaceproof.core import emit_receipt

from .constants import (
    HAB_TARGET_TEMP_C,
    HAB_TEMP_MAX_C,
    HAB_TEMP_MIN_C,
    HUMAN_METABOLIC_HEAT_W,
    HUMAN_O2_KG_PER_DAY,
    HUMAN_WATER_KG_PER_DAY,
    ISS_ECLSS_MTBF_HOURS,
    ISS_H2O_RECOVERY_RATIO,
    ISS_O2_CLOSURE_RATIO,
    MARS_AMBIENT_TEMP_K,
    MOXIE_O2_G_PER_HOUR,
    O2_PARTIAL_PRESSURE_MAX_KPA,
    O2_PARTIAL_PRESSURE_MIN_KPA,
    STEFAN_BOLTZMANN_W_M2_K4,
    TENANT_ID,
)


def calculate_o2_balance(
    crew: int,
    moxie_units: int = 1,
    eclss_closure: float = ISS_O2_CLOSURE_RATIO,
    power_available_w: float = 10000.0,
) -> dict:
    """Calculate O2 balance for colony.

    Args:
        crew: Number of crew members
        moxie_units: Number of MOXIE units
        eclss_closure: ECLSS O2 closure ratio (default ISS measured)
        power_available_w: Available power for MOXIE (each needs 300W)

    Returns:
        dict: O2 balance with production, consumption, net, reserve days.
    """
    # Consumption: crew metabolic
    consumption_kg_day = crew * HUMAN_O2_KG_PER_DAY

    # Production: MOXIE + ECLSS recycling
    moxie_production_kg_day = moxie_units * MOXIE_O2_G_PER_HOUR * 24 / 1000  # g/hr -> kg/day

    # ECLSS recycling recovers a fraction of exhaled CO2 back to O2
    # CO2 -> O2 via Sabatier + electrolysis chain
    recycled_kg_day = consumption_kg_day * eclss_closure

    total_production_kg_day = moxie_production_kg_day + recycled_kg_day
    net_balance_kg_day = total_production_kg_day - consumption_kg_day

    # Reserve days at current net rate (if positive, growing reserve)
    # Assume 30kg initial reserve per person
    initial_reserve_kg = crew * 30
    if net_balance_kg_day > 0:
        reserve_days = float("inf")  # Growing
    elif net_balance_kg_day < 0:
        reserve_days = initial_reserve_kg / abs(net_balance_kg_day)
    else:
        reserve_days = float("inf")  # Balanced

    return {
        "production_kg_day": total_production_kg_day,
        "moxie_production_kg_day": moxie_production_kg_day,
        "recycled_kg_day": recycled_kg_day,
        "consumption_kg_day": consumption_kg_day,
        "net_balance_kg_day": net_balance_kg_day,
        "reserve_days": reserve_days,
        "closure_ratio": total_production_kg_day / consumption_kg_day if consumption_kg_day > 0 else 0.0,
    }


def calculate_h2o_balance(
    crew: int,
    recovery_ratio: float = ISS_H2O_RECOVERY_RATIO,
    isru_production_kg_day: float = 0.0,
) -> dict:
    """Calculate water balance for colony.

    Args:
        crew: Number of crew members
        recovery_ratio: H2O recovery ratio (default ISS measured 98%)
        isru_production_kg_day: ISRU water extraction from regolith

    Returns:
        dict: H2O balance with recovery, consumption, net, reserve days.
    """
    # Consumption: drinking + hygiene
    consumption_kg_day = crew * HUMAN_WATER_KG_PER_DAY

    # Recovery: wastewater + humidity
    recovered_kg_day = consumption_kg_day * recovery_ratio

    # Total available
    total_production_kg_day = recovered_kg_day + isru_production_kg_day
    net_balance_kg_day = total_production_kg_day - consumption_kg_day

    # Reserve calculation
    initial_reserve_kg = crew * 100  # 100kg per person initial
    if net_balance_kg_day >= 0:
        reserve_days = float("inf")
    else:
        reserve_days = initial_reserve_kg / abs(net_balance_kg_day)

    return {
        "production_kg_day": total_production_kg_day,
        "recovered_kg_day": recovered_kg_day,
        "isru_kg_day": isru_production_kg_day,
        "consumption_kg_day": consumption_kg_day,
        "net_balance_kg_day": net_balance_kg_day,
        "reserve_days": reserve_days,
        "closure_ratio": total_production_kg_day / consumption_kg_day if consumption_kg_day > 0 else 0.0,
    }


def calculate_thermal_entropy(
    crew: int,
    equipment_power_w: float,
    radiator_area_m2: float,
    t_hab_c: float = HAB_TARGET_TEMP_C,
) -> dict:
    """Calculate thermal entropy balance.

    Positive net entropy = accumulating heat (bad).
    Negative net entropy = shedding heat (good).

    Args:
        crew: Number of crew members
        equipment_power_w: Total equipment power (becomes heat)
        radiator_area_m2: Radiator surface area
        t_hab_c: Habitat temperature in Celsius

    Returns:
        dict: Thermal entropy with generation, export, net rates.
    """
    t_hab_k = t_hab_c + 273.15

    # Entropy generation: metabolic + equipment
    metabolic_heat_w = crew * HUMAN_METABOLIC_HEAT_W
    total_heat_w = metabolic_heat_w + equipment_power_w

    # Entropy generation rate: dS/dt = Q/T (Joules/sec/Kelvin = Watts/Kelvin)
    entropy_generation_w_k = total_heat_w / t_hab_k

    # Entropy export via radiation: Q_rad = epsilon * sigma * A * (T_hab^4 - T_ambient^4)
    epsilon = 0.9  # Radiator emissivity
    q_rad = epsilon * STEFAN_BOLTZMANN_W_M2_K4 * radiator_area_m2 * (t_hab_k**4 - MARS_AMBIENT_TEMP_K**4)

    # Entropy export rate
    entropy_export_w_k = q_rad / t_hab_k if t_hab_k > 0 else 0.0

    net_entropy_w_k = entropy_generation_w_k - entropy_export_w_k

    return {
        "entropy_generation_w_k": entropy_generation_w_k,
        "entropy_export_w_k": entropy_export_w_k,
        "net_entropy_w_k": net_entropy_w_k,
        "metabolic_heat_w": metabolic_heat_w,
        "equipment_heat_w": equipment_power_w,
        "radiation_w": q_rad,
        "stable": net_entropy_w_k <= 0,  # Stable if exporting >= generating
    }


def calculate_eclss_reliability(
    mtbf_hours: float = ISS_ECLSS_MTBF_HOURS,
    redundancy_factor: float = 1.0,
    repair_capacity: float = 0.8,
    mission_duration_days: int = 500,
) -> float:
    """Calculate probability of ECLSS surviving mission without critical failure.

    Uses ISS actual MTBF (1752h) not design (10000h).
    Redundancy and repair capacity improve reliability.

    Args:
        mtbf_hours: Mean time between failures (default ISS actual)
        redundancy_factor: Redundancy level (1.0 = single, 2.0 = dual)
        repair_capacity: Probability of successful repair (0-1)
        mission_duration_days: Mission duration in days

    Returns:
        float: Reliability (0-1) - probability of no critical failure.
    """
    mission_hours = mission_duration_days * 24

    # Base failure rate (failures per hour)
    lambda_base = 1.0 / mtbf_hours

    # Effective MTBF with redundancy (series-parallel reliability)
    # For n-redundant system: effective_mtbf ≈ mtbf * (1 + 1/2 + ... + 1/n)
    harmonic_sum = sum(1.0 / i for i in range(1, int(redundancy_factor) + 1))
    effective_mtbf = mtbf_hours * harmonic_sum

    # Failure probability without repair
    p_fail_no_repair = 1.0 - math.exp(-mission_hours / effective_mtbf)

    # With repair capacity, we can recover from some failures
    # P(success) = P(no failure) + P(failure) * P(repair)
    p_success = (1 - p_fail_no_repair) + p_fail_no_repair * repair_capacity

    return min(max(p_success, 0.0), 1.0)


def calculate_life_support_entropy_rate(crew: int, eclss_config: dict) -> float:
    """Calculate Shannon entropy of life support state distribution.

    Measures system uncertainty/chaos. Higher = closer to failure.

    Args:
        crew: Number of crew members
        eclss_config: ECLSS configuration dict with closure ratios, etc.

    Returns:
        float: Entropy rate (negative = stable, positive = unstable).
    """
    # Get closure ratios (how closed the loop is)
    o2_closure = eclss_config.get("o2_closure", ISS_O2_CLOSURE_RATIO)
    h2o_closure = eclss_config.get("h2o_closure", ISS_H2O_RECOVERY_RATIO)

    # Get reliability factor
    mtbf = eclss_config.get("mtbf_hours", ISS_ECLSS_MTBF_HOURS)
    redundancy = eclss_config.get("redundancy_factor", 1.0)

    # Calculate component "stability" probabilities
    # Higher closure = more stable
    # Higher reliability = more stable

    reliability = calculate_eclss_reliability(mtbf_hours=mtbf, redundancy_factor=redundancy, mission_duration_days=500)

    # Combined stability score (geometric mean of factors)
    # All factors should be 0-1 where 1 = perfectly stable
    stability_factors = [o2_closure, h2o_closure, reliability]

    # Geometric mean
    product = 1.0
    for f in stability_factors:
        product *= max(f, 0.01)  # Avoid zero

    geometric_mean = product ** (1.0 / len(stability_factors))

    # Entropy rate: transform to negative for stable, positive for unstable
    # -1 = very stable, +1 = very unstable
    entropy_rate = 1.0 - 2.0 * geometric_mean

    return entropy_rate


def emit_life_support_balance_receipt(
    crew: int,
    o2_balance: dict,
    h2o_balance: dict,
    thermal: dict,
    reliability: float,
    entropy_rate: float,
) -> dict:
    """Emit life support balance receipt.

    Args:
        crew: Crew count
        o2_balance: O2 balance dict
        h2o_balance: H2O balance dict
        thermal: Thermal entropy dict
        reliability: ECLSS reliability
        entropy_rate: Life support entropy rate

    Returns:
        dict: Emitted receipt.
    """
    return emit_receipt(
        "life_support_balance",
        {
            "tenant_id": TENANT_ID,
            "crew_count": crew,
            "o2_closure_ratio": o2_balance["closure_ratio"],
            "o2_net_kg_day": o2_balance["net_balance_kg_day"],
            "o2_reserve_days": o2_balance["reserve_days"] if o2_balance["reserve_days"] != float("inf") else 9999,
            "h2o_closure_ratio": h2o_balance["closure_ratio"],
            "h2o_net_kg_day": h2o_balance["net_balance_kg_day"],
            "h2o_reserve_days": h2o_balance["reserve_days"] if h2o_balance["reserve_days"] != float("inf") else 9999,
            "thermal_stable": thermal["stable"],
            "net_entropy_w_k": thermal["net_entropy_w_k"],
            "eclss_reliability": reliability,
            "entropy_rate": entropy_rate,
        },
    )
