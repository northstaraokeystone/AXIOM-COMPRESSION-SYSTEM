"""BUILD C4: Hard physics calculations for thermal, atmosphere, and resource management.

Pure functions with real physics constants. Each function documents its formula
and source. No magic numbers - all constants named and sourced.

Source: CLAUDEME §8, AXIOM_Colony_Build_Strategy_v2.md §2.5-2.7
"""
import math
from typing import Dict

import numpy as np

from src.core import emit_receipt, TENANT_ID
from src.entropy import (
    HUMAN_METABOLIC_W,
    MOXIE_O2_G_PER_HR,
    ISS_WATER_RECOVERY,
    SOLAR_FLUX_MAX,
    KILOPOWER_KW,
)


# === PHYSICS CONSTANTS ===

STEFAN_BOLTZMANN = 5.67e-8
"""Stefan-Boltzmann constant in W/(m²·K⁴). Physics fundamental."""

MARS_AMBIENT_K = 210.0
"""Mars average surface temperature in Kelvin. NASA."""

HAB_THERMAL_MASS_J_PER_K = 5.0e6
"""Habitat thermal mass in J/K. ~5MJ to change hab by 1°C. Engineering estimate."""

SOLAR_PANEL_EFFICIENCY = 0.20
"""Solar panel efficiency on Mars. 20% typical. Engineering."""

EQUIPMENT_HEAT_FRACTION = 0.70
"""Fraction of equipment power becoming waste heat. 70% typical. Engineering."""

RADIATOR_EMISSIVITY = 0.9
"""Radiator surface emissivity. High-emissivity coating. Engineering."""

# === HUMAN PHYSIOLOGY CONSTANTS ===

HUMAN_O2_KG_PER_DAY = 0.84
"""Human oxygen consumption in kg/person/day. NASA physiology."""

HUMAN_CO2_KG_PER_DAY = 1.0
"""Human CO2 production in kg/person/day. NASA physiology."""

HUMAN_WATER_L_PER_DAY = 3.0
"""Human water consumption in L/person/day. NASA physiology."""

FOOD_KCAL_PER_DAY = 2000.0
"""Daily food requirement in kcal/person. NASA standard."""

FOOD_KG_PER_KCAL = 0.0003
"""Dry food mass per kcal. ~0.6 kg/2000 kcal. Engineering estimate."""

# === CHEMISTRY CONSTANTS ===

CO2_MOLAR_MASS = 44.01
"""CO2 molar mass in g/mol. Chemistry."""

H2_MOLAR_MASS = 2.016
"""H2 molar mass in g/mol. Chemistry."""

CH4_MOLAR_MASS = 16.04
"""CH4 molar mass in g/mol. Chemistry."""

H2O_MOLAR_MASS = 18.015
"""H2O molar mass in g/mol. Chemistry."""

# === MOXIE POWER REQUIREMENT ===

MOXIE_POWER_W_PER_UNIT = 300.0
"""Power required per MOXIE unit in W. NASA reference."""


# =============================================================================
# THERMAL FUNCTIONS
# =============================================================================

def solar_input(array_m2: float, flux_W_m2: float,
                efficiency: float = SOLAR_PANEL_EFFICIENCY) -> float:
    """Calculate solar power input in watts.

    Formula: array_m2 × flux_W_m2 × efficiency

    Args:
        array_m2: Solar array area in square meters.
        flux_W_m2: Solar flux in W/m² (varies with dust, season).
        efficiency: Panel efficiency (default 0.20).

    Returns:
        Power in watts. 0 if any input is negative.
    """
    if array_m2 < 0 or flux_W_m2 < 0 or efficiency < 0:
        return 0.0
    return array_m2 * flux_W_m2 * efficiency


def nuclear_input(kilopower_units: int) -> float:
    """Calculate nuclear power input in watts.

    Formula: kilopower_units × KILOPOWER_KW × 1000

    Args:
        kilopower_units: Number of Kilopower reactor units.

    Returns:
        Power in watts.
    """
    if kilopower_units < 0:
        return 0.0
    return kilopower_units * KILOPOWER_KW * 1000


def metabolic_heat(crew: int) -> float:
    """Calculate metabolic heat output from crew in watts.

    Formula: crew × HUMAN_METABOLIC_W

    Args:
        crew: Number of crew members.

    Returns:
        Heat in watts.
    """
    if crew < 0:
        return 0.0
    return crew * HUMAN_METABOLIC_W


def equipment_heat(power_draw_W: float) -> float:
    """Calculate waste heat from equipment in watts.

    Formula: power_draw_W × EQUIPMENT_HEAT_FRACTION

    Args:
        power_draw_W: Total equipment power draw in watts.

    Returns:
        Waste heat in watts.
    """
    if power_draw_W < 0:
        return 0.0
    return power_draw_W * EQUIPMENT_HEAT_FRACTION


def radiator_capacity(area_m2: float, T_hab_C: float,
                      T_ambient_K: float = MARS_AMBIENT_K) -> float:
    """Calculate radiator heat rejection capacity using Stefan-Boltzmann law.

    Formula: area_m2 × ε × σ × (T_hab_K⁴ - T_ambient_K⁴)
    where ε = RADIATOR_EMISSIVITY, σ = STEFAN_BOLTZMANN

    Args:
        area_m2: Radiator area in square meters.
        T_hab_C: Habitat temperature in Celsius.
        T_ambient_K: Ambient temperature in Kelvin (default Mars 210K).

    Returns:
        Heat rejection capacity in watts. 0 if T_hab <= T_ambient.
    """
    if area_m2 < 0:
        return 0.0

    T_hab_K = T_hab_C + 273.15

    # Cannot radiate if habitat is colder than ambient
    if T_hab_K <= T_ambient_K:
        return 0.0

    # Stefan-Boltzmann law: Q = ε × A × σ × (T_hot⁴ - T_cold⁴)
    return (area_m2 * RADIATOR_EMISSIVITY * STEFAN_BOLTZMANN *
            (T_hab_K**4 - T_ambient_K**4))


def thermal_balance(Q_in_W: float, Q_out_W: float) -> Dict[str, float]:
    """Calculate thermal balance metrics.

    Formula: delta_T_per_hour = (Q_in - Q_out) × 3600 / HAB_THERMAL_MASS_J_PER_K

    Args:
        Q_in_W: Total heat input in watts.
        Q_out_W: Total heat rejection in watts.

    Returns:
        Dict with:
        - delta_T_per_hour: Temperature change rate in °C/hour
        - time_to_critical_hours: Hours until 0°C or 40°C (from 22°C nominal)
        - status: "nominal"|"stressed"|"critical"
    """
    # Calculate temperature rate of change
    # dT/dt = (Q_in - Q_out) / C_thermal
    # Convert to per-hour: multiply by 3600
    delta_T_per_hour = (Q_in_W - Q_out_W) * 3600 / HAB_THERMAL_MASS_J_PER_K

    # Time to critical (assuming starting from 22°C nominal)
    T_nominal = 22.0
    T_critical_low = 0.0
    T_critical_high = 40.0

    if abs(delta_T_per_hour) < 1e-10:
        # Essentially zero rate
        time_to_critical_hours = float('inf')
    elif delta_T_per_hour > 0:
        # Heating: time to reach 40°C
        time_to_critical_hours = (T_critical_high - T_nominal) / delta_T_per_hour
    else:
        # Cooling: time to reach 0°C
        time_to_critical_hours = (T_nominal - T_critical_low) / abs(delta_T_per_hour)

    # Ensure non-negative
    time_to_critical_hours = max(0, time_to_critical_hours)

    # Determine status based on rate
    abs_rate = abs(delta_T_per_hour)
    if abs_rate <= 0.5:
        status = "nominal"
    elif abs_rate <= 2.0:
        status = "stressed"
    else:
        status = "critical"

    return {
        "delta_T_per_hour": delta_T_per_hour,
        "time_to_critical_hours": time_to_critical_hours,
        "status": status,
    }


# =============================================================================
# ATMOSPHERE FUNCTIONS
# =============================================================================

def moxie_o2(units: int, power_available_W: float) -> float:
    """Calculate MOXIE O2 production in kg/day.

    Base rate: MOXIE_O2_G_PER_HR × 24 / 1000 per unit = ~0.132 kg/day/unit
    Power-limited: if power < 300W per unit, output scales proportionally.

    Args:
        units: Number of MOXIE units.
        power_available_W: Total power available for MOXIE.

    Returns:
        O2 production in kg/day.
    """
    if units <= 0 or power_available_W <= 0:
        return 0.0

    # Base rate per unit: g/hr → kg/day
    base_rate_kg_per_day = MOXIE_O2_G_PER_HR * 24 / 1000  # ~0.132 kg/day

    # Power requirement per unit
    power_per_unit = MOXIE_POWER_W_PER_UNIT
    total_power_needed = units * power_per_unit

    # Power limiting factor
    if power_available_W >= total_power_needed:
        power_factor = 1.0
    else:
        power_factor = power_available_W / total_power_needed

    return units * base_rate_kg_per_day * power_factor


def sabatier(co2_kg: float, h2_kg: float, efficiency: float = 0.85) -> Dict[str, float]:
    """Calculate Sabatier reaction products.

    Stoichiometry: CO2 + 4H2 → CH4 + 2H2O
    Molar: 1 mol CO2 (44g) + 4 mol H2 (8g) → 1 mol CH4 (16g) + 2 mol H2O (36g)

    Args:
        co2_kg: CO2 input in kg.
        h2_kg: H2 input in kg.
        efficiency: Reaction efficiency (default 0.85).

    Returns:
        Dict with ch4_kg and h2o_kg produced.
    """
    if co2_kg <= 0 or h2_kg <= 0:
        return {"ch4_kg": 0.0, "h2o_kg": 0.0}

    # Convert kg to grams
    co2_g = co2_kg * 1000
    h2_g = h2_kg * 1000

    # Calculate moles
    co2_mol = co2_g / CO2_MOLAR_MASS
    h2_mol = h2_g / H2_MOLAR_MASS

    # Stoichiometry: 1 CO2 : 4 H2
    # Determine limiting reagent
    co2_limited_h2_mol = co2_mol * 4  # H2 needed if CO2 is limiting
    h2_limited_co2_mol = h2_mol / 4   # CO2 reacted if H2 is limiting

    if h2_mol >= co2_limited_h2_mol:
        # CO2 is limiting
        reacting_co2_mol = co2_mol
    else:
        # H2 is limiting
        reacting_co2_mol = h2_limited_co2_mol

    # Products: 1 mol CO2 → 1 mol CH4 + 2 mol H2O
    ch4_mol = reacting_co2_mol * efficiency
    h2o_mol = reacting_co2_mol * 2 * efficiency

    # Convert to kg
    ch4_kg = ch4_mol * CH4_MOLAR_MASS / 1000
    h2o_kg = h2o_mol * H2O_MOLAR_MASS / 1000

    return {"ch4_kg": ch4_kg, "h2o_kg": h2o_kg}


def human_o2(crew: int) -> float:
    """Calculate crew O2 consumption in kg/day.

    Formula: crew × HUMAN_O2_KG_PER_DAY

    Args:
        crew: Number of crew members.

    Returns:
        O2 consumption in kg/day.
    """
    if crew <= 0:
        return 0.0
    return crew * HUMAN_O2_KG_PER_DAY


def human_co2(crew: int) -> float:
    """Calculate crew CO2 production in kg/day.

    Formula: crew × HUMAN_CO2_KG_PER_DAY

    Args:
        crew: Number of crew members.

    Returns:
        CO2 production in kg/day.
    """
    if crew <= 0:
        return 0.0
    return crew * HUMAN_CO2_KG_PER_DAY


def atmosphere_balance(o2_production_kg: float, o2_consumption_kg: float,
                       co2_scrub_kg: float) -> Dict[str, float]:
    """Calculate atmosphere balance metrics.

    Args:
        o2_production_kg: O2 produced per day in kg.
        o2_consumption_kg: O2 consumed per day in kg.
        co2_scrub_kg: CO2 scrubbed per day in kg.

    Returns:
        Dict with:
        - net_o2_kg: Net O2 change per day
        - o2_days_reserve: Days of O2 reserve at current rate (inf if positive)
        - co2_buildup_rate: CO2 accumulation rate in kg/day
        - status: "nominal"|"stressed"|"critical"
    """
    # Net O2 balance
    net_o2_kg = o2_production_kg - o2_consumption_kg

    # O2 reserve days (assuming 100 kg baseline reserve)
    O2_RESERVE_BASELINE = 100.0
    if net_o2_kg >= 0:
        o2_days_reserve = float('inf')
    else:
        o2_days_reserve = O2_RESERVE_BASELINE / max(0.001, abs(net_o2_kg))

    # CO2 buildup rate
    # CO2 produced = O2 consumed × (CO2/O2 ratio)
    co2_ratio = HUMAN_CO2_KG_PER_DAY / HUMAN_O2_KG_PER_DAY
    co2_produced = o2_consumption_kg * co2_ratio
    co2_buildup_rate = co2_produced - co2_scrub_kg

    # Determine status
    if net_o2_kg >= 0 and co2_buildup_rate <= 0:
        status = "nominal"
    elif net_o2_kg < 0 and o2_days_reserve > 30:
        status = "stressed"
    elif o2_days_reserve <= 30 or co2_buildup_rate > 1.0:
        status = "critical"
    else:
        status = "stressed"

    return {
        "net_o2_kg": net_o2_kg,
        "o2_days_reserve": o2_days_reserve,
        "co2_buildup_rate": co2_buildup_rate,
        "status": status,
    }


# =============================================================================
# RESOURCE FUNCTIONS
# =============================================================================

def water_cycle(crew: int, recovery_rate: float = ISS_WATER_RECOVERY) -> Dict[str, float]:
    """Calculate water cycle metrics.

    Args:
        crew: Number of crew members.
        recovery_rate: Water recovery rate (default ISS 0.98).

    Returns:
        Dict with consumed_L, recovered_L, net_loss_L.
    """
    if crew <= 0:
        return {"consumed_L": 0.0, "recovered_L": 0.0, "net_loss_L": 0.0}

    consumed_L = crew * HUMAN_WATER_L_PER_DAY
    recovered_L = consumed_L * recovery_rate
    net_loss_L = consumed_L - recovered_L

    return {
        "consumed_L": consumed_L,
        "recovered_L": recovered_L,
        "net_loss_L": net_loss_L,
    }


def food_requirement(crew: int) -> float:
    """Calculate daily food requirement in kcal.

    Formula: crew × FOOD_KCAL_PER_DAY

    Args:
        crew: Number of crew members.

    Returns:
        Daily food requirement in kcal.
    """
    if crew <= 0:
        return 0.0
    return crew * FOOD_KCAL_PER_DAY


def power_budget(solar_W: float, nuclear_W: float,
                 consumption_W: float) -> Dict[str, float]:
    """Calculate power budget metrics.

    Args:
        solar_W: Solar power generation in watts.
        nuclear_W: Nuclear power generation in watts.
        consumption_W: Power consumption in watts.

    Returns:
        Dict with:
        - total_generation_W: Total power generation
        - net_power_W: Generation minus consumption
        - reserve_margin_pct: Percentage reserve margin
        - status: "nominal"|"stressed"|"critical"
    """
    total_generation_W = max(0, solar_W) + max(0, nuclear_W)
    net_power_W = total_generation_W - consumption_W

    # Reserve margin as percentage of generation
    if total_generation_W > 0:
        reserve_margin_pct = 100 * net_power_W / total_generation_W
    else:
        reserve_margin_pct = -100.0 if consumption_W > 0 else 0.0

    # Determine status
    if reserve_margin_pct > 20:
        status = "nominal"
    elif reserve_margin_pct >= 5:
        status = "stressed"
    else:
        status = "critical"

    return {
        "total_generation_W": total_generation_W,
        "net_power_W": net_power_W,
        "reserve_margin_pct": reserve_margin_pct,
        "status": status,
    }


def isru_closure(local_production: Dict[str, float],
                 consumption: Dict[str, float]) -> float:
    """Calculate ISRU closure ratio.

    Args:
        local_production: Dict of resource -> amount produced locally.
        consumption: Dict of resource -> amount consumed.

    Returns:
        Closure ratio 0-1. 1.0 = fully closed loop.
    """
    production_sum = sum(local_production.values())
    consumption_sum = sum(consumption.values())

    if consumption_sum <= 0.001:
        return 1.0

    return min(1.0, production_sum / consumption_sum)


# =============================================================================
# RECEIPT FUNCTION
# =============================================================================

def emit_subsystem_receipt(subsystem: str, metrics: Dict, status: str) -> Dict:
    """Emit a subsystem receipt.

    Convenience wrapper around emit_receipt for subsystem metrics.

    Args:
        subsystem: Subsystem name ("thermal"|"atmosphere"|"resource").
        metrics: Dict of subsystem-specific metrics.
        status: Status string ("nominal"|"stressed"|"critical").

    Returns:
        Complete receipt dict.
    """
    data = {
        "subsystem": subsystem,
        "metrics": metrics,
        "status": status,
    }
    return emit_receipt("subsystem", data)
