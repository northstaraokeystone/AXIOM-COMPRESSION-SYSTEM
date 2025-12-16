"""AXIOM-COLONY v3.1 Subsystems Module - Physics with confidence returns.

v3.1 KEY CHANGE: Every function returns {..., "confidence": float}
Low confidence → more decisions needed.
"""

import math
from functools import reduce

import numpy as np

from src.core import emit_receipt
from src.entropy import HUMAN_METABOLIC_W, KILOPOWER_KW

# Constants with confidence levels
STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴, confidence: 1.0
MARS_AMBIENT_K = 210  # K, confidence: 0.95
HUMAN_O2_KG_PER_DAY = 0.84  # confidence: 0.98
HUMAN_CO2_KG_PER_DAY = 1.0  # confidence: 0.98
HUMAN_WATER_L_PER_DAY = 2.5  # confidence: 0.95
HUMAN_KCAL_PER_DAY = 2000  # confidence: 0.95
SABATIER_EFFICIENCY_FLOOR = 0.70  # confidence: 0.50 (no Mars data)
MOXIE_POWER_W_PER_UNIT = 300  # confidence: 0.90
CROP_YIELD_KG_PER_M2_DAY = 0.01  # confidence: 0.60 (Biosphere 2)


# ─────────────────────────────────────────────────────────────────────────────
# THERMAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def solar_input(array_m2: float, flux: float = 590, efficiency: float = 0.2) -> dict:
    """Calculate solar input power.

    Returns: {watts, confidence: 0.90}
    """
    watts = array_m2 * flux * efficiency
    return {"watts": watts, "confidence": 0.90}


def nuclear_input(kilopower_units: int) -> dict:
    """Calculate nuclear power input.

    Returns: {watts, confidence: 0.95}
    """
    watts = kilopower_units * KILOPOWER_KW * 1000
    return {"watts": watts, "confidence": 0.95}


def metabolic_heat(crew: int) -> dict:
    """Calculate metabolic heat from crew.

    Returns: {watts, confidence: 0.98}
    """
    watts = crew * HUMAN_METABOLIC_W
    return {"watts": watts, "confidence": 0.98}


def radiator_capacity(area_m2: float, T_hab_K: float, T_ambient_K: float = MARS_AMBIENT_K) -> dict:
    """Calculate radiator heat rejection capacity.

    Returns: {watts, confidence: 0.85}
    """
    watts = area_m2 * STEFAN_BOLTZMANN * (T_hab_K**4 - T_ambient_K**4)
    return {"watts": watts, "confidence": 0.85}


def thermal_balance(Q_in_W: float, Q_out_W: float, mass_kg: float = 10000) -> dict:
    """Calculate thermal balance.

    Returns: {delta_T, T_hab_C, status, confidence}
    """
    # Specific heat capacity of habitat (mixed materials)
    c_p = 1000  # J/kg·K approximation
    net_Q = Q_in_W - Q_out_W
    delta_T = net_Q / (mass_kg * c_p) * 3600  # Per hour

    # Assume starting at 22°C
    T_hab_C = 22.0 + delta_T

    if T_hab_C < 0:
        status = "freezing"
    elif T_hab_C > 40:
        status = "overheating"
    elif T_hab_C < 15 or T_hab_C > 30:
        status = "warning"
    else:
        status = "nominal"

    # Confidence based on net heat flow
    conf = 0.90 if abs(net_Q) < 1000 else 0.80

    return {
        "delta_T": delta_T,
        "T_hab_C": T_hab_C,
        "status": status,
        "confidence": conf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ATMOSPHERE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def moxie_o2_production(units: int, power_W: float) -> dict:
    """Calculate MOXIE O2 production.

    Returns: {kg_per_day, confidence: 0.90}
    """
    # MOXIE produces 5.5 g/hr at 300W per unit
    hours_per_day = min(24, power_W / (MOXIE_POWER_W_PER_UNIT * units)) if units > 0 else 0
    kg_per_day = units * 5.5 * hours_per_day / 1000
    return {"kg_per_day": kg_per_day, "confidence": 0.90}


def sabatier_conversion(co2_kg: float, h2_kg: float, efficiency: float = 0.70) -> dict:
    """Calculate Sabatier reaction products.

    CO2 + 4H2 → CH4 + 2H2O

    Returns: {ch4_kg, h2o_kg, confidence: 0.50}
    """
    # Stoichiometry: 44g CO2 + 8g H2 → 16g CH4 + 36g H2O
    limiting = min(co2_kg / 44, h2_kg / 8)  # mol
    ch4_kg = limiting * 16 * efficiency
    h2o_kg = limiting * 36 * efficiency
    return {"ch4_kg": ch4_kg, "h2o_kg": h2o_kg, "confidence": 0.50}


def human_o2_consumption(crew: int) -> dict:
    """Calculate human O2 consumption.

    Returns: {kg_per_day, confidence: 0.98}
    """
    return {"kg_per_day": crew * HUMAN_O2_KG_PER_DAY, "confidence": 0.98}


def human_co2_production(crew: int) -> dict:
    """Calculate human CO2 production.

    Returns: {kg_per_day, confidence: 0.98}
    """
    return {"kg_per_day": crew * HUMAN_CO2_KG_PER_DAY, "confidence": 0.98}


def atmosphere_balance(o2_prod: float, o2_consume: float, co2_scrub: float, vol_m3: float) -> dict:
    """Calculate atmosphere balance.

    Returns: {O2_pct, CO2_ppm, status, confidence}
    """
    # Simplified model
    net_o2 = o2_prod - o2_consume
    # Standard atmosphere: 21% O2 at sea level, 400ppm CO2

    # O2 percentage change per day (simplified)
    O2_pct = 21.0 + (net_o2 / vol_m3) * 100

    # CO2 accumulation (simplified)
    CO2_ppm = 400 + max(0, (o2_consume - co2_scrub) * 1000)

    if O2_pct < 19.5:
        status = "hypoxic"
    elif O2_pct > 23.5:
        status = "hyperoxic"
    elif CO2_ppm > 5000:
        status = "toxic_co2"
    elif CO2_ppm > 2000:
        status = "elevated_co2"
    else:
        status = "nominal"

    return {
        "O2_pct": O2_pct,
        "CO2_ppm": CO2_ppm,
        "status": status,
        "confidence": 0.85,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def water_cycle(crew: int, recovery: float = 0.98) -> dict:
    """Calculate water cycle balance.

    Returns: {consumed, recovered, net_loss, confidence: 0.95}
    """
    consumed = crew * HUMAN_WATER_L_PER_DAY
    recovered = consumed * recovery
    net_loss = consumed - recovered
    return {
        "consumed": consumed,
        "recovered": recovered,
        "net_loss": net_loss,
        "confidence": 0.95,
    }


def food_requirement(crew: int) -> dict:
    """Calculate food requirements.

    Returns: {kcal_per_day, confidence: 0.95}
    """
    return {"kcal_per_day": crew * HUMAN_KCAL_PER_DAY, "confidence": 0.95}


def food_production(grow_area_m2: float) -> dict:
    """Calculate food production from agriculture.

    Returns: {kg_per_day, confidence: 0.60}
    """
    kg_per_day = grow_area_m2 * CROP_YIELD_KG_PER_M2_DAY
    return {"kg_per_day": kg_per_day, "confidence": 0.60}


def power_budget(solar_W: float, nuclear_W: float, consumption_W: float) -> dict:
    """Calculate power budget.

    Returns: {net_W, status, confidence}
    """
    total_gen = solar_W + nuclear_W
    net_W = total_gen - consumption_W

    if net_W < 0:
        status = "deficit"
    elif net_W < 1000:
        status = "tight"
    else:
        status = "surplus"

    # Confidence based on solar variability
    solar_fraction = solar_W / max(total_gen, 1)
    conf = 0.95 - 0.15 * solar_fraction  # More solar = less confident

    return {"net_W": net_W, "status": status, "confidence": conf}


def isru_closure(local_production: float, consumption: float) -> dict:
    """Calculate ISRU closure ratio.

    Returns: {ratio, confidence}
    """
    if consumption <= 0:
        return {"ratio": 1.0, "confidence": 0.50}
    ratio = local_production / consumption
    # Lower confidence for higher ratios (optimistic)
    conf = max(0.50, 0.90 - 0.4 * ratio)
    return {"ratio": min(ratio, 1.0), "confidence": conf}


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY FUNCTIONS (v3.1 NEW)
# ─────────────────────────────────────────────────────────────────────────────

def subsystem_confidence(outputs: list) -> float:
    """Geometric mean of confidences from subsystem outputs."""
    confidences = [o.get("confidence", 1.0) for o in outputs if isinstance(o, dict)]
    if not confidences:
        return 1.0
    return reduce(lambda x, y: x * y, confidences) ** (1 / len(confidences))


def uncertainty_to_decision_overhead(confidence: float) -> float:
    """Convert confidence to decision overhead.

    (1/conf - 1). 0.70 conf → 0.43 overhead
    """
    if confidence <= 0:
        return float('inf')
    return (1.0 / confidence) - 1.0
