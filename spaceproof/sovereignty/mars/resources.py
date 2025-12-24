"""Resource Margins Calculator.

Purpose: Calculate ISRU closure ratios and reserve margins.

THE PHYSICS:
    Self-sufficiency requires local production >= consumption.
    Closure ratio 1.0 = fully closed loop, no resupply needed.
    Buffer of 90+ days critical (synodic period = 780 days).
    Starship payload = 125t per flight, must optimize cargo mix.
"""

from typing import Any

from spaceproof.core import emit_receipt

from .constants import (
    BUFFER_DAYS_MINIMUM,
    BUFFER_DAYS_NOMINAL,
    HUMAN_FOOD_KG_PER_DAY,
    HUMAN_KCAL_PER_DAY,
    HUMAN_O2_KG_PER_DAY,
    HUMAN_WATER_KG_PER_DAY,
    ISRU_CLOSURE_TARGET,
    MARS_SYNODIC_PERIOD_DAYS,
    STARSHIP_PAYLOAD_KG,
    TENANT_ID,
)


def calculate_isru_closure(
    production: dict,
    consumption: dict,
) -> float:
    """Calculate ISRU closure ratio.

    1.0 = fully closed loop, no resupply needed.
    Categories: O2, H2O, food, fuel, spare parts.

    Args:
        production: Dict of resource production rates (kg/day)
        consumption: Dict of resource consumption rates (kg/day)

    Returns:
        float: Closure ratio (0-1). 1.0 = fully self-sufficient.
    """
    if not consumption:
        return 1.0

    total_production = 0.0
    total_consumption = 0.0

    # Critical resources get higher weights
    weights = {
        "o2": 3.0,  # Life critical
        "h2o": 3.0,  # Life critical
        "food": 3.0,  # Life critical
        "fuel": 1.0,  # Important but not immediate
        "spare_parts": 0.5,  # Can defer some maintenance
        "other": 0.5,
    }

    for resource, rate in consumption.items():
        weight = weights.get(resource, 0.5)
        prod_rate = production.get(resource, 0.0)

        total_consumption += rate * weight
        total_production += min(prod_rate, rate) * weight  # Can't exceed consumption

    if total_consumption <= 0:
        return 1.0

    return total_production / total_consumption


def calculate_reserve_buffer(
    reserves: dict,
    consumption_per_day: dict,
) -> dict:
    """Calculate buffer days per resource.

    Target: >= 90 days for critical resources.

    Args:
        reserves: Dict of resource reserves (kg)
        consumption_per_day: Dict of consumption rates (kg/day)

    Returns:
        dict: Buffer days per resource and overall status.
    """
    buffer_days = {}
    critical_met = True

    critical_resources = ["o2", "h2o", "food"]

    for resource, reserve in reserves.items():
        rate = consumption_per_day.get(resource, 0.0)
        if rate > 0:
            days = reserve / rate
        else:
            days = float("inf")

        buffer_days[resource] = days

        if resource in critical_resources and days < BUFFER_DAYS_MINIMUM:
            critical_met = False

    # Overall buffer: minimum of critical resources
    critical_buffers = [buffer_days.get(r, 0) for r in critical_resources if r in buffer_days]
    min_critical_buffer = min(critical_buffers) if critical_buffers else 0

    return {
        "buffer_days": buffer_days,
        "min_critical_buffer_days": min_critical_buffer,
        "critical_buffer_met": critical_met,
        "target_days": BUFFER_DAYS_MINIMUM,
    }


def calculate_resupply_cadence(
    closure_ratio: float,
    buffer_days: int,
    synodic_period: int = MARS_SYNODIC_PERIOD_DAYS,
) -> dict:
    """Calculate resupply requirements.

    If closure < 1.0 and buffer < synodic_period, colony is dependent.

    Args:
        closure_ratio: ISRU closure ratio (0-1)
        buffer_days: Current buffer days
        synodic_period: Days between launch windows (780)

    Returns:
        dict: Resupply requirements and risk level.
    """
    # Deficit rate: how much we're short per day (normalized)
    deficit_rate = 1.0 - closure_ratio

    if deficit_rate <= 0:
        # Fully closed loop
        return {
            "resupply_required": False,
            "flights_per_window": 0,
            "risk_level": "NONE",
            "margin_days": float("inf"),
        }

    # Days until reserves exhausted at current deficit
    if deficit_rate > 0:
        exhaustion_days = buffer_days / deficit_rate
    else:
        exhaustion_days = float("inf")

    # Compare to synodic period
    margin_days = exhaustion_days - synodic_period

    if margin_days >= synodic_period:
        risk_level = "LOW"
        flights_needed = 1
    elif margin_days >= BUFFER_DAYS_MINIMUM:
        risk_level = "MEDIUM"
        flights_needed = 2
    elif margin_days >= 0:
        risk_level = "HIGH"
        flights_needed = 3
    else:
        risk_level = "CRITICAL"
        flights_needed = 4  # Emergency resupply

    return {
        "resupply_required": True,
        "flights_per_window": flights_needed,
        "risk_level": risk_level,
        "margin_days": margin_days,
        "exhaustion_days": exhaustion_days,
    }


def calculate_starship_manifest(
    crew: int,
    duration_days: int,
    closure_ratio: float,
    current_reserves: dict | None = None,
) -> dict:
    """Calculate required Starship flights and cargo manifest.

    Payload: 125t per ship. Optimizes cargo mix for maximum closure improvement.

    Args:
        crew: Number of crew members
        duration_days: Mission duration in days
        closure_ratio: Current ISRU closure ratio
        current_reserves: Current reserve levels (optional)

    Returns:
        dict: Manifest with flights needed and cargo breakdown.
    """
    if current_reserves is None:
        current_reserves = {}

    # Calculate total consumption for duration
    consumption = {
        "o2": crew * HUMAN_O2_KG_PER_DAY * duration_days,
        "h2o": crew * HUMAN_WATER_KG_PER_DAY * duration_days,
        "food": crew * HUMAN_FOOD_KG_PER_DAY * duration_days,
    }

    # What's not produced locally must be supplied
    deficit_fraction = 1.0 - closure_ratio

    required_cargo = {}
    total_required_kg = 0.0

    for resource, amount in consumption.items():
        needed = amount * deficit_fraction
        reserve = current_reserves.get(resource, 0.0)
        additional = max(needed - reserve, 0.0)
        required_cargo[resource] = additional
        total_required_kg += additional

    # Add buffer (1.5x for safety margin)
    total_with_buffer = total_required_kg * 1.5

    # Add fixed overhead: spare parts, equipment, etc.
    overhead_kg = crew * 500  # 500kg per crew for spares/equipment

    total_cargo_kg = total_with_buffer + overhead_kg

    # Calculate flights
    flights = max(1, int(total_cargo_kg / STARSHIP_PAYLOAD_KG + 0.99))

    # Cargo breakdown per flight
    cargo_per_flight = total_cargo_kg / flights

    return {
        "flights_required": flights,
        "total_cargo_kg": total_cargo_kg,
        "cargo_per_flight_kg": cargo_per_flight,
        "cargo_breakdown": required_cargo,
        "overhead_kg": overhead_kg,
        "payload_capacity_kg": STARSHIP_PAYLOAD_KG,
        "utilization_pct": (cargo_per_flight / STARSHIP_PAYLOAD_KG) * 100,
    }


def identify_binding_resource(
    closure: dict,
) -> str:
    """Identify which resource is the binding constraint.

    Returns which resource has lowest closure ratio.
    This determines colony vulnerability.

    Args:
        closure: Dict of closure ratios per resource

    Returns:
        str: Name of binding (lowest closure) resource.
    """
    if not closure:
        return "none"

    binding = min(closure.items(), key=lambda x: x[1])
    return binding[0]


def calculate_resource_score(
    crew: int,
    production: dict,
    consumption: dict,
    reserves: dict,
) -> dict:
    """Calculate comprehensive resource score.

    Args:
        crew: Crew count
        production: Production rates
        consumption: Consumption rates
        reserves: Current reserves

    Returns:
        dict: Resource metrics including closure and buffer status.
    """
    closure_ratio = calculate_isru_closure(production, consumption)
    buffer_status = calculate_reserve_buffer(reserves, consumption)
    resupply = calculate_resupply_cadence(closure_ratio, int(buffer_status["min_critical_buffer_days"]))

    # Per-resource closure ratios
    resource_closures = {}
    for resource in consumption:
        prod = production.get(resource, 0.0)
        cons = consumption.get(resource, 0.01)  # Avoid div by zero
        resource_closures[resource] = min(prod / cons, 1.0)

    binding = identify_binding_resource(resource_closures)

    return {
        "closure_ratio": closure_ratio,
        "buffer_status": buffer_status,
        "resupply": resupply,
        "resource_closures": resource_closures,
        "binding_resource": binding,
    }


def emit_resource_balance_receipt(
    crew: int,
    metrics: dict,
) -> dict:
    """Emit resource balance receipt.

    Args:
        crew: Crew count
        metrics: Resource metrics

    Returns:
        dict: Emitted receipt.
    """
    return emit_receipt(
        "resource_balance",
        {
            "tenant_id": TENANT_ID,
            "crew_count": crew,
            "closure_ratio": metrics["closure_ratio"],
            "buffer_days": metrics["buffer_status"]["min_critical_buffer_days"],
            "buffer_met": metrics["buffer_status"]["critical_buffer_met"],
            "binding_resource": metrics["binding_resource"],
            "resupply_required": metrics["resupply"]["resupply_required"],
            "risk_level": metrics["resupply"]["risk_level"],
        },
    )
