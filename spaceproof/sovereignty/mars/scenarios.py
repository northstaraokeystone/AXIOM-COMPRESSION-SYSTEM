"""Mars Failure Scenarios.

Purpose: Define failure scenarios for Monte Carlo validation.

THE PHYSICS:
    - Probabilities from ISS data (where available)
    - Conservative estimates for Mars (no data available)
    - Cascade effects modeled (ECLSS failure -> O2 crisis -> emergency decisions)
    - Recovery actions have success probabilities based on crew skill
"""

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """Failure scenario definition."""

    name: str
    probability: float  # Annual occurrence rate
    trigger_day: int | str  # Day to trigger, or "random"
    duration_days: int  # How long the scenario lasts
    effects: dict = field(default_factory=dict)  # State changes during scenario
    recovery_actions: list = field(default_factory=list)  # Available crew responses
    description: str = ""


# === SCENARIO DEFINITIONS ===

DUST_STORM_GLOBAL = Scenario(
    name="DUST_STORM_GLOBAL",
    probability=0.10,  # Once per decade average
    trigger_day="random",
    duration_days=90,
    effects={
        "solar_flux_multiplier": 0.01,  # 1% of normal
        "visibility_km": 0.1,
        "eva_allowed": False,
    },
    recovery_actions=[
        {"action": "switch_to_nuclear", "success_probability": 0.95},
        {"action": "reduce_power_consumption", "success_probability": 0.99},
    ],
    description="Global dust storm reducing solar flux to 1% for 90 days",
)

DUST_STORM_REGIONAL = Scenario(
    name="DUST_STORM_REGIONAL",
    probability=0.50,  # Common occurrence
    trigger_day="random",
    duration_days=30,
    effects={
        "solar_flux_multiplier": 0.20,  # 20% of normal
        "visibility_km": 1.0,
        "eva_allowed": True,  # With restrictions
    },
    recovery_actions=[
        {"action": "increase_battery_usage", "success_probability": 0.98},
    ],
    description="Regional dust storm reducing solar flux to 20% for 30 days",
)

HAB_BREACH_SMALL = Scenario(
    name="HAB_BREACH_SMALL",
    probability=0.05,  # Rare but serious
    trigger_day="random",
    duration_days=2,  # 1-3 days to repair
    effects={
        "pressure_loss_rate_kpa_hour": 0.1,
        "emergency_mode": True,
    },
    recovery_actions=[
        {"action": "seal_with_patch", "success_probability": 0.90},
        {"action": "isolate_section", "success_probability": 0.95},
    ],
    description="Small puncture causing 0.1 kPa/hour pressure loss",
)

HAB_BREACH_LARGE = Scenario(
    name="HAB_BREACH_LARGE",
    probability=0.01,  # Very rare but catastrophic
    trigger_day="random",
    duration_days=0,  # Immediate crisis
    effects={
        "pressure_loss_rate_kpa_hour": 10.0,
        "emergency_mode": True,
        "evacuation_required": True,
    },
    recovery_actions=[
        {"action": "emergency_seal", "success_probability": 0.50},
        {"action": "evacuate_to_backup_hab", "success_probability": 0.80},
    ],
    description="Large breach causing rapid pressure loss, immediate action required",
)

ECLSS_O2_FAILURE = Scenario(
    name="ECLSS_O2_FAILURE",
    probability=0.30,  # ISS has ~3 critical anomalies/year
    trigger_day="random",
    duration_days=2,  # 24-72 hours to repair
    effects={
        "o2_production": 0.0,
        "emergency_mode": True,
    },
    recovery_actions=[
        {"action": "switch_to_backup", "success_probability": 0.85},
        {"action": "repair_primary", "success_probability": 0.70},
        {"action": "use_o2_reserves", "success_probability": 0.99},
    ],
    description="Primary O2 system offline for 24-72 hours",
)

ECLSS_H2O_FAILURE = Scenario(
    name="ECLSS_H2O_FAILURE",
    probability=0.20,
    trigger_day="random",
    duration_days=4,  # 48-120 hours to repair
    effects={
        "h2o_recovery": 0.0,
        "emergency_mode": True,
    },
    recovery_actions=[
        {"action": "switch_to_backup", "success_probability": 0.85},
        {"action": "water_rationing", "success_probability": 0.95},
    ],
    description="Water recovery system offline for 48-120 hours",
)

MOXIE_FAILURE = Scenario(
    name="MOXIE_FAILURE",
    probability=0.15,
    trigger_day="random",
    duration_days=14,  # 7-30 days to repair
    effects={
        "moxie_production": 0.0,
    },
    recovery_actions=[
        {"action": "repair_moxie", "success_probability": 0.60},
        {"action": "increase_eclss_recycling", "success_probability": 0.80},
    ],
    description="MOXIE O2 production offline for 7-30 days",
)

SABATIER_FAILURE = Scenario(
    name="SABATIER_FAILURE",
    probability=0.10,
    trigger_day="random",
    duration_days=14,  # 7-30 days to repair
    effects={
        "ch4_h2o_production": 0.0,
    },
    recovery_actions=[
        {"action": "repair_sabatier", "success_probability": 0.65},
        {"action": "vent_co2", "success_probability": 0.99},
    ],
    description="Sabatier reactor offline for 7-30 days",
)

CREW_MEDICAL_MINOR = Scenario(
    name="CREW_MEDICAL_MINOR",
    probability=0.80,  # Common
    trigger_day="random",
    duration_days=7,  # 3-14 days recovery
    effects={
        "crew_capacity_reduction": 1,  # One crew member at partial capacity
        "crew_partial": True,
    },
    recovery_actions=[
        {"action": "medical_treatment", "success_probability": 0.95},
        {"action": "rest_period", "success_probability": 0.99},
    ],
    description="Minor medical issue reducing one crew member's capacity",
)

CREW_MEDICAL_MAJOR = Scenario(
    name="CREW_MEDICAL_MAJOR",
    probability=0.10,
    trigger_day="random",
    duration_days=60,  # 30-180 days recovery
    effects={
        "crew_capacity_reduction": 1,  # One crew member fully incapacitated
        "crew_full_incapacitation": True,
    },
    recovery_actions=[
        {"action": "surgery", "success_probability": 0.70},
        {"action": "telemedicine_consultation", "success_probability": 0.50},
    ],
    description="Major medical issue fully incapacitating one crew member",
)

POWER_SYSTEM_DEGRADATION = Scenario(
    name="POWER_SYSTEM_DEGRADATION",
    probability=1.00,  # Inevitable
    trigger_day="random",
    duration_days=365,  # Permanent
    effects={
        "power_output_multiplier": 0.95,  # 5% degradation per year
    },
    recovery_actions=[
        {"action": "panel_cleaning", "success_probability": 0.80},
        {"action": "panel_replacement", "success_probability": 0.90},
    ],
    description="Solar panel degradation reducing output by 5% annually",
)

EQUIPMENT_FAILURE_RANDOM = Scenario(
    name="EQUIPMENT_FAILURE_RANDOM",
    probability=2.00,  # Twice per year average
    trigger_day="random",
    duration_days=4,  # 1-7 days to repair
    effects={
        "random_system_offline": True,
    },
    recovery_actions=[
        {"action": "repair", "success_probability": 0.75},
        {"action": "workaround", "success_probability": 0.85},
    ],
    description="Random equipment failure affecting non-critical system",
)

CONJUNCTION_BLACKOUT = Scenario(
    name="CONJUNCTION_BLACKOUT",
    probability=1.00,  # Deterministic every 780 days
    trigger_day=780,  # Predictable
    duration_days=14,
    effects={
        "earth_communication": 0.0,
        "earth_support": 0.0,
    },
    recovery_actions=[],  # No recovery possible, must endure
    description="Solar conjunction communication blackout for 14 days",
)

# === SCENARIO COLLECTION ===

SCENARIOS = {
    "BASELINE": None,  # No special scenario
    "DUST_STORM_GLOBAL": DUST_STORM_GLOBAL,
    "DUST_STORM_REGIONAL": DUST_STORM_REGIONAL,
    "HAB_BREACH_SMALL": HAB_BREACH_SMALL,
    "HAB_BREACH_LARGE": HAB_BREACH_LARGE,
    "ECLSS_O2_FAILURE": ECLSS_O2_FAILURE,
    "ECLSS_H2O_FAILURE": ECLSS_H2O_FAILURE,
    "MOXIE_FAILURE": MOXIE_FAILURE,
    "SABATIER_FAILURE": SABATIER_FAILURE,
    "CREW_MEDICAL_MINOR": CREW_MEDICAL_MINOR,
    "CREW_MEDICAL_MAJOR": CREW_MEDICAL_MAJOR,
    "POWER_SYSTEM_DEGRADATION": POWER_SYSTEM_DEGRADATION,
    "EQUIPMENT_FAILURE_RANDOM": EQUIPMENT_FAILURE_RANDOM,
    "CONJUNCTION_BLACKOUT": CONJUNCTION_BLACKOUT,
}

# Mandatory scenarios for Monte Carlo validation
MANDATORY_SCENARIOS = [
    "BASELINE",
    "DUST_STORM_GLOBAL",
    "HAB_BREACH_SMALL",
    "ECLSS_O2_FAILURE",
    "CREW_MEDICAL_MAJOR",
    "CONJUNCTION_BLACKOUT",
]


def get_scenario(name: str) -> Scenario | None:
    """Get scenario by name.

    Args:
        name: Scenario name

    Returns:
        Scenario object or None if not found.
    """
    return SCENARIOS.get(name)


def list_scenarios() -> list[str]:
    """List all available scenario names.

    Returns:
        List of scenario names.
    """
    return list(SCENARIOS.keys())


def get_mandatory_scenarios() -> list[Scenario]:
    """Get list of mandatory scenarios for validation.

    Returns:
        List of mandatory Scenario objects.
    """
    return [SCENARIOS[name] for name in MANDATORY_SCENARIOS if SCENARIOS.get(name)]
