"""Mars Computational Sovereignty Simulator.

THE PARADIGM INVERSION:
    Everyone models MASS (kg cargo) and ENERGY (watts power).
    Nobody models BITS (decisions/second to survive).
    The unmodeled dimension is the binding constraint.

THE PHYSICS:
    Mars colony sovereignty = information-theoretic threshold where
    internal decision capacity (bits/sec) exceeds Earth input capacity.
    Latency (3-22 min) is irreducible. Conjunction blackout (14 days)
    is deterministic. Decision paralysis is fatal.

Source: SpaceProof D20 Production Evolution - Mars Sovereignty Spec
"""

from .constants import (
    CREW_MIN_GEORGE_MASON,
    CREW_MIN_SALOTTI,
    DECISION_BIT_COMPLEXITY_CRITICAL,
    DECISION_BIT_COMPLEXITY_HIGH,
    DECISIONS_PER_DAY_CRITICAL,
    DECISIONS_PER_DAY_HIGH,
    HUMAN_CO2_KG_PER_DAY,
    HUMAN_METABOLIC_HEAT_W,
    HUMAN_O2_KG_PER_DAY,
    HUMAN_WATER_KG_PER_DAY,
    ISS_ANOMALIES_PER_YEAR,
    ISS_ECLSS_MTBF_HOURS,
    ISS_H2O_RECOVERY_RATIO,
    ISS_O2_CLOSURE_RATIO,
    MARS_CONJUNCTION_BLACKOUT_DAYS,
    MARS_LIGHT_DELAY_MAX_SEC,
    MARS_LIGHT_DELAY_MIN_SEC,
    MARS_SOLAR_FLUX_DUST_W_M2,
    MARS_SOLAR_FLUX_PEAK_W_M2,
    MARS_SYNODIC_PERIOD_DAYS,
    MOXIE_O2_G_PER_HOUR,
    SABATIER_EFFICIENCY,
    STARSHIP_CREW_CAPACITY,
    STARSHIP_PAYLOAD_KG,
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
    calculate_decision_latency_cost,
    calculate_earth_input_rate,
    calculate_internal_capacity,
    compute_sovereignty_threshold,
)
from .integrator import (
    calculate_sovereignty_score,
    compute_crew_size_threshold,
    generate_failure_tree,
    identify_binding_constraint,
    validate_against_research,
)
from .life_support import (
    calculate_eclss_reliability,
    calculate_h2o_balance,
    calculate_life_support_entropy_rate,
    calculate_o2_balance,
    calculate_thermal_entropy,
)
from .monte_carlo import (
    calculate_survival_probability,
    generate_scenario,
    identify_failure_modes,
    run_simulation,
    simulate_day,
)
from .resources import (
    calculate_isru_closure,
    calculate_reserve_buffer,
    calculate_resupply_cadence,
    calculate_starship_manifest,
    identify_binding_resource,
)
from .scenarios import SCENARIOS, Scenario

# High-level API functions
from .api import (
    calculate_mars_sovereignty,
    compare_configs,
    find_crew_threshold,
    generate_report,
    get_default_config,
    load_config,
)

__all__ = [
    # Constants
    "TENANT_ID",
    "ISS_ECLSS_MTBF_HOURS",
    "ISS_O2_CLOSURE_RATIO",
    "ISS_H2O_RECOVERY_RATIO",
    "ISS_ANOMALIES_PER_YEAR",
    "HUMAN_METABOLIC_HEAT_W",
    "HUMAN_O2_KG_PER_DAY",
    "HUMAN_CO2_KG_PER_DAY",
    "HUMAN_WATER_KG_PER_DAY",
    "MARS_SOLAR_FLUX_PEAK_W_M2",
    "MARS_SOLAR_FLUX_DUST_W_M2",
    "MARS_LIGHT_DELAY_MIN_SEC",
    "MARS_LIGHT_DELAY_MAX_SEC",
    "MARS_CONJUNCTION_BLACKOUT_DAYS",
    "MARS_SYNODIC_PERIOD_DAYS",
    "STARSHIP_PAYLOAD_KG",
    "STARSHIP_CREW_CAPACITY",
    "MOXIE_O2_G_PER_HOUR",
    "SABATIER_EFFICIENCY",
    "CREW_MIN_GEORGE_MASON",
    "CREW_MIN_SALOTTI",
    "DECISIONS_PER_DAY_CRITICAL",
    "DECISIONS_PER_DAY_HIGH",
    "DECISION_BIT_COMPLEXITY_CRITICAL",
    "DECISION_BIT_COMPLEXITY_HIGH",
    # Crew matrix
    "define_skill_matrix",
    "calculate_coverage",
    "calculate_redundancy",
    "identify_gaps",
    "compute_crew_entropy",
    # Life support
    "calculate_o2_balance",
    "calculate_h2o_balance",
    "calculate_thermal_entropy",
    "calculate_eclss_reliability",
    "calculate_life_support_entropy_rate",
    # Decision capacity
    "calculate_internal_capacity",
    "calculate_earth_input_rate",
    "calculate_decision_latency_cost",
    "compute_sovereignty_threshold",
    "calculate_conjunction_survival",
    # Resources
    "calculate_isru_closure",
    "calculate_reserve_buffer",
    "calculate_resupply_cadence",
    "calculate_starship_manifest",
    "identify_binding_resource",
    # Integrator
    "calculate_sovereignty_score",
    "identify_binding_constraint",
    "generate_failure_tree",
    "compute_crew_size_threshold",
    "validate_against_research",
    # Monte Carlo
    "run_simulation",
    "generate_scenario",
    "simulate_day",
    "calculate_survival_probability",
    "identify_failure_modes",
    # Scenarios
    "SCENARIOS",
    "Scenario",
    # API
    "calculate_mars_sovereignty",
    "load_config",
    "find_crew_threshold",
    "compare_configs",
    "generate_report",
    "get_default_config",
]
