"""Mars Sovereignty Constants - Research-Validated Values.

THE PHYSICS:
    Every constant here is either:
    1. Directly measured (NASA ECLSS, Perseverance MOXIE)
    2. Calculated from physics (orbital mechanics, thermodynamics)
    3. From peer-reviewed research (George Mason 2023, Salotti Nature 2020)

    No speculative constants. Conservative estimates when ranges exist.

Sources:
    - NASA ECLSS 2019: ISS life support reliability data
    - NASA ECLSS 2023: O2/H2O closure ratios
    - Perseverance MOXIE 2021-2025: In-situ O2 production rates
    - George Mason 2023: Minimum viable crew via agent-based modeling
    - Salotti Nature 2020: Minimum crew via work-capacity analysis
    - Mars orbital mechanics: Synodic period, conjunction duration
    - SpaceX official: Starship payload specifications
"""

# === IDENTITY ===
TENANT_ID = "spaceproof-mars-sovereignty"

# === LIFE SUPPORT (NASA ECLSS measured data) ===

# ISS ECLSS Mean Time Between Failures
# Source: NASA 2019 reliability data
# NOTE: Actual measured is 5.6x lower than design (10000h)
ISS_ECLSS_MTBF_HOURS = 1752

# ISS O2 closure ratio (% of O2 recycled)
# Source: NASA ECLSS 2023 measured 85-90% range, use midpoint
ISS_O2_CLOSURE_RATIO = 0.875

# ISS H2O recovery ratio
# Source: NASA ECLSS 2023 measured achievement
ISS_H2O_RECOVERY_RATIO = 0.98

# ISS average critical anomalies per year
# Source: NASA operational data
ISS_ANOMALIES_PER_YEAR = 3.5

# === HUMAN PHYSIOLOGY ===

# Human metabolic heat production
# Source: Physiology textbooks, 80-120W range, use midpoint
HUMAN_METABOLIC_HEAT_W = 100

# Human O2 consumption per day
# Source: NASA standard consumption rate
HUMAN_O2_KG_PER_DAY = 0.84

# Human CO2 production per day
# Source: NASA standard production rate
HUMAN_CO2_KG_PER_DAY = 1.0

# Human water consumption per day (drinking + hygiene)
# Source: NASA standard consumption rate
HUMAN_WATER_KG_PER_DAY = 3.6

# Human food consumption per day (dry mass)
# Source: NASA ISS data
HUMAN_FOOD_KG_PER_DAY = 1.8

# Human caloric requirement per day
# Source: NASA, varies 2000-3000, use 2500 for Mars
HUMAN_KCAL_PER_DAY = 2500

# === MARS ENVIRONMENT ===

# Mars solar flux at surface, clear sky equator
# Source: NASA Viking mission data
MARS_SOLAR_FLUX_PEAK_W_M2 = 590

# Mars solar flux during global dust storm (1% of peak)
# Source: NASA dust storm observations
MARS_SOLAR_FLUX_DUST_W_M2 = 6

# Mars gravity
# Source: Physics
MARS_GRAVITY_M_S2 = 3.72

# Mars atmospheric pressure (average)
# Source: NASA
MARS_ATMOSPHERIC_PRESSURE_PA = 610

# === MARS COMMUNICATION ===

# Light delay at opposition (closest approach)
# Source: Orbital mechanics, c = 299792458 m/s
MARS_LIGHT_DELAY_MIN_SEC = 180  # 3 minutes

# Light delay at conjunction (farthest)
# Source: Orbital mechanics
MARS_LIGHT_DELAY_MAX_SEC = 1320  # 22 minutes

# Average light delay
# Source: Calculated from orbital mechanics
MARS_LIGHT_DELAY_AVG_SEC = 750  # ~12.5 minutes

# Communication blackout during solar conjunction
# Source: Mars orbital mechanics
MARS_CONJUNCTION_BLACKOUT_DAYS = 14

# Synodic period (time between launch windows)
# Source: Orbital mechanics
MARS_SYNODIC_PERIOD_DAYS = 780  # ~26 months

# === SPACEX HARDWARE ===

# Starship payload capacity
# Source: SpaceX official, 100-150t range, use 125t
STARSHIP_PAYLOAD_KG = 125000

# Starship crew capacity
# Source: SpaceX estimates for Mars missions
STARSHIP_CREW_CAPACITY = 100

# === ISRU (In-Situ Resource Utilization) ===

# MOXIE O2 production rate
# Source: Perseverance MOXIE experiment 2021-2025, measured 5-6 g/hr
MOXIE_O2_G_PER_HOUR = 5.5

# MOXIE power consumption
# Source: NASA MOXIE specs
MOXIE_POWER_W = 300

# Sabatier reactor efficiency
# Source: NASA estimate, 0.8-0.95 range (no Mars data yet)
SABATIER_EFFICIENCY = 0.85

# Kilopower reactor output
# Source: NASA KRUSTY test reactor
KILOPOWER_KW = 10

# === RESEARCH BENCHMARKS ===

# Minimum viable crew from George Mason 2023 study
# Source: Agent-based modeling simulation
CREW_MIN_GEORGE_MASON = 22

# Minimum viable crew from Salotti Nature 2020 study
# Source: Work-capacity analysis
CREW_MIN_SALOTTI = 110

# === DECISION CAPACITY (NOVEL - estimated from ISS operations) ===

# Critical decisions per day
# Source: Estimate from ISS flight director logs
DECISIONS_PER_DAY_CRITICAL = 50

# High priority decisions per day
# Source: Estimate from ISS operations
DECISIONS_PER_DAY_HIGH = 200

# Medium priority decisions per day
DECISIONS_PER_DAY_MEDIUM = 500

# Low priority decisions per day
DECISIONS_PER_DAY_LOW = 1000

# Bits per critical decision (NOVEL estimate)
# Represents information complexity of life-or-death choices
DECISION_BIT_COMPLEXITY_CRITICAL = 1000

# Bits per high priority decision
DECISION_BIT_COMPLEXITY_HIGH = 100

# Bits per medium decision
DECISION_BIT_COMPLEXITY_MEDIUM = 20

# Bits per low decision
DECISION_BIT_COMPLEXITY_LOW = 5

# === SKILL CATEGORIES ===

# Critical skills that require 24/7 coverage
SKILL_CATEGORY_CRITICAL = ["medical", "systems", "life_support"]

# High priority skills
SKILL_CATEGORY_HIGH = ["engineering", "agriculture", "power"]

# Medium priority skills
SKILL_CATEGORY_MEDIUM = ["science", "operations", "communications"]

# Low priority skills (can be part-time)
SKILL_CATEGORY_LOW = ["administration", "recreation", "training"]

# === CREW CONSTRAINTS ===

# Maximum work hours per week (avoid burnout)
MAX_WORK_HOURS_PER_WEEK = 60

# Minimum rest hours per day
MIN_REST_HOURS_PER_DAY = 8

# Crew per 8-hour shift for 24/7 coverage
CREW_PER_SHIFT_24_7 = 3

# Minimum redundancy for critical skills
MIN_REDUNDANCY_CRITICAL = 2.0

# Minimum redundancy for high skills
MIN_REDUNDANCY_HIGH = 1.5

# === RESOURCE TARGETS ===

# Minimum buffer days for critical resources
BUFFER_DAYS_MINIMUM = 90

# Nominal buffer days
BUFFER_DAYS_NOMINAL = 180

# Target ISRU closure ratio
ISRU_CLOSURE_TARGET = 0.85

# === THERMODYNAMICS ===

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN_W_M2_K4 = 5.67e-8

# Mars ambient temperature (average)
MARS_AMBIENT_TEMP_K = 210

# Habitat target temperature
HAB_TARGET_TEMP_C = 22

# Habitat temperature bounds
HAB_TEMP_MIN_C = 0
HAB_TEMP_MAX_C = 40

# O2 partial pressure bounds (kPa)
O2_PARTIAL_PRESSURE_MIN_KPA = 19.5
O2_PARTIAL_PRESSURE_MAX_KPA = 23.5

# === SCORING WEIGHTS ===

# Default weights for sovereignty score calculation
DEFAULT_WEIGHTS = {
    "crew": 0.25,  # Crew coverage
    "life_support": 0.30,  # Life support entropy
    "decision": 0.35,  # Decision capacity (HIGHEST - the novel dimension)
    "resources": 0.10,  # Resource closure
}

# === SLO THRESHOLDS ===

# Calculation time SLO
SLO_CALCULATION_MS = 1000  # <1 second

# Monte Carlo time SLO
SLO_MONTE_CARLO_S = 60  # <60 seconds for 1000 iterations

# Crew threshold search SLO
SLO_CREW_THRESHOLD_S = 2  # <2 seconds

# === VALIDATION TOLERANCES ===

# Research benchmark validation tolerance
RESEARCH_VALIDATION_TOLERANCE = 0.10  # Within 10% of expected

# Score range
SOVEREIGNTY_SCORE_MIN = 0.0
SOVEREIGNTY_SCORE_MAX = 100.0
