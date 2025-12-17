"""constants.py - Centralized Physics Constants

ALL physics constants for AXIOM live here. Single source of truth.

Organization:
    - SHANNON: Entropy bounds and rates
    - MARS: Light delay and bandwidth parameters
    - GNN: Cache parameters and thresholds
    - PRUNING: Entropy pruning parameters
    - ALPHA: Effective alpha ranges and targets
    - TAU: Decision timing and costs
    - ABLATION: Testing modes and ranges

Source: CLAUDEME S8 + Grok analysis Dec 2025
"""

# === SHANNON ENTROPY BOUNDS ===

ENTROPY_ASYMPTOTE_E = 2.71828
"""physics: Shannon entropy bound ~e*ln(n). This is e (Euler's number).
NOT tunable - this is a physics constant from information theory.
Source: Grok - "Merkle batch entropy bounds as ~e*ln(n)" """

SHANNON_FLOOR_ALPHA = 2.71828
"""physics: Without engineering, alpha = e (Shannon baseline).
This is the FLOOR, not ceiling. Same value as ENTROPY_ASYMPTOTE_E.
Source: alpha_formula_spec.json - "e is the FLOOR (baseline)" """

ALPHA_CEILING_TARGET = 3.0
"""physics: Theoretical ceiling with ML optimization.
With full GNN + pruning stack, alpha approaches 3.0.
Source: alpha_formula_spec.json """

MIN_EFF_ALPHA_FLOOR = 2.7185
"""physics: Validated minimum effective alpha at floor.
From 1000-run sweep at 90d baseline testing."""

MIN_EFF_ALPHA_VALIDATED = 2.7185
"""physics: Validated minimum effective alpha from 1000-run sweep at 90d.
Alias for MIN_EFF_ALPHA_FLOOR for backward compatibility."""

# === MARS COMMUNICATION PARAMETERS ===

MARS_LIGHT_DELAY_MIN_S = 180
"""physics: Minimum Mars light delay in seconds (3 minutes).
Source: Physics - Mars at opposition."""

MARS_LIGHT_DELAY_MAX_S = 1320
"""physics: Maximum Mars light delay in seconds (22 minutes).
Source: Physics - Mars at conjunction."""

MARS_LIGHT_DELAY_AVG_S = 750
"""physics: Average Mars light delay in seconds (~12.5 minutes).
Source: Orbital average over synodic period."""

STARLINK_MARS_BANDWIDTH_MIN_MBPS = 2.0
"""physics: Minimum Starlink Mars relay bandwidth in Mbps.
Source: "2-10 Mbps 2025 sims" """

STARLINK_MARS_BANDWIDTH_MAX_MBPS = 10.0
"""physics: Maximum Starlink Mars relay bandwidth in Mbps.
Source: "2-10 Mbps 2025 sims" """

STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS = 4.0
"""physics: Expected (median) Starlink Mars relay bandwidth.
Source: Midpoint of range with pessimistic lean."""

# === HUMAN DECISION PARAMETERS ===

HUMAN_DECISION_RATE_BPS = 10
"""physics: Human decision rate in bits per second.
Source: Reviewer confirmed. Voice/gesture baseline."""

BITS_PER_DECISION = 9
"""physics: Bits required to encode a decision query/response cycle.
Derivation: log2(512) = 9 bits for typical decision space."""

# === GNN CACHE PARAMETERS ===

ASYMPTOTE_ALPHA = 2.72
"""physics: e-like stability ceiling from GNN saturation.
References ENTROPY_ASYMPTOTE_E, rounded for practical use."""

PRUNING_TARGET_ALPHA = 2.80
"""physics: Target effective alpha with ln(n) compression via pruning."""

CACHE_DEPTH_BASELINE = int(1e8)
"""physics: ~150d buffer at 50k entries/sol (10^8 entries)."""

CACHE_DEPTH_MIN = int(1e7)
"""physics: ~90d minimal coverage (10^7 entries)."""

CACHE_DEPTH_MAX = int(1e10)
"""physics: ~300d theoretical max (10^10 entries)."""

ENTRIES_PER_SOL = 50000
"""physics: Merkle batch scaling factor."""

# === OVERFLOW THRESHOLDS ===

OVERFLOW_THRESHOLD_DAYS = 200
"""physics: Cache overflow stoprule trigger (without pruning)."""

OVERFLOW_THRESHOLD_DAYS_PRUNED = 300
"""physics: Cache overflow threshold with pruning enabled (~50% extension)."""

OVERFLOW_CAPACITY_PCT = 0.95
"""physics: Halt at 95% saturation."""

QUORUM_FAIL_DAYS = 180
"""physics: Quorum degradation onset before cache overflow."""

CACHE_BREAK_DAYS = 200
"""physics: Cache overflow failure point."""

# === BLACKOUT PARAMETERS ===

BLACKOUT_BASE_DAYS = 43
"""physics: Mars solar conjunction maximum duration in days."""

BLACKOUT_SWEEP_MAX_DAYS = 200
"""physics: Maximum sweep duration for extreme blackout testing."""

BLACKOUT_PRUNING_TARGET_DAYS = 250
"""physics: Extended survival target with entropy pruning (250d at α>2.8)."""

# === RETENTION CURVE PARAMETERS ===

NONLINEAR_RETENTION_FLOOR = 1.25
"""physics: Asymptotic retention floor (better than linear at 90d)."""

RETENTION_BASE_FACTOR = 1.4
"""physics: Baseline retention factor at 43d blackout."""

CURVE_TYPE = "gnn_nonlinear"
"""physics: Model identifier - replaces linear."""

DECAY_LAMBDA = 0.003
"""physics: Calibrated decay constant for nonlinear retention."""

SATURATION_KAPPA = 0.05
"""physics: Saturation rate for asymptotic alpha."""

# === PRUNING PARAMETERS ===

LN_N_TRIM_FACTOR_BASE = 0.3
"""physics: Conservative initial redundancy reduction (30% of ln(n))."""

LN_N_TRIM_FACTOR_MAX = 0.5
"""physics: Aggressive pruning ceiling (50% of ln(n))."""

OVER_PRUNE_STOPRULE_THRESHOLD = 0.6
"""physics: StopRule triggers if trim_factor exceeds this (too aggressive)."""

ENTROPY_PRUNE_THRESHOLD = 0.1
"""physics: Branches with entropy < 0.1*ln(n) are pruning candidates."""

DEDUP_PRIORITY = 1.0
"""physics: Deterministic dedup runs first (zero risk)."""

PREDICTIVE_PRIORITY = 0.7
"""physics: GNN-predicted pruning weighted lower (bounded uncertainty)."""

MIN_PROOF_PATHS_RETAINED = 3
"""physics: Safety - always keep at least 3 audit/proof paths."""

MIN_CONFIDENCE_THRESHOLD = 0.7
"""physics: StopRule on predictive phase if GNN confidence < 0.7."""

MIN_QUORUM_FRACTION = 2/3
"""physics: Quorum must be maintained at >= 2/3 nodes after pruning."""

DEDUP_RATIO_EXPECTED = 0.15
"""physics: Expected dedup ratio for typical Merkle batches (>=15%)."""

PREDICTIVE_ACCURACY_TARGET = 0.85
"""physics: Target GNN prediction accuracy for low-entropy branches."""

# === TAU DECISION TIMING ===

TAU_DECISION_DECAY_S = 300
"""physics: Time constant for decision value decay in seconds (5 minutes).
Source: Grok - "Model effective rate as bw * exp(-delay/tau)" """

TAU_BASE_CURRENT_S = 300
"""physics: Current human-in-loop decision latency in seconds."""

TAU_MIN_AUTONOMY_S = 30
"""physics: Aggressive autonomy target (30s decision cycle, fully autonomous)."""

TAU_MIN_ACHIEVABLE_S = 30
"""physics: Physical floor: 30-second decision cycles with full autonomy."""

TAU_REFERENCE_S = 300.0
"""physics: Reference tau for baseline calculations."""

# === TAU COST PARAMETERS ===

TAU_COST_EXPONENT = 2.0
"""physics: Reducing τ by half costs 4x (quadratic cost scaling)."""

TAU_COST_BASE_M = 100
"""physics: Base cost in millions USD to halve τ from current."""

TAU_COST_INFLECTION_M = 400
"""physics: Inflection point for logistic curve in millions USD ($400M)."""

TAU_COST_STEEPNESS = 0.01
"""physics: Logistic curve steepness parameter (k)."""

AUTONOMY_INVESTMENT_MAX_M = 1000
"""physics: Maximum reasonable autonomy R&D spend ($1B)."""

BANDWIDTH_COST_PER_MBPS_M = 10
"""physics: Cost per Mbps upgrade in millions USD."""

COMPUTE_FLOPS_TO_DECISIONS = 1e-15
"""physics: Conversion factor from FLOPS to decisions/sec."""

# === META COMPRESSION (AI acceleration) ===

ITERATION_COMPRESSION_FACTOR = 7.5
"""physics: Midpoint of Grok's 5-10x for AI→AI loops."""

META_TAU_HUMAN_DAYS = 30
"""physics: Human-only R&D cycle time in days."""

META_TAU_AI_DAYS = 4
"""physics: AI-mediated R&D cycle time in days."""

# === VARIANCE RATIOS ===

DELAY_VARIANCE_RATIO = 7.33
"""physics: Ratio of delay range to minimum delay.
Source: Grok - "3-22 min delay varies more than bandwidth" """

BANDWIDTH_VARIANCE_RATIO = 4.0
"""physics: Ratio of bandwidth range to minimum bandwidth."""

# === ABLATION MODES ===

ABLATION_MODES = ["full", "no_cache", "no_prune", "baseline"]
"""physics: Four-mode isolation testing for ablation analysis."""

RETENTION_FACTOR_GNN_RANGE = (1.008, 1.015)
"""physics: Isolated GNN contribution from Grok ablation analysis."""

RETENTION_FACTOR_PRUNE_RANGE = (1.008, 1.015)
"""physics: Isolated pruning contribution from Grok ablation analysis."""

# === SPEC FILE PATHS ===

ENTROPY_PRUNING_SPEC_PATH = "data/entropy_pruning_spec.json"
"""Path to entropy pruning specification file."""

GNN_CACHE_SPEC_PATH = "data/gnn_cache_spec.json"
"""Path to GNN cache specification file."""

ALPHA_FORMULA_SPEC_PATH = "data/alpha_formula_spec.json"
"""Path to alpha formula specification file."""
