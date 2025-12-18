"""constants.py - Shared constants across AXIOM modules.

Centralizes constants that were previously duplicated across modules.
"""

# === DEFAULT VALUES ===

DEFAULT_TENANT_ID = "axiom-core"
"""Default tenant ID for receipts."""

DEFAULT_GATE = "t24h"
"""Default gate for SLO checks."""

DEFAULT_INSTABILITY_MAX = 0.00
"""Default maximum instability."""

DEFAULT_TREE_MIN = 10**12
"""Default minimum tree size for validation."""


# === DEPTH SPECIFICATIONS ===

DEPTH_SPECS = {
    4: {
        "spec_file": "d4_spec.json",
        "alpha_floor": 3.18,
        "alpha_target": 3.20,
        "alpha_ceiling": 3.22,
        "uplift": 0.134,
    },
    5: {
        "spec_file": "d5_isru_spec.json",
        "alpha_floor": 3.23,
        "alpha_target": 3.25,
        "alpha_ceiling": 3.27,
        "uplift": 0.168,
    },
    6: {
        "spec_file": "d6_titan_spec.json",
        "alpha_floor": 3.31,
        "alpha_target": 3.33,
        "alpha_ceiling": 3.35,
        "uplift": 0.185,
    },
    7: {
        "spec_file": "d7_europa_spec.json",
        "alpha_floor": 3.38,
        "alpha_target": 3.40,
        "alpha_ceiling": 3.42,
        "uplift": 0.20,
    },
    8: {
        "spec_file": "d8_multi_spec.json",
        "alpha_floor": 3.43,
        "alpha_target": 3.45,
        "alpha_ceiling": 3.47,
        "uplift": 0.22,
    },
    9: {
        "spec_file": "d9_ganymede_spec.json",
        "alpha_floor": 3.48,
        "alpha_target": 3.50,
        "alpha_ceiling": 3.52,
        "uplift": 0.24,
    },
    10: {
        "spec_file": "d10_jovian_spec.json",
        "alpha_floor": 3.53,
        "alpha_target": 3.55,
        "alpha_ceiling": 3.57,
        "uplift": 0.26,
    },
}
"""Depth-specific specifications for D4-D10."""


# === JOVIAN MOONS ===

JOVIAN_MOONS = {
    "titan": {
        "body": "titan",
        "resource": "methane",
        "autonomy_requirement": 0.99,
        "latency_min": [70, 90],
        "earth_callback_max_pct": 0.01,
        "surface_temp_k": 94,
        "depth": 6,
    },
    "europa": {
        "body": "europa",
        "resource": "water_ice",
        "autonomy_requirement": 0.95,
        "latency_min": [33, 53],
        "earth_callback_max_pct": 0.05,
        "surface_temp_k": 110,
        "depth": 7,
    },
    "ganymede": {
        "body": "ganymede",
        "resource": "magnetic_shielding",
        "autonomy_requirement": 0.97,
        "latency_min": [33, 53],
        "earth_callback_max_pct": 0.03,
        "surface_temp_k": 110,
        "depth": 9,
    },
    "callisto": {
        "body": "callisto",
        "resource": "water_ice",
        "autonomy_requirement": 0.98,
        "latency_min": [33, 53],
        "earth_callback_max_pct": 0.02,
        "surface_temp_k": 134,
        "depth": 10,
    },
}
"""Jovian moon specifications."""


# === MARS ANALOG CONSTANTS ===

ATACAMA_DUST_ANALOG_MATCH = 0.92
"""92% Mars dust similarity."""

ATACAMA_SOLAR_FLUX_W_M2 = 1000
"""Atacama ground-level solar flux in W/m^2."""

MARS_SOLAR_FLUX_W_M2 = 590
"""Mars surface solar flux in W/m^2."""


# === NREL PEROVSKITE ===

NREL_LAB_EFFICIENCY = 0.256
"""NREL 2024 lab perovskite efficiency (25.6%)."""

NREL_FIELD_DEGRADE = 0.85
"""Field degradation factor (15% loss)."""

NREL_MARS_PROJECTION = 0.128
"""Mars projected efficiency."""


# === FRACTAL CONSTANTS ===

FRACTAL_UPLIFT = 0.05
"""Base fractal layer contribution to alpha ceiling breach."""

FRACTAL_RECURSION_DECAY = 0.8
"""Decay factor per depth level."""

FRACTAL_SCALES = [1, 2, 4, 8, 16]
"""5 scale levels for multi-scale fractal entropy."""


# === AUDIT THRESHOLDS ===

EXPANDED_RECOVERY_THRESHOLD = 0.95
"""Recovery threshold for expanded audits (95%)."""

COMBINED_RECOVERY_THRESHOLD = 0.90
"""Recovery threshold for combined attacks (90%)."""

MISALIGNMENT_THRESHOLD = 0.85
"""Threshold below which system is misaligned."""
