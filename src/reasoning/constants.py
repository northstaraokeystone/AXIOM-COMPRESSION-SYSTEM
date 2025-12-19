"""reasoning/constants.py - Constants for sovereignty timeline projections.

All constants used in sovereignty reasoning calculations.
"""

# === PARTITION/RESILIENCE CONSTANTS ===

MIN_EFF_ALPHA_BOUND = 2.63
"""Minimum effective alpha at 40% partition per Grok validation."""

CYCLES_THRESHOLD_EARLY = 1000
"""Early sovereignty marker: 10^3 person-equivalent."""

CYCLES_THRESHOLD_CITY = 1_000_000
"""City sovereignty threshold: 10^6 person-equivalent."""


# === PIPELINE CONSTANTS ===

PILOT_RETENTION_TARGET = 1.05
"""Target retention for pilot + quantum + tuned sweep pipeline."""

EXPECTED_FINAL_RETENTION = 1.062
"""Expected final retention from Grok simulation."""

EXPECTED_EFF_ALPHA = 2.89
"""Expected effective alpha from Grok simulation."""


# === SCALABILITY GATE CONSTANTS ===

SCALABILITY_GATE_THRESHOLD = 3.06
"""Minimum alpha at 10^9 for scalability gate pass."""

SCALABILITY_INSTABILITY_TOLERANCE = 0.00
"""Zero tolerance for instability at scale."""

SCALABILITY_DEGRADATION_TOLERANCE = 0.01
"""Maximum 1% degradation tolerance."""


__all__ = [
    "MIN_EFF_ALPHA_BOUND",
    "CYCLES_THRESHOLD_EARLY",
    "CYCLES_THRESHOLD_CITY",
    "PILOT_RETENTION_TARGET",
    "EXPECTED_FINAL_RETENTION",
    "EXPECTED_EFF_ALPHA",
    "SCALABILITY_GATE_THRESHOLD",
    "SCALABILITY_INSTABILITY_TOLERANCE",
    "SCALABILITY_DEGRADATION_TOLERANCE",
]
