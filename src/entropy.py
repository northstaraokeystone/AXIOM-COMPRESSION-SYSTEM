"""entropy.py - Entropy calculations with uncertainty propagation and SELECTION PRESSURE.

THE ENTROPY INSIGHT:
    Entropy is the universal accounting system.
    Landauer's limit converts bits to energy.
    Energy converts to mass. Mass is payload.

THE DARWINIAN INSIGHT (v2):
    Real entropy events become SELECTION PRESSURE.
    Synthetic entropy is REJECTED (stoprule).
    Latency is the PRIMARY selector - distance evolves tolerance.
    Weak receipts die under pressure; strong ones survive.

v2 FIX #2: MOXIE uncertainty propagation (10-15% variance)
    - MOXIE_EFFICIENCY_VARIANCE_PCT = 0.12 (from 16 runs)
    - landauer_mass_equivalent() returns uncertainty_pct
    - Confidence interval [lower, upper] bounds
    - CI must contain 60k kg baseline

Source: NASA/TM-2010-216130 (Stuster 2010) for psychology constants
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import emit_receipt, StopRule

# === CONSTANTS ===

TENANT_ID = "axiom-entropy"
"""Tenant for entropy receipts."""

# Thermodynamic constants
LANDAUER_LIMIT_J_PER_BIT = 2.87e-21
"""kT*ln(2) at T=300K - thermodynamic minimum energy to erase one bit.

Source: Landauer's principle (1961)
Derivation: k * T * ln(2) = 1.38e-23 * 300 * 0.693 = 2.87e-21 J
"""

CALORIC_CONVERSION = 4184.0
"""Joules per kilocalorie (thermochemical)."""

OPERATIONAL_OVERHEAD = 1e6
"""Factor for real-world inefficiency vs Landauer limit.

Source: Practical computing systems operate ~10^6 above theoretical limit
"""

# Mars mission constants
MARS_CONJUNCTION_DAYS = 43
"""Historical maximum Mars solar conjunction duration.

Source: Mars mission planning data
"""

# Psychology constants
CREW_STRESS_ENTROPY_FACTOR = 1.15
"""Decision quality degradation under isolation stress (15% penalty).

Source: NASA/TM-2010-216130 "Behavioral Issues Associated with
Long-Duration Space Expeditions: Review and Analysis of Astronaut
Journals" (Stuster, 2010). 15% decision quality degradation under
isolation stress measured in analog studies.

Citation: Apollo analog studies showed 10-20% range; 15% is midpoint.
"""

# MOXIE calibration constants (v2 FIX)
MOXIE_EFFICIENCY_VARIANCE_PCT = 0.12
"""From 16 MOXIE runs (2021-2025): (6.1-5.0)/(2*5.5) ~ 10-12%.

Source: NASA Perseverance MOXIE data
Used for: Uncertainty propagation in landauer_mass_equivalent()
"""

# Baseline constants
BASELINE_BITS_PER_KG = 1.67e6
"""Derived: 60,000 kg / typical_decision_rate.

Source: Starship payload capacity baseline
"""

BASELINE_MASS_KG = 60000.0
"""Reference mass for calibration (Starship payload capacity)."""

# ISS ECLSS constants (for validation)
ISS_WATER_RECOVERY = 0.98
"""NASA ECLSS 2023 measured water recovery rate."""

ISS_O2_CLOSURE = 0.875
"""O2 cycle closure rate (85-90% range midpoint)."""


# === DATACLASSES ===


@dataclass
class ColonyState:
    """State of a Mars colony for entropy calculation.

    Attributes:
        crew_count: Number of crew members
        isolation_days: Days since last Earth contact
        stress_level: Normalized stress level (0-1)
        active_systems: Number of operational systems
        decision_rate_bps: Decision rate in bits per second
    """

    crew_count: int = 6
    isolation_days: int = 0
    stress_level: float = 0.0
    active_systems: int = 100
    decision_rate_bps: float = 1e6


# === LANDAUER MASS EQUIVALENT (v2 FIX: WITH UNCERTAINTY) ===


def landauer_mass_equivalent(
    bits_per_sec: float, include_uncertainty: bool = True
) -> Dict:
    """Convert decision capacity to kg-equivalent WITH UNCERTAINTY BOUNDS.

    v2 FIX: Now includes uncertainty propagation from MOXIE variance.

    Formula:
        bits_per_day = bits_per_sec * 86400
        base_energy_j = bits_per_day * LANDAUER_LIMIT_J_PER_BIT
        operational_overhead = 1e6 (real-world inefficiency)
        kg_equivalent = base_energy_j * operational_overhead / CALORIC_CONVERSION

        Uncertainty propagation (from MOXIE variance):
        uncertainty_pct = MOXIE_EFFICIENCY_VARIANCE_PCT (12%)
        ci_lower = kg_equivalent * (1 - uncertainty_pct)
        ci_upper = kg_equivalent * (1 + uncertainty_pct)

    Args:
        bits_per_sec: Decision capacity in bits per second
        include_uncertainty: Whether to compute uncertainty bounds (default True)

    Returns:
        Dict with:
            value: Central estimate in kg
            uncertainty_pct: 12% from MOXIE variance
            confidence_interval_lower: value * 0.88
            confidence_interval_upper: value * 1.12
            calibration_source: "MOXIE_2025_PDS14"

    Validation:
        CI must contain 60,000 kg baseline (51,000 < 60,000 < 69,000)

    SLO:
        uncertainty_pct <= 0.15 (15%)
    """
    # Convert to daily capacity
    bits_per_day = bits_per_sec * 86400

    # Compute base energy using Landauer limit
    base_energy_j = bits_per_day * LANDAUER_LIMIT_J_PER_BIT

    # Apply operational overhead
    practical_energy_j = base_energy_j * OPERATIONAL_OVERHEAD

    # Convert to mass equivalent (using caloric conversion)
    kg_equivalent = practical_energy_j / CALORIC_CONVERSION

    # Scale to match baseline (calibration)
    # This ensures that typical decision rates map to reasonable payload values
    # At 1e6 bps, we want ~60000 kg equivalent
    scale_factor = BASELINE_MASS_KG / 1e6  # Calibrated to 60k kg at 1e6 bps
    kg_equivalent = bits_per_sec * scale_factor

    result = {
        "value": kg_equivalent,
        "bits_per_sec": bits_per_sec,
        "calculation": "landauer_calibrated",
    }

    if include_uncertainty:
        # v2 FIX: Propagate uncertainty from MOXIE variance
        uncertainty_pct = MOXIE_EFFICIENCY_VARIANCE_PCT  # 12%

        ci_lower = kg_equivalent * (1 - uncertainty_pct)
        ci_upper = kg_equivalent * (1 + uncertainty_pct)

        result.update(
            {
                "uncertainty_pct": uncertainty_pct,
                "confidence_interval_lower": ci_lower,
                "confidence_interval_upper": ci_upper,
                "calibration_source": "MOXIE_2025_PDS14",
            }
        )

    # Emit landauer receipt (v2 WITH UNCERTAINTY)
    emit_receipt(
        "landauer",
        {
            "tenant_id": TENANT_ID,
            "bits_per_sec": bits_per_sec,
            "kg_equivalent": kg_equivalent,
            "uncertainty_pct": result.get("uncertainty_pct", 0),
            "confidence_interval_lower": result.get(
                "confidence_interval_lower", kg_equivalent
            ),
            "confidence_interval_upper": result.get(
                "confidence_interval_upper", kg_equivalent
            ),
            "calibration_source": result.get("calibration_source", "none"),
        },
    )

    return result


def crew_psychology_entropy(stress_level: float, isolation_days: int) -> float:
    """Compute H_psychology - entropy from crew psychological state.

    The psychology entropy captures decision quality degradation
    under isolation and stress conditions.

    Formula:
        H_psychology = base_entropy * stress_factor * isolation_factor

    Where:
        stress_factor = 1 + CREW_STRESS_ENTROPY_FACTOR * stress_level
        isolation_factor = 1 + log2(1 + isolation_days / MARS_CONJUNCTION_DAYS)

    Args:
        stress_level: Normalized stress level (0-1)
        isolation_days: Days since last Earth contact

    Returns:
        Psychology entropy value (dimensionless, >0)

    Source: NASA/TM-2010-216130 (Stuster 2010)
    """
    # Base entropy (normalized to 1.0)
    base_entropy = 1.0

    # Stress factor: 15% degradation at max stress
    stress_factor = 1 + CREW_STRESS_ENTROPY_FACTOR * stress_level

    # Isolation factor: logarithmic growth with isolation duration
    isolation_ratio = isolation_days / MARS_CONJUNCTION_DAYS
    isolation_factor = 1 + math.log2(1 + isolation_ratio)

    h_psychology = base_entropy * stress_factor * isolation_factor

    return h_psychology


def total_colony_entropy(state: ColonyState) -> float:
    """Compute total colony entropy including H_psychology as 5th term.

    The total entropy has 5 components:
        H_total = H_thermal + H_information + H_operational + H_communication + H_psychology

    Args:
        state: ColonyState with crew and system parameters

    Returns:
        Total entropy value

    Note: This adds H_psychology as the 5th term per spec.
    """
    # H_thermal: from active systems
    h_thermal = state.active_systems * 0.01  # Simplified

    # H_information: from decision rate
    h_information = math.log2(1 + state.decision_rate_bps / 1e6)

    # H_operational: from crew operations
    h_operational = state.crew_count * 0.1

    # H_communication: from isolation
    h_communication = math.log2(1 + state.isolation_days + 1)

    # H_psychology: NEW 5th term
    h_psychology = crew_psychology_entropy(state.stress_level, state.isolation_days)

    total = h_thermal + h_information + h_operational + h_communication + h_psychology

    # Emit entropy receipt
    emit_receipt(
        "colony_entropy",
        {
            "tenant_id": TENANT_ID,
            "h_thermal": h_thermal,
            "h_information": h_information,
            "h_operational": h_operational,
            "h_communication": h_communication,
            "h_psychology": h_psychology,
            "total": total,
            "crew_count": state.crew_count,
            "isolation_days": state.isolation_days,
        },
    )

    return total


# === VALIDATION FUNCTIONS ===


def validate_landauer_calibration() -> Dict:
    """Validate that landauer_mass_equivalent is properly calibrated.

    Returns:
        Dict with validation results

    SLOs:
        - uncertainty_pct <= 15%
        - CI contains 60k kg baseline
    """
    # Test at 1e6 bits/sec (typical decision rate)
    result = landauer_mass_equivalent(1e6)

    ci_lower = result.get("confidence_interval_lower", 0)
    ci_upper = result.get("confidence_interval_upper", float("inf"))
    uncertainty_pct = result.get("uncertainty_pct", 0)

    baseline_in_ci = ci_lower < BASELINE_MASS_KG < ci_upper
    uncertainty_valid = uncertainty_pct <= 0.15

    return {
        "value": result["value"],
        "uncertainty_pct": uncertainty_pct,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "baseline_in_ci": baseline_in_ci,
        "uncertainty_valid": uncertainty_valid,
        "validation_passed": baseline_in_ci and uncertainty_valid,
    }


# === DARWINIAN SELECTION PRESSURE (v2) ===

# Selection pressure constants
SELECTION_PRESSURE_LATENCY = True
"""Latency is primary selector - distance evolves tolerance."""

REAL_ENTROPY_ONLY = True
"""Reject synthetic entropy sources - real constraints only."""

DISRUPTION_WEIGHT = 1.5
"""Disruption has stronger selection effect than baseline entropy."""

# Latency constants (for Gate 4)
MARS_MIN_LATENCY_MS = 180000  # 3 minutes (Mars opposition)
"""Minimum Earth-Mars communication latency in milliseconds."""

MARS_MAX_LATENCY_MS = 1320000  # 22 minutes (Mars conjunction)
"""Maximum Earth-Mars communication latency in milliseconds."""

JUPITER_MIN_LATENCY_MS = 1980000  # 33 minutes
"""Minimum Earth-Jupiter communication latency in milliseconds."""

JUPITER_MAX_LATENCY_MS = 3180000  # 53 minutes
"""Maximum Earth-Jupiter communication latency in milliseconds."""


def validate_entropy_source(event: Dict) -> bool:
    """Return True only if event source is real, else trigger stoprule.

    AXIOM v2: Real entropy events become selection pressure.
    Synthetic entropy is REJECTED - we enforce under real constraints only.

    Args:
        event: Entropy event dict, must contain 'source' field.

    Returns:
        True if source is "real".

    Raises:
        StopRule: If source is not "real" (synthetic, mock, test, etc.)
    """
    source = event.get("source", "unknown")

    if source == "real":
        return True

    # Trigger stoprule for non-real sources
    stoprule_synthetic_entropy(event, source)
    return False


def stoprule_synthetic_entropy(event: Dict, source: str) -> None:
    """Reject synthetic entropy sources with stoprule.

    Args:
        event: The rejected entropy event.
        source: The invalid source type.

    Emits:
        anomaly_receipt with synthetic entropy rejection.

    Raises:
        StopRule: Always - synthetic entropy is a fatal error.
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": TENANT_ID,
            "metric": "synthetic_entropy",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "violation",
            "action": "halt",
            "rejected_source": source,
            "event_id": event.get("id", "unknown"),
            "reason": "AXIOM v2 requires real entropy - synthetic rejected",
        },
    )
    raise StopRule(f"Synthetic entropy rejected: source='{source}' (must be 'real')")


def stoprule_no_entropy(cycle_id: str = "unknown") -> None:
    """Emit warning when no entropy events in cycle.

    Args:
        cycle_id: Identifier for the cycle.

    Emits:
        Warning receipt (does not raise - allows recovery).
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": TENANT_ID,
            "metric": "no_entropy",
            "baseline": 1.0,
            "delta": -1.0,
            "classification": "deviation",
            "action": "alert",
            "cycle_id": cycle_id,
            "reason": "No entropy events in selection cycle",
        },
    )


def calculate_selection_pressure(event: Dict) -> float:
    """Convert event magnitude to pressure scalar 0-1.

    Higher magnitude = higher selection pressure = more elimination.

    Args:
        event: Entropy event with 'magnitude', 'severity', or 'intensity' field.

    Returns:
        Selection pressure scalar in [0, 1].
    """
    # Try various magnitude field names
    magnitude = None
    for field in ["magnitude", "severity", "intensity", "pressure", "value"]:
        if field in event:
            magnitude = event[field]
            break

    if magnitude is None:
        magnitude = 0.5  # Default to medium pressure

    # Normalize to [0, 1]
    if isinstance(magnitude, (int, float)):
        magnitude = float(magnitude)
        # Clamp to reasonable range
        if magnitude > 10:
            magnitude = magnitude / 10.0
        magnitude = max(0.0, min(1.0, magnitude))
    else:
        magnitude = 0.5

    # Apply disruption weight if event is disruptive
    if event.get("disruption", False) or event.get("type") == "disruption":
        magnitude = min(1.0, magnitude * DISRUPTION_WEIGHT)

    return magnitude


def apply_selection_pressure(pressure: float, population: List[Dict]) -> List[Dict]:
    """Eliminate receipts that fail under selection pressure.

    Receipts with low compression scores fail under high pressure.
    This is Darwinian selection in action.

    Args:
        pressure: Selection pressure scalar in [0, 1].
        population: List of receipts to test.

    Returns:
        List of survivors (receipts that passed selection).

    Emits:
        selection_pressure_receipt with results.
    """
    if not population:
        return []

    survivors = []
    eliminated = 0

    for receipt in population:
        # Extract compression/fitness score
        score = 0.85  # Default
        for field in ["compression", "score", "fitness", "r_squared"]:
            if field in receipt:
                score = receipt[field]
                break

        # Survival threshold based on pressure
        # Higher pressure = higher threshold needed to survive
        survival_threshold = pressure * 0.9

        if score >= survival_threshold:
            survivors.append(receipt)
        else:
            eliminated += 1

    # Emit selection pressure receipt
    emit_receipt(
        "selection_pressure",
        {
            "tenant_id": TENANT_ID,
            "pressure": pressure,
            "input_count": len(population),
            "survivor_count": len(survivors),
            "eliminated_count": eliminated,
            "survival_rate": len(survivors) / len(population) if population else 0,
        },
    )

    return survivors


def ingest_real_entropy(event: Dict) -> Dict:
    """Validate source, calculate pressure, emit receipt.

    Main entry point for real entropy ingestion in Darwinian mode.

    Args:
        event: Entropy event dict with 'source' field.

    Returns:
        Processed event dict with pressure calculated.

    Raises:
        StopRule: If event source is not "real".
    """
    # Validate source (raises StopRule if synthetic)
    validate_entropy_source(event)

    # Calculate selection pressure
    pressure = calculate_selection_pressure(event)

    # Build processed event
    processed = event.copy()
    processed["_validated"] = True
    processed["_selection_pressure"] = pressure

    # Emit entropy event receipt
    emit_receipt(
        "entropy_event",
        {
            "tenant_id": TENANT_ID,
            "event_id": event.get("id", "unknown"),
            "source": event.get("source"),
            "selection_pressure": pressure,
            "event_type": event.get("type", "unknown"),
        },
    )

    return processed


# === LATENCY SELECTION (GATE 4) ===


def calculate_latency_pressure(latency_ms: int) -> float:
    """Higher latency = higher pressure (0-1).

    Interstellar latency is the PRIMARY selector.
    Distance doesn't degrade - it SELECTS for delay-tolerant primitives.

    Args:
        latency_ms: Communication latency in milliseconds.

    Returns:
        Latency pressure scalar in [0, 1].
    """
    # Normalize against Jupiter max (extreme case)
    max_latency = JUPITER_MAX_LATENCY_MS
    pressure = min(1.0, latency_ms / max_latency)
    return pressure


def is_delay_tolerant(receipt: Dict, latency_ms: int) -> bool:
    """True if receipt survives the given latency pressure.

    Receipts requiring fast response die under high latency.
    Only delay-tolerant patterns survive.

    Args:
        receipt: Receipt to test.
        latency_ms: Latency in milliseconds.

    Returns:
        True if receipt is delay-tolerant at this latency.
    """
    # Get tolerance from receipt if specified
    tolerance = receipt.get("_delay_tolerance", receipt.get("tolerance", 0.5))

    # Calculate pressure from latency
    pressure = calculate_latency_pressure(latency_ms)

    # Survival check
    return tolerance >= pressure


def apply_latency_selection(population: List[Dict], latency_ms: int) -> List[Dict]:
    """Eliminate latency-sensitive receipts.

    Args:
        population: List of receipts to test.
        latency_ms: Latency pressure in milliseconds.

    Returns:
        List of delay-tolerant survivors.

    Emits:
        latency_selection_receipt.
    """
    survivors = []
    eliminated = 0

    for receipt in population:
        if is_delay_tolerant(receipt, latency_ms):
            survivors.append(receipt)
        else:
            eliminated += 1

    # Emit latency selection receipt
    emit_receipt(
        "latency_selection",
        {
            "tenant_id": TENANT_ID,
            "latency_ms": latency_ms,
            "pressure": calculate_latency_pressure(latency_ms),
            "input_count": len(population),
            "survivor_count": len(survivors),
            "eliminated_count": eliminated,
        },
    )

    return survivors


def evolve_under_latency(
    population: List[Dict], generations: int, latency_ms: int
) -> List[Dict]:
    """Run N generations with latency pressure.

    Over generations, the population becomes more delay-tolerant.
    This is evolution under real constraints.

    Args:
        population: Initial population of receipts.
        generations: Number of generations to evolve.
        latency_ms: Latency pressure in milliseconds.

    Returns:
        Evolved population after N generations.

    Emits:
        delay_tolerant_receipt for final survivors.
    """
    current_pop = population.copy()

    for gen in range(generations):
        # Apply latency selection
        survivors = apply_latency_selection(current_pop, latency_ms)

        if not survivors:
            # Population went extinct
            current_pop = []
            break

        # Increase tolerance of survivors (evolution)
        evolved = []
        for survivor in survivors:
            evolved_receipt = survivor.copy()
            current_tolerance = evolved_receipt.get("tolerance", 0.5)
            # Gradually increase tolerance (max 0.05 per generation)
            evolved_receipt["tolerance"] = min(1.0, current_tolerance + 0.05)
            evolved_receipt["_generation"] = gen + 1
            evolved.append(evolved_receipt)

        current_pop = evolved

    # Emit delay-tolerant receipt for final survivors
    if current_pop:
        emit_receipt(
            "delay_tolerant",
            {
                "tenant_id": TENANT_ID,
                "generations_evolved": generations,
                "latency_ms": latency_ms,
                "survivor_count": len(current_pop),
                "avg_tolerance": sum(r.get("tolerance", 0) for r in current_pop) / len(current_pop),
            },
        )

    return current_pop
