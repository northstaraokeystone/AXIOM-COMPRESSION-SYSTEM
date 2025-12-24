"""decision_augmented.py - AI vs Neuralink Augmentation Factors

THE AUGMENTATION INSIGHT:
    1 person + AI @ 5x factor = 5 person-equivalents.
    BUT: AI needs power, cooling, compute mass.
    Trade-off: Send 4 people + xAI compute OR send 20 people.

Source: SpaceProof v3.0 Multi-Tier Autonomy Network Evolution
Grok: "xAI autonomy for deep-space lacks policy framework"
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import math

from .core import emit_receipt

# === CONSTANTS ===

TENANT_ID = "spaceproof-augmentation"

# Augmentation factors (estimated from capabilities)
AI_AUGMENTATION_FACTOR = 5.0  # xAI multiplier (Grok collaboration estimate)
NEURALINK_AUGMENTATION_FACTOR = 20.0  # Neural interface (speculative)
HUMAN_ONLY_FACTOR = 1.0  # Baseline

# Power requirements (watts per factor point)
AI_POWER_PER_FACTOR_POINT = 100  # W (data center scaling)
NEURALINK_POWER_PER_FACTOR_POINT = 10  # W (lower power for neural interface)
HUMAN_METABOLIC_W = 100  # Baseline human power

# Mass requirements (kg per factor point)
AI_MASS_PER_FACTOR_POINT = 50  # kg (compute hardware)
NEURALINK_MASS_PER_FACTOR_POINT = 0.5  # kg (implant + support)

# Reliability factors
AI_RELIABILITY = 0.99  # Uptime
NEURALINK_RELIABILITY = 0.95  # Medical risk factor
HUMAN_RELIABILITY = 0.98  # Human error rate


class AugmentationType(Enum):
    """Types of decision augmentation."""

    HUMAN_ONLY = "human_only"
    AI_ASSISTED = "ai_assisted"
    NEURALINK_ASSISTED = "neuralink_assisted"
    HYBRID = "hybrid"  # AI + Neuralink


@dataclass
class AugmentationConfig:
    """Configuration for augmentation calculation.

    Attributes:
        augmentation_type: Type of augmentation
        compute_mass_kg: Available compute mass
        power_available_w: Available power
        reliability_required: Minimum reliability threshold
    """

    augmentation_type: AugmentationType = AugmentationType.HUMAN_ONLY
    compute_mass_kg: float = 0.0
    power_available_w: float = 0.0
    reliability_required: float = 0.90


@dataclass
class AugmentationResult:
    """Result of augmentation calculation.

    Attributes:
        base_factor: Nominal augmentation factor
        effective_factor: Actual factor after constraints
        power_required: Power needed in watts
        mass_required: Mass needed in kg
        reliability: Expected reliability
        feasible: Whether augmentation is feasible
        bottleneck: What limits the augmentation (if any)
    """

    base_factor: float
    effective_factor: float
    power_required: float
    mass_required: float
    reliability: float
    feasible: bool
    bottleneck: Optional[str] = None


def get_base_factor(augmentation_type: AugmentationType) -> float:
    """Get baseline augmentation factor for type.

    Args:
        augmentation_type: Type of augmentation

    Returns:
        Baseline factor multiplier
    """
    factors = {
        AugmentationType.HUMAN_ONLY: HUMAN_ONLY_FACTOR,
        AugmentationType.AI_ASSISTED: AI_AUGMENTATION_FACTOR,
        AugmentationType.NEURALINK_ASSISTED: NEURALINK_AUGMENTATION_FACTOR,
        AugmentationType.HYBRID: AI_AUGMENTATION_FACTOR * NEURALINK_AUGMENTATION_FACTOR / 10,  # Diminishing returns
    }
    return factors.get(augmentation_type, HUMAN_ONLY_FACTOR)


def calculate_augmentation_factor(augmentation_type: str, compute_mass_kg: float) -> float:
    """Calculate effective augmentation factor.

    Factor scales with available compute mass.
    Base factor achieved at 100kg compute, scales logarithmically.

    Args:
        augmentation_type: Type string (ai_assisted, neuralink_assisted, etc.)
        compute_mass_kg: Available compute mass in kg

    Returns:
        Effective augmentation factor
    """
    # Convert string to enum
    try:
        aug_type = AugmentationType(augmentation_type)
    except ValueError:
        aug_type = AugmentationType.HUMAN_ONLY

    base = get_base_factor(aug_type)

    if aug_type == AugmentationType.HUMAN_ONLY:
        return base

    # Scale factor with compute mass (logarithmic scaling)
    # Full factor at 100kg, scales down below that
    if compute_mass_kg <= 0:
        return HUMAN_ONLY_FACTOR

    mass_factor = math.log2(1 + compute_mass_kg) / math.log2(101)  # log2(1+100)
    effective = HUMAN_ONLY_FACTOR + (base - HUMAN_ONLY_FACTOR) * mass_factor

    return effective


def effective_crew_size(crew: int, augmentation: float) -> float:
    """Calculate effective crew size with augmentation.

    Args:
        crew: Physical crew count
        augmentation: Augmentation factor

    Returns:
        Effective crew (decision-making equivalents)
    """
    return crew * augmentation


def augmentation_energy_cost(augmentation_type: str, factor: float) -> float:
    """Calculate power required for augmentation.

    Args:
        augmentation_type: Type string
        factor: Augmentation factor

    Returns:
        Power required in watts
    """
    try:
        aug_type = AugmentationType(augmentation_type)
    except ValueError:
        return 0.0

    if aug_type == AugmentationType.HUMAN_ONLY:
        return 0.0
    elif aug_type == AugmentationType.AI_ASSISTED:
        return (factor - HUMAN_ONLY_FACTOR) * AI_POWER_PER_FACTOR_POINT
    elif aug_type == AugmentationType.NEURALINK_ASSISTED:
        return (factor - HUMAN_ONLY_FACTOR) * NEURALINK_POWER_PER_FACTOR_POINT
    elif aug_type == AugmentationType.HYBRID:
        # Split between AI and Neuralink
        ai_portion = (factor - HUMAN_ONLY_FACTOR) * 0.7
        neural_portion = (factor - HUMAN_ONLY_FACTOR) * 0.3
        return ai_portion * AI_POWER_PER_FACTOR_POINT + neural_portion * NEURALINK_POWER_PER_FACTOR_POINT

    return 0.0


def augmentation_mass_cost(augmentation_type: str, factor: float) -> float:
    """Calculate mass required for augmentation hardware.

    This is the inverse of calculate_augmentation_factor.
    Uses logarithmic scaling: full base factor achieved at 100kg.

    Args:
        augmentation_type: Type string
        factor: Augmentation factor

    Returns:
        Mass required in kg
    """
    try:
        aug_type = AugmentationType(augmentation_type)
    except ValueError:
        return 0.0

    if aug_type == AugmentationType.HUMAN_ONLY or factor <= HUMAN_ONLY_FACTOR:
        return 0.0

    # Get base factor for this type
    base_factor = get_base_factor(aug_type)

    # Inverse of logarithmic scaling from calculate_augmentation_factor
    # Original: mass_factor = log2(1 + mass) / log2(101)
    #           effective = 1.0 + (base - 1.0) * mass_factor
    # Inverse:  mass_factor = (effective - 1.0) / (base - 1.0)
    #           mass = 2^(mass_factor * log2(101)) - 1

    mass_factor = (factor - HUMAN_ONLY_FACTOR) / (base_factor - HUMAN_ONLY_FACTOR)
    mass_factor = max(0.0, min(1.0, mass_factor))  # Clamp to [0, 1]
    mass_required = 2 ** (mass_factor * math.log2(101)) - 1

    return mass_required


def get_reliability(augmentation_type: str) -> float:
    """Get reliability factor for augmentation type.

    Args:
        augmentation_type: Type string

    Returns:
        Reliability (0-1)
    """
    try:
        aug_type = AugmentationType(augmentation_type)
    except ValueError:
        return HUMAN_RELIABILITY

    reliabilities = {
        AugmentationType.HUMAN_ONLY: HUMAN_RELIABILITY,
        AugmentationType.AI_ASSISTED: AI_RELIABILITY,
        AugmentationType.NEURALINK_ASSISTED: NEURALINK_RELIABILITY,
        AugmentationType.HYBRID: AI_RELIABILITY * NEURALINK_RELIABILITY,  # Combined
    }
    return reliabilities.get(aug_type, HUMAN_RELIABILITY)


def validate_augmentation(
    crew: int,
    augmentation_type: str,
    compute_mass_kg: float,
    power_available: float,
) -> Dict:
    """Check if colony can support augmentation.

    Args:
        crew: Number of crew members
        augmentation_type: Type of augmentation
        compute_mass_kg: Available compute mass
        power_available: Available power in watts

    Returns:
        Dict with {feasible, factor, bottleneck, effective_crew, confidence}
    """
    factor = calculate_augmentation_factor(augmentation_type, compute_mass_kg)
    power_required = augmentation_energy_cost(augmentation_type, factor)
    mass_required = augmentation_mass_cost(augmentation_type, factor)
    reliability = get_reliability(augmentation_type)

    bottleneck = None
    feasible = True

    # Check power constraint
    if power_required > power_available:
        feasible = False
        bottleneck = "power"
        # Scale down factor to available power
        if augmentation_type == "ai_assisted":
            max_factor = HUMAN_ONLY_FACTOR + power_available / AI_POWER_PER_FACTOR_POINT
        elif augmentation_type == "neuralink_assisted":
            max_factor = HUMAN_ONLY_FACTOR + power_available / NEURALINK_POWER_PER_FACTOR_POINT
        else:
            max_factor = factor
        factor = min(factor, max_factor)

    # Check mass constraint
    if mass_required > compute_mass_kg:
        if not bottleneck:
            bottleneck = "mass"
        feasible = False

    effective = effective_crew_size(crew, factor)

    result = {
        "feasible": feasible,
        "factor": factor,
        "effective_crew": effective,
        "power_required": power_required,
        "mass_required": mass_required,
        "reliability": reliability,
        "bottleneck": bottleneck,
        "confidence": reliability if feasible else reliability * 0.5,
    }

    emit_receipt(
        "augmentation_receipt",
        {
            "tenant_id": TENANT_ID,
            "crew": crew,
            "augmentation_type": augmentation_type,
            "compute_mass_kg": compute_mass_kg,
            "power_available": power_available,
            "factor": factor,
            "effective_crew": effective,
            "feasible": feasible,
            "bottleneck": bottleneck,
        },
    )

    return result


def calculate_sovereignty_with_augmentation(
    crew: int,
    augmentation_type: str,
    compute_mass_kg: float,
    bandwidth_mbps: float,
    delay_s: float,
) -> Dict:
    """Calculate sovereignty with augmentation factor.

    Integrates augmentation into the sovereignty equation.

    Args:
        crew: Physical crew count
        augmentation_type: Type of augmentation
        compute_mass_kg: Available compute mass
        bandwidth_mbps: Earth communication bandwidth
        delay_s: Light delay in seconds

    Returns:
        Dict with sovereignty calculation results
    """
    # Calculate augmentation factor
    factor = calculate_augmentation_factor(augmentation_type, compute_mass_kg)
    effective = effective_crew_size(crew, factor)

    # Sovereignty calculation (from sovereignty_core.py)
    HUMAN_DECISION_RATE_BPS = 10
    internal_rate = math.log2(1 + effective * HUMAN_DECISION_RATE_BPS)
    external_rate = (bandwidth_mbps * 1e6) / (2 * delay_s)
    advantage = internal_rate - external_rate
    sovereign = advantage > 0

    # Find threshold crew (how many unaugmented crew needed for same result)
    threshold_unaugmented = int(math.ceil(effective))

    result = {
        "crew": crew,
        "augmentation_type": augmentation_type,
        "factor": factor,
        "effective_crew": effective,
        "internal_rate": internal_rate,
        "external_rate": external_rate,
        "advantage": advantage,
        "sovereign": sovereign,
        "threshold_unaugmented_crew": threshold_unaugmented,
    }

    emit_receipt(
        "augmented_sovereignty_receipt",
        {
            "tenant_id": TENANT_ID,
            **result,
        },
    )

    return result


def optimal_augmentation_mix(
    target_effective_crew: float,
    power_budget_w: float,
    mass_budget_kg: float,
) -> Dict:
    """Find optimal mix of crew and augmentation.

    Given constraints, find the best combination of physical crew
    and augmentation to achieve target effective crew.

    Args:
        target_effective_crew: Desired effective crew count
        power_budget_w: Available power
        mass_budget_kg: Available mass budget

    Returns:
        Dict with optimal configuration
    """
    best_config = None
    best_physical_crew = float("inf")

    # Try each augmentation type
    for aug_type in AugmentationType:
        aug_str = aug_type.value

        # Binary search for minimum physical crew
        for crew in range(1, int(target_effective_crew) + 1):
            factor = calculate_augmentation_factor(aug_str, mass_budget_kg)
            power = augmentation_energy_cost(aug_str, factor)

            if power > power_budget_w:
                continue

            effective = effective_crew_size(crew, factor)

            if effective >= target_effective_crew:
                if crew < best_physical_crew:
                    best_physical_crew = crew
                    best_config = {
                        "augmentation_type": aug_str,
                        "physical_crew": crew,
                        "factor": factor,
                        "effective_crew": effective,
                        "power_used": power,
                        "mass_used": augmentation_mass_cost(aug_str, factor),
                    }
                break

    if best_config is None:
        # Fallback to human-only
        best_config = {
            "augmentation_type": AugmentationType.HUMAN_ONLY.value,
            "physical_crew": int(math.ceil(target_effective_crew)),
            "factor": 1.0,
            "effective_crew": target_effective_crew,
            "power_used": 0.0,
            "mass_used": 0.0,
        }

    emit_receipt(
        "optimal_augmentation_receipt",
        {
            "tenant_id": TENANT_ID,
            "target_effective_crew": target_effective_crew,
            "power_budget_w": power_budget_w,
            "mass_budget_kg": mass_budget_kg,
            **best_config,
        },
    )

    return best_config
