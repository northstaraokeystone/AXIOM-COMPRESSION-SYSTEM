"""Mars Sovereignty Integrator.

Purpose: Combine all subsystems into unified sovereignty score (0-100%).

THE PHYSICS:
    Decision capacity weighted highest (0.35) because:
    1. Latency is irreducible: Physics mandates 3-22 min delay
    2. Conjunction is deterministic: 14-day blackout every 780 days
    3. ECLSS failures are stochastic: ISS averages 3-4 critical/year
    4. Decision paralysis is fatal: Cannot wait 22 min for depressurization

    The sovereignty threshold exists because of information theory,
    not resource scarcity.
"""

from typing import Any

from spaceproof.core import emit_receipt

from .constants import (
    CREW_MIN_GEORGE_MASON,
    CREW_MIN_SALOTTI,
    DEFAULT_WEIGHTS,
    RESEARCH_VALIDATION_TOLERANCE,
    SOVEREIGNTY_SCORE_MAX,
    SOVEREIGNTY_SCORE_MIN,
    TENANT_ID,
)


def calculate_sovereignty_score(
    crew_coverage: float,
    life_support_entropy: float,
    decision_capacity: float,
    resource_closure: float,
    weights: dict | None = None,
) -> float:
    """Calculate unified sovereignty score (0-100%).

    Formula: weighted sum of 4 subsystems.
    Default weights: {crew: 0.25, life_support: 0.30, decision: 0.35, resources: 0.10}
    Decision capacity weighted highest (THE NOVEL DIMENSION).

    Args:
        crew_coverage: Crew skill coverage ratio (0-1)
        life_support_entropy: Life support entropy rate (-1 to +1, negative is good)
        decision_capacity: Decision sovereignty ratio (internal/external, unbounded)
        resource_closure: ISRU closure ratio (0-1)
        weights: Optional custom weights dict

    Returns:
        float: Sovereignty score (0-100%).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Normalize inputs to 0-1 range

    # Crew coverage is already 0-1
    crew_score = max(0.0, min(1.0, crew_coverage))

    # Life support entropy: -1 (good) to +1 (bad), transform to 0-1
    # -1 -> 1.0, 0 -> 0.5, +1 -> 0.0
    life_support_score = max(0.0, min(1.0, 0.5 - life_support_entropy / 2))

    # Decision capacity: ratio of internal/external
    # >= 1 means sovereign, use sigmoid to normalize
    # decision_capacity can be 0 to inf
    import math

    if decision_capacity >= 1.0:
        # Already sovereign, score based on how much margin
        decision_score = 0.5 + 0.5 * (1 - 1 / (1 + decision_capacity))
    else:
        # Not sovereign, linear ramp
        decision_score = 0.5 * decision_capacity

    # Resource closure is already 0-1
    resource_score = max(0.0, min(1.0, resource_closure))

    # Weighted combination
    total_weight = (
        weights.get("crew", 0.25)
        + weights.get("life_support", 0.30)
        + weights.get("decision", 0.35)
        + weights.get("resources", 0.10)
    )

    if total_weight <= 0:
        return 0.0

    score = (
        crew_score * weights.get("crew", 0.25)
        + life_support_score * weights.get("life_support", 0.30)
        + decision_score * weights.get("decision", 0.35)
        + resource_score * weights.get("resources", 0.10)
    ) / total_weight

    # Convert to percentage
    return score * 100.0


def identify_binding_constraint(
    scores: dict,
) -> str:
    """Identify which subsystem limits overall score.

    Args:
        scores: Dict with subsystem scores {crew, life_support, decision, resources}

    Returns:
        str: Name of binding constraint (lowest scoring subsystem).
    """
    if not scores:
        return "unknown"

    # Find minimum score
    binding = min(scores.items(), key=lambda x: x[1])
    return binding[0]


def generate_failure_tree(
    config: dict,
    scores: dict,
) -> dict:
    """Generate failure mode tree.

    Each subsystem has failure probabilities.
    Cascades are modeled (ECLSS failure -> O2 crisis -> emergency decisions -> overload).

    Args:
        config: Colony configuration
        scores: Subsystem scores

    Returns:
        dict: Failure mode tree with probabilities and cascades.
    """
    tree = {
        "root": "colony_failure",
        "probability": 0.0,
        "children": [],
    }

    # Life support failures
    life_support_node = {
        "name": "life_support_failure",
        "probability": max(0.0, 1.0 - scores.get("life_support", 0.5)),
        "children": [
            {
                "name": "o2_crisis",
                "probability": 0.3,
                "cascade_to": "emergency_decisions",
            },
            {
                "name": "h2o_crisis",
                "probability": 0.2,
                "cascade_to": "resource_depletion",
            },
            {
                "name": "thermal_runaway",
                "probability": 0.1,
                "cascade_to": "hab_evacuation",
            },
        ],
    }

    # Decision capacity failures
    decision_node = {
        "name": "decision_failure",
        "probability": max(0.0, 1.0 - scores.get("decision", 0.5)),
        "children": [
            {
                "name": "delayed_response",
                "probability": 0.4,
                "cascade_to": "cascading_failure",
            },
            {
                "name": "information_overload",
                "probability": 0.3,
                "cascade_to": "wrong_decision",
            },
        ],
    }

    # Crew failures
    crew_node = {
        "name": "crew_failure",
        "probability": max(0.0, 1.0 - scores.get("crew", 0.5)),
        "children": [
            {
                "name": "skill_gap",
                "probability": 0.5,
                "cascade_to": "inability_to_repair",
            },
            {
                "name": "crew_incapacitation",
                "probability": 0.2,
                "cascade_to": "workload_overload",
            },
        ],
    }

    # Resource failures
    resource_node = {
        "name": "resource_failure",
        "probability": max(0.0, 1.0 - scores.get("resources", 0.5)),
        "children": [
            {
                "name": "supply_depletion",
                "probability": 0.4,
                "cascade_to": "rationing",
            },
            {
                "name": "isru_breakdown",
                "probability": 0.3,
                "cascade_to": "dependency_on_resupply",
            },
        ],
    }

    tree["children"] = [life_support_node, decision_node, crew_node, resource_node]

    # Calculate overall failure probability (OR of all subsystems)
    probs = [child["probability"] for child in tree["children"]]
    # P(any failure) = 1 - P(all succeed)
    p_all_succeed = 1.0
    for p in probs:
        p_all_succeed *= 1.0 - p
    tree["probability"] = 1.0 - p_all_succeed

    return tree


def compute_crew_size_threshold(
    target_score: float,
    config_template: dict,
    min_crew: int = 4,
    max_crew: int = 200,
) -> int:
    """Binary search for minimum crew achieving target score.

    This answers: "How many people for 95% sovereignty?"

    Args:
        target_score: Target sovereignty score (0-100)
        config_template: Base configuration template
        min_crew: Minimum crew to search
        max_crew: Maximum crew to search

    Returns:
        int: Minimum crew size achieving target score.
    """
    from .api import evaluate_config_with_crew

    low, high = min_crew, max_crew

    while low < high:
        mid = (low + high) // 2
        score = evaluate_config_with_crew(config_template, mid)

        if score >= target_score:
            high = mid
        else:
            low = mid + 1

    return low


def validate_against_research(
    crew: int,
    score: float,
) -> bool:
    """Validate output against research benchmarks.

    22 crew (George Mason) should yield ~95% score.
    110 crew (Salotti) should yield ~99.9% score.
    Returns True if within 10% of expected.

    Args:
        crew: Crew size
        score: Calculated sovereignty score

    Returns:
        bool: True if score matches research expectations.
    """
    # Expected scores based on research
    if crew <= CREW_MIN_GEORGE_MASON:
        # 22 crew -> ~95% expected
        expected = 95.0 * (crew / CREW_MIN_GEORGE_MASON)
    elif crew <= CREW_MIN_SALOTTI:
        # Linear interpolation from 95% to 99.9%
        progress = (crew - CREW_MIN_GEORGE_MASON) / (CREW_MIN_SALOTTI - CREW_MIN_GEORGE_MASON)
        expected = 95.0 + progress * 4.9
    else:
        # Above 110 crew -> approaching 100%
        expected = 99.9

    # Check if within tolerance
    tolerance = expected * RESEARCH_VALIDATION_TOLERANCE

    return abs(score - expected) <= tolerance


def calculate_comprehensive_sovereignty(
    crew_metrics: dict,
    life_support_metrics: dict,
    decision_metrics: dict,
    resource_metrics: dict,
    weights: dict | None = None,
) -> dict:
    """Calculate comprehensive sovereignty including all metrics.

    Args:
        crew_metrics: Crew coverage metrics
        life_support_metrics: Life support metrics
        decision_metrics: Decision capacity metrics
        resource_metrics: Resource metrics
        weights: Optional custom weights

    Returns:
        dict: Comprehensive sovereignty result.
    """
    # Extract scores from metrics
    crew_coverage = crew_metrics.get("coverage", 0.5)
    life_support_entropy = life_support_metrics.get("entropy_rate", 0.0)
    decision_ratio = decision_metrics.get("advantage_ratio", 1.0)
    resource_closure = resource_metrics.get("closure_ratio", 0.5)

    score = calculate_sovereignty_score(
        crew_coverage=crew_coverage,
        life_support_entropy=life_support_entropy,
        decision_capacity=decision_ratio,
        resource_closure=resource_closure,
        weights=weights,
    )

    # Component scores
    scores = {
        "crew": crew_coverage,
        "life_support": 0.5 - life_support_entropy / 2,
        "decision": min(decision_ratio, 2.0) / 2,  # Normalize
        "resources": resource_closure,
    }

    binding = identify_binding_constraint(scores)
    failure_tree = generate_failure_tree({}, scores)

    # Research validation
    crew_count = crew_metrics.get("crew_count", 22)
    validated = validate_against_research(crew_count, score)

    return {
        "sovereignty_score": score,
        "component_scores": scores,
        "binding_constraint": binding,
        "failure_tree": failure_tree,
        "research_validated": validated,
        "is_sovereign": decision_metrics.get("sovereign", False),
        "conjunction_survival": decision_metrics.get("conjunction_survival_probability", 0.0),
    }


def emit_mars_sovereignty_receipt(
    crew_count: int,
    sovereignty_result: dict,
    config: dict | None = None,
) -> dict:
    """Emit Mars sovereignty receipt.

    Args:
        crew_count: Number of crew
        sovereignty_result: Comprehensive sovereignty result
        config: Optional configuration

    Returns:
        dict: Emitted receipt.
    """
    return emit_receipt(
        "mars_sovereignty",
        {
            "tenant_id": TENANT_ID,
            "crew_count": crew_count,
            "sovereignty_score": sovereignty_result["sovereignty_score"],
            "is_sovereign": sovereignty_result["is_sovereign"],
            "binding_constraint": sovereignty_result["binding_constraint"],
            "conjunction_survival": sovereignty_result["conjunction_survival"],
            "research_validated": sovereignty_result["research_validated"],
            "component_scores": sovereignty_result["component_scores"],
        },
    )
