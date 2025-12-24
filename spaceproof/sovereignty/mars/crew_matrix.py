"""Crew Skill Matrix Calculator.

Purpose: Calculate crew skill coverage and redundancy for 24/7 operations.

THE PHYSICS:
    A Mars colony needs 24/7 coverage of critical skills.
    With 8-hour shifts + rest requirements, this means 3-4 crew per role.
    Redundancy factor >= 2 for critical skills (someone must always be backup).
    Shannon entropy of workload measures load balancing.
"""

import math
from typing import Any

from spaceproof.core import emit_receipt

from .constants import (
    CREW_PER_SHIFT_24_7,
    MAX_WORK_HOURS_PER_WEEK,
    MIN_REDUNDANCY_CRITICAL,
    MIN_REDUNDANCY_HIGH,
    SKILL_CATEGORY_CRITICAL,
    SKILL_CATEGORY_HIGH,
    SKILL_CATEGORY_LOW,
    SKILL_CATEGORY_MEDIUM,
    TENANT_ID,
)


def define_skill_matrix() -> dict:
    """Return required skills with criticality levels.

    Returns:
        dict: Skill matrix with categories and requirements.
            Categories: CRITICAL (medical, systems), HIGH (engineering, agriculture),
            MEDIUM (science, operations), LOW (administration).
    """
    return {
        # CRITICAL: Require 24/7 coverage, redundancy >= 2
        "medical": {
            "category": "CRITICAL",
            "min_crew": CREW_PER_SHIFT_24_7,
            "min_redundancy": MIN_REDUNDANCY_CRITICAL,
            "description": "Medical diagnosis, surgery, emergency response",
        },
        "systems": {
            "category": "CRITICAL",
            "min_crew": CREW_PER_SHIFT_24_7,
            "min_redundancy": MIN_REDUNDANCY_CRITICAL,
            "description": "ECLSS, habitat pressure, thermal control",
        },
        "life_support": {
            "category": "CRITICAL",
            "min_crew": CREW_PER_SHIFT_24_7,
            "min_redundancy": MIN_REDUNDANCY_CRITICAL,
            "description": "O2, H2O, CO2 scrubbing, waste processing",
        },
        # HIGH: Require daily coverage, redundancy >= 1.5
        "engineering": {
            "category": "HIGH",
            "min_crew": 2,
            "min_redundancy": MIN_REDUNDANCY_HIGH,
            "description": "Mechanical, electrical, structural maintenance",
        },
        "agriculture": {
            "category": "HIGH",
            "min_crew": 2,
            "min_redundancy": MIN_REDUNDANCY_HIGH,
            "description": "Food production, hydroponics, soil science",
        },
        "power": {
            "category": "HIGH",
            "min_crew": 2,
            "min_redundancy": MIN_REDUNDANCY_HIGH,
            "description": "Solar arrays, nuclear reactors, batteries",
        },
        # MEDIUM: Regular shifts, redundancy >= 1.0
        "science": {
            "category": "MEDIUM",
            "min_crew": 1,
            "min_redundancy": 1.0,
            "description": "Geology, biology, chemistry research",
        },
        "operations": {
            "category": "MEDIUM",
            "min_crew": 1,
            "min_redundancy": 1.0,
            "description": "EVA coordination, logistics, scheduling",
        },
        "communications": {
            "category": "MEDIUM",
            "min_crew": 1,
            "min_redundancy": 1.0,
            "description": "Earth comms, local network, data management",
        },
        # LOW: Part-time acceptable
        "administration": {
            "category": "LOW",
            "min_crew": 1,
            "min_redundancy": 0.5,
            "description": "Resource tracking, governance, records",
        },
        "training": {
            "category": "LOW",
            "min_crew": 0,
            "min_redundancy": 0.5,
            "description": "Cross-training, skill development",
        },
    }


def calculate_coverage(crew: list[dict], skills: dict | None = None) -> float:
    """Calculate coverage ratio (0.0-1.0).

    1.0 = all critical skills covered 24/7 with redundancy.
    Uses round-robin scheduling + rest requirements.

    Args:
        crew: List of crew member dicts with 'skills' key containing
              skill: proficiency pairs (0.0-1.0)
        skills: Skill matrix (uses default if None)

    Returns:
        float: Coverage ratio where 1.0 = full coverage.
    """
    if skills is None:
        skills = define_skill_matrix()

    if not crew:
        return 0.0

    total_required = 0.0
    total_covered = 0.0

    for skill_name, skill_info in skills.items():
        category = skill_info["category"]
        min_crew = skill_info["min_crew"]

        # Weight by category importance
        weight = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}.get(category, 1.0)

        # Count qualified crew for this skill
        qualified = 0.0
        for member in crew:
            member_skills = member.get("skills", {})
            proficiency = member_skills.get(skill_name, 0.0)
            qualified += proficiency

        required = min_crew * weight
        covered = min(qualified * weight, required)

        total_required += required
        total_covered += covered

    return total_covered / total_required if total_required > 0 else 0.0


def calculate_redundancy(crew: list[dict], skills: dict | None = None) -> dict:
    """Calculate redundancy factor per skill.

    Critical skills require >= 2 qualified crew.
    HIGH requires >= 1.5 average.

    Args:
        crew: List of crew member dicts with 'skills' key
        skills: Skill matrix (uses default if None)

    Returns:
        dict: Redundancy factor per skill.
    """
    if skills is None:
        skills = define_skill_matrix()

    redundancy = {}

    for skill_name, skill_info in skills.items():
        # Count fully qualified crew (proficiency >= 0.7)
        fully_qualified = 0
        partially_qualified = 0

        for member in crew:
            member_skills = member.get("skills", {})
            proficiency = member_skills.get(skill_name, 0.0)
            if proficiency >= 0.7:
                fully_qualified += 1
            elif proficiency >= 0.3:
                partially_qualified += 1

        # Redundancy = fully qualified + 0.5 * partial
        redundancy[skill_name] = fully_qualified + 0.5 * partially_qualified

    return redundancy


def identify_gaps(crew: list[dict], skills: dict | None = None) -> list[dict]:
    """Identify uncovered skills or under-redundant critical skills.

    Args:
        crew: List of crew member dicts with 'skills' key
        skills: Skill matrix (uses default if None)

    Returns:
        list: List of gap dicts with skill name, required, actual, gap type.
              Empty list = no gaps.
    """
    if skills is None:
        skills = define_skill_matrix()

    redundancy = calculate_redundancy(crew, skills)
    gaps = []

    for skill_name, skill_info in skills.items():
        required_redundancy = skill_info["min_redundancy"]
        actual_redundancy = redundancy.get(skill_name, 0.0)

        if actual_redundancy < required_redundancy:
            category = skill_info["category"]
            gap_severity = "CRITICAL" if category == "CRITICAL" else "WARNING" if category == "HIGH" else "INFO"

            gaps.append(
                {
                    "skill": skill_name,
                    "category": category,
                    "required_redundancy": required_redundancy,
                    "actual_redundancy": actual_redundancy,
                    "deficit": required_redundancy - actual_redundancy,
                    "severity": gap_severity,
                }
            )

    # Sort by severity (CRITICAL first)
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    gaps.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return gaps


def compute_crew_entropy(crew: list[dict], schedule: dict | None = None) -> float:
    """Compute Shannon entropy of crew workload distribution.

    Lower entropy = better load balancing.
    High entropy = overloaded individuals (failure risk).

    Args:
        crew: List of crew member dicts with 'workload_hours' key
        schedule: Optional schedule dict (for future extensions)

    Returns:
        float: Shannon entropy of workload distribution.
               0.0 = perfectly balanced, higher = more unbalanced.
    """
    if not crew:
        return 0.0

    # Get workloads, default to equal distribution
    workloads = []
    for member in crew:
        hours = member.get("workload_hours", MAX_WORK_HOURS_PER_WEEK / len(crew))
        workloads.append(max(hours, 0.01))  # Avoid zero

    total = sum(workloads)
    if total <= 0:
        return 0.0

    # Normalize to probabilities
    probs = [w / total for w in workloads]

    # Calculate Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Normalize by max entropy (uniform distribution)
    max_entropy = math.log2(len(crew)) if len(crew) > 1 else 1.0

    # Invert: 0 = balanced (high entropy), 1 = unbalanced (low entropy)
    normalized = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    return normalized


def emit_crew_coverage_receipt(
    crew: list[dict],
    coverage: float,
    redundancy: dict,
    gaps: list[dict],
    entropy: float,
) -> dict:
    """Emit crew coverage receipt.

    Args:
        crew: Crew configuration
        coverage: Coverage ratio
        redundancy: Redundancy factors
        gaps: Identified gaps
        entropy: Workload entropy

    Returns:
        dict: Emitted receipt.
    """
    return emit_receipt(
        "crew_coverage",
        {
            "tenant_id": TENANT_ID,
            "crew_count": len(crew),
            "coverage_ratio": coverage,
            "redundancy": redundancy,
            "gaps_count": len(gaps),
            "gaps": gaps[:5] if gaps else [],  # Limit to top 5
            "workload_entropy": entropy,
            "critical_gaps": len([g for g in gaps if g["severity"] == "CRITICAL"]),
        },
    )
