"""darwinian_enforce.py - Core Darwinian selection and law enforcement.

THE DARWINIAN INSIGHT:
    The receipt chain is a Darwinian replicator.
    High-compression paths replicate (amplified x2).
    Low-compression paths die (starved after 5 cycles).
    Real entropy applies selection pressure.
    Survivors after 10 generations become IMPOSED LAWS.

    Physics emerges not from observation but from survival.
    Laws are not discovered - they are the survivors of selection.

Source: AXIOM v2 - Darwinian Causal Enforcer
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .core import dual_hash, emit_receipt, StopRule

# === CONSTANTS ===

TENANT_ID = "axiom-darwinian"
"""Tenant for Darwinian enforcement receipts."""

# Selection parameters (from darwinian_spec.json)
AMPLIFY_FACTOR_HIGH_COMPRESSION = 2.0
"""Duplication weight for winners - paths with score >= 0.9 get 2x replication."""

STARVE_THRESHOLD_LOW_COMPRESSION = 0.8
"""Below this score = marked for death. Paths with score < 0.8 get starved."""

HIGH_COMPRESSION_THRESHOLD = 0.9
"""Above this score = amplify. Paths with score >= 0.9 get amplified."""

DARWINIAN_GENERATIONS_PER_CYCLE = 10
"""Cycles before law crystallization. Patterns must survive 10 generations."""

MAX_SURVIVAL_CYCLES = 5
"""Low-compression paths die after this many cycles of starvation."""

# Spec file path
SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "darwinian_spec.json"
)

# Storage for active laws
_active_laws: List[Dict] = []
_generation_tracker: Dict[str, int] = {}


# === SPEC LOADING ===


def load_spec() -> Dict:
    """Load darwinian_spec.json with dual-hash validation.

    Returns:
        Dict with spec data and _hash field containing dual-hash.

    Emits:
        darwinian_spec_load_receipt

    Raises:
        StopRule: If spec file not found or validation fails.
    """
    if not os.path.exists(SPEC_PATH):
        emit_receipt(
            "anomaly",
            {
                "tenant_id": TENANT_ID,
                "metric": "spec_not_found",
                "baseline": 0.0,
                "delta": -1.0,
                "classification": "violation",
                "action": "halt",
            },
        )
        raise StopRule(f"Darwinian spec not found: {SPEC_PATH}")

    with open(SPEC_PATH, "r") as f:
        content = f.read()
        spec = json.loads(content)

    # Compute dual-hash of spec content
    spec_hash = dual_hash(content)
    spec["_hash"] = spec_hash

    # Emit load receipt
    emit_receipt(
        "darwinian_spec_load",
        {
            "tenant_id": TENANT_ID,
            "version": spec.get("version", "unknown"),
            "spec_hash": spec_hash,
            "amplify_factor": spec.get("amplify_factor", AMPLIFY_FACTOR_HIGH_COMPRESSION),
            "starve_threshold": spec.get("starve_threshold", STARVE_THRESHOLD_LOW_COMPRESSION),
            "high_compression_threshold": spec.get("high_compression_threshold", HIGH_COMPRESSION_THRESHOLD),
        },
    )

    return spec


# === COMPRESSION SCORING ===


def score_compression(receipt: Dict) -> float:
    """Extract compression score from receipt.

    Args:
        receipt: Receipt dict, may contain 'compression', 'compression_ratio',
                 'score', or other compression-related fields.

    Returns:
        Compression score normalized to [0, 1].

    Emits:
        anomaly_receipt if score outside valid range.
    """
    # Try various field names for compression score
    score = None
    for field_name in ["compression", "compression_ratio", "score", "r_squared", "fitness"]:
        if field_name in receipt:
            score = receipt[field_name]
            break

    if score is None:
        # Default to neutral if no compression info
        score = 0.85

    # Validate range
    if not isinstance(score, (int, float)):
        score = 0.85

    score = float(score)

    # Stoprule for invalid compression
    if score < 0 or score > 1:
        stoprule_invalid_compression(score, receipt)

    return score


def stoprule_invalid_compression(score: float, receipt: Dict) -> None:
    """Handle invalid compression score.

    Args:
        score: The invalid score value.
        receipt: The receipt containing the invalid score.

    Emits:
        anomaly_receipt with invalid compression details.

    Raises:
        StopRule: Always - invalid compression is a fatal error.
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": TENANT_ID,
            "metric": "invalid_compression",
            "baseline": 0.5,
            "delta": score - 0.5,
            "classification": "violation",
            "action": "halt",
            "invalid_score": score,
            "receipt_id": receipt.get("id", "unknown"),
        },
    )
    raise StopRule(f"Invalid compression score: {score} (must be in [0, 1])")


# === FITNESS CLASSIFICATION ===


def classify_fitness(score: float) -> str:
    """Classify receipt fitness based on compression score.

    Args:
        score: Compression score in [0, 1].

    Returns:
        "amplify" if score >= 0.9 (high compression = replicate)
        "neutral" if 0.8 <= score < 0.9 (survive but don't replicate)
        "starve" if score < 0.8 (marked for death)
    """
    if score >= HIGH_COMPRESSION_THRESHOLD:
        return "amplify"
    elif score >= STARVE_THRESHOLD_LOW_COMPRESSION:
        return "neutral"
    else:
        return "starve"


# === PATH AMPLIFICATION ===


def amplify_path(receipt: Dict, factor: float = AMPLIFY_FACTOR_HIGH_COMPRESSION) -> List[Dict]:
    """Duplicate receipt with amplification weight.

    High-compression paths get replicated. This is Darwinian selection
    in action: winners reproduce.

    Args:
        receipt: Receipt to amplify.
        factor: Amplification factor (default 2.0 = duplicate).

    Returns:
        List of amplified receipts (original + duplicates).

    Emits:
        path_amplification_receipt for each copy.
    """
    # Create copies based on factor
    copies = []
    num_copies = int(factor)

    for i in range(num_copies):
        copy = receipt.copy()
        copy["_amplified"] = True
        copy["_amplification_factor"] = factor
        copy["_copy_index"] = i
        copy["_amplification_id"] = str(uuid.uuid4())[:8]

        # Track generation
        receipt_id = copy.get("id", str(uuid.uuid4())[:8])
        _generation_tracker[receipt_id] = _generation_tracker.get(receipt_id, 0) + 1
        copy["_generation"] = _generation_tracker[receipt_id]

        copies.append(copy)

    # Emit amplification receipt
    emit_receipt(
        "path_amplification",
        {
            "tenant_id": TENANT_ID,
            "original_id": receipt.get("id", "unknown"),
            "num_copies": num_copies,
            "factor": factor,
            "compression_score": score_compression(receipt),
            "generation": _generation_tracker.get(receipt.get("id", ""), 0),
        },
    )

    return copies


# === PATH STARVATION ===


def get_generation(receipt: Dict) -> int:
    """Track how many cycles a receipt path has survived.

    Args:
        receipt: Receipt to check.

    Returns:
        Generation count (number of cycles survived).
    """
    if "_generation" in receipt:
        return receipt["_generation"]

    receipt_id = receipt.get("id", str(uuid.uuid4())[:8])
    return _generation_tracker.get(receipt_id, 0)


def starve_path(receipt: Dict, generation: int = None) -> Optional[Dict]:
    """Mark receipt for pruning, return None if dead.

    Low-compression paths get starved. After MAX_SURVIVAL_CYCLES (5),
    they die completely.

    Args:
        receipt: Receipt to potentially starve.
        generation: Optional explicit generation count.

    Returns:
        Marked receipt if still alive, None if dead.

    Emits:
        path_starvation_receipt.
    """
    if generation is None:
        generation = get_generation(receipt)

    # Check if this path should die
    if generation > MAX_SURVIVAL_CYCLES:
        emit_receipt(
            "path_starvation",
            {
                "tenant_id": TENANT_ID,
                "receipt_id": receipt.get("id", "unknown"),
                "generation": generation,
                "status": "dead",
                "compression_score": score_compression(receipt),
                "cause": f"exceeded max_survival_cycles ({MAX_SURVIVAL_CYCLES})",
            },
        )
        return None

    # Mark for starvation but still alive
    starved = receipt.copy()
    starved["_starved"] = True
    starved["_starvation_generation"] = generation
    starved["_survival_remaining"] = MAX_SURVIVAL_CYCLES - generation

    emit_receipt(
        "path_starvation",
        {
            "tenant_id": TENANT_ID,
            "receipt_id": receipt.get("id", "unknown"),
            "generation": generation,
            "status": "starving",
            "survival_remaining": MAX_SURVIVAL_CYCLES - generation,
            "compression_score": score_compression(receipt),
        },
    )

    return starved


# === SELECTION CYCLE ===


def run_selection_cycle(receipts: List[Dict]) -> List[Dict]:
    """Apply amplify/starve to population, return survivors.

    This is the core Darwinian loop:
    1. Score each receipt's compression
    2. Classify fitness (amplify/neutral/starve)
    3. Amplify high-compression paths (x2 replication)
    4. Starve low-compression paths (die after 5 cycles)
    5. Return survivors

    Args:
        receipts: List of receipts to process.

    Returns:
        List of surviving (and amplified) receipts.

    Emits:
        selection_cycle_receipt with cycle statistics.
    """
    if not receipts:
        return []

    survivors = []
    amplified_count = 0
    starved_count = 0
    dead_count = 0
    neutral_count = 0

    for receipt in receipts:
        score = score_compression(receipt)
        fitness = classify_fitness(score)

        if fitness == "amplify":
            # High-compression: replicate
            amplified = amplify_path(receipt)
            survivors.extend(amplified)
            amplified_count += 1
        elif fitness == "neutral":
            # Neutral: survive but don't replicate
            survivors.append(receipt)
            neutral_count += 1
        else:
            # Starve: may die
            generation = get_generation(receipt) + 1
            result = starve_path(receipt, generation)
            if result is not None:
                survivors.append(result)
                starved_count += 1
            else:
                dead_count += 1

    # Check for amplification overflow
    stoprule_amplification_overflow(len(survivors), len(receipts))

    # Emit cycle receipt
    emit_receipt(
        "selection_cycle",
        {
            "tenant_id": TENANT_ID,
            "input_count": len(receipts),
            "survivor_count": len(survivors),
            "amplified_count": amplified_count,
            "neutral_count": neutral_count,
            "starved_count": starved_count,
            "dead_count": dead_count,
            "avg_input_score": sum(score_compression(r) for r in receipts) / len(receipts) if receipts else 0,
            "avg_survivor_score": sum(score_compression(r) for r in survivors) / len(survivors) if survivors else 0,
        },
    )

    return survivors


def stoprule_amplification_overflow(population_size: int, original_size: int, max_ratio: float = 10.0) -> None:
    """Check for population explosion from amplification.

    Args:
        population_size: Current population size.
        original_size: Original population size.
        max_ratio: Maximum allowed population ratio.

    Raises:
        StopRule: If population exceeds limit (applies emergency pressure).
    """
    if original_size == 0:
        return

    ratio = population_size / original_size
    if ratio > max_ratio:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": TENANT_ID,
                "metric": "amplification_overflow",
                "baseline": 1.0,
                "delta": ratio - 1.0,
                "classification": "deviation",
                "action": "escalate",
                "population_size": population_size,
                "original_size": original_size,
                "ratio": ratio,
                "max_ratio": max_ratio,
            },
        )
        raise StopRule(f"Amplification overflow: population ratio {ratio:.2f} exceeds max {max_ratio}")


# === LAW IMPOSITION (GATE 3) ===


def extract_surviving_pattern(survivors: List[Dict]) -> Dict:
    """Find common pattern in high-compression survivors.

    Args:
        survivors: List of receipts that survived selection.

    Returns:
        Dict with common pattern keys and values.

    Emits:
        Warning if no common pattern found.
    """
    if not survivors:
        stoprule_pattern_extraction_failed("No survivors to extract pattern from")
        return {}

    # Find receipts that have been amplified (high-compression winners)
    winners = [r for r in survivors if r.get("_amplified", False)]
    if not winners:
        winners = survivors

    # Extract common keys and values
    pattern = {}
    if winners:
        # Get keys present in all winners
        common_keys = set(winners[0].keys())
        for w in winners[1:]:
            common_keys &= set(w.keys())

        # Filter out internal keys
        common_keys = {k for k in common_keys if not k.startswith("_")}

        # Extract values that are consistent across all winners
        for key in common_keys:
            values = [w[key] for w in winners if key in w]
            if len(set(str(v) for v in values)) == 1:
                pattern[key] = values[0]

    if not pattern:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": TENANT_ID,
                "metric": "weak_pattern",
                "baseline": 1.0,
                "delta": 0.0,
                "classification": "deviation",
                "action": "alert",
                "survivor_count": len(survivors),
            },
        )

    return pattern


def stoprule_pattern_extraction_failed(reason: str) -> None:
    """Handle failed pattern extraction.

    Args:
        reason: Description of why extraction failed.

    Emits:
        Warning receipt (does not raise - allows recovery).
    """
    emit_receipt(
        "anomaly",
        {
            "tenant_id": TENANT_ID,
            "metric": "pattern_extraction_failed",
            "baseline": 0.0,
            "delta": -1.0,
            "classification": "deviation",
            "action": "alert",
            "reason": reason,
        },
    )


def is_law_candidate(pattern: Dict, generations: int) -> bool:
    """Check if pattern has survived enough generations to become law.

    Args:
        pattern: Extracted pattern from survivors.
        generations: Number of generations pattern has survived.

    Returns:
        True if generations >= DARWINIAN_GENERATIONS_PER_CYCLE (10).
    """
    return generations >= DARWINIAN_GENERATIONS_PER_CYCLE


def impose_law(pattern: Dict) -> str:
    """Crystallize pattern as law.

    Patterns that survive 10+ generations become IMPOSED laws.
    Future receipts must conform or be rejected.

    Args:
        pattern: Pattern to crystallize as law.

    Returns:
        Law ID.

    Emits:
        darwinian_law_receipt, law_imposition_receipt.
    """
    law_id = f"law_{str(uuid.uuid4())[:8]}"

    law = {
        "id": law_id,
        "pattern": pattern,
        "imposed_at": emit_receipt.__module__,  # Timestamp will be in receipt
        "generation": DARWINIAN_GENERATIONS_PER_CYCLE,
    }

    _active_laws.append(law)

    # Emit law receipts
    emit_receipt(
        "darwinian_law",
        {
            "tenant_id": TENANT_ID,
            "law_id": law_id,
            "pattern": pattern,
            "status": "crystallized",
            "generations_survived": DARWINIAN_GENERATIONS_PER_CYCLE,
        },
    )

    emit_receipt(
        "law_imposition",
        {
            "tenant_id": TENANT_ID,
            "law_id": law_id,
            "pattern_keys": list(pattern.keys()),
            "active_laws_count": len(_active_laws),
        },
    )

    return law_id


def get_active_laws() -> List[Dict]:
    """Return currently imposed laws.

    Returns:
        List of active law dicts.
    """
    return _active_laws.copy()


def validate_against_laws(receipt: Dict, laws: List[Dict] = None) -> bool:
    """Check receipt conforms to all active laws.

    Args:
        receipt: Receipt to validate.
        laws: Optional list of laws (uses _active_laws if None).

    Returns:
        True if receipt conforms to all laws.
    """
    if laws is None:
        laws = _active_laws

    for law in laws:
        pattern = law.get("pattern", {})
        for key, expected_value in pattern.items():
            if key in receipt:
                # Check if value matches (string comparison for safety)
                if str(receipt[key]) != str(expected_value):
                    return False

    return True


def enforce_laws(receipt: Dict) -> Optional[Dict]:
    """Apply laws, return None if violation.

    Args:
        receipt: Receipt to check.

    Returns:
        Receipt if it conforms, None if it violates.

    Emits:
        law_violation_receipt if receipt violates a law.
    """
    if not _active_laws:
        return receipt

    if validate_against_laws(receipt):
        return receipt

    # Find which law was violated
    for law in _active_laws:
        pattern = law.get("pattern", {})
        for key, expected_value in pattern.items():
            if key in receipt and str(receipt[key]) != str(expected_value):
                stoprule_law_violation(receipt, law, key, expected_value, receipt[key])
                return None

    return None


def stoprule_law_violation(
    receipt: Dict, law: Dict, key: str, expected: Any, actual: Any
) -> None:
    """Handle law violation.

    Args:
        receipt: Violating receipt.
        law: Law that was violated.
        key: Key that violated.
        expected: Expected value.
        actual: Actual value.

    Emits:
        law_violation_receipt.
    """
    emit_receipt(
        "law_violation",
        {
            "tenant_id": TENANT_ID,
            "receipt_id": receipt.get("id", "unknown"),
            "law_id": law.get("id", "unknown"),
            "violated_key": key,
            "expected_value": str(expected),
            "actual_value": str(actual),
            "action": "reject",
        },
    )


# === UTILITY FUNCTIONS ===


def reset_darwinian_state() -> None:
    """Reset all Darwinian state (for testing)."""
    global _active_laws, _generation_tracker
    _active_laws = []
    _generation_tracker = {}


def get_darwinian_info() -> Dict:
    """Get current Darwinian enforcer status.

    Returns:
        Dict with current configuration and state.
    """
    return {
        "amplify_factor": AMPLIFY_FACTOR_HIGH_COMPRESSION,
        "starve_threshold": STARVE_THRESHOLD_LOW_COMPRESSION,
        "high_compression_threshold": HIGH_COMPRESSION_THRESHOLD,
        "generations_per_cycle": DARWINIAN_GENERATIONS_PER_CYCLE,
        "max_survival_cycles": MAX_SURVIVAL_CYCLES,
        "active_laws_count": len(_active_laws),
        "tracked_generations": len(_generation_tracker),
        "paradigm": "Laws causally enforced by selective receipt amplification",
        "insight": "The receipt chain is a Darwinian replicator",
    }
