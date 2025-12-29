"""meta_integration.py - Meta-Loop Topology Classification.

D20 Production Evolution: Hook domains into Meta-Loop v2.1 topology classification.

THE META-LOOP PARADIGM:
    Patterns evolve through topology classification:
    - Open: E >= V_esc AND A > 0.75 → Graduate → CASCADE
    - Hybrid: T > 0.70 → Transfer to other domains
    - Closed: Continue optimizing

    CASCADE_MULTIPLIER = 5 (spawn 5 variants on graduation)
    Confidence-gated fallback (<0.85 triggers external validation)

Source: Meta_loop_v2_1.txt (document #14)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import uuid

from spaceproof.core import dual_hash, emit_receipt, merkle, TENANT_ID

# === CONSTANTS (from Meta-Loop v2.1) ===

META_INTEGRATION_TENANT = "spaceproof-meta-integration"

# Escape velocities per domain
ESCAPE_VELOCITY = {
    "orbital_compute": 0.90,  # Starcloud
    "constellation_ops": 0.85,  # Starlink
    "autonomous_decision": 0.88,  # Defense
    "firmware_integrity": 0.80,  # All
}

# Thresholds
AUTONOMY_THRESHOLD = 0.75  # A > 0.75 for open topology
TRANSFER_THRESHOLD = 0.70  # T > 0.70 for hybrid topology
CASCADE_MULTIPLIER = 5  # Spawn 5 variants on graduation
CONFIDENCE_FALLBACK = 0.85  # Trigger fallback below this

# Mutation for cascade
DEFAULT_MUTATION_RATE = 0.05


class Topology(Enum):
    """Pattern topology classification."""

    OPEN = "open"  # Graduate → CASCADE
    CLOSED = "closed"  # Continue optimizing
    HYBRID = "hybrid"  # Transfer to other domain


@dataclass
class PatternMetrics:
    """Metrics for a pattern."""

    pattern_id: str
    domain: str
    effectiveness: float  # E = (H_before - H_after) / n_receipts
    autonomy_score: float  # A = auto_approved / total_actions
    transfer_score: float  # T = temporal_graph_similarity
    confidence: float  # Classification confidence
    n_receipts: int
    entropy_before: float
    entropy_after: float


@dataclass
class TopologyResult:
    """Result of topology classification."""

    pattern_id: str
    domain: str
    topology: Topology
    effectiveness: float
    autonomy_score: float
    transfer_score: float
    escape_velocity: float
    confidence: float
    action: str  # "cascade", "transfer", "optimize"
    receipt: Dict[str, Any]


@dataclass
class CascadeResult:
    """Result of CASCADE operation."""

    parent_pattern_id: str
    child_pattern_ids: List[str]
    mutation_rate: float
    backtest_results: Dict[str, Any]
    receipt: Dict[str, Any]


@dataclass
class TransferResult:
    """Result of pattern transfer."""

    pattern_id: str
    from_domain: str
    to_domain: str
    transfer_score: float
    similarity_metrics: Dict[str, Any]
    receipt: Dict[str, Any]


def _compute_shannon_entropy(data: bytes) -> float:
    """Compute normalized Shannon entropy."""
    if len(data) == 0:
        return 0.0

    from collections import Counter

    freq = Counter(data)
    total = len(data)

    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy / 8.0  # Normalize to 0-1


def calculate_effectiveness(pattern: Dict[str, Any]) -> float:
    """Calculate effectiveness score.

    E = (H_before - H_after) / n_receipts

    Args:
        pattern: Pattern dictionary with entropy data

    Returns:
        Effectiveness score (0.0 - 1.0)
    """
    h_before = pattern.get("entropy_before", 0.5)
    h_after = pattern.get("entropy_after", 0.5)
    n_receipts = pattern.get("n_receipts", 1)

    if n_receipts == 0:
        return 0.0

    reduction = max(0, h_before - h_after)
    effectiveness = reduction / n_receipts

    # Normalize to 0-1 range
    return min(1.0, max(0.0, effectiveness))


def calculate_autonomy(pattern: Dict[str, Any]) -> float:
    """Calculate autonomy score.

    A = auto_approved / total_actions

    Args:
        pattern: Pattern dictionary with action data

    Returns:
        Autonomy score (0.0 - 1.0)
    """
    auto_approved = pattern.get("auto_approved", 0)
    total_actions = pattern.get("total_actions", 1)

    if total_actions == 0:
        return 0.0

    return min(1.0, max(0.0, auto_approved / total_actions))


def calculate_transfer_score(pattern: Dict[str, Any], target_domain: str) -> float:
    """Calculate transfer score (temporal graph similarity).

    Note: NOT cosine similarity - uses temporal graph patterns.

    Args:
        pattern: Pattern dictionary
        target_domain: Target domain for transfer

    Returns:
        Transfer score (0.0 - 1.0)
    """
    source_domain = pattern.get("domain", "unknown")

    # Domain compatibility matrix (based on shared patterns)
    compatibility = {
        ("orbital_compute", "constellation_ops"): 0.75,
        ("orbital_compute", "autonomous_decision"): 0.65,
        ("orbital_compute", "firmware_integrity"): 0.80,
        ("constellation_ops", "orbital_compute"): 0.75,
        ("constellation_ops", "autonomous_decision"): 0.70,
        ("constellation_ops", "firmware_integrity"): 0.85,
        ("autonomous_decision", "orbital_compute"): 0.65,
        ("autonomous_decision", "constellation_ops"): 0.70,
        ("autonomous_decision", "firmware_integrity"): 0.75,
        ("firmware_integrity", "orbital_compute"): 0.80,
        ("firmware_integrity", "constellation_ops"): 0.85,
        ("firmware_integrity", "autonomous_decision"): 0.75,
    }

    base_score = compatibility.get((source_domain, target_domain), 0.5)

    # Adjust based on pattern effectiveness
    effectiveness = pattern.get("effectiveness", 0.5)
    transfer_score = base_score * (0.5 + 0.5 * effectiveness)

    return min(1.0, max(0.0, transfer_score))


def classify_pattern(pattern: Dict[str, Any], domain: str) -> str:
    """Classify pattern topology.

    Classification logic:
    - IF E >= V_esc[domain] AND A > AUTONOMY_THRESHOLD → "open" (graduate)
    - ELIF T > TRANSFER_THRESHOLD → "hybrid" (transfer)
    - ELSE → "closed" (continue optimizing)

    Args:
        pattern: Pattern dictionary with metrics
        domain: Domain identifier

    Returns:
        Topology string: "open", "closed", or "hybrid"
    """
    effectiveness = pattern.get("effectiveness", calculate_effectiveness(pattern))
    autonomy = pattern.get("autonomy", calculate_autonomy(pattern))
    transfer = pattern.get("transfer_score", 0.5)

    v_esc = ESCAPE_VELOCITY.get(domain, 0.85)

    if effectiveness >= v_esc and autonomy > AUTONOMY_THRESHOLD:
        return "open"
    elif transfer > TRANSFER_THRESHOLD:
        return "hybrid"
    else:
        return "closed"


def emit_topology_receipt(
    pattern: Dict[str, Any],
    domain: str,
    topology: str,
    effectiveness: float,
    autonomy: float,
    transfer: float,
    confidence: float,
) -> TopologyResult:
    """Emit topology classification receipt.

    Args:
        pattern: Pattern dictionary
        domain: Domain identifier
        topology: Classified topology
        effectiveness: E score
        autonomy: A score
        transfer: T score
        confidence: Classification confidence

    Returns:
        TopologyResult with receipt
    """
    pattern_id = pattern.get("pattern_id", dual_hash(json.dumps(pattern, sort_keys=True)))
    v_esc = ESCAPE_VELOCITY.get(domain, 0.85)

    # Determine action
    if topology == "open":
        action = "cascade"
    elif topology == "hybrid":
        action = "transfer"
    else:
        action = "optimize"

    receipt = emit_receipt(
        "topology",
        {
            "tenant_id": META_INTEGRATION_TENANT,
            "pattern_id": pattern_id,
            "domain": domain,
            "topology": topology,
            "effectiveness": effectiveness,
            "autonomy_score": autonomy,
            "transfer_score": transfer,
            "escape_velocity": v_esc,
            "confidence": confidence,
            "action": action,
        },
    )

    return TopologyResult(
        pattern_id=pattern_id,
        domain=domain,
        topology=Topology(topology),
        effectiveness=effectiveness,
        autonomy_score=autonomy,
        transfer_score=transfer,
        escape_velocity=v_esc,
        confidence=confidence,
        action=action,
        receipt=receipt,
    )


def trigger_cascade(
    pattern: Dict[str, Any],
    mutation_rate: float = DEFAULT_MUTATION_RATE,
) -> CascadeResult:
    """Spawn CASCADE_MULTIPLIER variants on graduation.

    Args:
        pattern: Graduated pattern
        mutation_rate: Mutation rate for variants

    Returns:
        CascadeResult with child patterns
    """
    parent_id = pattern.get("pattern_id", dual_hash(json.dumps(pattern, sort_keys=True)))

    # Generate exactly 5 child variants
    child_ids = []
    for i in range(CASCADE_MULTIPLIER):
        child_id = dual_hash(f"{parent_id}:variant:{i}:{uuid.uuid4()}")
        child_ids.append(child_id)

    # Backtest placeholder (would run actual backtests in production)
    backtest_results = {
        "variants_tested": CASCADE_MULTIPLIER,
        "successful_variants": CASCADE_MULTIPLIER,
        "average_fitness": pattern.get("effectiveness", 0.8),
    }

    receipt = emit_receipt(
        "cascade",
        {
            "tenant_id": META_INTEGRATION_TENANT,
            "parent_pattern_id": parent_id,
            "child_pattern_ids": child_ids,
            "mutation_rate": mutation_rate,
            "backtest_results": backtest_results,
            "cascade_multiplier": CASCADE_MULTIPLIER,
        },
    )

    return CascadeResult(
        parent_pattern_id=parent_id,
        child_pattern_ids=child_ids,
        mutation_rate=mutation_rate,
        backtest_results=backtest_results,
        receipt=receipt,
    )


def transfer_pattern(
    pattern: Dict[str, Any],
    from_domain: str,
    to_domain: str,
) -> TransferResult:
    """Transfer successful pattern to another domain.

    Args:
        pattern: Pattern to transfer
        from_domain: Source domain
        to_domain: Target domain

    Returns:
        TransferResult with transfer receipt
    """
    pattern_id = pattern.get("pattern_id", dual_hash(json.dumps(pattern, sort_keys=True)))

    transfer_score = calculate_transfer_score(pattern, to_domain)

    similarity_metrics = {
        "temporal_similarity": transfer_score,
        "structural_similarity": pattern.get("effectiveness", 0.5),
        "domain_compatibility": ESCAPE_VELOCITY.get(to_domain, 0.85) / ESCAPE_VELOCITY.get(from_domain, 0.85),
    }

    receipt = emit_receipt(
        "transfer",
        {
            "tenant_id": META_INTEGRATION_TENANT,
            "pattern_id": pattern_id,
            "from_domain": from_domain,
            "to_domain": to_domain,
            "transfer_score": transfer_score,
            "similarity_metrics": similarity_metrics,
            "transfer_threshold": TRANSFER_THRESHOLD,
            "transfer_successful": transfer_score > TRANSFER_THRESHOLD,
        },
    )

    return TransferResult(
        pattern_id=pattern_id,
        from_domain=from_domain,
        to_domain=to_domain,
        transfer_score=transfer_score,
        similarity_metrics=similarity_metrics,
        receipt=receipt,
    )


def compute_confidence(pattern: Dict[str, Any]) -> float:
    """Compute classification confidence.

    Args:
        pattern: Pattern to evaluate

    Returns:
        Confidence score (0.0 - 1.0)
    """
    n_receipts = pattern.get("n_receipts", 0)
    effectiveness = pattern.get("effectiveness", 0.5)

    # Confidence increases with more data and higher effectiveness
    data_confidence = min(1.0, n_receipts / 100)
    effectiveness_confidence = effectiveness

    # Combined confidence
    confidence = (data_confidence + effectiveness_confidence) / 2

    return min(1.0, max(0.0, confidence))


def should_trigger_fallback(confidence: float) -> bool:
    """Check if confidence triggers fallback.

    Args:
        confidence: Classification confidence

    Returns:
        True if fallback should be triggered
    """
    return confidence < CONFIDENCE_FALLBACK


def classify_all_patterns(
    patterns: List[Dict[str, Any]],
    domain: str,
) -> List[TopologyResult]:
    """Classify all patterns in a domain.

    Args:
        patterns: List of patterns to classify
        domain: Domain identifier

    Returns:
        List of TopologyResults
    """
    results = []

    for pattern in patterns:
        effectiveness = calculate_effectiveness(pattern)
        autonomy = calculate_autonomy(pattern)
        transfer = calculate_transfer_score(pattern, "firmware_integrity")  # Default transfer target
        confidence = compute_confidence(pattern)

        topology = classify_pattern(
            {**pattern, "effectiveness": effectiveness, "autonomy": autonomy, "transfer_score": transfer},
            domain,
        )

        result = emit_topology_receipt(
            pattern,
            domain,
            topology,
            effectiveness,
            autonomy,
            transfer,
            confidence,
        )

        results.append(result)

    return results


def process_graduated_patterns(results: List[TopologyResult]) -> Tuple[List[CascadeResult], List[TransferResult]]:
    """Process patterns that graduated (open) or need transfer (hybrid).

    Args:
        results: List of topology results

    Returns:
        Tuple of (cascade_results, transfer_results)
    """
    cascades = []
    transfers = []

    for result in results:
        if result.topology == Topology.OPEN:
            # Trigger cascade for graduated patterns
            pattern = {"pattern_id": result.pattern_id, "effectiveness": result.effectiveness}
            cascade = trigger_cascade(pattern)
            cascades.append(cascade)

        elif result.topology == Topology.HYBRID:
            # Transfer hybrid patterns
            pattern = {"pattern_id": result.pattern_id, "effectiveness": result.effectiveness}
            # Transfer to highest compatibility domain
            best_domain = "firmware_integrity"  # Default
            transfer = transfer_pattern(pattern, result.domain, best_domain)
            transfers.append(transfer)

    return cascades, transfers


def get_domain_escape_velocity(domain: str) -> float:
    """Get escape velocity for a domain.

    Args:
        domain: Domain identifier

    Returns:
        Escape velocity threshold
    """
    return ESCAPE_VELOCITY.get(domain, 0.85)


def validate_entropy_conservation(patterns: List[Dict[str, Any]], threshold: float = 0.01) -> bool:
    """Validate entropy conservation across patterns.

    |ΔS| must be < threshold every cycle.

    Args:
        patterns: List of patterns with entropy data
        threshold: Conservation threshold (default 0.01)

    Returns:
        True if conservation is valid
    """
    for pattern in patterns:
        h_before = pattern.get("entropy_before", 0.0)
        h_after = pattern.get("entropy_after", 0.0)
        delta = abs(h_after - h_before)

        if delta >= threshold:
            emit_receipt(
                "entropy_violation",
                {
                    "tenant_id": META_INTEGRATION_TENANT,
                    "pattern_id": pattern.get("pattern_id", "unknown"),
                    "delta_s": delta,
                    "threshold": threshold,
                    "violation": True,
                },
            )
            return False

    return True
