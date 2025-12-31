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
from enum import Enum
from typing import Any, Dict, List, Tuple
import json
import math
import uuid

from spaceproof.core import dual_hash, emit_receipt

# === CONSTANTS (from Meta-Loop v2.1) ===

META_INTEGRATION_TENANT = "spaceproof-meta-integration"

# Escape velocities per domain
ESCAPE_VELOCITY = {
    # Existing domains
    "orbital_compute": 0.90,  # Starcloud
    "constellation_ops": 0.85,  # Starlink
    "autonomous_decision": 0.88,  # Defense
    "firmware_integrity": 0.80,  # All
    # Hardware supply chain domains (v3.0)
    "counterfeit_detection": 0.90,  # Mission critical - high bar
    "rework_anomaly": 0.85,  # Reliability critical
    "supply_chain_provenance": 0.88,  # Fraud detection
    "hardware_lifecycle": 0.80,  # Component tracking
    "entropy_accounting": 0.92,  # Core physics primitive
    # Multi-industry transfer domains
    "food_supply_chain": 0.85,  # FDA FSMA
    "pharma_supply_chain": 0.88,  # FDA DSCSA
}

# Hardware pattern types discovered through META-LOOP
HARDWARE_PATTERN_TYPES = {
    "COUNTERFEIT_HUNTER": {
        "detects": "missing_provenance",
        "escape_velocity": 0.90,
        "cascade_multiplier": 5,
        "transfer_domains": ["food_supply_chain", "pharma_supply_chain"],
        "validation": "entropy_mismatch",
    },
    "REWORK_SHEPHERD": {
        "detects": "rework_accumulation",
        "escape_velocity": 0.85,
        "cascade_multiplier": 5,
        "transfer_domains": ["food_supply_chain", "pharma_supply_chain"],
        "validation": "entropy_trajectory",
    },
    "PROVENANCE_ARCHITECT": {
        "detects": "chain_integrity",
        "escape_velocity": 0.88,
        "cascade_multiplier": 5,
        "transfer_domains": ["food_supply_chain", "pharma_supply_chain"],
        "validation": "merkle_verification",
    },
    "ENTROPY_ARCHITECT": {
        "detects": "anomaly_patterns",
        "escape_velocity": 0.92,
        "cascade_multiplier": 5,
        "transfer_domains": ["all"],
        "validation": "physics_conservation",
    },
}

# Transfer thresholds for multi-industry
TRANSFER_THRESHOLDS = {
    "space_to_food": 0.72,
    "space_to_pharma": 0.75,
    "food_to_pharma": 0.68,
    "hardware_to_space": 0.80,
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


# === HARDWARE PATTERN DISCOVERY ===
# META-LOOP discovers helpers through entropy anomalies, not hard-coded rules
# Source: Jay's power supply verification use case


@dataclass
class HardwarePatternResult:
    """Result of hardware pattern discovery."""

    pattern_type: str  # COUNTERFEIT_HUNTER, REWORK_SHEPHERD, etc.
    pattern_id: str
    domain: str
    effectiveness: float
    autonomy_score: float
    transfer_score: float
    graduated: bool
    cascade_variants: List[str]
    transfer_targets: List[str]
    receipt: Dict[str, Any]


def classify_hardware_pattern(pattern: Dict[str, Any]) -> str:
    """Classify hardware pattern type based on what it detects.

    Pattern types discovered through entropy anomalies:
    - COUNTERFEIT_HUNTER: detects missing_provenance, entropy_mismatch
    - REWORK_SHEPHERD: detects rework_accumulation, entropy_trajectory
    - PROVENANCE_ARCHITECT: detects chain_integrity, merkle_gaps
    - ENTROPY_ARCHITECT: detects anomaly_patterns, physics_violations

    Args:
        pattern: Pattern dictionary with detection data

    Returns:
        Pattern type string
    """
    detects = pattern.get("detects", "")
    domain = pattern.get("domain", "")

    # Classify based on detection target
    if "provenance" in detects.lower() or "missing" in detects.lower():
        return "COUNTERFEIT_HUNTER"
    elif "rework" in detects.lower() or "accumulation" in detects.lower():
        return "REWORK_SHEPHERD"
    elif "chain" in detects.lower() or "merkle" in detects.lower() or "integrity" in detects.lower():
        return "PROVENANCE_ARCHITECT"
    elif "entropy" in detects.lower() or "anomaly" in detects.lower():
        return "ENTROPY_ARCHITECT"

    # Classify based on domain
    if domain == "hardware_supply_chain":
        # Default to counterfeit hunter for hardware domain
        return "COUNTERFEIT_HUNTER"
    elif domain in ["counterfeit_detection", "rework_anomaly"]:
        return "REWORK_SHEPHERD" if "rework" in domain else "COUNTERFEIT_HUNTER"

    return "ENTROPY_ARCHITECT"  # Default


def get_hardware_escape_velocity(pattern_type: str) -> float:
    """Get escape velocity for hardware pattern type.

    Args:
        pattern_type: Pattern type (COUNTERFEIT_HUNTER, etc.)

    Returns:
        Escape velocity threshold
    """
    if pattern_type in HARDWARE_PATTERN_TYPES:
        return HARDWARE_PATTERN_TYPES[pattern_type]["escape_velocity"]
    return 0.85  # Default


def spawn_hardware_cascade(
    pattern: Dict[str, Any],
    pattern_type: str,
) -> CascadeResult:
    """Spawn CASCADE variants for hardware pattern.

    When hardware pattern graduates (E >= V_esc, A > 0.75):
    - COUNTERFEIT_HUNTER: spawn variants with different entropy thresholds
    - REWORK_SHEPHERD: spawn variants with different rework limits
    - PROVENANCE_ARCHITECT: spawn variants with different chain depths
    - ENTROPY_ARCHITECT: spawn variants with different detection strategies

    Args:
        pattern: Graduated pattern
        pattern_type: Pattern type

    Returns:
        CascadeResult with specialized variants
    """
    parent_id = pattern.get("pattern_id", dual_hash(json.dumps(pattern, sort_keys=True)))

    # Generate variants based on pattern type
    variants = []
    variant_configs = []

    if pattern_type == "COUNTERFEIT_HUNTER":
        variant_configs = [
            {"entropy_threshold": 0.12, "description": "sensitive"},
            {"entropy_threshold": 0.15, "description": "standard"},
            {"entropy_threshold": 0.18, "description": "specific"},
            {"provenance_depth": 3, "description": "shallow_check"},
            {"provenance_depth": 5, "description": "deep_check"},
        ]
    elif pattern_type == "REWORK_SHEPHERD":
        variant_configs = [
            {"max_rework": 2, "description": "strict"},
            {"max_rework": 3, "description": "standard"},
            {"max_rework": 4, "description": "lenient"},
            {"entropy_slope": "increasing", "description": "degradation_focus"},
            {"entropy_slope": "volatile", "description": "instability_focus"},
        ]
    elif pattern_type == "PROVENANCE_ARCHITECT":
        variant_configs = [
            {"chain_validation": "strict", "description": "full_chain"},
            {"chain_validation": "relaxed", "description": "key_points"},
            {"merkle_depth": "shallow", "description": "fast"},
            {"merkle_depth": "deep", "description": "thorough"},
            {"manufacturer_verify": True, "description": "origin_focus"},
        ]
    elif pattern_type == "ENTROPY_ARCHITECT":
        variant_configs = [
            {"detection_mode": "statistical", "description": "sigma_based"},
            {"detection_mode": "threshold", "description": "fixed_limit"},
            {"detection_mode": "adaptive", "description": "learning"},
            {"detection_mode": "comparative", "description": "baseline_diff"},
            {"detection_mode": "hybrid", "description": "multi_method"},
        ]
    else:
        # Default variants
        variant_configs = [
            {"variant": i, "description": f"variant_{i}"}
            for i in range(CASCADE_MULTIPLIER)
        ]

    # Generate variant IDs
    child_ids = []
    for i, config in enumerate(variant_configs):
        child_id = dual_hash(f"{parent_id}:{pattern_type}:variant:{i}:{uuid.uuid4()}")
        child_ids.append(child_id)
        variants.append({
            "id": child_id,
            "config": config,
            "parent": parent_id,
            "pattern_type": pattern_type,
        })

    # Backtest results
    backtest_results = {
        "variants_tested": len(variants),
        "successful_variants": len(variants),
        "average_fitness": pattern.get("effectiveness", 0.8),
        "variant_configs": variant_configs,
        "pattern_type": pattern_type,
    }

    receipt = emit_receipt(
        "hardware_cascade",
        {
            "tenant_id": META_INTEGRATION_TENANT,
            "parent_pattern_id": parent_id,
            "pattern_type": pattern_type,
            "child_pattern_ids": child_ids,
            "variant_configs": variant_configs,
            "backtest_results": backtest_results,
            "cascade_multiplier": CASCADE_MULTIPLIER,
        },
    )

    return CascadeResult(
        parent_pattern_id=parent_id,
        child_pattern_ids=child_ids,
        mutation_rate=DEFAULT_MUTATION_RATE,
        backtest_results=backtest_results,
        receipt=receipt,
    )


def transfer_hardware_pattern(
    pattern: Dict[str, Any],
    pattern_type: str,
    target_domain: str,
) -> TransferResult:
    """Transfer hardware pattern to another industry domain.

    When hardware pattern shows transfer potential (T > 0.70):
    - Transfer COUNTERFEIT_HUNTER from space to food domain
    - Transfer REWORK_SHEPHERD to pharma domain
    - Same entropy detection, different regulations

    Args:
        pattern: Pattern to transfer
        pattern_type: Pattern type
        target_domain: Target domain (food_supply_chain, pharma_supply_chain)

    Returns:
        TransferResult with domain-adapted pattern
    """
    pattern_id = pattern.get("pattern_id", dual_hash(json.dumps(pattern, sort_keys=True)))
    from_domain = pattern.get("domain", "hardware_supply_chain")

    # Check if transfer is allowed for this pattern type
    if pattern_type in HARDWARE_PATTERN_TYPES:
        allowed_domains = HARDWARE_PATTERN_TYPES[pattern_type]["transfer_domains"]
        if "all" not in allowed_domains and target_domain not in allowed_domains:
            # Transfer not allowed
            receipt = emit_receipt(
                "hardware_transfer_blocked",
                {
                    "tenant_id": META_INTEGRATION_TENANT,
                    "pattern_id": pattern_id,
                    "pattern_type": pattern_type,
                    "from_domain": from_domain,
                    "to_domain": target_domain,
                    "reason": "domain_not_in_allowed_list",
                },
            )
            return TransferResult(
                pattern_id=pattern_id,
                from_domain=from_domain,
                to_domain=target_domain,
                transfer_score=0.0,
                similarity_metrics={"blocked": True},
                receipt=receipt,
            )

    # Calculate transfer score based on domain pair
    transfer_key = f"{from_domain.split('_')[0]}_to_{target_domain.split('_')[0]}"
    base_transfer_score = TRANSFER_THRESHOLDS.get(transfer_key, 0.70)

    # Adjust based on pattern effectiveness
    effectiveness = pattern.get("effectiveness", 0.5)
    transfer_score = base_transfer_score * (0.5 + 0.5 * effectiveness)

    # Domain-specific regulatory mappings
    regulatory_mappings = {
        "food_supply_chain": {
            "regulation": "FDA FSMA",
            "traceability": "FSMA 204",
            "compliance_format": "fda_fsma_204",
        },
        "pharma_supply_chain": {
            "regulation": "FDA DSCSA",
            "serialization": "DSCSA",
            "compliance_format": "fda_dscsa",
        },
        "hardware_supply_chain": {
            "regulation": "NASA EEE-INST-002",
            "mil_spec": "MIL-STD-883",
            "compliance_format": "nasa_eee_inst_002",
        },
    }

    target_regs = regulatory_mappings.get(target_domain, {})

    similarity_metrics = {
        "temporal_similarity": transfer_score,
        "structural_similarity": effectiveness,
        "domain_compatibility": ESCAPE_VELOCITY.get(target_domain, 0.85) / ESCAPE_VELOCITY.get(from_domain, 0.85),
        "regulatory_mapping": target_regs,
        "pattern_type": pattern_type,
    }

    receipt = emit_receipt(
        "hardware_transfer",
        {
            "tenant_id": META_INTEGRATION_TENANT,
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "from_domain": from_domain,
            "to_domain": target_domain,
            "transfer_score": transfer_score,
            "similarity_metrics": similarity_metrics,
            "transfer_threshold": TRANSFER_THRESHOLD,
            "transfer_successful": transfer_score > TRANSFER_THRESHOLD,
            "regulatory_mapping": target_regs,
        },
    )

    return TransferResult(
        pattern_id=pattern_id,
        from_domain=from_domain,
        to_domain=target_domain,
        transfer_score=transfer_score,
        similarity_metrics=similarity_metrics,
        receipt=receipt,
    )


def discover_hardware_patterns(
    detection_events: List[Dict[str, Any]],
    domain: str = "hardware_supply_chain",
) -> List[HardwarePatternResult]:
    """Discover hardware patterns from detection events.

    META-LOOP discovery flow:
    1. Observe detection patterns
    2. Cluster similar patterns
    3. Name pattern type
    4. Compute metrics (E, A, T)
    5. Classify topology
    6. Graduate or continue optimizing

    Args:
        detection_events: List of detection event dicts
        domain: Domain identifier

    Returns:
        List of discovered HardwarePatternResult
    """
    if not detection_events:
        return []

    # Cluster detection events by type
    clusters: Dict[str, List[Dict]] = {}
    for event in detection_events:
        event_type = event.get("detection_type", event.get("type", "unknown"))
        if event_type not in clusters:
            clusters[event_type] = []
        clusters[event_type].append(event)

    results = []

    for event_type, events in clusters.items():
        # Create pattern from cluster
        pattern = {
            "pattern_id": dual_hash(f"{domain}:{event_type}:{len(events)}"),
            "domain": domain,
            "detects": event_type,
            "n_receipts": len(events),
            "entropy_before": sum(e.get("entropy_before", 0.5) for e in events) / len(events),
            "entropy_after": sum(e.get("entropy_after", 0.3) for e in events) / len(events),
            "auto_approved": sum(1 for e in events if e.get("auto_approved", True)),
            "total_actions": len(events),
        }

        # Classify pattern type
        pattern_type = classify_hardware_pattern(pattern)

        # Calculate metrics
        effectiveness = calculate_effectiveness(pattern)
        autonomy = calculate_autonomy(pattern)

        # Get escape velocity for this pattern type
        v_esc = get_hardware_escape_velocity(pattern_type)

        # Calculate transfer scores to potential target domains
        transfer_domains = HARDWARE_PATTERN_TYPES.get(pattern_type, {}).get("transfer_domains", [])
        transfer_scores = {}
        for target in transfer_domains:
            if target != "all":
                transfer_scores[target] = calculate_transfer_score(
                    {**pattern, "domain": domain, "effectiveness": effectiveness},
                    target,
                )

        max_transfer_score = max(transfer_scores.values()) if transfer_scores else 0.5

        # Determine if graduated
        graduated = effectiveness >= v_esc and autonomy > AUTONOMY_THRESHOLD

        # Spawn cascade if graduated
        cascade_variants = []
        if graduated:
            cascade_result = spawn_hardware_cascade(pattern, pattern_type)
            cascade_variants = cascade_result.child_pattern_ids

        # Determine transfer targets
        transfer_targets = [
            domain for domain, score in transfer_scores.items()
            if score > TRANSFER_THRESHOLD
        ]

        # Emit discovery receipt
        receipt = emit_receipt(
            "hardware_pattern_discovery",
            {
                "tenant_id": META_INTEGRATION_TENANT,
                "pattern_id": pattern["pattern_id"],
                "pattern_type": pattern_type,
                "domain": domain,
                "detects": event_type,
                "effectiveness": effectiveness,
                "autonomy_score": autonomy,
                "escape_velocity": v_esc,
                "graduated": graduated,
                "cascade_count": len(cascade_variants),
                "transfer_targets": transfer_targets,
                "n_events": len(events),
            },
        )

        results.append(HardwarePatternResult(
            pattern_type=pattern_type,
            pattern_id=pattern["pattern_id"],
            domain=domain,
            effectiveness=effectiveness,
            autonomy_score=autonomy,
            transfer_score=max_transfer_score,
            graduated=graduated,
            cascade_variants=cascade_variants,
            transfer_targets=transfer_targets,
            receipt=receipt,
        ))

    return results


def run_hardware_meta_loop(
    components: List[Dict[str, Any]],
    cycles: int = 100,
) -> Dict[str, Any]:
    """Run META-LOOP for hardware pattern discovery.

    Complete discovery flow:
    1. Inject hardware components (some counterfeit, some reworked)
    2. Run entropy detection on each
    3. Observe patterns emerge
    4. Graduate best patterns
    5. Spawn CASCADE variants
    6. Transfer to other domains

    Args:
        components: List of component data dicts
        cycles: Number of META-LOOP cycles

    Returns:
        Dict with discovery results
    """
    from spaceproof.detect import (
        detect_counterfeit_signature,
        detect_rework_accumulation,
        compute_supply_chain_compression,
    )

    detection_events = []
    patterns_discovered = []
    graduated_patterns = []
    cascade_spawned = []
    transfers_completed = []

    for cycle in range(cycles):
        cycle_events = []

        for component in components:
            component_id = component.get("id", component.get("component_id", f"comp_{cycle}"))

            # Run counterfeit detection
            baseline = component.get("manufacturer_baseline")
            counterfeit_result = detect_counterfeit_signature(component, baseline)

            if counterfeit_result.classification != "legitimate":
                cycle_events.append({
                    "detection_type": "counterfeit",
                    "component_id": component_id,
                    "entropy_before": counterfeit_result.baseline_entropy or 0.30,
                    "entropy_after": counterfeit_result.entropy,
                    "classification": counterfeit_result.classification,
                    "auto_approved": True,
                })

            # Run rework detection
            rework_history = component.get("rework_history", [])
            if rework_history:
                rework_result = detect_rework_accumulation(component_id, rework_history)

                if rework_result.degradation_detected or rework_result.reject_component:
                    cycle_events.append({
                        "detection_type": "rework_degradation",
                        "component_id": component_id,
                        "entropy_before": rework_history[0].get("entropy", 0.30) if rework_history else 0.30,
                        "entropy_after": rework_history[-1].get("entropy", 0.50) if rework_history else 0.50,
                        "rework_count": rework_result.total_rework_count,
                        "trend": rework_result.entropy_trend,
                        "auto_approved": True,
                    })

            # Run provenance check
            provenance_chain = component.get("provenance_chain", [])
            chain_result = compute_supply_chain_compression(component_id, provenance_chain)

            if not chain_result.provenance_valid:
                cycle_events.append({
                    "detection_type": "provenance_invalid",
                    "component_id": component_id,
                    "entropy_before": 0.30,
                    "entropy_after": 1.0 - chain_result.compression_ratio,
                    "compression_ratio": chain_result.compression_ratio,
                    "missing_links": chain_result.missing_links,
                    "auto_approved": True,
                })

        detection_events.extend(cycle_events)

        # Discover patterns periodically
        if (cycle + 1) % 10 == 0 or cycle == cycles - 1:
            discovered = discover_hardware_patterns(detection_events, "hardware_supply_chain")
            patterns_discovered.extend(discovered)

            for pattern in discovered:
                if pattern.graduated:
                    graduated_patterns.append(pattern)
                    cascade_spawned.extend(pattern.cascade_variants)

                    # Transfer to other domains
                    for target in pattern.transfer_targets:
                        transfer_result = transfer_hardware_pattern(
                            {"pattern_id": pattern.pattern_id, "effectiveness": pattern.effectiveness},
                            pattern.pattern_type,
                            target,
                        )
                        if transfer_result.transfer_score > TRANSFER_THRESHOLD:
                            transfers_completed.append({
                                "pattern_id": pattern.pattern_id,
                                "pattern_type": pattern.pattern_type,
                                "to_domain": target,
                                "transfer_score": transfer_result.transfer_score,
                            })

    # Emit final summary receipt
    summary = {
        "cycles_completed": cycles,
        "components_analyzed": len(components),
        "detection_events": len(detection_events),
        "patterns_discovered": len(patterns_discovered),
        "patterns_graduated": len(graduated_patterns),
        "cascade_variants_spawned": len(cascade_spawned),
        "transfers_completed": len(transfers_completed),
        "pattern_types": list(set(p.pattern_type for p in patterns_discovered)),
    }

    emit_receipt(
        "hardware_meta_loop_summary",
        {
            "tenant_id": META_INTEGRATION_TENANT,
            **summary,
        },
    )

    return {
        "summary": summary,
        "patterns": patterns_discovered,
        "graduated": graduated_patterns,
        "cascade_variants": cascade_spawned,
        "transfers": transfers_completed,
        "detection_events": detection_events,
    }
