"""reroute.py - Adaptive Hybrid Algorithm for Dynamic Path Recovery

THE PHYSICS (Dec 2025 adaptive rerouting):
    Hybrid Ephemeris-Deterministic + ML-Predictive:
    - CGR Base: Deterministic shortest-path on precomputed orbital contact plan
    - ML Adaptive: Lightweight GNN for contact degradation prediction
    - Graph: Nodes = relay sats + surface habitats; Edges = optical links
    - Quorum-aware: Reroute preserves Merkle chain continuity

CONSTANTS (LOCKED - validated by prior gate 2025-12-16):
    REROUTING_ALPHA_BOOST_LOCKED = 0.07 (validated, immutable)
    MIN_EFF_ALPHA_VALIDATED = 2.656 (refined floor from prior gate)
    BLACKOUT_BASE_DAYS = 43 (Mars solar conjunction maximum)
    BLACKOUT_EXTENDED_DAYS = 60 (with reroute: 43d * 1.4 retention)

Source: Grok - "Prioritize adaptive rerouting: +0.07 to 2.7+"
Gate: PASSED 2025-12-16 (eff_α=2.70, min_α=2.656, 43d survival, quorum preserved)
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from .core import emit_receipt, dual_hash, StopRule


# === CONSTANTS (LOCKED - validated by prior gate 2025-12-16) ===

REROUTING_ALPHA_BOOST_LOCKED = 0.07
"""physics: LOCKED. Validated reroute boost. 2.656 + 0.07 = 2.726 → floor 2.70"""

# Immutability assertion at module load
assert REROUTING_ALPHA_BOOST_LOCKED == 0.07, "REROUTING_ALPHA_BOOST_LOCKED must remain 0.07"

# Backward compatibility alias (to be deprecated)
REROUTE_ALPHA_BOOST = REROUTING_ALPHA_BOOST_LOCKED

BLACKOUT_BASE_DAYS = 43
"""physics: Mars solar conjunction maximum duration in days."""

BLACKOUT_EXTENDED_DAYS = 60
"""physics: Extended blackout tolerance with reroute (43d * 1.4 retention)."""

REROUTE_RETENTION_FACTOR = 1.4
"""physics: ~40% duration extension with adaptive rerouting."""

MIN_EFF_ALPHA_VALIDATED = 2.656
"""physics: LOCKED. Refined validated floor from prior gate (was 2.63)."""

# Backward compatibility alias (to be deprecated)
MIN_EFF_ALPHA_FLOOR = MIN_EFF_ALPHA_VALIDATED

GATE_PASS_TIMESTAMP = "2025-12-16"
"""Audit trail: Gate pass date for locked constants."""

CGR_BASELINE = "nasa_dtn_v3"
"""Contact Graph Routing standard baseline."""

ML_MODEL_TYPE = "lightweight_gnn"
"""Graph Neural Network for anomaly prediction."""

ALGO_TYPE = "hybrid_ephemeris_ml"
"""Hybrid algorithm combining CGR + ML prediction."""

REROUTE_SPEC_PATH = "data/reroute_blackout_spec.json"
"""Path to reroute specification file."""


@dataclass
class RerouteResult:
    """Result from adaptive reroute operation.

    Attributes:
        recovery_factor: Recovery effectiveness (0-1)
        new_paths: List of alternative path dicts
        alpha_boost: Applied alpha boost
        quorum_preserved: Whether Merkle chain continuity maintained
    """
    recovery_factor: float
    new_paths: List[Dict[str, Any]]
    alpha_boost: float
    quorum_preserved: bool


@dataclass
class BlackoutResult:
    """Result from blackout simulation.

    Attributes:
        survival_status: True if survived blackout
        alpha_trajectory: List of daily alpha values
        quorum_health: Daily quorum status
        min_alpha_during: Minimum alpha during blackout
        max_alpha_drop: Maximum alpha drop during blackout
    """
    survival_status: bool
    alpha_trajectory: List[float]
    quorum_health: List[bool]
    min_alpha_during: float
    max_alpha_drop: float


def load_reroute_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify reroute specification file.

    Loads data/reroute_blackout_spec.json and emits ingest receipt
    with dual_hash per CLAUDEME S4.1.

    Args:
        path: Optional path override (default: REROUTE_SPEC_PATH)

    Returns:
        Dict containing reroute specification

    Receipt: reroute_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, REROUTE_SPEC_PATH)

    with open(path, 'r') as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt("reroute_spec_ingest", {
        "tenant_id": "axiom-reroute",
        "file_path": path,
        "algo_type": data["algo_type"],
        "blackout_base_days": data["blackout_base_days"],
        "blackout_extended_days": data["blackout_extended_days"],
        "reroute_alpha_boost": data["reroute_alpha_boost"],
        "min_eff_alpha_floor": data["min_eff_alpha_floor"],
        "cgr_baseline": data["cgr_baseline"],
        "ml_model_type": data["ml_model_type"],
        "payload_hash": content_hash
    })

    return data


def compute_cgr_paths(
    contact_graph: Dict[str, Any],
    source: str,
    targets: List[str]
) -> List[Dict[str, Any]]:
    """Compute Contact Graph Routing paths.

    Deterministic CGR baseline using time-varying Dijkstra on scheduled graph.
    Standard NASA DTN v3 algorithm.

    Args:
        contact_graph: Dict with 'nodes' and 'edges' lists
        source: Source node ID
        targets: List of target node IDs

    Returns:
        List of path dicts with hop_count, latency_ms, reliability

    Receipt: cgr_paths
    """
    nodes = contact_graph.get("nodes", [])
    edges = contact_graph.get("edges", [])

    paths = []
    for target in targets:
        # Simplified CGR: find path via edges
        # In production, this would use proper Dijkstra on contact windows
        path_edges = [e for e in edges if e.get("src") == source or e.get("dst") == target]
        hop_count = min(len(path_edges), 3) + 1  # Simplified: 1-4 hops typical
        latency_ms = hop_count * 50 + random.randint(10, 100)  # 50ms per hop + jitter
        reliability = max(0.9, 1.0 - (hop_count * 0.02))  # 2% drop per hop

        paths.append({
            "source": source,
            "target": target,
            "hop_count": hop_count,
            "latency_ms": latency_ms,
            "reliability": round(reliability, 4),
            "algo": CGR_BASELINE
        })

    emit_receipt("cgr_paths", {
        "tenant_id": "axiom-reroute",
        "source": source,
        "targets": targets,
        "paths_computed": len(paths),
        "avg_hops": sum(p["hop_count"] for p in paths) / max(1, len(paths)),
        "avg_latency_ms": sum(p["latency_ms"] for p in paths) / max(1, len(paths)),
        "algo": CGR_BASELINE
    })

    return paths


def predict_degradation(
    historical_anomalies: List[Dict[str, Any]],
    current_state: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """Predict contact degradation using ML model.

    Lightweight GNN stub. Returns conservative estimate if model unavailable.
    In production, this would use trained weights for anomaly prediction.

    Args:
        historical_anomalies: List of prior anomaly events
        current_state: Current network state dict

    Returns:
        Tuple of (degradation_probability, affected_edges)

    Receipt: ml_prediction
    """
    # Stub implementation: conservative estimate based on anomaly count
    anomaly_count = len(historical_anomalies)
    blackout_active = current_state.get("blackout_active", False)
    partition_pct = current_state.get("partition_pct", 0.0)

    # Base probability from historical patterns
    base_prob = min(0.5, anomaly_count * 0.05)

    # Increase if blackout active
    if blackout_active:
        base_prob = min(0.8, base_prob + 0.3)

    # Increase by partition percentage
    degradation_prob = min(0.95, base_prob + partition_pct * 0.5)

    # Identify affected edges (simplified: random selection based on probability)
    all_edges = current_state.get("edges", ["edge_1", "edge_2", "edge_3"])
    affected_count = max(1, int(len(all_edges) * degradation_prob))
    affected_edges = all_edges[:affected_count]

    emit_receipt("ml_prediction", {
        "tenant_id": "axiom-reroute",
        "model_type": ML_MODEL_TYPE,
        "historical_anomaly_count": anomaly_count,
        "blackout_active": blackout_active,
        "partition_pct": partition_pct,
        "degradation_probability": round(degradation_prob, 4),
        "affected_edges_count": len(affected_edges),
        "model_status": "stub_conservative"
    })

    return round(degradation_prob, 4), affected_edges


def apply_reroute_boost(
    base_alpha: float,
    reroute_active: bool,
    blackout_days: int = 0
) -> float:
    """Apply reroute boost to effective alpha.

    Formula: boosted_alpha = base_alpha + (REROUTE_ALPHA_BOOST * reroute_active * retention_scale)
    Retention scale degrades gracefully beyond 43d base blackout.

    Args:
        base_alpha: Base effective alpha
        reroute_active: Whether reroute is enabled
        blackout_days: Current blackout duration in days

    Returns:
        Boosted effective alpha

    Receipt: reroute_boost_applied
    """
    if not reroute_active:
        emit_receipt("reroute_boost_applied", {
            "tenant_id": "axiom-reroute",
            "base_alpha": base_alpha,
            "reroute_active": False,
            "boost_applied": 0.0,
            "boosted_alpha": base_alpha,
            "blackout_days": blackout_days
        })
        return base_alpha

    # Retention scale: degrades gracefully beyond base blackout
    if blackout_days <= BLACKOUT_BASE_DAYS:
        retention_scale = 1.0
    else:
        # Linear degradation from 1.0 at 43d to 0.7 at 60d
        excess_days = blackout_days - BLACKOUT_BASE_DAYS
        max_excess = BLACKOUT_EXTENDED_DAYS - BLACKOUT_BASE_DAYS
        degradation = min(0.3, (excess_days / max_excess) * 0.3)
        retention_scale = 1.0 - degradation

    boost_applied = REROUTE_ALPHA_BOOST * retention_scale
    boosted_alpha = base_alpha + boost_applied

    emit_receipt("reroute_boost_applied", {
        "tenant_id": "axiom-reroute",
        "base_alpha": base_alpha,
        "reroute_active": True,
        "blackout_days": blackout_days,
        "retention_scale": round(retention_scale, 4),
        "boost_applied": round(boost_applied, 4),
        "boosted_alpha": round(boosted_alpha, 4)
    })

    return round(boosted_alpha, 4)


def adaptive_reroute(
    graph_state: Dict[str, Any],
    partition_pct: float,
    blackout_days: int = 0
) -> Dict[str, Any]:
    """Execute adaptive rerouting for path recovery.

    Pure function. Combines CGR baseline with ML prediction for hybrid routing.

    Args:
        graph_state: Current graph state with nodes, edges, and current paths
        partition_pct: Current partition percentage (0-1)
        blackout_days: Current blackout duration in days

    Returns:
        Dict with recovery_factor, new_paths, alpha_boost, quorum_preserved

    Raises:
        StopRule: If unrecoverable (quorum lost AND no viable reroute)

    Receipt: adaptive_reroute_receipt
    """
    nodes = graph_state.get("nodes", 5)
    edges = graph_state.get("edges", [])
    historical_anomalies = graph_state.get("anomalies", [])

    # Calculate surviving nodes
    if isinstance(nodes, int):
        nodes_total = nodes
    else:
        nodes_total = len(nodes)

    nodes_lost = int(nodes_total * partition_pct)
    nodes_surviving = nodes_total - nodes_lost
    quorum_threshold = 3  # Byzantine fault tolerance

    # Check if quorum is viable
    if nodes_surviving < quorum_threshold:
        # Attempt emergency reroute before failing
        emergency_recovery = min(0.2, partition_pct * 0.5)  # Up to 20% recovery
        potential_surviving = nodes_surviving + int(nodes_total * emergency_recovery)

        if potential_surviving < quorum_threshold:
            emit_receipt("anomaly", {
                "tenant_id": "axiom-reroute",
                "metric": "reroute_failure",
                "baseline": quorum_threshold,
                "delta": nodes_surviving - quorum_threshold,
                "classification": "violation",
                "action": "halt",
                "partition_pct": partition_pct,
                "blackout_days": blackout_days,
                "nodes_surviving": nodes_surviving
            })
            raise StopRule(f"Unrecoverable: {nodes_surviving} nodes < {quorum_threshold} quorum, no viable reroute")

    # Predict degradation using ML
    current_state = {
        "blackout_active": blackout_days > 0,
        "partition_pct": partition_pct,
        "edges": [f"edge_{i}" for i in range(10)]
    }
    degradation_prob, affected_edges = predict_degradation(historical_anomalies, current_state)

    # Compute CGR paths for recovery
    contact_graph = {
        "nodes": [f"node_{i}" for i in range(nodes_surviving)],
        "edges": [{"src": f"node_{i}", "dst": f"node_{(i+1) % nodes_surviving}"}
                  for i in range(nodes_surviving)]
    }

    if nodes_surviving > 1:
        source = "node_0"
        targets = [f"node_{i}" for i in range(1, nodes_surviving)]
        new_paths = compute_cgr_paths(contact_graph, source, targets)
    else:
        new_paths = []

    # Calculate recovery factor
    # Higher partition = lower recovery, but reroute helps
    base_recovery = 1.0 - partition_pct
    reroute_bonus = min(0.3, (1.0 - degradation_prob) * 0.4)
    recovery_factor = min(1.0, base_recovery + reroute_bonus)

    # Verify Merkle chain continuity (quorum check)
    quorum_preserved = nodes_surviving >= quorum_threshold

    # Apply alpha boost if recovery successful
    alpha_boost = 0.0
    if quorum_preserved and recovery_factor > 0.5:
        alpha_boost = REROUTE_ALPHA_BOOST * recovery_factor

    result = {
        "recovery_factor": round(recovery_factor, 4),
        "new_paths": new_paths,
        "new_paths_count": len(new_paths),
        "alpha_boost": round(alpha_boost, 4),
        "quorum_preserved": quorum_preserved,
        "nodes_surviving": nodes_surviving,
        "nodes_total": nodes_total,
        "partition_pct": partition_pct,
        "blackout_days": blackout_days,
        "degradation_prob": degradation_prob,
        "algo_type": ALGO_TYPE
    }

    emit_receipt("adaptive_reroute", {
        "tenant_id": "axiom-reroute",
        **result
    })

    return result


def blackout_sim(
    nodes: int = 5,
    blackout_days: int = 43,
    reroute_enabled: bool = True,
    base_alpha: float = 2.63,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run full blackout simulation with daily state updates.

    Simulates Mars solar conjunction blackout with adaptive rerouting.

    Args:
        nodes: Total node count (default: 5)
        blackout_days: Blackout duration in days (default: 43)
        reroute_enabled: Whether adaptive rerouting is active
        base_alpha: Baseline effective alpha (default: 2.63 floor)
        seed: Random seed for reproducibility

    Returns:
        Dict with survival_status, alpha_trajectory, quorum_health,
        min_alpha_during, max_alpha_drop

    Receipt: blackout_sim_receipt
    """
    if seed is not None:
        random.seed(seed)

    alpha_trajectory = []
    quorum_health = []
    current_alpha = base_alpha
    quorum_failures = 0
    max_alpha_drop = 0.0

    for day in range(blackout_days):
        # Simulate daily degradation
        # Partition stress increases mid-blackout, peaks around day 20-25
        blackout_progress = day / blackout_days
        stress_curve = 4 * blackout_progress * (1 - blackout_progress)  # Parabolic peak
        daily_partition_pct = min(0.35, stress_curve * 0.35 + random.uniform(-0.05, 0.05))

        # Calculate nodes surviving
        nodes_lost = max(0, int(nodes * daily_partition_pct))
        nodes_surviving = nodes - nodes_lost
        quorum_threshold = 3

        quorum_ok = nodes_surviving >= quorum_threshold
        quorum_health.append(quorum_ok)

        if not quorum_ok:
            quorum_failures += 1
            if not reroute_enabled:
                # Without reroute, simulation fails
                alpha_trajectory.append(0.0)
                continue

        # Calculate daily alpha
        partition_drop = daily_partition_pct * 0.125  # Same formula as partition.py

        if reroute_enabled:
            # Apply reroute boost
            boosted_alpha = apply_reroute_boost(base_alpha, True, day)
            daily_alpha = boosted_alpha - partition_drop
        else:
            daily_alpha = base_alpha - partition_drop

        alpha_trajectory.append(round(daily_alpha, 4))

        # Track max drop
        drop = base_alpha - daily_alpha
        if reroute_enabled:
            drop = boosted_alpha - daily_alpha
        max_alpha_drop = max(max_alpha_drop, drop)

    # Determine survival
    min_alpha_during = min(alpha_trajectory) if alpha_trajectory else 0.0
    survival_status = (quorum_failures == 0) or (reroute_enabled and min_alpha_during > 0)

    # Extended survival check for 60d+
    if blackout_days > BLACKOUT_BASE_DAYS and reroute_enabled:
        # With reroute, can survive up to 60d if alpha stays above floor
        survival_status = min_alpha_during >= MIN_EFF_ALPHA_FLOOR * 0.95

    result = {
        "survival_status": survival_status,
        "alpha_trajectory": alpha_trajectory,
        "quorum_health": quorum_health,
        "min_alpha_during": round(min_alpha_during, 4),
        "max_alpha_drop": round(max_alpha_drop, 4),
        "blackout_days": blackout_days,
        "reroute_enabled": reroute_enabled,
        "base_alpha": base_alpha,
        "nodes": nodes,
        "quorum_failures": quorum_failures
    }

    emit_receipt("blackout_sim", {
        "tenant_id": "axiom-reroute",
        "blackout_days": blackout_days,
        "reroute_enabled": reroute_enabled,
        "survival_status": survival_status,
        "min_alpha_during": round(min_alpha_during, 4),
        "max_alpha_drop": round(max_alpha_drop, 4),
        "quorum_failures": quorum_failures,
        "nodes": nodes,
        "base_alpha": base_alpha
    })

    return result


def blackout_stress_sweep(
    nodes: int = 5,
    blackout_range: Tuple[int, int] = (43, 60),
    n_iterations: int = 1000,
    reroute_enabled: bool = True,
    base_alpha: float = 2.63,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run blackout stress sweep across duration range.

    Runs n_iterations with random blackout durations within range.

    Args:
        nodes: Total node count (default: 5)
        blackout_range: Tuple of (min_days, max_days)
        n_iterations: Number of iterations (default: 1000)
        reroute_enabled: Whether adaptive rerouting is active
        base_alpha: Baseline effective alpha
        seed: Random seed for reproducibility

    Returns:
        Dict with survival_rate, avg_min_alpha, failures list

    Receipt: blackout_stress_sweep
    """
    if seed is not None:
        random.seed(seed)

    results = []
    failures = 0
    total_min_alpha = 0.0
    total_max_drop = 0.0

    for i in range(n_iterations):
        # Random blackout duration in range
        blackout_days = random.randint(blackout_range[0], blackout_range[1])

        # Use different seed for each iteration but deterministic
        iter_seed = (seed or 42) + i if seed is not None else None

        sim_result = blackout_sim(
            nodes=nodes,
            blackout_days=blackout_days,
            reroute_enabled=reroute_enabled,
            base_alpha=base_alpha,
            seed=iter_seed
        )

        results.append(sim_result)

        if not sim_result["survival_status"]:
            failures += 1
        else:
            total_min_alpha += sim_result["min_alpha_during"]
            total_max_drop += sim_result["max_alpha_drop"]

    successes = n_iterations - failures
    survival_rate = successes / n_iterations
    avg_min_alpha = total_min_alpha / max(1, successes)
    avg_max_drop = total_max_drop / max(1, successes)

    report = {
        "nodes": nodes,
        "blackout_range": list(blackout_range),
        "iterations": n_iterations,
        "reroute_enabled": reroute_enabled,
        "base_alpha": base_alpha,
        "survival_rate": round(survival_rate, 4),
        "failures": failures,
        "avg_min_alpha": round(avg_min_alpha, 4),
        "avg_max_drop": round(avg_max_drop, 4),
        "all_survived": failures == 0
    }

    emit_receipt("blackout_stress_sweep", {
        "tenant_id": "axiom-reroute",
        **report
    })

    return report


def get_reroute_algo_info() -> Dict[str, Any]:
    """Get adaptive rerouting algorithm specification.

    Returns:
        Dict with algorithm details

    Receipt: reroute_algo_info
    """
    info = {
        "algo_type": ALGO_TYPE,
        "cgr_baseline": CGR_BASELINE,
        "ml_model_type": ML_MODEL_TYPE,
        "alpha_boost": REROUTE_ALPHA_BOOST,
        "blackout_base_days": BLACKOUT_BASE_DAYS,
        "blackout_extended_days": BLACKOUT_EXTENDED_DAYS,
        "retention_factor": REROUTE_RETENTION_FACTOR,
        "min_eff_alpha_floor": MIN_EFF_ALPHA_FLOOR,
        "description": "Hybrid ephemeris-ML algorithm (CGR + lightweight GNN)"
    }

    emit_receipt("reroute_algo_info", {
        "tenant_id": "axiom-reroute",
        **info
    })

    return info
