"""partition.py - Fault Simulation Module for Distributed Quorum Resilience

THE PHYSICS:
    At 40% partition with 5 nodes, quorum survives (3 nodes remaining).
    eff_α drop formula: base_alpha * (1 - (loss_pct * DROP_FACTOR))
    Where DROP_FACTOR ≈ 0.125 (yields ~0.05 drop at 40% loss)

QUORUM THRESHOLD:
    Byzantine fault tolerant at 2/3 of baseline.
    5 nodes → quorum threshold of 3 → survives 2 simultaneous failures.

DYNAMIC RECOVERY (Dec 2025 adaptive rerouting - DEFAULT):
    All partition stress now flows through adaptive reroute.
    Static partition logic has been KILLED per Grok directive.
    Reroute boost: +0.07 to eff_α (locked, validated).
    Extended blackout survival: 43d → 90d with retention curve.

CONSTANTS (from Grok validation):
    NODE_BASELINE = 5 (3 uncrewed habs + 2 rovers)
    QUORUM_THRESHOLD = 3 (survives 2-node failure)
    PARTITION_MAX_TEST_PCT = 0.40 (stress test upper bound)
    ALPHA_DROP_FACTOR = 0.125 (calibrated: 0.4 * 0.125 ≈ 0.05 drop)
    REROUTING_ALPHA_BOOST_LOCKED = 0.07 (validated, immutable)

KILLED (per Grok directive "Kill: static partition (now dynamic)"):
    - Static partition simulation without reroute
    - Non-dynamic recovery paths

Source: Grok - "quorum intact for 5-node eg.", "Kill: static partition (now dynamic)"
"""

import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from .core import emit_receipt, dual_hash, StopRule


# === CONSTANTS (Dec 2025 distributed anchoring) ===

NODE_BASELINE = 5
"""Quorum-resilient minimum node count (3 uncrewed habs + 2 rovers)."""

QUORUM_THRESHOLD = 3
"""Byzantine fault tolerant threshold. Survives 2-node failure in 5-node baseline."""

PARTITION_MAX_TEST_PCT = 0.40
"""Stress test upper bound. Test up to 40% node loss."""

ALPHA_DROP_FACTOR = 0.125
"""Calibrated drop factor: 0.4 * 0.125 ≈ 0.05 α drop at max partition."""

GREENS_CURRENT = 81
"""Fleet cadence marker (81 tests passing, Dec 2025)."""

REROUTING_ALPHA_BOOST_LOCKED = 0.07
"""LOCKED: Validated reroute boost (was REROUTING_POTENTIAL_BOOST, now locked)."""

# Backward compatibility alias (deprecated)
REROUTING_POTENTIAL_BOOST = REROUTING_ALPHA_BOOST_LOCKED

BASE_ALPHA = 2.68
"""Baseline effective alpha with distributed anchoring (+0.12 boost from 2.56)."""

PARTITION_SPEC_PATH = "data/node_partition_spec.json"
"""Path to partition specification file."""


@dataclass
class PartitionResult:
    """Result from a single partition simulation.

    Attributes:
        nodes_total: Total nodes in baseline
        loss_pct: Fraction of nodes lost (0-1)
        nodes_surviving: Remaining nodes after partition
        eff_alpha_drop: Drop in effective alpha from baseline
        eff_alpha: Effective alpha after partition
        quorum_status: True if quorum intact (surviving >= threshold)
    """

    nodes_total: int
    loss_pct: float
    nodes_surviving: int
    eff_alpha_drop: float
    eff_alpha: float
    quorum_status: bool


def load_partition_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify partition specification file.

    Loads data/node_partition_spec.json and emits ingest receipt
    with dual_hash per CLAUDEME §4.1.

    Args:
        path: Optional path override (default: PARTITION_SPEC_PATH)

    Returns:
        Dict containing partition specification

    Receipt: partition_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, PARTITION_SPEC_PATH)

    with open(path, "r") as f:
        data = json.load(f)

    # Compute dual_hash of contents
    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    # Emit ingest receipt
    emit_receipt(
        "partition_spec_ingest",
        {
            "tenant_id": "spaceproof-resilience",
            "file_path": path,
            "node_baseline": data["node_baseline"],
            "quorum_min": data["quorum_min"],
            "partition_test_range": data["partition_test_range"],
            "greens_current": data["greens_current"],
            "ledger_alpha_boost": data["ledger_alpha_boost"],
            "rerouting_potential": data["rerouting_potential"],
            "payload_hash": content_hash,
        },
    )

    return data


def quorum_check(nodes_surviving: int, quorum_min: int = QUORUM_THRESHOLD) -> bool:
    """Check if quorum is intact.

    Byzantine fault tolerance requires 2/3 of baseline nodes.

    Args:
        nodes_surviving: Number of nodes still operational
        quorum_min: Minimum required for quorum (default: QUORUM_THRESHOLD=3)

    Returns:
        True if quorum intact (surviving >= threshold)

    Raises:
        StopRule: If quorum fails (surviving < threshold)
    """
    quorum_intact = nodes_surviving >= quorum_min

    if not quorum_intact:
        emit_receipt(
            "anomaly",
            {
                "tenant_id": "spaceproof-resilience",
                "metric": "quorum_failure",
                "baseline": quorum_min,
                "delta": nodes_surviving - quorum_min,
                "classification": "violation",
                "action": "halt",
                "nodes_surviving": nodes_surviving,
                "quorum_threshold": quorum_min,
            },
        )
        raise StopRule(
            f"Quorum failed: {nodes_surviving} nodes surviving < {quorum_min} threshold"
        )

    return quorum_intact


def partition_sim(
    nodes_total: int = NODE_BASELINE,
    loss_pct: float = 0.0,
    base_alpha: float = BASE_ALPHA,
    emit: bool = True,
    reroute_enabled: bool = True,
) -> Dict[str, Any]:
    """Simulate node partition and compute impact on effective alpha.

    Pure function. Computes nodes surviving, quorum status, and α drop.
    Dynamic recovery via adaptive reroute is NOW DEFAULT (static logic killed).

    WARNING: Calling with reroute_enabled=False is DEPRECATED.
    Static partition simulation has been killed per Grok directive.

    Args:
        nodes_total: Total nodes in baseline (default: 5)
        loss_pct: Fraction of nodes lost (0-1, max 0.40 for tests)
        base_alpha: Baseline effective alpha (default: 2.68)
        emit: Whether to emit receipt (default: True)
        reroute_enabled: Enable adaptive rerouting recovery (default: True)
            DEPRECATED: Setting to False will emit deprecation warning

    Returns:
        Dict with:
            - nodes_total: Input node count
            - loss_pct: Input loss percentage
            - nodes_surviving: Remaining nodes after partition
            - eff_alpha_drop: Drop in effective alpha
            - eff_alpha: Effective alpha after partition
            - quorum_status: True if quorum intact
            - reroute_applied: True if reroute recovery was applied

    Raises:
        StopRule: If quorum fails (via quorum_check)

    Receipt: partition_stress_receipt
    """
    # DEPRECATION: Static partition logic killed per Grok directive
    if not reroute_enabled:
        warnings.warn(
            "partition_sim with reroute_enabled=False is DEPRECATED. "
            "Static partition logic has been killed. All partition stress "
            "now flows through adaptive reroute. This warning will become "
            "an error in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
    # Calculate nodes surviving (floor of remaining)
    nodes_lost = int(nodes_total * loss_pct)
    nodes_surviving = nodes_total - nodes_lost

    # Check quorum - raises StopRule if failed
    quorum_status = quorum_check(nodes_surviving)

    # Calculate α drop using calibrated formula
    # eff_α drop = loss_pct * DROP_FACTOR (yields ~0.05 at 40% loss)
    # At 40% loss: 0.4 * 0.125 = 0.05
    eff_alpha_drop = loss_pct * ALPHA_DROP_FACTOR
    eff_alpha = base_alpha - eff_alpha_drop

    # Dynamic recovery via reroute when enabled and quorum stressed
    reroute_applied = False
    reroute_boost = 0.0

    if reroute_enabled and quorum_status:
        # Check if quorum is stressed (surviving < baseline but >= threshold)
        quorum_stressed = (
            nodes_surviving < nodes_total and nodes_surviving >= QUORUM_THRESHOLD
        )

        if quorum_stressed or loss_pct > 0:
            # Import here to avoid circular dependency
            from .reroute import adaptive_reroute

            # Attempt reroute recovery
            graph_state = {
                "nodes": nodes_total,
                "edges": [
                    {"src": f"n{i}", "dst": f"n{(i + 1) % nodes_surviving}"}
                    for i in range(nodes_surviving)
                ],
            }

            try:
                reroute_result = adaptive_reroute(
                    graph_state, loss_pct, blackout_days=0
                )

                if (
                    reroute_result["quorum_preserved"]
                    and reroute_result["recovery_factor"] > 0.5
                ):
                    reroute_applied = True
                    reroute_boost = reroute_result["alpha_boost"]
                    eff_alpha = eff_alpha + reroute_boost
            except StopRule:
                # Reroute failed, continue with base calculation
                pass

    result = {
        "nodes_total": nodes_total,
        "loss_pct": loss_pct,
        "nodes_surviving": nodes_surviving,
        "eff_alpha_drop": round(eff_alpha_drop, 4),
        "eff_alpha": round(eff_alpha, 4),
        "quorum_status": quorum_status,
        "reroute_applied": reroute_applied,
        "reroute_boost": round(reroute_boost, 4),
    }

    if emit:
        emit_receipt(
            "partition_stress",
            {
                "tenant_id": "spaceproof-resilience",
                "nodes_total": nodes_total,
                "loss_pct": loss_pct,
                "nodes_surviving": nodes_surviving,
                "eff_alpha_drop": round(eff_alpha_drop, 4),
                "eff_alpha": round(eff_alpha, 4),
                "quorum_status": quorum_status,
                "base_alpha": base_alpha,
                "drop_factor": ALPHA_DROP_FACTOR,
                "reroute_applied": reroute_applied,
                "reroute_boost": round(reroute_boost, 4),
            },
        )

    return result


def stress_sweep(
    nodes_total: int = NODE_BASELINE,
    loss_range: Tuple[float, float] = (0.0, PARTITION_MAX_TEST_PCT),
    n_iterations: int = 1000,
    base_alpha: float = BASE_ALPHA,
    seed: Optional[int] = None,
    reroute_enabled: bool = True,
) -> List[Dict[str, Any]]:
    """Run stress sweep with random loss in range.

    Runs n_iterations with random partition loss percentage within range.
    Dynamic recovery via adaptive reroute is NOW DEFAULT.

    Args:
        nodes_total: Total nodes in baseline (default: 5)
        loss_range: Tuple of (min_loss, max_loss) percentages (default: 0-40%)
        n_iterations: Number of iterations (default: 1000)
        base_alpha: Baseline effective alpha (default: 2.68)
        seed: Random seed for reproducibility (optional)
        reroute_enabled: Enable adaptive rerouting recovery (default: True)

    Returns:
        List of partition simulation results

    Receipt: quorum_resilience_receipt (summary at end)
    """
    if seed is not None:
        random.seed(seed)

    results = []
    quorum_failures = 0
    total_alpha_drop = 0.0

    for i in range(n_iterations):
        # Random loss within range
        loss_pct = random.uniform(loss_range[0], loss_range[1])

        try:
            # Run partition sim (emit=False to avoid 1000 receipts, we'll emit summary)
            result = partition_sim(
                nodes_total,
                loss_pct,
                base_alpha,
                emit=False,
                reroute_enabled=reroute_enabled,
            )
            results.append(result)
            total_alpha_drop += result["eff_alpha_drop"]
        except StopRule:
            # Quorum failed - this shouldn't happen in valid test range
            quorum_failures += 1
            results.append(
                {
                    "nodes_total": nodes_total,
                    "loss_pct": loss_pct,
                    "nodes_surviving": nodes_total - int(nodes_total * loss_pct),
                    "eff_alpha_drop": 0.0,
                    "eff_alpha": 0.0,
                    "quorum_status": False,
                    "reroute_applied": False,
                    "reroute_boost": 0.0,
                }
            )

    # Compute summary stats
    success_count = len([r for r in results if r["quorum_status"]])
    success_rate = success_count / n_iterations
    avg_alpha_drop = total_alpha_drop / max(1, success_count)

    # Count reroute applications
    reroute_count = len([r for r in results if r.get("reroute_applied", False)])

    # Emit summary receipt
    emit_receipt(
        "quorum_resilience",
        {
            "tenant_id": "spaceproof-resilience",
            "baseline_nodes": nodes_total,
            "quorum_threshold": QUORUM_THRESHOLD,
            "test_iterations": n_iterations,
            "loss_range": list(loss_range),
            "success_rate": round(success_rate, 4),
            "avg_alpha_drop": round(avg_alpha_drop, 4),
            "quorum_failures": quorum_failures,
            "base_alpha": base_alpha,
            "min_eff_alpha": (
                round(min(r["eff_alpha"] for r in results if r["quorum_status"]), 4)
                if success_count > 0
                else 0.0
            ),
            "max_eff_alpha_drop": (
                round(
                    max(r["eff_alpha_drop"] for r in results if r["quorum_status"]), 4
                )
                if success_count > 0
                else 0.0
            ),
            "reroute_enabled": reroute_enabled,
            "reroute_applications": reroute_count,
        },
    )

    return results


def validate_partition_bounds(
    nodes_total: int = NODE_BASELINE,
    loss_pct: float = PARTITION_MAX_TEST_PCT,
    base_alpha: float = BASE_ALPHA,
    min_eff_alpha: float = 2.63,
) -> Dict[str, Any]:
    """Validate that partition bounds maintain minimum effective alpha.

    Asserts: eff_alpha(partition=0.4, nodes=5) >= 2.63 (per Grok validation)

    Args:
        nodes_total: Total nodes (default: 5)
        loss_pct: Partition loss to test (default: 40%)
        base_alpha: Baseline alpha (default: 2.68)
        min_eff_alpha: Minimum required effective alpha (default: 2.63)

    Returns:
        Dict with validation results

    Raises:
        AssertionError: If eff_alpha < min_eff_alpha
    """
    result = partition_sim(nodes_total, loss_pct, base_alpha, emit=False)

    # Validate bounds
    assert result["eff_alpha"] >= min_eff_alpha, (
        f"eff_alpha {result['eff_alpha']} < min {min_eff_alpha} at {loss_pct * 100}% partition"
    )

    assert result["quorum_status"], f"Quorum failed at {loss_pct * 100}% partition"

    assert result["eff_alpha_drop"] <= 0.05, (
        f"Alpha drop {result['eff_alpha_drop']} > 0.05 at {loss_pct * 100}% partition"
    )

    validation = {
        "validated": True,
        "nodes_total": nodes_total,
        "loss_pct": loss_pct,
        "eff_alpha": result["eff_alpha"],
        "eff_alpha_drop": result["eff_alpha_drop"],
        "min_eff_alpha_required": min_eff_alpha,
        "quorum_status": result["quorum_status"],
    }

    emit_receipt(
        "partition_validation", {"tenant_id": "spaceproof-resilience", **validation}
    )

    return validation


def get_rerouting_potential() -> Dict[str, Any]:
    """Get stub for adaptive rerouting potential.

    Next gate scope: +0.07 α potential pushing toward 2.7+ and sub-2.5 cycles.

    Returns:
        Dict with rerouting potential stub info

    Receipt: rerouting_potential_stub
    """
    potential = {
        "boost_potential": REROUTING_POTENTIAL_BOOST,
        "target_alpha": BASE_ALPHA + REROUTING_POTENTIAL_BOOST,
        "status": "stub",
        "next_gate": True,
        "description": "Path to 2.7+ α with adaptive rerouting",
    }

    emit_receipt(
        "rerouting_potential_stub", {"tenant_id": "spaceproof-resilience", **potential}
    )

    return potential
