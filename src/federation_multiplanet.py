"""Multi-planet federation protocols.

Implements federated consensus across Mars, Venus, Mercury, and Jovian system
with lag-tolerant consensus, cross-planet arbitration, and eventual consistency.

Receipt Types:
    - federation_config_receipt: Configuration loaded
    - federation_init_receipt: Federation initialized
    - federation_planet_receipt: Planet added/removed
    - federation_sync_receipt: Sync completed
    - federation_consensus_receipt: Consensus result
    - federation_arbitration_receipt: Arbitration result
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core import TENANT_ID, dual_hash, emit_receipt

# Federation constants
FEDERATION_PLANETS = ["mars", "venus", "mercury", "jovian_system"]
DEFAULT_PLANETS = FEDERATION_PLANETS  # Backward-compatibility alias
FEDERATION_CONSENSUS_LAG_TOLERANCE = True
FEDERATION_AUTONOMY_MINIMUM = 0.995
FEDERATION_SYNC_INTERVAL_HOURS = 24
FEDERATION_ARBITRATION_ENABLED = True


@dataclass
class PlanetProfile:
    """Profile for a federated planet."""

    name: str
    gravity_g: float
    latency_min: float
    latency_max: float
    node_count: int
    autonomy_target: float
    status: str = "active"
    last_sync: Optional[str] = None


@dataclass
class FederationState:
    """Current federation state."""

    planets: Dict[str, PlanetProfile] = field(default_factory=dict)
    initialized: bool = False
    consensus_round: int = 0
    last_sync: Optional[str] = None
    disputes: List[Dict[str, Any]] = field(default_factory=list)


# Global federation state
_federation_state = FederationState()


def load_federation_config() -> Dict[str, Any]:
    """Load federation configuration from spec file.

    Returns:
        dict: Federation configuration.

    Receipt:
        federation_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "federation_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get(
        "federation_config",
        {
            "planets": FEDERATION_PLANETS,
            "consensus_lag_tolerance": FEDERATION_CONSENSUS_LAG_TOLERANCE,
            "autonomy_minimum": FEDERATION_AUTONOMY_MINIMUM,
            "sync_interval_hours": FEDERATION_SYNC_INTERVAL_HOURS,
            "arbitration_enabled": FEDERATION_ARBITRATION_ENABLED,
        },
    )

    emit_receipt(
        "federation_config_receipt",
        {
            "receipt_type": "federation_config_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "planets": config.get("planets", FEDERATION_PLANETS),
            "autonomy_minimum": config.get("autonomy_minimum", FEDERATION_AUTONOMY_MINIMUM),
            "arbitration_enabled": config.get(
                "arbitration_enabled", FEDERATION_ARBITRATION_ENABLED
            ),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )
    return config


def init_federation(planets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Initialize multi-planet federation.

    Args:
        planets: List of planets to include (default from config).

    Returns:
        dict: Initialization result.

    Receipt:
        federation_init_receipt
    """
    global _federation_state

    config = load_federation_config()
    if planets is None:
        planets = config.get("planets", FEDERATION_PLANETS)

    # Load planet profiles
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "federation_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    planet_profiles = spec.get("planet_profiles", {})

    _federation_state.planets = {}
    for planet in planets:
        profile = planet_profiles.get(planet, {})
        _federation_state.planets[planet] = PlanetProfile(
            name=planet,
            gravity_g=profile.get("gravity_g", 0.38),
            latency_min=profile.get("latency_min", 3),
            latency_max=profile.get("latency_max", 22),
            node_count=profile.get("node_count", 5),
            autonomy_target=profile.get("autonomy_target", 0.999),
            status="active",
            last_sync=datetime.utcnow().isoformat() + "Z",
        )

    _federation_state.initialized = True
    _federation_state.last_sync = datetime.utcnow().isoformat() + "Z"

    result = {
        "initialized": True,
        "planets": list(_federation_state.planets.keys()),
        "planet_count": len(_federation_state.planets),
        "consensus_lag_tolerance": config.get(
            "consensus_lag_tolerance", FEDERATION_CONSENSUS_LAG_TOLERANCE
        ),
        "arbitration_enabled": config.get(
            "arbitration_enabled", FEDERATION_ARBITRATION_ENABLED
        ),
    }

    emit_receipt(
        "federation_init_receipt",
        {
            "receipt_type": "federation_init_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "initialized": True,
            "planets": list(_federation_state.planets.keys()),
            "planet_count": len(_federation_state.planets),
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def add_planet(planet: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add planet to federation.

    Args:
        planet: Planet name.
        profile: Planet profile (uses defaults if None).

    Returns:
        dict: Addition result.

    Receipt:
        federation_planet_receipt
    """
    global _federation_state

    if not _federation_state.initialized:
        init_federation()

    if profile is None:
        profile = {
            "gravity_g": 0.38,
            "latency_min": 3,
            "latency_max": 22,
            "node_count": 5,
            "autonomy_target": 0.999,
        }

    _federation_state.planets[planet] = PlanetProfile(
        name=planet,
        gravity_g=profile.get("gravity_g", 0.38),
        latency_min=profile.get("latency_min", 3),
        latency_max=profile.get("latency_max", 22),
        node_count=profile.get("node_count", 5),
        autonomy_target=profile.get("autonomy_target", 0.999),
        status="active",
        last_sync=datetime.utcnow().isoformat() + "Z",
    )

    result = {
        "added": True,
        "planet": planet,
        "total_planets": len(_federation_state.planets),
    }

    emit_receipt(
        "federation_planet_receipt",
        {
            "receipt_type": "federation_planet_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "add",
            "planet": planet,
            "total_planets": len(_federation_state.planets),
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def remove_planet(planet: str) -> Dict[str, Any]:
    """Remove planet from federation.

    Args:
        planet: Planet name.

    Returns:
        dict: Removal result.

    Receipt:
        federation_planet_receipt
    """
    global _federation_state

    if planet in _federation_state.planets:
        del _federation_state.planets[planet]
        removed = True
    else:
        removed = False

    result = {
        "removed": removed,
        "planet": planet,
        "total_planets": len(_federation_state.planets),
    }

    emit_receipt(
        "federation_planet_receipt",
        {
            "receipt_type": "federation_planet_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "remove",
            "planet": planet,
            "removed": removed,
            "total_planets": len(_federation_state.planets),
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def sync_federation() -> Dict[str, Any]:
    """Synchronize all planets in federation.

    Returns:
        dict: Sync result.

    Receipt:
        federation_sync_receipt
    """
    global _federation_state

    if not _federation_state.initialized:
        init_federation()

    sync_results = []
    for planet_name, profile in _federation_state.planets.items():
        # Simulate sync with latency
        sync_latency = random.uniform(profile.latency_min, profile.latency_max)
        sync_success = random.random() > 0.01  # 99% success

        profile.last_sync = datetime.utcnow().isoformat() + "Z"
        profile.status = "active" if sync_success else "sync_failed"

        sync_results.append(
            {
                "planet": planet_name,
                "sync_success": sync_success,
                "latency_min": sync_latency,
            }
        )

    _federation_state.last_sync = datetime.utcnow().isoformat() + "Z"

    all_synced = all(r["sync_success"] for r in sync_results)

    result = {
        "sync_complete": all_synced,
        "planets_synced": sum(1 for r in sync_results if r["sync_success"]),
        "total_planets": len(sync_results),
        "sync_results": sync_results,
        "timestamp": _federation_state.last_sync,
    }

    emit_receipt(
        "federation_sync_receipt",
        {
            "receipt_type": "federation_sync_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "sync_complete": all_synced,
            "planets_synced": result["planets_synced"],
            "total_planets": len(sync_results),
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def run_consensus(proposal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run federated consensus across planets.

    Args:
        proposal: Proposal to vote on (uses test proposal if None).

    Returns:
        dict: Consensus result.

    Receipt:
        federation_consensus_receipt
    """
    global _federation_state

    if not _federation_state.initialized:
        init_federation()

    _federation_state.consensus_round += 1

    if proposal is None:
        proposal = {
            "id": f"proposal_{_federation_state.consensus_round}",
            "type": "resource_allocation",
            "data": {"resource": "bandwidth", "amount": 100},
        }

    # Collect votes from each planet
    votes = []
    for planet_name, profile in _federation_state.planets.items():
        # Simulate vote with lag compensation
        vote_latency = random.uniform(profile.latency_min, profile.latency_max)
        vote = random.random() > 0.1  # 90% approval rate
        votes.append(
            {
                "planet": planet_name,
                "vote": vote,
                "latency_min": vote_latency,
            }
        )

    # Calculate consensus
    config = load_federation_config()
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "federation_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)
    quorum = spec.get("arbitration_config", {}).get("quorum_fraction", 0.51)

    approval_count = sum(1 for v in votes if v["vote"])
    approval_rate = approval_count / len(votes)
    consensus_reached = approval_rate >= quorum

    result = {
        "consensus_reached": consensus_reached,
        "consensus_round": _federation_state.consensus_round,
        "proposal_id": proposal.get("id", "unknown"),
        "approval_rate": approval_rate,
        "quorum": quorum,
        "votes": votes,
    }

    emit_receipt(
        "federation_consensus_receipt",
        {
            "receipt_type": "federation_consensus_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "consensus_reached": consensus_reached,
            "consensus_round": _federation_state.consensus_round,
            "approval_rate": approval_rate,
            "quorum": quorum,
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def arbitrate_dispute(dispute: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Arbitrate cross-planet dispute.

    Args:
        dispute: Dispute to arbitrate (uses test dispute if None).

    Returns:
        dict: Arbitration result.

    Receipt:
        federation_arbitration_receipt
    """
    global _federation_state

    if not _federation_state.initialized:
        init_federation()

    if dispute is None:
        dispute = {
            "id": f"dispute_{len(_federation_state.disputes) + 1}",
            "type": "resource_conflict",
            "parties": ["mars", "venus"],
            "subject": "bandwidth_allocation",
        }

    # Collect arbitration votes
    arbitration_votes = []
    for planet_name, profile in _federation_state.planets.items():
        if planet_name not in dispute.get("parties", []):
            # Neutral parties vote
            vote = random.choice(dispute.get("parties", ["mars"]))
            arbitration_votes.append({"planet": planet_name, "vote": vote})

    # Determine winner by majority
    if arbitration_votes:
        vote_counts = {}
        for v in arbitration_votes:
            vote_counts[v["vote"]] = vote_counts.get(v["vote"], 0) + 1
        winner = max(vote_counts, key=vote_counts.get)
    else:
        # No neutral parties, use first party
        winner = dispute.get("parties", ["unknown"])[0]

    dispute["resolution"] = winner
    dispute["resolved"] = True
    _federation_state.disputes.append(dispute)

    result = {
        "resolved": True,
        "dispute_id": dispute.get("id", "unknown"),
        "winner": winner,
        "arbitration_votes": arbitration_votes,
    }

    emit_receipt(
        "federation_arbitration_receipt",
        {
            "receipt_type": "federation_arbitration_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "resolved": True,
            "dispute_id": dispute.get("id", "unknown"),
            "winner": winner,
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def measure_federation_health() -> Dict[str, Any]:
    """Measure overall federation health.

    Returns:
        dict: Health metrics.
    """
    global _federation_state

    if not _federation_state.initialized:
        init_federation()

    active_planets = sum(
        1 for p in _federation_state.planets.values() if p.status == "active"
    )
    total_planets = len(_federation_state.planets)

    avg_latency = 0.0
    if _federation_state.planets:
        avg_latency = sum(
            (p.latency_min + p.latency_max) / 2 for p in _federation_state.planets.values()
        ) / len(_federation_state.planets)

    return {
        "healthy": active_planets == total_planets,
        "active_planets": active_planets,
        "total_planets": total_planets,
        "availability": active_planets / max(1, total_planets),
        "avg_latency_min": avg_latency,
        "consensus_rounds": _federation_state.consensus_round,
        "disputes_resolved": len(_federation_state.disputes),
    }


def get_planet_status(planet: str) -> Dict[str, Any]:
    """Get status of a single planet.

    Args:
        planet: Planet name.

    Returns:
        dict: Planet status.
    """
    global _federation_state

    if planet not in _federation_state.planets:
        return {"error": f"Planet {planet} not in federation"}

    profile = _federation_state.planets[planet]
    return {
        "planet": planet,
        "status": profile.status,
        "gravity_g": profile.gravity_g,
        "latency_range": f"{profile.latency_min}-{profile.latency_max} min",
        "node_count": profile.node_count,
        "autonomy_target": profile.autonomy_target,
        "last_sync": profile.last_sync,
    }


def get_federation_status() -> Dict[str, Any]:
    """Get full federation status.

    Returns:
        dict: Federation status.
    """
    global _federation_state

    return {
        "initialized": _federation_state.initialized,
        "planets": list(_federation_state.planets.keys()),
        "planet_count": len(_federation_state.planets),
        "consensus_round": _federation_state.consensus_round,
        "last_sync": _federation_state.last_sync,
        "disputes_count": len(_federation_state.disputes),
        "health": measure_federation_health(),
    }


def simulate_partition(planets: List[str]) -> Dict[str, Any]:
    """Simulate network partition for specified planets.

    Args:
        planets: Planets to partition.

    Returns:
        dict: Partition simulation result.
    """
    global _federation_state

    partitioned = []
    for planet in planets:
        if planet in _federation_state.planets:
            _federation_state.planets[planet].status = "partitioned"
            partitioned.append(planet)

    return {
        "partitioned": partitioned,
        "partition_count": len(partitioned),
        "remaining_active": sum(
            1 for p in _federation_state.planets.values() if p.status == "active"
        ),
    }


def simulate_recovery(planets: List[str]) -> Dict[str, Any]:
    """Simulate recovery from partition.

    Args:
        planets: Planets to recover.

    Returns:
        dict: Recovery simulation result.
    """
    global _federation_state

    recovered = []
    for planet in planets:
        if planet in _federation_state.planets:
            _federation_state.planets[planet].status = "active"
            _federation_state.planets[planet].last_sync = (
                datetime.utcnow().isoformat() + "Z"
            )
            recovered.append(planet)

    return {
        "recovered": recovered,
        "recovery_count": len(recovered),
        "active_planets": sum(
            1 for p in _federation_state.planets.values() if p.status == "active"
        ),
    }
