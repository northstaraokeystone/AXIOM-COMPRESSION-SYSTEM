"""Cross-planet pattern migration for D19.

Migrate successful patterns between planetary swarms.
"""

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19 MIGRATION CONSTANTS ===

MIGRATION_LATENCY_TOLERANCE_MS = 5000
"""Maximum latency tolerance for migration."""

MIGRATION_FITNESS_THRESHOLD = 0.80
"""Minimum fitness at source for migration."""

MIGRATION_AGE_THRESHOLD = 50
"""Minimum cycles for migration eligibility."""


@dataclass
class MigrationTransfer:
    """Pattern transfer record."""

    transfer_id: str
    pattern_id: str
    source_planet: str
    dest_planet: str
    serialized_pattern: Dict
    status: str = "pending"
    latency_ms: float = 0.0
    created_at: str = ""


@dataclass
class MigrationManager:
    """Manager for cross-planet pattern migration."""

    manager_id: str
    transfers: Dict[str, MigrationTransfer] = field(default_factory=dict)
    successful_migrations: int = 0
    failed_migrations: int = 0


def init_migration(config: Dict = None) -> MigrationManager:
    """Initialize migration manager.

    Args:
        config: Optional configuration dict

    Returns:
        MigrationManager instance
    """
    manager_id = str(uuid.uuid4())[:8]
    return MigrationManager(manager_id=manager_id)


def identify_migration_candidates(manager: MigrationManager, source_planet: str) -> List[Dict]:
    """Identify patterns ready for migration.

    Migration criteria:
    - Fitness at source >= 0.80
    - Pattern age >= 50 cycles
    - Similar conditions at destination
    - Relay latency < 5000ms

    Args:
        manager: MigrationManager instance
        source_planet: Source planet identifier

    Returns:
        List of candidate patterns

    Receipt: migration_candidate_receipt
    """
    # Simulate identifying candidates based on planetary swarm
    candidates = []

    # Generate synthetic candidates for simulation
    for i in range(random.randint(2, 5)):
        candidate = {
            "pattern_id": f"pattern_{source_planet}_{i:02d}",
            "fitness": round(random.uniform(0.75, 0.95), 4),
            "age_cycles": random.randint(40, 100),
            "source_planet": source_planet,
        }

        if candidate["fitness"] >= MIGRATION_FITNESS_THRESHOLD and candidate["age_cycles"] >= MIGRATION_AGE_THRESHOLD:
            candidates.append(candidate)

    emit_receipt(
        "migration_candidate",
        {
            "receipt_type": "migration_candidate",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "manager_id": manager.manager_id,
            "source_planet": source_planet,
            "candidates_found": len(candidates),
            "payload_hash": dual_hash(json.dumps({"source": source_planet, "count": len(candidates)}, sort_keys=True)),
        },
    )

    return candidates


def prepare_pattern_transfer(manager: MigrationManager, pattern: Dict, dest_planet: str) -> Dict[str, Any]:
    """Serialize pattern for cross-planet transfer.

    Args:
        manager: MigrationManager instance
        pattern: Pattern dict to transfer
        dest_planet: Destination planet

    Returns:
        Transfer preparation result

    Receipt: transfer_preparation_receipt
    """
    transfer_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat() + "Z"

    # Serialize pattern for transfer
    serialized = {
        "pattern_id": pattern.get("pattern_id"),
        "fitness": pattern.get("fitness"),
        "source_planet": pattern.get("source_planet"),
        "serialized_at": now,
        "spline_coefficients": [[random.gauss(0, 0.1) for _ in range(5)] for _ in range(3)],
        "metadata": {
            "age_cycles": pattern.get("age_cycles", 0),
            "compression_ratio": random.uniform(0.85, 0.95),
        },
    }

    transfer = MigrationTransfer(
        transfer_id=transfer_id,
        pattern_id=pattern.get("pattern_id", "unknown"),
        source_planet=pattern.get("source_planet", "unknown"),
        dest_planet=dest_planet,
        serialized_pattern=serialized,
        status="prepared",
        created_at=now,
    )

    manager.transfers[transfer_id] = transfer

    result = {
        "transfer_id": transfer_id,
        "pattern_id": pattern.get("pattern_id"),
        "source_planet": pattern.get("source_planet"),
        "dest_planet": dest_planet,
        "status": "prepared",
        "serialized_size": len(json.dumps(serialized)),
    }

    emit_receipt(
        "transfer_preparation",
        {
            "receipt_type": "transfer_preparation",
            "tenant_id": TENANT_ID,
            "ts": now,
            "manager_id": manager.manager_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def execute_transfer(manager: MigrationManager, transfer: Dict) -> Dict[str, Any]:
    """Execute transfer via relay mesh.

    Args:
        manager: MigrationManager instance
        transfer: Transfer dict

    Returns:
        Transfer execution result

    Receipt: transfer_execution_receipt
    """
    transfer_id = transfer.get("transfer_id")
    if transfer_id not in manager.transfers:
        return {"error": "transfer_not_found", "transfer_id": transfer_id}

    xfer = manager.transfers[transfer_id]
    start_time = time.time()

    # Simulate relay transmission
    # Latency based on planetary distance
    planets = ["mars", "venus", "mercury", "jovian"]
    src_idx = planets.index(xfer.source_planet) if xfer.source_planet in planets else 0
    dst_idx = planets.index(xfer.dest_planet) if xfer.dest_planet in planets else 1
    distance_factor = abs(src_idx - dst_idx)

    latency_ms = 500 + (distance_factor * 1000) + random.uniform(0, 500)
    xfer.latency_ms = latency_ms

    # Check latency tolerance
    if latency_ms <= MIGRATION_LATENCY_TOLERANCE_MS:
        xfer.status = "transferred"
        success = True
    else:
        xfer.status = "failed_latency"
        success = False

    elapsed_ms = (time.time() - start_time) * 1000

    result = {
        "transfer_id": transfer_id,
        "pattern_id": xfer.pattern_id,
        "source_planet": xfer.source_planet,
        "dest_planet": xfer.dest_planet,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "status": xfer.status,
    }

    emit_receipt(
        "transfer_execution",
        {
            "receipt_type": "transfer_execution",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "manager_id": manager.manager_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def validate_migration(manager: MigrationManager, pattern_id: str, dest_planet: str) -> bool:
    """Validate pattern works at destination.

    Args:
        manager: MigrationManager instance
        pattern_id: Pattern identifier
        dest_planet: Destination planet

    Returns:
        True if pattern is valid at destination

    Receipt: migration_validation_receipt
    """
    # Simulate validation
    valid = random.random() > 0.1  # 90% success rate

    if valid:
        manager.successful_migrations += 1
    else:
        manager.failed_migrations += 1

    emit_receipt(
        "migration_validation",
        {
            "receipt_type": "migration_validation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "manager_id": manager.manager_id,
            "pattern_id": pattern_id,
            "dest_planet": dest_planet,
            "valid": valid,
            "payload_hash": dual_hash(json.dumps({"pattern_id": pattern_id, "valid": valid}, sort_keys=True)),
        },
    )

    return valid


def adapt_to_destination(manager: MigrationManager, pattern: Dict, dest_planet: str) -> Dict[str, Any]:
    """Adapt pattern to destination planet conditions.

    Args:
        manager: MigrationManager instance
        pattern: Pattern dict
        dest_planet: Destination planet

    Returns:
        Adapted pattern dict

    Receipt: adaptation_receipt
    """
    # Planet-specific adaptations
    adaptations = {
        "mars": {"latency_factor": 1.2, "entropy_adjustment": 0.05},
        "venus": {"latency_factor": 0.9, "entropy_adjustment": -0.03},
        "mercury": {"latency_factor": 0.7, "entropy_adjustment": -0.05},
        "jovian": {"latency_factor": 2.0, "entropy_adjustment": 0.10},
    }

    adaptation = adaptations.get(dest_planet, {"latency_factor": 1.0, "entropy_adjustment": 0.0})

    adapted = {
        **pattern,
        "dest_planet": dest_planet,
        "adapted": True,
        "latency_factor": adaptation["latency_factor"],
        "entropy_adjustment": adaptation["entropy_adjustment"],
        "fitness": pattern.get("fitness", 0.8) + adaptation["entropy_adjustment"],
    }

    emit_receipt(
        "adaptation",
        {
            "receipt_type": "adaptation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "manager_id": manager.manager_id,
            "pattern_id": pattern.get("pattern_id"),
            "dest_planet": dest_planet,
            "adaptation_applied": adaptation,
            "payload_hash": dual_hash(json.dumps(adaptation, sort_keys=True)),
        },
    )

    return adapted


def measure_migration_fitness(manager: MigrationManager, pattern_id: str) -> float:
    """Measure fitness at destination vs source.

    Args:
        manager: MigrationManager instance
        pattern_id: Pattern identifier

    Returns:
        Fitness ratio (dest/source)
    """
    # Find transfer for pattern
    for transfer in manager.transfers.values():
        if transfer.pattern_id == pattern_id:
            # Simulate fitness comparison
            source_fitness = transfer.serialized_pattern.get("fitness", 0.8)
            dest_fitness = source_fitness * random.uniform(0.9, 1.1)  # +/- 10%
            return round(dest_fitness / source_fitness, 4) if source_fitness > 0 else 1.0

    return 1.0


def get_migration_status() -> Dict[str, Any]:
    """Get current migration status.

    Returns:
        Migration status dict
    """
    return {
        "module": "autocatalytic.cross_planet_migration",
        "version": "19.0.0",
        "latency_tolerance_ms": MIGRATION_LATENCY_TOLERANCE_MS,
        "fitness_threshold": MIGRATION_FITNESS_THRESHOLD,
        "age_threshold": MIGRATION_AGE_THRESHOLD,
    }
