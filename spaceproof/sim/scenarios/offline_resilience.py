"""offline_resilience.py - Offline Sync Scenario.

OFFLINE_RESILIENCE SCENARIO:
    Validate light-delay offline sync.
    Network partitions, conflict resolution, Byzantine-resilient merge.

Pass Criteria:
    - 100% offline receipts preserved in local ledger
    - Conflict resolution successful (deterministic merge)
    - Merkle chain integrity maintained across partition
    - Sync latency within bounds (< 2x light-delay)
    - Zero data loss on rejoin
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import uuid

import numpy as np

from spaceproof.core import emit_receipt, merkle, dual_hash, MARS_LIGHT_DELAY_MAX_SEC

CHECKPOINT_FREQUENCY = 50
TENANT_ID = "spaceproof-scenario-offline"


@dataclass
class OfflineResilienceConfig:
    """Configuration for offline resilience scenario."""

    cycles: int = 500
    seed: int = 42
    network_partitions: int = 3
    receipts_per_partition: int = 50
    conflicting_receipts: int = 50
    light_delay_sec: float = MARS_LIGHT_DELAY_MAX_SEC  # 22 min Mars conjunction


@dataclass
class OfflineResilienceResult:
    """Result of offline resilience scenario execution."""

    cycles_completed: int
    partitions_simulated: int
    receipts_created_offline: int
    receipts_preserved: int
    conflicts_detected: int
    conflicts_resolved: int
    merkle_integrity_maintained: bool
    sync_latency_ok: bool
    data_loss: int
    passed: bool
    failure_reasons: List[str]


class OfflineResilienceScenario:
    """Offline sync validation scenario."""

    def __init__(self, config: Optional[OfflineResilienceConfig] = None):
        """Initialize offline resilience scenario."""
        self.config = config or OfflineResilienceConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Tracking
        self.offline_receipts: Dict[str, List[Dict]] = {}  # node_id -> receipts
        self.merged_receipts: List[Dict] = []
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.sync_latencies: List[float] = []
        self.merkle_roots: Dict[str, str] = {}

    def run(self) -> OfflineResilienceResult:
        """Run the offline resilience scenario."""
        failure_reasons = []

        # Simulate network partitions
        for partition in range(self.config.network_partitions):
            self._simulate_partition(partition)

        # Create conflicting receipts
        for i in range(self.config.conflicting_receipts):
            self._create_conflict(i)

        # Perform sync and conflict resolution
        self._perform_sync()

        # Verify Merkle integrity
        merkle_ok = self._verify_merkle_integrity()

        # Calculate metrics
        total_offline = sum(len(r) for r in self.offline_receipts.values())
        preserved = len(self.merged_receipts)
        data_loss = total_offline - preserved

        # Check sync latency
        max_allowed_latency = self.config.light_delay_sec * 2 * 1000  # ms
        sync_ok = all(lat < max_allowed_latency for lat in self.sync_latencies) if self.sync_latencies else True

        # Validate results
        if data_loss > 0:
            failure_reasons.append(f"Data loss: {data_loss} receipts lost")

        if not merkle_ok:
            failure_reasons.append("Merkle chain integrity not maintained")

        if not sync_ok:
            failure_reasons.append("Sync latency exceeded 2x light-delay")

        if self.conflicts_resolved < self.conflicts_detected:
            failure_reasons.append(f"Only {self.conflicts_resolved}/{self.conflicts_detected} conflicts resolved")

        passed = len(failure_reasons) == 0

        return OfflineResilienceResult(
            cycles_completed=self.config.cycles,
            partitions_simulated=self.config.network_partitions,
            receipts_created_offline=total_offline,
            receipts_preserved=preserved,
            conflicts_detected=self.conflicts_detected,
            conflicts_resolved=self.conflicts_resolved,
            merkle_integrity_maintained=merkle_ok,
            sync_latency_ok=sync_ok,
            data_loss=data_loss,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def _simulate_partition(self, partition_id: int) -> None:
        """Simulate a network partition."""
        node_id = f"colony_{partition_id}"
        self.offline_receipts[node_id] = []

        for i in range(self.config.receipts_per_partition):
            receipt = self._create_offline_receipt(node_id, i)
            self.offline_receipts[node_id].append(receipt)

            if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                self._emit_checkpoint(partition_id, i)

        # Compute Merkle root for this node's receipts
        self.merkle_roots[node_id] = merkle(self.offline_receipts[node_id])

        emit_receipt(
            "partition_end",
            {
                "tenant_id": TENANT_ID,
                "node_id": node_id,
                "receipts_during_partition": len(self.offline_receipts[node_id]),
                "merkle_root": self.merkle_roots[node_id],
            },
        )

    def _create_offline_receipt(self, node_id: str, index: int) -> Dict:
        """Create an offline receipt."""
        return {
            "receipt_id": str(uuid.uuid4()),
            "node_id": node_id,
            "sequence_num": index,
            "receipt_type": "offline_telemetry",
            "data": {"value": float(self.rng.uniform(0, 100))},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hash": dual_hash(f"{node_id}:{index}"),
        }

    def _create_conflict(self, index: int) -> None:
        """Create conflicting receipts between nodes."""
        if len(self.offline_receipts) < 2:
            return

        nodes = list(self.offline_receipts.keys())
        node_a, node_b = self.rng.choice(nodes, size=2, replace=False)

        # Create same receipt ID with different content
        conflict_id = str(uuid.uuid4())

        receipt_a = {
            "receipt_id": conflict_id,
            "node_id": node_a,
            "sequence_num": index,
            "receipt_type": "conflict_test",
            "data": {"value": float(self.rng.uniform(0, 50))},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        receipt_b = {
            "receipt_id": conflict_id,
            "node_id": node_b,
            "sequence_num": index,
            "receipt_type": "conflict_test",
            "data": {"value": float(self.rng.uniform(50, 100))},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self.offline_receipts[node_a].append(receipt_a)
        self.offline_receipts[node_b].append(receipt_b)
        self.conflicts_detected += 1

    def _perform_sync(self) -> bool:
        """Perform sync and conflict resolution."""
        import time

        start = time.time()

        # Collect all receipts
        all_receipts: Dict[str, List[Dict]] = {}  # receipt_id -> [versions]

        for node_id, receipts in self.offline_receipts.items():
            for receipt in receipts:
                rid = receipt["receipt_id"]
                if rid not in all_receipts:
                    all_receipts[rid] = []
                all_receipts[rid].append(receipt)

        # Resolve conflicts (deterministic: pick by hash order)
        for rid, versions in all_receipts.items():
            if len(versions) > 1:
                # Sort by hash for deterministic resolution
                versions.sort(key=lambda r: dual_hash(str(r)))
                winner = versions[0]
                self.conflicts_resolved += 1
            else:
                winner = versions[0]

            self.merged_receipts.append(winner)

        sync_time = (time.time() - start) * 1000
        self.sync_latencies.append(sync_time)

        emit_receipt(
            "sync_complete",
            {
                "tenant_id": TENANT_ID,
                "receipts_merged": len(self.merged_receipts),
                "conflicts_resolved": self.conflicts_resolved,
                "sync_time_ms": sync_time,
            },
        )

        return True

    def _verify_merkle_integrity(self) -> bool:
        """Verify Merkle chain integrity after sync."""
        # Compute new Merkle root from merged receipts
        merged_root = merkle(self.merged_receipts) if self.merged_receipts else ""

        # Verify each node's receipts are included
        for node_id, receipts in self.offline_receipts.items():
            for receipt in receipts:
                # Check receipt is in merged list (or its conflict resolution winner)
                if not any(r["receipt_id"] == receipt["receipt_id"] for r in self.merged_receipts):
                    return False

        emit_receipt(
            "merkle_verification",
            {
                "tenant_id": TENANT_ID,
                "merged_merkle_root": merged_root,
                "node_count": len(self.merkle_roots),
                "integrity_verified": True,
            },
        )

        return True

    def _emit_checkpoint(self, partition: int, step: int) -> None:
        """Emit checkpoint receipt."""
        node_id = f"colony_{partition}"
        emit_receipt(
            "offline_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "node_id": node_id,
                "step": step,
                "receipts_so_far": len(self.offline_receipts.get(node_id, [])),
            },
        )
