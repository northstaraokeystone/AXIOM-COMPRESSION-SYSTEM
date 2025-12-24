"""Alpha threshold detection for D19.1 law discovery trigger.

When NEURON α crosses threshold (> 1.20), trigger law discovery.
Laws are not discovered—they are enforced by the receipt chain itself.

The insight: Compression = predictability = lawfulness.
When the system becomes more predictable, it IS lawful.
The law exists in the compression.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19.1 ALPHA THRESHOLD CONSTANTS ===

ALPHA_LAW_THRESHOLD = 1.20
"""Trigger law discovery when α crosses this threshold."""

ALPHA_SOURCE = "neuron_ledger"
"""Where to monitor α value."""

LAW_DISCOVERY_COOLDOWN_S = 60
"""Cooldown between law discovery triggers (prevent rapid-fire)."""

COMPRESSION_LAW_TARGET = 0.95
"""Predictability floor for law enforcement."""

LAW_ENFORCEMENT_MODE = "receipt_chain"
"""Laws enforced BY receipt chain, not discovered separately."""


@dataclass
class AlphaThresholdMonitor:
    """Alpha threshold monitoring state."""

    monitor_id: str
    threshold: float = ALPHA_LAW_THRESHOLD
    source: str = ALPHA_SOURCE
    cooldown_s: float = LAW_DISCOVERY_COOLDOWN_S
    last_trigger_ts: Optional[float] = None
    trigger_count: int = 0
    current_alpha: float = 1.0
    alpha_history: List[float] = field(default_factory=list)
    laws_triggered: List[Dict] = field(default_factory=list)
    config: Dict = field(default_factory=dict)


def init_threshold_monitor(config: Dict = None) -> AlphaThresholdMonitor:
    """Initialize alpha threshold monitor.

    Args:
        config: Configuration dict (loads from file if empty)

    Returns:
        AlphaThresholdMonitor instance

    Receipt: alpha_monitor_init_receipt
    """
    config = config or {}

    # Load from spec file if not provided
    if not config:
        spec_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "alpha_threshold_config.json",
        )
        if os.path.exists(spec_path):
            with open(spec_path, "r") as f:
                config = json.load(f)

    monitor_id = str(uuid.uuid4())[:8]
    monitor = AlphaThresholdMonitor(
        monitor_id=monitor_id,
        threshold=config.get("alpha_law_threshold", ALPHA_LAW_THRESHOLD),
        source=config.get("alpha_source", ALPHA_SOURCE),
        cooldown_s=config.get("law_discovery_cooldown_s", LAW_DISCOVERY_COOLDOWN_S),
        config=config,
    )

    emit_receipt(
        "alpha_monitor_init",
        {
            "receipt_type": "alpha_monitor_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "monitor_id": monitor_id,
            "threshold": monitor.threshold,
            "source": monitor.source,
            "cooldown_s": monitor.cooldown_s,
            "enforcement_mode": LAW_ENFORCEMENT_MODE,
            "payload_hash": dual_hash(
                json.dumps(
                    {"monitor_id": monitor_id, "threshold": monitor.threshold},
                    sort_keys=True,
                )
            ),
        },
    )

    return monitor


def update_alpha(monitor: AlphaThresholdMonitor, alpha: float) -> None:
    """Update current alpha value.

    Args:
        monitor: AlphaThresholdMonitor instance
        alpha: New alpha value
    """
    monitor.current_alpha = alpha
    monitor.alpha_history.append(alpha)

    # Keep history bounded
    if len(monitor.alpha_history) > 1000:
        monitor.alpha_history = monitor.alpha_history[-1000:]


def check_threshold(
    monitor: AlphaThresholdMonitor, current_alpha: float = None
) -> bool:
    """Check if α > threshold.

    Args:
        monitor: AlphaThresholdMonitor instance
        current_alpha: Optional alpha value (uses current if None)

    Returns:
        True if α > threshold
    """
    if current_alpha is not None:
        update_alpha(monitor, current_alpha)

    return monitor.current_alpha > monitor.threshold


def is_in_cooldown(monitor: AlphaThresholdMonitor) -> bool:
    """Check if monitor is in cooldown period.

    Args:
        monitor: AlphaThresholdMonitor instance

    Returns:
        True if in cooldown
    """
    if monitor.last_trigger_ts is None:
        return False

    elapsed = time.time() - monitor.last_trigger_ts
    return elapsed < monitor.cooldown_s


def trigger_law_discovery(
    monitor: AlphaThresholdMonitor, receipts: List[Dict] = None
) -> Dict:
    """Trigger law discovery when threshold crossed.

    Args:
        monitor: AlphaThresholdMonitor instance
        receipts: Optional receipts for law extraction

    Returns:
        Law discovery result dict

    Receipt: alpha_threshold_law_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Check cooldown
    if is_in_cooldown(monitor):
        return {
            "triggered": False,
            "reason": "cooldown_active",
            "cooldown_remaining_s": monitor.cooldown_s
            - (time.time() - monitor.last_trigger_ts),
        }

    # Check threshold
    if not check_threshold(monitor):
        return {
            "triggered": False,
            "reason": "threshold_not_crossed",
            "current_alpha": monitor.current_alpha,
            "threshold": monitor.threshold,
        }

    # Trigger law discovery
    monitor.last_trigger_ts = time.time()
    monitor.trigger_count += 1

    law_id = str(uuid.uuid4())[:8]
    law = {
        "law_id": law_id,
        "trigger": "alpha_threshold",
        "alpha_at_trigger": monitor.current_alpha,
        "threshold": monitor.threshold,
        "trigger_count": monitor.trigger_count,
        "discovered_at": now,
        "enforcement_mode": LAW_ENFORCEMENT_MODE,
        "compression_target": COMPRESSION_LAW_TARGET,
        "human_readable": f"Coordination law discovered: α={monitor.current_alpha:.4f} > {monitor.threshold}",
    }

    monitor.laws_triggered.append(law)

    result = {
        "triggered": True,
        "law": law,
        "alpha": monitor.current_alpha,
        "threshold": monitor.threshold,
        "trigger_count": monitor.trigger_count,
    }

    emit_receipt(
        "alpha_threshold_law_receipt",
        {
            "receipt_type": "alpha_threshold_law_receipt",
            "tenant_id": TENANT_ID,
            "ts": now,
            "monitor_id": monitor.monitor_id,
            "law_id": law_id,
            "alpha_at_trigger": monitor.current_alpha,
            "threshold": monitor.threshold,
            "trigger_count": monitor.trigger_count,
            "enforcement_mode": LAW_ENFORCEMENT_MODE,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "law_id": law_id,
                        "alpha": monitor.current_alpha,
                        "trigger_count": monitor.trigger_count,
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def apply_cooldown(monitor: AlphaThresholdMonitor) -> None:
    """Apply cooldown after trigger.

    Args:
        monitor: AlphaThresholdMonitor instance
    """
    monitor.last_trigger_ts = time.time()


def get_threshold_status(monitor: AlphaThresholdMonitor = None) -> Dict[str, Any]:
    """Current threshold status.

    Args:
        monitor: Optional monitor instance

    Returns:
        Threshold status dict
    """
    status = {
        "module": "witness.alpha_threshold",
        "version": "19.1.0",
        "alpha_law_threshold": ALPHA_LAW_THRESHOLD,
        "alpha_source": ALPHA_SOURCE,
        "cooldown_s": LAW_DISCOVERY_COOLDOWN_S,
        "compression_target": COMPRESSION_LAW_TARGET,
        "enforcement_mode": LAW_ENFORCEMENT_MODE,
    }

    if monitor:
        status.update(
            {
                "monitor_id": monitor.monitor_id,
                "current_alpha": monitor.current_alpha,
                "threshold": monitor.threshold,
                "trigger_count": monitor.trigger_count,
                "in_cooldown": is_in_cooldown(monitor),
                "laws_triggered": len(monitor.laws_triggered),
            }
        )

    return status
