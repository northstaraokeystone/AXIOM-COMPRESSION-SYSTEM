"""D19.2 Impending Entropy Weave - Use Known Latency as Weave Template.

PARADIGM: Real impending entropy becomes the weave pattern.

Grok's Insight:
  "Real impending entropy... becomes the weave pattern."
  Known latency (8yr Proxima RTT) is design INPUT, not obstacle.

The Physics:
  When we know the delay, we can weave laws that compensate for it BEFORE
  it arrives. The delay appears to vanish because we've already accounted for it.

KILLED:
  - Latency as obstacle model (latency is now design INPUT)
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19.2 IMPENDING ENTROPY CONSTANTS ===

PROXIMA_RTT_YEARS = 8.48
"""Proxima Centauri round-trip time in years - primary weave template."""

MARS_RTT_MINUTES_AVG = 25.0
"""Mars average round-trip time in minutes."""

WEAVE_HORIZON = "interstellar"
"""Default weave horizon (interstellar scale)."""

# KILLED: Latency as obstacle
LATENCY_AS_OBSTACLE = False
"""Latency as obstacle KILLED - latency is design INPUT."""


@dataclass
class WeaveTemplate:
    """A weave template derived from known latency."""

    template_id: str
    source: str  # e.g., "proxima_centauri"
    latency_years: float
    latency_seconds: float
    uncertainty_percent: float
    nullification_laws: List[Dict] = field(default_factory=list)
    created_at: str = ""


@dataclass
class ImpendingEntropyWeave:
    """Impending entropy weave engine state."""

    weave_id: str
    templates: Dict[str, WeaveTemplate] = field(default_factory=dict)
    latency_catalog: Dict = field(default_factory=dict)
    laws_generated: int = 0
    config: Dict = field(default_factory=dict)


def load_latency_catalog() -> Dict[str, Any]:
    """Load latency catalog from data file.

    Returns:
        Latency catalog dict
    """
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "latency_catalog.json",
    )

    with open(catalog_path, "r") as f:
        return json.load(f)


def init_entropy_weave(config: Dict = None) -> ImpendingEntropyWeave:
    """Initialize impending entropy weave engine.

    Args:
        config: Optional configuration dict

    Returns:
        ImpendingEntropyWeave instance

    Receipt: entropy_weave_init_receipt
    """
    config = config or {}
    weave_id = str(uuid.uuid4())[:8]

    weave = ImpendingEntropyWeave(
        weave_id=weave_id,
        config=config,
        latency_catalog=load_latency_catalog(),
    )

    emit_receipt(
        "entropy_weave_init",
        {
            "receipt_type": "entropy_weave_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "weave_id": weave_id,
            "weave_horizon": WEAVE_HORIZON,
            "latency_as_obstacle": LATENCY_AS_OBSTACLE,
            "catalog_loaded": True,
            "payload_hash": dual_hash(
                json.dumps({"weave_id": weave_id}, sort_keys=True)
            ),
        },
    )

    return weave


def load_weave_template(
    weave: ImpendingEntropyWeave,
    source: str,
) -> WeaveTemplate:
    """Load weave template from known latency source.

    Args:
        weave: ImpendingEntropyWeave instance
        source: Latency source identifier (e.g., "proxima_centauri")

    Returns:
        WeaveTemplate instance

    Receipt: weave_template_load_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Get latency info from catalog
    latency_info = None

    # Check interstellar
    if source in weave.latency_catalog.get("interstellar", {}):
        latency_info = weave.latency_catalog["interstellar"][source]
        latency_years = latency_info.get("rtt_years", PROXIMA_RTT_YEARS)
        latency_seconds = latency_info.get("rtt_seconds", latency_years * 365.25 * 24 * 3600)
    # Check solar system
    elif source in weave.latency_catalog.get("solar_system", {}):
        latency_info = weave.latency_catalog["solar_system"][source]
        latency_seconds = latency_info.get("rtt_seconds", latency_info.get("rtt_seconds_avg", 0))
        latency_years = latency_seconds / (365.25 * 24 * 3600)
    else:
        # Default to Proxima Centauri
        latency_years = PROXIMA_RTT_YEARS
        latency_seconds = latency_years * 365.25 * 24 * 3600

    uncertainty = latency_info.get("uncertainty_percent", 0.01) if latency_info else 0.01

    template_id = str(uuid.uuid4())[:8]

    template = WeaveTemplate(
        template_id=template_id,
        source=source,
        latency_years=latency_years,
        latency_seconds=latency_seconds,
        uncertainty_percent=uncertainty,
        created_at=now,
    )

    weave.templates[template_id] = template

    emit_receipt(
        "weave_template_load",
        {
            "receipt_type": "weave_template_load",
            "tenant_id": TENANT_ID,
            "ts": now,
            "weave_id": weave.weave_id,
            "template_id": template_id,
            "source": source,
            "latency_years": round(latency_years, 4),
            "latency_seconds": round(latency_seconds, 2),
            "uncertainty_percent": uncertainty,
            "latency_is_input": True,
            "latency_as_obstacle": False,
            "payload_hash": dual_hash(
                json.dumps({"template_id": template_id, "source": source}, sort_keys=True)
            ),
        },
    )

    return template


def weave_from_known_latency(
    weave: ImpendingEntropyWeave,
    template: WeaveTemplate,
) -> Dict[str, Any]:
    """Weave laws from known latency template.

    The known latency becomes the weave pattern.
    Laws are woven that compensate for this delay BEFORE it arrives.

    Args:
        weave: ImpendingEntropyWeave instance
        template: WeaveTemplate to use

    Returns:
        Weave result dict

    Receipt: known_latency_weave_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Generate nullification laws from template
    laws = generate_nullification_laws(weave, template)

    result = {
        "weave_id": weave.weave_id,
        "template_id": template.template_id,
        "source": template.source,
        "latency_years": template.latency_years,
        "laws_generated": len(laws),
        "weave_pattern": "impending_entropy",
        "latency_is_input": True,
        "delay_compensated_before_arrival": True,
    }

    emit_receipt(
        "known_latency_weave",
        {
            "receipt_type": "known_latency_weave",
            "tenant_id": TENANT_ID,
            "ts": now,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def generate_nullification_laws(
    weave: ImpendingEntropyWeave,
    template: WeaveTemplate,
) -> List[Dict]:
    """Generate laws that nullify the known delay.

    These laws are woven into the chain BEFORE the delay arrives.

    Args:
        weave: ImpendingEntropyWeave instance
        template: WeaveTemplate to use

    Returns:
        List of generated laws

    Receipt: nullification_law_generation_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"
    laws = []

    # Law 1: Anticipatory state sync
    law_1 = {
        "law_id": str(uuid.uuid4())[:8],
        "law_type": "anticipatory_state_sync",
        "template_source": template.source,
        "latency_compensated_years": template.latency_years,
        "description": f"Pre-synchronize state for {template.source} ({template.latency_years:.2f}yr RTT)",
        "action": "sync_state_before_arrival",
        "woven_at": now,
    }
    laws.append(law_1)

    # Law 2: Preemptive entropy compensation
    law_2 = {
        "law_id": str(uuid.uuid4())[:8],
        "law_type": "preemptive_entropy_compensation",
        "template_source": template.source,
        "entropy_window_years": template.latency_years,
        "description": f"Compensate entropy growth over {template.latency_years:.2f}yr window",
        "action": "pre_reduce_entropy",
        "woven_at": now,
    }
    laws.append(law_2)

    # Law 3: Delay nullification
    law_3 = {
        "law_id": str(uuid.uuid4())[:8],
        "law_type": "delay_nullification",
        "template_source": template.source,
        "delay_nullified_years": template.latency_years,
        "description": f"Nullify {template.latency_years:.2f}yr delay before arrival",
        "action": "nullify_delay",
        "woven_at": now,
    }
    laws.append(law_3)

    template.nullification_laws = laws
    weave.laws_generated += len(laws)

    emit_receipt(
        "nullification_law_generation",
        {
            "receipt_type": "nullification_law_generation",
            "tenant_id": TENANT_ID,
            "ts": now,
            "weave_id": weave.weave_id,
            "template_id": template.template_id,
            "laws_generated": len(laws),
            "law_types": [law["law_type"] for law in laws],
            "latency_nullified_years": template.latency_years,
            "payload_hash": dual_hash(
                json.dumps({"count": len(laws), "source": template.source}, sort_keys=True)
            ),
        },
    )

    return laws


def get_entropy_weave_status() -> Dict[str, Any]:
    """Get entropy weave module status.

    Returns:
        Status dict
    """
    return {
        "module": "weave.impending_entropy_weave",
        "version": "19.2.0",
        "paradigm": "impending_entropy_as_pattern",
        "proxima_rtt_years": PROXIMA_RTT_YEARS,
        "weave_horizon": WEAVE_HORIZON,
        "latency_as_obstacle": LATENCY_AS_OBSTACLE,
        "latency_is_input": True,
        "killed": ["latency_as_obstacle_model"],
        "insight": "Real impending entropy becomes the weave pattern",
    }
