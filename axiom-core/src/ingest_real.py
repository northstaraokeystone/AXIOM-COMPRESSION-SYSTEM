"""ingest_real.py - Real Data Ingest

Purpose: Ingest REAL bandwidth and delay projections. No assumptions.

Source: Critical Review Dec 16, 2025 - "No validation against real data"
"""

import json
import os
import random
from typing import Dict, List

from .core import emit_receipt

# Get the data directory relative to this file
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_bandwidth_data() -> Dict:
    """Load bandwidth_mars_2025.json.

    Returns:
        Dict with bandwidth data:
        {
            "source": str,
            "values": {
                "minimum_mbps": float,
                "maximum_mbps": float,
                "expected_mbps": float,
                "uncertainty_pct": int
            }
        }
    """
    path = os.path.join(_DATA_DIR, "bandwidth_mars_2025.json")
    with open(path, "r") as f:
        return json.load(f)


def load_delay_data() -> Dict:
    """Load delay_mars_2025.json.

    Returns:
        Dict with delay data:
        {
            "source": str,
            "values": {
                "opposition_min_s": int,
                "conjunction_max_s": int,
                "average_s": int,
                "uncertainty_pct": int
            }
        }
    """
    path = os.path.join(_DATA_DIR, "delay_mars_2025.json")
    with open(path, "r") as f:
        return json.load(f)


def sample_bandwidth(n: int, seed: int = 42) -> List[float]:
    """Sample n bandwidth values from range with uncertainty.

    Args:
        n: Number of samples
        seed: Random seed for reproducibility

    Returns:
        List of bandwidth values in Mbps

    Sampling method:
        Uniform sampling from [minimum_mbps, maximum_mbps]
        with Gaussian noise scaled by uncertainty_pct
    """
    random.seed(seed)
    data = load_bandwidth_data()
    values = data["values"]

    min_bw = values["minimum_mbps"]
    max_bw = values["maximum_mbps"]
    uncertainty = values["uncertainty_pct"] / 100.0

    samples = []
    for i in range(n):
        # Base value: uniform in range
        base = random.uniform(min_bw, max_bw)

        # Add Gaussian noise scaled by uncertainty
        noise = random.gauss(0, base * uncertainty)
        sample = base + noise

        # Clamp to valid range
        sample = max(min_bw, min(max_bw, sample))
        samples.append(sample)

    return samples


def sample_delay(n: int, seed: int = 42) -> List[float]:
    """Sample n delay values from range.

    Args:
        n: Number of samples
        seed: Random seed for reproducibility

    Returns:
        List of delay values in seconds

    Sampling method:
        Uniform sampling from [opposition_min_s, conjunction_max_s]
        Note: uncertainty_pct is 0 for physics-based delays
    """
    random.seed(seed)
    data = load_delay_data()
    values = data["values"]

    min_delay = values["opposition_min_s"]
    max_delay = values["conjunction_max_s"]

    samples = []
    for i in range(n):
        # Uniform sampling across orbital range
        sample = random.uniform(min_delay, max_delay)
        samples.append(sample)

    return samples


def emit_data_ingest_receipt(data_type: str, source: str) -> Dict:
    """Emit CLAUDEME-compliant receipt for data ingest.

    Args:
        data_type: Type of data ("bandwidth" or "delay")
        source: Source attribution string

    Returns:
        Receipt dict
    """
    return emit_receipt("data_ingest", {
        "tenant_id": "axiom-core",
        "data_type": data_type,
        "source": source
    })
