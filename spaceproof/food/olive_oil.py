"""olive_oil.py - Olive Oil Adulteration Detection.

Detects adulteration via spectral entropy analysis.

Genuine oils: Natural random gradients in polyphenols/fatty acids → HIGH spectral entropy (3.8-4.6)
Adulterants: Homogenized vegetable oil mixtures → LOW spectral entropy (<3.2)

Detection target: ≥99.9% recall on validated adulterations
False positive: ≤1% on genuine samples

Compliance: FSMA 204 traceability, 21 CFR Part 11 electronic records

Source: Grok Research - $40B annual olive oil fraud
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from spaceproof.shared.verification_engine import VerificationEngine
from .entropy import spectral_entropy


def compute_spectral_entropy(
    spectral_scan: Union[np.ndarray, List[float], Dict],
) -> float:
    """Compute spectral entropy from NIR/hyperspectral scan.

    Uses Shannon entropy: H = -Σ(p_i * log2(p_i))
    Bins spectrum into 50 bins, normalizes, computes distribution entropy.

    Args:
        spectral_scan: NIR or hyperspectral wavelength readings
                      Can be array, list, or dict with 'readings' key

    Returns:
        Spectral entropy value
    """
    if isinstance(spectral_scan, dict):
        scan_data = spectral_scan.get("readings", spectral_scan.get("spectral_scan", []))
        if isinstance(scan_data, dict):
            scan_data = list(scan_data.values())
    elif isinstance(spectral_scan, list):
        scan_data = spectral_scan
    else:
        scan_data = spectral_scan

    return spectral_entropy(np.array(scan_data), bins=50)


def verify_olive_oil(
    batch_id: str,
    product_grade: str,
    spectral_scan: Union[np.ndarray, List[float]],
    provenance_chain: List[str],
    compliance_standard: str = "FSMA_204",
    config_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Verify olive oil authenticity via spectral entropy.

    Args:
        batch_id: Unique batch identifier
        product_grade: "extra_virgin" | "virgin" | "pure"
        spectral_scan: NIR or hyperspectral wavelength readings
        provenance_chain: Chain of custody (farm → processor → bottler → retailer)
        compliance_standard: "FSMA_204" | "CFR_Part_11"
        config_dir: Optional path to config directory

    Returns:
        Tuple of (verdict, receipt)
        verdict: "AUTHENTIC" | "COUNTERFEIT" | "SUSPICIOUS"
        receipt: Complete verification receipt

    Raises:
        ValueError: If product_grade is invalid
    """
    valid_grades = ["extra_virgin", "virgin", "pure"]
    if product_grade not in valid_grades:
        raise ValueError(f"Invalid product_grade '{product_grade}'. Must be one of: {valid_grades}")

    # Initialize verification engine
    engine = VerificationEngine(config_dir=config_dir)

    # Prepare sensor data
    if isinstance(spectral_scan, np.ndarray):
        sensor_data = spectral_scan
    else:
        sensor_data = np.array(spectral_scan)

    # Verify using engine
    result = engine.verify(
        domain="food",
        item_id=batch_id,
        sensor_data=sensor_data,
        baseline_key=f"olive_oil.{product_grade}",
        entropy_calculator=compute_spectral_entropy,
        receipt_type="olive_oil_verification",
        provenance_chain=provenance_chain,
        compliance_standard=compliance_standard,
        additional_fields={
            "product_grade": product_grade,
            "product_type": "olive_oil",
            "spectral_bins": 50,
        },
    )

    return result.verdict, result.receipt


def verify_olive_oil_batch(
    batches: List[Dict],
    config_dir: Optional[Path] = None,
) -> List[Tuple[str, Dict]]:
    """Verify multiple olive oil batches.

    Args:
        batches: List of dicts with keys:
            - batch_id: str
            - product_grade: str
            - spectral_scan: array-like
            - provenance_chain: List[str]
            - compliance_standard: str (optional)
        config_dir: Optional path to config directory

    Returns:
        List of (verdict, receipt) tuples
    """
    results = []

    for batch in batches:
        verdict, receipt = verify_olive_oil(
            batch_id=batch["batch_id"],
            product_grade=batch["product_grade"],
            spectral_scan=batch["spectral_scan"],
            provenance_chain=batch.get("provenance_chain", []),
            compliance_standard=batch.get("compliance_standard", "FSMA_204"),
            config_dir=config_dir,
        )
        results.append((verdict, receipt))

    return results
