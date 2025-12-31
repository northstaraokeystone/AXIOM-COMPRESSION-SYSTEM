"""honey.py - Honey Syrup Adulteration Detection.

Detects syrup adulteration via microstructural texture entropy + pollen analysis.

Genuine honey: Inherent randomness in pollen distribution + crystallization → HIGH texture entropy (4.2-5.1)
Syrup-adulterated: Lacks microstructural entropy → LOW texture entropy (<3.5)

Detection target: ≥99.9% recall
Compliance: FSMA 204, 21 CFR Part 11

Source: Grok Research - $10-40B annual honey fraud
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from spaceproof.shared.verification_engine import VerificationEngine
from .entropy import texture_entropy, pollen_diversity_entropy, combined_food_entropy


def compute_texture_entropy(
    texture_scan: Union[np.ndarray, List[float], Dict],
) -> float:
    """Compute texture entropy from optical microscopy or ultrasound imaging.

    Uses 2D spatial entropy on grayscale image.
    Analyzes local variance in pixel neighborhoods.

    Args:
        texture_scan: Optical microscopy or ultrasound image data
                     Can be array, list, or dict with 'image' key

    Returns:
        Texture entropy value
    """
    if isinstance(texture_scan, dict):
        scan_data = texture_scan.get("image", texture_scan.get("texture_scan", []))
        if isinstance(scan_data, dict):
            scan_data = list(scan_data.values())
    elif isinstance(texture_scan, list):
        scan_data = texture_scan
    else:
        scan_data = texture_scan

    return texture_entropy(np.array(scan_data), window_size=8)


def compute_pollen_entropy(
    pollen_data: Dict,
) -> float:
    """Compute pollen diversity entropy from species counts.

    Uses Shannon entropy of species distribution.

    Args:
        pollen_data: Dict with 'species' key mapping species names to counts,
                    or direct dict of species -> counts

    Returns:
        Pollen diversity entropy value
    """
    if "species" in pollen_data:
        species_counts = pollen_data["species"]
    else:
        # Assume direct species -> count mapping
        species_counts = {k: v for k, v in pollen_data.items()
                         if isinstance(v, (int, float)) and k != "total"}

    return pollen_diversity_entropy(species_counts)


def compute_combined_honey_entropy(
    texture_scan: Union[np.ndarray, List[float], Dict],
    pollen_data: Optional[Dict] = None,
) -> float:
    """Compute combined entropy for honey verification.

    Weighted: 60% texture + 40% pollen

    Args:
        texture_scan: Texture scan data
        pollen_data: Optional pollen analysis data

    Returns:
        Combined entropy value
    """
    texture_ent = compute_texture_entropy(texture_scan)

    if pollen_data is None:
        return texture_ent

    pollen_ent = compute_pollen_entropy(pollen_data)

    return combined_food_entropy(texture_ent, pollen_ent, primary_weight=0.6)


def verify_honey(
    batch_id: str,
    honey_type: str,
    texture_scan: Union[np.ndarray, List[float]],
    pollen_analysis: Optional[Dict] = None,
    provenance_chain: Optional[List[str]] = None,
    compliance_standard: str = "FSMA_204",
    config_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Verify honey authenticity via texture entropy + pollen analysis.

    Args:
        batch_id: Unique batch identifier
        honey_type: "manuka" | "wildflower" | "clover"
        texture_scan: Optical microscopy or ultrasound imaging data
        pollen_analysis: Optional dict with pollen count and species distribution
            - pollen_count: int (total pollen grains)
            - species: Dict[str, int] (species name -> count)
        provenance_chain: Chain of custody
        compliance_standard: "FSMA_204" | "CFR_Part_11"
        config_dir: Optional path to config directory

    Returns:
        Tuple of (verdict, receipt)
        verdict: "AUTHENTIC" | "COUNTERFEIT" | "SUSPICIOUS"
        receipt: Complete verification receipt

    Raises:
        ValueError: If honey_type is invalid
    """
    valid_types = ["manuka", "wildflower", "clover"]
    if honey_type not in valid_types:
        raise ValueError(f"Invalid honey_type '{honey_type}'. Must be one of: {valid_types}")

    # Initialize verification engine
    engine = VerificationEngine(config_dir=config_dir)

    # Prepare sensor data
    if isinstance(texture_scan, np.ndarray):
        sensor_data = texture_scan
    else:
        sensor_data = np.array(texture_scan)

    # Create combined entropy calculator with pollen data
    def combined_calculator(data):
        return compute_combined_honey_entropy(data, pollen_analysis)

    # Extract pollen metrics for receipt
    pollen_count = 0
    pollen_entropy = 0.0
    if pollen_analysis:
        pollen_count = pollen_analysis.get("pollen_count", 0)
        if "species" in pollen_analysis:
            pollen_count = sum(pollen_analysis["species"].values())
        pollen_entropy = compute_pollen_entropy(pollen_analysis)

    # Verify using engine
    result = engine.verify(
        domain="food",
        item_id=batch_id,
        sensor_data=sensor_data,
        baseline_key=f"honey.{honey_type}",
        entropy_calculator=combined_calculator,
        receipt_type="honey_verification",
        provenance_chain=provenance_chain or [],
        compliance_standard=compliance_standard,
        additional_fields={
            "honey_type": honey_type,
            "product_type": "honey",
            "texture_entropy": compute_texture_entropy(sensor_data),
            "pollen_entropy": pollen_entropy,
            "pollen_count": pollen_count,
        },
    )

    return result.verdict, result.receipt


def verify_honey_batch(
    batches: List[Dict],
    config_dir: Optional[Path] = None,
) -> List[Tuple[str, Dict]]:
    """Verify multiple honey batches.

    Args:
        batches: List of dicts with keys:
            - batch_id: str
            - honey_type: str
            - texture_scan: array-like
            - pollen_analysis: Dict (optional)
            - provenance_chain: List[str] (optional)
            - compliance_standard: str (optional)
        config_dir: Optional path to config directory

    Returns:
        List of (verdict, receipt) tuples
    """
    results = []

    for batch in batches:
        verdict, receipt = verify_honey(
            batch_id=batch["batch_id"],
            honey_type=batch["honey_type"],
            texture_scan=batch["texture_scan"],
            pollen_analysis=batch.get("pollen_analysis"),
            provenance_chain=batch.get("provenance_chain", []),
            compliance_standard=batch.get("compliance_standard", "FSMA_204"),
            config_dir=config_dir,
        )
        results.append((verdict, receipt))

    return results
