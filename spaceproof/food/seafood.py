"""seafood.py - Seafood Species Substitution Detection.

Detects species substitution via tissue entropy analysis.

Genuine tissue: Natural biological variance in structure → Species-specific entropy signature
Substitutes: Altered textural/density entropy (different species = different entropy)

Detection target: ≥99.9% recall
Compliance: FSMA 204 (high-risk food), 21 CFR Part 11

Source: Grok Research - Seafood species fraud detection
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from spaceproof.shared.verification_engine import VerificationEngine
from .entropy import texture_entropy, gradient_entropy


def compute_tissue_entropy(
    tissue_scan: Union[np.ndarray, List[float], Dict],
) -> float:
    """Compute tissue entropy from optical or ultrasound scan.

    3D spatial entropy on tissue structure.
    Analyzes density gradient variance.

    Args:
        tissue_scan: Optical or ultrasound tissue scan data
                    Can be array, list, or dict with 'scan' key

    Returns:
        Tissue entropy value
    """
    if isinstance(tissue_scan, dict):
        scan_data = tissue_scan.get("scan", tissue_scan.get("tissue_scan", []))
        if isinstance(scan_data, dict):
            scan_data = list(scan_data.values())
    elif isinstance(tissue_scan, list):
        scan_data = tissue_scan
    else:
        scan_data = tissue_scan

    arr = np.array(scan_data)

    # Combine texture and gradient entropy for tissue analysis
    text_ent = texture_entropy(arr)
    grad_ent = gradient_entropy(arr)

    # Weight: 70% texture + 30% gradient
    return text_ent * 0.7 + grad_ent * 0.3


def verify_seafood(
    sample_id: str,
    claimed_species: str,
    tissue_scan: Union[np.ndarray, List[float]],
    dna_barcode: Optional[str] = None,
    provenance_chain: Optional[List[str]] = None,
    compliance_standard: str = "FSMA_204",
    config_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Verify seafood species via tissue entropy analysis.

    If DNA barcode available: use as confirmation (100% accurate but expensive)
    If DNA unavailable: rely on tissue entropy alone

    Args:
        sample_id: Unique sample identifier
        claimed_species: "blue_crab" | "wild_salmon" | "cod" | "tuna"
        tissue_scan: Optical or ultrasound tissue scan
        dna_barcode: Optional DNA barcode sequence (if available)
        provenance_chain: Chain of custody
        compliance_standard: "FSMA_204" | "FSMA_204_high_risk" | "CFR_Part_11"
        config_dir: Optional path to config directory

    Returns:
        Tuple of (verdict, receipt)
        verdict: "AUTHENTIC" | "COUNTERFEIT" | "SUSPICIOUS"
        receipt: Complete verification receipt

    Raises:
        ValueError: If claimed_species is invalid
    """
    valid_species = ["blue_crab", "wild_salmon", "cod", "tuna"]
    if claimed_species not in valid_species:
        raise ValueError(f"Invalid claimed_species '{claimed_species}'. Must be one of: {valid_species}")

    # Initialize verification engine
    engine = VerificationEngine(config_dir=config_dir)

    # Prepare sensor data
    if isinstance(tissue_scan, np.ndarray):
        sensor_data = tissue_scan
    else:
        sensor_data = np.array(tissue_scan)

    # DNA validation (if available)
    dna_verified = False
    dna_contradicts_entropy = False

    if dna_barcode:
        # Simple DNA barcode validation (in real system, would match against database)
        # For now, we assume DNA barcode starting with species prefix is valid
        species_prefixes = {
            "blue_crab": ["BC", "CPB"],
            "wild_salmon": ["WS", "SAL"],
            "cod": ["COD", "GAD"],
            "tuna": ["TUN", "THU"],
        }
        valid_prefixes = species_prefixes.get(claimed_species, [])
        dna_verified = any(dna_barcode.upper().startswith(p) for p in valid_prefixes)

    # Verify using engine
    result = engine.verify(
        domain="food",
        item_id=sample_id,
        sensor_data=sensor_data,
        baseline_key=f"seafood.{claimed_species}",
        entropy_calculator=compute_tissue_entropy,
        receipt_type="seafood_verification",
        provenance_chain=provenance_chain or [],
        compliance_standard=compliance_standard,
        additional_fields={
            "claimed_species": claimed_species,
            "product_type": "seafood",
            "dna_verified": dna_verified,
            "dna_barcode_provided": dna_barcode is not None,
        },
    )

    # If DNA contradicts entropy result, flag as CRITICAL
    verdict = result.verdict
    receipt = result.receipt

    if dna_barcode and not dna_verified and verdict == "AUTHENTIC":
        # DNA says wrong species but entropy says authentic - flag contradiction
        receipt["dna_contradiction"] = True
        receipt["flags"] = receipt.get("flags", []) + ["dna_species_mismatch"]
        verdict = "SUSPICIOUS"
        receipt["verdict"] = verdict

    return verdict, receipt


def verify_seafood_batch(
    samples: List[Dict],
    config_dir: Optional[Path] = None,
) -> List[Tuple[str, Dict]]:
    """Verify multiple seafood samples.

    Args:
        samples: List of dicts with keys:
            - sample_id: str
            - claimed_species: str
            - tissue_scan: array-like
            - dna_barcode: str (optional)
            - provenance_chain: List[str] (optional)
            - compliance_standard: str (optional)
        config_dir: Optional path to config directory

    Returns:
        List of (verdict, receipt) tuples
    """
    results = []

    for sample in samples:
        verdict, receipt = verify_seafood(
            sample_id=sample["sample_id"],
            claimed_species=sample["claimed_species"],
            tissue_scan=sample["tissue_scan"],
            dna_barcode=sample.get("dna_barcode"),
            provenance_chain=sample.get("provenance_chain", []),
            compliance_standard=sample.get("compliance_standard", "FSMA_204"),
            config_dir=config_dir,
        )
        results.append((verdict, receipt))

    return results
