"""cancer_drugs.py - Cancer Drug Counterfeit Detection.

Detects counterfeit cancer drugs via API distribution entropy.

Genuine tablets: API distribution randomness from validated processes → Entropy (4.1-4.9)
Fakes: No API (zero entropy) OR wrong distribution (entropy outside range)

Detection target: ≥99.9% recall (treatment failure = death)
Compliance: 21 CFR Part 820 (QSR), ISO 13485

Source: Grok Research - 1/3 of some cancer drugs fake in certain markets
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from spaceproof.shared.verification_engine import VerificationEngine
from .entropy import api_distribution_entropy


# No-API detection threshold
NO_API_ENTROPY_THRESHOLD = 0.5


def compute_api_distribution_entropy(
    raman_map: Union[np.ndarray, List[float], Dict],
) -> float:
    """Compute API distribution entropy from Raman spectroscopy mapping.

    3D spatial entropy of active pharmaceutical ingredient.
    Analyzes uniformity vs expected variance.

    Args:
        raman_map: 3D Raman spectroscopy spatial mapping data

    Returns:
        API distribution entropy value
    """
    if isinstance(raman_map, dict):
        map_data = raman_map.get("map", raman_map.get("raman_map", []))
        if isinstance(map_data, dict):
            map_data = list(map_data.values())
    elif isinstance(raman_map, list):
        map_data = raman_map
    else:
        map_data = raman_map

    return api_distribution_entropy(np.array(map_data))


def detect_no_api(
    raman_map: Union[np.ndarray, List[float]],
    drug_name: str,
) -> bool:
    """Check if API spectral signature is present.

    If absent: immediate COUNTERFEIT verdict.

    Args:
        raman_map: Raman spectroscopy data
        drug_name: Expected drug name

    Returns:
        True if NO API detected (counterfeit), False if API present
    """
    entropy = compute_api_distribution_entropy(raman_map)

    # Very low entropy suggests no active ingredient
    return entropy < NO_API_ENTROPY_THRESHOLD


def verify_cancer_drug(
    drug_id: str,
    drug_name: str,
    raman_map: Union[np.ndarray, List[float]],
    tablet_structure: Optional[Dict] = None,
    provenance_chain: Optional[List[str]] = None,
    compliance_standard: str = "21_CFR_Part_820_QSR",
    config_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Verify cancer drug authenticity via API distribution entropy.

    First check: API present? (detect_no_api)
    If no API: COUNTERFEIT
    Else: compute API distribution entropy, verify against baseline

    Args:
        drug_id: Drug identifier/lot number
        drug_name: "imfinzi_120mg" | "keytruda_100mg" | "opdivo_240mg"
        raman_map: 3D Raman spectroscopy spatial mapping data
        tablet_structure: Optional dict with micro-CT or cross-section analysis
        provenance_chain: Chain of custody
        compliance_standard: "21_CFR_Part_820_QSR" | "ISO_13485"
        config_dir: Optional path to config directory

    Returns:
        Tuple of (verdict, receipt)
        verdict: "AUTHENTIC" | "COUNTERFEIT" | "SUSPICIOUS"
        receipt: Complete verification receipt

    Notes:
        - All cancer drugs are risk_level=CRITICAL
        - treatment_impact field included in receipt
    """
    valid_drugs = ["imfinzi_120mg", "keytruda_100mg", "opdivo_240mg"]
    if drug_name not in valid_drugs:
        raise ValueError(f"Invalid drug_name '{drug_name}'. Must be one of: {valid_drugs}")

    # Convert raman_map to array
    if isinstance(raman_map, dict):
        map_data = raman_map.get("map", raman_map.get("raman_map", []))
    elif isinstance(raman_map, np.ndarray):
        map_data = raman_map
    else:
        map_data = np.array(raman_map)

    if isinstance(map_data, list):
        map_data = np.array(map_data)

    # First check: is API present?
    no_api_detected = detect_no_api(map_data, drug_name)

    if no_api_detected:
        # Immediate COUNTERFEIT - no active ingredient
        from spaceproof.core import emit_receipt

        receipt = emit_receipt(
            "cancer_drug_verification",
            {
                "tenant_id": "spaceproof-verification",
                "domain": "medical",
                "item_id": drug_id,
                "drug_name": drug_name,
                "api_present": False,
                "verdict": "COUNTERFEIT",
                "confidence": 0.99,
                "risk_level": "CRITICAL",
                "treatment_impact": "Cancer treatment failure = death",
                "flags": ["no_api_detected", "immediate_rejection"],
                "provenance_chain": provenance_chain or [],
                "compliance_standard": compliance_standard,
            },
        )
        return "COUNTERFEIT", receipt

    # Initialize verification engine
    engine = VerificationEngine(config_dir=config_dir)

    # Get manufacturer from drug name
    manufacturers = {
        "imfinzi_120mg": "AstraZeneca",
        "keytruda_100mg": "Merck",
        "opdivo_240mg": "Bristol-Myers Squibb",
    }

    # Verify using engine
    result = engine.verify(
        domain="medical",
        item_id=drug_id,
        sensor_data=map_data,
        baseline_key=f"cancer_drug.{drug_name}",
        entropy_calculator=compute_api_distribution_entropy,
        receipt_type="cancer_drug_verification",
        provenance_chain=provenance_chain or [],
        compliance_standard=compliance_standard,
        risk_level="CRITICAL",
        additional_fields={
            "drug_name": drug_name,
            "product_type": "cancer_drug",
            "api_present": True,
            "manufacturer": manufacturers.get(drug_name, "Unknown"),
            "treatment_impact": "Cancer treatment failure = death",
        },
    )

    return result.verdict, result.receipt


def verify_cancer_drug_batch(
    drugs: List[Dict],
    config_dir: Optional[Path] = None,
) -> List[Tuple[str, Dict]]:
    """Verify multiple cancer drug samples.

    Args:
        drugs: List of dicts with keys:
            - drug_id: str
            - drug_name: str
            - raman_map: array-like
            - tablet_structure: Dict (optional)
            - provenance_chain: List[str] (optional)
            - compliance_standard: str (optional)
        config_dir: Optional path to config directory

    Returns:
        List of (verdict, receipt) tuples
    """
    results = []

    for drug in drugs:
        verdict, receipt = verify_cancer_drug(
            drug_id=drug["drug_id"],
            drug_name=drug["drug_name"],
            raman_map=drug["raman_map"],
            tablet_structure=drug.get("tablet_structure"),
            provenance_chain=drug.get("provenance_chain", []),
            compliance_standard=drug.get("compliance_standard", "21_CFR_Part_820_QSR"),
            config_dir=config_dir,
        )
        results.append((verdict, receipt))

    return results
