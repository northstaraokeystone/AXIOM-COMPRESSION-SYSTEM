"""glp1.py - GLP-1 Pen Counterfeit Detection (Ozempic, Wegovy).

Detects counterfeit GLP-1 pens via fill variance entropy.

Genuine pens: Manufacturing variance in fill/compression → Controlled entropy (2.8-3.4)
Fakes: Abnormal uniformity (too perfect, <2.5) OR material deviations (>3.8)

Detection target: ≥99.9% recall (life-threatening if missed)
False positive: Minimize (genuine patients need medication)

Compliance: 21 CFR Part 820 (QSR), ISO 13485

Source: Grok Research - 2023-2025 counterfeit GLP-1 surge
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from spaceproof.shared.verification_engine import VerificationEngine
from .entropy import fill_variance_entropy


# Lot number patterns by manufacturer
LOT_PATTERNS = {
    "ozempic": r"^OZP-\d{4}-\d{5}$",
    "wegovy": r"^WGY-\d{4}-\d{5}$",
    "novo_nordisk": r"^(OZP|WGY|NVL)-\d{4}-\d{5}$",
}


def compute_fill_entropy(
    fill_imaging: Union[Dict, np.ndarray, List[float]],
) -> float:
    """Compute fill entropy from optical + X-ray measurements.

    Analyzes fill level variance across multiple measurements.
    Variance in compression uniformity.

    Args:
        fill_imaging: Dict with fill_level, compression, etc.
                     Or array of measurements

    Returns:
        Fill entropy value
    """
    return fill_variance_entropy(fill_imaging)


def validate_lot_format(
    lot_number: str,
    device_type: str,
) -> bool:
    """Validate lot number format against manufacturer patterns.

    Args:
        lot_number: Lot number string
        device_type: "ozempic_0.5mg" | "ozempic_1mg" | "wegovy_1.7mg" | etc.

    Returns:
        True if lot format is valid
    """
    # Extract manufacturer from device type
    if "ozempic" in device_type.lower():
        pattern = LOT_PATTERNS["ozempic"]
    elif "wegovy" in device_type.lower():
        pattern = LOT_PATTERNS["wegovy"]
    else:
        pattern = LOT_PATTERNS["novo_nordisk"]

    return bool(re.match(pattern, lot_number))


def verify_glp1_pen(
    serial_number: str,
    device_type: str,
    fill_imaging: Union[Dict, np.ndarray, List[float]],
    lot_number: str,
    provenance_chain: Optional[List[str]] = None,
    compliance_standard: str = "21_CFR_Part_820_QSR",
    config_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Verify GLP-1 pen authenticity via fill variance entropy.

    Args:
        serial_number: Pen serial number
        device_type: "ozempic_0.5mg" | "ozempic_1mg" | "wegovy_1.7mg" | "wegovy_2.4mg"
        fill_imaging: Dict with fill measurements (fill_level, compression, etc.)
                     Or array of optical/X-ray measurements
        lot_number: Manufacturer lot number
        provenance_chain: Chain of custody (manufacturer → distributor → pharmacy)
        compliance_standard: "21_CFR_Part_820_QSR" | "ISO_13485"
        config_dir: Optional path to config directory

    Returns:
        Tuple of (verdict, receipt)
        verdict: "AUTHENTIC" | "COUNTERFEIT" | "SUSPICIOUS"
        receipt: Complete verification receipt

    Notes:
        - If lot_format invalid: immediate COUNTERFEIT verdict
        - If entropy outside range: COUNTERFEIT
        - All GLP-1 pens are risk_level=CRITICAL
    """
    valid_types = ["ozempic_0.5mg", "ozempic_1mg", "wegovy_1.7mg", "wegovy_2.4mg"]
    if device_type not in valid_types:
        raise ValueError(f"Invalid device_type '{device_type}'. Must be one of: {valid_types}")

    # Validate lot number format first
    lot_format_valid = validate_lot_format(lot_number, device_type)

    # If lot format is invalid, immediate COUNTERFEIT
    if not lot_format_valid:
        from spaceproof.core import emit_receipt

        receipt = emit_receipt(
            "glp1_verification",
            {
                "tenant_id": "spaceproof-verification",
                "domain": "medical",
                "item_id": serial_number,
                "device_type": device_type,
                "lot_number": lot_number,
                "lot_format_valid": False,
                "verdict": "COUNTERFEIT",
                "confidence": 0.95,
                "risk_level": "CRITICAL",
                "flags": ["invalid_lot_format", "immediate_rejection"],
                "provenance_chain": provenance_chain or [],
                "compliance_standard": compliance_standard,
            },
        )
        return "COUNTERFEIT", receipt

    # Initialize verification engine
    engine = VerificationEngine(config_dir=config_dir)

    # Prepare sensor data
    if isinstance(fill_imaging, dict):
        sensor_data = fill_imaging
    elif isinstance(fill_imaging, np.ndarray):
        sensor_data = fill_imaging
    else:
        sensor_data = np.array(fill_imaging)

    # Verify using engine
    result = engine.verify(
        domain="medical",
        item_id=serial_number,
        sensor_data=sensor_data,
        baseline_key=f"glp1_pen.{device_type}",
        entropy_calculator=compute_fill_entropy,
        receipt_type="glp1_verification",
        provenance_chain=provenance_chain or [],
        compliance_standard=compliance_standard,
        risk_level="CRITICAL",
        additional_fields={
            "device_type": device_type,
            "product_type": "glp1_pen",
            "lot_number": lot_number,
            "lot_format_valid": lot_format_valid,
            "manufacturer": "Novo Nordisk",
        },
    )

    return result.verdict, result.receipt


def verify_glp1_batch(
    pens: List[Dict],
    config_dir: Optional[Path] = None,
) -> List[Tuple[str, Dict]]:
    """Verify multiple GLP-1 pens.

    Args:
        pens: List of dicts with keys:
            - serial_number: str
            - device_type: str
            - fill_imaging: Dict or array
            - lot_number: str
            - provenance_chain: List[str] (optional)
            - compliance_standard: str (optional)
        config_dir: Optional path to config directory

    Returns:
        List of (verdict, receipt) tuples
    """
    results = []

    for pen in pens:
        verdict, receipt = verify_glp1_pen(
            serial_number=pen["serial_number"],
            device_type=pen["device_type"],
            fill_imaging=pen["fill_imaging"],
            lot_number=pen["lot_number"],
            provenance_chain=pen.get("provenance_chain", []),
            compliance_standard=pen.get("compliance_standard", "21_CFR_Part_820_QSR"),
            config_dir=config_dir,
        )
        results.append((verdict, receipt))

    return results
