"""botox.py - Botox Counterfeit Detection.

Detects counterfeit Botox vials via surface/solution entropy.

Genuine vials: Controlled random variance in surface finish → Entropy range (3.2-3.9)
Counterfeits: Altered surface entropy (cheap vials) + concentration anomalies

Detection target: ≥99.9% recall (botulism risk if fake)
Compliance: 21 CFR Part 820, ISO 13485

Source: Grok Research - Counterfeit Botox detection
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from spaceproof.shared.verification_engine import VerificationEngine
from .entropy import surface_topography_entropy, structural_entropy


def compute_surface_entropy(
    surface_scan: Union[np.ndarray, List[float], Dict],
) -> float:
    """Compute surface entropy from laser speckle or optical scanning.

    Analyzes surface topography randomness.

    Args:
        surface_scan: Laser speckle pattern or optical surface data

    Returns:
        Surface entropy value
    """
    if isinstance(surface_scan, dict):
        scan_data = surface_scan.get("scan", surface_scan.get("surface_scan", []))
        if isinstance(scan_data, dict):
            scan_data = list(scan_data.values())
    elif isinstance(surface_scan, list):
        scan_data = surface_scan
    else:
        scan_data = surface_scan

    return surface_topography_entropy(np.array(scan_data))


def compute_solution_entropy(
    solution_analysis: Dict,
) -> float:
    """Compute solution entropy from concentration analysis.

    Analyzes concentration variance and particulate distribution.

    Args:
        solution_analysis: Dict with concentration, particulate data

    Returns:
        Solution entropy value
    """
    values = []

    # Extract concentration-related values
    for key in ["concentration", "concentration_variance", "particulate_count",
                "particulate_distribution", "density"]:
        if key in solution_analysis:
            val = solution_analysis[key]
            if isinstance(val, (int, float)):
                values.append(val)
            elif isinstance(val, (list, np.ndarray)):
                values.extend(val if isinstance(val, list) else val.tolist())

    if not values:
        return 0.0

    arr = np.array(values)
    return structural_entropy(arr)


def compute_combined_botox_entropy(
    surface_scan: Union[np.ndarray, List[float], Dict],
    solution_analysis: Optional[Dict] = None,
) -> float:
    """Compute combined entropy for Botox verification.

    Weighted: 50% surface + 50% solution

    Args:
        surface_scan: Surface scan data
        solution_analysis: Optional solution analysis data

    Returns:
        Combined entropy value
    """
    surface_ent = compute_surface_entropy(surface_scan)

    if solution_analysis is None:
        return surface_ent

    solution_ent = compute_solution_entropy(solution_analysis)

    # 50/50 weighting
    return surface_ent * 0.5 + solution_ent * 0.5


def verify_botox_vial(
    vial_id: str,
    unit_count: int,
    surface_scan: Union[np.ndarray, List[float]],
    solution_analysis: Optional[Dict] = None,
    provenance_chain: Optional[List[str]] = None,
    compliance_standard: str = "21_CFR_Part_820_QSR",
    config_dir: Optional[Path] = None,
) -> Tuple[str, Dict]:
    """Verify Botox vial authenticity via surface/solution entropy.

    Args:
        vial_id: Vial identifier
        unit_count: 50 | 100 | 200 (units)
        surface_scan: Laser speckle or optical scanning data
        solution_analysis: Optional dict with concentration, particulate data
        provenance_chain: Chain of custody
        compliance_standard: "21_CFR_Part_820_QSR" | "ISO_13485"
        config_dir: Optional path to config directory

    Returns:
        Tuple of (verdict, receipt)
        verdict: "AUTHENTIC" | "COUNTERFEIT" | "SUSPICIOUS"
        receipt: Complete verification receipt
    """
    valid_units = [50, 100, 200]
    if unit_count not in valid_units:
        raise ValueError(f"Invalid unit_count {unit_count}. Must be one of: {valid_units}")

    # Initialize verification engine
    engine = VerificationEngine(config_dir=config_dir)

    # Prepare sensor data
    if isinstance(surface_scan, np.ndarray):
        sensor_data = surface_scan
    else:
        sensor_data = np.array(surface_scan)

    # Create combined entropy calculator
    def combined_calculator(data):
        return compute_combined_botox_entropy(data, solution_analysis)

    # Extract solution metrics for receipt
    surface_ent = compute_surface_entropy(sensor_data)
    solution_ent = 0.0
    if solution_analysis:
        solution_ent = compute_solution_entropy(solution_analysis)

    # Verify using engine
    result = engine.verify(
        domain="medical",
        item_id=vial_id,
        sensor_data=sensor_data,
        baseline_key=f"botox_vial.{unit_count}unit",
        entropy_calculator=combined_calculator,
        receipt_type="botox_verification",
        provenance_chain=provenance_chain or [],
        compliance_standard=compliance_standard,
        risk_level="CRITICAL",
        additional_fields={
            "unit_count": unit_count,
            "product_type": "botox_vial",
            "surface_entropy": surface_ent,
            "solution_entropy": solution_ent,
            "manufacturer": "Allergan",
        },
    )

    return result.verdict, result.receipt


def verify_botox_batch(
    vials: List[Dict],
    config_dir: Optional[Path] = None,
) -> List[Tuple[str, Dict]]:
    """Verify multiple Botox vials.

    Args:
        vials: List of dicts with keys:
            - vial_id: str
            - unit_count: int
            - surface_scan: array-like
            - solution_analysis: Dict (optional)
            - provenance_chain: List[str] (optional)
            - compliance_standard: str (optional)
        config_dir: Optional path to config directory

    Returns:
        List of (verdict, receipt) tuples
    """
    results = []

    for vial in vials:
        verdict, receipt = verify_botox_vial(
            vial_id=vial["vial_id"],
            unit_count=vial["unit_count"],
            surface_scan=vial["surface_scan"],
            solution_analysis=vial.get("solution_analysis"),
            provenance_chain=vial.get("provenance_chain", []),
            compliance_standard=vial.get("compliance_standard", "21_CFR_Part_820_QSR"),
            config_dir=config_dir,
        )
        results.append((verdict, receipt))

    return results
