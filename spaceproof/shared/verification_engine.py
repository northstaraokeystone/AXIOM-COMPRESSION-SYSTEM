"""verification_engine.py - Universal Entropy-Based Verification Engine.

THE PARADIGM:
    Aerospace: Rework accumulation → LOW entropy → COUNTERFEIT
    Food: Adulteration homogenization → LOW entropy → COUNTERFEIT
    Medical: Counterfeit uniformity → LOW/HIGH entropy → COUNTERFEIT

    Same physics. Same detection. Different domains.

Source: SpaceProof Multi-Domain Expansion v2.0
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
import numpy as np

from spaceproof.core import emit_receipt, dual_hash

# === CONSTANTS ===

TENANT_ID = "spaceproof-verification"

# Verdict thresholds (percentage of baseline range)
COUNTERFEIT_THRESHOLD_PCT = 0.85  # <85% of min = COUNTERFEIT
SUSPICIOUS_THRESHOLD_PCT = 1.15   # >115% of max = SUSPICIOUS

# Confidence calculation
MIN_CONFIDENCE = 0.50
MAX_CONFIDENCE = 0.99


@dataclass
class BaselineConfig:
    """Baseline configuration for a verification target."""

    entropy_min: float
    entropy_max: float
    domain: str
    item_type: str
    source: str = "Grok research"
    additional_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entropy_min": self.entropy_min,
            "entropy_max": self.entropy_max,
            "domain": self.domain,
            "item_type": self.item_type,
            "source": self.source,
            "additional_params": self.additional_params or {},
        }


@dataclass
class EntropyAnalysis:
    """Result of entropy analysis."""

    measured_entropy: float
    baseline_min: float
    baseline_max: float
    deviation_pct: float
    within_range: bool
    flags: List[str]
    comparison: str  # Human-readable explanation


@dataclass
class VerificationResult:
    """Result of verification."""

    verdict: Literal["AUTHENTIC", "COUNTERFEIT", "SUSPICIOUS"]
    confidence: float
    risk_score: float  # Inverse of confidence
    entropy_analysis: EntropyAnalysis
    receipt: Dict[str, Any]
    receipt_id: str
    domain: str
    item_id: str


class VerificationEngine:
    """Domain-agnostic entropy-based verification engine.

    Universal verification pattern:
    1. Compute entropy using domain-specific calculator
    2. Load baseline for item type
    3. Compare measured vs expected
    4. Classify verdict based on deviation
    5. Emit receipt with full context
    6. Return verdict + receipt
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize verification engine.

        Args:
            config_dir: Directory containing baseline JSON files.
                        Defaults to spaceproof/config/
        """
        if config_dir is None:
            # Default to config/ relative to spaceproof package
            config_dir = Path(__file__).parent.parent.parent / "config"

        self.config_dir = Path(config_dir)
        self._baseline_cache: Dict[str, Dict] = {}

    def load_baseline(self, domain: str, baseline_key: str) -> BaselineConfig:
        """Load baseline configuration for a domain and item type.

        Args:
            domain: "aerospace" | "food" | "medical"
            baseline_key: Dot-notation key like "olive_oil.extra_virgin"

        Returns:
            BaselineConfig with entropy bounds

        Raises:
            ValueError: If baseline not found
        """
        # Check cache first
        cache_key = f"{domain}.{baseline_key}"
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        # Load baseline file
        baseline_file = self.config_dir / f"{domain}_baselines.json"

        if not baseline_file.exists():
            raise ValueError(f"Baseline file not found: {baseline_file}")

        with open(baseline_file) as f:
            baselines = json.load(f)

        # Navigate to baseline using dot notation
        # Only split on first dot to preserve version numbers like "0.5mg"
        parts = baseline_key.split(".", 1)
        current = baselines

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Baseline not found: {domain}.{baseline_key}")

        # Extract entropy bounds
        entropy_min = current.get("entropy_min", current.get("spectral_entropy_min",
                                  current.get("texture_entropy_min", current.get("fill_entropy_min", 0.0))))
        entropy_max = current.get("entropy_max", current.get("spectral_entropy_max",
                                  current.get("texture_entropy_max", current.get("fill_entropy_max", 1.0))))

        config = BaselineConfig(
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            domain=domain,
            item_type=baseline_key,
            source=current.get("source", "Grok research"),
            additional_params=current,
        )

        # Cache for future use
        self._baseline_cache[cache_key] = config

        return config

    def classify_verdict(
        self,
        measured_entropy: float,
        baseline: BaselineConfig,
    ) -> tuple[Literal["AUTHENTIC", "COUNTERFEIT", "SUSPICIOUS"], float, List[str]]:
        """Classify verdict based on entropy comparison.

        Args:
            measured_entropy: Computed entropy value
            baseline: Baseline configuration

        Returns:
            Tuple of (verdict, confidence, flags)
        """
        flags: List[str] = []

        # Calculate thresholds
        min_threshold = baseline.entropy_min * COUNTERFEIT_THRESHOLD_PCT
        max_threshold = baseline.entropy_max * SUSPICIOUS_THRESHOLD_PCT

        # Check against thresholds
        if measured_entropy < min_threshold:
            # Too uniform - homogenization detected (adulteration/counterfeit)
            verdict = "COUNTERFEIT"
            deviation = (min_threshold - measured_entropy) / min_threshold
            confidence = min(MAX_CONFIDENCE, MIN_CONFIDENCE + deviation * 0.5)
            flags.append("homogenization_detected")
            flags.append(f"entropy_below_threshold_{min_threshold:.2f}")

        elif measured_entropy > max_threshold:
            # Abnormal variance - suspicious
            verdict = "SUSPICIOUS"
            deviation = (measured_entropy - max_threshold) / max_threshold
            confidence = min(MAX_CONFIDENCE, MIN_CONFIDENCE + deviation * 0.3)
            flags.append("abnormal_variance")
            flags.append(f"entropy_above_threshold_{max_threshold:.2f}")

        elif measured_entropy < baseline.entropy_min:
            # Below range but not critical
            verdict = "SUSPICIOUS"
            deviation = (baseline.entropy_min - measured_entropy) / baseline.entropy_min
            confidence = MIN_CONFIDENCE + deviation * 0.2
            flags.append("below_baseline_range")

        elif measured_entropy > baseline.entropy_max:
            # Above range but not critical
            verdict = "SUSPICIOUS"
            deviation = (measured_entropy - baseline.entropy_max) / baseline.entropy_max
            confidence = MIN_CONFIDENCE + deviation * 0.2
            flags.append("above_baseline_range")

        else:
            # Within expected range
            verdict = "AUTHENTIC"
            # Confidence based on how centered the measurement is
            midpoint = (baseline.entropy_min + baseline.entropy_max) / 2
            range_half = (baseline.entropy_max - baseline.entropy_min) / 2
            if range_half > 0:
                distance_from_center = abs(measured_entropy - midpoint) / range_half
                confidence = MAX_CONFIDENCE - distance_from_center * 0.15
            else:
                confidence = MAX_CONFIDENCE
            flags.append("within_expected_range")

        return verdict, confidence, flags

    def compute_deviation(self, measured_entropy: float, baseline: BaselineConfig) -> float:
        """Compute percentage deviation from baseline range.

        Args:
            measured_entropy: Computed entropy
            baseline: Baseline configuration

        Returns:
            Deviation percentage (0.0 = perfect center, positive = outside range)
        """
        midpoint = (baseline.entropy_min + baseline.entropy_max) / 2
        range_half = (baseline.entropy_max - baseline.entropy_min) / 2

        if range_half <= 0:
            return 0.0

        distance = abs(measured_entropy - midpoint)

        if measured_entropy < baseline.entropy_min or measured_entropy > baseline.entropy_max:
            # Outside range
            if measured_entropy < baseline.entropy_min:
                deviation = (baseline.entropy_min - measured_entropy) / range_half
            else:
                deviation = (measured_entropy - baseline.entropy_max) / range_half
            return deviation * 100  # As percentage
        else:
            # Inside range - negative deviation
            return (distance / range_half - 1) * 100

    def verify(
        self,
        domain: str,
        item_id: str,
        sensor_data: Union[Dict, np.ndarray, List[float]],
        baseline_key: str,
        entropy_calculator: Callable,
        receipt_type: str,
        provenance_chain: Optional[List[str]] = None,
        compliance_standard: Optional[str] = None,
        risk_level: Optional[str] = None,
        additional_fields: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Universal verification pattern.

        Args:
            domain: "aerospace" | "food" | "medical"
            item_id: Component ID, batch ID, or serial number
            sensor_data: Raw measurements from sensors
            baseline_key: Key to lookup in config/{domain}_baselines.json
            entropy_calculator: Function that computes entropy from sensor_data
            receipt_type: Type for emit_receipt()
            provenance_chain: Optional chain of custody
            compliance_standard: Optional compliance standard (FSMA_204, CFR_820_QSR, etc.)
            risk_level: Optional risk level for medical domain
            additional_fields: Optional additional fields for receipt

        Returns:
            VerificationResult with verdict, receipt, and analysis
        """
        # 1. Compute entropy using domain-specific calculator
        measured_entropy = entropy_calculator(sensor_data)

        # 2. Load baseline
        baseline = self.load_baseline(domain, baseline_key)

        # 3. Compare and classify
        verdict, confidence, flags = self.classify_verdict(measured_entropy, baseline)

        # 4. Compute deviation
        deviation_pct = self.compute_deviation(measured_entropy, baseline)

        # 5. Build entropy analysis
        within_range = baseline.entropy_min <= measured_entropy <= baseline.entropy_max

        if verdict == "AUTHENTIC":
            comparison = f"Entropy {measured_entropy:.3f} within expected range [{baseline.entropy_min:.3f}, {baseline.entropy_max:.3f}]"
        elif verdict == "COUNTERFEIT":
            comparison = f"Entropy {measured_entropy:.3f} below minimum threshold {baseline.entropy_min * COUNTERFEIT_THRESHOLD_PCT:.3f} - homogenization/counterfeit detected"
        else:
            comparison = f"Entropy {measured_entropy:.3f} outside expected range [{baseline.entropy_min:.3f}, {baseline.entropy_max:.3f}] - requires investigation"

        entropy_analysis = EntropyAnalysis(
            measured_entropy=measured_entropy,
            baseline_min=baseline.entropy_min,
            baseline_max=baseline.entropy_max,
            deviation_pct=deviation_pct,
            within_range=within_range,
            flags=flags,
            comparison=comparison,
        )

        # 6. Build receipt data
        # Hash sensor data (never include raw data in receipts)
        if isinstance(sensor_data, dict):
            sensor_hash = dual_hash(json.dumps(sensor_data, sort_keys=True))
        elif isinstance(sensor_data, np.ndarray):
            sensor_hash = dual_hash(sensor_data.tobytes())
        else:
            sensor_hash = dual_hash(json.dumps(list(sensor_data), sort_keys=True))

        receipt_data = {
            "tenant_id": TENANT_ID,
            "domain": domain,
            "item_id": item_id,
            "measured_entropy": measured_entropy,
            "baseline_min": baseline.entropy_min,
            "baseline_max": baseline.entropy_max,
            "deviation_pct": deviation_pct,
            "verdict": verdict,
            "confidence": confidence,
            "flags": flags,
            "sensor_data_hash": sensor_hash,
            "baseline_key": baseline_key,
            "provenance_chain": provenance_chain or [],
        }

        if compliance_standard:
            receipt_data["compliance_standard"] = compliance_standard

        if risk_level:
            receipt_data["risk_level"] = risk_level

        if additional_fields:
            receipt_data.update(additional_fields)

        # 7. Emit receipt
        receipt = emit_receipt(receipt_type, receipt_data)

        # 8. Return result
        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            risk_score=1.0 - confidence,
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", ""),
            domain=domain,
            item_id=item_id,
        )

    def verify_batch(
        self,
        domain: str,
        items: List[Dict],
        baseline_key: str,
        entropy_calculator: Callable,
        receipt_type: str,
        **kwargs,
    ) -> List[VerificationResult]:
        """Verify a batch of items.

        Args:
            domain: Domain name
            items: List of items with 'id' and 'sensor_data' keys
            baseline_key: Baseline key for all items
            entropy_calculator: Entropy calculator function
            receipt_type: Receipt type
            **kwargs: Additional arguments passed to verify()

        Returns:
            List of VerificationResult objects
        """
        results = []

        for item in items:
            result = self.verify(
                domain=domain,
                item_id=item.get("id", item.get("item_id", "unknown")),
                sensor_data=item.get("sensor_data", item),
                baseline_key=baseline_key,
                entropy_calculator=entropy_calculator,
                receipt_type=receipt_type,
                **kwargs,
            )
            results.append(result)

        return results
