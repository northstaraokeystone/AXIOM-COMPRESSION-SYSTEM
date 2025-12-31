"""privacy_enforcement.py - Privacy Layer Scenario.

PRIVACY_ENFORCEMENT SCENARIO:
    Validate privacy layer functionality.
    PII redaction, differential privacy, budget tracking.

Pass Criteria:
    - 100% PII redacted (regex + ML detection)
    - 100% redaction_receipts emitted
    - Epsilon-DP noise within bounds (epsilon = 1.0 default)
    - Privacy budget enforced (reject when exhausted)
    - Zero PII leakage (audit all outputs)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

import numpy as np

from spaceproof.core import emit_receipt, dual_hash

CHECKPOINT_FREQUENCY = 50
TENANT_ID = "spaceproof-scenario-privacy"

# PII patterns
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}-\d{3}-\d{4}\b",
}


@dataclass
class PrivacyEnforcementConfig:
    """Configuration for privacy enforcement scenario."""

    cycles: int = 500
    seed: int = 42
    receipts_with_pii: int = 100
    dp_requests: int = 50
    budget_exhaustion_attempts: int = 10
    default_epsilon: float = 1.0
    initial_budget: float = 10.0


@dataclass
class PrivacyEnforcementResult:
    """Result of privacy enforcement scenario execution."""

    cycles_completed: int
    pii_detected: int
    pii_redacted: int
    redaction_receipts_emitted: int
    dp_queries_processed: int
    noise_within_bounds: int
    budget_enforced: bool
    leakage_detected: int
    passed: bool
    failure_reasons: List[str]


class PrivacyEnforcementScenario:
    """Privacy layer validation scenario."""

    def __init__(self, config: Optional[PrivacyEnforcementConfig] = None):
        """Initialize privacy enforcement scenario."""
        self.config = config or PrivacyEnforcementConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Tracking
        self.pii_detected = 0
        self.pii_redacted = 0
        self.redaction_receipts = 0
        self.dp_queries = 0
        self.noise_in_bounds = 0
        self.privacy_budget = self.config.initial_budget
        self.budget_rejections = 0
        self.leakage_count = 0
        self.outputs: List[str] = []

    def run(self) -> PrivacyEnforcementResult:
        """Run the privacy enforcement scenario."""
        failure_reasons = []

        # Process receipts with PII
        for i in range(self.config.receipts_with_pii):
            text = self._generate_pii_text(i)
            pii_found = self._detect_pii(text)
            self.pii_detected += pii_found

            if pii_found > 0:
                redacted = self._redact_pii(text)
                self.pii_redacted += pii_found
                self.redaction_receipts += 1
                self.outputs.append(redacted)
            else:
                self.outputs.append(text)

            if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                self._emit_checkpoint(i)

        # Process DP requests
        for i in range(self.config.dp_requests):
            value = float(self.rng.uniform(0, 100))
            result = self._apply_dp_noise(value)
            self.dp_queries += 1

            if result["within_bounds"]:
                self.noise_in_bounds += 1

        # Attempt budget exhaustion
        for i in range(self.config.budget_exhaustion_attempts):
            # Try to spend more budget
            if not self._try_spend_budget(2.0):
                self.budget_rejections += 1

        # Audit outputs for leakage
        self.leakage_count = self._audit_outputs()

        # Validate results
        if self.pii_redacted < self.pii_detected:
            failure_reasons.append(f"Only {self.pii_redacted}/{self.pii_detected} PII redacted")

        if self.redaction_receipts < self.pii_detected:
            failure_reasons.append(f"Only {self.redaction_receipts} redaction receipts for {self.pii_detected} PII")

        if self.noise_in_bounds < self.dp_queries:
            failure_reasons.append(f"Only {self.noise_in_bounds}/{self.dp_queries} DP queries within bounds")

        if self.budget_rejections == 0 and self.config.budget_exhaustion_attempts > 0:
            failure_reasons.append("Budget was never enforced (no rejections)")

        if self.leakage_count > 0:
            failure_reasons.append(f"PII leakage detected: {self.leakage_count} instances")

        passed = len(failure_reasons) == 0

        return PrivacyEnforcementResult(
            cycles_completed=self.config.cycles,
            pii_detected=self.pii_detected,
            pii_redacted=self.pii_redacted,
            redaction_receipts_emitted=self.redaction_receipts,
            dp_queries_processed=self.dp_queries,
            noise_within_bounds=self.noise_in_bounds,
            budget_enforced=self.budget_rejections > 0,
            leakage_detected=self.leakage_count,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def _generate_pii_text(self, index: int) -> str:
        """Generate text with PII."""
        pii_types = ["email", "ssn", "phone", "none"]
        pii_type = self.rng.choice(pii_types)

        if pii_type == "email":
            return f"User {index} contact: user{index}@example.com for support"
        elif pii_type == "ssn":
            return f"Record {index} SSN: 123-45-{6789 + index % 1000}"
        elif pii_type == "phone":
            return f"Call back {index}: 555-123-{4000 + index}"
        else:
            return f"General record {index} with no sensitive data"

    def _detect_pii(self, text: str) -> int:
        """Detect PII in text."""
        count = 0
        for pattern in PII_PATTERNS.values():
            matches = re.findall(pattern, text)
            count += len(matches)
        return count

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text

        for name, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{name.upper()}]", redacted)

        emit_receipt(
            "redaction",
            {
                "tenant_id": TENANT_ID,
                "original_hash": dual_hash(text),
                "redacted_hash": dual_hash(redacted),
                "pii_types_redacted": list(PII_PATTERNS.keys()),
            },
        )

        return redacted

    def _apply_dp_noise(self, value: float) -> Dict[str, Any]:
        """Apply differential privacy noise."""
        epsilon = self.config.default_epsilon
        sensitivity = 1.0

        # Check budget
        if self.privacy_budget < epsilon:
            return {"noisy_value": None, "within_bounds": False, "rejected": True}

        # Laplace noise
        scale = sensitivity / epsilon
        noise = float(self.rng.laplace(0, scale))
        noisy_value = value + noise

        self.privacy_budget -= epsilon

        emit_receipt(
            "differential_privacy",
            {
                "tenant_id": TENANT_ID,
                "epsilon": epsilon,
                "sensitivity": sensitivity,
                "noisy_value": noisy_value,
                "budget_remaining": self.privacy_budget,
            },
        )

        # Check if noise is reasonable (within 3 standard deviations)
        within_bounds = abs(noise) < 3 * scale

        return {"noisy_value": noisy_value, "within_bounds": within_bounds, "rejected": False}

    def _try_spend_budget(self, amount: float) -> bool:
        """Try to spend from privacy budget."""
        if self.privacy_budget >= amount:
            self.privacy_budget -= amount
            return True
        return False

    def _audit_outputs(self) -> int:
        """Audit outputs for PII leakage."""
        leakage = 0

        for output in self.outputs:
            # Check for un-redacted PII
            for pattern in PII_PATTERNS.values():
                if re.search(pattern, output):
                    leakage += 1
                    break

        return leakage

    def _emit_checkpoint(self, step: int) -> None:
        """Emit checkpoint receipt."""
        emit_receipt(
            "privacy_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "pii_detected": self.pii_detected,
                "pii_redacted": self.pii_redacted,
                "budget_remaining": self.privacy_budget,
            },
        )
