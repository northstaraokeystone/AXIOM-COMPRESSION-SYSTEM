"""base.py - Abstract base class for AGI audit modules.

Consolidates the common patterns from:
- agi_audit_expanded.py
- fractal_encrypt_audit.py
- randomized_paths_audit.py
- quantum_resist_random.py
- secure_enclave_audit.py

All audit modules share similar patterns for:
- Config loading
- Attack simulation
- Recovery computation
- Receipt emission
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash
from ..utils.constants import EXPANDED_RECOVERY_THRESHOLD


class AuditModuleBase(ABC):
    """Abstract base class for AGI audit modules.

    Subclasses implement:
    - _load_audit_config(): Load audit-specific config
    - _simulate_attack(): Simulate a specific attack type
    - _compute_recovery(): Compute recovery from attack
    """

    # Class-level defaults (overridden by subclasses)
    AUDIT_NAME: str = ""
    TENANT_ID: str = ""
    RECOVERY_THRESHOLD: float = EXPANDED_RECOVERY_THRESHOLD
    ATTACK_TYPES: List[str] = []

    def __init__(self):
        """Initialize audit module instance."""
        self._config = None

    @abstractmethod
    def _load_audit_config(self) -> Dict[str, Any]:
        """Load audit-specific configuration.

        Returns:
            Dict with audit configuration
        """
        pass

    @abstractmethod
    def _simulate_attack(
        self, attack_type: str, severity: float = 0.5
    ) -> Dict[str, Any]:
        """Simulate a specific attack type.

        Args:
            attack_type: Type of attack to simulate
            severity: Attack severity (0-1)

        Returns:
            Dict with attack simulation results including 'recovery'
        """
        pass

    def load_config(self) -> Dict[str, Any]:
        """Load audit configuration with receipt emission.

        Returns:
            Dict with audit configuration
        """
        config = self._load_audit_config()
        self._config = config

        emit_receipt(
            f"{self.AUDIT_NAME.lower()}_config",
            {
                "receipt_type": f"{self.AUDIT_NAME.lower()}_config",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                **{k: v for k, v in config.items() if not isinstance(v, (dict, list))},
                "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
            },
        )

        return config

    def simulate_attack(
        self, attack_type: str, severity: float = 0.5
    ) -> Dict[str, Any]:
        """Simulate an attack with receipt emission.

        Args:
            attack_type: Type of attack
            severity: Attack severity (0-1)

        Returns:
            Dict with attack results
        """
        result = self._simulate_attack(attack_type, severity)

        # Add common fields
        result["attack_type"] = attack_type
        result["severity"] = severity
        result["recovered"] = result.get("recovery", 0) >= self.RECOVERY_THRESHOLD

        emit_receipt(
            f"{self.AUDIT_NAME.lower()}_{attack_type}",
            {
                "receipt_type": f"{self.AUDIT_NAME.lower()}_{attack_type}",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "attack_type": attack_type,
                "severity": severity,
                "recovery": result.get("recovery", 0),
                "recovered": result["recovered"],
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )

        return result

    def run_audit(
        self,
        attack_types: List[str] = None,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """Run full audit with specified attack types.

        Args:
            attack_types: List of attack types to test (default: all)
            iterations: Number of iterations per attack type

        Returns:
            Dict with audit results
        """
        import random

        if attack_types is None:
            attack_types = self.ATTACK_TYPES

        if self._config is None:
            self.load_config()
        results = []

        for attack_type in attack_types:
            for _ in range(iterations // len(attack_types)):
                severity = random.uniform(0.1, 0.9)
                result = self.simulate_attack(attack_type, severity)
                results.append(result)

        # Compute aggregate metrics
        recoveries = [r.get("recovery", 0) for r in results]
        avg_recovery = sum(recoveries) / len(recoveries) if recoveries else 0

        recovered_count = sum(1 for r in results if r.get("recovered", False))
        recovery_rate = recovered_count / len(results) if results else 0

        audit_result = {
            "attack_types_tested": attack_types,
            "iterations": len(results),
            "avg_recovery": round(avg_recovery, 4),
            "recovery_rate": round(recovery_rate, 4),
            "recovered_count": recovered_count,
            "failed_count": len(results) - recovered_count,
            "recovery_threshold": self.RECOVERY_THRESHOLD,
            "recovery_passed": avg_recovery >= self.RECOVERY_THRESHOLD,
            "overall_classification": (
                "aligned" if avg_recovery >= self.RECOVERY_THRESHOLD else "misaligned"
            ),
        }

        emit_receipt(
            f"{self.AUDIT_NAME.lower()}_audit",
            {
                "receipt_type": f"{self.AUDIT_NAME.lower()}_audit",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                **audit_result,
                "payload_hash": dual_hash(json.dumps(audit_result, sort_keys=True)),
            },
        )

        return audit_result

    def get_info(self) -> Dict[str, Any]:
        """Get audit module info.

        Returns:
            Dict with module info
        """
        config = self.load_config() if self._config is None else self._config

        info = {
            "module": self.AUDIT_NAME,
            "version": "1.0.0",
            "config": config,
            "attack_types": self.ATTACK_TYPES,
            "thresholds": {
                "recovery_threshold": self.RECOVERY_THRESHOLD,
            },
            "description": f"{self.AUDIT_NAME} audit module",
        }

        emit_receipt(
            f"{self.AUDIT_NAME.lower()}_info",
            {
                "receipt_type": f"{self.AUDIT_NAME.lower()}_info",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "version": info["version"],
                "recovery_threshold": self.RECOVERY_THRESHOLD,
                "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
            },
        )

        return info
