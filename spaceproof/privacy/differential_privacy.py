"""differential_privacy.py - Epsilon-differential privacy mechanisms.

Add calibrated noise to query results for privacy preservation.
"""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spaceproof.core import emit_receipt

# === CONSTANTS ===

PRIVACY_TENANT = "spaceproof-privacy"

# Default DP parameters
DEFAULT_EPSILON = 1.0
DEFAULT_DELTA = 1e-5
DEFAULT_SENSITIVITY = 1.0
MAX_EPSILON = 10.0


@dataclass
class PrivacyBudget:
    """Privacy budget tracker."""

    budget_id: str
    initial_budget: float
    remaining_budget: float
    queries_made: int
    reset_at: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "budget_id": self.budget_id,
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "queries_made": self.queries_made,
            "reset_at": self.reset_at,
            "created_at": self.created_at,
        }

    def spend(self, epsilon: float) -> bool:
        """Spend from budget.

        Args:
            epsilon: Amount to spend

        Returns:
            True if budget available, False if exhausted
        """
        if epsilon > self.remaining_budget:
            return False
        self.remaining_budget -= epsilon
        self.queries_made += 1
        return True


@dataclass
class DPResult:
    """Result of differential privacy operation."""

    operation_id: str
    mechanism: str
    epsilon: float
    delta: float
    sensitivity: float
    original_value: float
    noisy_value: float
    noise_added: float
    budget_remaining: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without original value for privacy)."""
        return {
            "operation_id": self.operation_id,
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            # Note: original_value is NOT included
            "noisy_value": self.noisy_value,
            "budget_remaining": self.budget_remaining,
            "timestamp": self.timestamp,
        }


# Global budget tracker
_budgets: Dict[str, PrivacyBudget] = {}


def get_or_create_budget(
    budget_id: str = "default",
    initial_budget: float = 10.0,
) -> PrivacyBudget:
    """Get or create a privacy budget.

    Args:
        budget_id: Budget identifier
        initial_budget: Initial budget if creating

    Returns:
        PrivacyBudget
    """
    if budget_id not in _budgets:
        reset_at = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).isoformat() + "Z"
        _budgets[budget_id] = PrivacyBudget(
            budget_id=budget_id,
            initial_budget=initial_budget,
            remaining_budget=initial_budget,
            queries_made=0,
            reset_at=reset_at,
        )
    return _budgets[budget_id]


def compute_sensitivity(
    query_type: str,
    data_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Compute sensitivity for query type.

    Args:
        query_type: Type of query (count, sum, average, etc.)
        data_range: Optional data range for bounded queries

    Returns:
        Sensitivity value
    """
    if query_type == "count":
        return 1.0
    elif query_type == "sum":
        if data_range:
            return abs(data_range[1] - data_range[0])
        return 1.0
    elif query_type == "average":
        if data_range:
            return abs(data_range[1] - data_range[0])
        return 1.0
    else:
        return DEFAULT_SENSITIVITY


def add_laplace_noise(
    value: float,
    epsilon: float = DEFAULT_EPSILON,
    sensitivity: float = DEFAULT_SENSITIVITY,
    budget_id: str = "default",
) -> DPResult:
    """Add Laplace noise to value.

    Args:
        value: Original value
        epsilon: Privacy parameter
        sensitivity: Query sensitivity
        budget_id: Budget to use

    Returns:
        DPResult with noisy value
    """
    budget = get_or_create_budget(budget_id)

    if not budget.spend(epsilon):
        # Budget exhausted - return very noisy result
        noise = np.random.laplace(0, sensitivity * 100)
        return DPResult(
            operation_id=str(uuid.uuid4()),
            mechanism="laplace_budget_exhausted",
            epsilon=epsilon,
            delta=0.0,
            sensitivity=sensitivity,
            original_value=value,
            noisy_value=value + noise,
            noise_added=noise,
            budget_remaining=0.0,
        )

    # Laplace scale = sensitivity / epsilon
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    noisy_value = value + noise

    return DPResult(
        operation_id=str(uuid.uuid4()),
        mechanism="laplace",
        epsilon=epsilon,
        delta=0.0,
        sensitivity=sensitivity,
        original_value=value,
        noisy_value=noisy_value,
        noise_added=noise,
        budget_remaining=budget.remaining_budget,
    )


def add_gaussian_noise(
    value: float,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    sensitivity: float = DEFAULT_SENSITIVITY,
    budget_id: str = "default",
) -> DPResult:
    """Add Gaussian noise to value.

    Args:
        value: Original value
        epsilon: Privacy parameter
        delta: Failure probability
        sensitivity: Query sensitivity
        budget_id: Budget to use

    Returns:
        DPResult with noisy value
    """
    budget = get_or_create_budget(budget_id)

    if not budget.spend(epsilon):
        # Budget exhausted
        noise = np.random.normal(0, sensitivity * 100)
        return DPResult(
            operation_id=str(uuid.uuid4()),
            mechanism="gaussian_budget_exhausted",
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            original_value=value,
            noisy_value=value + noise,
            noise_added=noise,
            budget_remaining=0.0,
        )

    # Gaussian scale = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, scale)
    noisy_value = value + noise

    return DPResult(
        operation_id=str(uuid.uuid4()),
        mechanism="gaussian",
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        original_value=value,
        noisy_value=noisy_value,
        noise_added=noise,
        budget_remaining=budget.remaining_budget,
    )


def check_privacy_budget(budget_id: str = "default") -> Dict[str, Any]:
    """Check privacy budget status.

    Args:
        budget_id: Budget identifier

    Returns:
        Budget status
    """
    budget = get_or_create_budget(budget_id)
    return {
        **budget.to_dict(),
        "is_exhausted": budget.remaining_budget <= 0,
        "utilization": (budget.initial_budget - budget.remaining_budget) / budget.initial_budget * 100,
    }


def emit_dp_receipt(result: DPResult) -> Dict[str, Any]:
    """Emit differential privacy receipt.

    Args:
        result: DPResult to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "differential_privacy",
        {
            "tenant_id": PRIVACY_TENANT,
            **result.to_dict(),
        },
    )


def reset_budget(budget_id: str = "default") -> None:
    """Reset a privacy budget.

    Args:
        budget_id: Budget identifier
    """
    if budget_id in _budgets:
        budget = _budgets[budget_id]
        budget.remaining_budget = budget.initial_budget
        budget.queries_made = 0


def clear_all_budgets() -> None:
    """Clear all budgets (for testing)."""
    global _budgets
    _budgets = {}


# Aliases for API compatibility
def get_privacy_budget(budget_id: str = "default") -> Dict[str, Any]:
    """Alias for check_privacy_budget."""
    return check_privacy_budget(budget_id)


def spend_privacy_budget(
    budget_id: str = "default",
    epsilon: float = 1.0,
) -> bool:
    """Spend from privacy budget.

    Args:
        budget_id: Budget identifier
        epsilon: Amount to spend

    Returns:
        True if successful, False if budget exhausted
    """
    budget = get_or_create_budget(budget_id)
    return budget.spend(epsilon)
