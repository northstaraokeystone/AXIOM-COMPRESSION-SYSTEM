"""protocol.py - ValidationModule Protocol Interface.

THE PROTOCOL PARADIGM:
    Each module must implement a base interface for:
    - Validation
    - Receipt field generation
    - Saga compensation (rollback)

This enables the 3-of-7 module selection pattern where each domain
configuration activates exactly three modules from the pool.

Available modules:
    - compress: Entropy reduction
    - witness: Observation/attestation (KAN-based)
    - sovereignty: Ownership verification
    - ledger: Immutable recording
    - detect: Anomaly detection
    - anchor: Cryptographic timestamping
    - loop: Feedback iteration

Source: SpaceProof D20 Production Evolution
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ProofInput:
    """Input for validation modules."""

    data: Any
    config: Dict[str, Any]
    context: Dict[str, Any]
    previous_results: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result from module validation."""

    passed: bool
    entropy_delta: float
    coherence_score: float
    data: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ModuleState:
    """State for compensation/rollback."""

    module_id: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    changes: List[Dict[str, Any]]


@runtime_checkable
class ValidationModule(Protocol):
    """Protocol that all validation modules must implement.

    This is the base interface for SpaceProof's 7 validation modules:
    compress, witness, sovereignty, ledger, detect, anchor, loop.

    Each module in the 3-of-7 selection must implement:
    - MODULE_ID: Unique identifier
    - CAPABILITIES: List of capabilities provided
    - validate(): Core validation logic
    - generate_receipt_fields(): Receipt field generation
    - compensate(): Saga rollback support
    """

    MODULE_ID: str
    CAPABILITIES: List[str]

    def validate(self, input: ProofInput) -> ValidationResult:
        """Execute module validation.

        Args:
            input: ProofInput with data, config, and context

        Returns:
            ValidationResult with pass/fail and metrics
        """
        ...

    def generate_receipt_fields(self) -> Dict:
        """Generate module-specific receipt fields.

        Returns:
            Dict of fields to include in receipt
        """
        ...

    def compensate(self, state: ModuleState) -> bool:
        """Execute compensating transaction for rollback.

        Args:
            state: ModuleState with before/after state

        Returns:
            True if compensation succeeded
        """
        ...


# === MODULE CAPABILITY CONSTANTS ===

CAP_COMPRESS = "entropy_reduction"
CAP_WITNESS = "law_discovery"
CAP_SOVEREIGNTY = "autonomy_calculation"
CAP_LEDGER = "immutable_storage"
CAP_DETECT = "anomaly_detection"
CAP_ANCHOR = "cryptographic_proof"
CAP_LOOP = "feedback_iteration"


# === MODULE REGISTRY ===


class ModuleRegistry:
    """Registry of available validation modules."""

    def __init__(self):
        self._modules: Dict[str, type] = {}
        self._instances: Dict[str, ValidationModule] = {}

    def register(self, module_class: type) -> None:
        """Register a module class.

        Args:
            module_class: Class implementing ValidationModule
        """
        if hasattr(module_class, "MODULE_ID"):
            self._modules[module_class.MODULE_ID] = module_class

    def get(self, module_id: str) -> Optional[ValidationModule]:
        """Get module instance by ID.

        Args:
            module_id: Module identifier

        Returns:
            ValidationModule instance or None
        """
        if module_id in self._instances:
            return self._instances[module_id]

        if module_id in self._modules:
            instance = self._modules[module_id]()
            self._instances[module_id] = instance
            return instance

        return None

    def get_by_capability(self, capability: str) -> List[ValidationModule]:
        """Get all modules with a capability.

        Args:
            capability: Capability to search for

        Returns:
            List of modules with the capability
        """
        result = []
        for module_id, module_class in self._modules.items():
            if hasattr(module_class, "CAPABILITIES"):
                if capability in module_class.CAPABILITIES:
                    result.append(self.get(module_id))
        return result

    def list_modules(self) -> List[str]:
        """List all registered module IDs.

        Returns:
            List of module IDs
        """
        return list(self._modules.keys())


# Global registry
_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    """Get the global module registry.

    Returns:
        ModuleRegistry instance
    """
    return _registry


def register_module(module_class: type) -> type:
    """Decorator to register a module class.

    Usage:
        @register_module
        class MyModule:
            MODULE_ID = "my_module"
            CAPABILITIES = ["some_capability"]
            ...

    Args:
        module_class: Class to register

    Returns:
        The class (unmodified)
    """
    _registry.register(module_class)
    return module_class


# === 3-OF-7 MODULE SELECTION ===


def validate_module_selection(modules: List[str]) -> bool:
    """Validate that exactly 3 modules are selected.

    Args:
        modules: List of module IDs

    Returns:
        True if selection is valid
    """
    if len(modules) != 3:
        return False

    valid_modules = {"compress", "witness", "sovereignty", "ledger", "detect", "anchor", "loop"}
    return all(m in valid_modules for m in modules)


def get_domain_modules(domain: str) -> List[str]:
    """Get the 3 modules for a domain configuration.

    Args:
        domain: Domain name (xai, doge, nasa, defense, dot)

    Returns:
        List of 3 module IDs
    """
    domain_modules = {
        "xai": ["compress", "witness", "sovereignty"],
        "doge": ["ledger", "detect", "anchor"],
        "nasa": ["compress", "sovereignty", "loop"],
        "defense": ["compress", "ledger", "anchor"],
        "dot": ["compress", "ledger", "detect"],
    }
    return domain_modules.get(domain.lower(), ["compress", "ledger", "anchor"])


# === MODULE COMPOSITION ===


def count_possible_compositions() -> int:
    """Count possible 3-of-7 module compositions.

    C(7,3) = 35 possible configurations.

    Returns:
        Number of possible configurations
    """
    from math import comb

    return comb(7, 3)


def list_all_compositions() -> List[List[str]]:
    """List all possible 3-of-7 module compositions.

    Returns:
        List of module ID lists
    """
    from itertools import combinations

    modules = ["compress", "witness", "sovereignty", "ledger", "detect", "anchor", "loop"]
    return [list(combo) for combo in combinations(modules, 3)]
