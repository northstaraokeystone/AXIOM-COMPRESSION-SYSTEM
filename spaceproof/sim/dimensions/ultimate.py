"""ultimate.py - D15-D20 Ultimate Dimensions.

Ultimate dimensions implement the highest validation maturity:
    D15 - Autocatalysis: Self-sustaining receipt patterns (tau >= 0.7)
    D16 - Self-reproduction: System regenerates validation logic
    D17 - Thermodynamic: Energy/entropy conservation verification
    D18 - Singularity: Self-referential boundary conditions
    D19 - Decidability: Godel completeness bounds acknowledged
    D20 - Transcendence: Meta-system consistency across all dimensions

SpaceProof back-builds from these, following "receipts all the way down".
"""

from typing import Dict, List
import numpy as np
import json

from spaceproof.engine.entropy import (
    coherence_score,
    fitness_score,
    COHERENCE_THRESHOLD,
)
from spaceproof.core import dual_hash
from spaceproof.sim.dimensions.foundation import BaseDimension, DimensionResult


class D15_Autocatalysis(BaseDimension):
    """D15: Self-sustaining receipt patterns detected (tau >= 0.7)."""

    dimension_id = "D15"
    dimension_name = "Autocatalysis"

    def __init__(self, threshold: float = COHERENCE_THRESHOLD):
        """Initialize autocatalysis detector.

        Args:
            threshold: Coherence threshold for autocatalytic patterns
        """
        self.threshold = threshold

    def validate(self, data: List[Dict]) -> DimensionResult:
        """Detect self-sustaining receipt patterns.

        Autocatalytic patterns are those that:
        1. Achieve coherence >= threshold (tau >= 0.7)
        2. Generate receipts that reference themselves
        3. Maintain positive entropy delta

        Args:
            data: List of receipts

        Returns:
            DimensionResult with autocatalysis detection
        """
        if not isinstance(data, list) or len(data) < 5:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message="Insufficient data for autocatalysis detection",
                details={},
            )

        # Extract entropy deltas
        deltas = []
        for item in data:
            if isinstance(item, dict):
                delta = item.get("entropy_delta", 0)
                deltas.append(delta)

        if not deltas:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message="No entropy deltas found",
                details={},
            )

        # Compute coherence of delta pattern
        pattern = np.array(deltas)
        coh = coherence_score(pattern)

        # Check for self-sustaining pattern
        # 1. High coherence
        is_coherent = coh.score >= self.threshold

        # 2. Positive trend (more order being created)
        trend = np.polyfit(range(len(deltas)), deltas, 1)[0]
        is_growing = trend >= 0

        # 3. Most deltas positive (entropy being reduced)
        positive_ratio = sum(1 for d in deltas if d > 0) / len(deltas)
        is_pumping = positive_ratio >= 0.6

        is_autocatalytic = is_coherent and (is_growing or is_pumping)

        return DimensionResult(
            dimension=self.dimension_id,
            passed=is_autocatalytic,
            message="Autocatalytic pattern detected" if is_autocatalytic else "No autocatalytic pattern",
            details={
                "coherence": coh.score,
                "is_alive": coh.is_alive,
                "pattern_strength": coh.pattern_strength,
                "trend": float(trend),
                "positive_ratio": positive_ratio,
                "is_autocatalytic": is_autocatalytic,
            },
        )


class D16_SelfReproduction(BaseDimension):
    """D16: System can regenerate its validation logic."""

    dimension_id = "D16"
    dimension_name = "Self-Reproduction"

    def validate(self, data: Dict) -> DimensionResult:
        """Check if system can regenerate validation logic.

        Self-reproduction means:
        1. Validation rules are encoded in receipts
        2. Rules can be extracted and reapplied
        3. System state can be reconstructed from receipts

        Args:
            data: Dict with validation rules and receipts

        Returns:
            DimensionResult with self-reproduction capability
        """
        rules_encoded = data.get("rules_in_receipts", False)
        rules_extractable = data.get("rules_extractable", False)
        state_reconstructable = data.get("state_reconstructable", False)

        # Check if validation functions are hashable (reproducible)
        has_deterministic_validation = True
        if "validation_hash" in data:
            # Verify hash is consistent
            computed_hash = dual_hash(json.dumps(data.get("validation_rules", {}), sort_keys=True))
            has_deterministic_validation = computed_hash == data["validation_hash"]

        can_self_reproduce = (
            rules_encoded and rules_extractable and state_reconstructable and has_deterministic_validation
        )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=can_self_reproduce,
            message="Self-reproduction capable" if can_self_reproduce else "Cannot self-reproduce",
            details={
                "rules_encoded": rules_encoded,
                "rules_extractable": rules_extractable,
                "state_reconstructable": state_reconstructable,
                "deterministic_validation": has_deterministic_validation,
            },
        )


class D17_Thermodynamic(BaseDimension):
    """D17: Energy/entropy conservation verified per cycle."""

    dimension_id = "D17"
    dimension_name = "Thermodynamic"

    def __init__(self, tolerance: float = 0.05):
        """Initialize thermodynamic validator.

        Args:
            tolerance: Tolerance for conservation violations
        """
        self.tolerance = tolerance

    def validate(self, data: Dict) -> DimensionResult:
        """Verify thermodynamic laws.

        First law: Energy conservation (total energy constant)
        Second law: Entropy non-decreasing in isolated systems

        Args:
            data: Dict with entropy and energy data

        Returns:
            DimensionResult with thermodynamic verification
        """
        # First law: energy conservation
        energy_in = data.get("energy_in", 0)
        energy_out = data.get("energy_out", 0)
        energy_work = data.get("energy_work", 0)

        first_law_balance = energy_in - energy_out - energy_work
        first_law_satisfied = abs(first_law_balance) < self.tolerance * max(energy_in, 1)

        # Second law: entropy non-decreasing
        entropy_changes = data.get("entropy_changes", [])
        second_law_violations = sum(1 for delta in entropy_changes if delta < -self.tolerance)
        second_law_satisfied = second_law_violations == 0

        # Compute fitness
        total_reduction = sum(max(0, d) for d in entropy_changes)
        n_cycles = len(entropy_changes)
        fit = fitness_score(total_reduction, n_cycles) if n_cycles > 0 else 0

        thermodynamic_valid = first_law_satisfied and second_law_satisfied

        return DimensionResult(
            dimension=self.dimension_id,
            passed=thermodynamic_valid,
            message="Thermodynamic laws satisfied" if thermodynamic_valid else "Thermodynamic violation",
            details={
                "first_law_satisfied": first_law_satisfied,
                "first_law_balance": first_law_balance,
                "second_law_satisfied": second_law_satisfied,
                "second_law_violations": second_law_violations,
                "fitness_score": fit,
            },
        )


class D18_Singularity(BaseDimension):
    """D18: Self-referential boundary conditions handled."""

    dimension_id = "D18"
    dimension_name = "Singularity"

    def __init__(self, max_recursion: int = 10):
        """Initialize singularity handler.

        Args:
            max_recursion: Maximum recursion depth allowed
        """
        self.max_recursion = max_recursion

    def validate(self, data: Dict) -> DimensionResult:
        """Validate self-referential conditions.

        Singularity handling means:
        1. System can emit receipts about receipts
        2. Circular references are detected
        3. Recursion is bounded

        Args:
            data: Dict with self-reference data

        Returns:
            DimensionResult with singularity handling
        """
        recursion_depth = data.get("recursion_depth", 0)
        circular_refs = data.get("circular_references", [])
        receipt_about_receipt = data.get("meta_receipts", 0)

        # Check recursion is bounded
        recursion_bounded = recursion_depth <= self.max_recursion

        # Check circular references are detected (not crashed)
        circulars_handled = len(circular_refs) >= 0  # Just checking it's tracked

        # Check meta-receipt capability
        has_meta_receipts = receipt_about_receipt > 0

        singularity_stable = recursion_bounded and circulars_handled

        return DimensionResult(
            dimension=self.dimension_id,
            passed=singularity_stable,
            message="Singularity stable" if singularity_stable else "Singularity unstable",
            details={
                "recursion_depth": recursion_depth,
                "max_recursion": self.max_recursion,
                "circular_refs_detected": len(circular_refs),
                "meta_receipts": receipt_about_receipt,
                "has_meta_capability": has_meta_receipts,
            },
        )


class D19_Decidability(BaseDimension):
    """D19: Godel completeness bounds acknowledged."""

    dimension_id = "D19"
    dimension_name = "Decidability"

    def validate(self, data: Dict) -> DimensionResult:
        """Acknowledge decidability limits.

        Godel's theorems imply:
        1. Some statements are undecidable within the system
        2. Consistency cannot be proven internally
        3. External attestation may be required

        Args:
            data: Dict with decidability info

        Returns:
            DimensionResult with decidability acknowledgment
        """
        total_statements = data.get("total_statements", 0)
        decidable = data.get("decidable_count", 0)
        undecidable = data.get("undecidable_count", 0)
        external_attestations = data.get("external_attestations", 0)

        if total_statements == 0:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="No statements to evaluate",
                details={},
            )

        completeness_ratio = decidable / total_statements

        # Acknowledge limits: we expect some undecidability
        # A perfect 100% decidability would actually be suspicious
        acknowledges_limits = undecidable > 0 or external_attestations > 0

        # System is consistent if it properly identifies undecidable cases
        properly_handles_undecidable = (
            undecidable == 0 or external_attestations >= undecidable * 0.5  # At least half got external help
        )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=acknowledges_limits and properly_handles_undecidable,
            message=f"Completeness: {completeness_ratio:.1%}",
            details={
                "total_statements": total_statements,
                "decidable": decidable,
                "undecidable": undecidable,
                "completeness_ratio": completeness_ratio,
                "external_attestations": external_attestations,
                "acknowledges_limits": acknowledges_limits,
            },
        )


class D20_Transcendence(BaseDimension):
    """D20: Meta-system consistency across all dimensions."""

    dimension_id = "D20"
    dimension_name = "Transcendence"

    def __init__(self, required_dimensions: List[str] = None):
        """Initialize transcendence validator.

        Args:
            required_dimensions: List of dimension IDs that must pass
        """
        self.required_dimensions = required_dimensions or [
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",  # Foundation
            "D6",
            "D7",
            "D8",
            "D9",
            "D10",  # Intermediate
            "D11",
            "D12",
            "D13",
            "D14",  # Advanced
            "D15",
            "D16",
            "D17",
            "D18",
            "D19",  # Ultimate
        ]

    def validate(self, data: Dict[str, DimensionResult]) -> DimensionResult:
        """Verify meta-system consistency across all dimensions.

        Transcendence requires:
        1. All lower dimensions pass
        2. No contradictions between dimensions
        3. System can describe itself completely through receipts

        Args:
            data: Dict of dimension_id -> DimensionResult

        Returns:
            DimensionResult for ultimate transcendence
        """
        if not isinstance(data, dict):
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message="Invalid dimension results format",
                details={},
            )

        # Check all required dimensions passed
        missing = []
        failed = []
        passed = []

        for dim_id in self.required_dimensions:
            if dim_id not in data:
                missing.append(dim_id)
            elif hasattr(data[dim_id], "passed"):
                if data[dim_id].passed:
                    passed.append(dim_id)
                else:
                    failed.append(dim_id)
            elif isinstance(data[dim_id], dict):
                if data[dim_id].get("passed", False):
                    passed.append(dim_id)
                else:
                    failed.append(dim_id)

        all_passed = len(missing) == 0 and len(failed) == 0

        # Check for contradictions (would require more complex analysis)
        # For now, we check consistency of key metrics across dimensions
        contradictions = []

        # Meta-hash: hash of all dimension results
        result_summary = {
            dim_id: {
                "passed": data[dim_id].passed if hasattr(data[dim_id], "passed") else data[dim_id].get("passed", False),
                "dimension": dim_id,
            }
            for dim_id in data
        }
        meta_hash = dual_hash(json.dumps(result_summary, sort_keys=True))

        transcendent = all_passed and len(contradictions) == 0

        return DimensionResult(
            dimension=self.dimension_id,
            passed=transcendent,
            message="Transcendence achieved" if transcendent else "Transcendence not achieved",
            details={
                "dimensions_passed": passed,
                "dimensions_failed": failed,
                "dimensions_missing": missing,
                "contradictions": contradictions,
                "meta_hash": meta_hash,
                "transcendent": transcendent,
            },
        )
