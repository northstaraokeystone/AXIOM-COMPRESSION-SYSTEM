"""Autocatalytic pattern package for D19 emergent swarm behavior.

Patterns that sustain themselves through self-reference.
A pattern is "alive" when it emits receipts that reference itself.
Birth = crosses autocatalysis threshold. Death = loses coherence.
"""

from .pattern_detector import (
    PatternDetector,
    init_detector,
    scan_receipt_stream,
    measure_self_reference,
    detect_autocatalysis,
    track_pattern_coherence,
    predict_pattern_fate,
    get_detector_status,
    SELF_REFERENCE_THRESHOLD,
)

from .pattern_lifecycle import (
    PatternLifecycle,
    init_lifecycle,
    birth_pattern,
    evaluate_fitness,
    apply_selection_pressure,
    kill_pattern,
    resurrect_pattern,
    get_active_patterns,
    get_superposition_patterns,
    PATTERN_BIRTH_FITNESS,
    PATTERN_DEATH_FITNESS,
)

from .cross_planet_migration import (
    MigrationManager,
    init_migration,
    identify_migration_candidates,
    prepare_pattern_transfer,
    execute_transfer,
    validate_migration,
    adapt_to_destination,
    measure_migration_fitness,
    get_migration_status,
    MIGRATION_LATENCY_TOLERANCE_MS,
)

from .fitness_evaluator import (
    compute_pattern_fitness,
    compute_multi_dimensional_fitness,
    thompson_sampling_select,
    compute_diversity_contribution,
    compute_recency_bonus,
)

__all__ = [
    # Pattern Detector
    "PatternDetector",
    "init_detector",
    "scan_receipt_stream",
    "measure_self_reference",
    "detect_autocatalysis",
    "track_pattern_coherence",
    "predict_pattern_fate",
    "get_detector_status",
    "SELF_REFERENCE_THRESHOLD",
    # Pattern Lifecycle
    "PatternLifecycle",
    "init_lifecycle",
    "birth_pattern",
    "evaluate_fitness",
    "apply_selection_pressure",
    "kill_pattern",
    "resurrect_pattern",
    "get_active_patterns",
    "get_superposition_patterns",
    "PATTERN_BIRTH_FITNESS",
    "PATTERN_DEATH_FITNESS",
    # Cross Planet Migration
    "MigrationManager",
    "init_migration",
    "identify_migration_candidates",
    "prepare_pattern_transfer",
    "execute_transfer",
    "validate_migration",
    "adapt_to_destination",
    "measure_migration_fitness",
    "get_migration_status",
    "MIGRATION_LATENCY_TOLERANCE_MS",
    # Fitness Evaluator
    "compute_pattern_fitness",
    "compute_multi_dimensional_fitness",
    "thompson_sampling_select",
    "compute_diversity_contribution",
    "compute_recency_bonus",
]

RECEIPT_SCHEMA = {
    "module": "src.autocatalytic",
    "receipt_types": [
        "pattern_scan_receipt",
        "autocatalysis_detection_receipt",
        "coherence_measurement_receipt",
        "fate_prediction_receipt",
        "pattern_birth_receipt",
        "fitness_evaluation_receipt",
        "selection_pressure_receipt",
        "pattern_death_receipt",
        "pattern_resurrection_receipt",
        "migration_candidate_receipt",
        "transfer_preparation_receipt",
        "transfer_execution_receipt",
        "migration_validation_receipt",
        "adaptation_receipt",
    ],
    "version": "19.0.0",
}
