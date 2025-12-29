"""Training pipeline - Transform interventions to training data.

Every human correction is training data, not just audit event.
Extracts, labels, scores quality, deduplicates, and exports examples.
"""

from .extractor import (
    extract_training_example,
    gather_input_context,
    format_bad_output,
    format_good_output,
    emit_training_example_receipt,
    emit_extraction_receipt,
    TrainingExample,
)

from .labeler import (
    apply_label,
    apply_labels,
    compute_label_distribution,
    get_labels_for_reason_code,
    get_label_schema,
    emit_labeling_receipt,
    LabeledExample,
)

from .exporter import (
    export_to_jsonl,
    export_to_huggingface,
    get_export_stats,
    emit_export_receipt,
    ExportResult,
)

from .quality import (
    score_example_quality,
    filter_by_quality,
    compute_quality_distribution,
    emit_quality_receipt,
    QualityScore,
)

from .dedup import (
    deduplicate_examples,
    compute_similarity,
    find_duplicates,
    emit_dedup_receipt,
    DedupResult,
)

from .feedback_loop import (
    add_to_retraining_queue,
    get_retraining_queue,
    prioritize_queue,
    process_retraining_batch,
    validate_learning,
    emit_feedback_receipt,
    FeedbackLoopState,
)

__all__ = [
    # Extractor
    "extract_training_example",
    "gather_input_context",
    "format_bad_output",
    "format_good_output",
    "emit_training_example_receipt",
    "emit_extraction_receipt",
    "TrainingExample",
    # Labeler
    "apply_label",
    "apply_labels",
    "compute_label_distribution",
    "get_labels_for_reason_code",
    "get_label_schema",
    "emit_labeling_receipt",
    "LabeledExample",
    # Exporter
    "export_to_jsonl",
    "export_to_huggingface",
    "get_export_stats",
    "emit_export_receipt",
    "ExportResult",
    # Quality
    "score_example_quality",
    "filter_by_quality",
    "compute_quality_distribution",
    "emit_quality_receipt",
    "QualityScore",
    # Dedup
    "deduplicate_examples",
    "compute_similarity",
    "find_duplicates",
    "emit_dedup_receipt",
    "DedupResult",
    # Feedback loop
    "add_to_retraining_queue",
    "get_retraining_queue",
    "prioritize_queue",
    "process_retraining_batch",
    "validate_learning",
    "emit_feedback_receipt",
    "FeedbackLoopState",
]
