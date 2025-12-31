"""Tests for spaceproof.training module."""

from spaceproof.training import (
    extract_training_example,
    gather_input_context,
    format_bad_output,
    format_good_output,
    TrainingExample,
    apply_label,
    get_labels_for_reason_code,
    LabeledExample,
    export_to_jsonl,
    get_export_stats,
    ExportResult,
    score_example_quality,
    filter_by_quality,
    QualityScore,
    deduplicate_examples,
    compute_similarity,
    DedupResult,
    add_to_retraining_queue,
    get_retraining_queue,
    process_retraining_batch,
    validate_learning,
    FeedbackLoopState,
)


def test_extract_training_example():
    """extract_training_example transforms intervention to example."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test justification",
    }
    example = extract_training_example(intervention)
    assert isinstance(example, TrainingExample)


def test_gather_input_context():
    """gather_input_context gathers context."""
    decision_id = "dec-001"
    context = gather_input_context(decision_id)
    assert context is not None


def test_format_bad_output():
    """format_bad_output formats original output."""
    output = {"type": "wrong", "value": 1}
    formatted = format_bad_output(output)
    assert formatted is not None


def test_format_good_output():
    """format_good_output formats corrected output."""
    output = {"type": "correct", "value": 2}
    formatted = format_good_output(output)
    assert formatted is not None


def test_apply_label():
    """apply_label adds label to example."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    example = extract_training_example(intervention)
    # Correct signature: apply_label(example, additional_labels=None)
    labeled = apply_label(example)
    assert isinstance(labeled, LabeledExample)


def test_get_labels_for_reason_code():
    """get_labels_for_reason_code returns labels."""
    labels = get_labels_for_reason_code("RE001")
    # Returns dict of label properties
    assert labels is not None


def test_export_to_jsonl():
    """export_to_jsonl exports examples to JSONL format."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    examples = [extract_training_example(intervention)]
    result = export_to_jsonl(examples)
    assert isinstance(result, ExportResult)


def test_get_export_stats():
    """get_export_stats returns stats."""
    stats = get_export_stats()
    assert isinstance(stats, dict)


def test_score_example_quality():
    """score_example_quality returns score."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    example = extract_training_example(intervention)
    score = score_example_quality(example)
    assert isinstance(score, QualityScore)


def test_filter_by_quality():
    """filter_by_quality removes low-quality examples."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    examples = [extract_training_example(intervention)]
    labeled = [apply_label(e) for e in examples]
    filtered = filter_by_quality(labeled, threshold=0.0)
    assert isinstance(filtered, list)


def test_deduplicate_examples():
    """deduplicate_examples removes duplicates."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    examples = [extract_training_example(intervention)]
    labeled = [apply_label(e) for e in examples]
    result = deduplicate_examples(labeled)
    assert isinstance(result, DedupResult)


def test_compute_similarity():
    """compute_similarity returns similarity score."""
    intervention1 = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    intervention2 = {
        "intervention_id": "int-002",
        "target_decision_id": "dec-002",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    ex1 = apply_label(extract_training_example(intervention1))
    ex2 = apply_label(extract_training_example(intervention2))
    similarity = compute_similarity(ex1, ex2)
    assert isinstance(similarity, float)


def test_add_to_retraining_queue():
    """add_to_retraining_queue adds example to queue."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    labeled = apply_label(extract_training_example(intervention))
    state = add_to_retraining_queue(labeled)
    assert isinstance(state, FeedbackLoopState)


def test_get_retraining_queue():
    """get_retraining_queue returns queue."""
    queue = get_retraining_queue()
    assert isinstance(queue, list)


def test_process_retraining_batch():
    """process_retraining_batch processes batch."""
    # Signature requires a batch of examples
    queue = get_retraining_queue()
    result = process_retraining_batch(queue)
    assert result is not None


def test_validate_learning():
    """validate_learning validates learning."""
    result = validate_learning()
    # Returns tuple (passed, details)
    assert result is not None
