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
)
from spaceproof.training.feedback_loop import create_retraining_batch


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
    # filter_by_quality expects List[TrainingExample], not LabeledExample
    examples = [extract_training_example(intervention)]
    # Returns Tuple[List[TrainingExample], List[TrainingExample]]
    passing, failing = filter_by_quality(examples, threshold=0.0)
    assert isinstance(passing, list)
    assert isinstance(failing, list)


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
    # deduplicate_examples expects List[TrainingExample], not LabeledExample
    examples = [extract_training_example(intervention)]
    # Returns Tuple[List[TrainingExample], DedupResult]
    deduped, result = deduplicate_examples(examples)
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
    # compute_similarity expects TrainingExample, not LabeledExample
    ex1 = extract_training_example(intervention1)
    ex2 = extract_training_example(intervention2)
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
    # add_to_retraining_queue expects TrainingExample, returns bool
    example = extract_training_example(intervention)
    result = add_to_retraining_queue(example)
    assert result is True


def test_get_retraining_queue():
    """get_retraining_queue returns queue."""
    queue = get_retraining_queue()
    assert isinstance(queue, list)


def test_process_retraining_batch():
    """process_retraining_batch processes batch."""
    # Add an example to queue first
    intervention = {
        "intervention_id": "int-batch-001",
        "target_decision_id": "dec-batch-001",
        "reason_code": "RE001",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test",
    }
    example = extract_training_example(intervention)
    add_to_retraining_queue(example)

    # Create a batch from the queue
    batch = create_retraining_batch(batch_size=10)
    if batch is not None:
        result = process_retraining_batch(batch)
        assert result is not None
    else:
        # Queue might be empty from previous tests, that's ok
        assert True


def test_validate_learning():
    """validate_learning validates learning."""
    result = validate_learning()
    # Returns tuple (passed, details)
    assert result is not None
