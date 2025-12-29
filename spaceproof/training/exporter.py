"""exporter.py - Export training examples to various formats.

Export to JSONL, HuggingFace datasets, and other ML platforms.
Maintains receipts for export operations and data lineage.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

from .extractor import TrainingExample
from .labeler import LabeledExample

# === CONSTANTS ===

TRAINING_TENANT = "spaceproof-training"
DEFAULT_EXPORT_DIR = Path(__file__).parent.parent.parent / "training_data"


@dataclass
class ExportResult:
    """Result of export operation."""

    export_id: str
    format: str
    path: str
    example_count: int
    total_bytes: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "export_id": self.export_id,
            "format": self.format,
            "path": self.path,
            "example_count": self.example_count,
            "total_bytes": self.total_bytes,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }


# Export statistics
_export_stats: Dict[str, Any] = {
    "total_exports": 0,
    "total_examples_exported": 0,
    "total_bytes": 0,
    "formats_used": {},
}


def format_example_for_jsonl(
    example: TrainingExample,
    labeled: Optional[LabeledExample] = None,
) -> Dict[str, Any]:
    """Format training example for JSONL export.

    Args:
        example: TrainingExample to format
        labeled: Optional LabeledExample with labels

    Returns:
        Dict ready for JSON serialization
    """
    output = {
        "id": example.example_id,
        "input": example.input_context.to_dict(),
        "bad_output": example.bad_output.to_dict(),
        "good_output": example.good_output.to_dict(),
        "metadata": {
            "source_decision_id": example.source_decision_id,
            "source_intervention_id": example.source_intervention_id,
            "timestamp": example.timestamp,
            "quality_score": example.quality_score,
            "retraining_priority": example.retraining_priority,
        },
    }

    if labeled:
        output["labels"] = labeled.labels
        output["metadata"]["label_confidence"] = labeled.label_confidence

    return output


def export_to_jsonl(
    examples: List[TrainingExample],
    output_path: Optional[Path] = None,
    labeled_examples: Optional[Dict[str, LabeledExample]] = None,
) -> ExportResult:
    """Export training examples to JSONL format.

    Args:
        examples: List of TrainingExample objects
        output_path: Output file path (optional)
        labeled_examples: Optional mapping of example_id to LabeledExample

    Returns:
        ExportResult with status
    """
    import uuid

    export_id = str(uuid.uuid4())

    # Default path
    if output_path is None:
        DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_EXPORT_DIR / f"training_{export_id[:8]}.jsonl"

    try:
        total_bytes = 0

        with open(output_path, "w") as f:
            for example in examples:
                labeled = labeled_examples.get(example.example_id) if labeled_examples else None
                formatted = format_example_for_jsonl(example, labeled)
                line = json.dumps(formatted, sort_keys=True) + "\n"
                f.write(line)
                total_bytes += len(line.encode("utf-8"))

                # Update example's exported_to field
                example.exported_to = str(output_path)

        # Update stats
        _export_stats["total_exports"] += 1
        _export_stats["total_examples_exported"] += len(examples)
        _export_stats["total_bytes"] += total_bytes
        _export_stats["formats_used"]["jsonl"] = _export_stats["formats_used"].get("jsonl", 0) + 1

        result = ExportResult(
            export_id=export_id,
            format="jsonl",
            path=str(output_path),
            example_count=len(examples),
            total_bytes=total_bytes,
        )

        # Emit receipt
        emit_receipt(
            "training_export",
            {
                "tenant_id": TRAINING_TENANT,
                **result.to_dict(),
            },
        )

        return result

    except Exception as e:
        return ExportResult(
            export_id=export_id,
            format="jsonl",
            path=str(output_path),
            example_count=0,
            total_bytes=0,
            success=False,
            error=str(e),
        )


def format_for_huggingface(
    example: TrainingExample,
    labeled: Optional[LabeledExample] = None,
) -> Dict[str, Any]:
    """Format training example for HuggingFace datasets.

    Args:
        example: TrainingExample to format
        labeled: Optional LabeledExample with labels

    Returns:
        Dict in HuggingFace format
    """
    # HuggingFace format: prompt/completion pairs
    prompt = json.dumps(example.input_context.to_dict())
    bad_completion = json.dumps(example.bad_output.to_dict())
    good_completion = json.dumps(example.good_output.to_dict())

    output = {
        "id": example.example_id,
        "prompt": prompt,
        "rejected": bad_completion,
        "chosen": good_completion,
        "quality_score": example.quality_score,
    }

    if labeled:
        output["category"] = labeled.category
        output["severity_weight"] = labeled.severity_weight

    return output


def export_to_huggingface(
    examples: List[TrainingExample],
    output_path: Optional[Path] = None,
    labeled_examples: Optional[Dict[str, LabeledExample]] = None,
) -> ExportResult:
    """Export training examples to HuggingFace dataset format.

    Args:
        examples: List of TrainingExample objects
        output_path: Output file path (optional)
        labeled_examples: Optional mapping of example_id to LabeledExample

    Returns:
        ExportResult with status
    """
    import uuid

    export_id = str(uuid.uuid4())

    # Default path
    if output_path is None:
        DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_EXPORT_DIR / f"hf_dataset_{export_id[:8]}.jsonl"

    try:
        total_bytes = 0

        with open(output_path, "w") as f:
            for example in examples:
                labeled = labeled_examples.get(example.example_id) if labeled_examples else None
                formatted = format_for_huggingface(example, labeled)
                line = json.dumps(formatted, sort_keys=True) + "\n"
                f.write(line)
                total_bytes += len(line.encode("utf-8"))

        # Update stats
        _export_stats["total_exports"] += 1
        _export_stats["total_examples_exported"] += len(examples)
        _export_stats["total_bytes"] += total_bytes
        _export_stats["formats_used"]["huggingface"] = _export_stats["formats_used"].get("huggingface", 0) + 1

        result = ExportResult(
            export_id=export_id,
            format="huggingface",
            path=str(output_path),
            example_count=len(examples),
            total_bytes=total_bytes,
        )

        return result

    except Exception as e:
        return ExportResult(
            export_id=export_id,
            format="huggingface",
            path=str(output_path),
            example_count=0,
            total_bytes=0,
            success=False,
            error=str(e),
        )


def get_export_stats() -> Dict[str, Any]:
    """Get export statistics.

    Returns:
        Dict with export stats
    """
    return _export_stats.copy()


def clear_export_stats() -> None:
    """Clear export statistics (for testing)."""
    global _export_stats
    _export_stats = {
        "total_exports": 0,
        "total_examples_exported": 0,
        "total_bytes": 0,
        "formats_used": {},
    }


def emit_export_receipt(result: ExportResult) -> Dict[str, Any]:
    """Emit receipt for export operation.

    Args:
        result: The export result

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "training_export",
        {
            "tenant_id": TRAINING_TENANT,
            **result.to_dict(),
        },
    )
