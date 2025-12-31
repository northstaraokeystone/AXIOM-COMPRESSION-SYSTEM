"""dedup.py - Deduplicate training examples.

Prevent duplicate or near-duplicate examples in training set.
Uses content hashing and similarity scoring for deduplication.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

from spaceproof.core import dual_hash

from .extractor import TrainingExample

# === CONSTANTS ===

# Similarity thresholds
EXACT_MATCH_THRESHOLD = 1.0
NEAR_DUPLICATE_THRESHOLD = 0.85


@dataclass
class DedupResult:
    """Result of deduplication operation."""

    original_count: int
    deduplicated_count: int
    duplicates_removed: int
    exact_duplicates: int
    near_duplicates: int
    duplicate_groups: List[List[str]]  # Groups of duplicate example IDs
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_count": self.original_count,
            "deduplicated_count": self.deduplicated_count,
            "duplicates_removed": self.duplicates_removed,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "duplicate_groups": self.duplicate_groups,
            "timestamp": self.timestamp,
        }


def compute_example_hash(example) -> str:
    """Compute hash of example content for exact duplicate detection.

    Args:
        example: TrainingExample or LabeledExample to hash

    Returns:
        Hash string
    """
    from .labeler import LabeledExample

    # Handle LabeledExample
    if isinstance(example, LabeledExample):
        content = {
            "example_id": example.example_id,
            "labels": example.labels,
            "category": example.category,
        }
        return dual_hash(json.dumps(content, sort_keys=True))

    # Handle TrainingExample
    content = {
        "input": example.input_context.to_dict() if hasattr(example, 'input_context') else {},
        "bad_output": example.bad_output.to_dict() if hasattr(example, 'bad_output') else {},
        "good_output": example.good_output.to_dict() if hasattr(example, 'good_output') else {},
        "label": example.label if hasattr(example, 'label') else {},
    }
    return dual_hash(json.dumps(content, sort_keys=True))


def compute_similarity(example1, example2) -> float:
    """Compute similarity score between two examples.

    Uses Jaccard similarity on tokenized content.
    Works with both TrainingExample and LabeledExample objects.

    Args:
        example1: First example (TrainingExample or LabeledExample)
        example2: Second example (TrainingExample or LabeledExample)

    Returns:
        Similarity score (0.0 - 1.0)
    """
    from .labeler import LabeledExample

    # Tokenize content
    def tokenize(example) -> Set[str]:
        tokens = set()

        # Handle LabeledExample (uses labels dict only)
        if isinstance(example, LabeledExample):
            label_str = json.dumps(example.labels, sort_keys=True)
            tokens.update(label_str.lower().split())
            tokens.add(example.category)
            tokens.update(example.subcategories)
            return tokens

        # Handle TrainingExample (full data)
        if hasattr(example, 'input_context') and example.input_context:
            input_str = json.dumps(example.input_context.to_dict(), sort_keys=True)
            tokens.update(input_str.lower().split())

        if hasattr(example, 'bad_output') and example.bad_output:
            bad_str = json.dumps(example.bad_output.to_dict(), sort_keys=True)
            tokens.update(bad_str.lower().split())

        if hasattr(example, 'good_output') and example.good_output:
            good_str = json.dumps(example.good_output.to_dict(), sort_keys=True)
            tokens.update(good_str.lower().split())

        # Add label tokens
        if hasattr(example, 'label') and example.label:
            label_str = json.dumps(example.label, sort_keys=True)
            tokens.update(label_str.lower().split())
        elif hasattr(example, 'labels') and example.labels:
            label_str = json.dumps(example.labels, sort_keys=True)
            tokens.update(label_str.lower().split())

        return tokens

    tokens1 = tokenize(example1)
    tokens2 = tokenize(example2)

    # Jaccard similarity
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def find_exact_duplicates(
    examples: List[TrainingExample],
) -> Tuple[Dict[str, List[TrainingExample]], int]:
    """Find exact duplicates based on content hash.

    Args:
        examples: List of examples to check

    Returns:
        Tuple of (hash_to_examples mapping, duplicate_count)
    """
    hash_map: Dict[str, List[TrainingExample]] = {}

    for example in examples:
        content_hash = compute_example_hash(example)
        if content_hash not in hash_map:
            hash_map[content_hash] = []
        hash_map[content_hash].append(example)

    duplicate_count = sum(len(v) - 1 for v in hash_map.values() if len(v) > 1)

    return hash_map, duplicate_count


def find_near_duplicates(
    examples: List[TrainingExample],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> List[List[TrainingExample]]:
    """Find near-duplicate groups based on similarity.

    Args:
        examples: List of examples to check
        threshold: Similarity threshold

    Returns:
        List of duplicate groups
    """
    duplicate_groups: List[List[TrainingExample]] = []
    processed: Set[str] = set()

    for i, example1 in enumerate(examples):
        if example1.example_id in processed:
            continue

        group = [example1]
        processed.add(example1.example_id)

        for j, example2 in enumerate(examples[i + 1 :], start=i + 1):
            if example2.example_id in processed:
                continue

            similarity = compute_similarity(example1, example2)
            if similarity >= threshold:
                group.append(example2)
                processed.add(example2.example_id)

        if len(group) > 1:
            duplicate_groups.append(group)

    return duplicate_groups


def find_duplicates(
    examples: List[TrainingExample],
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> Dict[str, Any]:
    """Find all duplicates (exact and near).

    Args:
        examples: List of examples to check
        near_duplicate_threshold: Threshold for near duplicates

    Returns:
        Dict with duplicate info
    """
    # Find exact duplicates
    hash_map, exact_count = find_exact_duplicates(examples)

    # Find near duplicates (only among non-exact duplicates)
    unique_examples = [v[0] for v in hash_map.values()]
    near_groups = find_near_duplicates(unique_examples, near_duplicate_threshold)

    near_count = sum(len(g) - 1 for g in near_groups)

    return {
        "exact_duplicates": exact_count,
        "near_duplicates": near_count,
        "total_duplicates": exact_count + near_count,
        "exact_groups": [[e.example_id for e in v] for v in hash_map.values() if len(v) > 1],
        "near_groups": [[e.example_id for e in g] for g in near_groups],
    }


def deduplicate_examples(
    examples,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    keep_strategy: str = "first",  # "first", "highest_quality", "latest"
) -> DedupResult:
    """Deduplicate training examples.

    Args:
        examples: List of examples to deduplicate (TrainingExample or LabeledExample)
        near_duplicate_threshold: Threshold for near duplicates
        keep_strategy: Strategy for choosing which duplicate to keep

    Returns:
        DedupResult with deduplication statistics
    """
    from .labeler import LabeledExample

    original_count = len(examples)

    if original_count == 0:
        return DedupResult(
            original_count=0,
            deduplicated_count=0,
            duplicates_removed=0,
            exact_duplicates=0,
            near_duplicates=0,
            duplicate_groups=[],
        )

    # Helper to get quality/timestamp with fallbacks
    def get_quality(e):
        if hasattr(e, 'quality_score'):
            return e.quality_score
        if hasattr(e, 'label_confidence'):
            return e.label_confidence
        return 0.5

    def get_timestamp(e):
        if hasattr(e, 'timestamp'):
            return e.timestamp
        if hasattr(e, 'labeled_at'):
            return e.labeled_at
        return ""

    # First pass: remove exact duplicates
    hash_map, exact_count = find_exact_duplicates(examples)

    # Keep one from each exact duplicate group
    unique_from_exact = []
    for group in hash_map.values():
        if keep_strategy == "first":
            unique_from_exact.append(group[0])
        elif keep_strategy == "highest_quality":
            unique_from_exact.append(max(group, key=get_quality))
        elif keep_strategy == "latest":
            unique_from_exact.append(max(group, key=get_timestamp))
        else:
            unique_from_exact.append(group[0])

    # Second pass: remove near duplicates
    near_groups = find_near_duplicates(unique_from_exact, near_duplicate_threshold)
    near_duplicate_ids: Set[str] = set()

    for group in near_groups:
        # Keep one from group, mark rest as duplicates
        if keep_strategy == "first":
            keeper = group[0]
        elif keep_strategy == "highest_quality":
            keeper = max(group, key=get_quality)
        elif keep_strategy == "latest":
            keeper = max(group, key=get_timestamp)
        else:
            keeper = group[0]

        for example in group:
            if example.example_id != keeper.example_id:
                near_duplicate_ids.add(example.example_id)

    near_count = len(near_duplicate_ids)

    # Final deduplicated list
    deduplicated = [e for e in unique_from_exact if e.example_id not in near_duplicate_ids]

    # Build duplicate groups for result
    all_groups = [[e.example_id for e in v] for v in hash_map.values() if len(v) > 1]
    all_groups.extend([[e.example_id for e in g] for g in near_groups])

    result = DedupResult(
        original_count=original_count,
        deduplicated_count=len(deduplicated),
        duplicates_removed=original_count - len(deduplicated),
        exact_duplicates=exact_count,
        near_duplicates=near_count,
        duplicate_groups=all_groups,
    )

    return result
