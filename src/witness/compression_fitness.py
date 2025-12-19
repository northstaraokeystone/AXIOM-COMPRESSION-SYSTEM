"""Compression fitness evaluation for discovered laws.

Uses MDL (Minimum Description Length) to score law quality.
"""

import math
from typing import Any, Dict, List


def compute_mdl_score(law: Dict, alpha: float = 1.0, beta: float = 0.10) -> float:
    """Compute MDL score for law.

    MDL = alpha * data_fit + beta * complexity
    Lower MDL = better law (more compression, less complexity)

    Args:
        law: Law dict
        alpha: Data fit weight (default 1.0)
        beta: Complexity weight (default 0.10)

    Returns:
        MDL score (lower is better)
    """
    # Data fit: inverse of compression ratio (higher compression = better fit)
    compression = law.get("compression_ratio", 0.5)
    data_fit = 1 - compression  # Lower is better

    # Complexity: based on spline coefficients
    coefficients = law.get("spline_coefficients", [])
    complexity = 0.0
    if coefficients:
        total_coeffs = sum(len(layer) for layer in coefficients if isinstance(layer, list))
        complexity = math.log2(total_coeffs + 1) / 10  # Normalized

    mdl = alpha * data_fit + beta * complexity

    return round(mdl, 6)


def compute_compression_fitness(law: Dict) -> float:
    """Compute overall compression fitness.

    Combines compression ratio, validation accuracy, and MDL.

    Args:
        law: Law dict

    Returns:
        Fitness score 0-1 (higher is better)
    """
    compression = law.get("compression_ratio", 0)
    validation = law.get("validation_accuracy", 0)
    mdl = compute_mdl_score(law)

    # Invert MDL (lower MDL = higher fitness)
    mdl_fitness = max(0, 1 - mdl)

    # Weighted combination
    fitness = 0.40 * compression + 0.35 * validation + 0.25 * mdl_fitness

    return round(fitness, 4)


def compute_spline_complexity(coefficients: List) -> float:
    """Compute complexity of spline coefficients.

    Args:
        coefficients: Nested list of spline coefficients

    Returns:
        Complexity score 0-1
    """
    if not coefficients:
        return 0.0

    # Count non-zero coefficients
    total = 0
    nonzero = 0

    def count_coeffs(items):
        nonlocal total, nonzero
        for item in items:
            if isinstance(item, list):
                count_coeffs(item)
            else:
                total += 1
                if abs(item) > 0.01:
                    nonzero += 1

    count_coeffs(coefficients)

    # Sparsity = ratio of nonzero coefficients
    sparsity = nonzero / total if total > 0 else 0

    # Complexity is inverse of sparsity
    complexity = 1 - sparsity

    return round(complexity, 4)


def rank_laws_by_fitness(laws: List[Dict]) -> List[Dict]:
    """Rank laws by compression fitness.

    Args:
        laws: List of law dicts

    Returns:
        Laws sorted by fitness (highest first)
    """
    ranked = []
    for law in laws:
        fitness = compute_compression_fitness(law)
        ranked.append({**law, "fitness_score": fitness})

    return sorted(ranked, key=lambda x: x["fitness_score"], reverse=True)
