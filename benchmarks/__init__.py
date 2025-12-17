"""benchmarks - Comparison benchmarks for AXIOM symbolic regression.

Modules:
    pysr_comparison: Compare AXIOM KAN to pySR symbolic regression
    symbolic_baselines: Other symbolic regression baselines
    report: Generate benchmark reports
"""

from .pysr_comparison import (
    run_pysr,
    run_axiom,
    compare,
    batch_compare,
    generate_table,
)

__all__ = [
    "run_pysr",
    "run_axiom",
    "compare",
    "batch_compare",
    "generate_table",
]
