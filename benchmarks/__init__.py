"""benchmarks - Compression Benchmark Suite

Compare SpaceProof KAN witness against:
    - pySR (2024 SOTA symbolic regression)
    - AI Feynman
    - Eureqa (legacy)

Source: SpaceProof Validation Lock v1
"""

from .pysr_comparison import (
    run_pysr,
    run_spaceproof,
    compare,
    batch_compare,
    generate_table,
)
from .symbolic_baselines import (
    run_ai_feynman,
    run_eureqa_stub,
    compare_all_baselines,
)
from .report import (
    generate_benchmark_report,
    format_comparison_table,
    emit_benchmark_summary,
)

__all__ = [
    # pySR
    "run_pysr",
    "run_spaceproof",
    "compare",
    "batch_compare",
    "generate_table",
    # Baselines
    "run_ai_feynman",
    "run_eureqa_stub",
    "compare_all_baselines",
    # Report
    "generate_benchmark_report",
    "format_comparison_table",
    "emit_benchmark_summary",
]
