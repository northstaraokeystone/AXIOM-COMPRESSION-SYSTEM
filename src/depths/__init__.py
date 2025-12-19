"""D19 Swarm Intelligence depth module.

Exports D19 orchestration functions.
"""

from .d19_swarm_intelligence import (
    # Constants
    D19_DEPTH,
    D19_SCALE,
    D19_PARADIGM,
    D19_ALPHA_FLOOR,
    D19_ALPHA_TARGET,
    D19_ALPHA_CEILING,
    D19_UPLIFT,
    D19_INSTABILITY_MAX,
    # Functions
    load_d19_config,
    run_d19,
    run_gate_1,
    run_gate_2,
    run_gate_1_2_parallel,
    run_gate_3,
    run_gate_4,
    run_gate_5,
    evaluate_innovation,
    calculate_alpha,
    get_d19_status,
)

__all__ = [
    # Constants
    "D19_DEPTH",
    "D19_SCALE",
    "D19_PARADIGM",
    "D19_ALPHA_FLOOR",
    "D19_ALPHA_TARGET",
    "D19_ALPHA_CEILING",
    "D19_UPLIFT",
    "D19_INSTABILITY_MAX",
    # Functions
    "load_d19_config",
    "run_d19",
    "run_gate_1",
    "run_gate_2",
    "run_gate_1_2_parallel",
    "run_gate_3",
    "run_gate_4",
    "run_gate_5",
    "evaluate_innovation",
    "calculate_alpha",
    "get_d19_status",
]

RECEIPT_SCHEMA = {
    "module": "src.depths",
    "receipt_types": [
        "d19_config_receipt",
        "d19_gate_1",
        "d19_gate_2",
        "d19_gate_1_2",
        "d19_gate_3",
        "d19_gate_4",
        "d19_gate_5",
        "d19_complete_receipt",
    ],
    "version": "19.0.0",
}
