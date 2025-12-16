"""AXIOM Compression System - Modular imports."""

# Core functions (always available)
from .core import (
    dual_hash,
    emit_receipt,
    merkle,
    StopRule,
    stoprule_hash_mismatch,
    stoprule_invalid_receipt,
    TENANT_ID,
    RECEIPT_SCHEMA,
    HAS_BLAKE3,
)

__all__ = [
    "dual_hash",
    "emit_receipt",
    "merkle",
    "StopRule",
    "stoprule_hash_mismatch",
    "stoprule_invalid_receipt",
    "TENANT_ID",
    "RECEIPT_SCHEMA",
    "HAS_BLAKE3",
]

# Optional: KAN module (requires torch)
try:
    from .kan_core import (
        DEFAULT_TOPOLOGY,
        SPLINE_DEGREE,
        SPLINE_KNOTS,
        PARAM_BUDGET,
        MDL_ALPHA,
        MDL_BETA,
        COMPLEXITY_THRESHOLD,
        GRAD_CLIP_NORM,
        RECEIPT_SCHEMAS,
        kan_init,
        spline_edge,
        forward_compress,
        complexity,
        mdl_loss,
        train_step,
        extract_equation,
        detect_dark_matter,
        persistence_match,
        checkpoint_save,
    )
    __all__.extend([
        "DEFAULT_TOPOLOGY", "SPLINE_DEGREE", "SPLINE_KNOTS", "PARAM_BUDGET",
        "MDL_ALPHA", "MDL_BETA", "COMPLEXITY_THRESHOLD", "GRAD_CLIP_NORM",
        "RECEIPT_SCHEMAS", "kan_init", "spline_edge", "forward_compress",
        "complexity", "mdl_loss", "train_step", "extract_equation",
        "detect_dark_matter", "persistence_match", "checkpoint_save",
    ])
except ImportError:
    pass  # torch not available, kan_core functions unavailable

# Optional: Topology module (requires ripser)
try:
    from .topology import (
        compute_persistence,
        classify_features,
        topology_loss_term,
        h1_interpretation,
        h2_interpretation,
        trivial_topology_check,
        topology_receipt_emit,
    )
    __all__.extend([
        "compute_persistence", "classify_features", "topology_loss_term",
        "h1_interpretation", "h2_interpretation", "trivial_topology_check",
        "topology_receipt_emit",
    ])
except ImportError:
    pass  # ripser not available, topology functions unavailable
