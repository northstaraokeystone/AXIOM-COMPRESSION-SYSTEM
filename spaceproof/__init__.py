"""SpaceProof v5.0.0 - Space-grade proof infrastructure.

No receipt, not real.

Part of ProofChain: SpaceProof | SpendProof | ClaimProof | VoteProof | OriginProof | GreenProof

CORE MODULES:
- core.py: Hashing and receipts (dual_hash, emit_receipt, merkle, StopRule)
- compress.py: Telemetry compression (10x+, 0.999 recall)
- witness.py: KAN/MDL law discovery
- anchor.py: Merkle proofs
- detect.py: Entropy-based anomaly detection
- ledger.py: Append-only receipt storage
- sovereignty.py: Autonomy threshold calculator
- loop.py: 60-second SENSE->ACTUATE cycle

DOMAIN GENERATORS:
- domain/galaxy.py: Galaxy rotation curves
- domain/colony.py: Mars colony simulation
- domain/telemetry.py: Fleet telemetry (Tesla/Starlink/SpaceX)

DEFENSE EXPANSION DOMAINS (v5.0):
- domain/orbital_compute.py: Starcloud orbital compute provenance
- domain/constellation_ops.py: Starlink maneuver audit chains
- domain/autonomous_decision.py: Defense DOD 3000.09 compliance
- domain/firmware_integrity.py: Supply chain verification

INTEGRATION LAYERS:
- meta_integration.py: Meta-Loop v2.1 topology classification
- context_router.py: Confidence-gated fallback
- mcp_server.py: MCP protocol for Claude Desktop

STAKEHOLDER CONFIGS:
- config/xai.yaml: Elon/xAI
- config/doge.yaml: DOGE
- config/dot.yaml: DOT
- config/defense.yaml: Defense
- config/nro.yaml: NRO

Source: D20 Production Evolution + Defense Expansion v1.0 (Dec 2025)
"""

# Core primitives
from .core import dual_hash, emit_receipt, merkle, StopRule

# Core module classes and functions
from .compress import (
    CompressionConfig,
    CompressionResult,
    compress,
    decompress,
    calculate_recall,
    validate_compression_slo,
    compress_stream,
)

from .witness import (
    KANConfig,
    KAN,
    train,
    crossover_detection,
    emit_witness_receipt,
    validate_compression_threshold,
)

from .anchor import (
    Proof,
    AnchorResult,
    create_proof,
    verify_proof,
    anchor_batch,
    verify_batch,
    chain_receipts,
)

from .detect import (
    BaselineStats,
    DetectionResult,
    shannon_entropy,
    entropy_delta,
    detect_anomaly,
    classify_anomaly,
    build_baseline,
    detect_fraud_pattern,
    estimate_improper_payments,
)

from .ledger import (
    LedgerEntry,
    Ledger,
    create_ledger,
    append_to_ledger,
    verify_ledger,
    save_ledger,
    load_ledger,
)

from .sovereignty import (
    SovereigntyConfig,
    SovereigntyResult,
    internal_rate,
    external_rate,
    sovereignty_advantage,
    is_sovereign,
    compute_sovereignty,
    find_threshold,
    sensitivity_analysis,
)

from .loop import (
    Action,
    CycleResult,
    Loop,
    run_loop_once,
    run_loop_continuous,
)

# Domain generators
from . import domain

# Defense expansion integration layers
from . import meta_integration
from . import context_router
from . import mcp_server

__version__ = "5.0.0"
__series__ = "ProofChain"

__all__ = [
    # Identity
    "__version__",
    "__series__",
    # Core primitives
    "dual_hash",
    "emit_receipt",
    "merkle",
    "StopRule",
    # Compress
    "CompressionConfig",
    "CompressionResult",
    "compress",
    "decompress",
    "calculate_recall",
    "validate_compression_slo",
    "compress_stream",
    # Witness
    "KANConfig",
    "KAN",
    "train",
    "crossover_detection",
    "emit_witness_receipt",
    "validate_compression_threshold",
    # Anchor
    "Proof",
    "AnchorResult",
    "create_proof",
    "verify_proof",
    "anchor_batch",
    "verify_batch",
    "chain_receipts",
    # Detect
    "BaselineStats",
    "DetectionResult",
    "shannon_entropy",
    "entropy_delta",
    "detect_anomaly",
    "classify_anomaly",
    "build_baseline",
    "detect_fraud_pattern",
    "estimate_improper_payments",
    # Ledger
    "LedgerEntry",
    "Ledger",
    "create_ledger",
    "append_to_ledger",
    "verify_ledger",
    "save_ledger",
    "load_ledger",
    # Sovereignty
    "SovereigntyConfig",
    "SovereigntyResult",
    "internal_rate",
    "external_rate",
    "sovereignty_advantage",
    "is_sovereign",
    "compute_sovereignty",
    "find_threshold",
    "sensitivity_analysis",
    # Loop
    "Action",
    "CycleResult",
    "Loop",
    "run_loop_once",
    "run_loop_continuous",
    # Domain
    "domain",
    # Defense expansion integration layers
    "meta_integration",
    "context_router",
    "mcp_server",
]
