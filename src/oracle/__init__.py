"""D19.3 Live Causality Oracle Package.

PARADIGM INVERSION:
  OLD: "Laws woven preemptively from projected future entropy"
  NEW: "Laws oracled directly from live chain's emergent causality"

The chain history IS the oracle. Interstellar latency silence becomes
selection pressure—gaps force emergence of minimal-sync laws from prior
chain alone.

This package replaces:
  - src/projection/ (KILLED)
  - src/weave/ (KILLED)

Components:
  - LiveHistoryOracle: Extract laws from actual chain history compression
  - CausalSubgraphExtractor: Extract maximal causal subgraphs
  - InstantIncorporator: Real-time oracle update on receipt arrival
  - GapSilenceEmergence: Latency-driven minimal-sync law selection
"""

from .live_history_oracle import (
    LiveHistoryOracle,
    init_oracle,
    load_chain_history,
    compute_history_compression,
    extract_laws_from_history,
    oracle_query,
    emit_oracle_receipt,
    get_oracle_status,
)

from .causal_subgraph_extractor import (
    CausalSubgraphExtractor,
    init_extractor,
    build_causal_graph,
    find_maximal_subgraphs,
    subgraph_to_law,
    validate_causal_invariance,
    emit_subgraph_receipt,
)

from .instant_incorporator import (
    InstantIncorporator,
    init_incorporator,
    on_receipt_arrival,
    update_compression,
    update_causal_graph,
    check_law_survival,
    emit_incorporation_receipt,
)

from .gap_silence_emergence import (
    GapSilenceEmergence,
    init_gap_detector,
    detect_gap,
    classify_gap,
    trigger_minimal_law_selection,
    minimal_sync_law,
    emit_gap_emergence_receipt,
)

__all__ = [
    # LiveHistoryOracle
    "LiveHistoryOracle",
    "init_oracle",
    "load_chain_history",
    "compute_history_compression",
    "extract_laws_from_history",
    "oracle_query",
    "emit_oracle_receipt",
    "get_oracle_status",
    # CausalSubgraphExtractor
    "CausalSubgraphExtractor",
    "init_extractor",
    "build_causal_graph",
    "find_maximal_subgraphs",
    "subgraph_to_law",
    "validate_causal_invariance",
    "emit_subgraph_receipt",
    # InstantIncorporator
    "InstantIncorporator",
    "init_incorporator",
    "on_receipt_arrival",
    "update_compression",
    "update_causal_graph",
    "check_law_survival",
    "emit_incorporation_receipt",
    # GapSilenceEmergence
    "GapSilenceEmergence",
    "init_gap_detector",
    "detect_gap",
    "classify_gap",
    "trigger_minimal_law_selection",
    "minimal_sync_law",
    "emit_gap_emergence_receipt",
]
