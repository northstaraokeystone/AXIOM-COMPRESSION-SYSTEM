"""Law witness package for D19 emergent governance.

This package implements KAN-based law discovery through compression.
Laws are witnessed, not programmed.
"""

from .kan_swarm import (
    SwarmKAN,
    init_swarm_kan,
    encode_swarm_state,
    train_on_coordination,
    compress_pattern,
    extract_law,
    validate_law,
    compare_laws,
    get_kan_status,
    KAN_ARCHITECTURE,
    MDL_ALPHA,
    MDL_BETA,
    COMPRESSION_TARGET,
)

from .law_discovery import (
    LawDiscovery,
    init_law_discovery,
    observe_swarm_cycle,
    detect_emerging_pattern,
    compress_pattern_to_law,
    validate_law_fitness,
    promote_law,
    demote_law,
    evolve_laws,
    get_active_laws,
    get_discovery_status,
    LAW_DISCOVERY_THRESHOLD,
    MAX_LAWS_PER_CYCLE,
)

from .governance_synthesis import (
    Protocol,
    synthesize_protocol,
    validate_protocol,
    deploy_protocol,
    monitor_protocol,
    retire_protocol,
    compare_to_hardcoded,
    get_active_protocols,
)

from .compression_fitness import (
    compute_compression_fitness,
    compute_mdl_score,
    compute_spline_complexity,
    rank_laws_by_fitness,
)

__all__ = [
    # KAN Swarm
    "SwarmKAN",
    "init_swarm_kan",
    "encode_swarm_state",
    "train_on_coordination",
    "compress_pattern",
    "extract_law",
    "validate_law",
    "compare_laws",
    "get_kan_status",
    "KAN_ARCHITECTURE",
    "MDL_ALPHA",
    "MDL_BETA",
    "COMPRESSION_TARGET",
    # Law Discovery
    "LawDiscovery",
    "init_law_discovery",
    "observe_swarm_cycle",
    "detect_emerging_pattern",
    "compress_pattern_to_law",
    "validate_law_fitness",
    "promote_law",
    "demote_law",
    "evolve_laws",
    "get_active_laws",
    "get_discovery_status",
    "LAW_DISCOVERY_THRESHOLD",
    "MAX_LAWS_PER_CYCLE",
    # Governance Synthesis
    "Protocol",
    "synthesize_protocol",
    "validate_protocol",
    "deploy_protocol",
    "monitor_protocol",
    "retire_protocol",
    "compare_to_hardcoded",
    "get_active_protocols",
    # Compression Fitness
    "compute_compression_fitness",
    "compute_mdl_score",
    "compute_spline_complexity",
    "rank_laws_by_fitness",
]

RECEIPT_SCHEMA = {
    "module": "src.witness",
    "receipt_types": [
        "swarm_encoding_receipt",
        "training_receipt",
        "law_extraction_receipt",
        "law_validation_receipt",
        "observation_receipt",
        "pattern_detection_receipt",
        "law_compression_receipt",
        "law_promotion_receipt",
        "law_demotion_receipt",
        "law_evolution_receipt",
        "synthesis_receipt",
        "deployment_receipt",
        "monitoring_receipt",
        "retirement_receipt",
    ],
    "version": "19.0.0",
}
