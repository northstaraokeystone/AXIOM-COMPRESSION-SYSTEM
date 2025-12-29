"""context_router.py - Context Engineering for Confidence-Gated Fallback.

D20 Production Evolution: Confidence-gated fallback for context enrichment.

THE CONTEXT ROUTER PARADIGM:
    Primary source: Receipt ledger
    Fallback: External validation if confidence < threshold

    Query Types and Sources:
    - Historical fact: Receipt ledger (0.95 threshold)
    - Pattern match: META-LOOP topology (0.85 threshold)
    - External validation: Web search (0.70 threshold)
    - Cross-domain: Temporal graph (0.80 threshold)

Source: Receipts_native_architecture_v2_0.txt (document #12)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json

from spaceproof.core import dual_hash, emit_receipt, merkle, TENANT_ID

# === CONSTANTS ===

CONTEXT_ROUTER_TENANT = "spaceproof-context-router"

# Confidence thresholds per query type
CONFIDENCE_THRESHOLDS = {
    "historical_fact": 0.95,
    "pattern_match": 0.85,
    "external_validation": 0.70,
    "cross_domain": 0.80,
}

# Default fallback threshold
DEFAULT_CONFIDENCE_FALLBACK = 0.85


class QueryType(Enum):
    """Types of context queries."""

    HISTORICAL_FACT = "historical_fact"
    PATTERN_MATCH = "pattern_match"
    EXTERNAL_VALIDATION = "external_validation"
    CROSS_DOMAIN = "cross_domain"


class ContextSource(Enum):
    """Context data sources."""

    RECEIPT_LEDGER = "receipt_ledger"
    META_LOOP_TOPOLOGY = "meta_loop_topology"
    TEMPORAL_GRAPH = "temporal_graph"
    WEB_SEARCH = "web_search"
    NONE = "none"


@dataclass
class ContextQuery:
    """A context query with metadata."""

    query_id: str
    query_type: QueryType
    query_text: str
    domain: str
    timestamp: str


@dataclass
class ContextResult:
    """Result from a context source."""

    source: ContextSource
    data: Dict[str, Any]
    confidence: float
    latency_ms: float


@dataclass
class RoutingDecision:
    """Context routing decision."""

    query_id: str
    query_type: QueryType
    primary_source: ContextSource
    fallback_source: Optional[ContextSource]
    confidence: float
    fallback_triggered: bool
    merged_result: bool
    receipt: Dict[str, Any]


def get_primary_source(query_type: QueryType) -> ContextSource:
    """Select primary context source for query type.

    Args:
        query_type: Type of query

    Returns:
        Primary ContextSource
    """
    source_map = {
        QueryType.HISTORICAL_FACT: ContextSource.RECEIPT_LEDGER,
        QueryType.PATTERN_MATCH: ContextSource.META_LOOP_TOPOLOGY,
        QueryType.EXTERNAL_VALIDATION: ContextSource.WEB_SEARCH,
        QueryType.CROSS_DOMAIN: ContextSource.TEMPORAL_GRAPH,
    }
    return source_map.get(query_type, ContextSource.RECEIPT_LEDGER)


def get_fallback_source(query_type: QueryType) -> Optional[ContextSource]:
    """Get fallback source for query type.

    Args:
        query_type: Type of query

    Returns:
        Fallback ContextSource or None
    """
    fallback_map = {
        QueryType.HISTORICAL_FACT: None,  # No fallback for historical facts
        QueryType.PATTERN_MATCH: ContextSource.RECEIPT_LEDGER,
        QueryType.EXTERNAL_VALIDATION: None,  # Fail gracefully
        QueryType.CROSS_DOMAIN: ContextSource.META_LOOP_TOPOLOGY,
    }
    return fallback_map.get(query_type)


def get_confidence_threshold(query_type: QueryType) -> float:
    """Get confidence threshold for query type.

    Args:
        query_type: Type of query

    Returns:
        Confidence threshold
    """
    return CONFIDENCE_THRESHOLDS.get(query_type.value, DEFAULT_CONFIDENCE_FALLBACK)


def compute_confidence(pattern: Dict[str, Any], source: ContextSource) -> float:
    """Calculate confidence score for pattern from source.

    Args:
        pattern: Pattern data
        source: Data source

    Returns:
        Confidence score (0.0 - 1.0)
    """
    # Base confidence by source reliability
    source_confidence = {
        ContextSource.RECEIPT_LEDGER: 0.95,
        ContextSource.META_LOOP_TOPOLOGY: 0.85,
        ContextSource.TEMPORAL_GRAPH: 0.80,
        ContextSource.WEB_SEARCH: 0.60,
        ContextSource.NONE: 0.0,
    }

    base = source_confidence.get(source, 0.5)

    # Adjust based on data quality
    data_quality = pattern.get("data_quality", 0.8)
    n_samples = pattern.get("n_samples", 0)

    # Sample size adjustment
    sample_factor = min(1.0, n_samples / 100) if n_samples > 0 else 0.5

    confidence = base * data_quality * (0.5 + 0.5 * sample_factor)

    return min(1.0, max(0.0, confidence))


def select_primary_source(query_type: str) -> str:
    """Choose primary context source for query.

    Args:
        query_type: Query type string

    Returns:
        Primary source name
    """
    try:
        qt = QueryType(query_type)
        source = get_primary_source(qt)
        return source.value
    except ValueError:
        return ContextSource.RECEIPT_LEDGER.value


def query_receipt_ledger(query: str, domain: str = "all") -> ContextResult:
    """Query the receipt ledger.

    Args:
        query: Query string
        domain: Domain filter

    Returns:
        ContextResult from ledger
    """
    import time
    start = time.time()

    # Simulate ledger query (in production, would query actual ledger)
    data = {
        "query": query,
        "domain": domain,
        "results": [],
        "total_matches": 0,
    }

    latency = (time.time() - start) * 1000

    return ContextResult(
        source=ContextSource.RECEIPT_LEDGER,
        data=data,
        confidence=0.95,
        latency_ms=latency,
    )


def query_meta_loop(query: str, domain: str = "all") -> ContextResult:
    """Query Meta-Loop topology.

    Args:
        query: Query string
        domain: Domain filter

    Returns:
        ContextResult from Meta-Loop
    """
    import time
    start = time.time()

    data = {
        "query": query,
        "domain": domain,
        "topology_matches": [],
        "patterns_found": 0,
    }

    latency = (time.time() - start) * 1000

    return ContextResult(
        source=ContextSource.META_LOOP_TOPOLOGY,
        data=data,
        confidence=0.85,
        latency_ms=latency,
    )


def query_temporal_graph(query: str, domain: str = "all") -> ContextResult:
    """Query temporal graph.

    Args:
        query: Query string
        domain: Domain filter

    Returns:
        ContextResult from temporal graph
    """
    import time
    start = time.time()

    data = {
        "query": query,
        "domain": domain,
        "graph_matches": [],
        "similarity_scores": [],
    }

    latency = (time.time() - start) * 1000

    return ContextResult(
        source=ContextSource.TEMPORAL_GRAPH,
        data=data,
        confidence=0.80,
        latency_ms=latency,
    )


def fallback_to_web(pattern: Dict[str, Any], query: str) -> ContextResult:
    """Trigger web search fallback if confidence < threshold.

    Args:
        pattern: Pattern needing enrichment
        query: Search query

    Returns:
        ContextResult from web search
    """
    import time
    start = time.time()

    # Simulate web search (in production, would use actual web search)
    data = {
        "query": query,
        "pattern_id": pattern.get("pattern_id", "unknown"),
        "search_results": [],
        "enrichment_data": {},
    }

    latency = (time.time() - start) * 1000

    emit_receipt(
        "web_fallback",
        {
            "tenant_id": CONTEXT_ROUTER_TENANT,
            "query": query,
            "pattern_id": pattern.get("pattern_id", "unknown"),
            "fallback_triggered": True,
        },
    )

    return ContextResult(
        source=ContextSource.WEB_SEARCH,
        data=data,
        confidence=0.60,
        latency_ms=latency,
    )


def merge_context(primary: ContextResult, fallback: ContextResult) -> Dict[str, Any]:
    """Merge primary + fallback results preserving provenance.

    Args:
        primary: Primary source result
        fallback: Fallback source result

    Returns:
        Merged context with provenance
    """
    merged = {
        "primary": {
            "source": primary.source.value,
            "data": primary.data,
            "confidence": primary.confidence,
        },
        "fallback": {
            "source": fallback.source.value,
            "data": fallback.data,
            "confidence": fallback.confidence,
        },
        "merged_confidence": (primary.confidence + fallback.confidence) / 2,
        "provenance": {
            "primary_source": primary.source.value,
            "fallback_source": fallback.source.value,
            "merge_timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }

    return merged


def emit_context_receipt(
    source: str,
    confidence: float,
    fallback_triggered: bool,
    query_type: str = "pattern_match",
    merged_result: bool = False,
) -> Dict[str, Any]:
    """Document context routing decision.

    Args:
        source: Primary source used
        confidence: Confidence score
        fallback_triggered: Whether fallback was used
        query_type: Type of query
        merged_result: Whether results were merged

    Returns:
        Context routing receipt
    """
    receipt = emit_receipt(
        "context_routing",
        {
            "tenant_id": CONTEXT_ROUTER_TENANT,
            "query_type": query_type,
            "primary_source": source,
            "confidence": confidence,
            "fallback_triggered": fallback_triggered,
            "merged_result": merged_result,
            "threshold": CONFIDENCE_THRESHOLDS.get(query_type, DEFAULT_CONFIDENCE_FALLBACK),
        },
    )

    return receipt


def route_query(
    query: ContextQuery,
    pattern: Optional[Dict[str, Any]] = None,
) -> RoutingDecision:
    """Route context query to appropriate source.

    Args:
        query: Context query
        pattern: Optional pattern for confidence calculation

    Returns:
        RoutingDecision with routing details
    """
    primary_source = get_primary_source(query.query_type)
    fallback_source = get_fallback_source(query.query_type)
    threshold = get_confidence_threshold(query.query_type)

    # Query primary source
    if primary_source == ContextSource.RECEIPT_LEDGER:
        primary_result = query_receipt_ledger(query.query_text, query.domain)
    elif primary_source == ContextSource.META_LOOP_TOPOLOGY:
        primary_result = query_meta_loop(query.query_text, query.domain)
    elif primary_source == ContextSource.TEMPORAL_GRAPH:
        primary_result = query_temporal_graph(query.query_text, query.domain)
    else:
        primary_result = ContextResult(
            source=primary_source,
            data={},
            confidence=0.5,
            latency_ms=0.0,
        )

    confidence = primary_result.confidence
    fallback_triggered = False
    merged_result = False

    # Check if fallback needed
    if confidence < threshold and fallback_source is not None:
        fallback_triggered = True
        if pattern is not None:
            fallback_result = fallback_to_web(pattern, query.query_text)
            merge_context(primary_result, fallback_result)
            merged_result = True
            confidence = (primary_result.confidence + fallback_result.confidence) / 2

    receipt = emit_context_receipt(
        source=primary_source.value,
        confidence=confidence,
        fallback_triggered=fallback_triggered,
        query_type=query.query_type.value,
        merged_result=merged_result,
    )

    return RoutingDecision(
        query_id=query.query_id,
        query_type=query.query_type,
        primary_source=primary_source,
        fallback_source=fallback_source if fallback_triggered else None,
        confidence=confidence,
        fallback_triggered=fallback_triggered,
        merged_result=merged_result,
        receipt=receipt,
    )


def enrich_pattern_with_context(
    pattern: Dict[str, Any],
    query_type: QueryType = QueryType.PATTERN_MATCH,
) -> Dict[str, Any]:
    """Enrich pattern with context data.

    Args:
        pattern: Pattern to enrich
        query_type: Type of context query

    Returns:
        Enriched pattern
    """
    query = ContextQuery(
        query_id=dual_hash(json.dumps(pattern, sort_keys=True)),
        query_type=query_type,
        query_text=pattern.get("description", ""),
        domain=pattern.get("domain", "unknown"),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    decision = route_query(query, pattern)

    enriched = {
        **pattern,
        "context": {
            "source": decision.primary_source.value,
            "confidence": decision.confidence,
            "fallback_used": decision.fallback_triggered,
            "routing_receipt_hash": decision.receipt.get("payload_hash", ""),
        },
    }

    return enriched


def should_trigger_external_validation(confidence: float, query_type: QueryType) -> bool:
    """Check if external validation should be triggered.

    Args:
        confidence: Current confidence
        query_type: Query type

    Returns:
        True if external validation needed
    """
    threshold = get_confidence_threshold(query_type)
    return confidence < threshold
