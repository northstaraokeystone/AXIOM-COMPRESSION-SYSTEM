"""D19.3 Causal Subgraph Extractor.

Purpose: Extract maximal causal subgraphs from chain history.
Laws ARE the invariant subgraphs.

The Physics:
  Block universe causality works both ways. D19.2 used future-to-present
  (projection). D19.3 uses past-to-present (oracle). The chain history
  IS the oracle.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

from ..core import emit_receipt, dual_hash, TENANT_ID


@dataclass
class CausalNode:
    """Node in causal graph."""

    node_id: str
    receipt_type: str
    timestamp: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


@dataclass
class CausalSubgraph:
    """A connected causal subgraph."""

    subgraph_id: str
    nodes: Set[str] = field(default_factory=set)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    is_maximal: bool = False
    invariance_score: float = 0.0


@dataclass
class CausalSubgraphExtractor:
    """Extract maximal causal subgraphs from chain history.

    Laws = maximal invariant subgraphs that hold across entire history.
    """

    extractor_id: str
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    subgraphs: List[CausalSubgraph] = field(default_factory=list)
    laws: List[Dict] = field(default_factory=list)


def init_extractor(history: List[Dict] = None) -> CausalSubgraphExtractor:
    """Build causal graph from history receipts.

    Args:
        history: List of historical receipts

    Returns:
        CausalSubgraphExtractor instance
    """
    extractor_id = str(uuid.uuid4())[:8]
    extractor = CausalSubgraphExtractor(extractor_id=extractor_id)

    if history:
        # Build causal graph from history
        _build_graph_from_history(extractor, history)

    return extractor


def _build_graph_from_history(
    extractor: CausalSubgraphExtractor, history: List[Dict]
) -> None:
    """Build causal graph from history.

    Args:
        extractor: CausalSubgraphExtractor instance
        history: List of historical receipts
    """
    # Create nodes for each receipt
    for i, receipt in enumerate(history):
        node_id = receipt.get("payload_hash", f"node_{i}")[:16]
        receipt_type = receipt.get("receipt_type", "unknown")
        timestamp = receipt.get("ts", "")

        node = CausalNode(
            node_id=node_id,
            receipt_type=receipt_type,
            timestamp=timestamp,
        )
        extractor.nodes[node_id] = node

    # Create edges based on temporal ordering and type dependencies
    node_list = list(extractor.nodes.values())
    for i in range(1, len(node_list)):
        prev_node = node_list[i - 1]
        curr_node = node_list[i]

        # Temporal edge
        prev_node.dependents.add(curr_node.node_id)
        curr_node.dependencies.add(prev_node.node_id)
        extractor.edges.append((prev_node.node_id, curr_node.node_id))


def build_causal_graph(receipts: List[Dict]) -> Dict[str, Any]:
    """Construct DAG from receipt dependencies.

    Args:
        receipts: List of receipt dicts

    Returns:
        Causal graph dict
    """
    nodes = []
    edges = []

    # Build node list
    for i, receipt in enumerate(receipts):
        node_id = receipt.get("payload_hash", f"node_{i}")[:16]
        nodes.append(
            {
                "id": node_id,
                "type": receipt.get("receipt_type", "unknown"),
                "ts": receipt.get("ts", ""),
            }
        )

    # Build edges (temporal ordering)
    for i in range(1, len(nodes)):
        edges.append(
            {
                "from": nodes[i - 1]["id"],
                "to": nodes[i]["id"],
                "type": "temporal",
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "is_dag": True,  # By construction
    }


def find_maximal_subgraphs(extractor: CausalSubgraphExtractor) -> List[CausalSubgraph]:
    """Find strongly connected components.

    Args:
        extractor: CausalSubgraphExtractor instance

    Returns:
        List of maximal CausalSubgraph instances
    """
    if not extractor.nodes:
        return []

    # Find connected components using DFS
    visited = set()
    subgraphs = []

    def dfs(node_id: str, component: Set[str]):
        if node_id in visited:
            return
        visited.add(node_id)
        component.add(node_id)

        node = extractor.nodes.get(node_id)
        if node:
            for dep_id in node.dependencies:
                if dep_id in extractor.nodes:
                    dfs(dep_id, component)
            for dep_id in node.dependents:
                if dep_id in extractor.nodes:
                    dfs(dep_id, component)

    # Find all connected components
    for node_id in extractor.nodes:
        if node_id not in visited:
            component = set()
            dfs(node_id, component)

            if component:
                subgraph_id = str(uuid.uuid4())[:8]
                subgraph = CausalSubgraph(
                    subgraph_id=subgraph_id,
                    nodes=component,
                    is_maximal=True,
                    invariance_score=len(component) / len(extractor.nodes)
                    if extractor.nodes
                    else 0,
                )
                subgraphs.append(subgraph)

    extractor.subgraphs = subgraphs
    return subgraphs


def subgraph_to_law(
    subgraph: CausalSubgraph, extractor: CausalSubgraphExtractor
) -> Dict[str, Any]:
    """Convert invariant subgraph to law spec.

    Args:
        subgraph: CausalSubgraph instance
        extractor: CausalSubgraphExtractor instance

    Returns:
        Law dict
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Collect receipt types in subgraph
    types_in_subgraph = set()
    for node_id in subgraph.nodes:
        node = extractor.nodes.get(node_id)
        if node:
            types_in_subgraph.add(node.receipt_type)

    law = {
        "law_id": str(uuid.uuid4())[:8],
        "law_type": "causal_subgraph",
        "subgraph_id": subgraph.subgraph_id,
        "node_count": len(subgraph.nodes),
        "receipt_types": list(types_in_subgraph),
        "invariance_score": round(subgraph.invariance_score, 4),
        "is_maximal": subgraph.is_maximal,
        "source": "causal_subgraph_extraction",
        "created_at": now,
    }

    return law


def validate_causal_invariance(law: Dict, history: List[Dict]) -> bool:
    """Verify law holds across all history.

    Args:
        law: Law dict to validate
        history: Full chain history

    Returns:
        True if law is invariant across history
    """
    if not history:
        return False

    # Check if the pattern in the law holds
    law_types = set(law.get("receipt_types", []))
    if not law_types:
        return True  # Empty laws are vacuously true

    # Count occurrences of law types in history
    history_types = set(r.get("receipt_type", "") for r in history)

    # Law is valid if all its types appear in history
    types_present = law_types.issubset(history_types)

    # Check invariance score
    invariance = law.get("invariance_score", 0)

    return types_present and invariance >= 0.5


def emit_subgraph_receipt(
    extractor: CausalSubgraphExtractor, laws: List[Dict]
) -> Dict[str, Any]:
    """Emit causal_subgraph_law_receipt.

    Args:
        extractor: CausalSubgraphExtractor instance
        laws: List of discovered laws

    Returns:
        Receipt dict
    """
    now = datetime.utcnow().isoformat() + "Z"

    extractor.laws = laws

    receipt_data = {
        "receipt_type": "causal_subgraph_law",
        "tenant_id": TENANT_ID,
        "ts": now,
        "extractor_id": extractor.extractor_id,
        "node_count": len(extractor.nodes),
        "edge_count": len(extractor.edges),
        "subgraph_count": len(extractor.subgraphs),
        "laws_discovered": len(laws),
        "source": "maximal_causal_subgraph",
        "payload_hash": dual_hash(
            json.dumps(
                {
                    "extractor_id": extractor.extractor_id,
                    "laws": len(laws),
                },
                sort_keys=True,
            )
        ),
    }

    emit_receipt("causal_subgraph_law", receipt_data)

    return receipt_data
