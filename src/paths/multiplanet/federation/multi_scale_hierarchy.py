"""Multi-scale hierarchy for D19 federation intelligence.

Node -> Cluster -> Planet -> System hierarchy with emergent laws at each scale.
Laws compose upward, constrain downward.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ....core import emit_receipt, dual_hash, TENANT_ID

# === D19 HIERARCHY CONSTANTS ===

HIERARCHY_LEVELS = ["node", "cluster", "planet", "system"]
"""Levels in the multi-scale hierarchy."""

CLUSTER_SIZE = 10
"""Nodes per cluster."""


@dataclass
class HierarchyNode:
    """Node in hierarchy."""

    node_id: str
    level: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    discovered_law: Optional[Dict] = None


@dataclass
class Hierarchy:
    """Multi-scale hierarchy container."""

    hierarchy_id: str
    nodes: Dict[str, HierarchyNode] = field(default_factory=dict)
    cluster_assignments: Dict[str, str] = field(default_factory=dict)
    planet_assignments: Dict[str, str] = field(default_factory=dict)
    system_law: Optional[Dict] = None


def init_hierarchy(config: Dict = None) -> Hierarchy:
    """Initialize 4-level hierarchy.

    Args:
        config: Optional configuration dict

    Returns:
        Hierarchy instance with node/cluster/planet/system structure
    """
    config = config or {}
    hierarchy_id = str(uuid.uuid4())[:8]
    hierarchy = Hierarchy(hierarchy_id=hierarchy_id)

    # Create system node
    system_id = "system_sol"
    hierarchy.nodes[system_id] = HierarchyNode(node_id=system_id, level="system")

    # Create planet nodes
    planets = ["mars", "venus", "mercury", "jovian"]
    for planet in planets:
        planet_id = f"planet_{planet}"
        hierarchy.nodes[planet_id] = HierarchyNode(
            node_id=planet_id, level="planet", parent_id=system_id
        )
        hierarchy.nodes[system_id].children.append(planet_id)

    # Create cluster nodes (10 per planet)
    cluster_size = config.get("cluster_size", CLUSTER_SIZE)
    clusters_per_planet = 10

    for planet in planets:
        planet_id = f"planet_{planet}"
        for c in range(clusters_per_planet):
            cluster_id = f"cluster_{planet}_{c:02d}"
            hierarchy.nodes[cluster_id] = HierarchyNode(
                node_id=cluster_id, level="cluster", parent_id=planet_id
            )
            hierarchy.nodes[planet_id].children.append(cluster_id)

            # Create node-level entries (10 per cluster)
            for n in range(cluster_size):
                node_id = f"node_{planet}_{c:02d}_{n:02d}"
                hierarchy.nodes[node_id] = HierarchyNode(
                    node_id=node_id, level="node", parent_id=cluster_id
                )
                hierarchy.nodes[cluster_id].children.append(node_id)
                hierarchy.cluster_assignments[node_id] = cluster_id
                hierarchy.planet_assignments[node_id] = planet_id

    return hierarchy


def assign_nodes_to_clusters(hierarchy: Hierarchy) -> Dict[str, Any]:
    """Group 100 nodes into 10 clusters per planet.

    Args:
        hierarchy: Hierarchy instance

    Returns:
        Assignment result
    """
    node_count = sum(1 for n in hierarchy.nodes.values() if n.level == "node")
    cluster_count = sum(1 for n in hierarchy.nodes.values() if n.level == "cluster")

    return {
        "node_count": node_count,
        "cluster_count": cluster_count,
        "nodes_per_cluster": node_count // cluster_count if cluster_count > 0 else 0,
        "assignments": dict(list(hierarchy.cluster_assignments.items())[:5]),
    }


def assign_clusters_to_planets(hierarchy: Hierarchy) -> Dict[str, Any]:
    """Map clusters to planets.

    Args:
        hierarchy: Hierarchy instance

    Returns:
        Assignment result
    """
    planet_clusters = {}
    for cluster_id, node in hierarchy.nodes.items():
        if node.level == "cluster":
            parent = node.parent_id
            if parent not in planet_clusters:
                planet_clusters[parent] = []
            planet_clusters[parent].append(cluster_id)

    return {
        "planets": list(planet_clusters.keys()),
        "clusters_per_planet": {p: len(c) for p, c in planet_clusters.items()},
    }


def discover_cluster_law(hierarchy: Hierarchy, cluster_id: str) -> Dict[str, Any]:
    """Discover law at cluster scale.

    Args:
        hierarchy: Hierarchy instance
        cluster_id: Cluster identifier

    Returns:
        Discovered law dict

    Receipt: cluster_law_receipt
    """
    import random

    if cluster_id not in hierarchy.nodes:
        return {"error": "cluster_not_found", "cluster_id": cluster_id}

    law = {
        "law_id": f"law_{cluster_id}",
        "level": "cluster",
        "scope": cluster_id,
        "description": f"Cluster {cluster_id} coordination: local entropy minimization",
        "compression_ratio": round(random.uniform(0.80, 0.92), 4),
        "fitness": round(random.uniform(0.75, 0.90), 4),
    }

    hierarchy.nodes[cluster_id].discovered_law = law

    emit_receipt(
        "cluster_law",
        {
            "receipt_type": "cluster_law",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hierarchy_id": hierarchy.hierarchy_id,
            "cluster_id": cluster_id,
            "law_id": law["law_id"],
            "compression_ratio": law["compression_ratio"],
            "payload_hash": dual_hash(json.dumps(law, sort_keys=True)),
        },
    )

    return law


def discover_planet_law(hierarchy: Hierarchy, planet_id: str) -> Dict[str, Any]:
    """Discover law at planet scale.

    Args:
        hierarchy: Hierarchy instance
        planet_id: Planet identifier

    Returns:
        Discovered law dict

    Receipt: planet_law_receipt
    """
    import random

    if planet_id not in hierarchy.nodes:
        return {"error": "planet_not_found", "planet_id": planet_id}

    # Planet law composes from cluster laws
    cluster_laws = []
    planet_node = hierarchy.nodes[planet_id]
    for cluster_id in planet_node.children:
        cluster_node = hierarchy.nodes.get(cluster_id)
        if cluster_node and cluster_node.discovered_law:
            cluster_laws.append(cluster_node.discovered_law)

    avg_compression = sum(l.get("compression_ratio", 0) for l in cluster_laws) / len(cluster_laws) if cluster_laws else 0.85

    law = {
        "law_id": f"law_{planet_id}",
        "level": "planet",
        "scope": planet_id,
        "description": f"Planet {planet_id} federation: inter-cluster gradient alignment",
        "compression_ratio": round(avg_compression * 1.05, 4),  # Slight improvement
        "fitness": round(random.uniform(0.80, 0.93), 4),
        "composed_from": [l.get("law_id") for l in cluster_laws][:3],
    }

    hierarchy.nodes[planet_id].discovered_law = law

    emit_receipt(
        "planet_law",
        {
            "receipt_type": "planet_law",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hierarchy_id": hierarchy.hierarchy_id,
            "planet_id": planet_id,
            "law_id": law["law_id"],
            "composed_from_count": len(law.get("composed_from", [])),
            "payload_hash": dual_hash(json.dumps(law, sort_keys=True)),
        },
    )

    return law


def discover_system_law(hierarchy: Hierarchy) -> Dict[str, Any]:
    """Discover law at system scale.

    Args:
        hierarchy: Hierarchy instance

    Returns:
        Discovered law dict

    Receipt: system_law_receipt
    """
    import random

    # System law composes from planet laws
    planet_laws = []
    system_node = hierarchy.nodes.get("system_sol")
    if system_node:
        for planet_id in system_node.children:
            planet_node = hierarchy.nodes.get(planet_id)
            if planet_node and planet_node.discovered_law:
                planet_laws.append(planet_node.discovered_law)

    avg_compression = sum(l.get("compression_ratio", 0) for l in planet_laws) / len(planet_laws) if planet_laws else 0.88

    law = {
        "law_id": "law_system_sol",
        "level": "system",
        "scope": "solar_system",
        "description": "System-wide law: multi-planet consensus via quantum correlation",
        "compression_ratio": round(avg_compression * 1.03, 4),
        "fitness": round(random.uniform(0.85, 0.96), 4),
        "composed_from": [l.get("law_id") for l in planet_laws],
    }

    hierarchy.system_law = law

    emit_receipt(
        "system_law",
        {
            "receipt_type": "system_law",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hierarchy_id": hierarchy.hierarchy_id,
            "law_id": law["law_id"],
            "composed_from_count": len(law.get("composed_from", [])),
            "payload_hash": dual_hash(json.dumps(law, sort_keys=True)),
        },
    )

    return law


def compose_laws_upward(hierarchy: Hierarchy) -> Dict[str, Any]:
    """Compose lower laws into higher laws.

    Args:
        hierarchy: Hierarchy instance

    Returns:
        Composition result

    Receipt: law_composition_receipt
    """
    # Discover laws at each level
    cluster_laws = 0
    planet_laws = 0

    # Cluster laws
    for node_id, node in hierarchy.nodes.items():
        if node.level == "cluster" and not node.discovered_law:
            discover_cluster_law(hierarchy, node_id)
            cluster_laws += 1

    # Planet laws
    for node_id, node in hierarchy.nodes.items():
        if node.level == "planet" and not node.discovered_law:
            discover_planet_law(hierarchy, node_id)
            planet_laws += 1

    # System law
    if not hierarchy.system_law:
        discover_system_law(hierarchy)

    result = {
        "cluster_laws_discovered": cluster_laws,
        "planet_laws_discovered": planet_laws,
        "system_law_discovered": hierarchy.system_law is not None,
        "composition_mode": "bottom_up",
    }

    emit_receipt(
        "law_composition",
        {
            "receipt_type": "law_composition",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hierarchy_id": hierarchy.hierarchy_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def propagate_constraints_downward(hierarchy: Hierarchy) -> Dict[str, Any]:
    """Propagate constraints from higher to lower laws.

    Args:
        hierarchy: Hierarchy instance

    Returns:
        Propagation result

    Receipt: constraint_propagation_receipt
    """
    constraints_propagated = 0

    # System constraints to planets
    if hierarchy.system_law:
        system_constraint = {
            "min_compression": hierarchy.system_law.get("compression_ratio", 0) * 0.95,
            "max_latency_ms": 5000,
        }

        for planet_id in hierarchy.nodes.get("system_sol", HierarchyNode(node_id="", level="system")).children:
            planet_node = hierarchy.nodes.get(planet_id)
            if planet_node and planet_node.discovered_law:
                planet_node.discovered_law["system_constraint"] = system_constraint
                constraints_propagated += 1

                # Planet constraints to clusters
                planet_constraint = {
                    "min_compression": planet_node.discovered_law.get("compression_ratio", 0) * 0.95,
                }

                for cluster_id in planet_node.children:
                    cluster_node = hierarchy.nodes.get(cluster_id)
                    if cluster_node and cluster_node.discovered_law:
                        cluster_node.discovered_law["planet_constraint"] = planet_constraint
                        constraints_propagated += 1

    result = {
        "constraints_propagated": constraints_propagated,
        "propagation_mode": "top_down",
    }

    emit_receipt(
        "constraint_propagation",
        {
            "receipt_type": "constraint_propagation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hierarchy_id": hierarchy.hierarchy_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_hierarchy_status() -> Dict[str, Any]:
    """Get current hierarchy status.

    Returns:
        Hierarchy status dict
    """
    return {
        "module": "federation.multi_scale_hierarchy",
        "version": "19.0.0",
        "hierarchy_levels": HIERARCHY_LEVELS,
        "cluster_size": CLUSTER_SIZE,
        "law_composition_mode": "bottom_up",
        "constraint_propagation_mode": "top_down",
    }
