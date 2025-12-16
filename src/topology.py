"""Manifold Structure Scanner - Persistent homology using Ripser.

Computes H0 (connected components), H1 (holes/loops), H2 (voids) from
velocity field point clouds. H1 features = dark matter topology.
H2 features = gravitational cavities.
"""

import base64
import json
from typing import Union

import numpy as np
from ripser import ripser
from scipy.stats import wasserstein_distance

from .core import dual_hash, emit_receipt


# Default threshold for filtering noise in persistence diagrams
PERSISTENCE_THRESHOLD = 0.01


def compute_persistence(velocity_field: np.ndarray) -> dict:
    """Compute persistent homology H0, H1, H2 from velocity field points.

    Args:
        velocity_field: (N, 2) or (N, 3) array of points

    Returns:
        Dict with H0, H1, H2 persistence diagrams as numpy arrays
    """
    if velocity_field.ndim == 1:
        velocity_field = velocity_field.reshape(-1, 1)

    if velocity_field.shape[0] < 3:
        return {
            "H0": np.array([]).reshape(0, 2),
            "H1": np.array([]).reshape(0, 2),
            "H2": np.array([]).reshape(0, 2),
        }

    # Ripser computation with maxdim=2 for H0, H1, H2
    result = ripser(velocity_field, maxdim=2)
    diagrams = result["dgms"]

    return {
        "H0": diagrams[0] if len(diagrams) > 0 else np.array([]).reshape(0, 2),
        "H1": diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2),
        "H2": diagrams[2] if len(diagrams) > 2 else np.array([]).reshape(0, 2),
    }


def classify_features(
    diagram: Union[dict, np.ndarray], threshold: float = PERSISTENCE_THRESHOLD
) -> list:
    """Filter noise and extract significant features from persistence diagram.

    Args:
        diagram: Either dict with H0/H1/H2 keys or single (N,2) diagram array
        threshold: Minimum persistence to keep (default 0.01)

    Returns:
        List of (birth, death, persistence) tuples for significant features
    """
    if isinstance(diagram, dict):
        # Combine H1 and H2 features (most interesting)
        all_features = []
        for key in ["H1", "H2"]:
            if key in diagram and len(diagram[key]) > 0:
                arr = diagram[key]
                for row in arr:
                    if len(row) >= 2 and np.isfinite(row[1]):
                        birth, death = row[0], row[1]
                        pers = death - birth
                        if pers >= threshold:
                            all_features.append((float(birth), float(death), float(pers)))
        return all_features
    else:
        # Single array
        features = []
        for row in diagram:
            if len(row) >= 2 and np.isfinite(row[1]):
                birth, death = row[0], row[1]
                pers = death - birth
                if pers >= threshold:
                    features.append((float(birth), float(death), float(pers)))
        return features


def topology_loss_term(obs: dict, pred: dict) -> float:
    """Compute Wasserstein distance between observed and predicted H1 diagrams.

    Args:
        obs: Observed persistence diagram dict with H1 key
        pred: Predicted persistence diagram dict with H1 key

    Returns:
        Wasserstein distance (float), 0.0 if either is empty
    """
    obs_h1 = obs.get("H1", np.array([]))
    pred_h1 = pred.get("H1", np.array([]))

    if len(obs_h1) == 0 or len(pred_h1) == 0:
        return 0.0

    # Flatten to persistence values for 1D Wasserstein
    obs_pers = []
    for row in obs_h1:
        if len(row) >= 2 and np.isfinite(row[1]):
            obs_pers.append(row[1] - row[0])

    pred_pers = []
    for row in pred_h1:
        if len(row) >= 2 and np.isfinite(row[1]):
            pred_pers.append(row[1] - row[0])

    if len(obs_pers) == 0 or len(pred_pers) == 0:
        return 0.0

    return float(wasserstein_distance(obs_pers, pred_pers))


def h1_interpretation(features: list) -> dict:
    """Interpret H1 features as dark matter topology indicators.

    Args:
        features: List of (birth, death, persistence) tuples from classify_features

    Returns:
        Dict with hole_count, max_persistence, total_persistence, interpretation
    """
    if not features:
        return {
            "hole_count": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "interpretation": "No topological holes detected - trivial H1",
        }

    persistences = [f[2] for f in features]
    max_pers = max(persistences)
    total_pers = sum(persistences)
    hole_count = len(features)

    # Interpretation based on persistence values
    if max_pers > 0.5:
        interpretation = "Strong H1 features - dark matter halo topology likely"
    elif max_pers > 0.1:
        interpretation = "Moderate H1 features - possible dark matter substructure"
    else:
        interpretation = "Weak H1 features - noise or minimal dark matter contribution"

    return {
        "hole_count": hole_count,
        "max_persistence": float(max_pers),
        "total_persistence": float(total_pers),
        "interpretation": interpretation,
    }


def h2_interpretation(features: list) -> dict:
    """Interpret H2 features as gravitational cavity indicators.

    Args:
        features: List of (birth, death, persistence) tuples from H2 diagram

    Returns:
        Dict with void_count, max_persistence, total_persistence, interpretation
    """
    if not features:
        return {
            "void_count": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "interpretation": "No topological voids detected - trivial H2",
        }

    persistences = [f[2] for f in features]
    max_pers = max(persistences)
    total_pers = sum(persistences)
    void_count = len(features)

    # Interpretation based on persistence values
    if max_pers > 0.5:
        interpretation = "Strong H2 features - significant gravitational cavities"
    elif max_pers > 0.1:
        interpretation = "Moderate H2 features - possible void structure"
    else:
        interpretation = "Weak H2 features - minimal cavity structure"

    return {
        "void_count": void_count,
        "max_persistence": float(max_pers),
        "total_persistence": float(total_pers),
        "interpretation": interpretation,
    }


def trivial_topology_check(diagram: dict) -> bool:
    """Check if topology is trivial (no significant H1/H2 features).

    Args:
        diagram: Dict with H0, H1, H2 keys

    Returns:
        True if no significant H1 or H2 features exist
    """
    h1_features = classify_features({"H1": diagram.get("H1", np.array([]))})
    h2_features = classify_features({"H2": diagram.get("H2", np.array([]))})

    return len(h1_features) == 0 and len(h2_features) == 0


def _encode_diagram(arr: np.ndarray) -> str:
    """Base64 encode a numpy array for receipt storage."""
    return base64.b64encode(arr.tobytes()).decode("ascii")


def topology_receipt_emit(
    galaxy_id: str, diagram: dict, interpretation: dict
) -> dict:
    """Emit topology_receipt with diagram and interpretation data.

    Args:
        galaxy_id: Galaxy identifier string
        diagram: Dict with H0, H1, H2 persistence diagrams
        interpretation: Dict with h1_interpretation and h2_interpretation results

    Returns:
        Emitted receipt dict
    """
    # Encode diagrams as base64 for storage
    encoded_diagrams = {}
    for key in ["H0", "H1", "H2"]:
        if key in diagram and len(diagram[key]) > 0:
            encoded_diagrams[key] = _encode_diagram(diagram[key])
        else:
            encoded_diagrams[key] = ""

    # Compute feature counts
    h1_features = classify_features({"H1": diagram.get("H1", np.array([]))})
    h2_features = classify_features({"H2": diagram.get("H2", np.array([]))})

    receipt_data = {
        "tenant_id": "axiom-0",
        "galaxy_id": galaxy_id,
        "diagrams_b64": encoded_diagrams,
        "h1_hole_count": len(h1_features),
        "h2_void_count": len(h2_features),
        "trivial_topology": trivial_topology_check(diagram),
        "interpretation": interpretation,
    }

    return emit_receipt("topology", receipt_data)
