"""axiom - Core Physics Compression Modules

THE AXIOM INSIGHT:
    Physics compresses. Symmetry predicts. Entropy bounds.
    The universe's laws are nature's compression algorithms.

Modules:
    cosmos: Galaxy rotation curve generation and loading
    witness: KAN-based physics discovery
    entropy: Information-theoretic measures and Landauer calibration
    colony: Mars colony state and psychology

Source: AXIOM Validation Lock v1
"""

from .cosmos import (
    generate_galaxy,
    generate_synthetic_dataset,
    PhysicsRegime,
)
from .witness import (
    KAN,
    train,
    crossover_detection,
    extract_symbolic,
)
from .entropy import (
    LANDAUER_LIMIT_J_PER_BIT,
    BASELINE_BITS_PER_KG,
    CREW_STRESS_ENTROPY_FACTOR,
    landauer_mass_equivalent,
    crew_psychology_entropy,
    total_colony_entropy,
)
from .colony import (
    ColonyState,
    CrewState,
    initialize_colony,
    update_colony_state,
)

__all__ = [
    # cosmos
    "generate_galaxy",
    "generate_synthetic_dataset",
    "PhysicsRegime",
    # witness
    "KAN",
    "train",
    "crossover_detection",
    "extract_symbolic",
    # entropy
    "LANDAUER_LIMIT_J_PER_BIT",
    "BASELINE_BITS_PER_KG",
    "CREW_STRESS_ENTROPY_FACTOR",
    "landauer_mass_equivalent",
    "crew_psychology_entropy",
    "total_colony_entropy",
    # colony
    "ColonyState",
    "CrewState",
    "initialize_colony",
    "update_colony_state",
]
