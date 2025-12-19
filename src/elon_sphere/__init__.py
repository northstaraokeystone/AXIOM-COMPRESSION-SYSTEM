"""src/elon_sphere/__init__.py - Elon-sphere integration package.

Provides integration hooks for Starlink relay analog, Grok inference,
xAI compute, and Tesla Dojo offload capabilities.
"""

from src.elon_sphere.starlink_relay import (
    load_starlink_config,
    initialize_starlink_mesh,
    simulate_laser_link,
    relay_hop_latency,
    analog_to_interstellar,
    mars_comms_proof,
    get_starlink_status,
)

from src.elon_sphere.grok_inference import (
    load_grok_config,
    initialize_grok_agents,
    parallel_inference,
    latency_tuning_loop,
    ensemble_integration,
    get_grok_status,
)

from src.elon_sphere.xai_compute import (
    load_xai_config,
    initialize_colossus,
    quantum_sim_batch,
    entanglement_modeling,
    scale_to_interstellar,
    get_xai_status,
)

from src.elon_sphere.dojo_offload import (
    load_dojo_config,
    initialize_dojo_cluster,
    offload_recursion_training,
    fractal_optimization_batch,
    retrieve_trained_model,
    get_dojo_status,
)

__all__ = [
    # Starlink
    "load_starlink_config",
    "initialize_starlink_mesh",
    "simulate_laser_link",
    "relay_hop_latency",
    "analog_to_interstellar",
    "mars_comms_proof",
    "get_starlink_status",
    # Grok
    "load_grok_config",
    "initialize_grok_agents",
    "parallel_inference",
    "latency_tuning_loop",
    "ensemble_integration",
    "get_grok_status",
    # xAI
    "load_xai_config",
    "initialize_colossus",
    "quantum_sim_batch",
    "entanglement_modeling",
    "scale_to_interstellar",
    "get_xai_status",
    # Dojo
    "load_dojo_config",
    "initialize_dojo_cluster",
    "offload_recursion_training",
    "fractal_optimization_batch",
    "retrieve_trained_model",
    "get_dojo_status",
]
