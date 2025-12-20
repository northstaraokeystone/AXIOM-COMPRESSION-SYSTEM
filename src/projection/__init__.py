"""D19.2 Projection Package - Future Path Projection Architecture.

PARADIGM INVERSION:
  OLD: "Observe pattern -> detect law -> enforce reactively"
  NEW: "Project future paths -> weave laws preemptively -> delay nullified before arrival"

The Physics (Block Universe):
  In block universe physics, the future already exists. We're not "predicting" -
  we're accessing a portion of spacetime that's already determined. Known latency
  (Proxima 8.48yr RTT) isn't an obstacle - it's INFORMATION about the future.

  When we know the delay, we can weave laws that compensate for it BEFORE it arrives.
  The delay appears to vanish because we've already accounted for it.

Grok's Core Insight:
  "Laws are not enforced reactively - they are woven preemptively
   from projected future entropy trajectories"

Modules:
  future_path_projection: Project receipt paths forward with light-speed bounds
  latency_bound_model: Light-speed constrained n-body model for path projection
  path_compression_estimator: Estimate compression of projected paths
"""

from .future_path_projection import (
    init_projection,
    load_latency_catalog,
    project_single_path,
    project_all_paths,
    apply_light_speed_bound,
    calculate_path_arrival,
    estimate_future_entropy,
    get_projection_status,
)

from .latency_bound_model import (
    init_model,
    add_body,
    calculate_geodesic,
    apply_gravitational_bending,
    validate_light_speed,
    get_arrival_time,
    get_model_status,
)

from .path_compression_estimator import (
    init_estimator,
    estimate_path_compression,
    estimate_batch_compression,
    rank_by_projected_compression,
    get_estimator_status,
)

__all__ = [
    # future_path_projection
    "init_projection",
    "load_latency_catalog",
    "project_single_path",
    "project_all_paths",
    "apply_light_speed_bound",
    "calculate_path_arrival",
    "estimate_future_entropy",
    "get_projection_status",
    # latency_bound_model
    "init_model",
    "add_body",
    "calculate_geodesic",
    "apply_gravitational_bending",
    "validate_light_speed",
    "get_arrival_time",
    "get_model_status",
    # path_compression_estimator
    "init_estimator",
    "estimate_path_compression",
    "estimate_batch_compression",
    "rank_by_projected_compression",
    "get_estimator_status",
]
