"""D19.2 Weave Package - Preemptive Law Weaving Architecture.

PARADIGM INVERSION:
  OLD: "Observe pattern -> detect law -> enforce reactively"
  NEW: "Project future paths -> weave laws preemptively -> delay nullified before arrival"

The Physics (Block Universe):
  Laws are not enforced reactively - they are woven preemptively from projected
  future entropy trajectories. The future already exists. When we know the delay,
  we can weave laws that compensate for it BEFORE it arrives.

Grok's Core Insight:
  "Laws are not enforced reactively - they are woven preemptively
   from projected future entropy trajectories"

Modules:
  preemptive_weave: Pre-amplify/starve based on PROJECTED compression
  impending_entropy_weave: Use known latency as weave template
  delay_nullification: Laws that nullify delay before arrival
  weave_to_chain: Insert woven laws into current chain
"""

from .preemptive_weave import (
    init_preemptive_weave,
    amplify_high_future_paths,
    starve_low_future_paths,
    apply_preemptive_selection,
    get_weave_status,
)

from .impending_entropy_weave import (
    init_entropy_weave,
    load_weave_template,
    weave_from_known_latency,
    generate_nullification_laws,
    get_entropy_weave_status,
)

from .delay_nullification import (
    init_nullification,
    nullify_known_delay,
    generate_preemptive_law,
    verify_nullification,
    get_nullification_status,
)

from .weave_to_chain import (
    init_weave_chain,
    insert_woven_law,
    batch_insert_laws,
    verify_chain_integrity,
    get_chain_status,
)

__all__ = [
    # preemptive_weave
    "init_preemptive_weave",
    "amplify_high_future_paths",
    "starve_low_future_paths",
    "apply_preemptive_selection",
    "get_weave_status",
    # impending_entropy_weave
    "init_entropy_weave",
    "load_weave_template",
    "weave_from_known_latency",
    "generate_nullification_laws",
    "get_entropy_weave_status",
    # delay_nullification
    "init_nullification",
    "nullify_known_delay",
    "generate_preemptive_law",
    "verify_nullification",
    "get_nullification_status",
    # weave_to_chain
    "init_weave_chain",
    "insert_woven_law",
    "batch_insert_laws",
    "verify_chain_integrity",
    "get_chain_status",
]
