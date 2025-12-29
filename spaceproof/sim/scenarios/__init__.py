"""SpaceProof Scenarios - Twelve-Scenario Validation Framework.

The scenarios probe different aspects of system behavior:

CORE SCENARIOS (xAI collaboration):
1. BASELINE: Normal operation with standard probability distributions
2. STRESS: Edge cases at 3-5x normal intensity with heavy-tail distributions
3. GENESIS: System initialization with bootstrap validation
4. SINGULARITY: Self-referential conditions where system audits itself
5. THERMODYNAMIC: Entropy conservation verification per second law
6. GODEL: Completeness bounds and decidability limits

DEFENSE EXPANSION SCENARIOS (Grok research):
7. ORBITAL_COMPUTE: Starcloud orbital compute provenance
8. CONSTELLATION_SCALE: Starlink maneuver audit at scale
9. AUTONOMOUS_ACCOUNTABILITY: Defense DOD 3000.09 compliance
10. FIRMWARE_SUPPLY_CHAIN: Firmware integrity chain verification

Each scenario implements:
- Specific input distribution generators
- Custom checkpoint frequencies
- Scenario-specific validation criteria
- Receipt patterns appropriate to the scenario
"""

from spaceproof.sim.scenarios.baseline import BaselineScenario
from spaceproof.sim.scenarios.stress import StressScenario
from spaceproof.sim.scenarios.genesis import GenesisScenario
from spaceproof.sim.scenarios.singularity import SingularityScenario
from spaceproof.sim.scenarios.thermodynamic import ThermodynamicScenario
from spaceproof.sim.scenarios.godel import GodelScenario

# Defense expansion scenarios
from spaceproof.sim.scenarios.orbital_compute import OrbitalComputeScenario
from spaceproof.sim.scenarios.constellation_scale import ConstellationScaleScenario
from spaceproof.sim.scenarios.autonomous_accountability import AutonomousAccountabilityScenario
from spaceproof.sim.scenarios.firmware_supply_chain import FirmwareSupplyChainScenario

__all__ = [
    # Core scenarios
    "BaselineScenario",
    "StressScenario",
    "GenesisScenario",
    "SingularityScenario",
    "ThermodynamicScenario",
    "GodelScenario",
    # Defense expansion scenarios
    "OrbitalComputeScenario",
    "ConstellationScaleScenario",
    "AutonomousAccountabilityScenario",
    "FirmwareSupplyChainScenario",
]
