"""SpaceProof Scenarios - Eighteen-Scenario Validation Framework.

The scenarios probe different aspects of system behavior:

CORE SCENARIOS:
1. BASELINE: Normal operation with standard probability distributions
2. STRESS: Edge cases at 3-5x normal intensity with heavy-tail distributions
3. GENESIS: System initialization with bootstrap validation
4. SINGULARITY: Self-referential conditions where system audits itself
5. THERMODYNAMIC: Entropy conservation verification per second law
6. GODEL: Completeness bounds and decidability limits

DEFENSE EXPANSION SCENARIOS:
7. ORBITAL_COMPUTE: Starcloud orbital compute provenance
8. CONSTELLATION_SCALE: Starlink maneuver audit at scale
9. AUTONOMOUS_ACCOUNTABILITY: Defense DOD 3000.09 compliance
10. FIRMWARE_SUPPLY_CHAIN: Firmware integrity chain verification

GOVERNANCE & TRAINING SCENARIOS (v2.0):
11. GOVERNANCE: Enterprise governance patterns (RACI, provenance, reason codes)
12. TRAINING_PRODUCTION: Training data factory validation

PRIVACY & OFFLINE SCENARIOS (v2.0):
13. PRIVACY_ENFORCEMENT: PII redaction and differential privacy
14. OFFLINE_RESILIENCE: Light-delay tolerant sync and conflict resolution

HARDWARE SUPPLY CHAIN SCENARIOS (v3.0):
15. HARDWARE_SUPPLY_CHAIN_DISCOVERY: META-LOOP discovers counterfeit/rework patterns
16. POWER_SUPPLY_PROTOTYPE: Jay's exact use case - module verification

Each scenario implements:
- Specific input distribution generators
- Custom checkpoint frequencies
- Scenario-specific validation criteria
- Receipt patterns appropriate to the scenario
"""

from spaceproof.sim.scenarios.baseline import (
    BaselineScenario,
    BaselineConfig,
    BaselineResult,
)
from spaceproof.sim.scenarios.stress import StressScenario, StressConfig, StressResult
from spaceproof.sim.scenarios.genesis import (
    GenesisScenario,
    GenesisConfig,
    GenesisResult,
)
from spaceproof.sim.scenarios.singularity import (
    SingularityScenario,
    SingularityConfig,
    SingularityResult,
)
from spaceproof.sim.scenarios.thermodynamic import (
    ThermodynamicScenario,
    ThermodynamicConfig,
    ThermodynamicResult,
)
from spaceproof.sim.scenarios.godel import GodelScenario, GodelConfig, GodelResult

# Defense expansion scenarios
from spaceproof.sim.scenarios.orbital_compute import (
    OrbitalComputeScenario,
    OrbitalComputeConfig,
    OrbitalComputeResult,
)
from spaceproof.sim.scenarios.constellation_scale import (
    ConstellationScaleScenario,
    ConstellationScaleConfig,
    ConstellationScaleResult,
)
from spaceproof.sim.scenarios.autonomous_accountability import (
    AutonomousAccountabilityScenario,
    AutonomousAccountabilityConfig,
    AutonomousAccountabilityResult,
)
from spaceproof.sim.scenarios.firmware_supply_chain import (
    FirmwareSupplyChainScenario,
    FirmwareSupplyChainConfig,
    FirmwareSupplyChainResult,
)

# v2.0 scenarios
from spaceproof.sim.scenarios.governance import (
    GovernanceScenario,
    GovernanceConfig,
    GovernanceResult,
)
from spaceproof.sim.scenarios.training_production import (
    TrainingProductionScenario,
    TrainingProductionConfig,
    TrainingProductionResult,
)
from spaceproof.sim.scenarios.privacy_enforcement import (
    PrivacyEnforcementScenario,
    PrivacyEnforcementConfig,
    PrivacyEnforcementResult,
)
from spaceproof.sim.scenarios.offline_resilience import (
    OfflineResilienceScenario,
    OfflineResilienceConfig,
    OfflineResilienceResult,
)

# v3.0 Hardware supply chain scenarios
from spaceproof.sim.scenarios.hardware_supply_chain import (
    HardwareSupplyChainScenario,
    HardwareSupplyChainConfig,
    HardwareSupplyChainResult,
    PowerSupplyPrototypeScenario,
    PowerSupplyPrototypeConfig,
    PowerSupplyPrototypeResult,
)

__all__ = [
    # Core scenarios
    "BaselineScenario",
    "BaselineConfig",
    "BaselineResult",
    "StressScenario",
    "StressConfig",
    "StressResult",
    "GenesisScenario",
    "GenesisConfig",
    "GenesisResult",
    "SingularityScenario",
    "SingularityConfig",
    "SingularityResult",
    "ThermodynamicScenario",
    "ThermodynamicConfig",
    "ThermodynamicResult",
    "GodelScenario",
    "GodelConfig",
    "GodelResult",
    # Defense expansion scenarios
    "OrbitalComputeScenario",
    "OrbitalComputeConfig",
    "OrbitalComputeResult",
    "ConstellationScaleScenario",
    "ConstellationScaleConfig",
    "ConstellationScaleResult",
    "AutonomousAccountabilityScenario",
    "AutonomousAccountabilityConfig",
    "AutonomousAccountabilityResult",
    "FirmwareSupplyChainScenario",
    "FirmwareSupplyChainConfig",
    "FirmwareSupplyChainResult",
    # v2.0 scenarios
    "GovernanceScenario",
    "GovernanceConfig",
    "GovernanceResult",
    "TrainingProductionScenario",
    "TrainingProductionConfig",
    "TrainingProductionResult",
    "PrivacyEnforcementScenario",
    "PrivacyEnforcementConfig",
    "PrivacyEnforcementResult",
    "OfflineResilienceScenario",
    "OfflineResilienceConfig",
    "OfflineResilienceResult",
    # v3.0 Hardware supply chain scenarios
    "HardwareSupplyChainScenario",
    "HardwareSupplyChainConfig",
    "HardwareSupplyChainResult",
    "PowerSupplyPrototypeScenario",
    "PowerSupplyPrototypeConfig",
    "PowerSupplyPrototypeResult",
]
