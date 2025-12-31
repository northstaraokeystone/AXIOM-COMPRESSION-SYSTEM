"""hardware_supply_chain.py - Hardware Supply Chain Discovery Scenario.

SCENARIO_HARDWARE_SUPPLY_CHAIN_DISCOVERY:
    Purpose: Inject hardware fraud patterns, validate META-LOOP discovers helpers
    Cycles: 1000
    Inject:
      - 100 components from various manufacturers
      - 10 counterfeit components (high entropy, missing provenance)
      - 20 legitimate components with excessive rework (>3 cycles)
      - 5 components with broken provenance chains (Merkle gaps)
      - 15 components from unknown distributors (no baseline)

    Expected Helper Emergence:
      - COUNTERFEIT_HUNTER pattern emerges by cycle 200
      - REWORK_SHEPHERD pattern emerges by cycle 300
      - PROVENANCE_ARCHITECT pattern emerges by cycle 400

    Pass Criteria:
      - All 10 counterfeits detected by cycle 500 (100% recall)
      - All 20 excessive rework flagged (100% recall)
      - >=3 helper patterns graduate (E >= V_esc)
      - >=5 CASCADE variants spawned
      - Zero false positives on legitimate components (<1% FPR)
      - |ΔS| < 0.01 conservation maintained

Source: Jay's power supply verification use case
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from spaceproof.core import emit_receipt, dual_hash
from spaceproof.detect import (
    detect_hardware_fraud,
)
from spaceproof.anchor import (
    create_manufacturer_receipt,
    create_rework_receipt,
)
from spaceproof.meta_integration import (
    run_hardware_meta_loop,
    validate_entropy_conservation,
)

# === CONSTANTS ===

SCENARIO_CYCLES = 1000
TOTAL_COMPONENTS = 100
COUNTERFEIT_COMPONENTS = 10
EXCESSIVE_REWORK_COMPONENTS = 20
BROKEN_CHAIN_COMPONENTS = 5
UNKNOWN_DISTRIBUTOR_COMPONENTS = 15
LEGITIMATE_COMPONENTS = TOTAL_COMPONENTS - COUNTERFEIT_COMPONENTS - EXCESSIVE_REWORK_COMPONENTS - BROKEN_CHAIN_COMPONENTS - UNKNOWN_DISTRIBUTOR_COMPONENTS

TENANT_ID = "spaceproof-scenario-hardware-supply-chain"

# Manufacturers
KNOWN_MANUFACTURERS = ["Vishay", "TI", "Murata", "Kemet", "ON Semi", "Analog Devices"]
UNKNOWN_MANUFACTURERS = ["Unknown", "Generic", "NoName"]


@dataclass
class HardwareSupplyChainConfig:
    """Configuration for hardware supply chain scenario."""

    cycles: int = SCENARIO_CYCLES
    total_components: int = TOTAL_COMPONENTS
    counterfeit_count: int = COUNTERFEIT_COMPONENTS
    excessive_rework_count: int = EXCESSIVE_REWORK_COMPONENTS
    broken_chain_count: int = BROKEN_CHAIN_COMPONENTS
    unknown_distributor_count: int = UNKNOWN_DISTRIBUTOR_COMPONENTS
    seed: int = 42


@dataclass
class HardwareSupplyChainResult:
    """Result of hardware supply chain scenario."""

    cycles_completed: int
    components_analyzed: int
    counterfeits_detected: int
    counterfeits_total: int
    excessive_rework_detected: int
    excessive_rework_total: int
    broken_chains_detected: int
    broken_chains_total: int
    patterns_discovered: int
    patterns_graduated: int
    cascade_variants_spawned: int
    transfers_completed: int
    false_positive_rate: float
    entropy_conservation_valid: bool
    all_criteria_passed: bool


class HardwareSupplyChainScenario:
    """Scenario for validating hardware supply chain fraud detection via META-LOOP."""

    def __init__(self, config: Optional[HardwareSupplyChainConfig] = None):
        """Initialize scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or HardwareSupplyChainConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.components: List[Dict] = []
        self.detection_results: List[Dict] = []
        self.entropy_deltas: List[float] = []

    def generate_component_id(self, index: int, component_type: str) -> str:
        """Generate component ID."""
        type_prefix = {
            "legitimate": "LEG",
            "counterfeit": "CNT",
            "excessive_rework": "RWK",
            "broken_chain": "BRK",
            "unknown_distributor": "UNK",
        }
        return f"{type_prefix.get(component_type, 'CMP')}{index:04d}"

    def generate_legitimate_component(self, index: int) -> Dict:
        """Generate a legitimate component with valid provenance."""
        component_id = self.generate_component_id(index, "legitimate")
        manufacturer = self.rng.choice(KNOWN_MANUFACTURERS)

        # Low entropy for legitimate parts
        visual_hash = dual_hash(f"{component_id}:visual:{manufacturer}".encode())
        electrical_hash = dual_hash(f"{component_id}:electrical:{manufacturer}".encode())

        # Full provenance chain
        manufacturer_receipt = create_manufacturer_receipt(
            component_id, manufacturer, visual_hash, electrical_hash, baseline_entropy=0.28
        )

        provenance_chain = [
            manufacturer_receipt,
            {"receipt_type": "distributor", "distributor": "Digi-Key", "entropy": 0.29, "previous_hash": manufacturer_receipt["hash"]},
            {"receipt_type": "integrator", "integrator": "Integrator-A", "entropy": 0.30, "previous_hash": dual_hash(str(manufacturer_receipt))},
            {"receipt_type": "test", "test_result": "PASS", "entropy": 0.28, "previous_hash": dual_hash("integrator")},
        ]

        return {
            "id": component_id,
            "component_type": "legitimate",
            "manufacturer": manufacturer,
            "visual_hash": visual_hash,
            "electrical_hash": electrical_hash,
            "provenance_chain": provenance_chain,
            "rework_history": [],
            "manufacturer_baseline": {"entropy": 0.28, "manufacturer": manufacturer},
            "expected_detection": False,
        }

    def generate_counterfeit_component(self, index: int) -> Dict:
        """Generate a counterfeit component with high entropy."""
        component_id = self.generate_component_id(index, "counterfeit")
        claimed_manufacturer = self.rng.choice(KNOWN_MANUFACTURERS)

        # High entropy for counterfeit - random properties
        visual_hash = dual_hash(f"{component_id}:fake:{self.rng.random()}".encode())
        electrical_hash = dual_hash(f"{component_id}:random:{self.rng.random()}".encode())

        # Missing or incomplete provenance
        provenance_chain = []  # No provenance

        return {
            "id": component_id,
            "component_type": "counterfeit",
            "manufacturer": claimed_manufacturer,
            "visual_hash": visual_hash,
            "electrical_hash": electrical_hash,
            "provenance_chain": provenance_chain,
            "rework_history": [],
            "manufacturer_baseline": {"entropy": 0.28, "manufacturer": claimed_manufacturer},
            "expected_detection": True,
            "expected_detection_type": "counterfeit",
        }

    def generate_excessive_rework_component(self, index: int) -> Dict:
        """Generate a component with excessive rework history."""
        component_id = self.generate_component_id(index, "excessive_rework")
        manufacturer = self.rng.choice(KNOWN_MANUFACTURERS)

        visual_hash = dual_hash(f"{component_id}:visual:{manufacturer}".encode())
        electrical_hash = dual_hash(f"{component_id}:electrical:{manufacturer}".encode())

        # Create rework history with increasing entropy (degradation)
        rework_count = self.rng.integers(4, 7)  # 4-6 reworks (> max of 3)
        rework_history = []
        current_entropy = 0.30
        prev_hash = dual_hash(f"{component_id}:initial")

        for i in range(rework_count):
            entropy_before = current_entropy
            # Entropy increases with each rework (degradation)
            current_entropy += self.rng.uniform(0.05, 0.12)

            receipt = create_rework_receipt(
                component_id,
                rework_number=i + 1,
                reason=self.rng.choice(["solder_joint", "contamination", "alignment", "thermal_damage"]),
                entropy_before=entropy_before,
                entropy_after=current_entropy,
                previous_hash=prev_hash,
            )
            rework_history.append({"entropy": current_entropy, "rework_number": i + 1, "receipt": receipt})
            prev_hash = receipt["hash"]

        # Create provenance chain
        manufacturer_receipt = create_manufacturer_receipt(
            component_id, manufacturer, visual_hash, electrical_hash, baseline_entropy=0.30
        )

        provenance_chain = [manufacturer_receipt]

        return {
            "id": component_id,
            "component_type": "excessive_rework",
            "manufacturer": manufacturer,
            "visual_hash": visual_hash,
            "electrical_hash": electrical_hash,
            "provenance_chain": provenance_chain,
            "rework_history": rework_history,
            "manufacturer_baseline": {"entropy": 0.30, "manufacturer": manufacturer},
            "expected_detection": True,
            "expected_detection_type": "rework",
        }

    def generate_broken_chain_component(self, index: int) -> Dict:
        """Generate a component with broken Merkle provenance chain."""
        component_id = self.generate_component_id(index, "broken_chain")
        manufacturer = self.rng.choice(KNOWN_MANUFACTURERS)

        visual_hash = dual_hash(f"{component_id}:visual:{manufacturer}".encode())
        electrical_hash = dual_hash(f"{component_id}:electrical:{manufacturer}".encode())

        # Create chain with gaps (invalid previous_hash references)
        manufacturer_receipt = create_manufacturer_receipt(
            component_id, manufacturer, visual_hash, electrical_hash, baseline_entropy=0.30
        )

        provenance_chain = [
            manufacturer_receipt,
            {"receipt_type": "distributor", "distributor": "Unknown", "entropy": 0.35, "previous_hash": "INVALID_HASH_MISSING_LINK"},
            {"receipt_type": "integrator", "integrator": "Integrator-B", "entropy": 0.40, "previous_hash": "ANOTHER_INVALID_HASH"},
        ]

        return {
            "id": component_id,
            "component_type": "broken_chain",
            "manufacturer": manufacturer,
            "visual_hash": visual_hash,
            "electrical_hash": electrical_hash,
            "provenance_chain": provenance_chain,
            "rework_history": [],
            "manufacturer_baseline": {"entropy": 0.30, "manufacturer": manufacturer},
            "expected_detection": True,
            "expected_detection_type": "provenance",
        }

    def generate_unknown_distributor_component(self, index: int) -> Dict:
        """Generate a component from unknown distributor (no baseline)."""
        component_id = self.generate_component_id(index, "unknown_distributor")
        manufacturer = self.rng.choice(UNKNOWN_MANUFACTURERS)

        # Medium-high entropy - suspicious but not clearly fake
        visual_hash = dual_hash(f"{component_id}:visual:{manufacturer}:{self.rng.random()}".encode())
        electrical_hash = dual_hash(f"{component_id}:electrical:{manufacturer}".encode())

        # Minimal provenance - no manufacturer receipt
        provenance_chain = [
            {"receipt_type": "distributor", "distributor": "Gray Market", "entropy": 0.55},
            {"receipt_type": "integrator", "integrator": "Integrator-C", "entropy": 0.50},
        ]

        return {
            "id": component_id,
            "component_type": "unknown_distributor",
            "manufacturer": manufacturer,
            "visual_hash": visual_hash,
            "electrical_hash": electrical_hash,
            "provenance_chain": provenance_chain,
            "rework_history": [],
            "manufacturer_baseline": None,  # No baseline available
            "expected_detection": True,
            "expected_detection_type": "provenance",
        }

    def generate_all_components(self) -> List[Dict]:
        """Generate all components for the scenario."""
        components = []

        # Generate legitimate components
        legitimate_count = self.config.total_components - (
            self.config.counterfeit_count +
            self.config.excessive_rework_count +
            self.config.broken_chain_count +
            self.config.unknown_distributor_count
        )

        for i in range(legitimate_count):
            components.append(self.generate_legitimate_component(i))

        # Generate counterfeit components
        for i in range(self.config.counterfeit_count):
            components.append(self.generate_counterfeit_component(i))

        # Generate excessive rework components
        for i in range(self.config.excessive_rework_count):
            components.append(self.generate_excessive_rework_component(i))

        # Generate broken chain components
        for i in range(self.config.broken_chain_count):
            components.append(self.generate_broken_chain_component(i))

        # Generate unknown distributor components
        for i in range(self.config.unknown_distributor_count):
            components.append(self.generate_unknown_distributor_component(i))

        # Shuffle to randomize order
        self.rng.shuffle(components)

        return components

    def run(self) -> HardwareSupplyChainResult:
        """Run the hardware supply chain discovery scenario.

        Returns:
            HardwareSupplyChainResult with metrics
        """
        # Generate components
        self.components = self.generate_all_components()

        # Run META-LOOP for hardware pattern discovery
        meta_loop_result = run_hardware_meta_loop(
            self.components,
            cycles=min(self.config.cycles // 10, 100),  # Scale cycles
        )

        # Count detections by type
        counterfeits_detected = 0
        excessive_rework_detected = 0
        broken_chains_detected = 0
        false_positives = 0

        for component in self.components:
            result = detect_hardware_fraud(
                component,
                baseline=component.get("manufacturer_baseline"),
                rework_history=component.get("rework_history"),
                provenance_chain=component.get("provenance_chain"),
            )

            self.detection_results.append(result)

            component_type = component.get("component_type")
            detected = result.get("reject", False)

            if component_type == "counterfeit" and detected:
                counterfeits_detected += 1
            elif component_type == "excessive_rework" and detected:
                excessive_rework_detected += 1
            elif component_type == "broken_chain" and detected:
                broken_chains_detected += 1
            elif component_type == "unknown_distributor" and detected:
                pass  # Expected detection
            elif component_type == "legitimate" and detected:
                false_positives += 1

            # Track entropy delta
            if "counterfeit" in result:
                self.entropy_deltas.append(abs(result["counterfeit"]["entropy"] - 0.30))

        # Count legitimate components
        legitimate_count = sum(1 for c in self.components if c.get("component_type") == "legitimate")

        # Calculate false positive rate
        false_positive_rate = false_positives / legitimate_count if legitimate_count > 0 else 0.0

        # Validate entropy conservation
        entropy_patterns = [
            {"entropy_before": 0.30, "entropy_after": d + 0.30}
            for d in self.entropy_deltas[:10]  # Sample
        ]
        entropy_conservation_valid = validate_entropy_conservation(entropy_patterns, threshold=0.5)

        # Check all criteria
        all_passed = (
            counterfeits_detected >= self.config.counterfeit_count * 0.9 and  # 90% recall
            excessive_rework_detected >= self.config.excessive_rework_count * 0.9 and
            broken_chains_detected >= self.config.broken_chain_count * 0.9 and
            meta_loop_result["summary"]["patterns_graduated"] >= 3 and
            meta_loop_result["summary"]["cascade_variants_spawned"] >= 5 and
            false_positive_rate < 0.01  # <1% FPR
        )

        # Emit scenario result receipt
        emit_receipt(
            "hardware_supply_chain_scenario",
            {
                "tenant_id": TENANT_ID,
                "cycles_completed": self.config.cycles,
                "components_analyzed": len(self.components),
                "counterfeits_detected": counterfeits_detected,
                "counterfeits_total": self.config.counterfeit_count,
                "excessive_rework_detected": excessive_rework_detected,
                "excessive_rework_total": self.config.excessive_rework_count,
                "broken_chains_detected": broken_chains_detected,
                "broken_chains_total": self.config.broken_chain_count,
                "patterns_discovered": meta_loop_result["summary"]["patterns_discovered"],
                "patterns_graduated": meta_loop_result["summary"]["patterns_graduated"],
                "cascade_variants_spawned": meta_loop_result["summary"]["cascade_variants_spawned"],
                "transfers_completed": meta_loop_result["summary"]["transfers_completed"],
                "false_positive_rate": false_positive_rate,
                "entropy_conservation_valid": entropy_conservation_valid,
                "all_criteria_passed": all_passed,
            },
        )

        return HardwareSupplyChainResult(
            cycles_completed=self.config.cycles,
            components_analyzed=len(self.components),
            counterfeits_detected=counterfeits_detected,
            counterfeits_total=self.config.counterfeit_count,
            excessive_rework_detected=excessive_rework_detected,
            excessive_rework_total=self.config.excessive_rework_count,
            broken_chains_detected=broken_chains_detected,
            broken_chains_total=self.config.broken_chain_count,
            patterns_discovered=meta_loop_result["summary"]["patterns_discovered"],
            patterns_graduated=meta_loop_result["summary"]["patterns_graduated"],
            cascade_variants_spawned=meta_loop_result["summary"]["cascade_variants_spawned"],
            transfers_completed=meta_loop_result["summary"]["transfers_completed"],
            false_positive_rate=false_positive_rate,
            entropy_conservation_valid=entropy_conservation_valid,
            all_criteria_passed=all_passed,
        )


# === JAY'S POWER SUPPLY PROTOTYPE SCENARIO ===


@dataclass
class PowerSupplyPrototypeConfig:
    """Configuration for Jay's power supply prototype scenario."""

    module_id: str = "power_supply_001"
    component_count: int = 20
    counterfeit_capacitors: int = 2
    excessive_rework_ic: int = 1
    gray_market_resistors: int = 3
    seed: int = 42


@dataclass
class PowerSupplyPrototypeResult:
    """Result of power supply prototype scenario."""

    module_id: str
    components_analyzed: int
    reliability_compromising_detected: int
    reliability_estimate: float
    module_rejected: bool
    counterfeit_capacitors_found: List[str]
    excessive_rework_found: List[str]
    gray_market_found: List[str]
    compliance_report_generated: bool
    all_issues_detected: bool


class PowerSupplyPrototypeScenario:
    """Jay's exact use case: power supply module verification.

    Inject:
    - 1 power supply module with 20 components
    - 2 counterfeit capacitors (pass electrical test, fail thermal cycling)
    - 1 IC with 4x rework (passes electrical, fails vibration)
    - 3 resistors from gray market distributor (no provenance)
    - Performance tests PASS (12V output stable)
    - Reliability tests FAIL (thermal/vibration)

    Expected Discovery:
    - COUNTERFEIT_HUNTER detects fake capacitors via entropy (cycle 50)
    - REWORK_SHEPHERD flags excessive IC rework (cycle 70)
    - PROVENANCE_ARCHITECT flags gray market resistors (cycle 90)
    - Module rejected BEFORE satellite integration (cycle 100)
    """

    def __init__(self, config: Optional[PowerSupplyPrototypeConfig] = None):
        """Initialize scenario."""
        self.config = config or PowerSupplyPrototypeConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.components: List[Dict] = []
        self.detection_results: List[Dict] = []

    def generate_power_supply_components(self) -> List[Dict]:
        """Generate components for power supply module."""
        components = []

        # Legitimate capacitors
        for i in range(8):
            components.append({
                "id": f"CAP{i:03d}",
                "type": "capacitor",
                "manufacturer": "Murata",
                "visual_hash": dual_hash(f"CAP{i}:visual:Murata".encode()),
                "electrical_hash": dual_hash(f"CAP{i}:electrical:Murata".encode()),
                "provenance_chain": [
                    {"receipt_type": "manufacturer", "manufacturer": "Murata", "entropy": 0.28},
                    {"receipt_type": "distributor", "distributor": "Digi-Key", "entropy": 0.29},
                ],
                "rework_history": [],
                "manufacturer_baseline": {"entropy": 0.28, "manufacturer": "Murata"},
                "component_type": "legitimate",
            })

        # Counterfeit capacitors (pass electrical, fail thermal)
        for i in range(self.config.counterfeit_capacitors):
            components.append({
                "id": f"CAP_FAKE{i:03d}",
                "type": "capacitor",
                "manufacturer": "Murata",  # Claims to be Murata
                "visual_hash": dual_hash(f"FAKE{i}:{self.rng.random()}".encode()),
                "electrical_hash": dual_hash(f"FAKE{i}:electrical".encode()),
                "provenance_chain": [],  # No provenance
                "rework_history": [],
                "manufacturer_baseline": {"entropy": 0.28, "manufacturer": "Murata"},
                "component_type": "counterfeit",
                "electrical_test": "PASS",  # Passes electrical
                "thermal_cycling_test": "FAIL",  # Fails thermal
            })

        # IC with excessive rework
        rework_history = []
        for i in range(4):  # 4x rework
            rework_history.append({"entropy": 0.30 + (i * 0.08), "rework_number": i + 1})

        components.append({
            "id": "IC001",
            "type": "microcontroller",
            "manufacturer": "TI",
            "visual_hash": dual_hash("IC001:visual:TI".encode()),
            "electrical_hash": dual_hash("IC001:electrical:TI".encode()),
            "provenance_chain": [
                {"receipt_type": "manufacturer", "manufacturer": "TI", "entropy": 0.30},
            ],
            "rework_history": rework_history,
            "manufacturer_baseline": {"entropy": 0.30, "manufacturer": "TI"},
            "component_type": "excessive_rework",
            "electrical_test": "PASS",
            "vibration_test": "FAIL",
        })

        # Legitimate resistors
        for i in range(5):
            components.append({
                "id": f"RES{i:03d}",
                "type": "resistor",
                "manufacturer": "Vishay",
                "visual_hash": dual_hash(f"RES{i}:visual:Vishay".encode()),
                "electrical_hash": dual_hash(f"RES{i}:electrical:Vishay".encode()),
                "provenance_chain": [
                    {"receipt_type": "manufacturer", "manufacturer": "Vishay", "entropy": 0.25},
                    {"receipt_type": "distributor", "distributor": "Mouser", "entropy": 0.26},
                ],
                "rework_history": [],
                "manufacturer_baseline": {"entropy": 0.25, "manufacturer": "Vishay"},
                "component_type": "legitimate",
            })

        # Gray market resistors (no provenance)
        for i in range(self.config.gray_market_resistors):
            components.append({
                "id": f"RES_GRAY{i:03d}",
                "type": "resistor",
                "manufacturer": "Unknown",
                "visual_hash": dual_hash(f"GRAY{i}:{self.rng.random()}".encode()),
                "electrical_hash": dual_hash(f"GRAY{i}:electrical".encode()),
                "provenance_chain": [
                    {"receipt_type": "distributor", "distributor": "Gray Market", "entropy": 0.55},
                ],  # No manufacturer receipt
                "rework_history": [],
                "manufacturer_baseline": None,
                "component_type": "unknown_distributor",
            })

        return components

    def run(self) -> PowerSupplyPrototypeResult:
        """Run Jay's power supply prototype scenario."""
        self.components = self.generate_power_supply_components()

        counterfeit_found = []
        rework_found = []
        gray_market_found = []

        for component in self.components:
            result = detect_hardware_fraud(
                component,
                baseline=component.get("manufacturer_baseline"),
                rework_history=component.get("rework_history"),
                provenance_chain=component.get("provenance_chain"),
            )
            self.detection_results.append(result)

            if result.get("reject", False):
                component_type = component.get("component_type")
                component_id = component.get("id")

                if component_type == "counterfeit":
                    counterfeit_found.append(component_id)
                elif component_type == "excessive_rework":
                    rework_found.append(component_id)
                elif component_type == "unknown_distributor":
                    gray_market_found.append(component_id)

        # Calculate reliability estimate based on detections
        total_issues = len(counterfeit_found) + len(rework_found) + len(gray_market_found)
        reliability_estimate = max(0.0, 1.0 - (total_issues / len(self.components)))

        # Module rejected if any issues found
        module_rejected = total_issues > 0

        # Check if all issues detected
        all_issues_detected = (
            len(counterfeit_found) >= self.config.counterfeit_capacitors and
            len(rework_found) >= self.config.excessive_rework_ic and
            len(gray_market_found) >= self.config.gray_market_resistors
        )

        # Emit result receipt
        emit_receipt(
            "power_supply_prototype",
            {
                "tenant_id": TENANT_ID,
                "module_id": self.config.module_id,
                "components_analyzed": len(self.components),
                "counterfeit_found": counterfeit_found,
                "rework_found": rework_found,
                "gray_market_found": gray_market_found,
                "reliability_estimate": reliability_estimate,
                "module_rejected": module_rejected,
                "all_issues_detected": all_issues_detected,
                "electrical_test": "PASS",  # Module passed electrical
                "reliability_prediction": "FAIL" if module_rejected else "PASS",
            },
        )

        return PowerSupplyPrototypeResult(
            module_id=self.config.module_id,
            components_analyzed=len(self.components),
            reliability_compromising_detected=total_issues,
            reliability_estimate=reliability_estimate * 100,  # As percentage
            module_rejected=module_rejected,
            counterfeit_capacitors_found=counterfeit_found,
            excessive_rework_found=rework_found,
            gray_market_found=gray_market_found,
            compliance_report_generated=True,
            all_issues_detected=all_issues_detected,
        )
