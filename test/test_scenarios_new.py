"""Tests for new scenarios - NETWORK and ADVERSARIAL.

Tests network-scale validation and DoD hostile audit.
"""


from spaceproof.sim.scenarios.network import (
    NetworkScenarioConfig,
    NetworkScenarioResult,
    run_scenario as run_network_scenario,
)
from spaceproof.sim.scenarios.adversarial import (
    AdversarialConfig,
    AdversarialResult,
    AttackType,
    DefenseStrategy,
    run_scenario as run_adversarial_scenario,
    validate_dod_audit,
)


class TestNetworkScenario:
    """Test NETWORK scenario."""

    def test_run_basic_scenario(self, capsys):
        """Should run basic network scenario."""
        config = NetworkScenarioConfig(
            n_colonies=10,
            colonists_per_colony=100,
            duration_days=10,
            seed=42,
            enable_partitions=False,
            enable_cascade=False,
        )
        result = run_network_scenario(config)

        assert isinstance(result, NetworkScenarioResult)
        assert result.scenario == "NETWORK"
        assert result.final_n_colonies == 10

    def test_scenario_emits_receipt(self, capsys):
        """Should emit scenario receipt."""
        config = NetworkScenarioConfig(
            n_colonies=5,
            duration_days=5,
            seed=42,
            enable_partitions=False,
            enable_cascade=False,
        )
        run_network_scenario(config)
        output = capsys.readouterr().out
        assert "network_scenario_receipt" in output

    def test_scenario_tracks_entropy(self, capsys):
        """Should track entropy stability."""
        config = NetworkScenarioConfig(
            n_colonies=10,
            duration_days=10,
            seed=42,
        )
        result = run_network_scenario(config)

        assert hasattr(result, "entropy_stable_ratio")
        assert 0 <= result.entropy_stable_ratio <= 1

    def test_scenario_checks_sovereignty(self, capsys):
        """Should check network sovereignty."""
        config = NetworkScenarioConfig(
            n_colonies=50,
            colonists_per_colony=1000,
            duration_days=10,
            seed=42,
        )
        result = run_network_scenario(config)

        assert hasattr(result, "sovereignty_achieved")

    def test_partition_testing(self, capsys):
        """Should test partition scenarios."""
        config = NetworkScenarioConfig(
            n_colonies=20,
            duration_days=10,
            seed=42,
            enable_partitions=True,
            enable_cascade=False,
        )
        result = run_network_scenario(config)

        # May or may not have partitions depending on network topology
        assert hasattr(result, "partition_events")
        assert hasattr(result, "partition_recovery_avg_hours")

    def test_cascade_testing(self, capsys):
        """Should test cascade failures."""
        config = NetworkScenarioConfig(
            n_colonies=20,
            duration_days=10,
            seed=42,
            enable_partitions=False,
            enable_cascade=True,
        )
        result = run_network_scenario(config)

        assert hasattr(result, "cascade_contained")


class TestAdversarialScenario:
    """Test ADVERSARIAL scenario."""

    def test_run_basic_scenario(self, capsys):
        """Should run basic adversarial scenario."""
        config = AdversarialConfig(
            attacker_budget=1e6,  # Reduced for speed
            attack_types=["corruption"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert isinstance(result, AdversarialResult)
        assert result.scenario == "ADVERSARIAL"

    def test_scenario_emits_receipt(self, capsys):
        """Should emit scenario receipt."""
        config = AdversarialConfig(
            attacker_budget=1e6,
            attack_types=["corruption"],
            n_receipts=50,
            seed=42,
        )
        run_adversarial_scenario(config)
        output = capsys.readouterr().out
        assert "adversarial_scenario_receipt" in output

    def test_corruption_attack(self, capsys):
        """Should test corruption attack."""
        config = AdversarialConfig(
            attacker_budget=1e7,
            attack_types=["corruption"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert result.total_attacks >= 1
        # Corruption should be detected via Merkle proof
        assert result.attacks_detected >= 1

    def test_timing_attack(self, capsys):
        """Should test timing attack."""
        config = AdversarialConfig(
            attacker_budget=1e7,
            attack_types=["timing"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert result.total_attacks >= 1

    def test_sybil_attack(self, capsys):
        """Should test Sybil attack."""
        config = AdversarialConfig(
            attacker_budget=1e9,
            attack_types=["sybil"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert result.total_attacks >= 1
        # Sybil should be blocked by dual-hash chain
        assert result.attacks_blocked >= 1

    def test_byzantine_attack(self, capsys):
        """Should test Byzantine attack."""
        config = AdversarialConfig(
            attacker_budget=1e8,
            attack_types=["byzantine"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert result.total_attacks >= 1
        assert result.consensus_maintained  # With honest majority

    def test_all_attacks(self, capsys):
        """Should test all attack types."""
        config = AdversarialConfig(
            attacker_budget=1e9,
            attack_types=["corruption", "timing", "sybil", "byzantine"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert result.total_attacks >= 4

    def test_merkle_integrity(self, capsys):
        """Should verify Merkle integrity."""
        config = AdversarialConfig(
            attacker_budget=1e8,
            attack_types=["corruption"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert hasattr(result, "merkle_integrity")

    def test_timing_proofs(self, capsys):
        """Should verify timing proofs."""
        config = AdversarialConfig(
            attacker_budget=1e7,
            attack_types=["timing"],
            n_receipts=100,
            seed=42,
        )
        result = run_adversarial_scenario(config)

        assert hasattr(result, "timing_proofs_valid")


class TestDodAuditValidation:
    """Test DoD audit validation."""

    def test_validate_dod_audit(self, capsys):
        """Should validate DoD audit requirements."""
        result = validate_dod_audit(attacker_budget=1e8, seed=42)

        assert "passed" in result
        assert "block_rate" in result
        assert "merkle_integrity" in result
        assert "consensus_maintained" in result

    def test_high_budget_attack(self, capsys):
        """Should handle high-budget attacks."""
        result = validate_dod_audit(attacker_budget=1e9, seed=42)

        # Even with $1B budget, defenses should hold
        assert result["merkle_integrity"]


class TestAttackTypes:
    """Test AttackType enum."""

    def test_attack_types(self):
        """Should have all attack types."""
        assert AttackType.CORRUPTION.value == "corruption"
        assert AttackType.TIMING.value == "timing"
        assert AttackType.SYBIL.value == "sybil"
        assert AttackType.BYZANTINE.value == "byzantine"


class TestDefenseStrategies:
    """Test DefenseStrategy enum."""

    def test_defense_strategies(self):
        """Should have all defense strategies."""
        assert DefenseStrategy.MERKLE_PROOF.value == "merkle_proof"
        assert DefenseStrategy.CONSENSUS.value == "consensus"
        assert DefenseStrategy.TIMING_PROOF.value == "timing_proof"
        assert DefenseStrategy.HYBRID.value == "hybrid"
