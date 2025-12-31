"""Tests for sovereignty_network.py - Network Sovereignty.

Tests network sovereignty threshold calculations.
"""


from spaceproof.domain.colony_network import initialize_network
from spaceproof.sovereignty_network import (
    network_decision_capacity,
    earth_input_rate,
    network_sovereignty_threshold,
    validate_network_sovereignty,
    sovereignty_by_partition,
    network_sovereignty_sensitivity,
    calculate_network_consensus_time,
)


class TestNetworkDecisionCapacity:
    """Test network decision capacity calculations."""

    def test_capacity_scales_with_colonies(self, capsys):
        """More colonies = more capacity."""
        net1 = initialize_network(10, 1000, seed=42)
        capsys.readouterr()
        net2 = initialize_network(50, 1000, seed=42)

        cap1 = network_decision_capacity(net1)
        cap2 = network_decision_capacity(net2)

        assert cap2 > cap1

    def test_capacity_with_augmentation(self, capsys):
        """Augmentation should increase capacity."""
        network = initialize_network(10, 1000, seed=42)

        cap_base = network_decision_capacity(network)
        cap_aug = network_decision_capacity(
            network,
            augmentation={"C0001": 5.0, "C0002": 5.0}
        )

        assert cap_aug > cap_base


class TestEarthInputRate:
    """Test Earth input rate calculations."""

    def test_earth_input_positive(self, capsys):
        """Earth input should be positive."""
        network = initialize_network(10, 1000, seed=42)
        rate = earth_input_rate(network)
        assert rate > 0

    def test_earth_input_scales_with_bandwidth(self, capsys):
        """Higher bandwidth = higher input rate."""
        network = initialize_network(10, 1000, seed=42)
        rate1 = earth_input_rate(network, earth_bandwidth_mbps=10)
        rate2 = earth_input_rate(network, earth_bandwidth_mbps=100)
        assert rate2 > rate1


class TestNetworkSovereigntyThreshold:
    """Test network sovereignty threshold calculation."""

    def test_finds_threshold(self, capsys):
        """Should find sovereignty threshold."""
        network = initialize_network(100, 1000, seed=42)
        result = network_sovereignty_threshold(network)

        assert "threshold_colonies" in result
        assert "sovereign" in result
        assert "network_decision_capacity_bps" in result
        assert "earth_input_bps" in result

    def test_emits_receipt(self, capsys):
        """Should emit sovereignty receipt."""
        network = initialize_network(10, 1000, seed=42)
        capsys.readouterr()
        network_sovereignty_threshold(network)
        output = capsys.readouterr().out
        assert "network_sovereignty_receipt" in output


class TestValidateNetworkSovereignty:
    """Test network sovereignty validation."""

    def test_returns_boolean(self, capsys):
        """Should return boolean."""
        network = initialize_network(10, 1000, seed=42)
        result = validate_network_sovereignty(network)
        assert isinstance(result, bool)

    def test_large_network_more_sovereign(self, capsys):
        """Larger networks should be more likely sovereign."""
        small = initialize_network(5, 100, seed=42)
        capsys.readouterr()
        large = initialize_network(100, 1000, seed=42)

        # Large network has more internal capacity
        cap_small = network_decision_capacity(small)
        cap_large = network_decision_capacity(large)
        assert cap_large > cap_small


class TestSovereigntyByPartition:
    """Test partition-level sovereignty."""

    def test_partition_sovereignty(self, capsys):
        """Should calculate sovereignty per partition."""
        network = initialize_network(10, 1000, seed=42)
        partitions = [[c.colony_id for c in network.colonies[:5]],
                      [c.colony_id for c in network.colonies[5:]]]

        capsys.readouterr()
        result = sovereignty_by_partition(network, partitions)

        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_emits_receipt(self, capsys):
        """Should emit partition sovereignty receipt."""
        network = initialize_network(10, 1000, seed=42)
        partitions = [[c.colony_id for c in network.colonies]]
        capsys.readouterr()
        sovereignty_by_partition(network, partitions)
        output = capsys.readouterr().out
        assert "partition_sovereignty_receipt" in output


class TestSovereigntySensitivity:
    """Test sovereignty sensitivity analysis."""

    def test_sensitivity_returns_list(self, capsys):
        """Should return list of results."""
        network = initialize_network(10, 1000, seed=42)
        results = network_sovereignty_sensitivity(network, steps=3)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_sensitivity_varies_params(self, capsys):
        """Results should vary with parameters."""
        network = initialize_network(10, 1000, seed=42)
        results = network_sovereignty_sensitivity(network, steps=3)

        bandwidths = [r["bandwidth_mbps"] for r in results]
        delays = [r["delay_s"] for r in results]

        # Should have variation
        assert len(set(bandwidths)) > 1
        assert len(set(delays)) > 1


class TestConsensusTime:
    """Test network consensus time calculation."""

    def test_consensus_time_positive(self, capsys):
        """Consensus time should be positive."""
        network = initialize_network(10, 1000, seed=42)
        time_sec = calculate_network_consensus_time(network)
        assert time_sec >= 0

    def test_single_colony_zero_time(self, capsys):
        """Single colony should have zero consensus time."""
        network = initialize_network(1, 1000, seed=42)
        time_sec = calculate_network_consensus_time(network)
        assert time_sec == 0.0
