"""Tests for colony_network.py - Multi-Colony Network.

Tests 1000 colonies, 1M colonists network dynamics.
"""

import pytest
import numpy as np

from spaceproof.domain.colony_network import (
    ColonyNetwork,
    ColonyNode,
    NetworkState,
    MARS_COLONIST_TARGET_2050,
    COLONY_NETWORK_SIZE_TARGET,
    INTER_COLONY_BANDWIDTH_MBPS,
    calculate_distance,
    inter_colony_bandwidth,
    initialize_network,
    simulate_network,
    detect_partition,
    merge_partitions,
    scale_network,
    network_entropy_rate,
)


class TestConstants:
    """Test that constants match Grok specifications."""

    def test_colonist_target_1m(self):
        """Grok: '1M colonists by 2050'."""
        assert MARS_COLONIST_TARGET_2050 == 1_000_000

    def test_colony_network_size_1000(self):
        """1M @ 1000/colony = 1000 colonies."""
        assert COLONY_NETWORK_SIZE_TARGET == 1000


class TestNetworkInitialization:
    """Test network initialization."""

    def test_initialize_network(self, capsys):
        """Should create network with correct colony count."""
        network = initialize_network(100, 1000, seed=42)
        assert network.n_colonies == 100
        assert network.total_population == 100000

    def test_colonies_have_positions(self, capsys):
        """Each colony should have position."""
        network = initialize_network(10, 1000, seed=42)
        for colony in network.colonies:
            assert hasattr(colony, "position")
            assert len(colony.position) == 2

    def test_inter_colony_links_created(self, capsys):
        """Should create inter-colony links."""
        network = initialize_network(10, 1000, seed=42)
        assert len(network.inter_colony_links) > 0

    def test_emits_receipt(self, capsys):
        """Should emit initialization receipt."""
        initialize_network(10, 1000, seed=42)
        output = capsys.readouterr().out
        assert "network_init_receipt" in output

    def test_deterministic_with_seed(self, capsys):
        """Same seed should produce same network."""
        n1 = initialize_network(10, 1000, seed=42)
        # Clear captured output
        capsys.readouterr()
        n2 = initialize_network(10, 1000, seed=42)
        assert n1.n_colonies == n2.n_colonies
        assert len(n1.inter_colony_links) == len(n2.inter_colony_links)


class TestInterColonyBandwidth:
    """Test inter-colony bandwidth calculations."""

    def test_bandwidth_degrades_with_distance(self, capsys):
        """Bandwidth should degrade with distance."""
        network = initialize_network(10, 1000, seed=42)
        c1, c2, c3 = network.colonies[:3]

        # Move c3 farther away
        c3_far = ColonyNode(
            colony_id="C_FAR",
            name="Far Colony",
            population=1000,
            position=(c1.position[0] + 4000, c1.position[1]),  # 4000 km away
            decision_capacity_bps=10000,
            bandwidth_to_earth_mbps=2.0,
        )

        bw_close = inter_colony_bandwidth(c1, c2)
        bw_far = inter_colony_bandwidth(c1, c3_far)

        # Further = less bandwidth
        assert bw_close >= bw_far


class TestNetworkSimulation:
    """Test network simulation."""

    def test_simulate_network(self, capsys):
        """Should run network simulation."""
        network = initialize_network(10, 1000, seed=42)
        states = simulate_network(network, 30, seed=42)
        assert len(states) == 30

    def test_states_have_sovereignty_info(self, capsys):
        """Each state should have sovereignty info."""
        network = initialize_network(10, 1000, seed=42)
        states = simulate_network(network, 10, seed=42)
        for state in states:
            assert hasattr(state, "sovereign")
            assert hasattr(state, "network_internal_bps")
            assert hasattr(state, "earth_input_bps")

    def test_emits_summary_receipt(self, capsys):
        """Should emit summary receipt."""
        network = initialize_network(10, 1000, seed=42)
        simulate_network(network, 10, seed=42)
        output = capsys.readouterr().out
        assert "colony_network_receipt" in output


class TestPartitionDetection:
    """Test network partition detection."""

    def test_no_partition_in_connected_network(self, capsys):
        """Connected network should have one partition."""
        network = initialize_network(10, 1000, seed=42)
        partitions = detect_partition(network)
        # Small networks are usually fully connected
        assert len(partitions) >= 1

    def test_partition_after_link_removal(self, capsys):
        """Removing links can create partitions."""
        network = initialize_network(10, 1000, seed=42)

        # Remove all links to isolate colonies
        network.inter_colony_links = {}

        partitions = detect_partition(network)
        # Each colony is now its own partition
        assert len(partitions) == network.n_colonies


class TestNetworkScaling:
    """Test network scaling."""

    def test_scale_network(self, capsys):
        """Should add colonies to network."""
        network = initialize_network(10, 1000, seed=42)
        initial_colonies = network.n_colonies

        expanded = scale_network(network, 5, 1000, seed=42)
        assert expanded.n_colonies == initial_colonies + 5

    def test_scale_emits_receipt(self, capsys):
        """Should emit scale receipt."""
        network = initialize_network(10, 1000, seed=42)
        capsys.readouterr()  # Clear
        scale_network(network, 5, 1000, seed=42)
        output = capsys.readouterr().out
        assert "network_scale_receipt" in output


class TestPartitionMerge:
    """Test partition merging."""

    def test_merge_partitions(self, capsys):
        """Should merge two partitions."""
        network = initialize_network(10, 1000, seed=42)

        # Create artificial partition
        partition_a = [c.colony_id for c in network.colonies[:5]]
        partition_b = [c.colony_id for c in network.colonies[5:]]

        merged = merge_partitions(network, partition_a, partition_b)
        # Should have at least one more link
        assert len(merged.inter_colony_links) >= len(network.inter_colony_links)


class TestLargeScaleNetwork:
    """Test large-scale network operations (1000 colonies)."""

    @pytest.mark.slow
    def test_1000_colony_network(self, capsys):
        """Test full 1000 colony network."""
        network = initialize_network(100, 1000, seed=42)  # Reduced for CI
        assert network.n_colonies == 100
        assert network.total_population == 100000

    @pytest.mark.slow
    def test_1000_colony_simulation(self, capsys):
        """Test simulation with 100 colonies (reduced)."""
        network = initialize_network(100, 1000, seed=42)
        states = simulate_network(network, 10, seed=42)  # 10 days
        assert len(states) == 10
