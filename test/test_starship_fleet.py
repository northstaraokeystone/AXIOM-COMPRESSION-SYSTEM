"""Tests for starship_fleet.py - Starship Fleet Model.

Tests 1000+ launches/year with entropy accounting.
"""

import pytest
import numpy as np

from spaceproof.domain.starship_fleet import (
    FleetConfig,
    FleetState,
    StarshipLaunch,
    STARSHIP_PAYLOAD_KG,
    STARSHIP_FLIGHTS_PER_YEAR,
    calculate_entropy_delivered,
    generate_launch_windows,
    simulate_fleet,
    fleet_to_colony_entropy,
    calculate_fleet_bandwidth,
)


class TestConstants:
    """Test that constants match Grok specifications."""

    def test_starship_payload_500t(self):
        """Grok: '500t payload'."""
        assert STARSHIP_PAYLOAD_KG == 500000

    def test_starship_flights_1000_per_year(self):
        """Grok: '1000 flights/year target'."""
        assert STARSHIP_FLIGHTS_PER_YEAR == 1000


class TestEntropyDelivered:
    """Test entropy delivery calculations."""

    def test_entropy_delivered_positive(self):
        """Entropy delivered should be positive."""
        entropy = calculate_entropy_delivered(500000, "mars_surface")
        assert entropy > 0

    def test_entropy_delivered_scales_with_payload(self):
        """More payload = more entropy."""
        e1 = calculate_entropy_delivered(100000, "mars_surface")
        e2 = calculate_entropy_delivered(500000, "mars_surface")
        assert e2 > e1

    def test_entropy_delivered_mars_surface_vs_orbit(self):
        """Surface delivery has different ambient entropy."""
        surface = calculate_entropy_delivered(100000, "mars_surface")
        orbit = calculate_entropy_delivered(100000, "mars_orbit")
        # Different ambient entropy = different delivered
        assert surface != orbit

    def test_entropy_delivered_never_negative(self):
        """Thermodynamic law: cannot deliver negative entropy."""
        entropy = calculate_entropy_delivered(0, "mars_surface")
        assert entropy >= 0


class TestLaunchWindows:
    """Test launch window generation."""

    def test_generate_correct_count(self):
        """Should generate requested number of windows."""
        windows = generate_launch_windows(2050, 100, seed=42)
        assert len(windows) == 100

    def test_windows_have_timestamps(self):
        """Each window should have timestamp."""
        windows = generate_launch_windows(2050, 10, seed=42)
        for w in windows:
            assert "timestamp" in w
            assert "window_id" in w

    def test_deterministic_with_seed(self):
        """Same seed should produce same windows."""
        w1 = generate_launch_windows(2050, 10, seed=42)
        w2 = generate_launch_windows(2050, 10, seed=42)
        assert w1 == w2


class TestFleetSimulation:
    """Test full fleet simulation."""

    def test_simulate_one_year(self, capsys):
        """Simulate one year with default config."""
        config = FleetConfig(n_starships=100)  # Reduced for speed
        states = simulate_fleet(config, duration_years=1, seed=42)
        assert len(states) == 1
        assert states[0].launches_this_year == 100

    def test_simulate_emits_receipts(self, capsys):
        """Simulation should emit receipts."""
        config = FleetConfig(n_starships=10)
        simulate_fleet(config, duration_years=1, seed=42)
        output = capsys.readouterr().out
        assert "starship_launch_receipt" in output
        assert "fleet_state_receipt" in output

    def test_fleet_state_has_required_fields(self):
        """FleetState should have all required fields."""
        config = FleetConfig(n_starships=10)
        states = simulate_fleet(config, duration_years=1, seed=42)
        state = states[0]
        assert hasattr(state, "launches_this_year")
        assert hasattr(state, "total_payload_delivered_kg")
        assert hasattr(state, "total_entropy_delivered")
        assert hasattr(state, "status")

    def test_cumulative_payload_increases(self):
        """Cumulative payload should increase over years."""
        config = FleetConfig(n_starships=50)
        states = simulate_fleet(config, duration_years=3, seed=42)
        for i in range(1, len(states)):
            assert states[i].total_payload_delivered_kg >= states[i-1].total_payload_delivered_kg

    def test_1000_launches_per_year(self, capsys):
        """Test full 1000 launches/year target."""
        config = FleetConfig(n_starships=STARSHIP_FLIGHTS_PER_YEAR)
        states = simulate_fleet(config, duration_years=1, seed=42)
        assert states[0].launches_this_year == 1000


class TestFleetToColony:
    """Test fleet to colony entropy distribution."""

    def test_distribute_to_colonies(self, capsys):
        """Should distribute entropy to colonies."""
        config = FleetConfig(n_starships=10)
        states = simulate_fleet(config, duration_years=1, seed=42)
        colony_ids = ["C0001", "C0002", "C0003"]
        distribution = fleet_to_colony_entropy(states[0], colony_ids)
        assert len(distribution) == 3
        assert all(v >= 0 for v in distribution.values())

    def test_empty_colony_list(self, capsys):
        """Empty colony list should return empty dict."""
        config = FleetConfig(n_starships=10)
        states = simulate_fleet(config, duration_years=1, seed=42)
        distribution = fleet_to_colony_entropy(states[0], [])
        assert distribution == {}


class TestFleetBandwidth:
    """Test fleet bandwidth calculation."""

    def test_bandwidth_from_entropy(self, capsys):
        """Should calculate bandwidth from entropy."""
        config = FleetConfig(n_starships=100)
        states = simulate_fleet(config, duration_years=1, seed=42)
        bandwidth = calculate_fleet_bandwidth(states[0])
        assert bandwidth > 0

    def test_bandwidth_scales_with_fleet(self, capsys):
        """More Starships = more bandwidth."""
        config1 = FleetConfig(n_starships=50)
        config2 = FleetConfig(n_starships=100)
        states1 = simulate_fleet(config1, duration_years=1, seed=42)
        states2 = simulate_fleet(config2, duration_years=1, seed=42)
        bw1 = calculate_fleet_bandwidth(states1[0])
        bw2 = calculate_fleet_bandwidth(states2[0])
        assert bw2 > bw1
