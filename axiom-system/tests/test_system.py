"""AXIOM-SYSTEM v2 Tests - System validation."""

import sys
sys.path.insert(0, '..')

import pytest


class TestCore:
    """Test core module."""

    def test_dual_hash(self):
        """Test dual hash format."""
        from src.core import dual_hash
        h = dual_hash("test")
        assert ":" in h, "Dual hash should contain colon"
        parts = h.split(":")
        assert len(parts) == 2, "Should have two hash parts"
        assert len(parts[0]) == 64, "SHA256 should be 64 hex chars"
        assert len(parts[1]) == 64, "BLAKE3 should be 64 hex chars"

    def test_emit_receipt(self):
        """Test receipt emission."""
        from src.core import emit_receipt
        r = emit_receipt("test", {"value": 42})
        assert r["receipt_type"] == "test"
        assert r["tenant_id"] == "axiom-system"
        assert "payload_hash" in r
        assert "ts" in r


class TestEntropy:
    """Test entropy module constants and functions."""

    def test_neuralink_multiplier(self):
        """Test Neuralink multiplier constant."""
        from src.entropy import NEURALINK_MULTIPLIER
        assert NEURALINK_MULTIPLIER == 1e5, "Should be 100,000x"

    def test_mdl_beta(self):
        """Test MDL beta constant."""
        from src.entropy import MDL_BETA
        assert MDL_BETA == 0.09, "Should be 0.09 for 96% compression"

    def test_sovereignty_threshold(self):
        """Test sovereignty thresholds."""
        from src.entropy import sovereignty_threshold
        assert sovereignty_threshold(False) == 25, "Baseline should be 25"
        assert sovereignty_threshold(True) == 5, "Neuralink should be 5"

    def test_kessler_threshold(self):
        """Test Kessler threshold constant."""
        from src.entropy import KESSLER_THRESHOLD
        assert KESSLER_THRESHOLD == 0.73, "Should be 73%"

    def test_internal_compression_rate(self):
        """Test internal compression rate with Neuralink."""
        from src.entropy import internal_compression_rate
        rate_base = internal_compression_rate(10, 0.8, 1e15, 0)
        rate_neuralink = internal_compression_rate(10, 0.8, 1e15, 1.0)
        assert rate_neuralink > rate_base * 100, "Neuralink should boost rate significantly"


class TestNetwork:
    """Test network module."""

    def test_create_relay_graph(self):
        """Test relay graph creation."""
        from src.network import create_relay_graph
        g = create_relay_graph(["moon", "mars"])
        assert "earth" in g.nodes()
        assert "moon" in g.nodes()
        assert "mars" in g.nodes()
        assert g.has_edge("earth", "moon")
        assert g.has_edge("earth", "mars")

    def test_add_relay(self):
        """Test adding relay edge."""
        from src.network import create_relay_graph, add_relay, shortest_relay_path
        g = create_relay_graph(["moon", "mars"])
        g = add_relay(g, "moon", "mars", 0.85)
        assert g.has_edge("moon", "mars")

    def test_bandwidth_allocation(self):
        """Test bandwidth allocation."""
        from src.network import allocate_bandwidth
        alloc = allocate_bandwidth(["moon", "mars"], {"moon": 0.4, "mars": 0.6})
        assert abs(sum(alloc.values()) - 1.0) < 0.01, "Should sum to 1.0"


class TestSystem:
    """Test system module."""

    def test_initialize_system(self):
        """Test system initialization."""
        from src.system import initialize_system, SystemConfig
        cfg = SystemConfig(duration_sols=10)
        state = initialize_system(cfg)
        assert "earth" in state.bodies
        assert "mars" in state.bodies
        assert state.sol == 0

    def test_run_simulation(self):
        """Test simulation run."""
        from src.system import run_simulation, SystemConfig
        cfg = SystemConfig(duration_sols=100, emit_receipts=False)
        result = run_simulation(cfg)
        assert result.sol == 100
        assert result.total_entropy > 0

    def test_moon_relay_impact(self):
        """Test Moon relay improves Mars external rate."""
        from src.system import run_simulation, SystemConfig

        cfg_no_relay = SystemConfig(duration_sols=100, moon_relay_enabled=False, emit_receipts=False)
        cfg_relay = SystemConfig(duration_sols=100, moon_relay_enabled=True, emit_receipts=False)

        result_no_relay = run_simulation(cfg_no_relay)
        result_relay = run_simulation(cfg_relay)

        mars_no_relay = result_no_relay.bodies.get("mars")
        mars_relay = result_relay.bodies.get("mars")

        if mars_no_relay and mars_relay:
            # Relay should improve external rate
            assert mars_relay.external_rate >= mars_no_relay.external_rate * 1.2


class TestOrbital:
    """Test orbital module."""

    def test_create_orbital_state(self):
        """Test orbital state creation."""
        from src.orbital import create_orbital_state, KESSLER_THRESHOLD
        state = create_orbital_state(70000, 5000)
        assert not state.kessler_active

        state2 = create_orbital_state(75000, 5000)
        assert state2.debris_ratio >= KESSLER_THRESHOLD

    def test_kessler_check(self):
        """Test Kessler threshold check."""
        from src.orbital import create_orbital_state, check_kessler
        state = create_orbital_state(65000)
        assert not check_kessler(state)

        state2 = create_orbital_state(75000)
        assert check_kessler(state2)


class TestProve:
    """Test prove module."""

    def test_bits_to_mass(self):
        """Test bits-to-mass equivalence."""
        from src.prove import bits_to_mass_equivalence
        equiv = bits_to_mass_equivalence(25)
        assert equiv["threshold_crew"] == 25
        assert equiv["kg_per_bit_per_sec"] > 0
        assert "implication" in equiv

    def test_neuralink_impact(self):
        """Test Neuralink impact formatting."""
        from src.prove import format_neuralink_impact
        s = format_neuralink_impact(25, 5)
        assert "25" in s
        assert "5" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
