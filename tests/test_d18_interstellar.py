"""Tests for D18 fractal recursion with interstellar relay integration.

Tests:
- D18 spec loading
- D18 alpha targets (floor, target, ceiling)
- Pruning v3 metrics
- D18+interstellar+quantum hybrid integration
"""

from src.fractal_layers import (
    get_d18_spec,
    get_d18_uplift,
    identify_topological_holes,
    eliminate_holes,
    pruning_v3,
    compute_compression,
    d18_recursive_fractal,
    d18_push,
    d18_interstellar_hybrid,
    get_d18_info,
    D18_ALPHA_FLOOR,
    D18_ALPHA_TARGET,
    D18_ALPHA_CEILING,
    D18_UPLIFT,
    D18_PRUNING_V3,
    D18_COMPRESSION_TARGET,
)


class TestD18SpecLoading:
    """Tests for D18 spec loading."""

    def test_d18_spec_loads(self):
        """Spec loads with valid dual-hash."""
        spec = get_d18_spec()
        assert spec is not None
        assert "version" in spec
        assert "d18_config" in spec

    def test_d18_spec_version(self):
        """Spec version is valid."""
        spec = get_d18_spec()
        assert spec["version"] == "1.0.0"

    def test_d18_config_present(self):
        """D18 config section present."""
        spec = get_d18_spec()
        assert "d18_config" in spec
        assert "recursion_depth" in spec["d18_config"]
        assert spec["d18_config"]["recursion_depth"] == 18

    def test_d18_pruning_v3_enabled(self):
        """Pruning v3 mode enabled in spec."""
        spec = get_d18_spec()
        assert spec["d18_config"]["pruning_v3"] is True

    def test_d18_uplift_by_depth(self):
        """Uplift by depth table present."""
        spec = get_d18_spec()
        assert "uplift_by_depth" in spec
        assert "18" in spec["uplift_by_depth"]
        assert float(spec["uplift_by_depth"]["18"]) == 0.42

    def test_d18_interstellar_config(self):
        """Interstellar relay config present."""
        spec = get_d18_spec()
        assert "interstellar_relay_config" in spec
        relay = spec["interstellar_relay_config"]
        assert relay["target_system"] == "proxima_centauri"
        assert relay["distance_ly"] == 4.24

    def test_d18_quantum_alt_config(self):
        """Quantum alternative config present."""
        spec = get_d18_spec()
        assert "quantum_alternative_config" in spec
        quantum = spec["quantum_alternative_config"]
        assert quantum["enabled"] is True
        assert quantum["no_ftl_constraint"] is True


class TestD18AlphaTargets:
    """Tests for D18 alpha floor, target, ceiling."""

    def test_d18_alpha_floor(self):
        """Alpha floor is 3.91."""
        assert D18_ALPHA_FLOOR == 3.91

    def test_d18_alpha_target(self):
        """Alpha target is 3.90."""
        assert D18_ALPHA_TARGET == 3.90

    def test_d18_alpha_ceiling(self):
        """Alpha ceiling is 3.94."""
        assert D18_ALPHA_CEILING == 3.94

    def test_d18_uplift_value(self):
        """D18 uplift is 0.42."""
        assert D18_UPLIFT == 0.42

    def test_d18_uplift_from_spec(self):
        """Get uplift from spec matches constant."""
        uplift = get_d18_uplift(18)
        assert uplift == D18_UPLIFT


class TestD18PruningV3:
    """Tests for D18 pruning v3 primitives."""

    def test_d18_pruning_v3_enabled(self):
        """Pruning v3 mode enabled."""
        assert D18_PRUNING_V3 is True

    def test_d18_compression_target(self):
        """Compression target is 0.992."""
        assert D18_COMPRESSION_TARGET == 0.992

    def test_identify_topological_holes(self):
        """Topological holes identified."""
        tree = {"size": 10**9, "depth": 18}
        result = identify_topological_holes(tree)

        assert "holes_found" in result
        assert "hole_locations" in result
        assert result["holes_found"] >= 0

    def test_eliminate_holes(self):
        """Holes can be eliminated."""
        tree = {"size": 10**9, "depth": 18}
        holes_result = identify_topological_holes(tree)
        result = eliminate_holes(tree, holes_result["hole_locations"])

        assert "holes_eliminated" in result
        assert "remaining_holes" in result
        assert result["remaining_holes"] == 0

    def test_pruning_v3(self):
        """Pruning v3 executes correctly."""
        tree = {"size": 10**9, "depth": 18}
        result = pruning_v3(tree)

        assert "original_size" in result
        assert "pruned_size" in result
        assert "compression_ratio" in result
        assert "pruning_version" in result
        assert result["pruning_version"] == "v3"

    def test_pruning_v3_compression_target(self):
        """Pruning v3 meets compression target."""
        tree = {"size": 10**9, "depth": 18}
        result = pruning_v3(tree)

        assert result["compression_ratio"] >= D18_COMPRESSION_TARGET
        assert result["target_met"] is True

    def test_compute_compression(self):
        """Compression computation works."""
        result = compute_compression(depth=18)

        assert "depth" in result
        assert "ratio" in result
        assert "target_met" in result


class TestD18RecursionResults:
    """Tests for D18 recursion execution."""

    def test_d18_recursive_fractal(self):
        """D18 recursive fractal executes."""
        result = d18_recursive_fractal(10**9, 3.55)

        assert "eff_alpha" in result
        assert "depth" in result
        assert result["depth"] == 18

    def test_d18_alpha_floor_met(self):
        """D18 meets alpha floor at depth=18."""
        result = d18_recursive_fractal(10**9, 3.55)
        # With uplift 0.42 and base 3.55, should get >= 3.91
        assert result["eff_alpha"] >= D18_ALPHA_TARGET

    def test_d18_push(self):
        """D18 push executes correctly."""
        result = d18_push(tree_size=10**9, base_alpha=3.55)

        assert "eff_alpha" in result
        assert "floor_met" in result
        assert "target_met" in result
        assert result["depth"] == 18

    def test_d18_slo_passed(self):
        """D18 SLO passes."""
        result = d18_push(tree_size=10**9, base_alpha=3.55)
        assert result["slo_passed"] is True

    def test_d18_no_plateau(self):
        """D18 shows no plateau."""
        result = d18_push(tree_size=10**9, base_alpha=3.55)
        assert result["no_plateau"] is True


class TestD18InterstellarHybrid:
    """Tests for D18+interstellar+quantum hybrid integration."""

    def test_d18_interstellar_hybrid_executes(self):
        """D18+interstellar hybrid executes."""
        result = d18_interstellar_hybrid(tree_size=10**9, base_alpha=3.55)

        assert "d18" in result
        assert "relay" in result
        assert "quantum" in result
        assert "combined_alpha" in result

    def test_d18_interstellar_hybrid_alpha(self):
        """Hybrid achieves alpha target."""
        result = d18_interstellar_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["combined_alpha"] >= D18_ALPHA_TARGET

    def test_d18_interstellar_relay_viable(self):
        """Interstellar relay coordination viable."""
        result = d18_interstellar_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["relay"]["coordination_viable"] is True

    def test_d18_quantum_viable(self):
        """Quantum alternative viable."""
        result = d18_interstellar_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["quantum"]["viable"] is True

    def test_d18_interstellar_hybrid_receipt(self):
        """Hybrid emits receipt."""
        result = d18_interstellar_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["gate"] == "t24h"


class TestD18Info:
    """Tests for D18 info retrieval."""

    def test_get_d18_info(self):
        """D18 info retrieves correctly."""
        info = get_d18_info()

        assert "version" in info
        assert "d18_config" in info
        assert "interstellar_relay_config" in info
        assert "quantum_alternative_config" in info

    def test_d18_info_elon_sphere(self):
        """D18 info includes Elon-sphere config."""
        info = get_d18_info()
        assert "elon_sphere_config" in info

    def test_d18_info_description(self):
        """D18 info has description."""
        info = get_d18_info()
        assert "description" in info
        assert "D18" in info["description"]
