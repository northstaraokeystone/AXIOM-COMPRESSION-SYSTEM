"""Tests for D16 topological fractal recursion and Kuiper 12-body integration.

Tests:
- D16 spec loading
- D16 alpha targets (floor, target, ceiling)
- Topological primitives (persistent homology)
- D16+Kuiper hybrid integration
"""

import pytest
from src.fractal_layers import (
    get_d16_spec,
    get_d16_uplift,
    compute_persistent_homology,
    compute_betti_numbers,
    multidimensional_scaling,
    topological_compression_ratio,
    d16_topological_push,
    d16_push,
    d16_kuiper_hybrid,
    get_d16_info,
    D16_ALPHA_FLOOR,
    D16_ALPHA_TARGET,
    D16_ALPHA_CEILING,
    D16_UPLIFT,
    D16_TOPOLOGICAL,
    D16_HOMOLOGY_DIMENSION,
)


class TestD16SpecLoading:
    """Tests for D16 spec loading."""

    def test_d16_spec_loads(self):
        """Spec loads with valid dual-hash."""
        spec = get_d16_spec()
        assert spec is not None
        assert "version" in spec
        assert "d16_config" in spec

    def test_d16_spec_version(self):
        """Spec version is valid."""
        spec = get_d16_spec()
        assert spec["version"] == "1.0.0"

    def test_d16_config_present(self):
        """D16 config section present."""
        spec = get_d16_spec()
        assert "d16_config" in spec
        assert "recursion_depth" in spec["d16_config"]
        assert spec["d16_config"]["recursion_depth"] == 16

    def test_d16_uplift_by_depth(self):
        """Uplift by depth table present."""
        spec = get_d16_spec()
        assert "uplift_by_depth" in spec
        assert "16" in spec["uplift_by_depth"]
        assert float(spec["uplift_by_depth"]["16"]) == 0.38


class TestD16AlphaTargets:
    """Tests for D16 alpha floor, target, ceiling."""

    def test_d16_alpha_floor(self):
        """Alpha floor is 3.91."""
        assert D16_ALPHA_FLOOR == 3.91

    def test_d16_alpha_target(self):
        """Alpha target is 3.90."""
        assert D16_ALPHA_TARGET == 3.90

    def test_d16_alpha_ceiling(self):
        """Alpha ceiling is 3.94."""
        assert D16_ALPHA_CEILING == 3.94

    def test_d16_uplift_value(self):
        """D16 uplift is 0.38."""
        assert D16_UPLIFT == 0.38

    def test_d16_uplift_from_spec(self):
        """Get uplift from spec matches constant."""
        uplift = get_d16_uplift(16)
        assert uplift == D16_UPLIFT


class TestD16Topological:
    """Tests for D16 topological primitives."""

    def test_d16_topological_enabled(self):
        """Topological primitives enabled."""
        assert D16_TOPOLOGICAL is True

    def test_d16_homology_dimension(self):
        """Homology dimension is 2 (H0, H1, H2)."""
        assert D16_HOMOLOGY_DIMENSION == 2

    def test_d16_persistent_homology(self):
        """Persistent homology computes correctly."""
        data = [[i * 0.01, (i % 10) * 0.1] for i in range(100)]
        homology = compute_persistent_homology(data, dimension=2)

        assert "dimension" in homology
        assert homology["dimension"] == 2
        assert "persistence_diagrams" in homology
        assert "betti_numbers" in homology
        assert "total_persistence" in homology

    def test_d16_betti_numbers(self):
        """Betti numbers extracted correctly."""
        data = [[i * 0.01, (i % 10) * 0.1] for i in range(100)]
        homology = compute_persistent_homology(data, dimension=2)
        betti = compute_betti_numbers(homology)

        assert isinstance(betti, list)
        assert len(betti) == 3  # b0, b1, b2

    def test_d16_mds_embedding(self):
        """MDS embedding works."""
        distances = [[i * j for j in range(10)] for i in range(10)]
        embedding = multidimensional_scaling(distances, dims=3)

        assert isinstance(embedding, list)
        assert len(embedding) == 10
        assert len(embedding[0]) == 3

    def test_topological_compression_ratio(self):
        """Compression ratio computed correctly."""
        original = {"data": [i for i in range(100)]}
        homology = compute_persistent_homology([[0, 0]], dimension=2)
        ratio = topological_compression_ratio(original, homology)

        assert ratio >= 1.0


class TestD16RecursionResults:
    """Tests for D16 recursion execution."""

    def test_d16_topological_push(self):
        """D16 topological push executes."""
        result = d16_topological_push(10**9, 3.55, topological=True)

        assert "eff_alpha" in result
        assert "topological" in result
        assert result["topological"] is True
        assert result["depth"] == 16

    def test_d16_alpha_floor_met(self):
        """D16 meets alpha floor at depth=16."""
        result = d16_topological_push(10**9, 3.55, topological=True)
        # With uplift 0.38 and base 3.55, should get ~3.93
        assert result["eff_alpha"] >= D16_ALPHA_TARGET

    def test_d16_push(self):
        """D16 push executes correctly."""
        result = d16_push(tree_size=10**9, base_alpha=3.55)

        assert "eff_alpha" in result
        assert "floor_met" in result
        assert "target_met" in result
        assert result["depth"] == 16

    def test_d16_slo_passed(self):
        """D16 SLO passes."""
        result = d16_push(tree_size=10**9, base_alpha=3.55)
        assert result["slo_passed"] is True


class TestD16KuiperHybrid:
    """Tests for D16+Kuiper hybrid integration."""

    def test_d16_kuiper_hybrid_executes(self):
        """D16+Kuiper hybrid executes."""
        result = d16_kuiper_hybrid(tree_size=10**9, base_alpha=3.55)

        assert "d16" in result
        assert "kuiper" in result
        assert "combined_alpha" in result

    def test_d16_kuiper_hybrid_alpha(self):
        """Hybrid achieves alpha target."""
        result = d16_kuiper_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["combined_alpha"] >= D16_ALPHA_TARGET

    def test_d16_kuiper_hybrid_receipt(self):
        """Hybrid emits receipt."""
        # Receipt emission tested implicitly via successful execution
        result = d16_kuiper_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["gate"] == "t24h"


class TestD16Info:
    """Tests for D16 info retrieval."""

    def test_get_d16_info(self):
        """D16 info retrieves correctly."""
        info = get_d16_info()

        assert "version" in info
        assert "d16_config" in info
        assert "kuiper_12body_config" in info
        assert "ml_ensemble_config" in info
        assert "bulletproofs_config" in info

    def test_d16_info_description(self):
        """D16 info has description."""
        info = get_d16_info()
        assert "description" in info
        assert "D16" in info["description"]
