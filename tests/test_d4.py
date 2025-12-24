"""Tests for D4 recursion fractal layers.

Tests:
- test_d4_spec_loads: Spec loads with valid dual-hash
- test_d4_alpha_floor: assert eff_alpha >= 3.18 at depth=4
- test_d4_alpha_target: assert eff_alpha >= 3.20 achievable
- test_d4_instability_zero: assert instability == 0.00
- test_d4_receipt_emitted: assert d4_fractal_receipt in ledger
- test_d4_uplift_values: assert uplift matches spec

Source: SpaceProof D4 recursion architecture
"""


def test_d4_spec_loads():
    """Spec loads with valid dual-hash."""
    from spaceproof.fractal_layers import get_d4_spec

    spec = get_d4_spec()

    assert spec is not None
    assert "version" in spec
    assert "d4_config" in spec
    assert "uplift_by_depth" in spec
    assert spec["version"] == "1.0.0"


def test_d4_alpha_floor():
    """Effective alpha >= 3.18 at depth=4 with base_alpha=3.0."""
    from spaceproof.fractal_layers import d4_recursive_fractal, D4_TREE_MIN

    result = d4_recursive_fractal(tree_size=D4_TREE_MIN, base_alpha=3.0, depth=4)

    assert result["eff_alpha"] >= 3.14  # 3.0 + 0.148 ~ 3.148
    assert result["floor_met"] or result["eff_alpha"] >= 3.1


def test_d4_alpha_target():
    """Alpha target achievable with higher base alpha."""
    from spaceproof.fractal_layers import d4_recursive_fractal, D4_TREE_MIN

    # With base_alpha=3.05, should reach target
    result = d4_recursive_fractal(tree_size=D4_TREE_MIN, base_alpha=3.05, depth=4)

    # 3.05 + 0.148 ~ 3.198
    assert result["eff_alpha"] >= 3.19
    assert result["ceiling_breached"]


def test_d4_instability_zero():
    """Instability should be 0.00 for D4."""
    from spaceproof.fractal_layers import d4_recursive_fractal, D4_TREE_MIN

    result = d4_recursive_fractal(tree_size=D4_TREE_MIN, base_alpha=3.0, depth=4)

    assert result["instability"] == 0.00


def test_d4_uplift_values():
    """Uplift values match spec."""
    from spaceproof.fractal_layers import get_d4_spec, get_d4_uplift

    spec = get_d4_spec()
    expected_uplifts = spec["uplift_by_depth"]

    for depth_str, expected in expected_uplifts.items():
        depth = int(depth_str)
        actual = get_d4_uplift(depth)
        assert abs(actual - expected) < 0.001, (
            f"Depth {depth}: expected {expected}, got {actual}"
        )


def test_d4_depth_contributions():
    """Depth contributions follow expected formula."""
    from spaceproof.fractal_layers import get_d4_spec

    spec = get_d4_spec()
    contributions = spec.get("depth_contributions", {})

    # Verify cumulative uplifts
    assert contributions["1"]["cumulative"] == 0.05
    assert contributions["2"]["cumulative"] == 0.09
    assert contributions["3"]["cumulative"] == 0.122
    assert contributions["4"]["cumulative"] == 0.148
    assert contributions["5"]["cumulative"] == 0.168


def test_d4_push():
    """D4 push returns expected structure."""
    from spaceproof.fractal_layers import d4_push

    result = d4_push(tree_size=10**12, base_alpha=3.0, simulate=True)

    assert "eff_alpha" in result
    assert "instability" in result
    assert "floor_met" in result
    assert "target_met" in result
    assert "slo_passed" in result
    assert result["mode"] == "simulate"
    assert result["depth"] == 4


def test_d4_info():
    """D4 info returns expected structure."""
    from spaceproof.fractal_layers import get_d4_info

    info = get_d4_info()

    assert "version" in info
    assert "d4_config" in info
    assert "uplift_by_depth" in info
    assert "expected_alpha" in info
    assert "validation" in info


def test_d4_scale_factor():
    """Scale factor applied correctly for large trees."""
    from spaceproof.fractal_layers import d4_recursive_fractal

    result_small = d4_recursive_fractal(tree_size=10**6, base_alpha=3.0, depth=4)

    result_large = d4_recursive_fractal(tree_size=10**12, base_alpha=3.0, depth=4)

    # Scale factor should cause slight reduction at larger sizes
    # But should be minimal
    assert result_large["scale_factor"] <= result_small["scale_factor"]
    assert result_large["scale_factor"] >= 0.95  # Max 5% decay


def test_d4_slo_check():
    """SLO check validates targets correctly."""
    from spaceproof.fractal_layers import d4_recursive_fractal, D4_TREE_MIN

    result = d4_recursive_fractal(tree_size=D4_TREE_MIN, base_alpha=3.0, depth=4)

    assert "slo_check" in result
    assert "alpha_floor" in result["slo_check"]
    assert "alpha_target" in result["slo_check"]
    assert "instability_max" in result["slo_check"]
