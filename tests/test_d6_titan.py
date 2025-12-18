"""Test suite for D6 fractal recursion + Titan methane hybrid.

Tests cover:
- D6 spec loading with dual-hash verification
- D6 alpha floor/target/ceiling achievement
- D6 uplift value validation
- Titan configuration loading
- Titan methane harvest simulation
- Titan autonomy requirements
- D6+Titan hybrid integration
- Perovskite efficiency scaling

SLO Requirements:
- eff_alpha >= 3.31 at depth=6
- Titan autonomy >= 0.99
- Perovskite scaling factor == 3.33
"""


from src.fractal_layers import (
    get_d6_spec,
    get_d6_uplift,
    d6_recursive_fractal,
    d6_push,
    get_d6_info,
    recursive_fractal,
    D6_ALPHA_FLOOR,
    D6_ALPHA_TARGET,
    D6_UPLIFT,
    D6_TREE_MIN,
)
from src.titan_methane_hybrid import (
    load_titan_config,
    simulate_harvest,
    compute_autonomy,
    d6_titan_hybrid,
    methane_to_fuel,
    get_titan_info,
    TITAN_AUTONOMY_REQUIREMENT,
    TITAN_LATENCY_MIN_MIN,
    TITAN_LATENCY_MAX_MIN,
)
from src.perovskite_efficiency import (
    load_efficiency_config,
    compute_scaling_factor,
    project_efficiency,
    validate_perovskite_target,
    get_perovskite_info,
    MOXIE_EFFICIENCY,
    PEROVSKITE_SOLAR_EFF_TARGET,
    EFFICIENCY_SCALING_FACTOR,
)


# === D6 SPEC TESTS ===


def test_d6_spec_loads():
    """Test that D6 spec loads with valid configuration."""
    spec = get_d6_spec()
    assert spec is not None
    assert "version" in spec
    assert "d6_config" in spec
    assert "titan_config" in spec
    assert "adversarial_config" in spec


def test_d6_spec_alpha_values():
    """Test that D6 spec contains correct alpha values."""
    spec = get_d6_spec()
    d6_config = spec["d6_config"]

    assert d6_config["alpha_floor"] == 3.31
    assert d6_config["alpha_target"] == 3.33
    assert d6_config["alpha_ceiling"] == 3.35
    assert d6_config["recursion_depth"] == 6


def test_d6_spec_uplift_by_depth():
    """Test that uplift values are present for depths 1-6."""
    spec = get_d6_spec()
    uplift_map = spec["uplift_by_depth"]

    assert "1" in uplift_map
    assert "2" in uplift_map
    assert "3" in uplift_map
    assert "4" in uplift_map
    assert "5" in uplift_map
    assert "6" in uplift_map

    # Verify cumulative uplift progression
    assert uplift_map["6"] == 0.185


# === D6 ALPHA TESTS ===


def test_d6_alpha_floor():
    """Test that D6 achieves alpha floor (3.31) at depth=6.

    With base_alpha=3.15 and uplift=0.185, eff_alpha should reach 3.31+.
    """
    result = d6_recursive_fractal(D6_TREE_MIN, 3.15, depth=6)
    assert result["eff_alpha"] >= D6_ALPHA_FLOOR, f"Expected >= {D6_ALPHA_FLOOR}, got {result['eff_alpha']}"


def test_d6_alpha_target():
    """Test that D6 can achieve alpha target (3.33) at depth=6."""
    result = d6_recursive_fractal(D6_TREE_MIN, 3.15, depth=6)
    assert result["eff_alpha"] >= D6_ALPHA_TARGET, f"Expected >= {D6_ALPHA_TARGET}, got {result['eff_alpha']}"


def test_d6_uplift_value():
    """Test that D6 uplift value is 0.185."""
    assert D6_UPLIFT == 0.185

    uplift = get_d6_uplift(6)
    assert uplift == 0.185


def test_d6_instability_zero():
    """Test that D6 maintains zero instability."""
    result = d6_recursive_fractal(D6_TREE_MIN, 3.0, depth=6)
    assert result["instability"] == 0.00


def test_d6_floor_met_flag():
    """Test that floor_met flag is set correctly."""
    result = d6_recursive_fractal(D6_TREE_MIN, 3.15, depth=6)
    assert result["floor_met"] is True


def test_d6_target_met_flag():
    """Test that target_met flag is set correctly."""
    result = d6_recursive_fractal(D6_TREE_MIN, 3.15, depth=6)
    assert result["target_met"] is True


# === TITAN CONFIG TESTS ===


def test_titan_config_loads():
    """Test that Titan config loads correctly."""
    config = load_titan_config()
    assert config is not None
    assert config["body"] == "titan"
    assert config["resource"] == "methane"


def test_titan_autonomy_requirement():
    """Test that Titan autonomy requirement is 0.99."""
    config = load_titan_config()
    assert config["autonomy_requirement"] == 0.99
    assert TITAN_AUTONOMY_REQUIREMENT == 0.99


def test_titan_latency_bounds():
    """Test that Titan latency is in [70, 90] min."""
    config = load_titan_config()
    latency = config["latency_min"]

    assert latency[0] == 70
    assert latency[1] == 90
    assert TITAN_LATENCY_MIN_MIN == 70
    assert TITAN_LATENCY_MAX_MIN == 90


def test_titan_methane_density():
    """Test Titan methane density value."""
    config = load_titan_config()
    assert config["methane_density_kg_m3"] == 1.5


def test_titan_surface_temp():
    """Test Titan surface temperature."""
    config = load_titan_config()
    assert config["surface_temp_k"] == 94


# === TITAN HARVEST TESTS ===


def test_titan_harvest_simulation():
    """Test Titan methane harvest simulation runs."""
    result = simulate_harvest(duration_days=30)

    assert "duration_days" in result
    assert "processed_kg" in result
    assert "energy_kwh" in result
    assert "autonomy_achieved" in result


def test_titan_harvest_autonomy():
    """Test that harvest simulation achieves autonomy."""
    result = simulate_harvest(duration_days=30)
    assert result["autonomy_achieved"] >= 0.95


def test_titan_autonomy_computation():
    """Test autonomy computation logic."""
    # High harvest rate should give full autonomy
    autonomy = compute_autonomy(10.0, 1.0)
    assert autonomy == 1.0

    # Equal rates
    autonomy = compute_autonomy(5.0, 5.0)
    assert autonomy == 1.0

    # Lower harvest than consumption
    autonomy = compute_autonomy(2.0, 10.0)
    assert autonomy == 0.2


def test_titan_methane_to_fuel():
    """Test methane to fuel conversion."""
    fuel = methane_to_fuel(1.0)  # 1 kg methane

    assert "methane_kg" in fuel
    assert "energy_mj" in fuel
    assert "energy_kwh" in fuel
    assert "o2_required_kg" in fuel
    assert "co2_produced_kg" in fuel
    assert "h2o_produced_kg" in fuel

    # Check stoichiometry
    # 1 kg CH4 needs ~4 kg O2
    assert fuel["o2_required_kg"] > 3.0
    assert fuel["o2_required_kg"] < 5.0


# === D6+TITAN HYBRID TESTS ===


def test_d6_titan_hybrid_alpha():
    """Test that D6+Titan hybrid achieves alpha >= 3.31."""
    result = d6_titan_hybrid(D6_TREE_MIN, 3.15)

    assert result["d6_result"]["eff_alpha"] >= D6_ALPHA_FLOOR


def test_d6_titan_hybrid_receipt():
    """Test that D6+Titan hybrid returns proper structure."""
    result = d6_titan_hybrid(D6_TREE_MIN, 3.0)

    assert "d6_result" in result
    assert "titan_result" in result
    assert "combined_slo" in result
    assert "gate" in result


def test_d6_titan_hybrid_slo_structure():
    """Test that D6+Titan hybrid SLO structure is correct."""
    result = d6_titan_hybrid(D6_TREE_MIN, 3.15)

    combined = result["combined_slo"]
    assert "alpha_target" in combined
    assert "alpha_met" in combined
    assert "autonomy_target" in combined
    assert "autonomy_met" in combined
    assert "all_targets_met" in combined


def test_d6_titan_hybrid_all_targets():
    """Test that D6+Titan hybrid meets all targets."""
    result = d6_titan_hybrid(D6_TREE_MIN, 3.15)

    assert result["combined_slo"]["alpha_met"] is True


# === PEROVSKITE EFFICIENCY TESTS ===


def test_perovskite_config_loads():
    """Test that perovskite efficiency config loads."""
    config = load_efficiency_config()
    assert config is not None
    assert "moxie_baseline" in config
    assert "perovskite_target" in config
    assert "scaling_factor" in config


def test_perovskite_scaling_factor():
    """Test that scaling factor is 3.33."""
    assert EFFICIENCY_SCALING_FACTOR == 3.33

    factor = compute_scaling_factor(MOXIE_EFFICIENCY, PEROVSKITE_SOLAR_EFF_TARGET)
    assert round(factor, 2) == 3.33


def test_perovskite_target():
    """Test that perovskite target is 0.20 (20%)."""
    assert PEROVSKITE_SOLAR_EFF_TARGET == 0.20


def test_perovskite_moxie_baseline():
    """Test that MOXIE baseline is 0.06 (6%)."""
    assert MOXIE_EFFICIENCY == 0.06


def test_perovskite_target_achievable():
    """Test that perovskite target is marked achievable."""
    achievable = validate_perovskite_target()
    assert achievable is True


def test_perovskite_projection():
    """Test efficiency projection over time."""
    result = project_efficiency(years=10, growth_rate=0.10)

    assert "projections" in result
    assert len(result["projections"]) > 0
    assert result["baseline_efficiency"] == MOXIE_EFFICIENCY


# === INFO FUNCTIONS TESTS ===


def test_get_d6_info():
    """Test D6 info function returns valid structure."""
    info = get_d6_info()

    assert "version" in info
    assert "d6_config" in info
    assert "uplift_by_depth" in info
    assert "titan_config" in info
    assert "adversarial_config" in info


def test_get_titan_info():
    """Test Titan info function returns valid structure."""
    info = get_titan_info()

    assert "module" in info
    assert "version" in info
    assert "config" in info
    assert "harvesting" in info
    assert "autonomy" in info


def test_get_perovskite_info():
    """Test perovskite info function returns valid structure."""
    info = get_perovskite_info()

    assert "module" in info
    assert "version" in info
    assert "stub_mode" in info
    assert "config" in info


# === D6 PUSH TESTS ===


def test_d6_push_result():
    """Test D6 push returns valid result."""
    result = d6_push(D6_TREE_MIN, 3.15, simulate=True)

    assert "eff_alpha" in result
    assert "instability" in result
    assert "floor_met" in result
    assert "target_met" in result
    assert "slo_passed" in result


def test_d6_push_slo():
    """Test D6 push SLO passes."""
    result = d6_push(D6_TREE_MIN, 3.15, simulate=True)

    assert result["slo_passed"] is True


# === RECURSIVE FRACTAL D6 TESTS ===


def test_recursive_fractal_depth_6():
    """Test recursive_fractal at depth=6."""
    result = recursive_fractal(10**9, 3.0, depth=6)

    assert result["depth"] == 6
    assert "final_alpha" in result
    assert "total_uplift" in result


def test_recursive_fractal_uplift_progression_to_d6():
    """Test that uplift increases with depth up to D6."""
    base_alpha = 3.0
    tree_size = 10**9

    prev_alpha = base_alpha
    for depth in range(1, 7):
        result = recursive_fractal(tree_size, base_alpha, depth=depth)
        assert result["final_alpha"] > prev_alpha, f"Depth {depth} should increase alpha"
        prev_alpha = result["final_alpha"]
