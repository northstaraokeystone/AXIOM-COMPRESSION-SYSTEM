"""Test suite for D5 fractal recursion + MOXIE-calibrated ISRU hybrid.

Tests cover:
- D5 spec loading with dual-hash verification
- D5 alpha floor/target/ceiling achievement
- D5 uplift value validation
- MOXIE calibration data validation
- ISRU closure target achievement
- D5+ISRU hybrid integration
- Mars MOXIE integration

SLO Requirements:
- eff_alpha >= 3.23 at depth=5
- MOXIE o2_total_g == 122
- ISRU closure >= 0.80 (target: 0.85)
"""

import pytest

from src.fractal_layers import (
    get_d5_spec,
    get_d5_uplift,
    d5_recursive_fractal,
    d5_push,
    get_d5_info,
    recursive_fractal,
    D5_ALPHA_FLOOR,
    D5_ALPHA_TARGET,
    D5_UPLIFT,
    D5_TREE_MIN,
)
from src.isru_hybrid import (
    moxie_calibration,
    load_moxie_calibration,
    simulate_o2_production,
    compute_isru_closure,
    d5_isru_hybrid,
    compute_o2_autonomy,
    get_isru_info,
    MOXIE_O2_TOTAL_G,
    MOXIE_O2_PEAK_G_HR,
    MOXIE_O2_AVG_G_HR,
    ISRU_CLOSURE_TARGET,
)
from src.paths.mars.core import (
    integrate_moxie,
    compute_o2_autonomy as mars_compute_o2_autonomy,
    simulate_dome_moxie,
)


# === D5 SPEC TESTS ===


def test_d5_spec_loads():
    """Test that D5 spec loads with valid configuration."""
    spec = get_d5_spec()
    assert spec is not None
    assert "version" in spec
    assert "d5_config" in spec
    assert "moxie_calibration" in spec
    assert "isru_config" in spec


def test_d5_spec_alpha_values():
    """Test that D5 spec contains correct alpha values."""
    spec = get_d5_spec()
    d5_config = spec["d5_config"]

    assert d5_config["alpha_floor"] == 3.23
    assert d5_config["alpha_target"] == 3.25
    assert d5_config["alpha_ceiling"] == 3.27
    assert d5_config["recursion_depth"] == 5


def test_d5_spec_uplift_by_depth():
    """Test that uplift values are present for depths 1-5."""
    spec = get_d5_spec()
    uplift_map = spec["uplift_by_depth"]

    assert "1" in uplift_map
    assert "2" in uplift_map
    assert "3" in uplift_map
    assert "4" in uplift_map
    assert "5" in uplift_map

    # Verify cumulative uplift progression
    assert uplift_map["1"] == 0.05
    assert uplift_map["2"] == 0.09
    assert uplift_map["3"] == 0.122
    assert uplift_map["4"] == 0.148
    assert uplift_map["5"] == 0.168


# === D5 ALPHA TESTS ===


def test_d5_alpha_floor():
    """Test that D5 achieves alpha floor (3.23) at depth=5.

    With base_alpha=3.07 and uplift=0.168, eff_alpha should reach 3.23+.
    """
    # D5 provides +0.168 uplift, so need base >= 3.07 for floor 3.23
    result = d5_recursive_fractal(D5_TREE_MIN, 3.07, depth=5)
    assert result["eff_alpha"] >= D5_ALPHA_FLOOR, f"Expected >= {D5_ALPHA_FLOOR}, got {result['eff_alpha']}"


def test_d5_alpha_target():
    """Test that D5 can achieve alpha target (3.25) at depth=5."""
    result = d5_recursive_fractal(D5_TREE_MIN, 3.1, depth=5)
    assert result["eff_alpha"] >= D5_ALPHA_TARGET, f"Expected >= {D5_ALPHA_TARGET}, got {result['eff_alpha']}"


def test_d5_uplift_value():
    """Test that D5 uplift value is 0.168."""
    assert D5_UPLIFT == 0.168

    uplift = get_d5_uplift(5)
    assert uplift == 0.168


def test_d5_instability_zero():
    """Test that D5 maintains zero instability."""
    result = d5_recursive_fractal(D5_TREE_MIN, 3.0, depth=5)
    assert result["instability"] == 0.00


def test_d5_floor_met_flag():
    """Test that floor_met flag is set correctly."""
    result = d5_recursive_fractal(D5_TREE_MIN, 3.1, depth=5)
    assert result["floor_met"] is True


def test_d5_target_met_flag():
    """Test that target_met flag is set correctly."""
    result = d5_recursive_fractal(D5_TREE_MIN, 3.1, depth=5)
    assert result["target_met"] is True


# === MOXIE CALIBRATION TESTS ===


def test_moxie_total():
    """Test that MOXIE O2 total is 122g."""
    assert MOXIE_O2_TOTAL_G == 122

    calibration = moxie_calibration()
    assert calibration["o2_total_g"] == 122


def test_moxie_peak():
    """Test that MOXIE O2 peak rate is 12 g/hr."""
    assert MOXIE_O2_PEAK_G_HR == 12

    calibration = moxie_calibration()
    assert calibration["o2_peak_g_hr"] == 12


def test_moxie_avg():
    """Test that MOXIE O2 average rate is 5.5 g/hr."""
    assert MOXIE_O2_AVG_G_HR == 5.5

    calibration = moxie_calibration()
    assert calibration["o2_avg_g_hr"] == 5.5


def test_moxie_validation():
    """Test that MOXIE calibration passes validation."""
    calibration = moxie_calibration()
    assert calibration["validated"] is True


def test_moxie_load_calibration():
    """Test that load_moxie_calibration returns valid data."""
    calibration = load_moxie_calibration()
    assert "o2_total_g" in calibration
    assert "o2_peak_g_hr" in calibration
    assert "o2_avg_g_hr" in calibration
    assert "source" in calibration


# === ISRU CLOSURE TESTS ===


def test_isru_closure_target():
    """Test that ISRU closure target is 0.85."""
    assert ISRU_CLOSURE_TARGET == 0.85


def test_isru_closure_achievable():
    """Test that ISRU closure >= 0.80 is achievable with sufficient MOXIE units.

    With MOXIE avg rate of 5.5 g/hr, 4 crew consuming 0.84 kg/day each:
    - Consumption: 4 * 0.84 = 3.36 kg/day
    - Need production: 3.36 * 0.80 = 2.69 kg/day
    - MOXIE produces: 5.5 g/hr * 24 = 132 g/day per unit
    - Need: 2690 / 132 = ~20 MOXIE units for 80% closure
    - Use 30 units to ensure >= 0.80 closure
    """
    # With 30 MOXIE units for 4 crew
    production_result = simulate_o2_production(24, 4, 30)

    production = {"o2": production_result["production_kg"]}
    consumption = {"o2": production_result["consumption_kg"]}

    closure = compute_isru_closure(production, consumption)
    assert closure >= 0.80, f"Expected closure >= 0.80, got {closure}"


def test_isru_closure_computation():
    """Test ISRU closure computation logic."""
    production = {"o2": 1.0}
    consumption = {"o2": 1.0}

    closure = compute_isru_closure(production, consumption)
    assert closure == 1.0


def test_isru_closure_zero_consumption():
    """Test ISRU closure with zero consumption."""
    production = {"o2": 1.0}
    consumption = {"o2": 0.0}

    closure = compute_isru_closure(production, consumption)
    assert closure == 0.0


# === D5+ISRU HYBRID TESTS ===


def test_d5_isru_hybrid_alpha():
    """Test that D5+ISRU hybrid achieves alpha >= 3.23."""
    result = d5_isru_hybrid(D5_TREE_MIN, 3.1, crew=4, hours=24, moxie_units=10)

    assert result["d5_result"]["eff_alpha"] >= D5_ALPHA_FLOOR


def test_d5_isru_hybrid_receipt():
    """Test that D5+ISRU hybrid emits receipt."""
    result = d5_isru_hybrid(D5_TREE_MIN, 3.0, crew=4, hours=24, moxie_units=10)

    # Check structure
    assert "d5_result" in result
    assert "isru_result" in result
    assert "closure" in result
    assert "combined_slo" in result


def test_d5_isru_hybrid_slo_structure():
    """Test that D5+ISRU hybrid SLO structure is correct."""
    result = d5_isru_hybrid(D5_TREE_MIN, 3.1, crew=4, hours=24, moxie_units=10)

    combined = result["combined_slo"]
    assert "alpha_target" in combined
    assert "alpha_met" in combined
    assert "closure_target" in combined
    assert "closure_met" in combined
    assert "all_targets_met" in combined


def test_d5_isru_hybrid_production():
    """Test that D5+ISRU hybrid includes production metrics."""
    result = d5_isru_hybrid(D5_TREE_MIN, 3.0, crew=4, hours=24, moxie_units=10)

    isru = result["isru_result"]
    assert "production_kg" in isru
    assert "consumption_kg" in isru
    assert "balance_kg" in isru
    assert "self_sufficient" in isru


# === O2 AUTONOMY TESTS ===


def test_o2_autonomy_computable():
    """Test that O2 autonomy metric returns valid float."""
    # 10 MOXIE units producing at avg rate
    production_rate = (MOXIE_O2_AVG_G_HR * 10) / 1000  # kg/hr

    autonomy = compute_o2_autonomy(production_rate, crew=4)

    assert isinstance(autonomy, float)
    assert autonomy >= 0.0
    assert autonomy <= 10.0  # Capped at 10.0


def test_o2_autonomy_self_sufficient():
    """Test O2 autonomy self-sufficiency flag."""
    # High production rate
    autonomy = compute_o2_autonomy(1.0, crew=1)
    assert autonomy >= 1.0  # Should be self-sufficient


def test_o2_autonomy_zero_crew():
    """Test O2 autonomy with zero crew."""
    autonomy = compute_o2_autonomy(0.1, crew=0)
    assert autonomy == 0.0


# === MARS MOXIE INTEGRATION TESTS ===


def test_mars_moxie_integration():
    """Test that Mars path uses MOXIE data."""
    result = integrate_moxie()

    assert result["integrated"] is True
    assert "moxie_calibration" in result
    assert "production_scaled" in result


def test_mars_moxie_calibration_values():
    """Test that Mars MOXIE integration has correct calibration values."""
    result = integrate_moxie()
    moxie = result["moxie_calibration"]

    assert moxie["o2_total_g"] == 122
    assert moxie["o2_peak_g_hr"] == 12
    assert moxie["o2_avg_g_hr"] == 5.5


def test_mars_o2_autonomy_computable():
    """Test that Mars O2 autonomy metric works."""
    production_rate = 0.1  # kg/hr
    crew = 4

    autonomy = mars_compute_o2_autonomy(production_rate, crew)

    assert isinstance(autonomy, float)


def test_mars_dome_moxie_simulation():
    """Test Mars dome simulation with MOXIE."""
    result = simulate_dome_moxie(crew=4, duration_days=30, moxie_units=50)

    assert result["moxie_enabled"] is True
    assert result["moxie_units"] == 50
    assert "o2_production" in result
    assert "o2_consumption" in result
    assert "o2_closure" in result


# === INFO FUNCTIONS TESTS ===


def test_get_d5_info():
    """Test D5 info function returns valid structure."""
    info = get_d5_info()

    assert "version" in info
    assert "d5_config" in info
    assert "uplift_by_depth" in info
    assert "moxie_calibration" in info
    assert "isru_config" in info


def test_get_isru_info():
    """Test ISRU info function returns valid structure."""
    info = get_isru_info()

    assert "moxie_calibration" in info
    assert "isru_config" in info
    assert "d5_integration" in info
    assert "consumption_rates" in info


# === D5 PUSH TESTS ===


def test_d5_push_result():
    """Test D5 push returns valid result."""
    result = d5_push(D5_TREE_MIN, 3.0, simulate=True)

    assert "eff_alpha" in result
    assert "instability" in result
    assert "floor_met" in result
    assert "target_met" in result
    assert "slo_passed" in result


def test_d5_push_slo():
    """Test D5 push SLO passes."""
    result = d5_push(D5_TREE_MIN, 3.1, simulate=True)

    # With base_alpha=3.1 and uplift=0.168, should pass floor
    assert result["slo_passed"] is True


# === RECURSIVE FRACTAL D5 TESTS ===


def test_recursive_fractal_depth_5():
    """Test recursive_fractal at depth=5."""
    result = recursive_fractal(10**9, 3.0, depth=5)

    assert result["depth"] == 5
    assert "final_alpha" in result
    assert "total_uplift" in result


def test_recursive_fractal_uplift_progression():
    """Test that uplift increases with depth."""
    base_alpha = 3.0
    tree_size = 10**9

    prev_alpha = base_alpha
    for depth in range(1, 6):
        result = recursive_fractal(tree_size, base_alpha, depth=depth)
        assert result["final_alpha"] > prev_alpha, f"Depth {depth} should increase alpha"
        prev_alpha = result["final_alpha"]


# === O2 PRODUCTION SIMULATION TESTS ===


def test_simulate_o2_production():
    """Test O2 production simulation."""
    result = simulate_o2_production(24, 4, 10)

    assert "production_g" in result
    assert "production_kg" in result
    assert "consumption_kg" in result
    assert "balance_kg" in result
    assert "self_sufficient" in result


def test_simulate_o2_production_scaling():
    """Test that O2 production scales with MOXIE units."""
    result_10 = simulate_o2_production(24, 4, 10)
    result_20 = simulate_o2_production(24, 4, 20)

    # Double units should double production
    assert result_20["production_kg"] == pytest.approx(result_10["production_kg"] * 2, rel=0.01)


def test_simulate_o2_consumption_scaling():
    """Test that O2 consumption scales with crew."""
    result_4 = simulate_o2_production(24, 4, 10)
    result_8 = simulate_o2_production(24, 8, 10)

    # Double crew should double consumption
    assert result_8["consumption_kg"] == pytest.approx(result_4["consumption_kg"] * 2, rel=0.01)
