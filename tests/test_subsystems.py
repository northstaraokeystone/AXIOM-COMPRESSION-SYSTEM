"""Tests for BUILD C4: subsystems.py physics calculations.

Verify physics calculations match expected values with real formulas.
"""
import math
import pytest

from src.subsystems import (
    # Constants
    STEFAN_BOLTZMANN,
    MARS_AMBIENT_K,
    HAB_THERMAL_MASS_J_PER_K,
    SOLAR_PANEL_EFFICIENCY,
    EQUIPMENT_HEAT_FRACTION,
    RADIATOR_EMISSIVITY,
    HUMAN_O2_KG_PER_DAY,
    HUMAN_CO2_KG_PER_DAY,
    HUMAN_WATER_L_PER_DAY,
    FOOD_KCAL_PER_DAY,
    FOOD_KG_PER_KCAL,
    CO2_MOLAR_MASS,
    H2_MOLAR_MASS,
    CH4_MOLAR_MASS,
    H2O_MOLAR_MASS,
    MOXIE_POWER_W_PER_UNIT,
    # Thermal functions
    solar_input,
    nuclear_input,
    metabolic_heat,
    equipment_heat,
    radiator_capacity,
    thermal_balance,
    # Atmosphere functions
    moxie_o2,
    sabatier,
    human_o2,
    human_co2,
    atmosphere_balance,
    # Resource functions
    water_cycle,
    food_requirement,
    power_budget,
    isru_closure,
    # Receipt function
    emit_subsystem_receipt,
)
from src.entropy import (
    HUMAN_METABOLIC_W,
    MOXIE_O2_G_PER_HR,
    ISS_WATER_RECOVERY,
    KILOPOWER_KW,
    SOLAR_FLUX_MAX,
    SOLAR_FLUX_DUST,
)


# =============================================================================
# THERMAL FUNCTION TESTS
# =============================================================================

class TestSolarInput:
    """Tests for solar_input function."""

    def test_solar_input_basic(self):
        """200 m2 x 590 W/m2 x 0.20 = 23600 W."""
        result = solar_input(200, 590, 0.20)
        assert result == pytest.approx(23600, rel=1e-6)

    def test_solar_input_zero_flux(self):
        """flux=0 -> 0 W."""
        result = solar_input(200, 0, 0.20)
        assert result == 0.0

    def test_solar_input_dust_storm(self):
        """flux=6 W/m2 (dust storm) -> much lower output."""
        normal = solar_input(200, SOLAR_FLUX_MAX, 0.20)
        dust = solar_input(200, SOLAR_FLUX_DUST, 0.20)
        assert dust < normal * 0.05  # Less than 5% of normal
        assert dust == pytest.approx(200 * 6 * 0.20, rel=1e-6)  # 240 W

    def test_solar_input_negative_guards(self):
        """Negative inputs return 0."""
        assert solar_input(-100, 590, 0.20) == 0.0
        assert solar_input(200, -590, 0.20) == 0.0
        assert solar_input(200, 590, -0.20) == 0.0

    def test_solar_input_default_efficiency(self):
        """Default efficiency is SOLAR_PANEL_EFFICIENCY (0.20)."""
        result = solar_input(100, 500)
        assert result == pytest.approx(100 * 500 * SOLAR_PANEL_EFFICIENCY, rel=1e-6)


class TestNuclearInput:
    """Tests for nuclear_input function."""

    def test_nuclear_input_basic(self):
        """2 units x 10 kW = 20000 W."""
        result = nuclear_input(2)
        assert result == pytest.approx(20000, rel=1e-6)

    def test_nuclear_input_zero(self):
        """0 units -> 0 W."""
        result = nuclear_input(0)
        assert result == 0.0

    def test_nuclear_input_negative(self):
        """Negative units -> 0 W."""
        result = nuclear_input(-1)
        assert result == 0.0

    def test_nuclear_input_single_unit(self):
        """1 unit = KILOPOWER_KW * 1000."""
        result = nuclear_input(1)
        assert result == pytest.approx(KILOPOWER_KW * 1000, rel=1e-6)


class TestMetabolicHeat:
    """Tests for metabolic_heat function."""

    def test_metabolic_heat_basic(self):
        """10 crew x 100 W = 1000 W."""
        result = metabolic_heat(10)
        assert result == pytest.approx(1000, rel=1e-6)

    def test_metabolic_heat_zero(self):
        """0 crew -> 0 W."""
        result = metabolic_heat(0)
        assert result == 0.0

    def test_metabolic_heat_negative(self):
        """Negative crew -> 0 W."""
        result = metabolic_heat(-5)
        assert result == 0.0

    def test_metabolic_heat_uses_constant(self):
        """Uses HUMAN_METABOLIC_W from entropy.py."""
        result = metabolic_heat(1)
        assert result == HUMAN_METABOLIC_W


class TestEquipmentHeat:
    """Tests for equipment_heat function."""

    def test_equipment_heat_basic(self):
        """10000 W x 0.70 = 7000 W."""
        result = equipment_heat(10000)
        assert result == pytest.approx(7000, rel=1e-6)

    def test_equipment_heat_zero(self):
        """0 W power -> 0 W heat."""
        result = equipment_heat(0)
        assert result == 0.0

    def test_equipment_heat_negative(self):
        """Negative power -> 0 W heat."""
        result = equipment_heat(-5000)
        assert result == 0.0

    def test_equipment_heat_uses_fraction(self):
        """Uses EQUIPMENT_HEAT_FRACTION."""
        result = equipment_heat(1000)
        assert result == pytest.approx(1000 * EQUIPMENT_HEAT_FRACTION, rel=1e-6)


class TestRadiatorCapacity:
    """Tests for radiator_capacity function."""

    def test_radiator_capacity_positive(self):
        """T_hab > T_ambient -> positive output."""
        result = radiator_capacity(100, 22, MARS_AMBIENT_K)
        assert result > 0

    def test_radiator_capacity_cold(self):
        """T_hab < T_ambient -> returns 0 (can't radiate)."""
        # T_hab_K = -70 + 273.15 = 203.15 K < 210 K
        result = radiator_capacity(100, -70, MARS_AMBIENT_K)
        assert result == 0.0

    def test_radiator_capacity_equal_temps(self):
        """T_hab = T_ambient -> returns 0."""
        # T_hab_K = 210 - 273.15 = -63.15 C
        result = radiator_capacity(100, -63.15, MARS_AMBIENT_K)
        assert result == 0.0

    def test_radiator_capacity_stefan_boltzmann(self):
        """Verify Stefan-Boltzmann formula."""
        area = 50
        T_hab_C = 25
        T_hab_K = T_hab_C + 273.15
        T_ambient_K = MARS_AMBIENT_K

        expected = (area * RADIATOR_EMISSIVITY * STEFAN_BOLTZMANN *
                    (T_hab_K**4 - T_ambient_K**4))
        result = radiator_capacity(area, T_hab_C, T_ambient_K)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_radiator_capacity_negative_area(self):
        """Negative area -> 0."""
        result = radiator_capacity(-100, 22, MARS_AMBIENT_K)
        assert result == 0.0


class TestThermalBalance:
    """Tests for thermal_balance function."""

    def test_thermal_balance_equilibrium(self):
        """Q_in = Q_out -> delta_T ~ 0, status nominal."""
        result = thermal_balance(5000, 5000)
        assert abs(result["delta_T_per_hour"]) < 1e-6
        assert result["time_to_critical_hours"] == float('inf')
        assert result["status"] == "nominal"

    def test_thermal_balance_heating(self):
        """Q_in > Q_out -> delta_T > 0, time_to_critical finite."""
        result = thermal_balance(10000, 5000)
        assert result["delta_T_per_hour"] > 0
        assert result["time_to_critical_hours"] < float('inf')
        assert result["time_to_critical_hours"] > 0

    def test_thermal_balance_cooling(self):
        """Q_in < Q_out -> delta_T < 0."""
        result = thermal_balance(3000, 8000)
        assert result["delta_T_per_hour"] < 0

    def test_thermal_balance_formula(self):
        """Verify delta_T formula."""
        Q_in = 6000
        Q_out = 4000
        expected_delta = (Q_in - Q_out) * 3600 / HAB_THERMAL_MASS_J_PER_K

        result = thermal_balance(Q_in, Q_out)
        assert result["delta_T_per_hour"] == pytest.approx(expected_delta, rel=1e-6)

    def test_thermal_balance_status_nominal(self):
        """Small rate -> nominal."""
        # delta_T = 0.3 C/hr -> nominal (within [-0.5, 0.5])
        # delta_T = (Q_in - Q_out) * 3600 / 5e6
        # 0.3 = (Q_in - Q_out) * 3600 / 5e6
        # Q_in - Q_out = 0.3 * 5e6 / 3600 = 416.67
        result = thermal_balance(5417, 5000)
        assert result["status"] == "nominal"

    def test_thermal_balance_status_stressed(self):
        """Medium rate -> stressed (delta_T in [-2, 2] but outside [-0.5, 0.5])."""
        # delta_T = 1.0 C/hr -> stressed
        # Q_in - Q_out = 1.0 * 5e6 / 3600 = 1388.89
        result = thermal_balance(6400, 5000)
        assert result["status"] == "stressed"

    def test_thermal_balance_status_critical(self):
        """Large rate -> critical (delta_T outside [-2, 2])."""
        # delta_T = 3.0 C/hr -> critical
        # Q_in - Q_out = 3.0 * 5e6 / 3600 = 4166.67
        result = thermal_balance(10000, 5000)
        assert result["status"] == "critical"


# =============================================================================
# ATMOSPHERE FUNCTION TESTS
# =============================================================================

class TestMoxieO2:
    """Tests for moxie_o2 function."""

    def test_moxie_o2_basic(self):
        """1 unit full power -> ~0.132 kg/day."""
        result = moxie_o2(1, 1000)  # Plenty of power
        expected = MOXIE_O2_G_PER_HR * 24 / 1000
        assert result == pytest.approx(expected, rel=1e-6)

    def test_moxie_o2_power_limited(self):
        """Low power -> reduced output."""
        full_power = moxie_o2(2, 1000)  # 2 units need 600W, have 1000
        low_power = moxie_o2(2, 300)    # 2 units need 600W, have 300 (50%)
        assert low_power == pytest.approx(full_power * 0.5, rel=1e-6)

    def test_moxie_o2_zero_units(self):
        """0 units -> 0 output."""
        result = moxie_o2(0, 1000)
        assert result == 0.0

    def test_moxie_o2_zero_power(self):
        """0 power -> 0 output."""
        result = moxie_o2(2, 0)
        assert result == 0.0

    def test_moxie_o2_multiple_units(self):
        """Multiple units scale linearly."""
        one_unit = moxie_o2(1, 500)
        three_units = moxie_o2(3, 1500)  # Same power per unit
        assert three_units == pytest.approx(one_unit * 3, rel=1e-6)


class TestSabatier:
    """Tests for sabatier function."""

    def test_sabatier_stoichiometry(self):
        """Mass balance: inputs ~ outputs x efficiency."""
        # Stoichiometry: 44g CO2 + 8g H2 -> 16g CH4 + 36g H2O
        # Total input: 52g, Total output: 52g (before efficiency)
        co2_kg = 0.044  # 44g = 1 mol
        h2_kg = 0.008   # 8g = 4 mol
        result = sabatier(co2_kg, h2_kg, efficiency=1.0)

        # Expected: 16g CH4, 36g H2O
        assert result["ch4_kg"] == pytest.approx(0.016, rel=0.01)
        assert result["h2o_kg"] == pytest.approx(0.036, rel=0.01)

    def test_sabatier_limiting_h2(self):
        """Less H2 than CO2 -> H2 limits output."""
        co2_kg = 1.0   # Excess CO2
        h2_kg = 0.001  # Only 1g H2 -> 0.5 mol H2 -> 0.125 mol CO2 reacted
        result = sabatier(co2_kg, h2_kg, efficiency=1.0)

        # H2 limits: 0.5 mol H2 reacts with 0.125 mol CO2
        # Products: 0.125 mol CH4 (2g), 0.25 mol H2O (4.5g)
        assert result["ch4_kg"] < 0.005  # Should be ~0.002 kg
        assert result["h2o_kg"] < 0.01   # Should be ~0.0045 kg

    def test_sabatier_limiting_co2(self):
        """Less CO2 than H2 -> CO2 limits output."""
        co2_kg = 0.001  # Only 1g CO2 -> 0.0227 mol CO2
        h2_kg = 1.0     # Excess H2
        result = sabatier(co2_kg, h2_kg, efficiency=1.0)

        # CO2 limits: 0.0227 mol CO2 reacts
        # Products: 0.0227 mol CH4, 0.0455 mol H2O
        expected_ch4 = 0.0227 * CH4_MOLAR_MASS / 1000  # ~0.000364 kg
        expected_h2o = 0.0455 * H2O_MOLAR_MASS / 1000  # ~0.000820 kg
        assert result["ch4_kg"] == pytest.approx(expected_ch4, rel=0.02)
        assert result["h2o_kg"] == pytest.approx(expected_h2o, rel=0.02)

    def test_sabatier_efficiency(self):
        """Efficiency factor applied to products."""
        result_full = sabatier(0.044, 0.008, efficiency=1.0)
        result_85 = sabatier(0.044, 0.008, efficiency=0.85)

        assert result_85["ch4_kg"] == pytest.approx(result_full["ch4_kg"] * 0.85, rel=1e-6)
        assert result_85["h2o_kg"] == pytest.approx(result_full["h2o_kg"] * 0.85, rel=1e-6)

    def test_sabatier_zero_inputs(self):
        """Zero inputs -> zero outputs."""
        assert sabatier(0, 0.008) == {"ch4_kg": 0.0, "h2o_kg": 0.0}
        assert sabatier(0.044, 0) == {"ch4_kg": 0.0, "h2o_kg": 0.0}


class TestHumanO2:
    """Tests for human_o2 function."""

    def test_human_o2_basic(self):
        """10 crew -> 8.4 kg/day."""
        result = human_o2(10)
        assert result == pytest.approx(8.4, rel=1e-6)

    def test_human_o2_zero(self):
        """0 crew -> 0."""
        assert human_o2(0) == 0.0

    def test_human_o2_negative(self):
        """Negative crew -> 0."""
        assert human_o2(-5) == 0.0

    def test_human_o2_uses_constant(self):
        """Uses HUMAN_O2_KG_PER_DAY."""
        result = human_o2(1)
        assert result == HUMAN_O2_KG_PER_DAY


class TestHumanCo2:
    """Tests for human_co2 function."""

    def test_human_co2_basic(self):
        """10 crew -> 10.0 kg/day."""
        result = human_co2(10)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_human_co2_zero(self):
        """0 crew -> 0."""
        assert human_co2(0) == 0.0

    def test_human_co2_negative(self):
        """Negative crew -> 0."""
        assert human_co2(-5) == 0.0

    def test_human_co2_uses_constant(self):
        """Uses HUMAN_CO2_KG_PER_DAY."""
        result = human_co2(1)
        assert result == HUMAN_CO2_KG_PER_DAY


class TestAtmosphereBalance:
    """Tests for atmosphere_balance function."""

    def test_atmosphere_balance_positive(self):
        """Production > consumption with adequate CO2 scrubbing -> nominal."""
        # For nominal: net_o2 >= 0 AND co2_buildup <= 0
        # CO2 produced = o2_consumption * (CO2/O2 ratio) = 4.2 * (1.0/0.84) = 5.0 kg/day
        # Need co2_scrub >= 5.0 for nominal status
        result = atmosphere_balance(5.0, 4.2, 5.0)
        assert result["net_o2_kg"] == pytest.approx(0.8, rel=1e-6)
        assert result["o2_days_reserve"] == float('inf')
        assert result["co2_buildup_rate"] <= 0  # No CO2 buildup
        assert result["status"] == "nominal"

    def test_atmosphere_balance_deficit(self):
        """Production < consumption -> stressed/critical."""
        # Large deficit -> critical
        result = atmosphere_balance(1.0, 8.4, 5.0)
        assert result["net_o2_kg"] < 0
        assert result["o2_days_reserve"] < float('inf')
        # With 100kg reserve and -7.4 kg/day deficit, ~13.5 days
        assert result["o2_days_reserve"] < 30
        assert result["status"] == "critical"

    def test_atmosphere_balance_stressed(self):
        """Small deficit with >30 days reserve -> stressed."""
        # net = 4.0 - 4.2 = -0.2 kg/day
        # days = 100 / 0.2 = 500 days
        result = atmosphere_balance(4.0, 4.2, 5.0)
        assert result["net_o2_kg"] < 0
        assert result["o2_days_reserve"] > 30
        assert result["status"] == "stressed"

    def test_atmosphere_balance_co2_buildup(self):
        """CO2 buildup when scrubbing < production."""
        # 10 crew consume 8.4 kg O2, produce 10 kg CO2
        # CO2 buildup = 10 - scrub
        result = atmosphere_balance(8.4, 8.4, 5.0)
        # CO2 produced = 8.4 * (1.0/0.84) = 10 kg/day
        # Buildup = 10 - 5 = 5 kg/day
        assert result["co2_buildup_rate"] == pytest.approx(5.0, rel=0.01)

    def test_atmosphere_balance_co2_critical(self):
        """High CO2 buildup -> critical."""
        result = atmosphere_balance(8.4, 8.4, 0.0)  # No scrubbing
        assert result["co2_buildup_rate"] > 1.0
        assert result["status"] == "critical"


# =============================================================================
# RESOURCE FUNCTION TESTS
# =============================================================================

class TestWaterCycle:
    """Tests for water_cycle function."""

    def test_water_cycle_basic(self):
        """10 crew, 98% recovery -> net_loss = 0.6 L/day."""
        result = water_cycle(10, ISS_WATER_RECOVERY)
        assert result["consumed_L"] == pytest.approx(30.0, rel=1e-6)
        assert result["recovered_L"] == pytest.approx(29.4, rel=1e-6)
        assert result["net_loss_L"] == pytest.approx(0.6, rel=1e-6)

    def test_water_cycle_perfect(self):
        """100% recovery -> net_loss = 0."""
        result = water_cycle(10, 1.0)
        assert result["net_loss_L"] == 0.0

    def test_water_cycle_no_recovery(self):
        """0% recovery -> net_loss = consumed."""
        result = water_cycle(10, 0.0)
        assert result["net_loss_L"] == result["consumed_L"]

    def test_water_cycle_zero_crew(self):
        """0 crew -> all zeros."""
        result = water_cycle(0)
        assert result["consumed_L"] == 0.0
        assert result["recovered_L"] == 0.0
        assert result["net_loss_L"] == 0.0

    def test_water_cycle_default_recovery(self):
        """Default recovery is ISS_WATER_RECOVERY."""
        result = water_cycle(5)
        expected_consumed = 5 * HUMAN_WATER_L_PER_DAY
        expected_recovered = expected_consumed * ISS_WATER_RECOVERY
        assert result["consumed_L"] == expected_consumed
        assert result["recovered_L"] == expected_recovered


class TestFoodRequirement:
    """Tests for food_requirement function."""

    def test_food_requirement_basic(self):
        """10 crew -> 20000 kcal/day."""
        result = food_requirement(10)
        assert result == pytest.approx(20000, rel=1e-6)

    def test_food_requirement_zero(self):
        """0 crew -> 0."""
        assert food_requirement(0) == 0.0

    def test_food_requirement_negative(self):
        """Negative crew -> 0."""
        assert food_requirement(-5) == 0.0

    def test_food_requirement_uses_constant(self):
        """Uses FOOD_KCAL_PER_DAY."""
        result = food_requirement(1)
        assert result == FOOD_KCAL_PER_DAY


class TestPowerBudget:
    """Tests for power_budget function."""

    def test_power_budget_surplus(self):
        """Generation > consumption -> nominal."""
        result = power_budget(20000, 10000, 20000)
        assert result["total_generation_W"] == 30000
        assert result["net_power_W"] == 10000
        assert result["reserve_margin_pct"] == pytest.approx(33.33, rel=0.01)
        assert result["status"] == "nominal"

    def test_power_budget_deficit(self):
        """Generation < consumption -> critical."""
        result = power_budget(5000, 5000, 15000)
        assert result["total_generation_W"] == 10000
        assert result["net_power_W"] == -5000
        assert result["reserve_margin_pct"] < 0
        assert result["status"] == "critical"

    def test_power_budget_stressed(self):
        """Reserve margin 5-20% -> stressed."""
        # 10000 total, 8500 consumption -> 15% margin
        result = power_budget(5000, 5000, 8500)
        assert result["reserve_margin_pct"] == pytest.approx(15, rel=0.01)
        assert result["status"] == "stressed"

    def test_power_budget_edge_nominal(self):
        """Reserve margin just above 20% -> nominal."""
        # 10000 total, 7900 consumption -> 21% margin
        result = power_budget(5000, 5000, 7900)
        assert result["reserve_margin_pct"] > 20
        assert result["status"] == "nominal"

    def test_power_budget_zero_generation(self):
        """Zero generation with consumption -> critical."""
        result = power_budget(0, 0, 1000)
        assert result["status"] == "critical"


class TestIsruClosure:
    """Tests for isru_closure function."""

    def test_isru_closure_full(self):
        """Production = consumption -> 1.0."""
        production = {"o2": 10, "water": 20}
        consumption = {"o2": 10, "water": 20}
        assert isru_closure(production, consumption) == 1.0

    def test_isru_closure_partial(self):
        """Production < consumption -> ratio < 1.0."""
        production = {"o2": 5, "water": 10}
        consumption = {"o2": 10, "water": 20}
        # Production sum: 15, Consumption sum: 30
        assert isru_closure(production, consumption) == pytest.approx(0.5, rel=1e-6)

    def test_isru_closure_zero_consumption(self):
        """consumption=0 -> ratio = 1.0."""
        production = {"o2": 10}
        consumption = {}
        assert isru_closure(production, consumption) == 1.0

    def test_isru_closure_excess_production(self):
        """Production > consumption -> capped at 1.0."""
        production = {"o2": 50, "water": 50}
        consumption = {"o2": 10, "water": 10}
        assert isru_closure(production, consumption) == 1.0

    def test_isru_closure_empty_production(self):
        """Empty production with consumption -> 0."""
        production = {}
        consumption = {"o2": 10}
        assert isru_closure(production, consumption) == 0.0


# =============================================================================
# RECEIPT FUNCTION TESTS
# =============================================================================

class TestEmitSubsystemReceipt:
    """Tests for emit_subsystem_receipt function."""

    def test_emit_subsystem_receipt_fields(self, capsys):
        """Receipt has subsystem, metrics, status."""
        metrics = {"Q_in_W": 5000, "Q_out_W": 4500}
        result = emit_subsystem_receipt("thermal", metrics, "nominal")

        # Check receipt structure
        assert result["receipt_type"] == "subsystem"
        assert result["subsystem"] == "thermal"
        assert result["metrics"] == metrics
        assert result["status"] == "nominal"
        assert "ts" in result
        assert "tenant_id" in result
        assert "payload_hash" in result

    def test_emit_subsystem_receipt_atmosphere(self, capsys):
        """Atmosphere subsystem receipt."""
        metrics = {"net_o2_kg": 0.5, "co2_buildup_rate": 0.1}
        result = emit_subsystem_receipt("atmosphere", metrics, "stressed")

        assert result["subsystem"] == "atmosphere"
        assert result["status"] == "stressed"

    def test_emit_subsystem_receipt_resource(self, capsys):
        """Resource subsystem receipt."""
        metrics = {"water_L": 100, "power_W": 20000}
        result = emit_subsystem_receipt("resource", metrics, "critical")

        assert result["subsystem"] == "resource"
        assert result["status"] == "critical"

    def test_emit_subsystem_receipt_prints_json(self, capsys):
        """Receipt is printed to stdout as JSON."""
        emit_subsystem_receipt("thermal", {"test": 1}, "nominal")
        captured = capsys.readouterr()
        assert "subsystem" in captured.out
        assert "thermal" in captured.out


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_thermal_system_integration(self):
        """Test complete thermal system calculation."""
        # 10 crew, 200m2 solar, 2 nuclear units, 100m2 radiator at 22C
        crew = 10
        array_m2 = 200
        flux = SOLAR_FLUX_MAX
        kilopower_units = 2
        radiator_m2 = 100
        T_hab_C = 22

        # Calculate inputs
        q_solar = solar_input(array_m2, flux)
        q_nuclear = nuclear_input(kilopower_units)
        q_metabolic = metabolic_heat(crew)
        q_equipment = equipment_heat(q_solar + q_nuclear)

        Q_in = q_solar + q_nuclear + q_metabolic + q_equipment
        Q_out = radiator_capacity(radiator_m2, T_hab_C)

        # Check thermal balance
        balance = thermal_balance(Q_in, Q_out)
        assert "delta_T_per_hour" in balance
        assert "time_to_critical_hours" in balance
        assert "status" in balance

    def test_atmosphere_system_integration(self):
        """Test complete atmosphere system calculation."""
        crew = 10
        moxie_units = 2
        power_W = 1000

        # Calculate O2 production and consumption
        o2_prod = moxie_o2(moxie_units, power_W)
        o2_cons = human_o2(crew)
        co2_prod = human_co2(crew)

        # Sabatier converts some CO2
        h2_available = 0.5  # kg
        sabatier_result = sabatier(co2_prod * 0.5, h2_available)

        # Check atmosphere balance
        co2_scrub = co2_prod * 0.5  # Assume we scrub half
        balance = atmosphere_balance(o2_prod, o2_cons, co2_scrub)
        assert "net_o2_kg" in balance
        assert "status" in balance

    def test_resource_system_integration(self):
        """Test complete resource system calculation."""
        crew = 10
        solar_W = solar_input(200, SOLAR_FLUX_MAX)
        nuclear_W = nuclear_input(2)

        # Water cycle
        water = water_cycle(crew)
        assert water["consumed_L"] == crew * HUMAN_WATER_L_PER_DAY

        # Food requirement
        food = food_requirement(crew)
        assert food == crew * FOOD_KCAL_PER_DAY

        # Power budget
        consumption = 25000  # Typical colony consumption
        power = power_budget(solar_W, nuclear_W, consumption)
        assert "status" in power

        # ISRU closure
        production = {"water": water["recovered_L"], "food": 0}
        consumption_dict = {"water": water["consumed_L"], "food": food * FOOD_KG_PER_KCAL}
        closure = isru_closure(production, consumption_dict)
        assert 0 <= closure <= 1
