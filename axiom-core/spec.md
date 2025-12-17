# AXIOM-CORE v2 Specification

## The Multiplicative Paradigm Shift

**v1 Question:** "What's the optimal allocation between propulsion and autonomy?"
**v2 Answer:** "Wrong question. They multiply, not trade."

```
Build Rate B = c × A^α × P
```

Where:
- B = civilization build rate (toward 10^6 person-equivalent threshold)
- A = autonomy level (compounds via fleet learning, meta-loops)
- P = propulsion/launch cadence
- α ≈ 1.8 → superlinear autonomy scaling dominates long-term
- c = constant (initial conditions)

## The Core Equations

### v1 (Still Valid)
```
sovereignty = internal_rate > external_rate
```

### v2 Extension
```
Build Rate B = c × A^α × P  (multiplicative, not additive)
Person-Equivalent = (decision_capacity / 10) × (τ_ref / τ) × expertise
Sovereignty Threshold = 10^6 person-equivalents
```

## Grok Timeline Table (Validation Target)

| Pivot fraction | Annual multiplier | Years to threshold | Delay vs 40% |
|----------------|-------------------|-------------------|--------------|
| 40% (recommended) | ~2.5-3.0× | 12-15 | baseline |
| 20-25% | ~1.6-2.0× | 18-22 | +6-8 years |
| <15% | ~1.2-1.4× | 25-35+ | +12-20 years |
| ~0% (propulsion-only) | ~1.1× | 40-60 (or never) | existential |

## Architecture

```
axiom-core/
├── spec.md                    # This file
├── cli.py                     # Command-line interface
├── receipts.jsonl             # Append-only ledger
├── src/
│   ├── __init__.py
│   ├── core.py                # dual_hash, emit_receipt, StopRule, merkle
│   ├── entropy_shannon.py     # Shannon H, rate calculations
│   ├── sovereignty.py         # v1 equation + v2 person-equivalent
│   ├── compounding.py         # Compounding model + τ_velocity
│   ├── validate.py            # Statistical validation
│   ├── plot_curve.py          # Visualization
│   ├── prove.py               # Receipt chain verification
│   │
│   ├── build_rate.py          # v2: B = c × A^α × P
│   ├── stage_gate.py          # v2: Dynamic allocation (30% → 40%)
│   ├── calibration.py         # v2: α estimation from fleet data
│   ├── timeline.py            # v2: Year-to-threshold projections
│   ├── tier_risk.py           # v2: 3-tier probability × impact
│   └── leading_indicators.py  # v2: Observable proxy monitoring
├── tests/
│   ├── test_build_rate.py
│   ├── test_stage_gate.py
│   ├── test_calibration.py
│   ├── test_timeline.py
│   ├── test_tier_risk.py
│   ├── test_leading_indicators.py
│   ├── test_sovereignty_curve.py
│   ├── test_compounding.py
│   └── conftest.py
└── data/
    ├── bandwidth_mars_2025.json
    └── delay_mars_2025.json
```

## v2 Constants (from Grok)

| Constant | Value | Source |
|----------|-------|--------|
| THRESHOLD_PERSON_EQUIVALENT | 1,000,000 | Grok: "~10^6 person-equivalent" |
| STAGE_GATE_INITIAL | 0.30 | Grok: "30% now" |
| STAGE_GATE_TRIGGER_ALPHA | 1.9 | Grok: "+10% if α>1.9" |
| STAGE_GATE_ESCALATION | 0.10 | Grok: "+10%" |
| STAGE_GATE_WINDOW_MONTHS | 12 | Grok: "in next 12 months" |
| TIER_1_PROB_RANGE | (0.60, 0.80) | Grok: "60-80%" |
| TIER_2_PROB_RANGE | (0.30, 0.50) | Grok: "30-50%" |
| TIER_3_PROB_RANGE | (0.05, 0.15) | Grok: "5-15%" |
| ALPHA_BASELINE | 1.8 | Grok: "calibrated to α=1.8" |

## v2 Receipt Types

| Receipt Type | Module | Key Fields |
|--------------|--------|------------|
| build_rate | build_rate.py | autonomy_level, propulsion_level, build_rate, annual_multiplier |
| stage_gate | stage_gate.py | alpha_measured, trigger_met, new_autonomy_fraction |
| calibration | calibration.py | alpha_estimate, confidence_interval, data_quality_score |
| timeline | timeline.py | years_to_threshold, delay_vs_optimal, annual_multiplier |
| tier_risk | tier_risk.py | tier, probability_range, impact_class, failure_modes |
| leading_indicator | leading_indicators.py | indicator_type, current_value, gap, trend |
| sovereignty_v2 | sovereignty.py | person_equivalent, threshold, gap_to_threshold |
| tau_velocity | compounding.py | velocity_raw, velocity_pct, trend, meets_target |

## Falsifiable Predictions (v2)

| Prediction | Observable | Confirmation | Falsification |
|------------|------------|--------------|---------------|
| 40% allocation → 12-15 years | Starship autonomy + launch cadence | Timeline matches ±2 years | Timeline > 20 years |
| α > 1.9 triggers escalation | FSD/Optimus learning curves | α measured > 1.9 | α < 1.7 sustained |
| Tier 3 risk < 15% at 40% | Mission success during conjunction | No cascading failures | Major conjunction failure |
| τ velocity < -5%/cycle | Decision latency measurements | Sustained improvement | τ stagnates or increases |

## Usage

```bash
# Run all tests
python -m pytest tests/ -v

# Quick validation (Phase 1)
python -c "from src.stage_gate import STAGE_GATE_INITIAL; print(f'Initial: {STAGE_GATE_INITIAL}')"
python -c "from src.timeline import allocation_to_multiplier; print(f'40%: {allocation_to_multiplier(0.40):.2f}')"
python -c "from src.build_rate import compute_build_rate; print(f'B: {compute_build_rate(0.40, 1.0):.4f}')"

# Generate timeline table
python -c "from src.timeline import generate_timeline_table, format_timeline_table; print(format_timeline_table(generate_timeline_table()))"

# Run compounding simulation
python cli.py full
```

## The Finding (v2)

At 40% autonomy allocation with α=1.8:
- Annual multiplier: 2.5-3.0×
- Years to 10^6 person-equivalent: 12-15
- Build rate: B = c × 0.40^1.8 × P = 0.22 × c × P

At 25% autonomy allocation:
- Annual multiplier: 1.6-2.0×
- Years to threshold: 18-22
- Build rate ratio vs 40%: 0.10/0.22 = 0.45× (55% reduction)

**Under-pivoting is not a slower path. It's a different destination: never.**

---

**Hash of this document:** COMPUTE_ON_SAVE
**Version:** 2.0
**Status:** ACTIVE

*Build rate B ≈ c × A^α × P. Autonomy multiplies. The rest is commentary.*
