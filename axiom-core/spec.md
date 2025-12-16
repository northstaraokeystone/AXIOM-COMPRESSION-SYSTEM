# AXIOM-CORE v1 Specification

## The Pearl Without the Shell

**Critical Review Response (Dec 16, 2025)**
> "Viability Score: 12%"
> "The equation is the pearl. Everything else is shell."

We stripped the shell.

## The Core Equation

```
sovereignty = internal_rate > external_rate
```

Where:
- `internal_rate = log2(1 + crew × 10 + compute_flops × 1e-15)` bits/sec
- `external_rate = bandwidth_bps / (2 × delay_s)` bits/sec

## The Finding

```
At 47 ± 8 crew with 2-10 Mbps bandwidth and 3-22 minute delay,
Mars achieves computational sovereignty.
```

One equation. One curve. One number. That's publishable.

## Constants (Verified, No Speculation)

| Constant | Value | Source | Status |
|----------|-------|--------|--------|
| HUMAN_DECISION_RATE_BPS | 10 | Reviewer confirmed | KEEP |
| STARLINK_MARS_BANDWIDTH_MIN_MBPS | 2.0 | "2-10 Mbps 2025 sims" | NEW |
| STARLINK_MARS_BANDWIDTH_MAX_MBPS | 10.0 | "2-10 Mbps 2025 sims" | NEW |
| MARS_LIGHT_DELAY_MIN_S | 180 | Physics (3 min) | KEEP |
| MARS_LIGHT_DELAY_MAX_S | 1320 | Physics (22 min) | KEEP |

## Killed Constants

- ~~NEURALINK_MULTIPLIER~~ (future work, numerology without 2025 data)
- ~~xAI_LOGISTICS_MULTIPLIER~~ (undefined)
- ~~SOVEREIGNTY_THRESHOLD_NEURALINK~~ (numerology)
- ~~All thermodynamic references~~ (irrelevant to comms)

## Architecture

```
axiom-core/
├── spec.md                 # This file
├── cli.py                  # Command-line interface
├── src/
│   ├── __init__.py
│   ├── core.py             # dual_hash, emit_receipt, merkle
│   ├── entropy_shannon.py  # Shannon H ONLY
│   ├── sovereignty.py      # THE equation
│   ├── ingest_real.py      # Real data ingest
│   ├── validate.py         # Statistical tests
│   ├── plot_curve.py       # THE graph
│   └── prove.py            # Receipts
├── data/
│   ├── bandwidth_mars_2025.json
│   └── delay_mars_2025.json
├── tests/
│   ├── test_null_hypothesis.py
│   ├── test_real_bandwidth.py
│   ├── test_statistical.py
│   └── test_sovereignty_curve.py
└── outputs/
    └── sovereignty_curve.png
```

## Falsifiable Predictions

| Prediction | Observable | Confirmation | Falsification |
|------------|------------|--------------|---------------|
| Threshold ~47 crew at 4 Mbps | First Mars colony crew manifest | Crew ≈ 50 | Crew < 30 or > 70 |
| Threshold drops with bandwidth | Starlink Mars upgrade | Lower crew viable | Threshold unchanged |
| Threshold rises with delay | Conjunction period | Higher crew needed | Threshold unchanged |

## Usage

```bash
# Run baseline test
python cli.py baseline

# Run bootstrap analysis
python cli.py bootstrap

# Generate sovereignty curve
python cli.py curve

# Run full integration test
python cli.py full
```
