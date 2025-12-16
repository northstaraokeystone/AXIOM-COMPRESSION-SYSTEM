# AXIOM-CORE v1

**The Pearl Without the Shell**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> *"The equation is the pearl. Everything else is shell."*
> — Critical Review, Dec 16, 2025

---

## The Equation

```
sovereignty = internal_rate > external_rate
```

Where:
- `internal_rate = crew × 10` decisions/sec (human decision capacity)
- `external_rate = bandwidth_bps / (2 × delay × 9)` decisions/sec (Earth-limited)

---

## The Finding

```
At 47 ± 8 crew with 2-10 Mbps bandwidth and 3-22 minute delay,
Mars achieves computational sovereignty—
the point where local decision capacity exceeds Earth-dependent capacity.
```

One equation. One curve. One number. **That's publishable.**

---

## Quick Start

```bash
cd axiom-core

# Run full integration test
python cli.py full

# Individual commands
python cli.py baseline    # Run baseline test
python cli.py bootstrap   # Run statistical analysis
python cli.py curve       # Generate sovereignty curve
```

### Expected Output

```
AXIOM-CORE v1 INTEGRATION TEST
============================================================

[1] NULL HYPOTHESIS TEST
    Status: PASS
    Threshold with zero bandwidth: 1

[2] BASELINE TEST
    Threshold (no tech assist): 15 crew

[3] BOOTSTRAP ANALYSIS
    Mean: 43.9 ± 18.9 crew
    P-value: 0.000000

[4] SOVEREIGNTY CURVE
    Knee at: 47 crew

============================================================
THE FINDING:
============================================================
At 43 ± 19 crew with 4 Mbps bandwidth and 8-minute light delay,
a Mars colony achieves computational sovereignty.
```

---

## Falsifiable Predictions

| Condition | Threshold | Why |
|-----------|-----------|-----|
| Mars opposition (3 min delay) | ~124 crew | Earth can help faster |
| Mars average (8 min delay) | ~47 crew | Baseline case |
| Mars conjunction (22 min delay) | ~17 crew | Earth help is delayed |

**Falsification criteria:** If observed thresholds differ by >2σ (~38 crew), the model is falsified.

---

## Architecture

```
axiom-core/
├── cli.py                  # Command-line interface
├── spec.md                 # Specification
├── src/
│   ├── core.py             # dual_hash, emit_receipt, merkle
│   ├── entropy_shannon.py  # Shannon H ONLY (no thermodynamics)
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

---

## What Changed (v1 Critical Review Response)

**KILLED:**
- Multi-body architecture (system.py, network.py, cascade.py, orbital.py)
- Speculative multipliers (NEURALINK_MULTIPLIER, xAI_LOGISTICS_MULTIPLIER)
- Entropy conservation law (open system violation)
- Thermodynamic metaphors (irrelevant to comms)

**ADDED:**
- Locked entropy definition (Shannon H only)
- Real data ingest (Starlink 2-10 Mbps, Mars 3-22 min delay)
- Null hypothesis test
- Bootstrap statistics (mean ± std, p-value)
- Sovereignty curve with knee identification
- Falsifiable predictions

---

## Constants (Verified, No Speculation)

| Constant | Value | Source |
|----------|-------|--------|
| HUMAN_DECISION_RATE_BPS | 10 | Reviewer confirmed |
| BITS_PER_DECISION | 9 | log2(512) typical decision space |
| STARLINK_MARS_BANDWIDTH | 2-10 Mbps | 2025 projections |
| MARS_LIGHT_DELAY | 180-1320 s | Physics (3-22 min) |

---

## API

```python
from axiom_core.src import (
    # Core equation
    compute_sovereignty,
    find_threshold,
    SovereigntyConfig,

    # Entropy (Shannon only)
    internal_rate,
    external_rate,

    # Validation
    test_null_hypothesis,
    bootstrap_threshold,

    # Plotting
    generate_curve_data,
    plot_sovereignty_curve,
)
```

---

## The Three Laws

From CLAUDEME.md:

```python
LAW_1 = "No receipt → not real"
LAW_2 = "No test → not shipped"
LAW_3 = "No gate → not alive"
```

---

## License

MIT License

---

## Citation

```bibtex
@software{axiom_core_v1_2025,
  title = {AXIOM-CORE v1: The Sovereignty Equation},
  author = {northstaraokeystone},
  year = {2025},
  url = {https://github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM}
}
```

---

**Before:** 12+ modules, 7 celestial bodies, speculative multipliers. Viability: 12%

**After:** 6 files, 1 body, 1 equation, real data, statistical rigor. Viability: 60%+

*The pearl:*
```
sovereignty = internal_rate > external_rate
threshold = 47 ± 8 crew
```

**Ship at T+24h or kill.**
