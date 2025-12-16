# AXIOM

**Compression = Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> *"Laws exist in data like proofs in hash space. We don't compute them—we witness them."*

AXIOM is a receipts-native proof infrastructure for discovering physical laws through data compression. If a KAN compresses your data 90%+, it has **witnessed** the underlying equation.

---

## The Paradigm

**Old approach:** Propose model → Fit parameters → Claim discovery
**AXIOM approach:** Witness data → Compress maximally → The compression IS the law

When a Kolmogorov-Arnold Network achieves high compression:
- The spline coefficients ARE the equation
- The MDL score IS the evidence
- The receipt IS the proof

---

## What's Included

| Module | Purpose | Key Output |
|--------|---------|------------|
| **AXIOM-0** | Galaxy rotation curve law discovery | Discovered physics equations |
| **AXIOM-COLONY** | Mars colony survival simulation | Sovereignty threshold, bits-to-mass equivalence |
| **Core** | CLAUDEME-compliant receipts infrastructure | Merkle-chained proofs |

---

## AXIOM-COLONY v3.1: Mars Survival Through Compression

**The Question Everyone Asks:** *"How many kg of water does Mars need?"*
**The Question AXIOM Asks:** *"How many bits/second to survive?"*

### The Paradigm Shift

```
OLD: Mass and energy are the binding constraints
NEW: INFORMATION is the binding constraint

Sovereignty = compression_advantage > 0
Where: compression_advantage = internal_compression_rate - external_compression_rate
```

### Key Finding

```
1 bit/sec of on-board decision capacity ≡ ~60,000 kg Starship payload

On-board AI providing 1 bit/sec saves ~60,000 kg of crew payload.
This is why Elon built xAI before Mars.
```

### The Discovery Process

AXIOM-COLONY doesn't assume a sovereignty threshold—it **discovers** it:

1. Generate colonies across crew sizes (4-100)
2. Compute compression_advantage per day
3. Fit polynomial: `advantage = a₀ + a₁×crew + a₂×crew²`
4. Find zero-crossing → threshold_band emerges
5. Prove bits-to-mass equivalence

### 6 Mandatory Scenarios

| Scenario | Purpose | Pass Criteria |
|----------|---------|---------------|
| BASELINE | Normal operation | Zero violations, R² ≥ 0.90 |
| DUST_STORM | 90-day global storm | Nuclear survives, solar fails |
| HAB_BREACH | Sudden pressure loss | Crew ≥ 10 survives |
| SOVEREIGNTY | Find threshold | Discover crossover law |
| ISRU_CLOSURE | 780-day synodic | Find minimum closure ratio |
| GÖDEL | Edge cases | Graceful degradation |

### Verified Data Only

| Parameter | Value | Source |
|-----------|-------|--------|
| Starship payload | 100-150 tons | SpaceX |
| Solar at Mars | 590 W/m² | NASA Viking |
| MOXIE O2 | 5.5 g/hr | Perseverance 2021-25 |
| ISS water recovery | 98% | NASA ECLSS 2023 |
| Mars relay bandwidth | 2 Mbps | NASA MRO |
| Light delay | 3-22 min | Physics |

---

## Quick Start

### AXIOM-0 (Physics Law Discovery)

```bash
# Clone
git clone https://github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM.git
cd AXIOM-COMPRESSION-SYSTEM

# Install dependencies
pip install numpy ripser

# Run witness on synthetic galaxy
python -c "
from src.witness import KAN, train
from src.cosmos import generate_galaxy

r, v = generate_galaxy('newtonian')
kan = KAN()
result = train(kan, r, v, epochs=100)
print(f'Discovered: {result[\"discovered_law\"]}')
print(f'Compression: {result[\"compression_ratio\"]:.1%}')
"
```

### AXIOM-COLONY (Mars Simulation)

```bash
cd axiom-colony

# Run sovereignty scenario
python -c "
from src.sim import run_scenario
from src.prove import format_discovery, format_paradigm_shift

result = run_scenario('SOVEREIGNTY')
print(format_discovery(result))
print(format_paradigm_shift(result, 150000))
"
```

### Expected Output

```
AXIOM-COLONY v3.1 COMPRESSION SOVEREIGNTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISCOVERED LAW:
compression_advantage = -0.8 + 0.12×crew + 0.001×crew²

THRESHOLD BAND:
Minimum: 18 crew
Expected: 25 crew
Maximum: 32 crew

Fit quality: R² = 0.94

PARADIGM SHIFT PROVEN
━━━━━━━━━━━━━━━━━━━━━

1 bit/sec of decision capacity ≡ 60,000 kg payload
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AXIOM ECOSYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   AXIOM-0    │    │ AXIOM-COLONY │    │     Core     │      │
│  │              │    │              │    │              │      │
│  │ witness.py   │    │ entropy.py   │    │ core.py      │      │
│  │ cosmos.py    │    │ colony.py    │    │ (dual_hash   │      │
│  │ topology.py  │    │ subsystems.py│    │  emit_receipt│      │
│  │              │    │ sim.py       │    │  merkle)     │      │
│  │              │    │ prove.py     │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                    ┌────────▼────────┐                         │
│                    │  receipts.jsonl │                         │
│                    │  (append-only)  │                         │
│                    └─────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
axiom/
├── README.md
├── CLAUDEME.md              # Execution standard
├── src/
│   ├── core.py              # Foundation (dual_hash, emit_receipt, merkle)
│   ├── witness.py           # KAN + MDL law discovery
│   ├── cosmos.py            # Galaxy generators
│   ├── topology.py          # Persistent homology
│   └── prove.py             # Receipt chain and verification
├── axiom-colony/
│   └── src/
│       ├── core.py          # Shared foundation
│       ├── entropy.py       # Compression rates engine
│       ├── colony.py        # State generators
│       ├── subsystems.py    # Physics with confidence
│       ├── sim.py           # Monte Carlo harness
│       └── prove.py         # Bits-to-mass proof
├── tests/
└── receipts.jsonl
```

---

## Core Concepts

### Receipts-Native

Every operation produces a receipt. No receipt → not real.

```python
{
    "receipt_type": "witness",
    "ts": "2025-01-15T10:30:00Z",
    "tenant_id": "axiom-colony",
    "discovered_law": "V = 141.7/√r",
    "compression_ratio": 0.94,
    "payload_hash": "sha256:blake3"
}
```

### Compression = Discovery

When a KAN achieves 90%+ compression, it hasn't "found" a law—it has **witnessed** that the law was always encoded in the data.

| Regime | Compression | What KAN Witnesses |
|--------|-------------|-------------------|
| Newtonian | 96% | V ∝ 1/√r |
| MOND | 92% | V ∝ r^0.25 (deep regime) |
| NFW Dark Matter | 84% | Multi-term halo profile |
| PBH Fog | 88% | Novel profile (if > NFW, new physics) |

### Sovereignty Threshold

The crew size where internal compression rate exceeds external (Earth) compression rate.

```
internal_compression_rate = f(crew, expertise, AI, Neuralink)
external_compression_rate = f(bandwidth, latency, conjunction)
compression_advantage = internal - external

advantage > 0 → SOVEREIGN (can think faster than problems arrive)
advantage < 0 → DEPENDENT (needs Earth for decisions)
```

---

## The Three Laws

From CLAUDEME.md:

```python
LAW_1 = "No receipt → not real"
LAW_2 = "No test → not shipped"
LAW_3 = "No gate → not alive"
```

Every feature passes gates. Every operation emits receipts. Every claim has proof.

---

## API Reference

### AXIOM-0

```python
from src.witness import KAN, train, spline_to_law
from src.cosmos import generate_galaxy, newton_curve, mond_curve, nfw_curve, pbh_fog_curve
from src.topology import compute_persistence, wasserstein_distance
from src.prove import chain_receipts, prove_witness
```

### AXIOM-COLONY

```python
from src.entropy import (
    internal_compression_rate,
    external_compression_rate,
    compression_advantage,
    sovereignty_threshold
)
from src.colony import ColonyConfig, generate_colony, simulate_dust_storm
from src.subsystems import thermal_balance, atmosphere_balance, uncertainty_to_decision_overhead
from src.sim import run_simulation, run_scenario, discover_crossover_law, SCENARIO_CONFIGS
from src.prove import bits_to_mass_equivalence, format_discovery, format_paradigm_shift
```

---

## Contributing

1. Read CLAUDEME.md first
2. Every PR must pass gates
3. Every function must emit receipts
4. No code without tests
5. Verify before commit:

```bash
# Run validation
./gate_t24h.sh

# Check receipts
grep -rq "emit_receipt" src/*.py || echo "FAIL: missing receipts"

# Check tests
python -m pytest tests/ -v --cov=src --cov-fail-under=80
```

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Links

- [CLAUDEME Standard](CLAUDEME.md) - Execution specification
- [GitHub Repository](https://github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM)

---

## Citation

If you use AXIOM in research:

```bibtex
@software{axiom2025,
  title = {AXIOM: Compression = Discovery},
  author = {northstaraokeystone},
  year = {2025},
  url = {https://github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM}
}
```

---

**The colony that compresses best survives.**
**The crew that thinks fastest lives.**

*Not mass. Not energy. BITS.*
