# SpaceProof

**Space-grade proof infrastructure. No receipt, not real.**

Part of the ProofChain series: SpaceProof | SpendProof | ClaimProof | VoteProof | OriginProof | GreenProof

> **v3.0: Multi-Tier Autonomy Network** - Scale from single colony to 1M colonists across 1000 colonies with AI/Neuralink augmentation

## What SpaceProof Does

| Module | Purpose | Value |
|--------|---------|-------|
| compress | 10x telemetry compression | Bandwidth savings |
| witness | Physics law discovery (KAN/MDL) | Accelerated R&D |
| sovereignty | Autonomy threshold calculation | Mars crew sizing |
| detect | Entropy anomaly detection | Fraud detection |
| ledger | Receipt storage | Full audit trail |
| anchor | Merkle proofs | Tamper-proof |
| loop | 60-second SENSE->ACTUATE cycle | Automated improvement |
| **v3.0 Modules** | | |
| starship_fleet | 1000 launches/year model | Entropy delivery |
| colony_network | Multi-colony network dynamics | 1M colonist scale |
| decision_augmented | AI (5x) / Neuralink (20x) augmentation | Crew optimization |
| sovereignty_network | Network sovereignty threshold | Distributed autonomy |
| autonomy_tiers | LEO/Mars/Deep-space tiers | Light-delay adaptation |

## Quick Start

```bash
# Install
pip install -e .

# Run tests
pytest tests/ -v

# CLI usage
python cli.py --help

# Test receipt emission
python cli.py --test

# Run with stakeholder config
python cli.py --config xai --test
python cli.py --config doge --test
```

### v3.0 Multi-Tier Features

```bash
# AI Augmentation: 4 crew + AI = 20 crew human-only
python -c "
from spaceproof.decision_augmented import effective_crew_size
print(f'4 crew + AI = {effective_crew_size(4, \"ai\")} effective crew')
"

# Network Sovereignty
python -c "
from spaceproof.domain.colony_network import initialize_network
from spaceproof.sovereignty_network import network_sovereignty_threshold
net = initialize_network(100, 1000, seed=42)
result = network_sovereignty_threshold(net)
print(f'Threshold: {result[\"threshold_colonies\"]} colonies for sovereignty')
"

# Run NETWORK scenario (1M colonist validation)
python -c "
from spaceproof.sim.scenarios.network import run_scenario, NetworkScenarioConfig
config = NetworkScenarioConfig(n_colonies=10, duration_days=30)
result = run_scenario(config)
print(f'NETWORK: {\"PASS\" if result.passed else \"FAIL\"}')
"
```

## Stakeholder Configs

| Config | Stakeholder | Primary Modules | Key Value |
|--------|-------------|-----------------|-----------|
| `configs/xai.yaml` | Elon/xAI | compress, witness, sovereignty | Mars crew threshold |
| `configs/doge.yaml` | DOGE | ledger, detect, anchor | $162B fraud target |
| `configs/dot.yaml` | DOT | compress, ledger, detect | Compliance |
| `configs/defense.yaml` | Defense | compress, ledger, anchor | Fire-control lineage |
| `configs/nro.yaml` | NRO | compress, anchor, sovereignty | Constellation governance |

## Module Taxonomy

```
core.py (foundation)
    |
compress.py <- witness.py
    |           |
sovereignty.py <- detect.py <- decision_augmented.py (v3.0)
    |                              |
sovereignty_network.py (v3.0) <- colony_network.py (v3.0)
    |                              |
  ledger.py <- anchor.py       starship_fleet.py (v3.0)
    |                              |
  loop.py (harness) <-------- autonomy_tiers.py (v3.0)
```

## Complete Architecture

```
spaceproof/
├── core.py                       # Foundation: dual_hash, emit_receipt, merkle
├── compress.py                   # 10x telemetry compression
├── witness.py                    # Physics law discovery (KAN/MDL)
├── detect.py                     # Entropy anomaly detection
├── ledger.py                     # Receipt storage
├── anchor.py                     # Merkle proofs
├── loop.py                       # 60-second SENSE→ACTUATE cycle
│
├── sovereignty_core.py           # Core sovereignty calculations
├── sovereignty_network.py        # Network sovereignty threshold
├── decision_augmented.py         # AI/Neuralink augmentation
│
├── cli/                          # Command-line interface
│   ├── __init__.py
│   ├── args.py
│   └── dispatch.py
│
├── domain/                       # Domain generators
│   ├── galaxy.py                 # Galaxy rotation curves
│   ├── colony.py                 # Mars colony simulation
│   ├── telemetry.py              # Fleet telemetry
│   ├── starship_fleet.py         # 1000 launches/year model
│   └── colony_network.py         # 1M colonist multi-colony network
│
├── engine/                       # Core engine components
│   ├── entropy.py                # Entropy calculations
│   ├── gates.py                  # Gate validation
│   ├── protocol.py               # Protocol interfaces
│   ├── receipts.py               # Receipt primitives
│   └── saga.py                   # Saga orchestration
│
├── sovereignty/                  # Sovereignty calculations
│   └── mars/                     # Mars sovereignty calculator (v3.0)
│       ├── api.py                # Public API
│       ├── constants.py          # Mars constants
│       ├── crew_matrix.py        # Crew skill matrix
│       ├── decision_capacity.py  # Decision capacity calculations
│       ├── integrator.py         # Sovereignty score integration
│       ├── life_support.py       # Life support constraints
│       ├── monte_carlo.py        # Monte Carlo validation
│       ├── resources.py          # ISRU and resource calculations
│       └── scenarios.py          # Test scenarios
│
├── tiers/                        # Multi-tier autonomy (v3.0)
│   └── autonomy_tiers.py         # LEO/Mars/Deep-space tiers
│
└── sim/                          # Simulation framework
    ├── monte_carlo.py            # Monte Carlo engine
    ├── dimensions/               # Scenario dimensions
    │   ├── foundation.py
    │   ├── intermediate.py
    │   ├── advanced.py
    │   └── ultimate.py
    └── scenarios/                # Test scenarios
        ├── baseline.py
        ├── stress.py
        ├── genesis.py
        ├── godel.py
        ├── singularity.py
        ├── thermodynamic.py
        ├── network.py            # 1M colonist validation
        └── adversarial.py        # DoD hostile audit
```

## Domain Generators

| Domain | Path | Purpose |
|--------|------|---------|
| galaxy | spaceproof/domain/galaxy.py | Galaxy rotation curve generation |
| colony | spaceproof/domain/colony.py | Mars colony state simulation |
| telemetry | spaceproof/domain/telemetry.py | Fleet telemetry (Tesla/Starlink/SpaceX) |
| starship_fleet | spaceproof/domain/starship_fleet.py | 1000 launches/year model (v3.0) |
| colony_network | spaceproof/domain/colony_network.py | Multi-colony 1M colonist network (v3.0) |

## Mars Sovereignty Calculator (v3.0)

Complete implementation in `spaceproof/sovereignty/mars/`:

| Module | Purpose |
|--------|---------|
| api.py | Calculate Mars crew sovereignty score |
| constants.py | Mars-specific constants (light delay, ISRU targets, etc.) |
| crew_matrix.py | Skill distribution and crew optimization |
| decision_capacity.py | Decision-making capacity calculations |
| integrator.py | Integrate all factors into sovereignty score |
| life_support.py | ECLSS and habitat constraints |
| monte_carlo.py | Monte Carlo validation of sovereignty |
| resources.py | ISRU, water recycling, food production |
| scenarios.py | Baseline, research-validated scenarios |

## SLOs

| Module | Metric | Threshold | Stoprule |
|--------|--------|-----------|----------|
| compress | compression_ratio | >=10 | FAIL |
| compress | recall | >=0.999 | FAIL |
| witness | training_time | <=60s | WARN |
| detect | false_positive_rate | <0.01 | FAIL |
| loop | cycle_time | <=60s | WARN |
| anchor | verify_time | <=2s | WARN |
| **v3.0 SLOs** | | | |
| colony_network | entropy_stable_ratio | >=0.95 | FAIL |
| sovereignty_network | sovereign_colonies | >=MIN | WARN |
| starship_fleet | launches_per_year | >=1000 | WARN |
| adversarial | attack_detection_rate | >=0.99 | FAIL |

## Core Primitives

All modules import from `spaceproof.core`:

```python
from spaceproof.core import dual_hash, emit_receipt, merkle, StopRule
```

- `dual_hash(data)` - SHA256:BLAKE3 format
- `emit_receipt(type, data)` - Create CLAUDEME-compliant receipt
- `merkle(items)` - Compute Merkle root
- `StopRule` - Exception for stoprule violations

## Gates

### T+2h: SKELETON
- [ ] spec.md exists
- [ ] ledger_schema.json exists
- [ ] cli.py emits receipt

### T+24h: MVP
- [ ] All modules importable
- [ ] All tests pass
- [ ] All receipts emit

### T+48h: HARDENED
- [ ] 80% coverage on core
- [ ] All configs work
- [ ] Documentation complete

## The Physics

Information theory unifies all domains:
- **Compression = Discovery** - High compression ratio reveals underlying structure
- **Entropy = Fraud Detection** - Anomalies in entropy signal violations
- **Bits/sec = Sovereignty** - When internal processing exceeds external input
- **Receipts = Trust** - Immutable audit trail for every operation

## v3.0 Multi-Tier Autonomy

**The Paradigm Inversion:** Mars is hard because Earth can't make decisions for you fast enough.

Light-speed delay FORCES computational sovereignty:
- **LEO (0s delay):** Real-time Earth control possible
- **Mars (3-22 min delay):** Colony must decide autonomously between communication windows
- **Deep Space (4.3 years):** Complete independence required

### v3.0 Scenarios

| Scenario | Purpose | Key Validation |
|----------|---------|----------------|
| NETWORK | 1M colonist scale | Entropy stability ≥95%, partition recovery <48h |
| ADVERSARIAL | DoD hostile audit | Attack detection ≥99%, Byzantine consensus |

### Augmentation Factors

| Type | Factor | Effective Crew |
|------|--------|----------------|
| Human only | 1.0x | crew |
| AI-assisted | 5.0x | crew × 5 |
| Neuralink | 20.0x | crew × 20 |

**Key insight:** 4 crew + AI (4×5=20) equals 20 crew human-only for sovereignty calculations.

## ProofChain Series

SpaceProof is the flagship engine. Future repositories:
- **SpendProof** - Government spending (uses detect + ledger)
- **ClaimProof** - Healthcare fraud (uses detect + anchor)
- **VoteProof** - Election integrity (uses anchor + ledger)
- **OriginProof** - Supply chain (uses compress + anchor)
- **GreenProof** - Climate accountability (uses detect + witness)

## License

MIT

---

*Space-grade proof. No receipt, not real. Ship at T+24h or kill.*
