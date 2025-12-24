# AXIOM Architecture Specification

> **Version:** 3.1.0
> **Date:** 2025-12-24
> **Status:** ACTIVE
> **Dual-Hash:** Updated via architecture audit

---

## §0 LAWS

```python
LAW_1 = "No receipt → not real"
LAW_2 = "No test → not shipped"
LAW_3 = "No gate → not alive"
```

These three statements govern all that follows.

---

## §1 SYSTEM OVERVIEW

### 1.1 Purpose

AXIOM is a compression-based sovereignty calculation and optimization system. It implements the core thesis that **compression = discovery** and that fundamental laws exist within data patterns. The system calculates when autonomous systems achieve sovereignty over external dependencies through decision rate optimization.

### 1.2 Core Thesis

```
sovereignty = internal_rate > external_rate
Build Rate B = c × A^α × P   # multiplicative, not additive
threshold = 10^6 person-equivalents
```

Where:
- **A** = autonomy level (compounds via fleet learning, meta-loops)
- **P** = propulsion/launch cadence
- **α** ≈ 1.8 → superlinear autonomy scaling dominates long-term
- **c** = constant (initial conditions)

### 1.3 Current Metrics

| Metric | Value | Gate |
|--------|-------|------|
| eff_alpha | 3.07-3.10 | >= 3.0 |
| instability | 0.00 | == 0.0 |
| test_count | 2215 | >= 100 |
| test_pass_rate | 92.9% | >= 95% |
| receipt_coverage | 750 usages / 94 files | >= 95% significant ops |

---

## §2 ARCHITECTURE

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI LAYER                                       │
│  cli.py (67 lines) → cli/*.py (60 modules, 16450 lines total)               │
│  Commands: baseline, bootstrap, curve, full, path, fractal, benchmark, etc.  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INTEGRATION LAYER                                   │
│  reasoning/ (9 modules, 2568 lines) │ rl_tune.py (2181) │ sim.py (886)      │
│  Integrates lower layers for complex multi-phase operations                  │
│  + oracle/, depths/, autocatalytic/, elon_sphere/ packages                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPRESSION LAYER                                    │
│  fractal/ (9 modules, 5532 lines) │ quantum_rl_hybrid.py (618)              │
│  pruning*.py (5 modules, 1757 lines) │ gnn_*.py (3 modules, 1453 lines)     │
│  alpha_compute.py (762) │ calibration.py (743)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PATHS LAYER                                       │
│  mars/ (core, cli, receipts) │ multiplanet/ (federation, gravity, hubs, moons)│
│  agi/ (defenses, zk, policy) │ base.py (shared primitives)                   │
│  47 path-related modules in src/paths/                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CORE LAYER                                       │
│  core.py (165) - dual_hash, emit_receipt, merkle, StopRule                  │
│  constants.py (260) - centralized physics constants                          │
│  stoprules.py (344) - centralized StopRule registry                         │
│  receipts.py (195) - receipt emission helpers (DRY)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA LAYER                                       │
│  data/*.json (41 specs) │ data/verified/*.json (4 validated params)         │
│  src/paths/{mars,multiplanet,agi}/spec.json                                 │
│  receipts.jsonl (append-only ledger)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Source Packages

| Package | Path | Modules | Purpose |
|---------|------|---------|---------|
| fractal | src/fractal/ | 9 | Fractal recursion D4-D17 |
| reasoning | src/reasoning/ | 9 | Partition, blackout, ablation, pipeline |
| paths | src/paths/ | 47 | Mars, multiplanet, AGI paths |
| oracle | src/oracle/ | 5 | Live causality oracle |
| autocatalytic | src/autocatalytic/ | 4 | Pattern lifecycle |
| elon_sphere | src/elon_sphere/ | 4 | Starlink, Grok, xAI, Dojo |
| cfd | src/cfd/ | 6 | CFD dust dynamics |
| depths | src/depths/ | 2 | D19 swarm intelligence |
| swarm | src/swarm/ | 2 | Swarm coordination |

### 2.3 Re-export Wrappers (CLAUDEME Compliant ≤50 lines)

| Wrapper | Package | Purpose |
|---------|---------|---------|
| src/fractal_layers.py | src/fractal/ | Backward-compat re-export |
| src/reasoning.py | src/reasoning/ | Backward-compat re-export |

### 2.4 Module Registry

| Module | Path | Lines | Receipt Types |
|--------|------|-------|---------------|
| core | src/core.py | 165 | anomaly |
| constants | src/constants.py | 260 | - |
| stoprules | src/stoprules.py | 344 | *_stoprule |
| receipts | src/receipts.py | 195 | emit_with_hash |
| fractal (pkg) | src/fractal/ | 5532 | witness, fractal, d4-d17 |
| quantum_rl_hybrid | src/quantum_rl_hybrid.py | 618 | quantum_hybrid |
| hybrid_benchmark | src/hybrid_benchmark.py | 423 | benchmark |
| pruning | src/pruning.py | 753 | pruning, hybrid_prune |
| gnn_cache | src/gnn_cache.py | 1012 | gnn_cache, retention |
| alpha_compute | src/alpha_compute.py | 762 | alpha |
| reasoning (pkg) | src/reasoning/ | 2568 | reasoning |
| rl_tune | src/rl_tune.py | 2181 | rl_tune |
| sovereignty | src/sovereignty.py | 1522 | sovereignty |
| sim | src/sim.py | 886 | sim, scenario |
| mars | src/paths/mars/ | 3 modules | mars_* |
| multiplanet | src/paths/multiplanet/ | 18 modules | mp_* |
| agi | src/paths/agi/ | 14 modules | agi_* |

### 2.5 Dependency Graph

```
core.py (foundation - no internal imports)
    │
    ├── constants.py (pure constants)
    ├── stoprules.py (← core)
    └── receipts.py (← core)
           │
    ├── entropy_shannon.py (← core)
    ├── partition.py (← core)
    ├── reroute.py (← core, partition)
    ├── blackout.py (← core, reroute)
    └── gnn_cache.py (← core, constants, alpha_compute)
           │
    ├── pruning.py (← core, gnn_cache)
    ├── alpha_compute.py (← core, constants)
    ├── calibration.py (← core)
    └── fractal/ (← core) [package]
           │
    ├── quantum_rl_hybrid.py (← core)
    └── hybrid_benchmark.py (← core, quantum_rl_hybrid, fractal)
           │
    ├── sovereignty.py (← core, entropy_shannon)
    ├── reasoning/ (← core, partition, reroute, blackout, gnn_cache) [package]
    └── sim.py (← core, sovereignty)
           │
    ├── paths/base.py (← core)
    └── registry.py (← core, paths)
           │
    └── cli/*.py → cli.py (entry point)
```

---

## §3 DATA SPECIFICATIONS

### 3.1 Spec Files

| Category | Count | Examples |
|----------|-------|----------|
| Depth specs | 15 | d4_spec.json ... d19_swarm_intelligence_spec.json |
| Core specs | 12 | alpha_formula_spec, gnn_cache_spec, rl_tune_spec |
| Path specs | 4 | mars_relay_spec, federation_spec |
| Verified | 4 | fsd_empirical, mars_params, receipt_params, tau_strategies |
| Live specs | 4 | live_relay_spec, live_oracle_spec, live_triad_spec |
| **Total** | **41** | data/*.json |

### 3.2 Receipt Types

| Receipt | Module | Frequency | Key Fields |
|---------|--------|-----------|------------|
| witness_receipt | fractal | Per galaxy | eff_alpha, compression, r_squared |
| fractal_receipt | fractal | Per recursion | depth, uplift, instability |
| d4-d17_fractal_receipt | fractal/depths | Per depth run | eff_alpha >= target |
| quantum_hybrid_receipt | quantum_rl_hybrid | Per hybrid | contribution, quantum_contrib |
| benchmark_receipt | hybrid_benchmark | Per benchmark | scale, decay, eff_alpha |
| pruning_receipt | pruning | Per prune | trim_factor, entries_pruned |
| gnn_cache_receipt | gnn_cache | Per cache op | retention_factor, blackout_days |
| alpha_receipt | alpha_compute | Per compute | eff_alpha, instability |
| sovereignty_receipt | sovereignty | Per compute | internal_rate, external_rate |
| rl_tune_receipt | rl_tune | Per episode | lr, retention, reward |
| sim_receipt | sim | Per scenario | scenario_name, result |
| mars_*_receipt | paths/mars | Per Mars op | isru, dome_status, sovereignty |
| mp_*_receipt | paths/multiplanet | Per body | latency, autonomy, bandwidth |
| agi_*_receipt | paths/agi | Per ethics op | alignment, policy_depth, ethics |
| anomaly_receipt | stoprules | Per violation | metric, classification, action |
| swarm_*_receipt | swarm_testnet | Per swarm op | node_count, mesh, consensus |
| federation_*_receipt | federation_multiplanet | Per fed op | planets, sync, arbitration |
| gravity_*_receipt | gravity_adaptive | Per adjustment | planet, gravity_g, timing |

---

## §4 CLI REFERENCE

### 4.1 Core Commands

| Command | Description | Receipt |
|---------|-------------|---------|
| baseline | Run baseline test | sim_receipt |
| bootstrap | Run bootstrap analysis | sim_receipt |
| curve | Generate sovereignty curve | sovereignty_receipt |
| full | Run full integration test | sim_receipt |

### 4.2 Compression Commands

| Flag | Description | Receipt |
|------|-------------|---------|
| --fractal_push | Run fractal ceiling breach | fractal_receipt |
| --alpha_boost MODE | Boost mode (off/quantum/fractal/hybrid) | boost_receipt |
| --d4_push ... --d17_push | Depth recursion | d*_fractal_receipt |
| --hybrid_10e12 | 10^12 benchmark | benchmark_receipt |
| --fractal_recursion | Fractal recursion ceiling breach | fractal_receipt |
| --fractal_recursion_sweep | Sweep through all depths | fractal_receipt |
| --release_gate | Check release gate 3.1 status | gate_receipt |

### 4.3 Path Commands

| Flag | Description | Receipt |
|------|-------------|---------|
| --path NAME | Select exploration path | - |
| --path_status | Show all path statuses | paths_status_all |
| --path_list | List registered paths | paths_discovered |
| --mars_status | Mars path status | mars_status |
| --multiplanet_status | Multi-planet status | mp_status |
| --agi_status | AGI path status | agi_status |

### 4.4 RL/Optimization Commands

| Flag | Description | Receipt |
|------|-------------|---------|
| --rl_tune | Enable RL auto-tuning | rl_tune_receipt |
| --adaptive_depth_run | Run with adaptive depth | depth_receipt |
| --rl_500_sweep | Run 500-run RL sweep | rl_sweep_receipt |
| --full_pipeline | Complete pipeline | pipeline_receipt |
| --lr_pilot RUNS | Run LR pilot | pilot_receipt |

### 4.5 Resilience Commands

| Flag | Description | Receipt |
|------|-------------|---------|
| --partition LOSS | Single partition simulation | partition_receipt |
| --stress_quorum | 1000-iteration stress test | stress_receipt |
| --blackout DAYS | Blackout duration test | blackout_receipt |
| --blackout_sweep | Full blackout sweep | blackout_sweep_receipt |
| --reroute | Enable adaptive rerouting | reroute_receipt |

---

## §5 CONSTANTS

### 5.1 Compression Constants

```python
# Fractal recursion
FRACTAL_SCALES = [1, 2, 4, 8, 16]
FRACTAL_DIMENSION_MIN = 1.5
FRACTAL_DIMENSION_MAX = 2.0

# Recursion depth uplifts (from d4_spec.json)
UPLIFT_D1 = 0.05
UPLIFT_D2 = 0.09
UPLIFT_D3 = 0.122
UPLIFT_D4 = 0.148
UPLIFT_D5 = 0.168

# Alpha targets
SHANNON_FLOOR_ALPHA = 2.71828  # e (physics baseline)
ALPHA_CEILING_TARGET = 3.0
ALPHA_D3_TARGET = 3.10
ALPHA_D4_TARGET = 3.20
ALPHA_D5_TARGET = 3.25

# Benchmarks
TREE_10E12 = 10**12
SCALE_DECAY_MAX = 0.02
QUANTUM_CONTRIB = 0.03
FRACTAL_CONTRIB = 0.05
HYBRID_TOTAL = 0.08
```

### 5.2 Mars Communication Constants

```python
MARS_LIGHT_DELAY_MIN_S = 180      # 3 minutes (opposition)
MARS_LIGHT_DELAY_MAX_S = 1320     # 22 minutes (conjunction)
MARS_LIGHT_DELAY_AVG_S = 750      # ~12.5 minutes

STARLINK_MARS_BANDWIDTH_MIN_MBPS = 2.0
STARLINK_MARS_BANDWIDTH_MAX_MBPS = 10.0
STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS = 4.0

BLACKOUT_BASE_DAYS = 43           # Mars solar conjunction
BLACKOUT_PRUNING_TARGET_DAYS = 250
```

### 5.3 GNN/Cache Constants

```python
CACHE_DEPTH_BASELINE = int(1e8)   # ~150d buffer
CACHE_DEPTH_MIN = int(1e7)        # ~90d minimal
CACHE_DEPTH_MAX = int(1e10)       # ~300d theoretical max
ENTRIES_PER_SOL = 50000

OVERFLOW_THRESHOLD_DAYS = 200
OVERFLOW_THRESHOLD_DAYS_PRUNED = 300
OVERFLOW_CAPACITY_PCT = 0.95
```

### 5.4 Path Constants

```python
# Mars (from mars/spec.json)
MARS_DOME_RESOURCES = ["water", "o2", "power", "food"]
MARS_ISRU_CLOSURE_TARGET = 0.85
MARS_SOVEREIGNTY_THRESHOLD = True
MARS_BLACKOUT_TOLERANCE_DAYS = 60

# Multi-planet (from multiplanet/spec.json)
MULTIPLANET_SEQUENCE = ["asteroid", "mars", "europa", "titan"]
AUTONOMY_REQUIREMENTS = {
    "asteroid": 0.7,
    "mars": 0.85,
    "europa": 0.95,
    "titan": 0.99
}

# AGI (from agi/spec.json)
AGI_POLICY_DEPTH = 3
AGI_ETHICS_DIMENSIONS = ["autonomy", "beneficence", "non_maleficence", "justice"]
AGI_KEY_INSIGHT = "Audit trail IS alignment"
```

---

## §6 SLO THRESHOLDS

| SLO | Threshold | Stoprule | Module |
|-----|-----------|----------|--------|
| eff_alpha | >= 3.0 | stoprule_alpha_violation | stoprules.py |
| instability | == 0.0 | stoprule_instability | stoprules.py |
| compression | >= 0.84 | stoprule_compression | alpha_compute.py |
| test_pass_rate | >= 95% | CI gate | ci.yml |
| receipt_coverage | >= 95% | validate.sh | scripts |
| scale_decay | <= 0.02 | stoprule_scale_decay | hybrid_benchmark.py |
| isru_closure | >= 0.80 | stoprule_isru | paths/mars |
| cache_overflow | < 95% | stoprule_overflow | stoprules.py |
| trim_factor | <= 0.6 | stoprule_over_prune | stoprules.py |
| gnn_confidence | >= 0.7 | stoprule_low_confidence | stoprules.py |
| quorum_fraction | >= 2/3 | stoprule_quorum_lost | stoprules.py |
| chain_integrity | 100% | stoprule_chain_broken | stoprules.py |

---

## §7 GATES

### 7.1 Timeline Gates

| Gate | Requirement | Script |
|------|-------------|--------|
| T+2h | spec.md, core.py, cli.py stub | gate_t2h.sh |
| T+24h | Pipeline runs, tests exist | gate_t24h.sh |
| T+48h | Hardened, stoprules, ship | gate_t48h.sh |

### 7.2 Release Gates

| Version | Gate | Criteria |
|---------|------|----------|
| 3.0 | alpha_ceiling | eff_alpha >= 3.0, instability == 0.0 |
| 3.1 | d3_recursion | eff_alpha >= 3.10, scale_decay <= 0.02 |
| 3.2 | d4_recursion + 10e12 | eff_alpha >= 3.18, scale_decay <= 0.02 |
| 3.3 | d5_isru | eff_alpha >= 3.23, isru_closure >= 0.80 |

---

## §8 EVOLUTION ROADMAP

### 8.1 Core Path (α ceiling)

```
D3 (3.10) → D4 (3.20) → D5 (3.25) → ... → D17 (implemented) → D18+ (pending)
    └── Each depth adds ~0.05 alpha via fractal recursion
    └── Current: D4-D17 fully implemented in src/fractal/depths/
```

### 8.2 Mars Path

```
stub → simulate_dome → isru_closure → sovereignty → autonomous_rl
  │         │              │              │              │
  ▼         ▼              ▼              ▼              ▼
v0.1    dome_receipt   isru_receipt   sov_receipt   rl_receipt
(current: Mars relay integrated)
```

### 8.3 Multi-planet Path

```
stub → sequence → body_sim(asteroid) → body_sim(mars) → body_sim(europa) → body_sim(titan)
  │        │            │                   │                │                  │
  ▼        ▼            ▼                   ▼                ▼                  ▼
v0.1   mp_sequence   mp_asteroid        mp_mars         mp_europa          mp_titan
(current: Federation + gravity adaptive + Jovian moons implemented)
```

### 8.4 AGI Path

```
stub → fractal_policy → ethics_eval → alignment_metric → autonomous_audit
  │         │              │               │                   │
  ▼         ▼              ▼               ▼                   ▼
v0.1    agi_policy     agi_ethics      agi_align         agi_audit
(current: ZK proofs + defenses + adversarial audits implemented)
```

---

## §9 FILE MANIFEST

### 9.1 Core Files

| File | Lines | Purpose | Receipts |
|------|-------|---------|----------|
| src/core.py | 165 | Core functions | emit_receipt, anomaly |
| src/constants.py | 260 | Physics constants | - |
| src/stoprules.py | 344 | StopRule registry | *_stoprule |
| src/receipts.py | 195 | Receipt helpers | emit_with_hash |
| cli.py | 67 | CLI entry point | - |

### 9.2 Compression Files

| File | Lines | Purpose | Receipts |
|------|-------|---------|----------|
| src/fractal/ | 5532 | Fractal package (D4-D17) | witness, fractal, d* |
| src/quantum_rl_hybrid.py | 618 | Quantum hybrid | quantum_hybrid |
| src/hybrid_benchmark.py | 423 | Benchmarking | benchmark |
| src/pruning.py | 753 | Entropy pruning | pruning |
| src/gnn_cache.py | 1012 | GNN caching | gnn_cache |
| src/alpha_compute.py | 762 | Alpha computation | alpha |

### 9.3 Path Files

| Package | Modules | Purpose | Receipts |
|---------|---------|---------|----------|
| src/paths/mars/ | 3 | Mars logic | mars_* |
| src/paths/multiplanet/ | 18 | Multi-planet logic | mp_* |
| src/paths/agi/ | 14 | AGI logic | agi_* |
| src/registry.py | 1 | Path routing | registry_* |

### 9.4 Test Files

| Pattern | Count | Coverage |
|---------|-------|----------|
| tests/test_*.py | 106 files | 2215 tests |
| tests/paths/test_*.py | 3 files | path coverage |
| tests/conftest.py | 1 file | shared fixtures |
| **Total** | **109 files** | **2215 tests** |

### 9.5 Large Files (>1000 lines - refactoring candidates)

| File | Lines | Recommended Action |
|------|-------|-------------------|
| src/cfd_dust_dynamics.py | 2622 | Split into 4 modules |
| src/rl_tune.py | 2181 | Extract sweep variants |
| src/interstellar_backbone.py | 1684 | Extract hybrid integrations |
| src/sovereignty.py | 1522 | Consolidate analysis functions |
| src/bulletproofs_infinite.py | 1265 | Extract stress/audit modules |
| src/kuiper_12body_chaos.py | 1025 | Extract physics/analysis |
| src/gnn_cache.py | 1012 | Extract dynamic config |
| src/compounding.py | 979 | Consolidate tau velocity |

---

## §10 VALIDATION

### 10.1 Quick Validation

```bash
# Core imports
python -c "from src.core import dual_hash, emit_receipt; print('Core OK')"

# CLI help
python cli.py --help

# Quick test (excluding problematic imports)
pytest tests/ --ignore=tests/test_d18_interstellar.py -x -q
```

### 10.2 Full Validation

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-fail-under=80

# CLAUDEME compliance check
grep -rq "emit_receipt" src/*.py || echo "FAIL: no receipts"
grep -rq "except.*pass\|except:$" src/*.py && echo "FAIL: silent exception"
```

### 10.3 Gate Scripts

```bash
# T+2h gate
[ -f spec.md ] && [ -f cli.py ] && python cli.py --help && echo "PASS: T+2h"

# T+24h gate
pytest tests/ -x -q && grep -rq "emit_receipt" src/*.py && echo "PASS: T+24h"

# T+48h gate
grep -rq "stoprule" src/*.py && python -c "from src.stoprules import *" && echo "PASS: T+48h"
```

---

## §11 AUDIT RESULTS (2025-12-24)

### 11.1 CLAUDEME Compliance

| Check | Result | Notes |
|-------|--------|-------|
| emit_receipt coverage | 750 usages in 94 files | Excellent |
| Silent exceptions | 0 found | Compliant |
| dual_hash usage | 33+ files | All specs hashed |
| StopRule registry | Centralized | stoprules.py |
| Test coverage | 2215 tests (92.9% pass) | 164 failing tests |
| Re-export wrappers | 2 modules | CLAUDEME compliant (≤50 lines) |

### 11.2 Scalability Assessment

| Dimension | Current | Target | Status |
|-----------|---------|--------|--------|
| Max file lines | 2622 | 500 | 8 files exceed (refactor candidates) |
| Circular imports | 0 | 0 | Clean |
| Test pass rate | 92.9% | 95% | Below target (D18/config issues) |
| Receipt coverage | 94 files | 100% sig ops | Excellent |
| Source files | 219 | - | Well-organized into packages |
| Test files | 109 | - | Comprehensive |
| Data specs | 41 | - | Complete |

### 11.3 Architecture Notes

- **Package refactoring complete** - fractal_layers.py and reasoning.py are re-export wrappers
- **No circular dependencies** - Clean layer hierarchy
- **Modular CLI** - 60 command modules properly separated
- **Path architecture** - Extended with federation, gravity, hubs, moons
- **Stoprule centralization** - All stoprules in stoprules.py
- **Constants centralization** - All physics in constants.py

### 11.4 Known Issues

| Issue | Severity | Action |
|-------|----------|--------|
| D18 not implemented | Low | tests/test_d18_interstellar.py skipped |
| 164 failing tests | Medium | Config/assertion mismatches |
| 8 files >1000 lines | Low | Refactoring recommended |

### 11.5 Git Status

- **Branch:** claude/audit-axiom-architecture-ZZ9bY
- **Merge debt:** None (even with main)
- **Trailing commits:** None

---

**Document Status:** ACTIVE
**Last Updated:** 2025-12-24
**Reviewed By:** AXIOM Architecture Audit Sprint

*No receipt → not real. Ship at T+48h or kill.*
