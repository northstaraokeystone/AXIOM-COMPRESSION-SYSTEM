# AXIOM-0 v2: THE WITNESS PROTOCOL

> "Laws exist in data like proofs exist in hash space. We don't compute them—we witness them."

## CHANGELOG (v2.0)

### Breaking Changes
- **TORCH REMOVED**: Entire system now numpy-only
- **KAN Rewritten**: Uses np.linalg.lstsq instead of gradient descent
- **File Renames**: galaxy.py → cosmos.py, residuals.py → topology.py

### New Features
- **6 Mandatory Scenarios**: BASELINE, STRESS, DISCOVERY, TOPOLOGY, REPRODUCIBILITY, GODEL
- **Receipt Chain**: Merkle proofs for all witness events
- **Topology Analysis**: H1/H2 persistence for DM-as-geometry hypothesis
- **Publication Formatter**: Academic-ready output with chi-squared statistics

### Architecture Changes
- Compression via linear least squares (not gradient descent)
- All modules emit CLAUDEME-compliant receipts
- Simulation-first validation (no feature ships without all scenarios passing)

---

## Overview

AXIOM-0 v2 tests whether Kolmogorov-Arnold Networks can discover physics laws through data compression. The core thesis: **compression = discovery**.

When a KAN achieves 90%+ compression on galaxy rotation curves, the spline coefficients ARE the equation. The MDL score IS the evidence.

---

## Quick Start
```bash
# Install dependencies (numpy only!)
pip install numpy ripser

# Run smoke test
python -c "from src.sim import run_simulation, SimConfig; s=run_simulation(SimConfig(n_cycles=10, n_galaxies_per_regime=2)); print(f'Passed: {s.passed}')"

# Run all 6 scenarios (THE SHIP GATE)
python -c "from src.sim import run_all_scenarios; r=run_all_scenarios(); print(f'ALL PASSED: {r[\"passed\"]}')"
```

---

## Architecture
```
cosmos.py          witness.py         topology.py        prove.py
(4 regimes) ──────► (KAN + MDL) ──────► (H1/H2) ──────► (merkle chain)
     │                   │                  │                │
     └───────────────────┴──────────────────┴────────────────┘
                                   │
                               sim.py
                        (6 mandatory scenarios)
```

---

## Modules

### core.py
Foundation functions per CLAUDEME standard:
- `dual_hash()`: SHA256:BLAKE3 format
- `emit_receipt()`: Creates timestamped, hashed receipts
- `merkle()`: Computes merkle root
- `StopRule`: Exception for constraint violations

### witness.py
Kolmogorov-Arnold Network implementation (NUMPY ONLY):
- `KAN`: Network class with B-spline edges
- `train()`: Solves coefficients via np.linalg.lstsq
- `spline_to_law()`: Extracts human-readable equation
- `classify_spline()`: sqrt|linear|inverse|log|power|complex

### cosmos.py
Synthetic galaxy generators:
- `newton_curve()`: V = sqrt(GM/r)
- `mond_curve()`: Deep MOND regime
- `nfw_curve()`: NFW dark matter halo
- `pbh_fog_curve()`: Novel PBH distribution
- `batch_generate()`: N galaxies per regime

### topology.py
Persistent homology analysis:
- `compute_persistence()`: H0/H1/H2 diagrams via ripser
- `wasserstein_distance()`: L1 distance between diagrams
- `classify_topology()`: newtonian|mond|dm_halo|novel
- `analyze_galaxy()`: Full pipeline with receipt

### prove.py
Receipt chain and verification:
- `chain_receipts()`: Merkle root of all witnesses
- `prove_witness()`: Generate proof path for single witness
- `verify_proof()`: Verify witness against root
- `format_for_publication()`: Academic-ready text

### sim.py
Monte Carlo validation harness:
- `run_simulation()`: Execute full simulation
- `run_scenario()`: Run named scenario with preset config
- `run_all_scenarios()`: THE SHIP GATE

---

## The 6 Mandatory Scenarios

No AXIOM feature ships without ALL scenarios passing.

| Scenario | Purpose | Pass Criteria |
|----------|---------|---------------|
| BASELINE | Standard operation | 1000 cycles, zero violations |
| STRESS | High noise tolerance | Newton >=80%, no NaN |
| DISCOVERY | Novel physics detection | PBH > NFW compression |
| TOPOLOGY | Geometry correlation | H1 matches regime |
| REPRODUCIBILITY | Determinism | Cross-seed variance <5% |
| GODEL | Edge case handling | Graceful degradation |

---

## Dependencies
```
numpy>=1.21.0
ripser>=0.6.0
```

**NO TORCH. NO TENSORFLOW. NO GPU.**

---

## Receipts

Every action produces a CLAUDEME-compliant receipt:

| Receipt | Module | Frequency |
|---------|--------|-----------|
| witness_receipt | witness.py | Per galaxy |
| cosmos_receipt | cosmos.py | Per batch |
| topology_receipt | topology.py | Per galaxy |
| chain_receipt | prove.py | Per simulation |
| sim_cycle_receipt | sim.py | Per cycle |
| violation_receipt | sim.py | Per violation |
| discovery_receipt | sim.py | On novel finding |

---

## The Core Thesis

**Old paradigm:** Propose model → Fit parameters → Claim discovery

**AXIOM paradigm:** Witness data → Compress maximally → Compression IS the law

If KAN compresses PBH fog better than NFW on identical mass profiles, we've witnessed new physics. The merkle root is the timestamp. The receipts are the evidence.

---

## License

MIT License

---

## Citation

If AXIOM helps your research:
```bibtex
@software{axiom2024,
  title={AXIOM: Law Discovery Through Compression},
  author={northstaraokeystone},
  year={2024},
  url={https://github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM}
}
```
