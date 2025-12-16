# AXIOM

Compression IS discovery.

## What It Does

| File | Lines | Purpose |
|------|-------|---------|
| `cosmos.py` | 71 | Generate synthetic galaxy rotation curves (4 physics regimes) |
| `src/kan_core.py` | 341 | KAN witness: learn rotation law via MDL minimization |
| `prove.py` | 59 | Receipt chain with dual-hash merkle proofs |
| `tweet.py` | 47 | AI-to-AI output formatting for Grok consumption |

Total: 518 lines. No dependencies beyond stdlib + torch.

## Quick Start

```bash
git clone https://github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM
cd AXIOM-COMPRESSION-SYSTEM/axiom
python -c "from cosmos import batch_generate; from src.kan_core import train_step; from prove import prove; from tweet import format_batch; import json; galaxies = batch_generate(n_per_regime=25); print('Generated', len(galaxies), 'galaxies'); print('Run KAN training on each galaxy, then prove() and format_batch()')"
```

Expected output: Galaxy data ready for KAN compression witness protocol.

## Architecture

```
cosmos.py           →  src/kan_core.py    →  prove.py        →  tweet.py
(generate curves)      (KAN + MDL)           (merkle chain)     (format)

Data flow:
1. cosmos: r,v arrays for 4 regimes (Newton/MOND/NFW/PBH)
2. kan_core: fit spline network, extract law via equation discovery
3. prove: aggregate receipts → merkle root
4. tweet: format for Grok analysis
```

## Output Format

Single galaxy receipt:
```
G:synth_pbh_0 PBH
Law:y = 0.87*x^0.42 + 1.2
C:94.3% MSE:0.012
✓ ⊢a3f5c2d1
```

Batch proof thread (tweet format):
```
N:100 Newton:23/25 MOND:24/25 NFW:24/25 PBH:25/25
Δ PBH-NFW:+4% ⊢a3f5c2d1

Δ=+4% PBH>NFW same synthetic data.
Confounders to reject H₀:PBH≠DM_proxy?
```

Grok prompt:
```
Δ=+4% PBH>NFW. Confounders to reject null?
```

## The Question

**Falsifiable hypothesis:**

If KAN compresses PBH fog rotation curves more efficiently than NFW dark matter halos on identical synthetic data, then either:

1. PBH fog has lower intrinsic entropy (simpler physics)
2. KAN architecture is biased toward cored profiles
3. Synthetic data generation favors PBH by construction

**What Grok response means:**

- Grok identifies confounders → test design flaw found
- Grok validates result → proceed to real galaxy data
- Grok requests more data → increase n_per_regime

The null hypothesis is: PBH compression rate = NFW compression rate.
Reject if Δ > 5% with p < 0.05 after confounder analysis.
