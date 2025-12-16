# KAN Core Specification

## Overview

Kolmogorov-Arnold Network engine where spline functions on edges encode physics equations. The network structure IS the physical law.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `baryonic_field` | `torch.Tensor` | 1D tensor of baryonic mass distribution at radius points |
| `topology` | `list[int]` | Network layer structure, default `[1, 5, 1]` |
| `n_params` | `int` | Maximum parameter budget, default `10000` |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `velocity_field` | `torch.Tensor` | Predicted velocity at each radius point |
| `edge_classifications` | `dict` | Edge ID to classification (Newtonian/MOND-like/Complex) |
| `equations` | `dict` | Edge ID to formula string |
| `dark_matter_detected` | `bool` | True if complexity exceeds threshold |

## Receipts

### training_receipt
Emitted by `train_step()` after each training iteration.
```json
{
  "receipt_type": "training",
  "ts": "ISO8601",
  "tenant_id": "str",
  "galaxy_id": "str",
  "epoch": "int",
  "loss": "float",
  "mse": "float",
  "complexity": "int",
  "grad_norm": "float",
  "payload_hash": "str"
}
```

### interpretation_receipt
Emitted by `extract_equation()` after analyzing converged network.
```json
{
  "receipt_type": "interpretation",
  "ts": "ISO8601",
  "tenant_id": "str",
  "galaxy_id": "str",
  "edge_classifications": "dict[str, str]",
  "equations": "dict[str, str]",
  "dark_matter_topology": "bool",
  "payload_hash": "str"
}
```

### discovery_receipt
Emitted when a new physical law is discovered.
```json
{
  "receipt_type": "discovery",
  "ts": "ISO8601",
  "tenant_id": "str",
  "law_type": "str",
  "formula": "str",
  "galaxies_explained": "list[str]",
  "description_length_bits": "int",
  "payload_hash": "str"
}
```

### checkpoint_receipt
Emitted by `checkpoint_save()` for audit trail.
```json
{
  "receipt_type": "checkpoint",
  "ts": "ISO8601",
  "tenant_id": "str",
  "path": "str",
  "merkle_root": "str",
  "payload_hash": "str"
}
```

### init_receipt
Emitted by `kan_init()` when network is created.
```json
{
  "receipt_type": "init",
  "ts": "ISO8601",
  "tenant_id": "str",
  "topology": "list[int]",
  "param_count": "int",
  "coefficient_hash": "str",
  "payload_hash": "str"
}
```

## SLOs

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| Training step latency | < 100ms | `time(train_step())` |
| Complexity calculation | < 10ms | `time(complexity())` |
| Parameter budget | < 10,000 | `sum(p.numel() for p in network.parameters())` |

## Stoprules

| Condition | Action | Receipt |
|-----------|--------|---------|
| `param_count > n_params` | Raise `StopRule` | anomaly_receipt |
| `grad_norm > 100` | Clip gradients | training_receipt with clipped norm |
| Network fails to converge | Continue training | training_receipt |

## Rollback

1. Load last checkpoint via `torch.load(path)`
2. Verify `state_dict` integrity via coefficient hash
3. Resume training from saved epoch
4. All training receipts are preserved in checkpoint

## Constants

| Name | Value | Purpose |
|------|-------|---------|
| `DEFAULT_TOPOLOGY` | `[1, 5, 1]` | 1 input -> 5 hidden -> 1 output |
| `SPLINE_DEGREE` | `3` | B-spline degree |
| `SPLINE_KNOTS` | `10` | Number of knots per spline |
| `PARAM_BUDGET` | `10000` | Max parameters allowed |
| `MDL_ALPHA` | `1.0` | Weight for MSE term |
| `MDL_BETA` | `0.1` | Weight for complexity penalty |
| `COMPLEXITY_THRESHOLD` | `50` | Bits threshold for dark matter detection |
| `GRAD_CLIP_NORM` | `1.0` | Gradient clipping max norm |

## Network Interpretation

When the network converges:
- **y = x**: Newtonian gravity - linear relationship between mass and velocity
- **y = x + sqrt(x)**: MOND-like - Modified Newtonian Dynamics pattern
- **Complex topology required**: Dark matter - residual structure that cannot be compressed

The network doesn't learn physics. It compresses physics. When you minimize description length, only laws remain.
