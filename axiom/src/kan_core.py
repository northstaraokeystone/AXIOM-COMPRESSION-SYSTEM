"""Kolmogorov-Arnold Network Core - Network-as-Equation Engine.

The network structure IS the physical law:
- Edges converge to y=x -> Newtonian gravity
- Edges converge to y=x+sqrt(x) -> MOND invented
- Hidden nodes required -> Dark matter topology discovered
"""

import hashlib
import json
from datetime import datetime
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# === CONSTANTS ===
DEFAULT_TOPOLOGY = [1, 5, 1]
SPLINE_DEGREE = 3
SPLINE_KNOTS = 10
PARAM_BUDGET = 10000
MDL_ALPHA = 1.0
MDL_BETA = 0.1
COMPLEXITY_THRESHOLD = 50
GRAD_CLIP_NORM = 1.0

# === RECEIPT SCHEMAS ===
RECEIPT_SCHEMAS = {
    "training": {
        "receipt_type": "training",
        "ts": "ISO8601",
        "tenant_id": "str",
        "galaxy_id": "str",
        "epoch": "int",
        "loss": "float",
        "mse": "float",
        "complexity": "int",
        "grad_norm": "float",
        "payload_hash": "str",
    },
    "interpretation": {
        "receipt_type": "interpretation",
        "ts": "ISO8601",
        "tenant_id": "str",
        "galaxy_id": "str",
        "edge_classifications": "dict[str, str]",
        "equations": "dict[str, str]",
        "dark_matter_topology": "bool",
        "payload_hash": "str",
    },
    "discovery": {
        "receipt_type": "discovery",
        "ts": "ISO8601",
        "tenant_id": "str",
        "law_type": "str",
        "formula": "str",
        "galaxies_explained": "list[str]",
        "description_length_bits": "int",
        "payload_hash": "str",
    },
}


class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently."""
    pass


def dual_hash(data: Union[bytes, str]) -> str:
    """SHA256:BLAKE3 - ALWAYS use this, never single hash."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Every function calls this. No exceptions."""
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": data.get("tenant_id", "axiom"),
        "payload_hash": dual_hash(json.dumps(data, sort_keys=True)),
        **data,
    }
    print(json.dumps(receipt), flush=True)
    return receipt


def _bspline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Compute B-spline basis functions of given degree."""
    n_basis = len(knots) - degree - 1
    basis = torch.zeros(x.shape[0], n_basis, device=x.device, dtype=x.dtype)
    for i in range(n_basis):
        basis[:, i] = _bspline_basis_single(x, knots, i, degree)
    return basis


def _bspline_basis_single(x: torch.Tensor, knots: torch.Tensor, i: int, k: int) -> torch.Tensor:
    """Recursive B-spline basis function computation."""
    if k == 0:
        return ((x >= knots[i]) & (x < knots[i + 1])).float()
    left_num = x - knots[i]
    left_den = knots[i + k] - knots[i]
    left = torch.where(left_den > 1e-10, left_num / left_den, torch.zeros_like(x))
    right_num = knots[i + k + 1] - x
    right_den = knots[i + k + 1] - knots[i + 1]
    right = torch.where(right_den > 1e-10, right_num / right_den, torch.zeros_like(x))
    return left * _bspline_basis_single(x, knots, i, k - 1) + right * _bspline_basis_single(x, knots, i + 1, k - 1)


class SplineEdge(nn.Module):
    """B-spline edge that transforms input via learned spline."""

    def __init__(self, n_knots: int = SPLINE_KNOTS, degree: int = SPLINE_DEGREE):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        n_coeffs = n_knots - degree - 1
        self.coeffs = nn.Parameter(torch.zeros(n_coeffs))
        knots = torch.linspace(0, 1, n_knots)
        self.register_buffer("knots", knots)
        nn.init.xavier_uniform_(self.coeffs.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x_norm = torch.clamp(x_norm, 0, 0.999)
        basis = _bspline_basis(x_norm.flatten(), self.knots, self.degree)
        out = (basis @ self.coeffs).view(x.shape)
        return out


class KANLayer(nn.Module):
    """KAN layer with spline edges between input and output nodes."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edges = nn.ModuleList([
            nn.ModuleList([SplineEdge() for _ in range(in_features)])
            for _ in range(out_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] if x.dim() > 1 else 1
        x = x.view(batch_size, -1)
        out = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        for j in range(self.out_features):
            for i in range(self.in_features):
                out[:, j] += self.edges[j][i](x[:, i])
        return out


class KANNetwork(nn.Module):
    """Kolmogorov-Arnold Network with B-spline edges."""

    def __init__(self, topology: list):
        super().__init__()
        self.topology = topology
        self.layers = nn.ModuleList([
            KANLayer(topology[i], topology[i + 1])
            for i in range(len(topology) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def spline_edge(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Apply B-spline transformation with given coefficients."""
    n_coeffs = coeffs.shape[0]
    n_knots = n_coeffs + SPLINE_DEGREE + 1
    knots = torch.linspace(0, 1, n_knots, device=x.device, dtype=x.dtype)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x_norm = torch.clamp(x_norm, 0, 0.999)
    basis = _bspline_basis(x_norm.flatten(), knots, SPLINE_DEGREE)
    return (basis @ coeffs).view(x.shape)


def kan_init(topology: list = None, n_params: int = PARAM_BUDGET) -> nn.Module:
    """Initialize KAN network with parameter budget check."""
    if topology is None:
        topology = DEFAULT_TOPOLOGY
    network = KANNetwork(topology)
    param_count = sum(p.numel() for p in network.parameters())
    if param_count > n_params:
        raise StopRule(f"Parameter budget exceeded: {param_count} > {n_params}")
    coeff_data = torch.cat([p.flatten() for p in network.parameters()])
    emit_receipt("init", {
        "tenant_id": "axiom",
        "topology": topology,
        "param_count": param_count,
        "coefficient_hash": dual_hash(coeff_data.detach().numpy().tobytes()),
    })
    return network


def forward_compress(network: nn.Module, baryonic_field: torch.Tensor) -> torch.Tensor:
    """Forward pass: mass -> velocity via learned equation."""
    return network(baryonic_field)


def complexity(network: nn.Module) -> int:
    """Compute Kolmogorov complexity proxy in estimated bits."""
    total_abs = 0.0
    nnz_edges = 0
    overhead_per_edge = 8
    for param in network.parameters():
        total_abs += param.abs().sum().item()
        nnz_edges += (param.abs() > 0.001).sum().item()
    bits = int(total_abs + nnz_edges * overhead_per_edge)
    return bits


def mdl_loss(pred: torch.Tensor, obs: torch.Tensor, network: nn.Module) -> torch.Tensor:
    """MDL loss: alpha * MSE + beta * complexity."""
    mse = torch.mean((pred - obs) ** 2)
    comp = complexity(network)
    return MDL_ALPHA * mse + MDL_BETA * torch.tensor(comp, dtype=mse.dtype, device=mse.device)


def train_step(network: nn.Module, batch: tuple, optimizer: Optimizer) -> tuple:
    """Single training step with receipt emission."""
    baryonic_field, velocity_obs = batch
    optimizer.zero_grad()
    pred = forward_compress(network, baryonic_field)
    loss = mdl_loss(pred, velocity_obs, network)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    mse_val = torch.mean((pred - velocity_obs) ** 2).item()
    comp_val = complexity(network)
    emit_receipt("training", {
        "tenant_id": "axiom",
        "galaxy_id": "batch",
        "epoch": 0,
        "loss": loss.item(),
        "mse": mse_val,
        "complexity": comp_val,
        "grad_norm": float(grad_norm),
    })
    return loss.item(), float(grad_norm)


def _classify_spline(coeffs: torch.Tensor, x_test: torch.Tensor) -> tuple:
    """Classify spline as Newtonian, MOND-like, or Complex."""
    y = spline_edge(x_test, coeffs)
    y_linear = x_test
    y_mond = x_test + torch.sqrt(torch.abs(x_test) + 1e-8)
    linear_err = torch.mean((y - y_linear) ** 2).item()
    mond_err = torch.mean((y - y_mond) ** 2).item()
    if linear_err < 0.01:
        return "Newtonian", "y = x"
    elif mond_err < 0.05:
        return "MOND-like", "y = x + sqrt(x)"
    else:
        return "Complex", "Dark matter topology required"


def extract_equation(trained_network: nn.Module) -> dict:
    """Extract equation classifications from converged network."""
    result = {}
    x_test = torch.linspace(0.1, 1.0, 100)
    edge_id = 0
    for layer_idx, layer in enumerate(trained_network.layers):
        for j, out_edges in enumerate(layer.edges):
            for i, edge in enumerate(out_edges):
                classification, formula = _classify_spline(edge.coeffs, x_test)
                result[f"L{layer_idx}_E{edge_id}"] = (classification, formula)
                edge_id += 1
    classifications = {k: v[0] for k, v in result.items()}
    equations = {k: v[1] for k, v in result.items()}
    dark_matter = any(c == "Complex" for c in classifications.values())
    emit_receipt("interpretation", {
        "tenant_id": "axiom",
        "galaxy_id": "trained",
        "edge_classifications": classifications,
        "equations": equations,
        "dark_matter_topology": dark_matter,
    })
    return result


def detect_dark_matter(network: nn.Module, threshold: int = COMPLEXITY_THRESHOLD) -> bool:
    """Detect if dark matter topology is required."""
    comp = complexity(network)
    return comp > threshold


def persistence_match(obs_diagram: np.ndarray, pred_diagram: np.ndarray) -> float:
    """Compute Wasserstein distance between persistence diagrams."""
    if obs_diagram.size == 0 or pred_diagram.size == 0:
        return 0.0
    try:
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        cost = cdist(obs_diagram, pred_diagram, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(np.sqrt(np.sum(cost[row_ind, col_ind] ** 2)))
    except ImportError:
        diff = np.mean((obs_diagram.flatten() - pred_diagram.flatten()[:obs_diagram.size]) ** 2)
        return float(np.sqrt(diff))


def checkpoint_save(
    network: nn.Module,
    path: str,
    training_receipts: list,
    interpretation_receipt: dict = None,
    discovery_receipt: dict = None,
) -> str:
    """Save checkpoint with receipts for audit trail."""
    from .kan_core import dual_hash, emit_receipt
    checkpoint = {
        "state_dict": network.state_dict(),
        "training_receipts": training_receipts,
        "interpretation_receipt": interpretation_receipt,
        "discovery_receipt": discovery_receipt,
    }
    torch.save(checkpoint, path)
    all_receipts = training_receipts + ([interpretation_receipt] if interpretation_receipt else [])
    all_receipts += [discovery_receipt] if discovery_receipt else []
    merkle_root = dual_hash(json.dumps([str(r) for r in all_receipts], sort_keys=True))
    emit_receipt("checkpoint", {
        "tenant_id": "axiom",
        "path": path,
        "merkle_root": merkle_root,
    })
    return path
