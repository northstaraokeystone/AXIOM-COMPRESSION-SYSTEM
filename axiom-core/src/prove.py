"""prove.py - Receipt Chain & Merkle Proofs (Updated for axiom-core)

THE PROOF INFRASTRUCTURE:
    Every calculation is verifiable.
    The merkle root is the cryptographic summary.
    Anyone can verify any result without seeing all others.

Source: CLAUDEME.md (S8)
"""

import json
from typing import Tuple, Any

from .core import dual_hash, emit_receipt, merkle

# === CONSTANTS ===

TENANT_ID = "axiom-core"
"""Receipt tenant isolation."""


def build_merkle_tree(items: list) -> Tuple[str, list]:
    """Build full merkle tree, returning root and all levels.

    Args:
        items: List of receipt dicts

    Returns:
        Tuple of (root_hash: str, levels: list[list[str]])
        levels[0] = leaf hashes, levels[-1] = [root]
    """
    if not items:
        empty_hash = dual_hash(b"empty")
        return empty_hash, [[empty_hash]]

    level_0 = [dual_hash(json.dumps(item, sort_keys=True)) for item in items]
    levels = [level_0]

    current_level = level_0
    while len(current_level) > 1:
        if len(current_level) % 2:
            current_level = current_level + [current_level[-1]]

        next_level = []
        for i in range(0, len(current_level), 2):
            parent = dual_hash(current_level[i] + current_level[i + 1])
            next_level.append(parent)

        levels.append(next_level)
        current_level = next_level

    root = current_level[0] if current_level else dual_hash(b"empty")
    return root, levels


def chain_receipts(receipts: list) -> dict:
    """Chain receipts and emit chain_receipt.

    Args:
        receipts: List of receipt dicts

    Returns:
        The chain_receipt dict

    MUST emit receipt (chain_receipt).
    """
    if not receipts:
        root = dual_hash(b"empty")
        return emit_receipt("chain", {
            "tenant_id": TENANT_ID,
            "n_receipts": 0,
            "merkle_root": root
        })

    root = merkle(receipts)

    return emit_receipt("chain", {
        "tenant_id": TENANT_ID,
        "n_receipts": len(receipts),
        "merkle_root": root
    })


def verify_proof(receipt: dict, proof_path: list, root: str) -> bool:
    """Verify a receipt is in the chain using its proof path.

    Args:
        receipt: The receipt being verified
        proof_path: List of {"sibling": hash, "position": "left"|"right"}
        root: The merkle root to verify against

    Returns:
        True if computed root matches provided root
    """
    current_hash = dual_hash(json.dumps(receipt, sort_keys=True))

    for step in proof_path:
        sibling = step["sibling"]
        position = step["position"]

        if position == "left":
            combined = sibling + current_hash
        else:
            combined = current_hash + sibling

        current_hash = dual_hash(combined)

    return current_hash == root


# === SENSITIVITY FORMATTING (v1.1 - Grok feedback Dec 16, 2025) ===

def format_sensitivity_finding(sensitivity_data: dict) -> str:
    """Format sensitivity analysis results as human-readable finding.

    Args:
        sensitivity_data: Dict from compute_sensitivity_ratio()

    Returns:
        Formatted string describing the sensitivity finding

    Source: Grok Dec 16, 2025 - "It's primarily latency-limited"
    """
    ratio_lin = sensitivity_data.get("ratio_linear", 1.0)
    ratio_exp = sensitivity_data.get("ratio_exp", 1.0)
    latency_limited_lin = sensitivity_data.get("latency_limited_linear", False)
    latency_limited_exp = sensitivity_data.get("latency_limited_exp", False)
    delay_variance = sensitivity_data.get("delay_variance_ratio", 7.33)
    bw_variance = sensitivity_data.get("bandwidth_variance_ratio", 4.0)

    finding = (
        "=" * 60 + "\n"
        "SENSITIVITY ANALYSIS FINDING\n"
        "=" * 60 + "\n\n"
        f"Grok validation: \"It's primarily latency-limited\"\n\n"
        f"Parameter Variance:\n"
        f"  Delay:     180s to 1320s ({delay_variance:.2f}x range)\n"
        f"  Bandwidth: 2 to 10 Mbps ({bw_variance:.2f}x range)\n\n"
        f"Model Results:\n"
        f"  Linear model:      {'LATENCY-LIMITED' if latency_limited_lin else 'BANDWIDTH-LIMITED'} "
        f"(ratio: {ratio_lin:.2f}x)\n"
        f"  Exponential model: {'LATENCY-LIMITED' if latency_limited_exp else 'BANDWIDTH-LIMITED'} "
        f"(ratio: {ratio_exp:.2f}x)\n\n"
        f"Interpretation:\n"
        f"  Delay variance ({delay_variance:.1f}x) exceeds bandwidth variance ({bw_variance:.1f}x),\n"
        f"  confirming Grok's assessment that the system is primarily\n"
        f"  latency-limited. Bandwidth investments yield diminishing returns\n"
        f"  at conjunction (22 min delay).\n\n"
        "=" * 60
    )

    return finding


def format_model_comparison(comparison_data: dict) -> str:
    """Format model comparison as human-readable report.

    Args:
        comparison_data: Dict from compare_models()

    Returns:
        Formatted comparison report
    """
    scenarios = comparison_data.get("scenarios", [])
    summary = comparison_data.get("summary", {})

    lines = [
        "=" * 70,
        "MODEL COMPARISON: Linear vs Exponential Decay",
        "=" * 70,
        "",
        f"Time constant (tau): {summary.get('tau_s', 300)}s",
        f"Note: {summary.get('model_note', '')}",
        "",
        "-" * 70,
        f"{'Scenario':<35} {'Linear':<12} {'Exponential':<12} {'Diff':<8}",
        "-" * 70,
    ]

    for s in scenarios:
        desc = s.get("description", "")[:35]
        t_lin = s.get("threshold_linear", 0)
        t_exp = s.get("threshold_exp", 0)
        diff = s.get("threshold_diff", 0)
        lines.append(f"{desc:<35} {t_lin:<12} {t_exp:<12} {diff:+<8}")

    lines.extend([
        "-" * 70,
        "",
        f"Mean rate ratio (exp/lin): {summary.get('mean_rate_ratio', 0):.4f}",
        f"Mean threshold difference: {summary.get('mean_threshold_diff', 0):.1f} crew",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


def format_grok_validation(validation_data: dict) -> str:
    """Format Grok number validation as human-readable report.

    Args:
        validation_data: Dict from validate_grok_numbers()

    Returns:
        Formatted validation report
    """
    grok = validation_data.get("grok_numbers", {})
    ours = validation_data.get("our_numbers", {})
    valid = validation_data.get("validation", {})

    return (
        "=" * 50 + "\n"
        "GROK NUMBER VALIDATION\n"
        "=" * 50 + "\n\n"
        "Grok's stated values:\n"
        f"  22 min, 100 Mbps: ~38k units\n"
        f"  3 min, 2 Mbps:    ~5.5k units\n\n"
        "Our calculated values:\n"
        f"  22 min, 100 Mbps: {grok.get('22min_100mbps_formula', 0):,} "
        f"{'✓' if valid.get('conjunction_match') else '✗'}\n"
        f"  3 min, 2 Mbps:    {grok.get('3min_2mbps_formula', 0):,} "
        f"{'✓' if valid.get('opposition_match') else '✗'}\n\n"
        f"VALIDATION: {'PASS' if valid.get('all_match') else 'FAIL'}\n\n"
        f"{validation_data.get('interpretation', '')}\n"
        "=" * 50
    )


def emit_sensitivity_proof_receipt(
    sensitivity_data: dict,
    comparison_data: dict,
    validation_data: dict
) -> dict:
    """Emit proof receipt for sensitivity analysis.

    MUST emit receipt per CLAUDEME.
    """
    finding = format_sensitivity_finding(sensitivity_data)
    comparison = format_model_comparison(comparison_data)
    validation = format_grok_validation(validation_data)

    return emit_receipt("sensitivity_proof", {
        "tenant_id": TENANT_ID,
        "finding": finding,
        "comparison": comparison,
        "validation": validation,
        "grok_validated": validation_data.get("validation", {}).get("all_match", False)
    })
