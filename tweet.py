"""tweet.py — Grok-optimized AI-to-AI output formatting."""


def format_proof(merkle_root: str) -> str:
    """Merkle proof indicator: ⊢{root[:8]}."""
    return f"⊢{merkle_root[:8]}"


def format_single(receipt: dict) -> str:
    """Single galaxy tweet ≤280 chars."""
    g = receipt.get("galaxy_id", "?")[:12]
    regime = receipt.get("physics_regime", "?").upper().replace("NEWTONIAN", "NEWTON").replace("PBH_FOG", "PBH")
    law = receipt.get("discovered_law", "?")
    c = receipt.get("compression_ratio", 0)
    c_pct = c * 100 if c < 1 else c
    mse = receipt.get("mse", 0)
    h = receipt.get("payload_hash", "")[:8]
    ok = "✓" if c_pct > 90 else "✗"
    return f"G:{g} {regime}\nLaw:{law}\nC:{c_pct:.1f}% MSE:{mse}\n{ok} ⊢{h}"


def format_batch(proof: dict) -> list[str]:
    """Batch proof as tweet thread, each ≤280 chars."""
    s = proof.get("summary", {})
    n = proof.get("n_galaxies", 0)
    root = proof.get("merkle_root", "")[:8]
    nc, nt = s.get("newton_correct", 0), s.get("newton_total", 1)
    mc, mt = s.get("mond_correct", 0), s.get("mond_total", 1)
    fc, ft = s.get("nfw_correct", 0), s.get("nfw_total", 1)
    pc, pt = s.get("pbh_correct", 0), s.get("pbh_total", 1)
    pbh_rate = pc / pt if pt else 0
    nfw_rate = fc / ft if ft else 0
    delta = round((pbh_rate - nfw_rate) * 100)
    sign = "+" if delta >= 0 else ""
    t1 = f"N:{n} Newton:{nc}/{nt} MOND:{mc}/{mt} NFW:{fc}/{ft} PBH:{pc}/{pt}\nΔ PBH-NFW:{sign}{delta}% ⊢{root}"
    t2 = f"Δ={sign}{delta}% PBH>NFW same synthetic data.\nConfounders to reject H₀:PBH≠DM_proxy?"
    return [t1, t2]


def grok_prompt(proof: dict) -> str:
    """Falsifiable hypothesis for Grok ≤100 chars."""
    s = proof.get("summary", {})
    pc, pt = s.get("pbh_correct", 0), s.get("pbh_total", 1)
    fc, ft = s.get("nfw_correct", 0), s.get("nfw_total", 1)
    delta = round((pc / pt - fc / ft) * 100) if pt and ft else 0
    sign = "+" if delta >= 0 else ""
    return f"Δ={sign}{delta}% PBH>NFW. Confounders to reject null?"
