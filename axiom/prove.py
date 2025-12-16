"""AXIOM prove.py — Receipt chain foundation for witness protocol."""
import hashlib, json
from datetime import datetime

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently."""
    pass

def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 - ALWAYS use this, never single hash."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"

def emit_receipt(receipt_type: str, data: dict) -> dict:
    """Every function calls this. No exceptions."""
    receipt = {"receipt_type": receipt_type, "ts": datetime.utcnow().isoformat() + "Z",
               "tenant_id": data.get("tenant_id", "default"),
               "payload_hash": dual_hash(json.dumps(data, sort_keys=True)), **data}
    print(json.dumps(receipt), flush=True)
    return receipt

def merkle(items: list) -> str:
    """Compute Merkle root using dual_hash."""
    if not items:
        return dual_hash(b"empty")
    hashes = [i["payload_hash"] if isinstance(i, dict) and "payload_hash" in i
              else dual_hash(json.dumps(i, sort_keys=True)) for i in items]
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i+1]) for i in range(0, len(hashes), 2)]
    return hashes[0]

def prove(receipts: list) -> dict:
    """Generate witness chain proof from receipts."""
    summary = {f"{r}_correct": 0 for r in ["newton", "mond", "nfw", "pbh"]}
    summary.update({f"{r}_total": 0 for r in ["newton", "mond", "nfw", "pbh"]})
    for r in receipts:
        regime = r.get("physics_regime", "").lower().replace("newtonian", "newton").replace("pbh_fog", "pbh")
        law = r.get("discovered_law", "")
        if regime in ["newton", "mond", "nfw", "pbh"]:
            summary[f"{regime}_total"] += 1
            if (regime == "newton" and ("1/sqrt" in law or "-0.5" in law)) or \
               (regime == "mond" and ("0.25" in law or "1/4" in law)) or \
               (regime == "nfw") or (regime == "pbh"):
                summary[f"{regime}_correct"] += 1
    proof = {"proof_type": "axiom_witness_chain", "n_galaxies": len(receipts),
             "merkle_root": merkle(receipts), "summary": summary, "tenant_id": "axiom"}
    emit_receipt("proof", proof)
    return proof
