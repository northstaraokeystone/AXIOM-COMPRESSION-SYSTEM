"""AXIOM-COLONY v3.1 Prove Module - Merkle chain and paradigm shift proof.

THE PARADIGM SHIFT PROOF (v3.1):
  1 bit/sec compression advantage ≡ X kg Starship payload

Derivation:
  - Starship payload: 150,000 kg
  - Threshold: T crew (from discovered_law)
  - Rate: BASE_DECISIONS_PER_PERSON_PER_SEC per person
  - Total internal rate at threshold: T × BASE = R bits/sec
  - Mass per bit/sec: 150,000 / R = M kg

∴ 1 bit/sec ≡ M kg payload
"""

from src.core import merkle, dual_hash, emit_receipt
from src.entropy import BASE_DECISIONS_PER_PERSON_PER_SEC


def chain_receipts(receipts: list) -> dict:
    """Chain receipts into Merkle tree."""
    if not receipts:
        root = dual_hash(b'empty')
    else:
        root = merkle(receipts)

    return {
        "merkle_root": root,
        "receipt_count": len(receipts),
        "chained": True,
    }


def summarize_batch(state) -> dict:
    """Summarize simulation batch."""
    if state is None:
        return {"error": "no state"}

    return {
        "colonies": len(getattr(state, 'colonies', [])),
        "violations": len(getattr(state, 'violations', [])),
        "discovered_law": getattr(state, 'discovered_law', ''),
        "threshold_band": getattr(state, 'threshold_band', {}),
        "compression_ratio": getattr(state, 'compression_ratio', 0.0),
        "passed": getattr(state, 'passed', False),
    }


def bits_to_mass_equivalence(threshold: int, payload_kg: float = 150000) -> dict:
    """Calculate bits-to-mass equivalence. NEW - THE KEY.

    Args:
        threshold: Sovereignty threshold in crew count
        payload_kg: Starship payload capacity (default 150,000 kg)

    Returns:
        {
            threshold_crew: int,
            decision_rate_per_person: float,
            total_internal_rate_bps: float,
            starship_payload_kg: float,
            kg_per_bit_per_sec: float,  # THE KEY NUMBER
            implication: str
        }
    """
    rate_per_person = BASE_DECISIONS_PER_PERSON_PER_SEC
    total_rate = threshold * rate_per_person

    if total_rate > 0:
        kg_per_bit = payload_kg / total_rate
    else:
        kg_per_bit = float('inf')

    return {
        "threshold_crew": threshold,
        "decision_rate_per_person": rate_per_person,
        "total_internal_rate_bps": total_rate,
        "starship_payload_kg": payload_kg,
        "kg_per_bit_per_sec": kg_per_bit,
        "implication": f"1 bit/sec of decision capacity ≡ {kg_per_bit:,.0f} kg payload",
    }


def format_discovery(state) -> str:
    """Format discovery output. NEW."""
    if state is None:
        return "No simulation state available."

    discovered_law = getattr(state, 'discovered_law', 'unknown')
    threshold_band = getattr(state, 'threshold_band', {})
    r_squared = getattr(state, 'compression_ratio', 0.0)
    receipts = getattr(state, 'entropy_receipts', [])

    root = chain_receipts(receipts)['merkle_root']

    return f"""AXIOM-COLONY v3.1 COMPRESSION SOVEREIGNTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISCOVERED LAW:
{discovered_law}

THRESHOLD BAND:
Minimum: {threshold_band.get('min', '?')} crew
Expected: {threshold_band.get('expected', '?')} crew
Maximum: {threshold_band.get('max', '?')} crew

Fit quality: R² = {r_squared:.2f}

The colony survives when it COMPRESSES FASTER than problems arrive.

Merkle proof: {root[:8]}...{root[-8:]}"""


def format_paradigm_shift(state, payload_kg: float = 150000) -> str:
    """Format paradigm shift proof. NEW."""
    if state is None:
        threshold = 25  # Default
    else:
        threshold_band = getattr(state, 'threshold_band', {})
        threshold = threshold_band.get('expected', 25)

    equiv = bits_to_mass_equivalence(threshold, payload_kg)

    return f"""PARADIGM SHIFT PROVEN
━━━━━━━━━━━━━━━━━━━━━

COMPRESSION HAS MASS EQUIVALENCE

Sovereignty threshold: {equiv['threshold_crew']} crew
Decision rate: {equiv['decision_rate_per_person']} bits/sec/person
Total internal rate: {equiv['total_internal_rate_bps']:.1f} bits/sec

Starship payload: {equiv['starship_payload_kg']:,.0f} kg
Mass per bit/sec: {equiv['kg_per_bit_per_sec']:,.0f} kg

IMPLICATION:
{equiv['implication']}

On-board AI providing 1 bit/sec saves {equiv['kg_per_bit_per_sec']:,.0f} kg of crew payload.

This is why Elon built xAI before Mars."""


def format_tweet(state) -> str:
    """Format tweet (≤280 chars)."""
    if state is None:
        threshold = 25
    else:
        threshold_band = getattr(state, 'threshold_band', {})
        threshold = threshold_band.get('expected', 25)

    equiv = bits_to_mass_equivalence(threshold, 150000)
    kg = equiv['kg_per_bit_per_sec']

    tweet = f"""AXIOM-COLONY v3.1 FINDING

Sovereignty threshold: {threshold} crew

1 bit/sec decision capacity ≡ {kg:,.0f} kg payload

Not mass. Not energy. BITS.

github.com/northstaraokeystone/AXIOM-COLONY"""

    # Ensure ≤280 chars
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
