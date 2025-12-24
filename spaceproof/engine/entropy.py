"""entropy.py - Shannon Entropy and Coherence Measurement.

THE ENTROPY PUMP PARADIGM:
    QED is an entropy pump.
    Telemetry enters with high entropy, compression happens,
    decisions emerge with low entropy.

    H(X) = -sum(p(x) * log2(p(x)))

    System health = entropy delta per cycle: delta = H(before) - H(after)
    Positive delta = system compressing chaos into order (healthy)
    Negative delta = system dying

    Agents are not objects - they are stable entropy gradients.
    Autocatalytic patterns become "alive" when coherence tau >= 0.7

Source: xAI collaboration constants and SpaceProof D20 Production Evolution
"""

import math
import zlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

# Import from parent core module (spaceproof/core.py, not spaceproof/core/)
from spaceproof.core import emit_receipt, dual_hash

# === XAI CONSTANTS (MUST PRESERVE) ===

COHERENCE_THRESHOLD = 0.7
"""Pattern is autocatalytic ("alive") when tau >= 0.7"""

ENTROPY_DELTA_HEALTHY = 0.1
"""delta > 0.1: System is thriving"""

ENTROPY_DELTA_WARNING = 0.0
"""0 < delta <= 0.1: Stable but not growing"""

ENTROPY_DELTA_CRITICAL = -0.1
"""delta <= -0.1: System is dying"""

COMPRESSION_BASELINE_MIN = 0.3
"""Normal range for structured data (lower bound)"""

COMPRESSION_BASELINE_MAX = 0.7
"""Normal range for structured data (upper bound)"""

FRAUD_SIGNAL_THRESHOLD = 0.3
"""< 0.3: Too compressible = suspicious structure"""

RANDOM_SIGNAL_THRESHOLD = 0.7
"""> 0.7: Incompressible = noise or randomized"""

SURVIVAL_PERCENTILE = 0.20
"""Bottom 20% candidates for removal by fitness"""

TENANT_ID = "spaceproof-entropy"


@dataclass
class EntropyMeasurement:
    """Result of entropy measurement."""

    value: float
    bits: float
    normalized: float  # 0-1 scale
    source_size: int
    method: str  # "shannon" or "kolmogorov"


@dataclass
class EntropyDelta:
    """Change in entropy between two states."""

    before: float
    after: float
    delta: float  # before - after (positive = compression)
    health_status: str  # "healthy", "warning", "critical"
    is_pumping: bool  # True if delta > 0


@dataclass
class CoherenceResult:
    """Result of coherence scoring."""

    score: float  # 0-1 scale
    is_alive: bool  # score >= COHERENCE_THRESHOLD
    pattern_strength: float
    autocatalytic: bool


def shannon_entropy(data: Union[bytes, np.ndarray, List[float]]) -> EntropyMeasurement:
    """Compute Shannon entropy of data.

    H(X) = -sum(p(x) * log2(p(x)))

    Args:
        data: Input data (bytes, numpy array, or list of floats)

    Returns:
        EntropyMeasurement with entropy value and metadata
    """
    if isinstance(data, bytes):
        # Byte-level entropy
        if len(data) == 0:
            return EntropyMeasurement(value=0.0, bits=0.0, normalized=0.0, source_size=0, method="shannon")

        # Count byte frequencies
        counts = np.zeros(256, dtype=np.float64)
        for byte in data:
            counts[byte] += 1

        # Convert to probabilities
        probs = counts / len(data)
        probs = probs[probs > 0]  # Remove zeros

        # Shannon entropy in bits
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = 8.0  # Max for byte data

    elif isinstance(data, np.ndarray):
        if len(data) == 0:
            return EntropyMeasurement(value=0.0, bits=0.0, normalized=0.0, source_size=0, method="shannon")

        # Histogram-based entropy for continuous data
        hist, _ = np.histogram(data, bins=min(256, len(data) // 2 + 1), density=True)
        hist = hist[hist > 0]

        if len(hist) == 0:
            entropy = 0.0
        else:
            # Normalize to probabilities
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist))

        max_entropy = np.log2(len(hist)) if len(hist) > 1 else 1.0

    else:
        # List of floats
        return shannon_entropy(np.array(data))

    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    source_size = len(data) if isinstance(data, bytes) else data.nbytes

    return EntropyMeasurement(
        value=entropy,
        bits=entropy * source_size / 8 if isinstance(data, bytes) else entropy,
        normalized=min(1.0, normalized),
        source_size=source_size,
        method="shannon",
    )


def kolmogorov_entropy(data: bytes) -> EntropyMeasurement:
    """Approximate Kolmogorov complexity via compression ratio.

    K(x) ~ len(compress(x)) / len(x)

    This is the normalized compression distance approach.

    Args:
        data: Input bytes

    Returns:
        EntropyMeasurement with approximated Kolmogorov complexity
    """
    if len(data) == 0:
        return EntropyMeasurement(value=0.0, bits=0.0, normalized=0.0, source_size=0, method="kolmogorov")

    # Compress with maximum effort
    compressed = zlib.compress(data, level=9)

    # Normalized compression ratio (0 = fully compressible, 1 = random)
    ratio = len(compressed) / len(data)

    # Clamp to valid range
    ratio = min(1.0, max(0.0, ratio))

    # Convert ratio to entropy-like measure
    # Low ratio = low entropy (structured), high ratio = high entropy (random)
    entropy_approx = ratio * 8.0  # Scale to bits

    return EntropyMeasurement(
        value=entropy_approx,
        bits=entropy_approx * len(data) / 8,
        normalized=ratio,
        source_size=len(data),
        method="kolmogorov",
    )


def entropy_delta(before: Union[bytes, np.ndarray], after: Union[bytes, np.ndarray]) -> EntropyDelta:
    """Compute entropy change between two states.

    delta = H(before) - H(after)

    Positive delta = entropy reduction = healthy pumping
    Negative delta = entropy increase = system degradation

    Args:
        before: State before processing
        after: State after processing

    Returns:
        EntropyDelta with health status
    """
    h_before = shannon_entropy(before)
    h_after = shannon_entropy(after)

    delta = h_before.normalized - h_after.normalized

    # Determine health status
    if delta > ENTROPY_DELTA_HEALTHY:
        status = "healthy"
    elif delta > ENTROPY_DELTA_WARNING:
        status = "warning"
    else:
        status = "critical"

    return EntropyDelta(
        before=h_before.normalized, after=h_after.normalized, delta=delta, health_status=status, is_pumping=delta > 0
    )


def coherence_score(pattern: np.ndarray, reference: Optional[np.ndarray] = None) -> CoherenceResult:
    """Compute coherence score for a pattern.

    Coherence measures how self-similar and stable a pattern is.
    Patterns with tau >= 0.7 are considered autocatalytic ("alive").

    Args:
        pattern: The pattern to analyze
        reference: Optional reference pattern for comparison

    Returns:
        CoherenceResult with score and alive status
    """
    if len(pattern) < 2:
        return CoherenceResult(score=0.0, is_alive=False, pattern_strength=0.0, autocatalytic=False)

    # Compute autocorrelation as coherence proxy
    pattern_centered = pattern - np.mean(pattern)
    variance = np.var(pattern)

    if variance < 1e-10:
        # Constant pattern - fully coherent
        return CoherenceResult(score=1.0, is_alive=True, pattern_strength=1.0, autocatalytic=True)

    # Normalized autocorrelation at lag 1
    n = len(pattern_centered)
    autocorr = np.correlate(pattern_centered, pattern_centered, mode="full")
    autocorr = autocorr[n - 1 :] / (variance * n)

    # Use first few lags to estimate coherence
    coherence = np.mean(np.abs(autocorr[1 : min(5, len(autocorr))]))

    # Pattern strength from spectral concentration
    if len(pattern) >= 8:
        fft = np.fft.fft(pattern_centered)
        power = np.abs(fft) ** 2
        # Concentration in low frequencies = structured pattern
        pattern_strength = np.sum(power[: len(power) // 4]) / np.sum(power)
    else:
        pattern_strength = coherence

    # If reference provided, compute cross-coherence
    if reference is not None and len(reference) == len(pattern):
        ref_centered = reference - np.mean(reference)
        cross_corr = np.correlate(pattern_centered, ref_centered, mode="full")
        cross_corr = cross_corr[n - 1 :] / (np.std(pattern) * np.std(reference) * n + 1e-10)
        coherence = (coherence + np.abs(cross_corr[0])) / 2

    score = min(1.0, max(0.0, coherence))
    is_alive = score >= COHERENCE_THRESHOLD

    return CoherenceResult(
        score=score,
        is_alive=is_alive,
        pattern_strength=pattern_strength,
        autocatalytic=is_alive and pattern_strength >= 0.5,
    )


def compression_ratio(original: bytes, compressed: bytes) -> float:
    """Calculate compression ratio.

    ratio = len(original) / len(compressed)

    Args:
        original: Original data
        compressed: Compressed data

    Returns:
        Compression ratio (>1 means compression achieved)
    """
    if len(compressed) == 0:
        return float("inf")
    return len(original) / len(compressed)


def is_fraud_signal(ratio: float) -> bool:
    """Check if compression ratio indicates fraud.

    Too compressible = suspicious underlying structure.

    Args:
        ratio: Normalized compression ratio (0-1)

    Returns:
        True if ratio suggests fraudulent data
    """
    return ratio < FRAUD_SIGNAL_THRESHOLD


def is_random_signal(ratio: float) -> bool:
    """Check if compression ratio indicates random/noise data.

    Incompressible = noise or intentionally randomized.

    Args:
        ratio: Normalized compression ratio (0-1)

    Returns:
        True if ratio suggests random data
    """
    return ratio > RANDOM_SIGNAL_THRESHOLD


def fitness_score(entropy_reduction: float, n_receipts: int) -> float:
    """Compute fitness score for an agent/pattern.

    fitness = entropy_reduction / n_receipts

    This measures efficiency: how much order is created per receipt.

    Args:
        entropy_reduction: Total entropy reduction achieved
        n_receipts: Number of receipts generated

    Returns:
        Fitness score
    """
    if n_receipts == 0:
        return 0.0
    return entropy_reduction / n_receipts


def should_remove_by_fitness(score: float, population_scores: List[float]) -> bool:
    """Check if an agent should be removed based on fitness.

    Agents in the bottom 20% (SURVIVAL_PERCENTILE) are candidates for removal.

    Args:
        score: Agent's fitness score
        population_scores: All fitness scores in population

    Returns:
        True if agent is in bottom percentile
    """
    if len(population_scores) == 0:
        return False

    threshold = np.percentile(population_scores, SURVIVAL_PERCENTILE * 100)
    return score <= threshold


def emit_entropy_receipt(
    source_id: str,
    measurement: EntropyMeasurement,
    delta: Optional[EntropyDelta] = None,
    coherence: Optional[CoherenceResult] = None,
) -> dict:
    """Emit receipt for entropy measurement.

    Args:
        source_id: Identifier for the data source
        measurement: Entropy measurement result
        delta: Optional entropy delta
        coherence: Optional coherence result

    Returns:
        Receipt dict
    """
    payload = {
        "tenant_id": TENANT_ID,
        "source_id": source_id,
        "entropy_value": measurement.value,
        "entropy_bits": measurement.bits,
        "entropy_normalized": measurement.normalized,
        "source_size": measurement.source_size,
        "method": measurement.method,
    }

    if delta is not None:
        payload["delta"] = delta.delta
        payload["delta_health"] = delta.health_status
        payload["is_pumping"] = delta.is_pumping

    if coherence is not None:
        payload["coherence_score"] = coherence.score
        payload["is_alive"] = coherence.is_alive
        payload["autocatalytic"] = coherence.autocatalytic

    return emit_receipt("entropy", payload)


# === THOMPSON SAMPLING (replaces Boltzmann selection) ===


@dataclass
class ThompsonState:
    """State for Thompson sampling."""

    alpha: float  # Successes + 1
    beta: float  # Failures + 1

    @classmethod
    def new(cls) -> "ThompsonState":
        """Create new uninformed prior (Beta(1,1) = Uniform)."""
        return cls(alpha=1.0, beta=1.0)

    def sample(self) -> float:
        """Draw from posterior Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, success: bool) -> "ThompsonState":
        """Update posterior with observation."""
        if success:
            return ThompsonState(self.alpha + 1, self.beta)
        else:
            return ThompsonState(self.alpha, self.beta + 1)

    def mean(self) -> float:
        """Expected value of posterior."""
        return self.alpha / (self.alpha + self.beta)


def thompson_select(agents: List[ThompsonState]) -> int:
    """Select agent using Thompson sampling.

    Thompson sampling balances exploration vs exploitation
    by sampling from posterior distributions.

    Preferred over Boltzmann selection per xAI constants.

    Args:
        agents: List of ThompsonState for each agent

    Returns:
        Index of selected agent
    """
    if len(agents) == 0:
        raise ValueError("Cannot select from empty agent list")

    samples = [agent.sample() for agent in agents]
    return int(np.argmax(samples))
