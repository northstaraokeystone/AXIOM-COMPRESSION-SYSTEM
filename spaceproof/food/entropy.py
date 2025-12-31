"""entropy.py - Food-Specific Entropy Calculators.

All entropy calculations for food domain:
- Spectral entropy (NIR/hyperspectral for oils, wines)
- Texture entropy (2D spatial for honey, processed foods)
- Gradient entropy (moisture/density for grains, produce)
- Pollen diversity entropy (species distribution for honey)

Source: Grok Research - Food fraud detection methods
"""

import numpy as np
from typing import Dict, List, Union


def spectral_entropy(
    spectrum: Union[np.ndarray, List[float]],
    bins: int = 50,
) -> float:
    """Compute Shannon entropy of spectral wavelength distribution.

    Used for: Olive oil, wine, spices (NIR/hyperspectral analysis)

    Genuine products: Natural random gradients → HIGH entropy (3.8-4.6)
    Adulterants: Homogenized mixtures → LOW entropy (<3.2)

    Args:
        spectrum: Array of wavelength intensity readings
        bins: Number of bins for histogram (default 50)

    Returns:
        Spectral entropy value (typically 0-8 bits)
    """
    if isinstance(spectrum, list):
        spectrum = np.array(spectrum)

    if len(spectrum) == 0:
        return 0.0

    # Normalize to positive values
    spectrum = np.abs(spectrum)

    if np.sum(spectrum) == 0:
        return 0.0

    # Create histogram distribution
    hist, _ = np.histogram(spectrum, bins=bins, density=True)

    # Remove zeros and normalize
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    # Shannon entropy: H = -Σ(p_i * log2(p_i))
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def texture_entropy(
    image: np.ndarray,
    window_size: int = 8,
) -> float:
    """Compute 2D spatial entropy using sliding window.

    Used for: Honey (crystallization patterns), processed foods

    Genuine products: Inherent microstructural randomness → HIGH entropy
    Adulterated: Uniform texture from added syrups → LOW entropy

    Args:
        image: 2D grayscale image array (or flattened)
        window_size: Size of sliding window for local variance

    Returns:
        Texture entropy value (typically 0-8 bits)
    """
    if image.size == 0:
        return 0.0

    # Flatten if needed for 1D analysis
    if len(image.shape) > 1:
        # Calculate local variance in windows
        image_flat = image.flatten()
    else:
        image_flat = image

    # Normalize to 0-255 range
    if image_flat.max() > 0:
        image_norm = (image_flat - image_flat.min()) / (image_flat.max() - image_flat.min()) * 255
    else:
        image_norm = image_flat

    # Histogram-based entropy
    hist, _ = np.histogram(image_norm, bins=256, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def gradient_entropy(
    scan: np.ndarray,
    axis: int = 0,
) -> float:
    """Compute entropy of moisture/density gradients.

    Used for: Grains, produce, any food with spatial variation

    Genuine products: Natural biological gradients → HIGH entropy
    Processed/adulterated: Uniform distribution → LOW entropy

    Args:
        scan: N-dimensional scan data (density, moisture, etc.)
        axis: Axis along which to compute gradient

    Returns:
        Gradient entropy value
    """
    if scan.size == 0:
        return 0.0

    # Flatten if 1D
    if len(scan.shape) == 1:
        gradient = np.diff(scan)
    else:
        gradient = np.diff(scan, axis=axis).flatten()

    if len(gradient) == 0:
        return 0.0

    # Normalize gradient
    if np.std(gradient) > 0:
        gradient_norm = (gradient - np.mean(gradient)) / np.std(gradient)
    else:
        return 0.0

    # Histogram of gradients
    hist, _ = np.histogram(gradient_norm, bins=50, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def pollen_diversity_entropy(
    pollen_counts: Dict[str, int],
) -> float:
    """Compute Shannon entropy of pollen species distribution.

    Used for: Honey authentication (pollen analysis)

    Genuine honey: Diverse pollen species → HIGH entropy
    Adulterated/fake: Few or no pollen → LOW entropy

    Args:
        pollen_counts: Dict mapping species names to counts

    Returns:
        Pollen diversity entropy value
    """
    if not pollen_counts:
        return 0.0

    counts = np.array(list(pollen_counts.values()), dtype=float)

    if np.sum(counts) == 0:
        return 0.0

    # Normalize to probabilities
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)


def combined_food_entropy(
    primary_entropy: float,
    secondary_entropy: float,
    primary_weight: float = 0.6,
) -> float:
    """Combine two entropy measurements with weighting.

    Args:
        primary_entropy: Primary entropy measurement
        secondary_entropy: Secondary entropy measurement
        primary_weight: Weight for primary (default 0.6)

    Returns:
        Weighted combined entropy
    """
    secondary_weight = 1.0 - primary_weight
    return primary_entropy * primary_weight + secondary_entropy * secondary_weight
