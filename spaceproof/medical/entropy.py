"""entropy.py - Medical-Specific Entropy Calculators.

All entropy calculations for medical domain:
- Structural entropy (internal device structure via micro-CT, ultrasound)
- Surface topography entropy (surface finish via laser speckle, optical)
- Dimensional entropy (micro-variations in dimensions)
- API distribution entropy (active ingredient spatial distribution via Raman)

Source: Grok Research - Medical counterfeit detection methods
"""

import numpy as np
from typing import Dict, List, Union


def structural_entropy(
    scan: Union[np.ndarray, List[float]],
) -> float:
    """Compute internal device structure variance entropy.

    Used for: Medical device internal structure (micro-CT, ultrasound)

    Genuine devices: Controlled manufacturing variance → Expected entropy range
    Counterfeits: Wrong materials or poor construction → Abnormal entropy

    Args:
        scan: 3D scan data (flattened or multi-dimensional)

    Returns:
        Structural entropy value
    """
    if isinstance(scan, list):
        scan = np.array(scan)

    if scan.size == 0:
        return 0.0

    scan_flat = scan.flatten()

    # Normalize
    if scan_flat.max() > scan_flat.min():
        scan_norm = (scan_flat - scan_flat.min()) / (scan_flat.max() - scan_flat.min())
    else:
        return 0.0

    # Histogram-based entropy
    hist, _ = np.histogram(scan_norm, bins=100, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def surface_topography_entropy(
    scan: Union[np.ndarray, List[float]],
) -> float:
    """Compute surface finish randomness entropy.

    Used for: Vial/device surface analysis (laser speckle, optical scanning)

    Genuine vials: Controlled surface finish → Expected entropy range
    Counterfeits: Cheap materials with different texture → Abnormal entropy

    Args:
        scan: Surface topography data (laser speckle pattern, etc.)

    Returns:
        Surface entropy value
    """
    if isinstance(scan, list):
        scan = np.array(scan)

    if scan.size == 0:
        return 0.0

    scan_flat = scan.flatten()

    # Local variance analysis (sliding window)
    window_size = min(10, len(scan_flat) // 10 + 1)

    if window_size < 2:
        # Fall back to histogram-based entropy
        hist, _ = np.histogram(scan_flat, bins=50, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        hist = hist / np.sum(hist)
        return float(-np.sum(hist * np.log2(hist)))

    local_vars = []
    for i in range(0, len(scan_flat) - window_size, window_size // 2):
        window = scan_flat[i:i + window_size]
        local_vars.append(np.var(window))

    if not local_vars:
        return 0.0

    local_vars = np.array(local_vars)

    # Entropy of local variance distribution
    hist, _ = np.histogram(local_vars, bins=50, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def dimensional_entropy(
    measurements: Union[np.ndarray, List[float]],
) -> float:
    """Compute micro-variations in dimensions entropy.

    Used for: Device dimensions (precision 3D scan, micrometer measurements)

    Genuine devices: Controlled tolerance → Narrow variance distribution
    Counterfeits: Poor precision → Different variance pattern

    Args:
        measurements: Repeated dimension measurements

    Returns:
        Dimensional entropy value
    """
    if isinstance(measurements, list):
        measurements = np.array(measurements)

    if len(measurements) < 2:
        return 0.0

    # Calculate variance entropy
    mean_val = np.mean(measurements)

    if mean_val == 0:
        return 0.0

    # Normalized deviations from mean
    deviations = (measurements - mean_val) / mean_val

    # Histogram of deviations
    hist, _ = np.histogram(deviations, bins=50, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def api_distribution_entropy(
    raman_map: Union[np.ndarray, List[float]],
) -> float:
    """Compute active pharmaceutical ingredient spatial distribution entropy.

    Used for: Tablets/capsules (Raman spectroscopy mapping)

    Genuine tablets: API distribution randomness from validated processes
    Counterfeits: No API (near-zero entropy) or wrong distribution

    Args:
        raman_map: 3D Raman spectroscopy spatial mapping data

    Returns:
        API distribution entropy value
    """
    if isinstance(raman_map, list):
        raman_map = np.array(raman_map)

    if raman_map.size == 0:
        return 0.0

    raman_flat = raman_map.flatten()

    # Check for no-API (very low signal)
    signal_mean = np.mean(np.abs(raman_flat))
    if signal_mean < 0.01:  # Threshold for "no signal"
        return 0.1  # Near-zero entropy indicates no API

    # Normalize
    if raman_flat.max() > raman_flat.min():
        raman_norm = (raman_flat - raman_flat.min()) / (raman_flat.max() - raman_flat.min())
    else:
        return 0.5  # Uniform signal

    # Spatial entropy calculation
    hist, _ = np.histogram(raman_norm, bins=100, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def fill_variance_entropy(
    fill_measurements: Union[Dict, np.ndarray, List[float]],
) -> float:
    """Compute fill level variance entropy for injectable pens.

    Used for: GLP-1 pens, insulin pens (optical + X-ray measurements)

    Genuine pens: Manufacturing variance in fill → Expected entropy range (2.8-3.4)
    Counterfeits: Abnormal uniformity (too perfect) OR material deviations

    Args:
        fill_measurements: Fill level measurements or dict with 'fill_level', 'compression'

    Returns:
        Fill variance entropy value
    """
    if isinstance(fill_measurements, dict):
        # Extract fill-related measurements
        values = []
        for key in ["fill_level", "compression", "uniformity_score", "variance"]:
            if key in fill_measurements:
                val = fill_measurements[key]
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, (list, np.ndarray)):
                    values.extend(val if isinstance(val, list) else val.tolist())

        if not values:
            values = list(fill_measurements.values())
            values = [v for v in values if isinstance(v, (int, float))]

        measurements = np.array(values)
    elif isinstance(fill_measurements, list):
        measurements = np.array(fill_measurements)
    else:
        measurements = fill_measurements

    if measurements.size == 0:
        return 0.0

    # Normalize to 0-1 range
    if measurements.max() > measurements.min():
        norm = (measurements - measurements.min()) / (measurements.max() - measurements.min())
    else:
        # All same value - very low entropy (suspicious uniformity)
        return 0.5

    # Histogram entropy
    hist, _ = np.histogram(norm, bins=50, density=True)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    hist = hist / np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)
