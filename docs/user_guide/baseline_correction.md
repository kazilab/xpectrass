# Baseline Correction

Baseline correction removes instrumental artifacts and background signals from FTIR spectra.

## Overview

The xpectrass baseline module wraps **pybaselines** to provide 50+ baseline correction algorithms through a unified interface.

```python
from xpectrass.utils import baseline_correction, baseline_method_names

# See all available methods
print(baseline_method_names())

# Apply baseline correction
corrected = baseline_correction(intensities, method='airpls', lam=1e6)
```

## Available Methods

### Whittaker-Based Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `asls` | Asymmetric Least Squares | General purpose |
| `airpls` | Adaptive Iteratively Reweighted Penalized LS | **Default choice for FTIR** |
| `arpls` | Asymmetrically Reweighted Penalized LS | Strong baselines |
| `iasls` | Improved AsLS | Better peak preservation |
| `psalsa` | Peak-Screening AsLS | Sharp peaks |
| `aspls` | Adaptive Smoothness Penalized LS | Variable smoothness |

### Polynomial-Based Methods

| Method | Description |
|--------|-------------|
| `poly` | Standard polynomial fit |
| `modpoly` | Modified polynomial |
| `imodpoly` | Iterative modified polynomial |
| `penalized_poly` | Penalized polynomial |
| `loess` | Local regression |

### Morphological Methods

| Method | Description |
|--------|-------------|
| `mor` | Morphological opening |
| `imor` | Iterative morphological |
| `mormol` | Morphological and mollification |
| `rolling_ball` | Rolling ball algorithm |
| `tophat` | Top-hat transform |

### Spline-Based Methods

| Method | Description |
|--------|-------------|
| `mixture_model` | Mixture model approach |
| `irsqr` | Iteratively reweighted spline quantile regression |
| `pspline_asls` | Penalized spline AsLS |

### Custom Methods

| Method | Description |
|--------|-------------|
| `median_filter` | Median filter baseline |
| `adaptive_window` | Adaptive minimum filter |

## Function Reference

### baseline_correction

```python
corrected = baseline_correction(
    intensities,           # 1-D array of intensities
    method='airpls',       # Algorithm name
    window_size=101,       # For custom windowed filters
    poly_order=4,          # For polynomial methods
    clip_negative=True,    # Set negative values to 0
    return_baseline=False, # Return (corrected, baseline) tuple
    **kwargs               # Method-specific parameters
)
```

#### Common Parameters by Method

**For Whittaker methods (`asls`, `airpls`, `arpls`, etc.):**
- `lam`: Smoothness parameter (typically 1e4 to 1e8). Higher = smoother baseline.
- `p`: Asymmetry parameter (typically 0.001 to 0.1). Lower = less peak influence.

**For polynomial methods:**
- `poly_order`: Polynomial degree (typically 2-6)

## Evaluation

Compare baseline methods using RFZN and NAR metrics:

```python
from xpectrass.utils import evaluate_all_samples

# Define flat regions (known baseline-only areas)
flat_windows = [(2500, 2600), (3350, 3450)]

# Evaluate all methods
rfzn, nar, snr = evaluate_all_samples(df, flat_windows)

# Lower RFZN and NAR = better baseline correction
print("Best methods by RFZN:", rfzn.mean().sort_values().head())
```

### Metrics

| Metric | Full Name | Interpretation |
|--------|-----------|----------------|
| RFZN | Residual Flat-Zone Noise | RMS of corrected signal in known baseline regions. Lower = better. |
| NAR | Negative Area Ratio | Fraction of negative area. Lower = better. |
| SNR | Signal-to-Noise Ratio | Peak height / noise. Higher = better. |

## Visualization

```python
from xpectrass.utils import plot_corrected_spectrum

plot_corrected_spectrum(df, sample_name='HDPE1', method='airpls')
```

## Recommendations for Plastics

| Plastic Type | Recommended Method | Notes |
|--------------|-------------------|-------|
| HDPE, LDPE | `airpls` | Strong CH peaks, smooth baseline |
| PET | `asls` or `airpls` | Complex spectrum |
| PP | `airpls` | Similar to PE |
| PS | `airpls` | Aromatic features |
| PVC | `airpls` or `arpls` | May have strong baseline drift |

## Example

```python
import numpy as np
from xpectrass.utils import baseline_correction

# Load spectrum
wavenumbers = np.linspace(400, 4000, 3751)
intensities = load_spectrum('HDPE1.csv')

# Apply baseline correction
corrected = baseline_correction(
    intensities,
    method='airpls',
    lam=1e6
)

# Get baseline for visualization
corrected, baseline = baseline_correction(
    intensities,
    method='airpls',
    lam=1e6,
    return_baseline=True
)
```
