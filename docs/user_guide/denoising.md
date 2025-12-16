# Denoising

The denoising module provides multiple methods for reducing noise in FTIR spectra while preserving important spectral features.

## Overview

```python
from xpectrass.utils import denoise, denoise_method_names

# See available methods
print(denoise_method_names())
# ['gaussian', 'lowpass', 'median', 'moving_average', 'savgol', 'wavelet', 'whittaker']

# Apply denoising
denoised = denoise(intensities, method='savgol', window_length=15, polyorder=3)
```

## Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `savgol` | Savitzky-Golay filter | **Default choice** - preserves peak shape |
| `wavelet` | Discrete wavelet transform | Variable noise levels |
| `moving_average` | Simple box filter | Quick preprocessing |
| `gaussian` | Gaussian smoothing | Low-frequency noise |
| `median` | Non-linear median filter | Spike/impulse noise |
| `whittaker` | Penalized least squares | Balanced smoothing |
| `lowpass` | Butterworth filter | High-frequency noise |

## Method Details

### Savitzky-Golay (Recommended)

Fits successive sub-sets of adjacent data points with a low-degree polynomial. Excellent for preserving peak shapes and positions.

```python
denoised = denoise(
    intensities,
    method='savgol',
    window_length=15,  # Must be odd
    polyorder=3        # Polynomial order
)
```

**Guidelines:**
- Larger `window_length` = more smoothing
- Higher `polyorder` = better peak preservation
- Typical: `window_length=11-21`, `polyorder=2-4`

### Wavelet Denoising

Multi-resolution approach that decomposes the signal into wavelet coefficients and thresholds small (noise) coefficients.

```python
denoised = denoise(
    intensities,
    method='wavelet',
    wavelet='db4',        # Wavelet family
    level=3,              # Decomposition level
    threshold_mode='soft' # 'soft' or 'hard'
)
```

**Common wavelets:** `'db4'`, `'db6'`, `'sym4'`, `'coif2'`

### Median Filter

Non-linear filter that replaces each value with the median of neighboring values. Excellent for removing spike noise.

```python
denoised = denoise(
    intensities,
    method='median',
    kernel_size=5  # Must be odd
)
```

### Gaussian Filter

Convolves the signal with a Gaussian kernel.

```python
denoised = denoise(
    intensities,
    method='gaussian',
    sigma=2.0  # Standard deviation
)
```

### Moving Average

Simple box filter that averages neighboring points.

```python
denoised = denoise(
    intensities,
    method='moving_average',
    window=11
)
```

### Whittaker Smoother

Penalized least squares that balances fidelity with smoothness.

```python
denoised = denoise(
    intensities,
    method='whittaker',
    lam=1e4,  # Smoothness parameter
    d=2       # Derivative order
)
```

### Low-Pass Filter

Butterworth filter that removes high-frequency components.

```python
denoised = denoise(
    intensities,
    method='lowpass',
    cutoff=0.1,  # Normalized frequency (0-1)
    order=4      # Filter order
)
```

## Evaluation

### SNR Estimation

```python
from xpectrass.utils import estimate_snr

# Compare raw vs denoised
snr_improvement = estimate_snr(y_raw, y_denoised)
print(f"SNR improvement: {snr_improvement:.1f} dB")
```

### Batch Evaluation

```python
from xpectrass.utils import evaluate_denoising

# Compare methods on dataset
results = evaluate_denoising(df, methods=['savgol', 'wavelet', 'gaussian'])

# Results include SNR, smoothness, and fidelity metrics
print(results.groupby('method').mean())
```

## Visualization

```python
from xpectrass.utils import plot_denoising_comparison

plot_denoising_comparison(
    intensities,
    wavenumbers,
    methods=['savgol', 'wavelet', 'gaussian'],
    sample_name='HDPE1'
)
```

## Recommendations

| Situation | Recommended Method |
|-----------|-------------------|
| General FTIR | `savgol` (window=15, polyorder=3) |
| High noise | `wavelet` with higher level |
| Spike noise | `median` first, then `savgol` |
| Real-time processing | `moving_average` (fastest) |
| Best peak preservation | `savgol` with small window |

## Example

```python
from xpectrass.utils import denoise
import numpy as np

# Apply two-stage denoising for spike + random noise
spectrum = load_spectrum('sample.csv')

# Stage 1: Remove spikes
no_spikes = denoise(spectrum, method='median', kernel_size=3)

# Stage 2: Smooth
smooth = denoise(no_spikes, method='savgol', window_length=11, polyorder=3)
```
