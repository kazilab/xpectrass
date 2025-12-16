# Normalization

Normalization standardizes spectral intensities to enable comparison between samples measured under different conditions.

## Overview

```python
from xpectrass.utils import normalize, normalize_method_names

# See available methods
print(normalize_method_names())
# ['area', 'max', 'minmax', 'peak', 'range', 'snv', 'vector']

# Apply normalization
normalized = normalize(intensities, method='snv')
```

## Available Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| `snv` | (x - mean) / std | **Default** - removes scatter effects |
| `vector` | x / ‖x‖₂ | Compare spectral shapes |
| `minmax` | (x - min) / (max - min) | Scale to [0, 1] |
| `area` | x / sum(\|x\|) | Total area = 1 |
| `peak` | x / x[ref] | Normalize to reference peak |
| `range` | x / (max - min) | Preserve relative intensities |
| `max` | x / max(\|x\|) | Maximum = 1 |

## Method Details

### Standard Normal Variate (SNV) - Recommended

Centers and scales each spectrum to have mean=0 and std=1. Effectively removes multiplicative scatter effects.

```python
normalized = normalize(intensities, method='snv')
# Result: mean ≈ 0, std ≈ 1
```

### Vector Normalization (L2)

Scales spectrum to unit length (Euclidean norm = 1).

```python
normalized = normalize(intensities, method='vector')
# Result: ||normalized||₂ = 1
```

### Min-Max Normalization

Scales values to a specified range (default [0, 1]).

```python
normalized = normalize(
    intensities,
    method='minmax',
    feature_range=(0, 1)
)
```

### Area Normalization

Scales so total absolute area equals 1.

```python
normalized = normalize(intensities, method='area')
# Result: sum(|normalized|) = 1
```

### Peak Normalization

Normalizes by intensity at a specific peak position.

```python
normalized = normalize(
    intensities,
    method='peak',
    peak_idx=1500  # Index of reference peak
)
```

## Scaling Methods for PCA/PLS

### Mean Centering

Essential preprocessing for PCA - centers each variable (wavenumber) across samples.

```python
from xpectrass.utils import mean_center

# Returns centered data and mean for reconstruction
centered, mean = mean_center(spectra_matrix, axis=0)
```

### Auto-Scaling

Mean centering + unit variance scaling. Each variable has mean=0, std=1.

```python
from xpectrass.utils import auto_scale

scaled, mean, std = auto_scale(spectra_matrix)
```

### Pareto Scaling

Less aggressive than auto-scaling - divides by sqrt(std) instead of std.

```python
from xpectrass.utils import pareto_scale

scaled, mean, std = pareto_scale(spectra_matrix)
```

## Detrending

Remove polynomial trends (often combined with SNV):

```python
from xpectrass.utils import detrend, snv_detrend

# Linear detrending
detrended = detrend(intensities, order=1)

# SNV + detrending (common combination)
snv_dt = snv_detrend(intensities, detrend_order=1)
```

## Batch Operations

### Normalize Multiple Spectra

```python
from xpectrass.utils import normalize_batch

normalized_matrix = normalize_batch(spectra_matrix, method='snv')
```

### DataFrame Operations

```python
from xpectrass.utils import normalize_dataframe, mean_center_dataframe

# Normalize Polars DataFrame
normalized_df = normalize_dataframe(df, method='snv')

# Mean center DataFrame
centered_df, mean = mean_center_dataframe(df)
```

## Comparison

| Method | Removes Offset | Removes Scale Diff | Preserves Shape | PCA Ready |
|--------|---------------|-------------------|-----------------|-----------|
| SNV | ✓ | ✓ | ✓ | ✓ |
| Vector | ✗ | ✓ | ✓ | Needs centering |
| MinMax | Partial | ✓ | ✓ | Needs centering |
| Area | ✗ | ✓ | ✓ | Needs centering |
| Mean Center | ✓ | ✗ | ✓ | ✓ |
| Auto-Scale | ✓ | ✓ | ✗ | ✓ |

## Recommendations

| Task | Recommended Method |
|------|-------------------|
| General preprocessing | `snv` |
| Classification | `snv` or `vector` |
| PCA/PLS | `snv` + `mean_center` or `auto_scale` |
| Quantitative analysis | `peak` (internal standard) |
| Visual comparison | `minmax` |

## Example

```python
from xpectrass.utils import (
    normalize, normalize_batch, mean_center,
    snv_detrend
)
import numpy as np

# Single spectrum
spectrum = load_spectrum('sample.csv')

# SNV normalization
snv_spectrum = normalize(spectrum, method='snv')

# SNV + detrending (scatter correction)
snv_dt = snv_detrend(spectrum)

# Batch processing
spectra = np.vstack([load_spectrum(f) for f in files])

# Normalize all
normalized = normalize_batch(spectra, method='snv')

# Mean center for PCA
centered, mean = mean_center(normalized)
```
