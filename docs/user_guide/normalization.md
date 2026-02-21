# Normalization

Normalization standardizes spectral intensities to enable comparison between samples measured under different conditions.

## Overview

### Using FTIRdataprocessing Class (Recommended)

The easiest way to apply normalization is through the `FTIRdataprocessing` class with built-in evaluation:

```python
from xpectrass import FTIRdataprocessing

# Initialize with your data
ftir = FTIRdataprocessing(df, label_column="type")

# Convert to absorbance first (normalization expects absorbance-scale spectra)
df_abs = ftir.convert(mode="to_absorbance", plot=False)

# Step 1: Evaluate all normalization methods to find the best one
norm_results = ftir.find_normalization_method(
    data=df_abs,
    methods="FTIR",
    n_splits=5,
)

# Step 2: View evaluation results
print(norm_results.head())

# Step 3: Apply the best method
best_method = norm_results.iloc[0]["method"]
ftir.normalize(data=df_abs, method=best_method, plot=False)

# Step 4: Access normalized data
normalized_df = ftir.df_norm
```

### Using Utility Functions Directly

For standalone use or custom pipelines:

```python
from xpectrass.utils import normalize, normalize_method_names

# See available methods
print(normalize_method_names())
# e.g. ['adaptive_regional', 'area', 'curvature_weighted', ..., 'snv', 'vector']

# Apply normalization to a single spectrum
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
centered, mean = mean_center(spectra_matrix, axis=0, return_mean=True)
```

### Auto-Scaling

Mean centering + unit variance scaling. Each variable has mean=0, std=1.

```python
from xpectrass.utils import auto_scale

scaled, mean, std = auto_scale(spectra_matrix, return_params=True)
```

### Pareto Scaling

Less aggressive than auto-scaling - divides by sqrt(std) instead of std.

```python
from xpectrass.utils import pareto_scale

scaled, mean, std = pareto_scale(spectra_matrix, return_params=True)
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
from xpectrass.utils import normalize_df

normalized_matrix = normalize_df(spectra_matrix, method="snv")
```

### DataFrame Operations

```python
from xpectrass.utils import normalize_df, mean_center

# Normalize Polars DataFrame
normalized_df = normalize_df(
    df,
    method="snv",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)

# Mean-center spectral matrix for PCA
spectral_cols = [c for c in normalized_df.columns if c not in ["study", "sample_id", "type", "environmental", "resolution"]]
centered_matrix, mean = mean_center(normalized_df[spectral_cols].to_numpy(), return_mean=True)
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
    normalize, normalize_df, mean_center,
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
normalized = normalize_df(spectra, method="snv")

# Mean center for PCA
centered, mean = mean_center(normalized, return_mean=True)
```
