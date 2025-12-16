# Scatter Correction

Scatter correction removes light scattering effects that cause baseline shifts and multiplicative intensity variations.

## Overview

```python
from xpectrass.utils import scatter_correction, scatter_method_names

# Available methods
print(scatter_method_names())
# ['emsc', 'msc', 'snv', 'snv_detrend']

# Apply correction (requires matrix of spectra)
corrected = scatter_correction(spectra_matrix, method='msc')
```

## Methods

### MSC (Multiplicative Scatter Correction)

Corrects for additive and multiplicative scatter effects by regressing each spectrum against a reference.

**Model:** spectrum = a + b × reference
**Correction:** (spectrum - a) / b

```python
corrected = scatter_correction(
    spectra_matrix,
    method='msc',
    reference=None  # Uses mean if None
)
```

### EMSC (Extended MSC)

Extends MSC with polynomial baseline terms for more complex scatter patterns.

**Model:** spectrum = a + b × reference + c₁x + c₂x² + ...

```python
corrected = scatter_correction(
    spectra_matrix,
    method='emsc',
    poly_order=2
)
```

### SNV (Standard Normal Variate)

Per-spectrum normalization (mean=0, std=1). Simple but effective.

```python
corrected = scatter_correction(spectra_matrix, method='snv')
```

### SNV + Detrend

SNV followed by polynomial detrending to remove residual slope.

```python
corrected = scatter_correction(
    spectra_matrix,
    method='snv_detrend',
    detrend_order=1
)
```

## Comparison

| Method | Removes Offset | Removes Scale | Removes Curvature | Requires Reference |
|--------|---------------|---------------|-------------------|-------------------|
| MSC | ✓ | ✓ | ✗ | ✓ (or mean) |
| EMSC | ✓ | ✓ | ✓ | ✓ (or mean) |
| SNV | ✓ | ✓ | ✗ | ✗ |
| SNV+Detrend | ✓ | ✓ | Partial | ✗ |

## When to Use

- **MSC/EMSC:** When samples have similar composition but different physical properties (particle size, path length)
- **SNV:** Quick correction when no reference is available
- **SNV+Detrend:** When SNV leaves residual slope

## Single Spectrum MSC

```python
from xpectrass.utils import msc_single

corrected, offset, scale = msc_single(spectrum, reference)
```

## Evaluation

```python
from xpectrass.utils import evaluate_scatter_correction

metrics = evaluate_scatter_correction(spectra_matrix, method='msc')
print(f"Variance reduction: {metrics['variance_ratio']:.2f}")
print(f"Correlation improvement: {metrics['correlation_improvement']:.3f}")
```

## Example

```python
from xpectrass.utils import scatter_correction
import numpy as np

# Load spectra matrix (n_samples × n_wavenumbers)
spectra = load_all_spectra()

# MSC correction using mean as reference
msc_corrected = scatter_correction(spectra, method='msc')

# Compare variance before/after
print(f"Variance before: {np.var(spectra, axis=0).mean():.4f}")
print(f"Variance after: {np.var(msc_corrected, axis=0).mean():.4f}")
```
