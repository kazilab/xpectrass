# Scatter Correction

Scatter correction removes light-scattering effects that introduce baseline shifts and multiplicative intensity differences.

## Overview

```python
import numpy as np
from xpectrass.utils import scatter_correction, scatter_method_names

print(scatter_method_names())
# ['emsc', 'msc', 'snv', 'snv_detrend']

# Single-spectrum example
reference = np.mean(spectra_matrix, axis=0)
corrected = scatter_correction(
    intensities=spectra_matrix[0],
    method="msc",
    reference=reference,
)
```

## Methods

### MSC (Multiplicative Scatter Correction)

```python
reference = np.mean(spectra_matrix, axis=0)
corrected = scatter_correction(
    intensities=spectra_matrix[0],
    method="msc",
    reference=reference,
)
```

### EMSC (Extended MSC)

```python
reference = np.mean(spectra_matrix, axis=0)
corrected = scatter_correction(
    intensities=spectra_matrix[0],
    method="emsc",
    reference=reference,
    poly_order=2,
)
```

### SNV

```python
corrected = scatter_correction(
    intensities=spectra_matrix[0],
    method="snv",
)
```

### SNV + Detrend

```python
corrected = scatter_correction(
    intensities=spectra_matrix[0],
    method="snv_detrend",
    detrend_order=1,
)
```

## Batch DataFrame Correction

For full datasets, use `apply_scatter_correction`:

```python
from xpectrass.data import load_jung_2018
from xpectrass.utils import apply_scatter_correction, convert_spectra

df = load_jung_2018().head(100)
df_abs = convert_spectra(
    df,
    mode="to_absorbance",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)

corrected_df = apply_scatter_correction(
    data=df_abs,
    method="msc",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)
```

## Single Spectrum MSC Diagnostics

```python
from xpectrass.utils import msc_single

corrected, offset, scale = msc_single(spectrum, reference)
```

## Quick Quality Check

```python
import numpy as np

variance_before = np.var(spectra_matrix, axis=0).mean()
variance_after = np.var(corrected_matrix, axis=0).mean()

print(f"Variance before: {variance_before:.4f}")
print(f"Variance after: {variance_after:.4f}")
```
