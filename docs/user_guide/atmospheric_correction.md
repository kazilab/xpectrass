# Atmospheric Correction

Atmospheric correction removes CO2 and H2O vapor interference from FTIR spectra.

## Overview

In xpectrass, atmospheric correction is typically applied to full DataFrames (rows = samples, columns = wavenumbers).

```python
from xpectrass.data import load_jung_2018
from xpectrass.utils import atmospheric_correction

df = load_jung_2018().head(50)

corrected_df = atmospheric_correction(
    data=df,
    method="interpolate",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)
```

## Interference Regions

| Band | Wavenumber Range | Source |
|------|------------------|--------|
| CO2 | 2300-2400 cm^-1 | Asymmetric stretch |
| H2O bend | 1350-1900 cm^-1 | Bending mode |
| H2O stretch | 3550-3900 cm^-1 | Stretching modes |

## Methods

Valid methods for `atmospheric_correction(..., method=...)`:

- `interpolate` (default)
- `spline`
- `reference`
- `zero`
- `exclude`

```python
# Spline interpolation
df_spline = atmospheric_correction(
    data=df,
    method="spline",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)

# Set atmospheric regions to local baseline
df_zero = atmospheric_correction(
    data=df,
    method="zero",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)
```

## Custom Regions

Use `co2_ranges` and `h2o_ranges` (lists of tuples):

```python
df_custom = atmospheric_correction(
    data=df,
    method="interpolate",
    co2_ranges=[(2280, 2420)],
    h2o_ranges=[(1400, 1850), (3600, 3850)],
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
)
```

## Detection

Check atmospheric interference on a single spectrum:

```python
import numpy as np
from xpectrass.utils import identify_atmospheric_features

spectral_cols = [c for c in df.columns if c not in ["study", "sample_id", "type", "environmental", "resolution"]]
wavenumbers = np.array([float(c) for c in spectral_cols])
intensities = df.iloc[0][spectral_cols].to_numpy(dtype=float)

report = identify_atmospheric_features(intensities, wavenumbers)

if report["co2_detected"]:
    print("CO2 interference detected")
if report["h2o_detected"]:
    print("H2O interference detected")
```

## When to Use

| Measurement Mode | Recommendation |
|------------------|----------------|
| ATR | Often minimal correction needed |
| Transmission | Usually recommended |
| External reflection | Often helpful |
