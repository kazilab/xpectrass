# Atmospheric Correction

Atmospheric correction removes CO₂ and H₂O vapor interference from FTIR spectra.

## Overview

FTIR measurements can be affected by atmospheric CO₂ and water vapor absorption, especially in transmission mode. This module provides methods to correct for these interferences.

```python
from xpectrass.utils import atmospheric_correction, get_atmospheric_regions

# Standard atmospheric regions
regions = get_atmospheric_regions()
# {'co2': (2300, 2400), 'h2o': [(1350, 1900), (3550, 3900)]}

# Apply correction
corrected = atmospheric_correction(intensities, wavenumbers, method='interpolate')
```

## Interference Regions

| Band | Wavenumber Range | Source |
|------|------------------|--------|
| CO₂ | 2300-2400 cm⁻¹ | Asymmetric stretch |
| H₂O bend | 1350-1900 cm⁻¹ | Bending mode |
| H₂O stretch | 3550-3900 cm⁻¹ | Stretching modes |

## Methods

### Interpolation (Default)

Linear interpolation across affected regions using boundary values.

```python
corrected = atmospheric_correction(
    intensities,
    wavenumbers,
    method='interpolate'
)
```

### Spline Interpolation

Cubic spline for smoother transitions.

```python
corrected = atmospheric_correction(
    intensities,
    wavenumbers,
    method='spline'
)
```

### Reference Subtraction

Subtract a scaled reference atmospheric spectrum.

```python
corrected = atmospheric_correction(
    intensities,
    wavenumbers,
    method='reference',
    reference_spectrum=atm_reference,
    reference_scale=1.0  # Auto-fit if not provided
)
```

### Zero Baseline

Set affected regions to local baseline level.

```python
corrected = atmospheric_correction(
    intensities,
    wavenumbers,
    method='zero'
)
```

### Exclude (Mark as NaN)

Mark affected regions for exclusion from analysis.

```python
corrected = atmospheric_correction(
    intensities,
    wavenumbers,
    method='exclude'
)
```

## Custom Regions

```python
corrected = atmospheric_correction(
    intensities,
    wavenumbers,
    method='interpolate',
    co2_range=(2280, 2420),
    h2o_ranges=[(1400, 1850), (3600, 3850)]
)
```

## Detection

Check for atmospheric interference:

```python
from xpectrass.utils import identify_atmospheric_features

report = identify_atmospheric_features(intensities, wavenumbers)

if report['co2_detected']:
    print("CO₂ interference detected")
if report['h2o_detected']:
    print("H₂O interference detected")
```

## When to Use

| Measurement Mode | Recommendation |
|-----------------|----------------|
| ATR | Usually not needed (short path length) |
| Transmission | Often needed |
| External reflection | May be needed |

> **Note:** For ATR measurements with controlled atmospheres, atmospheric correction is often unnecessary.
