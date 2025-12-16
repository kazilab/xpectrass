# Spectral Derivatives

Spectral derivatives enhance spectral resolution and remove baseline effects.

## Overview

Derivatives are computed using the Savitzky-Golay algorithm, which combines smoothing with differentiation.

```python
from xpectrass.utils import spectral_derivative, first_derivative, second_derivative

# First derivative
d1 = first_derivative(intensities)

# Second derivative
d2 = second_derivative(intensities)
```

## Benefits

### First Derivative
- Removes constant baseline offset
- Resolves overlapping bands
- Enhances small spectral differences
- Shifts peaks slightly from original positions

### Second Derivative
- Removes linear baseline
- Sharpens peaks (appear as negative minima)
- Increases noise - requires good smoothing
- Widely used in FTIR for band identification

## Functions

### spectral_derivative

General-purpose derivative function:

```python
deriv = spectral_derivative(
    intensities,
    order=1,           # Derivative order (1, 2, 3, ...)
    window_length=15,  # Savitzky-Golay window (odd)
    polyorder=3,       # Polynomial order
    delta=1.0          # Sample spacing
)
```

### first_derivative

```python
d1 = first_derivative(
    intensities,
    window_length=15,
    polyorder=3
)
```

### second_derivative

```python
d2 = second_derivative(
    intensities,
    window_length=15,
    polyorder=4  # Higher order for 2nd derivative
)
```

### gap_derivative

Norris-Williams gap derivative - more noise-resistant:

```python
from xpectrass.utils import gap_derivative

d_gap = gap_derivative(
    intensities,
    gap=5,      # Points to skip
    segment=5   # Points to average
)
```

## Parameter Guidelines

### Window Length
- Larger = more smoothing, less noise
- Smaller = better resolution, more noise
- Typical: 9-21 for FTIR

### Polynomial Order
- Must be less than window_length
- Higher = better peak preservation
- For 1st derivative: polyorder ≥ 2
- For 2nd derivative: polyorder ≥ 3

## Noise Considerations

Derivatives amplify high-frequency noise:
- 1st derivative: noise increases
- 2nd derivative: noise increases significantly

**Solutions:**
1. Increase window_length
2. Pre-smooth with denoising
3. Use gap derivative

## Batch Processing

```python
from xpectrass.utils import derivative_batch

# Apply to matrix of spectra
derivatives = derivative_batch(
    spectra_matrix,
    order=1,
    window_length=15
)
```

## Visualization

```python
from xpectrass.utils import plot_derivatives

plot_derivatives(
    intensities,
    wavenumbers,
    orders=[0, 1, 2],  # Original, 1st, 2nd
    sample_name='HDPE1'
)
```

## Example

```python
from xpectrass.utils import spectral_derivative
import numpy as np

# Load spectrum
intensities = load_spectrum('sample.csv')
wavenumbers = np.linspace(400, 4000, 3751)

# Second derivative for peak identification
d2 = spectral_derivative(intensities, order=2, window_length=17)

# Find peaks (minima in 2nd derivative)
peak_indices = np.where(d2[1:-1] < d2[:-2]) & (d2[1:-1] < d2[2:])
peak_wavenumbers = wavenumbers[1:-1][peak_indices]
```
