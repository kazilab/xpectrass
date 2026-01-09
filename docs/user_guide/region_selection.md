# Region Selection

Select or exclude specific wavenumber regions for focused analysis.

## Overview

```python
from xpectrass.utils import select_region, exclude_regions, FTIR_REGIONS

# Predefined regions
print(FTIR_REGIONS.keys())
# fingerprint, ch_stretch, carbonyl, aromatic, ...

# Select regions
fingerprint_df = select_region(df, 'fingerprint')
```

## Predefined Regions

### Main Regions

| Name | Range (cm⁻¹) | Description |
|------|--------------|-------------|
| `full` | 400-4000 | Complete spectrum |
| `fingerprint` | 400-1500 | Unique molecular signatures |
| `functional` | 1500-4000 | Functional group region |

### Functional Groups

| Name | Range (cm⁻¹) | Description |
|------|--------------|-------------|
| `ch_stretch` | 2800-3100 | C-H stretching |
| `ch_bend` | 1350-1480 | C-H bending |
| `carbonyl` | 1650-1800 | C=O stretch |
| `aromatic` | 1400-1600 | Aromatic ring |
| `oh_stretch` | 3200-3600 | O-H stretching |
| `ether` | 1000-1300 | C-O-C stretch |

### Plastic-Specific

| Name | Range (cm⁻¹) | Plastic |
|------|--------------|---------|
| `hdpe_ldpe` | 700-750 | PE identification |
| `pp_methyl` | 1370-1380 | PP CH₃ deformation |
| `ps_aromatic` | 690-760 | PS benzene |
| `pet_ester` | 1710-1730 | PET C=O |
| `pvc_ccl` | 600-700 | PVC C-Cl |

### Atmospheric

| Name | Range (cm⁻¹) | Source |
|------|--------------|--------|
| `co2` | 2300-2400 | CO₂ |
| `h2o_bend` | 1350-1900 | H₂O |
| `h2o_stretch` | 3550-3900 | H₂O |

## Functions

### select_region

Select specific wavenumber ranges:

```python
# By name
fingerprint = select_region(df, 'fingerprint')

# By range
ch_region = select_region(df, (2800, 3100))

# Multiple regions
selected = select_region(df, [(400, 1500), (2800, 3100)])
```

### exclude_regions

Remove specific ranges:

```python
# Exclude atmospheric
no_atm = exclude_regions(df, 'co2')

# Exclude multiple
clean = exclude_regions(df, [
    (2300, 2400),   # CO2
    (3550, 3900)    # H2O
])
```

### exclude_atmospheric

Convenience function to exclude all atmospheric bands:

```python
from xpectrass.utils import exclude_atmospheric

clean_df = exclude_atmospheric(df)
```

## NumPy Functions

For working with arrays directly:

```python
from xpectrass.utils import select_region_np, select_regions_np

# Single region
selected_int, selected_wn = select_region_np(
    intensities, wavenumbers, start=400, end=1500
)

# Multiple regions
selected_int, selected_wn = select_regions_np(
    intensities, wavenumbers,
    regions=[(400, 1500), (2800, 3100)]
)
```

## Analysis

Analyze intensity statistics across regions:

```python
from xpectrass.utils import analyze_regions

stats = analyze_regions(df)
print(stats)
#         region  start_cm  end_cm  n_points  mean_intensity  ...
# 0  fingerprint     400    1500      1150          97.5     ...
# 1   ch_stretch    2800    3100       312          85.2     ...
```

## Helper Functions

```python
from xpectrass.utils import (
    get_region_names,    # List all region names
    get_region_range,    # Get range for named region
    get_wavenumbers,     # Extract wavenumber array from df
    get_spectra_matrix   # Extract spectra as numpy matrix
)

# Get wavenumber array
wavenumbers = get_wavenumbers(df)

# Get spectra matrix
spectra = get_spectra_matrix(df)  # Shape: (n_samples, n_wavenumbers)
```

## Example: Classification Regions

```python
from xpectrass.utils import select_region

# Key regions for plastic classification
classification_regions = [
    (400, 1800),    # Fingerprint + carbonyl
    (2700, 3100)    # CH stretch region
]

training_df = select_region(df, classification_regions)
print(f"Reduced from {len(get_wavenumbers(df))} to {len(get_wavenumbers(training_df))} features")
```
