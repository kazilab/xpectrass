# Data Validation

The data validation module ensures your FTIR spectral data is complete, consistent, and ready for preprocessing.

## Overview

Data validation is the first step in any preprocessing workflow. It checks for:

- **Completeness**: All expected samples and classes present
- **Missing values**: NaN or infinite values in intensity data
- **Intensity ranges**: Values within expected bounds
- **Wavenumber consistency**: Matching spectral grids across samples
- **Duplicates**: Duplicate sample names

## Functions

### validate_spectra

```python
from xpectrass.utils import validate_spectra

report = validate_spectra(
    df,
    expected_samples_per_class=500,
    expected_classes=['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC'],
    wavenumber_range=(399.0, 4000.0),
    intensity_range=(0.0, 150.0),
    verbose=True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | pl.DataFrame | required | Wide-format DataFrame with sample, label, and wavenumber columns |
| `expected_samples_per_class` | int | 500 | Expected number of samples per class |
| `expected_classes` | list | HDPE, LDPE, PET, PP, PS, PVC | Expected class labels |
| `wavenumber_range` | tuple | (399.0, 4000.0) | Expected wavenumber range |
| `intensity_range` | tuple | (0.0, 150.0) | Valid intensity range for %T |
| `verbose` | bool | True | Print validation report |

#### Returns

Dictionary with validation results:

```python
{
    'valid': True,              # Overall pass/fail
    'n_samples': 3000,          # Total samples
    'n_wavenumbers': 3751,      # Spectral points
    'class_counts': {...},      # Samples per label
    'missing_values': 0,        # NaN/Inf count
    'out_of_range': {...},      # Intensity issues
    'wavenumber_check': {...},  # Range info
    'duplicates': [],           # Duplicate names
    'issues': []                # Issue descriptions
}
```

### detect_outlier_spectra

Detect spectra that deviate significantly from the dataset:

```python
from xpectrass.utils import detect_outlier_spectra

result = detect_outlier_spectra(
    df,
    method='zscore',  # 'zscore', 'iqr', or 'mad'
    threshold=3.0
)

print(f"Found {result['n_outliers']} outlier spectra")
print("Outlier samples:", result['outlier_samples'])
```

#### Methods

| Method | Description |
|--------|-------------|
| `zscore` | Flag spectra with mean intensity > threshold std from global mean |
| `iqr` | Interquartile range method (1.5 × IQR default) |
| `mad` | Median Absolute Deviation (robust to outliers) |

### check_wavenumber_consistency

Verify all files have consistent wavenumber grids:

```python
from xpectrass.utils import check_wavenumber_consistency

result = check_wavenumber_consistency(
    file_paths=['file1.csv', 'file2.csv', ...],
    skiprows=15,
    tolerance=0.1
)

if result['consistent']:
    print("All files have matching wavenumber grids")
else:
    print("Mismatched files:", result['mismatched_files'])
```

## Example Output

```
============================================================
FTIR DATA VALIDATION REPORT
============================================================
Total samples:      3000
Wavenumber points:  3751
Classes:            ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']
Samples per class:  {'HDPE': 500, 'LDPE': 500, 'PET': 500, 'PP': 500, 'PS': 500, 'PVC': 500}
Missing values:     0
Out of range:       12
Duplicates:         0
------------------------------------------------------------
ISSUES FOUND:
  ⚠ 12 samples have intensities outside [0.0, 150.0]
------------------------------------------------------------
VALIDATION STATUS:  PASSED ✓
============================================================
```

## Best Practices

1. **Always validate before preprocessing**: Catch data issues early
2. **Check for outliers**: Remove or investigate anomalous spectra
3. **Verify class balance**: Ensure equal representation for classification
4. **Document exclusions**: Keep track of any removed samples
