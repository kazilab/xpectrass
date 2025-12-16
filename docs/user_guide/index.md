# User Guide

```{toctree}
:maxdepth: 2

preprocessing_pipeline
data_validation
baseline_correction
denoising
normalization
atmospheric_correction
spectral_derivatives
scatter_correction
region_selection
```

## Overview

This user guide provides detailed documentation for each preprocessing step in the xpectrass pipeline. Each section covers:

- **Purpose**: Why this preprocessing step is important
- **Available Methods**: All implemented algorithms
- **Parameters**: Configurable options
- **Examples**: Code snippets and use cases
- **Best Practices**: Recommendations for FTIR plastic classification

## Preprocessing Pipeline Order

The recommended preprocessing order is:

```
1. Data Validation     → Ensure data quality
2. Region Selection    → Focus on relevant wavenumbers
3. Atmospheric Corr.   → Remove CO₂/H₂O interference
4. Baseline Correction → Remove instrumental artifacts
5. Denoising          → Reduce high-frequency noise
6. Scatter Correction  → Correct for scattering effects
7. Normalization      → Standardize intensity scales
8. Derivatives        → Enhance spectral features (optional)
9. Mean Centering     → Prepare for PCA/PLS
```

## Configuration

All preprocessing steps can be configured through the `PreprocessingConfig` class or via presets:

```python
from xpectrass.preprocessing_pipeline import PreprocessingConfig, FTIRPreprocessor

config = PreprocessingConfig(
    baseline=True,
    baseline_params={'method': 'airpls', 'lam': 1e6},
    denoise=True,
    denoise_params={'method': 'savgol', 'window_length': 15},
    normalize=True,
    normalize_params={'method': 'snv'},
    mean_center=True
)

pipe = FTIRPreprocessor(config)
```
