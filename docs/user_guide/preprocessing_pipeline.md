# Preprocessing Pipeline

The `FTIRPreprocessor` class provides a unified interface for applying multiple preprocessing steps to FTIR spectral data.

## Overview

The preprocessing pipeline follows an sklearn-style API with `fit()`, `transform()`, and `fit_transform()` methods:

```python
from xpectrass.preprocessing_pipeline import FTIRPreprocessor, create_preprocessor

# Using a preset
pipe = create_preprocessor('standard')

# Fit and transform
processed = pipe.fit_transform(df)

# Or separately
pipe.fit(df_train)
processed_test = pipe.transform(df_test)
```

## Configuration

### PreprocessingConfig

All preprocessing options are controlled via the `PreprocessingConfig` dataclass:

```python
from xpectrass.preprocessing_pipeline import PreprocessingConfig

config = PreprocessingConfig(
    # Data validation
    validate=True,
    validation_params={'expected_samples_per_class': 500},
    
    # Baseline correction
    baseline=True,
    baseline_params={'method': 'airpls', 'lam': 1e6},
    
    # Denoising
    denoise=True,
    denoise_params={'method': 'savgol', 'window_length': 15, 'polyorder': 3},
    
    # Normalization
    normalize=True,
    normalize_params={'method': 'snv'},
    
    # Atmospheric correction
    atmospheric=False,  # Often not needed for ATR
    atmospheric_params={'method': 'interpolate'},
    
    # Spectral derivatives
    derivatives=False,
    derivative_params={'order': 1, 'window_length': 15},
    
    # Scatter correction
    scatter=False,
    scatter_params={'method': 'msc'},
    
    # Region selection
    region_selection=False,
    regions=[(400, 4000)],
    exclude_atmospheric_regions=False,
    
    # Mean centering
    mean_center=True,
)
```

### Available Presets

Use `create_preprocessor()` with preset names for quick setup:

| Preset | Configuration |
|--------|--------------|
| `minimal` | Baseline (airpls) + SNV normalization |
| `standard` | Baseline + Denoise (savgol) + SNV + Mean center |
| `classification` | Standard + Region selection (fingerprint + CH stretch) |
| `pca` | Standard + 1st derivative |
| `raw` | Validation only, no preprocessing |

```python
# Get preset configuration
from xpectrass.preprocessing_pipeline import get_preset_config

config = get_preset_config('classification')
print(config.regions)  # [(400, 1800), (2700, 3100)]
```

### Custom Configuration with Overrides

```python
# Start from preset and override specific options
pipe = create_preprocessor('standard', 
    baseline_params={'method': 'asls', 'lam': 1e7},
    derivatives=True
)
```

## Processing Order

The default processing order is:

1. `validate` - Data validation
2. `region_selection` - Select/exclude wavenumber regions
3. `atmospheric` - Atmospheric correction
4. `baseline` - Baseline correction
5. `denoise` - Noise reduction
6. `scatter` - Scatter correction
7. `normalize` - Intensity normalization
8. `derivatives` - Spectral derivatives
9. `mean_center` - Mean centering

Customize the order if needed:

```python
config = PreprocessingConfig(
    order=['validate', 'baseline', 'normalize', 'derivatives', 'mean_center']
)
```

## Fitted Parameters

After fitting, the pipeline stores learned parameters:

```python
pipe.fit(df)

# Access fitted parameters
print(pipe.fitted_params_['mean'])  # Mean spectrum for centering
print(pipe.fitted_params_['scatter_reference'])  # Reference for MSC
print(pipe.validation_report_)  # Validation results
```

## Complete Example

```python
from xpectrass.utils import process_batch_files, validate_spectra
from xpectrass.preprocessing_pipeline import FTIRPreprocessor, PreprocessingConfig
import glob

# Load data
files = glob.glob('FTIR-PLASTIC-c4/*/*.csv')
df = process_batch_files(files)

# Configure pipeline
config = PreprocessingConfig(
    validate=True,
    baseline=True,
    baseline_params={'method': 'airpls'},
    denoise=True,
    denoise_params={'method': 'savgol', 'window_length': 11},
    normalize=True,
    normalize_params={'method': 'vector'},
    region_selection=True,
    regions=[(400, 1800), (2700, 3100)],
    mean_center=True
)

# Create and apply pipeline
pipe = FTIRPreprocessor(config)
processed = pipe.fit_transform(df)

print(f"Original shape: {df.shape}")
print(f"Processed shape: {processed.shape}")
```
