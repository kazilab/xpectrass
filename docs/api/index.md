# API Reference

```{toctree}
:maxdepth: 2

preprocessing_pipeline
utils
```

## Module Overview

```
xpectrass/
├── preprocessing_pipeline    # Main FTIRPreprocessor class
└── utils/
    ├── data_validation      # validate_spectra, detect_outlier_spectra
    ├── baseline             # baseline_correction (50+ methods)
    ├── denoise              # denoise (7 methods)
    ├── normalization        # normalize, mean_center, auto_scale
    ├── atmospheric          # atmospheric_correction
    ├── derivatives          # spectral_derivative
    ├── scatter_correction   # scatter_correction (MSC, EMSC, SNV)
    ├── region_selection     # select_region, exclude_regions
    ├── file_management      # process_batch_files, import_data
    └── plotting             # Visualization utilities
```

## Quick Import Guide

```python
# Main pipeline
from xpectrass.preprocessing_pipeline import (
    FTIRPreprocessor,
    PreprocessingConfig,
    create_preprocessor,
    get_preset_config
)

# Individual utilities
from xpectrass.utils import (
    # Data loading
    process_batch_files,
    import_data,
    
    # Validation
    validate_spectra,
    detect_outlier_spectra,
    
    # Baseline
    baseline_correction,
    baseline_method_names,
    
    # Denoising
    denoise,
    denoise_method_names,
    
    # Normalization
    normalize,
    mean_center,
    auto_scale,
    
    # Atmospheric
    atmospheric_correction,
    
    # Derivatives
    spectral_derivative,
    first_derivative,
    second_derivative,
    
    # Scatter
    scatter_correction,
    
    # Region selection
    select_region,
    exclude_regions,
    FTIR_REGIONS,
    get_wavenumbers,
    get_spectra_matrix
)
```
