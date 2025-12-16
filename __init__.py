"""
xpectrass - FTIR/ToF-SIMS Spectral Analysis Suite
==================================================

A comprehensive toolkit for preprocessing and analyzing
FTIR and ToF-SIMS spectral data for classification and identification.

Quick Start
-----------
>>> from xpectrass import FTIRPreprocessor, create_preprocessor
>>> from xpectrass.utils import process_batch_files
>>>
>>> # Load data
>>> df = process_batch_files(files)
>>>
>>> # Create and apply preprocessing pipeline
>>> pipe = create_preprocessor('standard')
>>> processed = pipe.fit_transform(df)

Modules
-------
- preprocessing_pipeline : Main FTIRPreprocessor class and presets
- utils : Individual preprocessing functions
    - data_validation : Data validation utilities
    - baseline : Baseline correction (50+ methods)
    - denoise : Noise reduction (7 methods)
    - normalization : Intensity normalization (7+ methods)
    - atmospheric : CO2/H2O correction
    - derivatives : Spectral derivatives
    - scatter_correction : MSC, EMSC, SNV
    - region_selection : Wavenumber region handling
    - file_management : Data loading utilities
"""

__version__ = "0.0.1"
__author__ = "FTIR-PLASTIC Team"
__email__ = "xpectrass@kazilab.se"
__license__ = "MIT"

# Main pipeline components
from .preprocessing_pipeline import (
    FTIRPreprocessor,
    PreprocessingConfig,
    create_preprocessor,
    get_preset_config,
)

# Commonly used utilities (for convenience)
from .utils import (
    # File management
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
    
    # Derivatives
    spectral_derivative,
    first_derivative,
    second_derivative,
    
    # Region selection
    select_region,
    exclude_regions,
    FTIR_REGIONS,
    get_wavenumbers,
    get_spectra_matrix,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Main pipeline
    "FTIRPreprocessor",
    "PreprocessingConfig", 
    "create_preprocessor",
    "get_preset_config",
    
    # Utilities
    "process_batch_files",
    "import_data",
    "validate_spectra",
    "detect_outlier_spectra",
    "baseline_correction",
    "baseline_method_names",
    "denoise",
    "denoise_method_names",
    "normalize",
    "mean_center",
    "auto_scale",
    "spectral_derivative",
    "first_derivative",
    "second_derivative",
    "select_region",
    "exclude_regions",
    "FTIR_REGIONS",
    "get_wavenumbers",
    "get_spectra_matrix",
]
