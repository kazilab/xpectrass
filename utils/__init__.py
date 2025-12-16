"""
xpectrass - FTIR/ToF-SIMS Spectral Analysis Utilities
======================================================

A comprehensive toolkit for preprocessing and analyzing spectral data.
"""

# File management
from .file_management import import_data, process_batch_files

# Baseline correction
from .baseline import (
    baseline_correction, baseline_method_names,
    evaluate_all_samples, plot_corrected_spectrum,
    plot_metric_boxes, plot_metric_boxes_masked
)

# Data validation
from .data_validation import (
    validate_spectra, detect_outlier_spectra, check_wavenumber_consistency
)

# Denoising
from .denoise import (
    denoise, denoise_method_names, estimate_snr, 
    evaluate_denoising, plot_denoising_comparison
)

# Normalization
from .normalization import (
    normalize, normalize_method_names, mean_center,
    auto_scale, pareto_scale, detrend, snv_detrend,
    normalize_batch, normalize_dataframe, mean_center_dataframe
)

# Atmospheric correction
from .atmospheric import (
    atmospheric_correction, get_atmospheric_regions,
    identify_atmospheric_features
)

# Spectral derivatives
from .derivatives import (
    spectral_derivative, first_derivative, second_derivative,
    gap_derivative, derivative_batch, plot_derivatives
)

# Scatter correction
from .scatter_correction import (
    scatter_correction, scatter_method_names,
    msc_single, evaluate_scatter_correction
)

# Region selection
from .region_selection import (
    select_region, exclude_regions, exclude_atmospheric,
    get_region_names, get_region_range, analyze_regions,
    get_wavenumbers, get_spectra_matrix, FTIR_REGIONS,
    select_region_np, select_regions_np
)

# Plotting
from .plotting import plot_ftir_spectra

__all__ = [
    # File management
    'import_data', 'process_batch_files',
    # Baseline
    'baseline_correction', 'baseline_method_names', 'evaluate_all_samples',
    'plot_corrected_spectrum', 'plot_metric_boxes', 'plot_metric_boxes_masked',
    # Validation
    'validate_spectra', 'detect_outlier_spectra', 'check_wavenumber_consistency',
    # Denoising
    'denoise', 'denoise_method_names', 'estimate_snr', 
    'evaluate_denoising', 'plot_denoising_comparison',
    # Normalization
    'normalize', 'normalize_method_names', 'mean_center',
    'auto_scale', 'pareto_scale', 'detrend', 'snv_detrend',
    'normalize_batch', 'normalize_dataframe', 'mean_center_dataframe',
    # Atmospheric
    'atmospheric_correction', 'get_atmospheric_regions', 'identify_atmospheric_features',
    # Derivatives
    'spectral_derivative', 'first_derivative', 'second_derivative',
    'gap_derivative', 'derivative_batch', 'plot_derivatives',
    # Scatter
    'scatter_correction', 'scatter_method_names', 'msc_single', 'evaluate_scatter_correction',
    # Region selection
    'select_region', 'exclude_regions', 'exclude_atmospheric',
    'get_region_names', 'get_region_range', 'analyze_regions',
    'get_wavenumbers', 'get_spectra_matrix', 'FTIR_REGIONS',
    # Plotting
    'plot_ftir_spectra',
]
