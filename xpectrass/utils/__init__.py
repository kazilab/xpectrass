"""
xpectrass - FTIR/ToF-SIMS Spectral Analysis Utilities
======================================================

A comprehensive toolkit for preprocessing and analyzing spectral data.
"""

# File management
from .file_management import import_data, process_batch_files
from .interpolate import combine_datasets

# Plotting
from .plotting import plot_ftir_spectra, compare_ftir_spectra
from .plottingx import (
    plot_mean_spectra_by_class,
    plot_overlay_mean_spectra,
    plot_coefficient_of_variation,
    plot_spectral_heatmap
)
from .plotting_stats import (
    perform_anova_analysis,
    plot_correlation_matrix
)
from .plotting_dim import (
    perform_pca_analysis,
    perform_tsne_analysis,
    perform_umap_analysis,
    perform_plsda_analysis,
    perform_oplsda_analysis
)
from .plotting_clus import (
    perform_kmeans_clustering,
    perform_hierarchical_clustering
)

# Conversion between Transmittance and Absorbance
from .trans_abs import convert_spectra

# Atmospheric Correction
from .atmospheric import (
    atmospheric_correction,
    identify_atmospheric_features,
    exclude_and_interpolate_regions
)

# Baseline correction
from .baseline import (
    evaluate_baseline_correction_methods,
    baseline_correction,
    baseline_method_names,
    apply_baseline_correction,
    plot_baseline_correction_metric_boxes,
    plot_baseline_correction_metric_boxes,
    plot_baseline_correction_metric_boxes_masked,
    find_best_baseline_method
)

# Denoising
from .denoise import (
    denoise, 
    denoise_method_names,
    apply_denoising, 
    estimate_snr, 
    evaluate_denoising_methods,
    plot_denoising_evaluation,
    plot_denoising_evaluation_summary,
    plot_denoising_comparison,
    find_best_denoising_method,
)

# Normalization
from .normalization import (
    normalize,
    normalize_method_names,
    mean_center,
    auto_scale,
    pareto_scale,
    detrend,
    snv_detrend,
    normalize_df
)

# Normalization evaluation
from .normalization_eval import (
    evaluate_norm_methods,
    evaluate_one_method,
    FTIRNormalizer,
    spectral_angle,
    within_group_mean_sam,
    zscore_robust
)

# Spectral derivatives
from .derivatives import (
    spectral_derivative, 
    first_derivative, 
    second_derivative,
    gap_derivative,
    derivative_with_smoothing,
    derivative_batch, 
    plot_derivatives
)
# Data validation
from .data_validation import (
    validate_spectra, 
    detect_outlier_spectra, 
    check_wavenumber_consistency
)
# Region selection
from .region_selection import (
    select_region, 
    exclude_regions, 
    exclude_atmospheric,
    get_region_names, 
    get_region_range, 
    analyze_regions,
    get_wavenumbers, 
    get_spectra_matrix, 
    FTIR_REGIONS,
    select_region_np, 
    select_regions_np,
)
# Scatter correction
from .scatter_correction import (
    scatter_correction,
    scatter_method_names,
    apply_scatter_correction,
    msc_single
)

from .warnings import log_and_suppress_warnings

# Machine Learning
from .ml import (
    prepare_data,
    get_all_models,
    evaluate_model,
    evaluate_all_models,
    plot_confusion_matrix,
    calculate_multiclass_metrics,
    print_multiclass_metrics,
    plot_model_comparison,
    categorize_model,
    plot_family_comparison,
    plot_efficiency_analysis,
    plot_overfitting_analysis,
    tune_top_models,
    compare_datasets_performance,
    generate_recommendations,
    explain_model_shap,
    plot_shap_decision
)

__all__ = [
    # File management
    'import_data', 
    'process_batch_files',
    'combine_datasets',
    # Plotting
    'plot_ftir_spectra',
    'compare_ftir_spectra',
    'plot_mean_spectra_by_class',
    'plot_overlay_mean_spectra',
    'plot_coefficient_of_variation',
    'plot_spectral_heatmap',
    'perform_anova_analysis',
    'plot_correlation_matrix',
    'perform_pca_analysis',
    'perform_tsne_analysis',
    'perform_umap_analysis',
    'perform_plsda_analysis',
    'perform_oplsda_analysis',
    'perform_kmeans_clustering',
    'perform_hierarchical_clustering',
    # Conversion
    'convert_spectra',
    # Atmospheric
    'atmospheric_correction',
    'identify_atmospheric_features',
    'exclude_and_interpolate_regions',
    # Baseline
    'evaluate_baseline_correction_methods',
    'baseline_correction', 
    'baseline_method_names',
    'apply_baseline_correction',
    'plot_baseline_correction_metric_boxes',
    'plot_baseline_correction_metric_boxes', 
    'plot_baseline_correction_metric_boxes_masked',
    'find_best_baseline_method',
    # Denoising
    'denoise', 
    'denoise_method_names', 
    'apply_denoising',
    'estimate_snr', 
    'evaluate_denoising_methods',
    'plot_denoising_evaluation',
    'plot_denoising_evaluation_summary',
    'plot_denoising_comparison',
    'find_best_denoising_method',
    # Normalization
    'normalize',
    'normalize_method_names',
    'mean_center',
    'auto_scale',
    'pareto_scale',
    'detrend',
    'snv_detrend',
    'normalize_df',
    # Normalization evaluation
    'evaluate_norm_methods',
    'evaluate_one_method',
    'FTIRNormalizer',
    'spectral_angle',
    'within_group_mean_sam',
    'zscore_robust',
    # Derivatives
    'spectral_derivative', 
    'first_derivative', 
    'second_derivative',
    'gap_derivative', 
    'derivative_with_smoothing',
    'derivative_batch', 
    'plot_derivatives',
    # Validation
    'validate_spectra', 
    'detect_outlier_spectra', 
    'check_wavenumber_consistency',
    # Region selection
    'select_region', 
    'exclude_regions', 
    'exclude_atmospheric',
    'get_region_names', 
    'get_region_range', 
    'analyze_regions',
    'get_wavenumbers', 
    'get_spectra_matrix', 
    'FTIR_REGIONS',
    'select_region_np', 
    'select_regions_np',
    # Scatter
    'scatter_correction',
    'scatter_method_names',
    'apply_scatter_correction',
    'msc_single',
    # Warnings
    'log_and_suppress_warnings',
    # Machine Learning
    'prepare_data',
    'get_all_models',
    'evaluate_model',
    'evaluate_all_models',
    'plot_confusion_matrix',
    'calculate_multiclass_metrics',
    'print_multiclass_metrics',
    'plot_model_comparison',
    'categorize_model',
    'plot_family_comparison',
    'plot_efficiency_analysis',
    'plot_overfitting_analysis',
    'tune_top_models',
    'compare_datasets_performance',
    'generate_recommendations',
    'explain_model_shap',
    'plot_shap_decision',
    # interpolate
    'combine_datasets'
]
