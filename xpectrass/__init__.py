"""
xpectrass - FTIR/ToF-SIMS Spectral Analysis Suite
==================================================

A comprehensive toolkit for preprocessing and analyzing
FTIR and ToF-SIMS spectral data for classification and identification.

Quick Start
-----------
>>> from xpectrass import FTIRdataprocessing, FTIRdataanalysis
>>> from xpectrass.utils import process_batch_files
>>>
>>> # Load data
>>> df = process_batch_files('files')
>>>
>>> # Create and apply preprocessing pipeline
>>> pipe = FTIRdataprocessing(data='df')
>>> processed = pipe.run()

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
- data : Bundled datasets (6 FTIR datasets from various studies)
"""

__version__ = "0.0.3"
__author__ = "Data Analysis Team @KaziLab.se"
__email__ = "xpectrass@kazilab.se"
__license__ = "MIT"

# Main pipeline components
from .main import FTIRdataprocessing, FTIRdataanalysis

# Data loading
from .data import (
    load_jung_2018,
    load_kedzierski_2019,
    load_kedzierski_2019_u,
    load_frond_2021,
    load_villegas_camacho_2024_c4,
    load_villegas_camacho_2024_c8,
    load_all_datasets,
    load_datasets,
    get_data_info,
)

# Commonly used utilities
from .utils import (
    # File management
    process_batch_files,
    import_data,
    combine_datasets,

    # Plotting
    plot_ftir_spectra,
    compare_ftir_spectra,
    plot_mean_spectra_by_class,
    plot_overlay_mean_spectra,
    plot_coefficient_of_variation,
    plot_spectral_heatmap,
    perform_anova_analysis,
    plot_correlation_matrix,
    perform_pca_analysis,
    perform_tsne_analysis,
    perform_umap_analysis,
    perform_plsda_analysis,
    perform_oplsda_analysis,
    perform_kmeans_clustering,
    perform_hierarchical_clustering,

    # Transmittance/Absorbance conversion
    convert_spectra,

    # Atmospheric correction
    atmospheric_correction,
    identify_atmospheric_features,
    exclude_and_interpolate_regions,

    # Baseline correction
    evaluate_baseline_correction_methods,
    baseline_correction,
    baseline_method_names,
    apply_baseline_correction,
    plot_baseline_correction_metric_boxes,
    plot_baseline_correction_metric_boxes,
    plot_baseline_correction_metric_boxes_masked,
    find_best_baseline_method,

    # Denoising
    denoise, 
    denoise_method_names, 
    apply_denoising,
    estimate_snr, 
    evaluate_denoising_methods,
    plot_denoising_evaluation,
    plot_denoising_evaluation_summary,
    plot_denoising_comparison,
    find_best_denoising_method,
    
    # Normalization
    normalize, 
    normalize_method_names, 
    mean_center,
    auto_scale, 
    pareto_scale, 
    detrend, 
    snv_detrend,
    normalize_df,

    # Normalization evaluation
    evaluate_norm_methods,
    evaluate_one_method,
    FTIRNormalizer,
    spectral_angle,
    within_group_mean_sam,
    zscore_robust,

    # Spectral derivatives
    spectral_derivative, 
    first_derivative, 
    second_derivative,
    gap_derivative,
    derivative_with_smoothing,
    derivative_batch, 
    plot_derivatives,

    # Region selection
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
    
    # Data validation
    validate_spectra, 
    detect_outlier_spectra, 
    check_wavenumber_consistency,
    
    # Scatter correction
    scatter_correction,
    scatter_method_names,
    apply_scatter_correction,
    msc_single,

    # Warnings
    log_and_suppress_warnings,

    # Machine Learning
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
    # Version info
    "__version__",
    "__author__",

    # Main classes
    'FTIRdataprocessing',
    'FTIRdataanalysis',

    # Data loading
    'load_jung_2018',
    'load_kedzierski_2019',
    'load_kedzierski_2019_u',
    'load_frond_2021',
    'load_villegas_camacho_2024_c4',
    'load_villegas_camacho_2024_c8',
    'load_all_datasets',
    'load_datasets',
    'get_data_info',

    # File management
    'process_batch_files',
    'import_data',
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

    # Transmittance/Absorbance conversion
    'convert_spectra',

    # Atmospheric correction
    'atmospheric_correction',
    'identify_atmospheric_features',
    'exclude_and_interpolate_regions',

    # Baseline correction
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

    # Spectral derivatives
    'spectral_derivative', 
    'first_derivative', 
    'second_derivative',
    'gap_derivative',
    'derivative_with_smoothing',
    'derivative_batch', 
    'plot_derivatives',

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

    # Data validation
    'validate_spectra', 
    'detect_outlier_spectra', 
    'check_wavenumber_consistency',
    
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
    'plot_shap_decision'
]
