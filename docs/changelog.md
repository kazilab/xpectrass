# Changelog

All notable changes to xpectrass will be documented in this file.

## [1.0.0] - 2024-12-06

### Added

#### Preprocessing Pipeline
- `FTIRPreprocessor` class with sklearn-style fit/transform API
- `PreprocessingConfig` dataclass for configuration
- Preset configurations: minimal, standard, classification, pca, raw
- `create_preprocessor()` convenience function

#### Data Validation
- `validate_spectra()` for comprehensive data validation
- `detect_outlier_spectra()` with zscore, IQR, and MAD methods
- `check_wavenumber_consistency()` for file consistency checks

#### Denoising
- 7 denoising methods: savgol, wavelet, moving_average, gaussian, median, whittaker, lowpass
- `evaluate_denoising()` for method comparison
- `plot_denoising_comparison()` visualization

#### Normalization
- 7 normalization methods: snv, vector, minmax, area, peak, range, max
- `mean_center()` for PCA preparation
- `auto_scale()` and `pareto_scale()` for variable scaling
- `snv_detrend()` combined SNV + detrending
- DataFrame integration: `normalize_dataframe()`, `mean_center_dataframe()`

#### Atmospheric Correction
- 5 correction methods: interpolate, spline, reference, zero, exclude
- CO₂ and H₂O region handling
- `identify_atmospheric_features()` detection

#### Spectral Derivatives
- Savitzky-Golay based derivatives
- `first_derivative()` and `second_derivative()` convenience functions
- `gap_derivative()` Norris-Williams algorithm
- `derivative_batch()` for matrix processing

#### Scatter Correction
- MSC (Multiplicative Scatter Correction)
- EMSC (Extended MSC)
- SNV and SNV+Detrend
- `evaluate_scatter_correction()` metrics

#### Region Selection
- Predefined FTIR regions for plastics
- `select_region()` and `exclude_regions()` functions
- `exclude_atmospheric()` convenience function
- `analyze_regions()` statistics
- NumPy array functions: `select_region_np()`, `select_regions_np()`

### Enhanced

#### Baseline Correction
- Updated `__init__.py` exports
- Integration with new preprocessing pipeline

#### File Management
- Better error handling
- Progress bar support

### Documentation
- ReadTheDocs-style documentation
- Comprehensive user guide
- Full API reference
- 7 practical examples
