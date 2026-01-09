# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-01-09

### Added
- Complete `FTIRdataprocessing` class with evaluation-first preprocessing pipeline
- Complete `FTIRdataanalysis` class for statistical analysis and machine learning
- 6 bundled FTIR plastic datasets from published studies (2018-2024):
  - Jung et al. 2018
  - Kedzierski et al. 2019 (2 variants)
  - Frond et al. 2021
  - Villegas-Camacho et al. 2024 (C4 and C8 fractions)
- 50+ baseline correction algorithms via pybaselines
- 7 denoising methods (Savitzky-Golay, wavelet, median, Gaussian, etc.)
- 7+ normalization methods (SNV, vector, min-max, area, peak, PQN, entropy)
- 20+ machine learning classification models
- Model explainability with SHAP values
- Dimensionality reduction: PCA, t-SNE, UMAP, PLS-DA, OPLS-DA
- Comprehensive evaluation methods for preprocessing steps
- Atmospheric correction for CO₂ and H₂O interference
- Spectral derivatives (1st, 2nd, gap derivatives)
- Complete documentation with user guides and API reference
- Interactive Jupyter notebooks for method selection and analysis

### Changed
- Updated author to "Data Analysis Team @KaziLab.se"
- Updated copyright to "2026 @KaziLab.se"
- Updated contact email to xpectrass@kazilab.se
- Reorganized package structure for better modularity
- Enhanced visualization capabilities across all modules

### Fixed
- Improved error handling and validation
- Enhanced data type support (Pandas and Polars)
- Better memory management for large datasets

## [0.0.1] - 2024-12-01

### Added
- Initial release
- Basic preprocessing utilities
- Baseline correction methods
- Normalization functions
- Data validation tools
