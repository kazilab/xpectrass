# Changelog

For the complete version history, see the [main CHANGELOG.md](https://github.com/kazilab/xpectrass/blob/main/CHANGELOG.md) on GitHub.

## Latest Release: v0.0.2 (2026-01-09)

### Major Features

**Preprocessing Pipeline**
- Complete `FTIRdataprocessing` class with evaluation-first approach
- 50+ baseline correction algorithms via pybaselines
- 7 denoising methods with comprehensive evaluation
- 17+ normalization methods with classification-based selection
- Atmospheric correction for CO₂ and H₂O interference

**Machine Learning & Analysis**
- Complete `FTIRdataanalysis` class for classification and analysis
- 20+ machine learning models (XGBoost, LightGBM, CatBoost, SVM, etc.)
- Model explainability with SHAP values
- Hyperparameter tuning for top models
- Dimensionality reduction: PCA, t-SNE, UMAP, PLS-DA, OPLS-DA

**Bundled Datasets**
- 6 FTIR plastic datasets from published studies (2018-2024)
- Jung et al. 2018 (~500 spectra)
- Kedzierski et al. 2019 (2 variants, ~4,000 spectra)
- Frond et al. 2021 (~400 spectra)
- Villegas-Camacho et al. 2024 (C4 and C8, ~6,000 spectra)

**Documentation & Examples**
- Comprehensive user guide with step-by-step tutorials
- Interactive Jupyter notebooks for method selection
- Complete API reference
- Real-world examples and use cases

### Improvements

- Enhanced visualization capabilities across all modules
- Better error handling and validation
- Support for both Pandas and Polars DataFrames
- Improved memory management for large datasets
- Professional packaging for PyPI distribution

---

For older versions and detailed changes, see [CHANGELOG.md](https://github.com/kazilab/xpectrass/blob/main/CHANGELOG.md).
