# xpectrass

**Xpectrass - From preprocessing to Machine Learning for Spectral Data**

A comprehensive Python toolkit for FTIR spectral data preprocessing, analysis, and machine learning classification.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/xpectrass/badge/?version=latest)](https://xpectrass.readthedocs.io/)
[![Version](https://img.shields.io/badge/version-0.0.4-green)](https://github.com/kazilab/xpectrass)

## Overview

Xpectrass provides an end-to-end pipeline for FTIR spectra classification, from raw spectral data to machine learning predictions with model explainability. The library is built around two main classes:

- **`FTIRdataprocessing`**: Comprehensive preprocessing pipeline with evaluation-first approach
- **`FTIRdataanalysis`**: Statistical analysis, dimensionality reduction, and machine learning

## Key Features

### 🔬 Preprocessing Pipeline
- **Evaluation-First Philosophy**: Automatically find the best preprocessing parameters for your data
- **9 Preprocessing Steps** with multiple methods for each step
- **50+ Baseline Correction** algorithms via pybaselines (airpls, asls, arpls, etc.)
- **7 Denoising Methods** (Savitzky-Golay, wavelet, median, Gaussian, etc.)
- **17+ Normalization Methods** (SNV, vector, min-max, area, peak, PQN, entropy)
- **Atmospheric Correction** (CO₂/H₂O removal and interpolation)
- **Spectral Derivatives** (1st, 2nd, gap derivatives with smoothing)
- **Real-time Visualization** at every preprocessing step

### 📊 Analysis & Visualization
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, PLS-DA, OPLS-DA
- **Statistical Analysis**: ANOVA, correlation analysis, coefficient of variation
- **Clustering**: K-means, hierarchical clustering with dendrograms
- **Interactive Plots**: Mean spectra, heatmaps, overlay plots, and more

### 🤖 Machine Learning
- **20+ Classification Models**: Random Forest, XGBoost, LightGBM, SVM, Neural Networks, etc.
- **Automated Evaluation**: Cross-validation, confusion matrices, performance metrics
- **Hyperparameter Tuning**: Automatic optimization of top-performing models
- **Model Explainability**: SHAP analysis for feature importance
- **Comparison Visualizations**: Family comparison, efficiency analysis, overfitting detection

### 📦 Bundled Datasets
- **6 Pre-loaded FTIR Plastic Datasets** from published studies (2018-2024)
- Ready-to-use examples for testing and learning
- Datasets: Jung 2018, Kedzierski 2019, Frond 2021, Villegas-Camacho 2024

## Installation

### From PyPI (when published)

```bash
pip install xpectrass
```

### From Source

```bash
git clone https://github.com/kazilab/xpectrass.git
cd xpectrass
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Option 1: Use Bundled Datasets

Xpectrass comes with 6 pre-loaded FTIR plastic datasets:

```python
from xpectrass import FTIRdataprocessing
from xpectrass.data import load_jung_2018, get_data_info

# See available datasets
print(get_data_info())

# Load a dataset
df = load_jung_2018()

# Remove duplicate spectra
df = df.drop_duplicates(subset=['sample_id'])
print(f"Loaded {len(df)} spectra")

# Start preprocessing
ftir = FTIRdataprocessing(df, label_column="type")
```

### Option 2: Load Your Own Data

```python
from xpectrass import FTIRdataprocessing
from xpectrass.utils import process_batch_files
import glob
import pandas as pd

# Load single CSV file
df = pd.read_csv("ftir_data.csv", index_col=0)

# Or load multiple files
files = glob.glob('data/plastics/*.csv')
df = process_batch_files(files)

print(f"Loaded {len(df)} spectra with {len(df.columns)-1} wavenumbers")
```

**Data Format:**
- Rows: Individual spectra
- Columns: One label column + wavenumber columns (e.g., "400.0", "401.0", ...)
- Index: Sample names or IDs

Example CSV structure:
```
sample_id,type,400.0,401.0,402.0,...,4000.0
HDPE_001,HDPE,0.123,0.125,0.128,...,0.045
PP_001,PP,0.098,0.102,0.105,...,0.038
```

## Basic Preprocessing Workflow

### Step-by-Step Approach

```python
from xpectrass import FTIRdataprocessing

# Initialize
ftir = FTIRdataprocessing(
    df,
    label_column="type",  # Name of your label column
    wn_min=400,           # Minimum wavenumber
    wn_max=4000           # Maximum wavenumber
)

# Step 1: Convert to absorbance (if data is in transmittance)
ftir.convert(mode="to_absorbance", plot=True)

# Step 2: Evaluate and apply denoising
ftir.find_denoising_method(n_samples=50, plot=True)
ftir.denoise_spect(method="savgol", window_length=15)

# Step 3: Evaluate and apply baseline correction
ftir.find_baseline_method(n_samples=50, plot=True)
ftir.plot_rfzn_nar_snr(metric_name="SNR")  # Visualize evaluation metrics
ftir.correct_baseline(method="asls", plot=False)

# Step 4: Remove atmospheric interference (CO₂, H₂O)
ftir.exclude_interpolate(method="spline", plot=True)

# Step 5: Evaluate and apply normalization
ftir.find_normalization_method()
ftir.normalize(method="snv")

# Step 6: Compare all processing stages
ftir.plot_multiple_spec()

# Get processed data
processed_df = ftir.df_norm
```

### Quick Run with Defaults

For rapid prototyping:

```python
ftir = FTIRdataprocessing(df, label_column="type")

# Run entire pipeline with default settings
ftir.run()

# Get final processed data
processed_df = ftir.df_norm
```

## Basic Analysis Workflow

After preprocessing, use `FTIRdataanalysis` for visualization and machine learning:

```python
from xpectrass import FTIRdataanalysis

LABEL_COLUMN = "type"
MIN_SAMPLE_NUMBER_FOR_GROUP = 10
# remove small groups from processed data
processed_df = processed_df.dropna(subset=[LABEL_COLUMN])
group_counts=processed_df[LABEL_COLUMN].value_counts()
valid_groups = group_counts[group_counts >= MIN_SAMPLE_NUMBER_FOR_GROUP].index
processed_df = processed_df[processed_df[LABEL_COLUMN].isin(valid_groups)]

# Initialize analysis
analysis = FTIRdataanalysis(processed_df, label_column=LABEL_COLUMN)

# Visualize mean spectra by class
analysis.plot_mean_spectra()

# Plot spectral heatmap
analysis.plot_heatmap()

# Dimensionality reduction
analysis.plot_pca()
analysis.plot_tsne()
analysis.plot_umap()

# Statistical analysis
analysis.perform_anova()
analysis.plot_correlation()
```

## Machine Learning

```python
# Prepare data for ML
analysis.ml_prepare_data()

# Run all classification models
results = analysis.run_all_models()
print(results.sort_values('test_f1', ascending=False))

# Tune top performing models
tuned_results = analysis.model_parameter_tuning()

# Explain model predictions with SHAP
analysis.explain_by_shap()
analysis.local_shap_plot()
```

## Main Features

### Preprocessing Methods

| Category | Methods Available |
|----------|-------------------|
| **Baseline Correction** | 50+ methods: airpls, asls, arpls, poly, mor, rubberband, snip, etc. |
| **Denoising** | Savitzky-Golay, wavelet, median, Gaussian, bilateral, Wiener, FFT |
| **Normalization** | SNV, vector, min-max, area, peak, PQN, entropy-weighted |
| **Atmospheric Correction** | CO₂/H₂O region exclusion and spline/linear interpolation |
| **Scatter Correction** | MSC, EMSC, SNV+detrend |
| **Spectral Derivatives** | 1st, 2nd, gap derivatives with Savitzky-Golay smoothing |
| **Data Validation** | Completeness checks, range validation, outlier detection |
| **Region Selection** | 13 predefined FTIR regions for plastic analysis |

### Analysis Capabilities

| Category | Methods |
|----------|---------|
| **Visualization** | Mean spectra, overlay plots, heatmaps, coefficient of variation |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP, PLS-DA, OPLS-DA with loadings plots |
| **Clustering** | K-means (with elbow plot), hierarchical (with dendrogram) |
| **Statistics** | ANOVA (wavenumber-wise), correlation matrices |

### Machine Learning Models

**20+ Classification Algorithms:**
- Ensemble: Random Forest, Extra Trees, AdaBoost, Gradient Boosting
- Boosting: XGBoost, LightGBM (multiple configurations)
- SVM: Linear, RBF, Polynomial kernels
- Linear: Logistic Regression, Ridge, SGD
- Neighbors: K-Nearest Neighbors (multiple K values)
- Neural Networks: Multi-Layer Perceptron (multiple architectures)
- Naive Bayes: Gaussian, Multinomial
- Discriminant Analysis: LDA, QDA

## Bundled Datasets

Load pre-processed FTIR datasets for immediate use:

```python
from xpectrass.data import (
    load_jung_2018,
    load_kedzierski_2019,
    load_frond_2021,
    load_villegas_camacho_2024_c4,
    load_all_datasets,
    get_data_info
)

# Load a specific dataset
df = load_jung_2018()

# View all available datasets
info = get_data_info()
print(info)

# Load all datasets
all_data = load_all_datasets()
```

**Available Datasets:**
- Jung et al. 2018 (~500 spectra, multiple polymer types)
- Kedzierski et al. 2019 (2 variants, ~300 spectra each)
- Frond et al. 2021 (~400 spectra)
- Villegas-Camacho et al. 2024 (C4 and C8 fractions, ~600 each)

## Loading Your Own Data

```python
from xpectrass.utils import process_batch_files
import glob

# Load multiple CSV files
files = glob.glob('data/plastics/*.csv')
df = process_batch_files(files)

# Load single file
import pandas as pd
df = pd.read_csv("my_ftir_data.csv", index_col=0)
```

**Expected Data Format:**
- Rows: Individual spectra
- Columns: One label column + wavenumber columns (e.g., "400.0", "401.0", ...)
- Index: Sample identifiers

## Documentation

Full documentation is available at [xpectrass.readthedocs.io](https://xpectrass.readthedocs.io/).

**User Guide Sections:**
- [Getting Started](https://xpectrass.readthedocs.io/en/latest/getting_started.html)
- [Preprocessing Pipeline](https://xpectrass.readthedocs.io/en/latest/user_guide/preprocessing_pipeline.html)
- [Data Loading](https://xpectrass.readthedocs.io/en/latest/user_guide/data_loading.html)
- [Analysis & Visualization](https://xpectrass.readthedocs.io/en/latest/user_guide/analysis.html)
- [Machine Learning](https://xpectrass.readthedocs.io/en/latest/user_guide/machine_learning.html)
- [API Reference](https://xpectrass.readthedocs.io/en/latest/api/index.html)

### Building Documentation Locally

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

## Requirements

### Core Dependencies
- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Pandas ≥ 1.3.0
- Polars ≥ 0.15.0

### Signal Processing
- PyBaselines ≥ 1.0.0
- PyWavelets ≥ 1.1.0

### Visualization
- Matplotlib ≥ 3.4.0
- Seaborn ≥ 0.11.0

### Machine Learning
- scikit-learn ≥ 1.0.0
- XGBoost ≥ 1.5.0
- LightGBM ≥ 3.3.0
- UMAP-learn ≥ 0.5.0
- SHAP ≥ 0.41.0

### Utilities
- tqdm ≥ 4.60.0
- joblib ≥ 1.0.0

## Project Structure

```
xpectrass/
├── __init__.py           # Main package exports
├── main.py               # FTIRdataprocessing & FTIRdataanalysis classes
├── data/                 # Bundled FTIR datasets
│   └── __init__.py
└── utils/                # Preprocessing & analysis utilities
    ├── baseline.py       # 50+ baseline correction methods
    ├── denoise.py        # 7 denoising methods
    ├── normalization.py  # 7+ normalization methods
    ├── atmospheric.py    # CO₂/H₂O correction
    ├── derivatives.py    # Spectral derivatives
    ├── scatter_correction.py  # MSC, EMSC, SNV
    ├── region_selection.py    # FTIR region handling
    ├── data_validation.py     # Data quality checks
    ├── ml.py                  # Machine learning models
    ├── plotting*.py           # Visualization functions
    └── ...
```

## Philosophy

### Evaluation-First Approach

Xpectrass uses an **evaluation-first philosophy**: instead of guessing preprocessing parameters, the library provides built-in evaluation methods to find the optimal settings for your specific data.

```python
# Evaluate all baseline methods
ftir.find_baseline_method(n_samples=50, plot=True)
ftir.plot_rfzn_nar_snr()  # Visualize metrics

# Apply the best method
ftir.correct_baseline(method="asls")
```

### State Management

The `FTIRdataprocessing` class maintains state through the entire pipeline, storing intermediate results for easy access and comparison:

```python
ftir.df              # Original data
ftir.converted_df    # After conversion
ftir.df_atm         # After atmospheric correction
ftir.df_corr        # After baseline correction
ftir.df_denoised    # After denoising
ftir.df_norm        # After normalization
ftir.df_deriv       # After derivatives
```

## Use Cases

- **Plastic Classification**: Identify polymer types from FTIR spectra
- **Quality Control**: Detect contamination or degradation in materials
- **Environmental Analysis**: Classify microplastics in environmental samples
- **Material Science**: Characterize polymer blends and composites
- **Method Development**: Compare preprocessing and classification strategies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{xpectrass,
  author = {Data Analysis Team @KaziLab.se},
  title = {Xpectrass - From preprocessing to Machine Learning for Spectral Data},
  year = {2026},
  url = {https://github.com/kazilab/xpectrass}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

### Development Setup

```bash
git clone https://github.com/kazilab/xpectrass.git
cd xpectrass
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Contact

- **Email**: xpectrass@kazilab.se
- **GitHub**: [github.com/kazilab/xpectrass](https://github.com/kazilab/xpectrass)
- **Documentation**: [xpectrass.readthedocs.io](https://xpectrass.readthedocs.io/)
- **Issues**: [github.com/kazilab/xpectrass/issues](https://github.com/kazilab/xpectrass/issues)

## Acknowledgments

Built with ❤️ by the Data Analysis Team @KaziLab.se

## Version History

### v0.0.4 (Current)
- Fixed documentation
- Bug fixes and stability improvements

### v0.0.3 (Current)
- Removed CatBoost dependency for simpler installation
- Bug fixes and stability improvements

### v0.0.2
- Complete documentation overhaul
- Added `FTIRdataprocessing` and `FTIRdataanalysis` classes
- 6 bundled FTIR datasets
- 20+ machine learning models with SHAP explainability
- Comprehensive evaluation methods for all preprocessing steps
- Advanced visualization and statistical analysis tools

### v0.0.1
- Initial release
- Basic preprocessing utilities
