# User Guide

```{toctree}
:maxdepth: 2

preprocessing_pipeline
data_loading
analysis
machine_learning
data_validation
baseline_correction
denoising
normalization
atmospheric_correction
spectral_derivatives
scatter_correction
region_selection
```

## Overview

This user guide provides comprehensive documentation for the xpectrass FTIR data processing and analysis library. The library is built around two main classes:

- **FTIRdataprocessing**: Handles all preprocessing steps with evaluation and visualization
- **FTIRdataanalysis**: Provides statistical analysis, dimensionality reduction, clustering, and machine learning

Each section covers:

- **Purpose**: Why this feature is important
- **Available Methods**: All implemented algorithms
- **Parameters**: Configurable options
- **Examples**: Code snippets and use cases
- **Best Practices**: Recommendations for FTIR plastic classification

## Quick Start

### Basic Preprocessing Workflow

```python
from xpectrass import FTIRdataprocessing
import pandas as pd

# Load your FTIR data
df = pd.read_csv("ftir_data.csv", index_col=0)

# Initialize the preprocessing pipeline
ftir = FTIRdataprocessing(
    df,
    label_column="type",
    wn_min=400,
    wn_max=4000
)

# Step 1: Convert to absorbance (if needed)
ftir.convert(mode="to_absorbance", plot=True)

# Step 2: Remove atmospheric interference
ftir.exclude_interpolate(method="spline", plot=True)

# Step 3: Find and apply best baseline correction
ftir.find_baseline_method(n_samples=50, plot=True)
ftir.correct_baseline(method="asls", plot=True)

# Step 4: Find and apply best denoising
ftir.find_denoising_method(n_samples=50, plot=True)
ftir.denoise_spect(method="savgol")

# Step 5: Evaluate and apply normalization
ftir.find_normalization_method(plot=True)
ftir.normalize(method="snv")

# Step 6: Compare all processing stages
ftir.plot_multiple_spec(sample="Sample1")

# Get the processed data
processed_df = ftir.df_norm
```

### Basic Analysis Workflow

```python
from xpectrass import FTIRdataanalysis

# Initialize analysis with processed data
analysis = FTIRdataanalysis(processed_df, label_column="type")

# Visualize spectra
analysis.plot_mean_spectra(by_class=True)
analysis.plot_heatmap()

# Dimensionality reduction
analysis.plot_pca(n_components=3)
analysis.plot_tsne()
analysis.plot_umap()

# Statistical analysis
analysis.perform_anova()
analysis.plot_correlation()

# Machine learning
analysis.ml_prepare_data(test_size=0.2)
results = analysis.run_all_models()
analysis.model_parameter_tuning(top_n=3)
```

## Preprocessing Pipeline Order

The recommended preprocessing order is:

```
1. Data Validation     → Ensure data quality
2. Conversion          → Transmittance ↔ Absorbance conversion
3. Atmospheric Corr.   → Remove CO₂/H₂O interference
4. Baseline Correction → Remove instrumental artifacts
5. Denoising          → Reduce high-frequency noise
6. Scatter Correction  → Correct for scattering effects (optional)
7. Normalization      → Standardize intensity scales
8. Derivatives        → Enhance spectral features (optional)
```

## Key Features

### Evaluation-First Approach

Xpectrass uses an evaluation-first philosophy - for each major preprocessing step, you can evaluate multiple methods to find the best one for your data:

- **Baseline correction**: 50+ methods evaluated using RFZN, NAR, and SNR metrics
- **Denoising**: 7 methods evaluated using spectral quality metrics
- **Normalization**: 7+ methods evaluated using consistency and quality metrics

### Bundled Datasets

The library includes 6 pre-loaded FTIR plastic datasets from published studies:

```python
from xpectrass.data import load_jung_2018, load_all_datasets

# Load a specific dataset
df = load_jung_2018()

# Load all datasets
all_data = load_all_datasets()
```

### Comprehensive Machine Learning

Built-in support for 20+ classification models with:
- Automatic hyperparameter tuning
- SHAP explainability analysis
- Cross-validation and performance metrics
- Model comparison visualizations

## Main Classes

### FTIRdataprocessing

The preprocessing class maintains state through the entire pipeline:

| Attribute | Description |
|-----------|-------------|
| `df` | Original data |
| `converted_df` | After transmittance/absorbance conversion |
| `df_atm` | After atmospheric correction |
| `df_corr` | After baseline correction |
| `df_denoised` | After denoising |
| `df_norm` | After normalization |
| `df_deriv` | After derivative calculation |

### FTIRdataanalysis

The analysis class provides visualization, statistics, and machine learning:

| Category | Methods |
|----------|---------|
| **Visualization** | `plot_mean_spectra`, `plot_overlay_spectra`, `plot_heatmap`, `plot_cv` |
| **Dimensionality Reduction** | `plot_pca`, `plot_tsne`, `plot_umap`, `plot_plsda`, `plot_oplsda` |
| **Clustering** | `plot_kmeans_clus`, `plot_hierarchical_clus` |
| **Statistics** | `perform_anova`, `plot_correlation` |
| **Machine Learning** | `run_all_models`, `model_parameter_tuning`, `explain_by_shap` |
