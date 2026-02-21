# Getting Started

This guide will help you get started with xpectrass for FTIR spectral data analysis.

## Prerequisites

- Python 3.8+
- NumPy, SciPy, Pandas, Polars
- PyBaselines, PyWavelets
- Matplotlib, scikit-learn

## Installation

### From PyPI

```bash
pip install xpectrass
```

### From Source

```bash
git clone https://github.com/kazilab/xpectrass.git
cd xpectrass
pip install -e .
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

## Key Methods Quick Reference

### FTIRdataprocessing

| Method | Purpose |
|--------|---------|
| `convert()` | Transmittance ↔ Absorbance conversion |
| `exclude_interpolate()` | Remove atmospheric interference |
| `find_baseline_method()` | Evaluate baseline correction methods |
| `correct_baseline()` | Apply baseline correction |
| `find_denoising_method()` | Evaluate denoising methods |
| `denoise_spect()` | Apply denoising |
| `find_normalization_method()` | Evaluate normalization methods |
| `normalize()` | Apply normalization |
| `derivatives()` | Calculate spectral derivatives |
| `plot_multiple_spec()` | Compare all processing stages |
| `run()` | Execute full pipeline with defaults |

### FTIRdataanalysis

| Method | Purpose |
|--------|---------|
| `plot_mean_spectra()` | Plot mean spectra by class |
| `plot_pca()` | PCA analysis |
| `plot_tsne()` | t-SNE analysis |
| `plot_umap()` | UMAP analysis |
| `plot_plsda()` | PLS-DA analysis |
| `perform_anova()` | ANOVA statistical test |
| `ml_prepare_data()` | Prepare train/test split |
| `run_all_models()` | Evaluate all ML models |
| `model_parameter_tuning()` | Tune hyperparameters |
| `explain_by_shap()` | SHAP explainability |

## Getting Help

- Acknowledgements: [Credits and tooling assistance](acknowledgements.md)
- Documentation: [Read the Docs](https://xpectrass.readthedocs.io)
- GitHub: [github.com/kazilab/xpectrass](https://github.com/kazilab/xpectrass)
- Issues: [Report bugs or request features](https://github.com/kazilab/xpectrass/issues)
