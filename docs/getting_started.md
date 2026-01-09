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
,type,400.0,401.0,402.0,...,4000.0
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

# Step 2: Remove atmospheric interference (CO₂, H₂O)
ftir.exclude_interpolate(method="spline", plot=True)

# Step 3: Evaluate and apply baseline correction
ftir.find_baseline_method(n_samples=50, plot=True)
ftir.plot_rfzn_nar_snr()  # Visualize evaluation metrics
ftir.correct_baseline(method="asls", plot=False)

# Step 4: Evaluate and apply denoising
ftir.find_denoising_method(n_samples=50, plot=True)
ftir.denoise_spect(method="savgol", window_length=15)

# Step 5: Evaluate and apply normalization
ftir.find_normalization_method(plot=True)
ftir.normalize(method="snv")

# Step 6: Compare all processing stages
ftir.plot_multiple_spec(sample="HDPE_001")

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

# Initialize analysis
analysis = FTIRdataanalysis(processed_df, label_column="type")

# Visualize mean spectra by class
analysis.plot_mean_spectra(by_class=True)

# Plot spectral heatmap
analysis.plot_heatmap()

# Dimensionality reduction
analysis.plot_pca(n_components=3)
analysis.plot_tsne(perplexity=30)
analysis.plot_umap(n_neighbors=15)

# Statistical analysis
analysis.perform_anova()
analysis.plot_correlation()
```

## Machine Learning

```python
# Prepare data for ML
analysis.ml_prepare_data(test_size=0.2, random_state=42)

# Run all classification models
results = analysis.run_all_models()
print(results.sort_values('f1_score', ascending=False))

# Tune top performing models
tuned_results = analysis.model_parameter_tuning(top_n=3)

# Explain model predictions with SHAP
analysis.explain_by_shap(model='RandomForest', X=analysis.X_test)
analysis.local_shap_plot()
```

## Complete Example

Here's a complete workflow from data loading to machine learning:

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_jung_2018

# 1. Load data
df = load_jung_2018()
print(f"Loaded {len(df)} spectra")

# 2. Preprocessing
ftir = FTIRdataprocessing(df, label_column="type", wn_min=400, wn_max=4000)

ftir.convert(mode="to_absorbance")
ftir.exclude_interpolate(method="spline")
ftir.find_baseline_method(n_samples=50)
ftir.correct_baseline(method="asls")
ftir.find_denoising_method(n_samples=50)
ftir.denoise_spect(method="savgol")
ftir.normalize(method="snv")

processed_df = ftir.df_norm
print(f"Preprocessing complete: {processed_df.shape}")

# 3. Visualization and Analysis
analysis = FTIRdataanalysis(processed_df, label_column="type")

analysis.plot_mean_spectra(by_class=True)
analysis.plot_pca(n_components=3)

# 4. Machine Learning
analysis.ml_prepare_data(test_size=0.2)
results = analysis.run_all_models()

print("\nTop 5 Models:")
print(results.nlargest(5, 'f1_score')[['model', 'accuracy', 'f1_score']])

# 5. Model tuning
tuned = analysis.model_parameter_tuning(top_n=1)
print(f"\nBest model after tuning: {tuned.iloc[0]['model']}")
print(f"Tuned F1 score: {tuned.iloc[0]['f1_score']:.3f}")
```

## Data Validation

Always validate your data before preprocessing:

```python
from xpectrass.utils import validate_spectra

# Validate data
report = validate_spectra(df, verbose=True)

if report['valid']:
    print("✓ Data passed all validation checks!")
else:
    print("✗ Data validation failed:")
    for issue in report['issues']:
        print(f"  - {issue}")
```

## Understanding the Pipeline

### Key Concepts

1. **State Management**: `FTIRdataprocessing` stores results at each step (df, converted_df, df_atm, df_corr, df_denoised, df_norm, df_deriv)

2. **Evaluation First**: Always evaluate methods before applying:
   - `find_baseline_method()` → `correct_baseline()`
   - `find_denoising_method()` → `denoise_spect()`
   - `find_normalization_method()` → `normalize()`

3. **Visualization**: Most methods have a `plot` parameter for immediate feedback

4. **Flexibility**: Skip or reorder steps as needed for your data

### Common Preprocessing Orders

**For ATR-FTIR data:**
```
Convert → Baseline → Denoise → Normalize
(Skip atmospheric correction for ATR)
```

**For Transmission FTIR data:**
```
Convert → Atmospheric → Baseline → Denoise → Normalize
(Include atmospheric correction)
```

**For PCA/PLS analysis:**
```
... → Normalize → Derivatives (1st or 2nd)
(Add derivatives before dimensionality reduction)
```

## Troubleshooting

### Issue: "Could not detect wavenumber columns"

**Solution**: Ensure your wavenumber columns are numeric or can be converted to float.

```python
# Fix non-numeric wavenumber columns
df.columns = [col if col == 'type' else float(col.replace('cm', ''))
              for col in df.columns]
```

### Issue: Baseline correction creates negative values

**Solution**: Increase the asymmetry parameter or try a different method:

```python
# More conservative baseline
ftir.correct_baseline(method="asls", lam=1e6, p=0.1)

# Or try a different method
ftir.correct_baseline(method="airpls")
```

### Issue: Denoising over-smooths peaks

**Solution**: Reduce window size or try different methods:

```python
# Smaller window
ftir.denoise_spect(method="savgol", window_length=7)

# Different method
ftir.denoise_spect(method="wavelet")
```

## Next Steps

- Read the [User Guide](user_guide/index.md) for detailed documentation
- Check out [Data Loading](user_guide/data_loading.md) to explore bundled datasets
- See [Preprocessing Pipeline](user_guide/preprocessing_pipeline.md) for advanced options
- Learn about [Analysis](user_guide/analysis.md) for visualization and statistics
- Explore [Machine Learning](user_guide/machine_learning.md) for classification workflows
- Check the [API Reference](api/index.md) for complete function documentation
- See [Examples](examples.md) for real-world use cases

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

- Documentation: [Read the Docs](https://xpectrass.readthedocs.io)
- GitHub: [github.com/kazilab/xpectrass](https://github.com/kazilab/xpectrass)
- Issues: [Report bugs or request features](https://github.com/kazilab/xpectrass/issues)
