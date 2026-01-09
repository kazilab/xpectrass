# Preprocessing Pipeline

The `FTIRdataprocessing` class provides a comprehensive interface for FTIR spectral data preprocessing with built-in evaluation and visualization at every step.

## Overview

The `FTIRdataprocessing` class streamlines FTIR data preprocessing workflows by:

- Maintaining state through each processing step
- Providing evaluation methods to find optimal parameters
- Offering visualization at every stage
- Supporting both pandas and polars DataFrames
- Storing all intermediate results for easy access

```python
from xpectrass import FTIRdataprocessing
import pandas as pd

# Load your data
df = pd.read_csv("ftir_data.csv", index_col=0)

# Initialize the pipeline
ftir = FTIRdataprocessing(
    df,
    label_column="type",
    wn_min=400,
    wn_max=4000
)
```

## Initialization Parameters

### FTIRdataprocessing.__init__()

```python
FTIRdataprocessing(
    df,                                    # DataFrame with samples as rows
    label_column="type",                   # Label column name
    exclude_columns=None,                  # Additional non-spectral columns
    wn_min=None,                          # Minimum wavenumber (cm⁻¹)
    wn_max=None,                          # Maximum wavenumber (cm⁻¹)
    exclude_regions=EXCLUDE_REGIONS,      # Regions to remove completely
    interpolate_regions=INTERPOLATE_REGIONS, # Regions to interpolate
    flat_windows=FLAT_WINDOWS,            # Flat regions for baseline eval
    baseline_methods=FTIR_BASELINE_METHODS,  # Methods to evaluate
    denoising_methods=FTIR_DENOISING_METHODS, # Methods to evaluate
    normalization_methods=FTIR_NORMALIZATION_METHODS, # Methods to evaluate
    sample_selection="random",            # "random" or "stratified"
    random_state=None,                    # Random seed for reproducibility
    n_jobs=-1                             # Parallel processing cores
)
```

### Default Regions

The class comes with sensible defaults for FTIR plastic analysis:

```python
# Default exclude regions (removed from data)
EXCLUDE_REGIONS = [
    (0, 680),       # Below 680 cm⁻¹ (CO₂ bending)
    (3500, 5000)    # Above 3500 cm⁻¹ (O-H stretch)
]

# Default interpolation regions (atmospheric interference)
INTERPOLATE_REGIONS = [
    (1250, 2400)    # H₂O + CO₂ region
]

# Default flat windows (for baseline evaluation)
FLAT_WINDOWS = [
    (1800, 1900),   # Between fingerprint and CH regions
    (2400, 2700)    # Between CO₂ and CH stretch
]
```

## Processing Workflow

### 1. Data Conversion

Convert between transmittance and absorbance:

```python
# Convert to absorbance
ftir.convert(mode="to_absorbance", plot=True)

# Convert to transmittance
ftir.convert(mode="to_transmittance", plot=False)

# Access converted data
converted_data = ftir.converted_df
```

### 2. Atmospheric Correction

Remove CO₂ and H₂O interference:

```python
# Exclude and interpolate atmospheric regions
ftir.exclude_interpolate(
    method="spline",      # "spline", "linear", "polynomial"
    plot=True
)

# Access corrected data
atm_corrected = ftir.df_atm
```

### 3. Baseline Correction

#### Step 3a: Evaluate Baseline Methods

```python
# Evaluate all baseline methods on sample spectra
ftir.find_baseline_method(
    n_samples=50,         # Number of spectra to test
    plot=True             # Show evaluation plots
)

# View evaluation results
print(ftir.rfzn_tbl)  # Residual Flatness in Zero Noise
print(ftir.nar_tbl)   # Negative Absorbance Ratio
print(ftir.snr_tbl)   # Signal-to-Noise Ratio

# Plot evaluation metrics
ftir.plot_rfzn_nar_snr()
```

**Metrics Interpretation:**
- **RFZN** (lower is better): RMS of residual in flat regions
- **NAR** (lower is better): Fraction of negative area after correction
- **SNR** (higher is better): Peak height / noise level

#### Step 3b: Apply Best Baseline Method

```python
# Apply baseline correction
ftir.correct_baseline(
    method="asls",        # or "airpls", "arpls", etc.
    plot=True,
    **kwargs              # Method-specific parameters
)

# Access baseline-corrected data
baseline_corrected = ftir.df_corr
```

**Common baseline methods:**
- `asls`: Asymmetric Least Squares (fast, good general purpose)
- `airpls`: Adaptive Iteratively Reweighted PLS (recommended for FTIR)
- `arpls`: Asymmetrically Reweighted PLS (strong baselines)
- `rubberband`: Rubberband baseline (fast, simple)
- `snip`: Statistics-sensitive Non-linear Iterative Peak-clipping

### 4. Denoising

#### Step 4a: Evaluate Denoising Methods

```python
# Evaluate denoising methods
ftir.find_denoising_method(
    n_samples=50,
    plot=True
)

# View results
print(ftir.denoising_results)

# Plot evaluation
ftir.plot_denoising_eval()
```

#### Step 4b: Apply Best Denoising Method

```python
# Apply denoising
ftir.denoise_spect(
    method="savgol",      # See available methods below
    window_length=15,     # Odd integer
    polyorder=3,          # For Savitzky-Golay
    plot=False
)

# Access denoised data
denoised = ftir.df_denoised
```

**Available denoising methods:**
- `savgol`: Savitzky-Golay filter (recommended)
- `wavelet`: Wavelet denoising
- `gaussian`: Gaussian smoothing
- `median`: Median filter
- `bilateral`: Bilateral filter
- `wiener`: Wiener filter
- `fft`: FFT-based filtering

### 5. Normalization

#### Step 5a: Evaluate Normalization Methods

```python
# Evaluate normalization methods
ftir.find_normalization_method(
    n_samples=50,
    plot=True
)

# View results
print(ftir.norm_eval_results)
```

#### Step 5b: Apply Normalization

```python
# Apply normalization
ftir.normalize(
    method="snv",         # See methods below
    plot=False
)

# Access normalized data
normalized = ftir.df_norm
```

**Available normalization methods:**
- `snv`: Standard Normal Variate (recommended for solids)
- `vector`: Vector (L2) normalization
- `minmax`: Min-Max scaling (0 to 1)
- `area`: Area normalization
- `peak`: Peak normalization
- `pqn`: Probabilistic Quotient Normalization
- `entropy`: Entropy-weighted normalization

### 6. Spectral Derivatives

Calculate first or second derivatives:

```python
# Calculate first derivative
ftir.derivatives(
    derivative_type="first",   # "first", "second", or "gap"
    window_length=15,          # Smoothing window
    polyorder=3,               # Polynomial order
    plot=True
)

# Plot derivative comparison
ftir.plot_deriv()

# Access derivative data
derivative = ftir.df_deriv
```

### 7. Visualization and Comparison

#### Plot Individual Spectra

```python
# Plot current data state
ftir.plot()
```

#### Compare All Processing Stages

```python
# Compare all stages for a specific sample
ftir.plot_multiple_spec(
    sample="Sample_001",      # Sample name
    figsize=(15, 12)
)
```

This shows:
- Original spectrum
- After conversion
- After atmospheric correction
- After baseline correction
- After denoising
- After normalization
- After derivatives (if applied)

## Accessing Processed Data

All intermediate results are stored as DataFrame attributes:

```python
# Original data
original = ftir.df

# After each step
converted = ftir.converted_df
atm_corrected = ftir.df_atm
baseline_corrected = ftir.df_corr
denoised = ftir.df_denoised
normalized = ftir.df_norm
derivative = ftir.df_deriv

# Evaluation results
baseline_metrics = ftir.rfzn_tbl, ftir.nar_tbl, ftir.snr_tbl
denoise_metrics = ftir.denoising_results
norm_metrics = ftir.norm_eval_results
```

## Complete Example

```python
from xpectrass import FTIRdataprocessing
from xpectrass.utils import process_batch_files
import glob

# Load multiple FTIR spectra
files = glob.glob('data/plastics/*.csv')
df = process_batch_files(files)

# Initialize pipeline
ftir = FTIRdataprocessing(
    df,
    label_column="polymer_type",
    wn_min=400,
    wn_max=4000
)

# Full preprocessing workflow
print("Step 1: Converting to absorbance...")
ftir.convert(mode="to_absorbance", plot=False)

print("Step 2: Removing atmospheric interference...")
ftir.exclude_interpolate(method="spline", plot=False)

print("Step 3: Evaluating baseline methods...")
ftir.find_baseline_method(n_samples=100, plot=True)
ftir.plot_rfzn_nar_snr()

print("Step 4: Applying best baseline correction...")
ftir.correct_baseline(method="asls", plot=False)

print("Step 5: Evaluating denoising methods...")
ftir.find_denoising_method(n_samples=100, plot=True)

print("Step 6: Applying denoising...")
ftir.denoise_spect(method="savgol", window_length=15)

print("Step 7: Evaluating normalization methods...")
ftir.find_normalization_method(plot=True)

print("Step 8: Applying normalization...")
ftir.normalize(method="snv")

# Compare processing stages
ftir.plot_multiple_spec(sample="HDPE_001")

# Get final processed data
processed_data = ftir.df_norm

print(f"Original shape: {df.shape}")
print(f"Processed shape: {processed_data.shape}")
print("Preprocessing complete!")
```

## Quick Run Method

For quick testing with default parameters:

```python
# Run entire pipeline with defaults
ftir.run()

# This executes:
# 1. Atmospheric correction
# 2. Baseline correction (airpls)
# 3. Denoising (savgol)
# 4. Normalization (snv)
```

## Advanced Features

### Custom Baseline Parameters

```python
# AsLS with custom parameters
ftir.correct_baseline(
    method="asls",
    lam=1e6,       # Smoothness (1e4 to 1e8)
    p=0.01         # Asymmetry (0.001 to 0.1)
)

# AirPLS with custom parameters
ftir.correct_baseline(
    method="airpls",
    lam=1e6,
    max_iter=50
)
```

### Custom Denoising Parameters

```python
# Savitzky-Golay with custom window
ftir.denoise_spect(
    method="savgol",
    window_length=21,    # Must be odd
    polyorder=5
)

# Wavelet denoising
ftir.denoise_spect(
    method="wavelet",
    wavelet="db4",
    level=3
)
```

### Stratified Sampling for Evaluation

```python
# Use stratified sampling for evaluation (better for imbalanced classes)
ftir = FTIRdataprocessing(
    df,
    label_column="type",
    sample_selection="stratified"
)

ftir.find_baseline_method(n_samples=50)
```

## Tips and Best Practices

1. **Always evaluate first**: Use `find_baseline_method()`, `find_denoising_method()`, and `find_normalization_method()` before applying corrections
2. **Use appropriate sample sizes**: 50-100 samples is usually sufficient for evaluation
3. **Check intermediate results**: Use `plot=True` to visualize each step
4. **Compare processing stages**: Use `plot_multiple_spec()` to see the effect of each step
5. **Save evaluation results**: Store `rfzn_tbl`, `denoising_results`, etc. for reproducibility
6. **Use defaults as starting point**: The default regions and methods are optimized for FTIR plastics
7. **Consider your data**: ATR-FTIR may not need atmospheric correction; transmission FTIR does

## Next Steps

- See [Baseline Correction](baseline_correction.md) for detailed baseline method documentation
- See [Denoising](denoising.md) for denoising algorithm details
- See [Normalization](normalization.md) for normalization method comparison
- See [Analysis](analysis.md) for post-processing analysis with `FTIRdataanalysis`
