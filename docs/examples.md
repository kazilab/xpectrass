# Examples

This page provides complete examples for common FTIR preprocessing workflows.

## Example 1: Basic Preprocessing

Load data and apply standard preprocessing for classification:

```python
import glob
import sys
sys.path.insert(0, '/path/to/scripts')

from xpectrass.utils import process_batch_files, validate_spectra
from xpectrass.preprocessing_pipeline import create_preprocessor

# Load all plastic spectra
files = glob.glob('FTIR-PLASTIC-c4/*/*.csv')
df = process_batch_files(files, show_progress=True)

print(f"Loaded {df.height} spectra with {len(df.columns) - 2} wavenumbers")

# Validate data
report = validate_spectra(df, verbose=True)

# Create and apply standard preprocessing
pipe = create_preprocessor('standard')
processed = pipe.fit_transform(df)

print("Preprocessing complete!")
```

---

## Example 2: Custom Pipeline Configuration

Configure each preprocessing step individually:

```python
from xpectrass.preprocessing_pipeline import FTIRPreprocessor, PreprocessingConfig

# Custom configuration
config = PreprocessingConfig(
    # Enable validation
    validate=True,
    
    # Baseline correction with ASLS
    baseline=True,
    baseline_params={'method': 'asls', 'lam': 1e7, 'p': 0.01},
    
    # Wavelet denoising
    denoise=True,
    denoise_params={'method': 'wavelet', 'wavelet': 'db4', 'level': 3},
    
    # Vector normalization
    normalize=True,
    normalize_params={'method': 'vector'},
    
    # Select classification regions
    region_selection=True,
    regions=[(400, 1800), (2700, 3100)],
    
    # Mean center for PCA
    mean_center=True
)

pipe = FTIRPreprocessor(config)
processed = pipe.fit_transform(df)
```

---

## Example 3: Evaluating Baseline Methods

Compare baseline correction methods on your data:

```python
from xpectrass.utils import (
    process_batch_files, baseline_method_names,
    evaluate_all_samples, plot_metric_boxes
)

# Load data
files = glob.glob('FTIR-PLASTIC-c4/*/*.csv')[:100]  # Subset for speed
df = process_batch_files(files)

# Define flat regions (known baseline-only areas)
flat_windows = [(2500, 2600), (3350, 3450)]

# Evaluate all baseline methods
rfzn, nar, snr = evaluate_all_samples(df, flat_windows)

# Visualize results
plot_metric_boxes(rfzn, metric_name="RFZN")
plot_metric_boxes(nar, metric_name="NAR")

# Find best methods
print("Top 5 methods by RFZN (lower is better):")
print(rfzn.mean().sort_values().head())
```

---

## Example 4: Comparing Denoising Methods

```python
from xpectrass.utils import (
    process_batch_files, denoise, denoise_method_names,
    evaluate_denoising, plot_denoising_comparison,
    get_wavenumbers, get_spectra_matrix
)

# Load single sample
df = process_batch_files(['FTIR-PLASTIC-c4/HDPE_c4/HDPE1.csv'])

wavenumbers = get_wavenumbers(df)
spectrum = get_spectra_matrix(df)[0]

# Visual comparison
plot_denoising_comparison(
    spectrum, wavenumbers,
    methods=['savgol', 'wavelet', 'gaussian'],
    sample_name='HDPE1'
)

# Quantitative evaluation
results = evaluate_denoising(df, n_samples=50)
print(results.groupby('method')[['snr_db', 'fidelity']].mean())
```

---

## Example 5: PCA Preprocessing

Complete preprocessing for PCA analysis:

```python
from xpectrass.preprocessing_pipeline import create_preprocessor
from xpectrass.utils import get_spectra_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and preprocess
files = glob.glob('FTIR-PLASTIC-c4/*/*.csv')
df = process_batch_files(files)

# Use PCA preset (includes 1st derivative)
pipe = create_preprocessor('pca')
processed = pipe.fit_transform(df)

# Extract matrix for PCA
X = get_spectra_matrix(processed)
labels = processed['label'].to_list()

# Perform PCA
pca = PCA(n_components=5)
scores = pca.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
for label in set(labels):
    mask = [l == label for l in labels]
    plt.scatter(scores[mask, 0], scores[mask, 1], label=label, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.title('PCA of Preprocessed FTIR Spectra')
plt.show()
```

---

## Example 6: Classification Pipeline

Complete pipeline for plastic classification:

```python
from xpectrass.preprocessing_pipeline import create_preprocessor
from xpectrass.utils import process_batch_files, get_spectra_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
files = glob.glob('FTIR-PLASTIC-c4/*/*.csv')
df = process_batch_files(files)

# Preprocess with classification preset
pipe = create_preprocessor('classification')
processed = pipe.fit_transform(df)

# Prepare for sklearn
X = get_spectra_matrix(processed)
y = processed['label'].to_list()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Example 7: Region Analysis

Analyze important spectral regions for each plastic type:

```python
from xpectrass.utils import (
    process_batch_files, analyze_regions, 
    select_region, FTIR_REGIONS
)

# Load data
df = process_batch_files(glob.glob('FTIR-PLASTIC-c4/*/*.csv'))

# Analyze predefined regions
stats = analyze_regions(df)
print(stats)

# Custom region analysis
plastic_regions = [
    (700, 750),    # CH2 rocking (PE)
    (1710, 1730),  # Ester C=O (PET)
    (1370, 1380),  # CH3 symmetric (PP)
    (690, 760),    # Aromatic (PS)
    (600, 700),    # C-Cl (PVC)
]

custom_stats = analyze_regions(df, regions=plastic_regions)
print(custom_stats)
```
