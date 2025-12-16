# Getting Started

This guide will help you get started with the xpectrass preprocessing pipeline.

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

## Basic Usage

### Loading Data

FTIR spectral data should be in CSV format with wavenumber and intensity columns:

```python
from xpectrass.utils import process_batch_files
import glob

# Load multiple spectra
files = glob.glob('data/HDPE_c4/*.csv')
df = process_batch_files(files, skiprows=15)

print(f"Loaded {df.height} spectra")
print(f"Columns: sample, label, + {len(df.columns) - 2} wavenumbers")
```

### Data Validation

Always validate your data before preprocessing:

```python
from xpectrass.utils import validate_spectra

report = validate_spectra(df, verbose=True)
# Prints detailed validation report

if report['valid']:
    print("Data passed all validation checks!")
```

### Quick Preprocessing

Use the pipeline with presets for common workflows:

```python
from xpectrass.preprocessing_pipeline import create_preprocessor

# Standard preprocessing pipeline
pipe = create_preprocessor('standard')
processed = pipe.fit_transform(df)
```

### Available Presets

| Preset | Description |
|--------|-------------|
| `minimal` | Baseline + SNV normalization only |
| `standard` | Baseline → Denoise → Normalize → Mean center |
| `classification` | Optimized for plastic classification |
| `pca` | Standard + first derivative (for PCA/PLS) |
| `raw` | Validation only, no preprocessing |

## Next Steps

- Read the [User Guide](user_guide/index.md) for detailed preprocessing options
- Check the [API Reference](api/index.md) for complete function documentation
- See [Examples](examples.md) for real-world use cases
