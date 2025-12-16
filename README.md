# xpectrass

A comprehensive preprocessing toolkit for spectral classification.

[![Python Version](https://img.shields.io/pypi/pyversions/xpectrass.svg)](https://pypi.org/project/xpectrass/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/xpectrass/badge/?version=latest)](https://xpectrass.readthedocs.io/)

## Features

- **9 Preprocessing Steps** with multiple configurable methods
- **50+ Baseline Correction** algorithms via pybaselines
- **sklearn-compatible API** with fit/transform pattern
- **Preset Configurations** for common use cases
- **Polars DataFrame Support** for high-performance data handling

### Preprocessing Pipeline

| Step | Methods Available |
|------|-------------------|
| Data Validation | Completeness, range checks, outlier detection |
| Baseline Correction | 50+ methods (airpls, asls, poly, mor, ...) |
| Denoising | Savitzky-Golay, wavelet, median, Gaussian, ... |
| Normalization | SNV, vector, min-max, area, peak normalization |
| Atmospheric Correction | CO₂/H₂O interpolation, reference subtraction |
| Spectral Derivatives | 1st/2nd derivative, gap derivative |
| Scatter Correction | MSC, EMSC, SNV+detrend |
| Region Selection | Predefined FTIR regions for plastics |
| Mean Centering | For PCA/PLS preparation |

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

```python
from xpectrass import FTIRPreprocessor, create_preprocessor
from xpectrass.utils import process_batch_files
import glob

# Load FTIR data
files = glob.glob('data/*.csv')
df = process_batch_files(files)

# Standard preprocessing pipeline
pipe = create_preprocessor('standard')
processed = pipe.fit_transform(df)
```

### Custom Configuration

```python
from xpectrass import FTIRPreprocessor, PreprocessingConfig

config = PreprocessingConfig(
    baseline_params={'method': 'airpls', 'lam': 1e6},
    denoise_params={'method': 'savgol', 'window_length': 15},
    normalize_params={'method': 'snv'},
    region_selection=True,
    regions=[(400, 1800), (2700, 3100)],
)

pipe = FTIRPreprocessor(config)
processed = pipe.fit_transform(df)
```

### Available Presets

| Preset | Description |
|--------|-------------|
| `minimal` | Baseline + SNV normalization |
| `standard` | Baseline → Denoise → Normalize → Mean center |
| `classification` | Optimized for plastic classification |
| `pca` | Standard + first derivative |
| `raw` | Validation only |

## Individual Utilities

```python
from xpectrass.utils import (
    # Baseline
    baseline_correction, baseline_method_names,
    
    # Denoising
    denoise, denoise_method_names,
    
    # Normalization
    normalize, mean_center, auto_scale,
    
    # Derivatives
    spectral_derivative, first_derivative,
    
    # Region selection
    select_region, FTIR_REGIONS,
)

# Apply individual preprocessing
corrected = baseline_correction(spectrum, method='airpls')
denoised = denoise(corrected, method='savgol')
normalized = normalize(denoised, method='snv')
```

## Documentation

Full documentation is available at [xpectrass.readthedocs.io](https://xpectrass.readthedocs.io/).

### Building Documentation Locally

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, Pandas, Polars
- PyBaselines, PyWavelets
- Matplotlib, scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{xpectrass,
  author = {FTIR-PLASTIC Team},
  title = {xpectrass: FTIR/ToF-SIMS Spectral Analysis Suite},
  year = {2024},
  url = {https://github.com/kazilab/xpectrass}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
