# xpectrass Documentation

**A comprehensive spectral preprocessing toolkit for spectra-based classification.**

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
user_guide/index
api/index
examples
changelog
```

## Overview

xpectrass (Spectral Analysis Suite) provides a complete preprocessing pipeline for FTIR spectroscopy data, specifically designed for plastic identification and classification tasks.

### Key Features

- **9 preprocessing steps** with multiple configurable methods
- **50+ baseline correction algorithms** via pybaselines integration
- **sklearn-compatible API** with fit/transform pattern
- **Preset configurations** for common use cases
- **Polars DataFrame support** for high-performance data handling

### Quick Start

```python
from xpectrass.preprocessing_pipeline import create_preprocessor
from xpectrass.utils import process_batch_files

# Load FTIR data
files = glob.glob('data/*.csv')
df = process_batch_files(files)

# Create and apply preprocessing pipeline
pipe = create_preprocessor('standard')
processed = pipe.fit_transform(df)
```

### Installation

```bash
# Install from PyPI
pip install xpectrass

# Or install from source
git clone https://github.com/kazilab/xpectrass.git
cd xpectrass
pip install -e .
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
