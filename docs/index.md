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
from xpectrass import FTIRdataprocessing
from xpectrass.data import load_jung_2018, get_data_info

# See available datasets
print(get_data_info())

# Load a dataset
df = load_jung_2018()
print(f"Loaded {len(df)} spectra")

# Start preprocessing
ftir = FTIRdataprocessing(df, label_column="type")

Your data:
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
