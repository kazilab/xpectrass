# Data Loading

Xpectrass includes built-in FTIR plastic datasets and utilities for loading your own data.

## Bundled Datasets

The library comes with 6 pre-loaded FTIR plastic datasets from published studies, perfect for testing, tutorials, and reproducible research.

### Available Datasets

```python
from xpectrass.data import get_data_info

# View all available datasets
info = get_data_info()
print(info)
```

| Dataset | Study | Samples | Polymer Types | Year |
|---------|-------|---------|---------------|------|
| `load_jung_2018()` | Jung et al. | ~500 | PE, PP, PS, PET, PVC, etc. | 2018 |
| `load_kedzierski_2019()` | Kedzierski et al. | ~300 | Various plastics | 2019 |
| `load_kedzierski_2019_u()` | Kedzierski et al. (unprocessed) | ~300 | Various plastics | 2019 |
| `load_frond_2021()` | Frond et al. | ~400 | Common polymers | 2021 |
| `load_villegas_camacho_2024_c4()` | Villegas-Camacho et al. (C4) | ~600 | Microplastics | 2024 |
| `load_villegas_camacho_2024_c8()` | Villegas-Camacho et al. (C8) | ~600 | Microplastics | 2024 |

### Loading Individual Datasets

```python
from xpectrass.data import (
    load_jung_2018,
    load_kedzierski_2019,
    load_frond_2021,
    load_villegas_camacho_2024_c4
)

# Load Jung 2018 dataset
df = load_jung_2018()
print(f"Loaded {len(df)} spectra")
print(f"Columns: {df.columns[:10]}...")
print(f"Unique types: {df['type'].unique()}")

# Load Villegas-Camacho 2024 (C4 fraction)
df_c4 = load_villegas_camacho_2024_c4()
print(f"C4 fraction: {len(df_c4)} spectra")
```

### Loading All Datasets

```python
from xpectrass.data import load_all_datasets

# Load all datasets as a dictionary
all_data = load_all_datasets()

for name, df in all_data.items():
    print(f"{name}: {len(df)} spectra, {len(df.columns)-1} wavenumbers")
```

### Loading Specific Datasets

```python
from xpectrass.data import load_datasets

# Load only specific datasets
selected_data = load_datasets([
    'jung_2018',
    'frond_2021',
    'villegas_camacho_2024_c4'
])

print(f"Loaded {len(selected_data)} datasets")
```

### Dataset Information

Each dataset includes:
- **type column**: Polymer type (HDPE, LDPE, PP, PS, PET, PVC, etc.)
- **Wavenumber columns**: Typically 400-4000 cm⁻¹ range
- **Index**: Sample identifiers

```python
df = load_jung_2018()

# Inspect dataset
print(f"Shape: {df.shape}")
print(f"Label column: 'type'")
print(f"Wavenumber range: {df.columns[1]} to {df.columns[-1]}")
print(f"Polymer types: {df['type'].value_counts()}")
```

## Loading Your Own Data

### From Single CSV File

```python
import pandas as pd
from xpectrass import FTIRdataprocessing

# Load single file
df = pd.read_csv("my_ftir_data.csv", index_col=0)

# Start processing
ftir = FTIRdataprocessing(df, label_column="polymer_type")
```

**Expected CSV format:**
```
,type,400.0,401.0,402.0,...,4000.0
Sample_1,HDPE,0.123,0.125,0.128,...,0.045
Sample_2,PP,0.098,0.102,0.105,...,0.038
Sample_3,PE,0.115,0.118,0.121,...,0.041
```

### From Multiple CSV Files (Batch Loading)

```python
from xpectrass.utils import process_batch_files
import glob

# Load all CSV files in a directory
files = glob.glob('data/plastics/*.csv')
df = process_batch_files(files)

print(f"Loaded {len(df)} spectra from {len(files)} files")
```

**process_batch_files() parameters:**

```python
df = process_batch_files(
    file_list,              # List of file paths
    skiprows=0,             # Skip header rows (e.g., skiprows=15 for Opus files)
    label_from='filename',  # 'filename' or 'column'
    label_column=None,      # Column name if label_from='column'
    sep=',',                # CSV separator
    decimal='.',            # Decimal separator
)
```

### From Directory Structure

If your files are organized by polymer type:

```
data/
  HDPE/
    sample_001.csv
    sample_002.csv
  PP/
    sample_001.csv
    sample_002.csv
  PET/
    sample_001.csv
```

```python
import os
import glob
import pandas as pd

def load_from_directory_structure(base_path):
    """Load FTIR files organized by type in subdirectories."""
    data_list = []

    # Iterate through subdirectories
    for polymer_type in os.listdir(base_path):
        type_path = os.path.join(base_path, polymer_type)

        if os.path.isdir(type_path):
            # Load all CSV files in this subdirectory
            files = glob.glob(os.path.join(type_path, '*.csv'))

            for file in files:
                df_temp = pd.read_csv(file, index_col=0)
                df_temp['type'] = polymer_type
                data_list.append(df_temp)

    # Combine all data
    df = pd.concat(data_list, ignore_index=True)
    return df

# Load data
df = load_from_directory_structure('data')
print(f"Loaded {len(df)} spectra")
```

### From Excel Files

```python
import pandas as pd

# Load from Excel
df = pd.read_excel("ftir_data.xlsx", sheet_name="Spectra", index_col=0)

# Process as usual
from xpectrass import FTIRdataprocessing
ftir = FTIRdataprocessing(df, label_column="type")
```

## Data Format Requirements

### Required Format

1. **DataFrame structure**: Samples as rows, wavenumbers as columns
2. **Label column**: One column containing sample labels/types
3. **Wavenumber columns**: Numeric column names (400.0, 401.0, ...) or convertible to float
4. **Index**: Sample identifiers (optional but recommended)

### Valid Wavenumber Column Formats

The library automatically handles various column name formats:

```python
# All of these work:
# Format 1: Pure numeric
"400.0", "401.0", "402.0"

# Format 2: With units
"400.0cm", "401.0cm", "402.0cm"

# Format 3: Scientific notation
"4.00e2", "4.01e2", "4.02e2"

# Format 4: String numbers (will be converted)
"400", "401", "402"
```

The library uses robust wavenumber detection to handle edge cases automatically.

## Data Validation

Always validate your data after loading:

```python
from xpectrass.utils import validate_spectra

# Validate loaded data
df = load_jung_2018()
report = validate_spectra(df, verbose=True)

if report['valid']:
    print("✓ Data is valid!")
else:
    print("✗ Data validation failed:")
    for issue in report['issues']:
        print(f"  - {issue}")
```

### Validation Checks

The validation function checks for:
- Missing values (NaN, inf)
- Negative intensity values (for absorbance)
- Out-of-range values
- Sufficient samples per class
- Wavenumber continuity
- Data type consistency

## Combining Datasets

### Combine Bundled Datasets

```python
from xpectrass.data import load_jung_2018, load_frond_2021
from xpectrass.utils import combine_datasets
import pandas as pd

# Load individual datasets
df1 = load_jung_2018()
df2 = load_frond_2021()

# Simple concatenation (if wavenumber ranges match)
df_combined = pd.concat([df1, df2], ignore_index=True)

print(f"Combined: {len(df_combined)} spectra")
```

### Combine with Interpolation

If datasets have different wavenumber ranges:

```python
from xpectrass.utils import interpolate_to_common_grid

# Interpolate to common wavenumber grid
df1_interp = interpolate_to_common_grid(df1, target_wn=df2.columns[1:])
df_combined = pd.concat([df1_interp, df2], ignore_index=True)
```

## Data Preprocessing Before Analysis

After loading, typical workflow:

```python
from xpectrass import FTIRdataprocessing
from xpectrass.data import load_jung_2018

# 1. Load data
df = load_jung_2018()

# 2. Initialize preprocessing
ftir = FTIRdataprocessing(
    df,
    label_column="type",
    wn_min=400,
    wn_max=4000
)

# 3. Apply preprocessing pipeline
ftir.convert(mode="to_absorbance")
ftir.exclude_interpolate(method="spline")
ftir.find_baseline_method(n_samples=50)
ftir.correct_baseline(method="asls")
ftir.denoise_spect(method="savgol")
ftir.normalize(method="snv")

# 4. Get processed data
processed_df = ftir.df_norm
```

## Exporting Processed Data

Save your processed data for later use:

```python
# Save to CSV
processed_df.to_csv("processed_ftir_data.csv")

# Save to Excel
processed_df.to_excel("processed_ftir_data.xlsx", sheet_name="Processed")

# Save to Parquet (efficient for large datasets)
processed_df.to_parquet("processed_ftir_data.parquet")

# Load processed data later
import pandas as pd
df = pd.read_csv("processed_ftir_data.csv", index_col=0)
```

## Tips and Best Practices

1. **Use bundled datasets for testing**: Perfect for learning and validating workflows
2. **Validate after loading**: Always run `validate_spectra()` on your data
3. **Check wavenumber ranges**: Ensure your data covers the spectral region of interest
4. **Inspect class distribution**: Check for class imbalance before machine learning
5. **Save intermediate results**: Save preprocessed data to avoid reprocessing
6. **Document data sources**: Keep track of where your data came from

## Example: Complete Data Loading Workflow

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_jung_2018, get_data_info
from xpectrass.utils import validate_spectra

# 1. Explore available datasets
print("Available datasets:")
print(get_data_info())

# 2. Load a dataset
df = load_jung_2018()
print(f"\nLoaded Jung 2018 dataset:")
print(f"  - Samples: {len(df)}")
print(f"  - Wavenumbers: {len(df.columns)-1}")
print(f"  - Polymer types: {df['type'].nunique()}")

# 3. Check class distribution
print(f"\nClass distribution:")
print(df['type'].value_counts())

# 4. Validate data
report = validate_spectra(df, verbose=False)
print(f"\nValidation: {'✓ Passed' if report['valid'] else '✗ Failed'}")

# 5. Preprocess
print("\nPreprocessing...")
ftir = FTIRdataprocessing(df, label_column="type")
ftir.run()  # Quick run with defaults

# 6. Analyze
print("\nAnalyzing...")
analysis = FTIRdataanalysis(ftir.df_norm, label_column="type")
analysis.plot_pca(n_components=3)

# 7. Save processed data
ftir.df_norm.to_csv("jung_2018_processed.csv")
print("\n✓ Processing complete! Saved to jung_2018_processed.csv")
```

## Common Data Loading Patterns

### Pattern 1: Load and Preprocess Bundled Data

```python
from xpectrass import FTIRdataprocessing
from xpectrass.data import load_jung_2018

df = load_jung_2018()
ftir = FTIRdataprocessing(df, label_column="type")
ftir.run()
```

### Pattern 2: Load Multiple Files from Directory

```python
from xpectrass.utils import process_batch_files
import glob

files = glob.glob('data/**/*.csv', recursive=True)
df = process_batch_files(files)
```

### Pattern 3: Load and Combine Multiple Datasets

```python
from xpectrass.data import load_datasets
import pandas as pd

datasets = load_datasets(['jung_2018', 'frond_2021'])
df = pd.concat(datasets.values(), ignore_index=True)
```

## Next Steps

- See [Preprocessing Pipeline](preprocessing_pipeline.md) for data processing
- See [Data Validation](data_validation.md) for quality checks
- See [Analysis](analysis.md) for visualization and statistics
- See [Machine Learning](machine_learning.md) for classification workflows
