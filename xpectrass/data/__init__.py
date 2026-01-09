"""
Data loading utilities for XpectraS package.

This module provides convenient functions to load the bundled FTIR datasets
that are stored as xz-compressed CSV files.

Available Datasets
------------------
- frond_2021
- jung_2018
- kedzierski_2019.csv.xz
- 'kedzierski_2019_U.csv.xz'
- villegas_camacho_2024_c4
- villegas_camacho_2024_c8

Example Usage
-------------
>>> from xpectrass_v002.data import (
...     load_frond_2021,
...     load_jung_2018,
...     load_kedzierski_2019,
...     load_kedzierski_2019_u,
...     load_villegas_camacho_2024_c4,
...     load_villegas_camacho_2024_c8,
...     load_all_datasets
... )
>>>
>>> # Load individual dataset
>>> df_frond = load_frond_2021()
>>> print(df_frond.shape)
>>>
>>> # Load all datasets at once
>>> datasets = load_all_datasets()
>>> for name, df in datasets.items():
...     print(f"{name}: {df.shape}")
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Optional
import lzma

# Get the data directory path
_DATA_DIR = Path(__file__).parent

# Dataset file mapping
_DATASETS = {
    'jung_2018': 'jung_2018.csv.xz',
    'kedzierski_2019': 'kedzierski_2019.csv.xz',
    'kedzierski_2019_u': 'kedzierski_2019_u.csv.xz',
    'frond_2021': 'frond_2021.csv.xz',
    'villegas_camacho_2024_c4': 'villegas_camacho_2024_c4.csv.xz',
    'villegas_camacho_2024_c8': 'villegas_camacho_2024_c8.csv.xz',
}


def _load_compressed_csv(file_path: Path) -> pd.DataFrame:
    """
    Load an xz-compressed CSV file using Polars.

    Parameters
    ----------
    file_path : Path
        Path to the .csv.xz file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Please ensure all dataset files are present in the data directory."
        )

    # Polars can read xz-compressed CSV files directly
    return pd.read_csv(file_path, compression='xz')


def load_frond_2021() -> pd.DataFrame:
    """
    Load the Frond 2021 FTIR dataset.

    Returns
    -------
    pd.DataFrame
        Frond 2021 dataset as a Polars DataFrame.

    Examples
    --------
    >>> df = load_frond_2021()
    >>> print(df.shape)
    >>> print(df.columns[:5])
    """
    file_path = _DATA_DIR / _DATASETS['frond_2021']
    return _load_compressed_csv(file_path)


def load_jung_2018() -> pd.DataFrame:
    """
    Load the Jung 2018 FTIR dataset.

    Returns
    -------
    pd.DataFrame
        Jung 2018 dataset as a Polars DataFrame.

    Examples
    --------
    >>> df = load_jung_2018()
    >>> print(df.shape)
    >>> print(df.columns[:5])
    """
    file_path = _DATA_DIR / _DATASETS['jung_2018']
    return _load_compressed_csv(file_path)


def load_kedzierski_2019() -> pd.DataFrame:
    """
    Load the Kedzierski 2019 FTIR dataset.

    Returns
    -------
    pd.DataFrame
        Kedzierski 2019 dataset as a Polars DataFrame.

    Examples
    --------
    >>> df = load_kedzierski_2019()
    >>> print(df.shape)
    >>> print(df.columns[:5])
    """
    file_path = _DATA_DIR / _DATASETS['kedzierski_2019']
    return _load_compressed_csv(file_path)


def load_kedzierski_2019_u() -> pd.DataFrame:
    """
    Load the Kedzierski 2019 U FTIR dataset.

    Returns
    -------
    pd.DataFrame
        Kedzierski 2019 U dataset as a Polars DataFrame.

    Examples
    --------
    >>> df = load_kedzierski_2019_u()
    >>> print(df.shape)
    >>> print(df.columns[:5])
    """
    file_path = _DATA_DIR / _DATASETS['kedzierski_2019_u']
    return _load_compressed_csv(file_path)


def load_villegas_camacho_2024_c4() -> pd.DataFrame:
    """
    Load the Villegas-Camacho 2024 C4 FTIR dataset.

    Returns
    -------
    pd.DataFrame
        Villegas-Camacho 2024 C4 dataset as a Polars DataFrame.

    Examples
    --------
    >>> df = load_villegas_camacho_2024_c4()
    >>> print(df.shape)
    >>> print(df.columns[:5])
    """
    file_path = _DATA_DIR / _DATASETS['villegas_camacho_2024_c4']
    return _load_compressed_csv(file_path)


def load_villegas_camacho_2024_c8() -> pd.DataFrame:
    """
    Load the Villegas-Camacho 2024 C8 FTIR dataset.

    Returns
    -------
    pd.DataFrame
        Villegas-Camacho 2024 C8 dataset as a Polars DataFrame.

    Examples
    --------
    >>> df = load_villegas_camacho_2024_c8()
    >>> print(df.shape)
    >>> print(df.columns[:5])
    """
    file_path = _DATA_DIR / _DATASETS['villegas_camacho_2024_c8']
    return _load_compressed_csv(file_path)


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all available FTIR datasets.

    Returns
    -------
    dict of str -> pd.DataFrame
        Dictionary with dataset names as keys and their respective
        DataFrames as values.

    Examples
    --------
    >>> datasets = load_all_datasets()
    >>> for name, df in datasets.items():
    ...     print(f"{name}: {df.shape}")
    """
    return {
        'jung_2018': load_jung_2018(),
        'kedzierski_2019': load_kedzierski_2019(),
        'kedzierski_2019_u': load_kedzierski_2019_u(),
        'frond_2021': load_frond_2021(),
        'villegas_camacho_2024_c4': load_villegas_camacho_2024_c4(),
        'villegas_camacho_2024_c8': load_villegas_camacho_2024_c8(),
    }


def load_datasets(datasets: Optional[list] = None) -> Dict[str, pd.DataFrame]:
    """
    Load specific datasets by name.

    Parameters
    ----------
    datasets : list of str, optional
        List of dataset names to load. Available options:
        'jung_2018', 'frond_2021',
        'villegas_camacho_2024_c4', 'villegas_camacho_2024_c8'.
        If None, loads all datasets.

    Returns
    -------
    dict of str -> pd.DataFrame
        Dictionary with requested dataset names as keys and their
        respective DataFrames as values.

    Examples
    --------
    >>> datasets = load_datasets(['frond_2021', 'jung_2018'])
    >>> print(datasets.keys())
    dict_keys(['frond_2021', 'jung_2018'])
    """
    if datasets is None:
        return load_all_datasets()

    loader_map = {
        'jung_2018': load_jung_2018,
        'kedzierski_2019': load_kedzierski_2019(),
        'kedzierski_2019_u': load_kedzierski_2019_u(),
        'frond_2021': load_frond_2021,
        'villegas_camacho_2024_c4': load_villegas_camacho_2024_c4,
        'villegas_camacho_2024_c8': load_villegas_camacho_2024_c8,
    }

    result = {}
    for name in datasets:
        if name not in loader_map:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available datasets: {list(loader_map.keys())}"
            )
        result[name] = loader_map[name]()

    return result


def get_data_info() -> Dict[str, dict]:
    """
    Get information about available datasets.

    Returns
    -------
    dict
        Dictionary containing dataset information including
        existence status, file paths, and sizes.

    Examples
    --------
    >>> info = get_data_info()
    >>> for name, details in info.items():
    ...     print(f"{name}: exists={details['exists']}, size={details['size_mb']:.2f} MB")
    """
    info = {}

    for name, filename in _DATASETS.items():
        file_path = _DATA_DIR / filename
        exists = file_path.exists()

        info[name] = {
            'exists': exists,
            'path': str(file_path),
            'filename': filename,
            'size_mb': file_path.stat().st_size / (1024 * 1024) if exists else 0.0
        }

    return info


# Make functions available at package level
__all__ = [
    'load_jung_2018',
    'load_kedzierski_2019',
    'load_kedzierski_2019_u'
    'load_frond_2021',
    'load_villegas_camacho_2024_c4',
    'load_villegas_camacho_2024_c8',
    'load_all_datasets',
    'load_datasets',
    'get_data_info',
]
