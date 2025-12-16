from __future__ import annotations
from pathlib import Path
from typing import Iterable, Literal, Union
from pathlib import Path
import pandas as pd
import polars as pl
import itertools
from typing import Union, Optional
import tqdm


##############################################################################
# 1.  Import helpers (assumed to exist in your code base)                    #
##############################################################################
# - import_data(file)           → pandas  (single sample, index="wavenumber")
# - import_data_pl(file)        → polars  (same schema)
##############################################################################

def process_batch_files(
    files: Iterable[Union[str, Path]],
    skiprows: int = 15,
    separator: str = ',',
    engine: Literal["pd", "pl"] = "pl",
    concat_how: Literal["vertical", "vertical_relaxed"] = "vertical",
    keep_index: bool = True,
    show_progress: bool = True,
) -> pl.DataFrame:
    """
    Import a batch of FT-IR CSVs and concatenate them into one Polars frame.

    Parameters
    ----------
    files : iterable of str | Path
        Paths to the spectral CSV files.
    skiprows : int, default 15
        Number of rows to skip at the start of the file (e.g. metadata).
    separator : str, default ','
        Delimiter for the CSV file.
    engine : {'pd', 'pl'}, default 'pl'
        • 'pd'  → read via the pandas-based importer, then convert to Polars.  
        • 'pl'  → read directly via the Polars importer (faster).
    concat_how : {'vertical', 'vertical_relaxed'}, default 'vertical_relaxed'
        • 'vertical'          → schemas must match exactly; raises if not.  
        • 'vertical_relaxed'  → union by column name; missing cols filled with
                                 nulls (Polars ≥ 0.20).
    keep_index : bool, default True
        If True and *engine* is 'pd', include the pandas Index as a column
        when converting to Polars (`include_index=True`).  Recommended because
        your importer names the index "sample".
    show_progress : bool, default True
        Toggle the tqdm progress bar.

    Returns
    -------
    pl.DataFrame
        All spectra stacked row-wise (each row = one sample).
    """
    dfs: list[pl.DataFrame] = []

    iterator = tqdm.tqdm(files, desc="Importing", disable=not show_progress)
    for file in iterator:
        file = Path(file)                       # normalise
        if engine == "pd":
            df_pd: pd.DataFrame = import_data_pd(file, skiprows, separator)
            df_pl = pl.from_pandas(df_pd, include_index=keep_index)
        elif engine == "pl":
            df_pl = import_data_pl(file, skiprows, separator)
        else:
            raise ValueError("engine must be 'pd' or 'pl'")

        dfs.append(df_pl)

    if not dfs:
        raise ValueError("No DataFrames were created from the provided file list.")

    # Concatenate; 'vertical_relaxed' aligns by header
    try:
        final_df = pl.concat(dfs, how=concat_how, rechunk=True)
    except pl.ColumnNotFoundError as e:
        raise ValueError(
            "Column mis-match across files. "
            "Try concat_how='vertical_relaxed' or inspect individual schemas."
        ) from e

    return final_df


# import data

def import_data(
    file_path: Union[str, Path],
    engine: str = 'pl',
    skiprows: int = 15,
    separator: str = ','
):
    """
    Load a single‐sample CSV of spectral data, set the wavenumber index,
    transpose so samples are rows, and attach a simple sample label.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file to import.
    skiprows : int, default 15
        Number of rows to skip at the start of the file (e.g. metadata).
    index_col : int, default 0
        Which column to use as the DataFrame’s index (the wavenumber axis).
    separator : str, default ','
        Delimiter for the CSV file.
    engine: str, default 'pd', or 'pl'

    Returns
    -------
    pd.DataFrame pl.DataFrame
        Transposed DataFrame (`samples` × `wavenumber`) with:
        - Index name = "sample"
        - Column name = wavenumber values, index name = "wavenumber"
        - A `"label"` column containing the alphabetic prefix of the sample name
    """
    if engine == 'pd':
        df = import_data_pd(file_path, skiprows, separator)
    else:
        df = import_data_pl(file_path, skiprows, separator)
    return df

# Import data using Pandas

def import_data_pd(
    file_path: Union[str, Path],
    skiprows: int = 15,
    sep: str = ','
) -> pd.DataFrame:
    """
    Load a single‐sample CSV of spectral data, set the wavenumber index,
    transpose so samples are rows, and attach a simple sample label.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file to import.
    skiprows : int, default 15
        Number of rows to skip at the start of the file (e.g. metadata).
    index_col : int, default 0
        Which column to use as the DataFrame’s index (the wavenumber axis).
    sep : str, default ','
        Delimiter for the CSV file.

    Returns
    -------
    pd.DataFrame
        Transposed DataFrame (`samples` × `wavenumber`) with:
        - Index name = "sample"
        - Column name = wavenumber values, index name = "wavenumber"
        - A `"label"` column containing the alphabetic prefix of the sample name
    """
    # Ensure we have a Path object
    file_path = Path(file_path)

    # Read CSV and use first column as the wavenumber index
    df = pd.read_csv(
        file_path,
        skiprows=skiprows,
        index_col=0,
        header=None,
        sep=sep
    )
    df.index.name = "wavenumber"

    # Derive sample name from filename (stem = filename without suffix)
    sample_name = file_path.stem
    df.columns = [sample_name]

    # Transpose so each sample is a row
    df_transposed = df.T
    df_transposed.index.name = "sample"

    # Create a simple label: the prefix of letters before any digits
    letters = itertools.takewhile(lambda ch: not ch.isdigit(), sample_name)
    sample_label = ''.join(letters)
    df_transposed["label"] = sample_label

    return df_transposed


# Import data using Polars


def import_data_pl(
    file_path: Union[str, Path],
    skiprows: int = 15,
    sep: str = ","
) -> pl.DataFrame:
    """
    Load a single-sample spectral CSV into a Polars DataFrame, reshape it so that
    each sample is a row with wavenumbers as columns, and attach a simple sample label.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file.
    skiprows : int, default 15
        Number of lines to skip before reading data (e.g., metadata).
    sep : str, default ","
        Field delimiter for the CSV.

    Returns
    -------
    pl.DataFrame
        A wide DataFrame where:
          - Each row is a sample (row index = filename stem).
          - Columns are wavenumbers (as floats or ints).
          - A "label" column holds the alphabetic prefix of the sample name.
    """
    # Normalize path and extract sample name
    file_path = Path(file_path)
    sample_name = file_path.stem

    # Read the CSV file without header, and assign columns
    new_columns = ["wavenumber", sample_name]
    df = pl.read_csv(
        source=str(file_path),
        skip_rows=skiprows,
        has_header=False,
        separator=sep,
        new_columns=new_columns
    )

    # Add a "sample" column for pivoting
    df = df.with_columns([
        pl.lit(sample_name).alias("sample")
    ])

    # Pivot into wide form: one row per sample, columns are wavenumbers
    df_wide = df.pivot(
        values=sample_name,
        index="sample",
        columns="wavenumber",
    )

    # Derive the "label" (alphabetic prefix before digits) and append
    prefix = "".join(itertools.takewhile(lambda ch: not ch.isdigit(), sample_name))
    df_wide = df_wide.with_columns([
        pl.lit(prefix).alias("label")
    ])

    # Reorder columns so "sample" and "label" come first
    cols = df_wide.columns
    ordered = ["sample", "label"] + [c for c in cols if c not in ("sample", "label")]
    return df_wide.select(ordered)
