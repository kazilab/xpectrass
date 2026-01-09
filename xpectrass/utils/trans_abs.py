import pandas as pd
import polars as pl
import numpy as np
from typing import Union, List, Optional

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)


def convert_spectra(
    data: Union[pd.DataFrame, pl.DataFrame],
    mode: str = "auto",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Convert FTIR spectral data between transmittance and absorbance.

    Each row represents a sample, and numerical columns represent wavenumbers/frequencies.
    Non-spectral columns (metadata) are preserved without conversion.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format DataFrame where rows = samples, columns = wavenumbers.
        Should contain numerical columns with spectral data and optional
        metadata columns (e.g., 'sample', 'label').
    mode : str, default "auto"
        Conversion direction:
        - "to_absorbance" or "t2a": Convert %Transmittance to Absorbance
        - "to_transmittance" or "a2t": Convert Absorbance to %Transmittance
        - "auto": Automatically detect data type and convert to the opposite format
    label_column : str, default "label"
        Name of the label/group column to exclude from conversion.
    exclude_columns : list[str], optional
        Additional column names to exclude from conversion (e.g., 'sample', 'id').
        If None, automatically excludes non-numeric columns.
    wn_min : float, optional
        Minimum wavenumber to include in conversion (filters columns before conversion).
        Use this to restrict which spectral data is processed (more efficient).
        If None, uses full spectrum range. Consistent with other modules
        (denoise.py, baseline.py, plotting.py, etc.).
    wn_max : float, optional
        Maximum wavenumber to include in conversion (filters columns before conversion).
        Use this to restrict which spectral data is processed (more efficient).
        If None, uses full spectrum range. Consistent with other modules
        (denoise.py, baseline.py, plotting.py, etc.).

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Converted DataFrame (same type as input) with spectral data converted
        and metadata columns preserved.

    Notes
    -----
    • Uses shared spectral utilities for robust column detection and sorting
    • Spectral columns are automatically sorted by wavenumber (ascending)
    • Handles unsorted or non-contiguous wavenumber columns correctly
    • **wn_min/wn_max**: Filter which columns to process (efficient, reduces data)
    • Converted data maintains sorted column order
    • Filters out non-spectral numeric columns (outside 200-8000 cm⁻¹ range)

    Conversion formulas:
    - Transmittance to Absorbance: A = -log₁₀(T/100)
    - Absorbance to Transmittance: T = 100 × 10^(-A)

    The function automatically:
    - Clips transmittance values to [1e-12, None] to avoid invalid log operations
    - Clips absorbance values to [-0.5, None] (only lower bound to prevent overflow)
    - Note: Negative absorbance values are preserved and should be handled by
      subsequent baseline correction steps

    Examples
    --------
    >>> # Convert transmittance to absorbance
    >>> df_abs = convert_spectra(df_trans, mode="to_absorbance")

    >>> # Convert absorbance to transmittance
    >>> df_trans = convert_spectra(df_abs, mode="to_transmittance")

    >>> # Specify additional columns to preserve
    >>> df_abs = convert_spectra(df_trans, mode="to_absorbance",
    ...                          exclude_columns=["sample", "label", "batch_id"])

    >>> # Auto-detect and convert
    >>> df_converted = convert_spectra(df_wide, mode="auto")

    >>> # Convert only specific wavenumber range (efficient)
    >>> df_abs = convert_spectra(df_trans, mode="to_absorbance", wn_min=1000, wn_max=2000)
    """
    # Normalize mode string
    mode = mode.lower().strip()

    # Handle auto mode - detect data type and convert accordingly
    if mode == "auto":
        detected_type = _detect_data_type(
            data,
            label_column=label_column,
            exclude_columns=exclude_columns,
            wn_min=wn_min,
            wn_max=wn_max
        )
        if detected_type == "transmittance":
            conversion_type = "to_absorbance"
            print(f"Auto-detected: Transmittance → Converting to Absorbance")
        else:
            conversion_type = "to_transmittance"
            print(f"Auto-detected: Absorbance → Converting to Transmittance")
    elif mode in ["to_absorbance", "t2a", "transmittance_to_absorbance", "trans_to_abs"]:
        conversion_type = "to_absorbance"
    elif mode in ["to_transmittance", "a2t", "absorbance_to_transmittance", "abs_to_trans"]:
        conversion_type = "to_transmittance"
    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Use 'to_absorbance', 'to_transmittance', or 'auto'."
        )

    # Determine if input is polars or pandas
    is_polars = isinstance(data, pl.DataFrame)

    # Convert to pandas for easier manipulation
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # --- Identify and sort spectral columns using shared utilities -----------
    # Build exclusion list
    exclude = [label_column] if label_column in df.columns else []
    if exclude_columns:
        if isinstance(exclude_columns, str):
            exclude.append(exclude_columns)
        else:
            exclude.extend(exclude_columns)

    # First, get ALL spectral columns (without filtering) to identify metadata
    all_spectral_cols, _ = _infer_spectral_columns(
        df,
        exclude_columns=exclude,
        wn_min=None,  # Get ALL spectral columns
        wn_max=None
    )

    # Identify true metadata columns (non-spectral columns)
    metadata_cols = [c for c in df.columns if c not in all_spectral_cols]

    # Now get filtered spectral columns (with wn_min/wn_max if provided)
    spectral_cols, wn_values = _infer_spectral_columns(
        df,
        exclude_columns=exclude,
        wn_min=wn_min,
        wn_max=wn_max
    )

    if len(spectral_cols) == 0:
        raise ValueError("No spectral columns found for conversion!")

    # Sort columns by wavenumber (ascending: 400 → 4000)
    spectral_cols_sorted, wn_sorted, _ = _sort_spectral_columns(
        spectral_cols, wn_values
    )

    # Extract spectral data as numpy array (in sorted order)
    spectra = df[spectral_cols_sorted].to_numpy()

    # Perform conversion
    if conversion_type == "to_absorbance":
        # A = -log10(T/100)
        # Check for negative or zero transmittance (physically invalid)
        n_negative = np.sum(spectra < 0)
        n_zero = np.sum(spectra == 0)
        if n_negative > 0 or n_zero > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Found {n_negative} negative and {n_zero} zero transmittance values. "
                f"These are physically invalid and will be clipped to 0.01% for conversion. "
                f"This indicates data quality issues in the input."
            )

        # Clip transmittance to avoid log(0) and handle negative values
        spectra_clipped = np.clip(spectra, 1e-12, None)
        converted_spectra = -np.log10(spectra_clipped / 100.0)
    else:  # to_transmittance
        # T = 100 × 10^(-A)
        # No clipping - allow negative absorbance values (will be handled by baseline correction)
        # Only clip maximum to prevent overflow
        spectra_clipped = np.clip(spectra, -0.5, None)
        converted_spectra = 100.0 * np.power(10, -spectra_clipped)

    # Create new dataframe with converted values
    # Build all columns at once to avoid fragmentation
    result_dict = {}

    # Add metadata columns first
    for col in metadata_cols:
        result_dict[col] = df[col].values

    # Add converted spectral columns in sorted order
    for i, col in enumerate(spectral_cols_sorted):
        result_dict[col] = converted_spectra[:, i]

    # Create dataframe from dictionary (much faster, no fragmentation)
    df_converted = pd.DataFrame(result_dict, index=df.index)

    # Convert back to polars if input was polars
    if is_polars:
        df_converted = pl.from_pandas(df_converted)

    return df_converted


def _detect_data_type(
    data: Union[pd.DataFrame, pl.DataFrame],
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    threshold: float = 10.0,
) -> str:
    """
    Detect whether spectral data is in transmittance or absorbance format.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        DataFrame with spectral data
    label_column : str, default "label"
        Name of the label column to exclude from analysis
    exclude_columns : list[str], optional
        Additional column names to exclude from analysis
    wn_min : float, optional
        Minimum wavenumber to consider for detection
    wn_max : float, optional
        Maximum wavenumber to consider for detection
    threshold : float, default 10.0
        Values above this threshold suggest transmittance data

    Returns
    -------
    str
        Either "transmittance" or "absorbance"

    Examples
    --------
    >>> data_type = _detect_data_type(df)
    >>> print(f"Data appears to be: {data_type}")
    """
    # Convert to pandas if needed
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    else:
        df = data

    # Build exclusion list
    exclude = [label_column] if label_column in df.columns else []
    if exclude_columns:
        if isinstance(exclude_columns, str):
            exclude.append(exclude_columns)
        else:
            exclude.extend(exclude_columns)

    # Use spectral_utils for robust column detection
    spectral_cols, wn_values = _infer_spectral_columns(
        df,
        exclude_columns=exclude,
        wn_min=wn_min,
        wn_max=wn_max
    )

    # Get max value from first sample
    if len(df) > 0 and len(spectral_cols) > 0:
        max_val = df[spectral_cols].iloc[0].max()

        if max_val > threshold:
            return "transmittance"
        else:
            return "absorbance"
    else:
        raise ValueError("No spectral data found to analyze!")


'''
# Example usage:

# Convert transmittance to absorbance
df_abs = convert_spectra(df_wide, mode="to_absorbance")

# Convert absorbance to transmittance
df_trans = convert_spectra(df_abs, mode="to_transmittance")

# Auto mode - automatically detect and convert
df_converted = convert_spectra(df_wide, mode="auto")
# Output: "Auto-detected: Transmittance → Converting to Absorbance"

# Convert with custom excluded columns
df_abs = convert_spectra(df_wide, mode="to_absorbance",
                         exclude_columns=["sample", "label", "batch_id"])

# Convert only specific wavenumber range (efficient)
df_abs = convert_spectra(df_trans, mode="to_absorbance", wn_min=1000, wn_max=2000)

# Combined: filter range + exclude metadata
df_abs = convert_spectra(
    df_trans,
    mode="to_absorbance",
    wn_min=500,
    wn_max=3500,
    exclude_columns=["temperature", "batch_id"]
)
'''
