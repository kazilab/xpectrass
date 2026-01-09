"""
Scatter Correction Module for FTIR Spectral Preprocessing
==========================================================

Provides multiplicative scatter correction (MSC), extended MSC (EMSC),
and related methods for correcting light scattering effects.

**IMPORTANT**: This module expects absorbance data (AU), not transmittance (%).
Convert transmittance to absorbance first using convert_spectra() from trans_abs.py

Features:
- Single spectrum correction via scatter_correction()
- Batch DataFrame processing via apply_scatter_correction()
- Automatic column detection and sorting by wavenumber
- Performance optimized for large datasets (vectorized operations)
- Pandas and Polars DataFrame support

Logging:
This module uses Python's logging module for warnings and informational messages.
Configure the logger to control output:

    import logging
    logging.getLogger('utils.scatter_correction').setLevel(logging.INFO)  # Show all messages
    logging.getLogger('utils.scatter_correction').setLevel(logging.ERROR)  # Only errors

Available Methods:
Run scatter_method_names() to see all available correction methods.
Common methods: msc, emsc, snv, snv_detrend
"""

from __future__ import annotations
from typing import Union, Tuple, Optional, List
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)

# Optional dependency: polars support is best-effort
try:
    import polars as pl  # type: ignore
except Exception:
    pl = None  # type: ignore

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)


def scatter_correction(
    intensities: Union[np.ndarray, list, tuple],
    wavenumbers: Optional[Union[np.ndarray, list, tuple]] = None,
    method: str = "msc",
    reference: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply scatter correction to a single FTIR spectrum.

    Parameters
    ----------
    intensities : array-like
        Raw intensity values (1-D). Absorbance data (AU), not transmittance (%).
    wavenumbers : array-like, optional
        X-axis values (wavenumbers in cm⁻¹). Ensures API consistency
        with other preprocessing modules (baseline, denoise, atmospheric).
        Not used in calculations but validates data integrity.
    method : str, default "msc"
        Correction method:
        - 'msc': Multiplicative Scatter Correction
        - 'emsc': Extended MSC (includes polynomial baseline terms)
        - 'snv': Standard Normal Variate (per-spectrum normalization)
        - 'snv_detrend': SNV followed by polynomial detrending
    reference : np.ndarray, optional
        Reference spectrum for MSC/EMSC. If None, cannot be applied
        (use apply_scatter_correction for batch processing with automatic reference).
        Must have same length as intensities.
    **kwargs : method-specific parameters
        emsc: poly_order (default 2)
        snv_detrend: detrend_order (default 1)

    Returns
    -------
    np.ndarray
        Scatter-corrected intensity values. NaN values in input are preserved at their
        original positions; correction is applied only to finite values.

    Raises
    ------
    ValueError
        If method requires reference spectrum but none provided, or if
        reference length doesn't match intensities.

    Notes
    -----
    NaN Handling:
        - If input contains NaN values, they are preserved in output
        - Correction is computed only on finite values
        - If all values are NaN, returns array of NaN

    Methods requiring reference (msc, emsc):
        - For single spectrum, reference must be provided explicitly
        - For batch processing, use apply_scatter_correction() which computes
          mean reference automatically
    """
    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("`intensities` must be a 1-D array-like object.")

    # Validate wavenumbers if provided
    if wavenumbers is not None:
        x = np.asarray(wavenumbers, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("`wavenumbers` must be a 1-D array-like object.")
        if len(x) != len(y):
            raise ValueError("`wavenumbers` and `intensities` must have the same length.")

        # Check if wavenumbers are monotonic (good practice)
        from .spectral_utils import _is_monotonic_strict
        if not _is_monotonic_strict(x):
            logger.warning(
                "Wavenumbers are not strictly monotonic. Scatter correction assumes "
                "uniform spacing or sorted data. Results may be unexpected."
            )

    # Handle NaN values: preserve positions but compute correction on finite values only
    nan_mask = ~np.isfinite(y)
    has_nans = np.any(nan_mask)

    if has_nans:
        # If all values are NaN, return NaN array
        if np.all(nan_mask):
            return np.full_like(y, np.nan)

        # Store original array and extract finite values
        y_original = y.copy()
        y_finite = y[~nan_mask]

        # Need at least 2 points for correction
        if len(y_finite) < 2:
            logger.warning(
                f"Insufficient finite data points ({len(y_finite)}) for scatter correction. "
                "Returning original spectrum with NaN preserved."
            )
            return y_original

        # Also filter reference if provided
        if reference is not None:
            ref_finite = reference[~nan_mask]
        else:
            ref_finite = None
    else:
        y_finite = y
        ref_finite = reference

    # Validate reference for methods that need it
    if method in ["msc", "emsc"]:
        if reference is None:
            raise ValueError(
                f"Method '{method}' requires a reference spectrum. "
                "Either provide 'reference' parameter, or use apply_scatter_correction() "
                "for batch processing with automatic reference calculation."
            )
        if len(reference) != len(y):
            raise ValueError(
                f"Reference spectrum length ({len(reference)}) must match "
                f"intensities length ({len(y)})."
            )

    # Apply correction to finite values
    try:
        if method == "msc":
            corrected_finite = _msc_single(y_finite, ref_finite, **kwargs)
        elif method == "emsc":
            corrected_finite = _emsc_single(y_finite, ref_finite, **kwargs)
        elif method == "snv":
            corrected_finite = _snv_single(y_finite, **kwargs)
        elif method == "snv_detrend":
            corrected_finite = _snv_detrend_single(y_finite, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                "Valid options: msc, emsc, snv, snv_detrend"
            )
    except Exception as e:
        if isinstance(e, ValueError) and "Unknown method" in str(e):
            raise
        raise RuntimeError(
            f"Scatter correction failed for method '{method}'. "
            f"Error: {type(e).__name__}: {str(e)}. "
            f"Check parameter compatibility with method documentation."
        ) from e

    # Restore NaN positions if needed
    if has_nans:
        result = np.full_like(y, np.nan)
        result[~nan_mask] = corrected_finite
        return result
    else:
        return corrected_finite


def scatter_method_names() -> List[str]:
    """Return list of available scatter correction method names."""
    return sorted(["msc", "emsc", "snv", "snv_detrend"])


# ---------------------------------------------------------------------------
#                    DATAFRAME-COMPATIBLE BATCH SCATTER CORRECTION
# ---------------------------------------------------------------------------

def apply_scatter_correction(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    method: str = "msc",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    reference: Optional[np.ndarray] = None,
    show_progress: bool = True,
    **kwargs
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """
    Apply scatter correction to a DataFrame of FTIR spectra (batch processing).

    Works with both pandas and polars DataFrames. Each row is a sample,
    numerical columns are wavenumbers. Applies scatter correction to all samples.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format DataFrame where rows = samples, columns = wavenumbers.
        Should contain numerical columns with spectral data and optional
        metadata columns (e.g., 'sample', 'label').
    method : str, default "msc"
        Scatter correction method. Options:
        - 'msc': Multiplicative Scatter Correction
        - 'emsc': Extended MSC (includes polynomial baseline terms)
        - 'snv': Standard Normal Variate (per-spectrum normalization)
        - 'snv_detrend': SNV followed by polynomial detrending
    label_column : str, default "label"
        Name of the label/group column to exclude from correction.
    exclude_columns : list[str], optional
        Additional column names to exclude from correction (e.g., 'sample', 'id').
    wn_min : float, optional
        Minimum wavenumber for column detection (default: 200.0 cm⁻¹).
        Columns with wavenumbers below this value will be excluded.
    wn_max : float, optional
        Maximum wavenumber for column detection (default: 8000.0 cm⁻¹).
        Columns with wavenumbers above this value will be excluded.
    reference : np.ndarray, optional
        Reference spectrum for MSC/EMSC. If None, uses mean of all spectra.
        Must match the length of spectral columns.
    show_progress : bool, default True
        If True, display a progress bar during processing.
    **kwargs : additional parameters
        Method-specific parameters:
        - emsc: poly_order (default 2)
        - snv_detrend: detrend_order (default 1)

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Scatter-corrected DataFrame (same type as input) with spectral data
        corrected and metadata columns preserved. Output columns are sorted
        by ascending wavenumber for standardization.

    NaN Handling
    ------------
    Robustly handles NaN (missing) values in spectral data:
    - NaN values are preserved in output at their original positions
    - Correction is computed only on finite values
    - If an entire spectrum is NaN, it remains as NaN
    - For MSC/EMSC, reference spectrum is computed from finite values only

    Performance
    -----------
    Optimized for large datasets using:
    - Robust wavenumber column detection (parses column names, not dtype)
    - Automatic column sorting to ensure monotonic wavenumber order
    - Vectorized numpy array access (no DataFrame.loc overhead)
    - Pre-allocated output arrays (no dynamic list appending)
    - Progress tracking via tqdm

    Examples
    --------
    >>> # Apply MSC scatter correction to all samples
    >>> df_corrected = apply_scatter_correction(df_wide, method="msc")

    >>> # Use EMSC with custom polynomial order
    >>> df_corrected = apply_scatter_correction(
    ...     df_wide,
    ...     method="emsc",
    ...     poly_order=3
    ... )

    >>> # Use SNV (no reference needed)
    >>> df_corrected = apply_scatter_correction(df_wide, method="snv")

    >>> # Works with both pandas and polars
    >>> df_pd_corrected = apply_scatter_correction(df_pandas)
    >>> df_pl_corrected = apply_scatter_correction(df_polars)

    >>> # Disable progress bar for cleaner output
    >>> df_corrected = apply_scatter_correction(df_wide, show_progress=False)
    """
    # Determine if input is polars or pandas
    is_polars = (pl is not None) and isinstance(data, pl.DataFrame)

    # Convert to pandas for processing
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Identify columns to exclude from correction
    if exclude_columns is None:
        exclude_columns = []
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    else:
        exclude_columns = list(exclude_columns)

    # Always exclude the label column if it exists
    if label_column in df.columns and label_column not in exclude_columns:
        exclude_columns.append(label_column)

    # Identify spectral columns by parsing column names as wavenumbers
    numeric_cols, wavenumbers = _infer_spectral_columns(df, exclude_columns, wn_min, wn_max)
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(numeric_cols, wavenumbers)

    # Warn if columns will be reordered
    if not np.array_equal(sort_idx, np.arange(len(sort_idx))):
        logger.warning(
            "Spectral columns are not in ascending wavenumber order. "
            "Output DataFrame will have columns sorted by ascending wavenumber for standardization."
        )

    # OPTIMIZATION: Extract numpy array and pre-allocate result
    spectral_data = df[sorted_cols].values.astype(np.float64)
    n_samples = spectral_data.shape[0]
    n_wavenumbers = spectral_data.shape[1]

    # VALIDATION: Check if data appears to be transmittance instead of absorbance
    sample_size = min(100, n_samples)
    sample_data = spectral_data[:sample_size, :].flatten()
    sample_data_finite = sample_data[np.isfinite(sample_data)]

    if len(sample_data_finite) > 0:
        median_val = np.median(sample_data_finite)
        p95_val = np.percentile(sample_data_finite, 95)

        if p95_val > 10.0 and median_val > 1.0:
            raise ValueError(
                f"Input data appears to be transmittance (%) rather than absorbance (AU). "
                f"Detected: median={median_val:.2f}, 95th percentile={p95_val:.2f}. "
                f"Scatter correction should be performed on absorbance for physical validity. "
                f"Please convert your data first using: "
                f"convert_spectra(data, mode='to_absorbance') from trans_abs.py"
            )

    # Compute reference spectrum if needed and not provided
    if method in ["msc", "emsc"] and reference is None:
        # Use mean of all finite values at each wavenumber
        reference = np.nanmean(spectral_data, axis=0)
        logger.info(f"Computed reference spectrum as mean of {n_samples} samples (NaN-aware).")

    corrected_data = np.empty((n_samples, n_wavenumbers), dtype=np.float64)

    # Apply scatter correction to each sample with progress bar
    iterator = tqdm(
        range(n_samples),
        desc=f"Scatter correction ({method})",
        disable=not show_progress,
        dynamic_ncols=True
    )

    for i in iterator:
        intensities = spectral_data[i, :]

        # Apply scatter correction (pass reference for msc/emsc methods)
        corrected_data[i, :] = scatter_correction(
            intensities=intensities,
            wavenumbers=sorted_wavenumbers,
            method=method,
            reference=reference,
            **kwargs
        )

    # PHYSICAL CONSTRAINT VALIDATION: Check for negative absorbance after correction
    finite_mask = np.isfinite(corrected_data)
    if np.any(finite_mask):
        n_negative = np.sum(corrected_data[finite_mask] < 0)
        if n_negative > 0:
            min_negative = np.min(corrected_data[finite_mask])
            pct_negative = 100.0 * n_negative / np.sum(finite_mask)
            logger.warning(
                f"Scatter correction produced {n_negative} negative absorbance values "
                f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                f"This is physically invalid. "
                f"Recommendations: (1) Apply baseline correction before scatter correction, "
                f"(2) Try different scatter correction method (e.g., 'snv' instead of 'msc'), "
                f"or (3) Check that input data is absorbance, not transmittance."
            )

    # Reconstruct DataFrame with corrected spectral data
    df_corrected_data = pd.DataFrame(
        corrected_data,
        index=df.index,
        columns=sorted_cols
    )

    # Merge back with original metadata (columns not in sorted_cols)
    metadata_cols = [c for c in df.columns if c not in sorted_cols]
    if metadata_cols:
        df_final = pd.concat([df[metadata_cols], df_corrected_data], axis=1)
    else:
        df_final = df_corrected_data

    # Reorder columns to ensure metadata comes first
    final_cols = metadata_cols + sorted_cols
    df_final = df_final[final_cols]

    # Convert back to polars if input was polars
    if is_polars:
        df_final = pl.from_pandas(df_final)

    return df_final


# ---------------------------------------------------------------------------
#                           INDIVIDUAL METHODS
# ---------------------------------------------------------------------------

def _msc_single(
    spectrum: np.ndarray,
    reference: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Multiplicative Scatter Correction (MSC) for a single spectrum.

    Corrects for additive (baseline offset) and multiplicative
    (path length) scatter effects by regressing the spectrum
    against a reference spectrum.

    Model: spectrum = a + b * reference
    Corrected: (spectrum - a) / b

    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum (1-D array).
    reference : np.ndarray
        Reference spectrum (same length as spectrum).
    **kwargs : ignored
        For API consistency.

    Returns
    -------
    np.ndarray
        MSC-corrected spectrum.
    """
    # Fit linear regression: spectrum = a + b * reference
    # Using least squares: [1, ref] @ [a, b]^T = spectrum
    X = np.column_stack([np.ones_like(reference), reference])
    coeffs = np.linalg.lstsq(X, spectrum, rcond=None)[0]
    a, b = coeffs[0], coeffs[1]

    # Avoid division by zero
    if abs(b) < 1e-10:
        b = 1.0

    corrected = (spectrum - a) / b
    return corrected


def _emsc_single(
    spectrum: np.ndarray,
    reference: np.ndarray,
    poly_order: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Extended Multiplicative Scatter Correction (EMSC) for a single spectrum.

    Extends MSC by including polynomial baseline terms to handle
    more complex scatter patterns.

    Model: spectrum = a + b * reference + c1*x + c2*x² + ...

    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum (1-D array).
    reference : np.ndarray
        Reference spectrum (same length as spectrum).
    poly_order : int, default 2
        Order of polynomial baseline terms.
    **kwargs : ignored
        For API consistency.

    Returns
    -------
    np.ndarray
        EMSC-corrected spectrum.
    """
    n_points = len(spectrum)

    # Create normalized x values for polynomial
    x = np.linspace(-1, 1, n_points)

    # Build design matrix: [1, reference, x, x², ...]
    X = [np.ones(n_points), reference]
    for p in range(1, poly_order + 1):
        X.append(x ** p)
    X = np.column_stack(X)

    coeffs = np.linalg.lstsq(X, spectrum, rcond=None)[0]

    # Reconstruct baseline (polynomial terms only)
    baseline = coeffs[0]  # intercept
    for p in range(1, poly_order + 1):
        baseline += coeffs[2 + p - 1] * (x ** p)

    b = coeffs[1]  # multiplicative term
    if abs(b) < 1e-10:
        b = 1.0

    corrected = (spectrum - baseline) / b
    return corrected


def _snv_single(
    spectrum: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Apply SNV (Standard Normal Variate) to a single spectrum.

    SNV normalizes the spectrum to have mean=0 and std=1.

    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum (1-D array).
    **kwargs : ignored
        For API consistency.

    Returns
    -------
    np.ndarray
        SNV-normalized spectrum.
    """
    mean = np.mean(spectrum)
    std = np.std(spectrum)

    # Avoid division by zero
    if std < 1e-10:
        std = 1.0

    return (spectrum - mean) / std


def _snv_detrend_single(
    spectrum: np.ndarray,
    detrend_order: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Apply SNV followed by polynomial detrending to a single spectrum.

    Removes residual baseline slope after SNV correction.

    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum (1-D array).
    detrend_order : int, default 1
        Order of polynomial for detrending (1 = linear, 2 = quadratic, etc.).
    **kwargs : ignored
        For API consistency.

    Returns
    -------
    np.ndarray
        SNV-corrected and detrended spectrum.
    """
    # First apply SNV
    snv_spectrum = _snv_single(spectrum)

    # Then detrend
    n_points = len(snv_spectrum)
    x = np.arange(n_points)

    coeffs = np.polyfit(x, snv_spectrum, detrend_order)
    trend = np.polyval(coeffs, x)
    corrected = snv_spectrum - trend

    return corrected


# ---------------------------------------------------------------------------
#                           HELPER FUNCTIONS (Deprecated)
# ---------------------------------------------------------------------------

def msc_single(
    spectrum: np.ndarray,
    reference: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Apply MSC to a single spectrum and return coefficients.

    **Deprecated**: Use scatter_correction() with method='msc' instead.

    This function is retained for backward compatibility only.

    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum.
    reference : np.ndarray
        Reference spectrum.

    Returns
    -------
    corrected : np.ndarray
        Corrected spectrum.
    a : float
        Offset coefficient.
    b : float
        Scaling coefficient.
    """
    X = np.column_stack([np.ones_like(reference), reference])
    coeffs = np.linalg.lstsq(X, spectrum, rcond=None)[0]
    a, b = coeffs[0], coeffs[1]

    if abs(b) < 1e-10:
        b = 1.0

    corrected = (spectrum - a) / b
    return corrected, a, b
