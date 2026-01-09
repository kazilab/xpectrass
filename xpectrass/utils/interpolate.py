"""
FT-IR spectral resampling: interpolate spectra from different instruments to a common wavenumber grid.

Designed for wide DataFrames where rows are samples, numeric columns are wavenumbers,
and non-numeric columns are metadata (sample name, group, etc.).

**CRITICAL: Data must be in ABSORBANCE (AU), not Transmittance (%)**

This module enforces absorbance-only interpolation for physical validity according to
Beer-Lambert law. Interpolating transmittance is mathematically incorrect due to the
non-linear relationship between transmittance and concentration.

Recommended Workflow:
    1. Import raw data (typically in Transmittance %)
    2. Convert to Absorbance using convert_spectra(mode="to_absorbance")
    3. Interpolate/resample the absorbance data using this module
    4. Apply downstream processing (denoise, baseline, normalize, etc.)

Example Pipeline:
    >>> from .trans_abs import convert_spectra
    >>> from .interpolate import resample_spectra
    >>>
    >>> # Step 1: Import (transmittance from instrument)
    >>> df_raw = import_data_pl("ftir_data.csv")
    >>>
    >>> # Step 2: Convert to absorbance (REQUIRED before interpolation)
    >>> df_abs = convert_spectra(df_raw, mode="to_absorbance")
    >>>
    >>> # Step 3: Interpolate absorbance data
    >>> df_resampled, wn_grid = resample_spectra(
    ...     df_abs, wn_min=400, wn_max=4000, resolution=2, method="pchip"
    ... )
    >>>
    >>> # Step 4: Continue with downstream processing
    >>> df_denoised = apply_denoising(df_resampled, method="savgol")

Features:
- Combine FT-IR data from different studies with different wavenumber grids
- Single spectrum interpolation via interpolate_spectrum()
- Batch DataFrame processing via resample_spectra()
- Multi-study data fusion via combine_datasets()
- Automatic column detection and sorting by wavenumber
- Performance optimized for large datasets (vectorized operations, parallel processing)
- Pandas and Polars DataFrame support
- Automatic validation: rejects transmittance data, warns about negative absorbance

Usage:
    # Single dataset resampling (absorbance data required)
    df_resampled, wn_grid = resample_spectra(df_abs, wn_min=400, wn_max=4000, resolution=2, method="pchip")

    # Combine multiple studies onto common grid (all must be absorbance)
    combined_df, wn_grid = combine_datasets([df1_abs, df2_abs, df3_abs], wn_min=400, wn_max=4000, resolution=2)

Interpolation methods:
    - "linear": Fast, no dependencies. Good for densely sampled spectra.
    - "pchip": Shape-preserving, no overshoot. **Recommended for FT-IR peaks** (avoids negative absorbance).
    - "akima": Smooth, handles outliers well. Good for noisy data.
    - "cubic": Cubic spline. Smooth but can oscillate near sharp peaks (may produce negative absorbance).

Physical Constraints:
    - Absorbance should be non-negative (A >= 0)
    - The module validates input data and warns if interpolation produces negative values
    - Use 'pchip' method to minimize overshoot artifacts

Logging:
This module uses Python's logging module for warnings and informational messages.
Configure the logger to control output:

    import logging
    logging.getLogger('utils.interpolate').setLevel(logging.INFO)  # Show all messages
    logging.getLogger('utils.interpolate').setLevel(logging.ERROR)  # Only errors
"""

from __future__ import annotations

from typing import Literal, Tuple, List, Optional, Union
import logging
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns,
    _is_monotonic_strict
)

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Optional dependency: polars support is best-effort
try:
    import polars as pl  # type: ignore
except Exception:
    pl = None  # type: ignore


InterpolationMethod = Literal["linear", "pchip", "akima", "cubic"]
DataMode = Literal["auto", "absorbance", "normalized"]


def make_wn_grid(
    wn_min: float = 400.0,
    wn_max: float = 4000.0,
    resolution: float = 2.0,
    descending: bool = True,
) -> np.ndarray:
    """
    Generate a uniform wavenumber grid.

    Parameters
    ----------
    wn_min : float
        Minimum wavenumber (cm⁻¹).
    wn_max : float
        Maximum wavenumber (cm⁻¹).
    resolution : float
        Grid spacing (cm⁻¹).
    descending : bool
        If True (FT-IR convention), grid runs from wn_max to wn_min.

    Returns
    -------
    np.ndarray
        1D array of wavenumber values.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    lo, hi = min(wn_min, wn_max), max(wn_min, wn_max)
    n_points = int(np.floor((hi - lo) / resolution)) + 1
    grid = lo + resolution * np.arange(n_points, dtype=float)

    return grid[::-1] if descending else grid


def _clean_spectrum(
    wn: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by wavenumber ascending, remove NaNs, and average duplicates."""
    wn = np.asarray(wn, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if wn.shape != y.shape:
        raise ValueError("wn and y must have the same length")

    # Remove non-finite values
    valid = np.isfinite(wn) & np.isfinite(y)
    wn, y = wn[valid], y[valid]

    if wn.size == 0:
        return wn, y

    # Sort ascending by wavenumber
    order = np.argsort(wn)
    wn, y = wn[order], y[order]

    # Handle duplicates by averaging
    if np.all(np.diff(wn) != 0):
        return wn, y

    unique_wn, indices, counts = np.unique(wn, return_index=True, return_counts=True)
    y_averaged = np.array([
        np.mean(y[idx:idx + count])
        for idx, count in zip(indices, counts)
    ])

    return unique_wn, y_averaged


def _get_interpolator(method: InterpolationMethod):
    """
    Return the appropriate interpolation function.
    
    Parameters
    ----------
    method : str
        One of "linear", "pchip", "akima", "cubic".
        
    Returns
    -------
    callable
        Function with signature (wn_source, y_source, wn_target) -> y_interpolated
    """
    if method == "linear":
        # Use numpy (no scipy needed)
        def linear_interp(wn_src, y_src, wn_target):
            return np.interp(wn_target, wn_src, y_src)
        return linear_interp
    
    # All other methods require scipy
    try:
        from scipy import interpolate as scipy_interp
    except ImportError as e:
        raise ImportError(
            f"SciPy is required for method='{method}'. "
            "Install with: pip install scipy, or use method='linear'."
        ) from e
    
    if method == "pchip":
        # Piecewise Cubic Hermite Interpolating Polynomial
        # - Shape-preserving (monotonic between data points)
        # - No overshoot or undershoot
        # - Excellent for spectral peaks
        def pchip_interp(wn_src, y_src, wn_target):
            interpolator = scipy_interp.PchipInterpolator(wn_src, y_src, extrapolate=False)
            return interpolator(wn_target)
        return pchip_interp
    
    if method == "akima":
        # Akima spline
        # - Smooth, avoids oscillations
        # - Less sensitive to outliers than cubic spline
        # - Good for noisy spectra
        def akima_interp(wn_src, y_src, wn_target):
            interpolator = scipy_interp.Akima1DInterpolator(wn_src, y_src)
            return interpolator(wn_target)
        return akima_interp
    
    if method == "cubic":
        # Cubic spline
        # - Very smooth (C2 continuous)
        # - Can oscillate/overshoot near sharp peaks
        # - Better for smooth baselines than sharp features
        def cubic_interp(wn_src, y_src, wn_target):
            interpolator = scipy_interp.CubicSpline(wn_src, y_src, extrapolate=False)
            return interpolator(wn_target)
        return cubic_interp
    
    raise ValueError(
        f"Unknown interpolation method: '{method}'. "
        "Choose from: 'linear', 'pchip', 'akima', 'cubic'."
    )


def interpolate_spectrum(
    wn: np.ndarray,
    absorbance: np.ndarray,
    wn_grid: np.ndarray,
    method: InterpolationMethod = "linear",
    data_mode: DataMode = "auto",
) -> np.ndarray:
    """
    Interpolate a single spectrum to a target wavenumber grid.

    **IMPORTANT**: This function expects absorbance data (AU), not transmittance (%).
    Interpolation should be performed on absorbance for physical validity according
    to Beer-Lambert law. If your data is in transmittance, convert it first using
    convert_spectra(mode="to_absorbance") from trans_abs.py.

    Parameters
    ----------
    wn : array-like
        Original wavenumber values.
    absorbance : array-like
        Original absorbance values (AU). Must be absorbance, not transmittance.
        Physical range: typically [0, ~3], though higher values are possible.
        If you have transmittance (%), convert first using convert_spectra().
    wn_grid : np.ndarray
        Target wavenumber grid.
    method : str
        Interpolation method: "linear", "pchip", "akima", or "cubic".
        - "linear": Fast, no scipy needed. Good for dense data.
        - "pchip": Shape-preserving, recommended for FT-IR peaks (no overshoot).
        - "akima": Smooth, handles outliers well.
        - "cubic": Cubic spline, very smooth but can overshoot (may produce negative values).
    data_mode : str, default "auto"
        Type of input data, controls validation:
        - "auto": Automatically detect if data looks like transmittance and raise error
        - "absorbance": Skip validation, assume data is absorbance (skips negative warnings)
        - "normalized": Skip all validation, for SNV/mean-centered/derivative data
                       (allows negative values without warnings)

    Returns
    -------
    np.ndarray
        Interpolated absorbance values aligned with wn_grid.
        Points outside the original range are set to NaN.

    Raises
    ------
    ValueError
        If data_mode="auto" and input data appears to be transmittance (%).

    Warnings
    --------
    - Warns if interpolation produces negative absorbance values (only in "absorbance" mode)
    - Warns if using cubic method with noisy data (can cause artifacts)

    Notes
    -----
    Data Mode Selection:
        - Use "auto" for raw absorbance data from instruments
        - Use "absorbance" to skip transmittance check but keep negative warnings
        - Use "normalized" for SNV, mean-centered, derivative, or any preprocessed data
          where negative values are expected

    Physical Constraints:
        - Absorbance should be non-negative (A >= 0)
        - Interpolation may produce small negative values in noisy data, especially
          with cubic splines. Use 'pchip' method to avoid overshoot.
        - Transmittance interpolation is mathematically incorrect due to non-linear
          Beer-Lambert relationship. Always interpolate absorbance, not transmittance.

    Examples
    --------
    >>> # Raw absorbance data (auto-detect transmittance)
    >>> interpolated = interpolate_spectrum(wn, absorbance, wn_grid, method="pchip")
    
    >>> # Normalized data (skip all validation, negative values OK)
    >>> interpolated = interpolate_spectrum(wn, snv_data, wn_grid, data_mode="normalized")
    """
    wn_grid = np.asarray(wn_grid, dtype=float).ravel()
    wn_clean, y_clean = _clean_spectrum(wn, absorbance)

    # Need at least 2 points to interpolate
    if wn_clean.size < 2:
        return np.full_like(wn_grid, np.nan)

    # VALIDATION: Check if data appears to be transmittance instead of absorbance
    # Only perform this check in "auto" mode
    if data_mode == "auto":
        y_finite = y_clean[np.isfinite(y_clean)]
        if len(y_finite) > 0:
            median_val = np.median(y_finite)
            p95_val = np.percentile(y_finite, 95)

            # Heuristic: if 95th percentile > 10.0 AND median > 1.0, likely transmittance (%)
            # Transmittance range: [0, 100]%, Absorbance range: typically [0, ~3] AU
            if p95_val > 10.0 and median_val > 1.0:
                raise ValueError(
                    f"Input data appears to be transmittance (%) rather than absorbance (AU). "
                    f"Detected: median={median_val:.2f}, 95th percentile={p95_val:.2f}. "
                    f"Interpolation must be performed on absorbance for physical validity. "
                    f"Please convert your data first using: "
                    f"convert_spectra(data, mode='to_absorbance') from trans_abs.py, "
                    f"or set data_mode='absorbance' if you're sure this is absorbance data."
                )

    # NUMERICAL STABILITY: Check for near-duplicate wavenumbers or extreme values
    # These can cause numerical instability in cubic splines
    if len(wn_clean) > 1:
        wn_diff = np.diff(wn_clean)
        min_diff = np.min(np.abs(wn_diff))

        # Check for near-duplicate wavenumbers (spacing < 0.001 cm⁻¹)
        if min_diff < 1e-3:
            logger.warning(
                f"Input wavenumbers have very small spacing (min={min_diff:.6f} cm⁻¹). "
                f"This may cause numerical instability with cubic/spline methods. "
                f"Consider using method='linear' or 'pchip' for more stable interpolation."
            )

        # Check for extreme absorbance values (only in auto/absorbance mode)
        if data_mode in ("auto", "absorbance"):
            y_finite = y_clean[np.isfinite(y_clean)]
            if len(y_finite) > 0:
                max_abs_val = np.max(np.abs(y_finite))
                if max_abs_val > 100:
                    logger.warning(
                        f"Input has extreme absorbance values (max={max_abs_val:.2f}). "
                        f"This is unusual and may indicate data quality issues or unit errors. "
                        f"Typical absorbance range: [0, ~3] AU. "
                        f"If this is normalized data, use data_mode='normalized' to suppress this warning."
                    )

    # Check for overlap between source and target
    src_min, src_max = wn_clean[0], wn_clean[-1]
    grid_min, grid_max = np.nanmin(wn_grid), np.nanmax(wn_grid)

    if src_max < grid_min or src_min > grid_max:
        return np.full_like(wn_grid, np.nan)

    # Get interpolation function
    interp_func = _get_interpolator(method)

    # CRITICAL FIX: np.interp and scipy interpolators expect ASCENDING target values
    # wn_grid may be descending (FTIR convention: 4000 → 400 cm⁻¹)
    # Solution: interpolate on sorted grid, then reorder to match original wn_grid order
    is_descending = len(wn_grid) > 1 and wn_grid[0] > wn_grid[-1]

    if is_descending:
        # Sort wn_grid to ascending, keep track of original order
        wn_grid_sorted_idx = np.argsort(wn_grid)
        wn_grid_sorted = wn_grid[wn_grid_sorted_idx]

        # Interpolate on sorted (ascending) grid
        interpolated_sorted = interp_func(wn_clean, y_clean, wn_grid_sorted)
        interpolated_sorted = np.asarray(interpolated_sorted, dtype=float)

        # Reorder back to match original descending wn_grid
        # Create inverse permutation to map sorted → original order
        inverse_idx = np.empty_like(wn_grid_sorted_idx)
        inverse_idx[wn_grid_sorted_idx] = np.arange(len(wn_grid_sorted_idx))
        interpolated = interpolated_sorted[inverse_idx]
    else:
        # wn_grid already ascending (or single point), interpolate directly
        interpolated = interp_func(wn_clean, y_clean, wn_grid)
        interpolated = np.asarray(interpolated, dtype=float)

    # Set out-of-range values to NaN (no extrapolation)
    out_of_range = (wn_grid < src_min) | (wn_grid > src_max)
    interpolated[out_of_range] = np.nan

    # PHYSICAL CONSTRAINT VALIDATION: Check for negative absorbance
    # Only warn in "auto" or "absorbance" mode (not for normalized data)
    # Negative absorbance is physically impossible (violates Beer-Lambert law)
    # Can occur with cubic interpolation in noisy data due to overshoot
    if data_mode in ("auto", "absorbance"):
        finite_mask = np.isfinite(interpolated)
        if np.any(finite_mask):
            n_negative = np.sum(interpolated[finite_mask] < 0)
            if n_negative > 0:
                min_negative = np.min(interpolated[finite_mask])
                pct_negative = 100.0 * n_negative / np.sum(finite_mask)
                logger.warning(
                    f"Interpolation produced {n_negative} negative absorbance values "
                    f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                    f"This is physically invalid and may indicate: "
                    f"(1) noisy input data, (2) overshoot from '{method}' interpolation, "
                    f"or (3) baseline drift. "
                    f"Recommendations: Use method='pchip' (shape-preserving, no overshoot), "
                    f"apply denoising before interpolation, or perform baseline correction first. "
                    f"If this is normalized data, use data_mode='normalized' to suppress this warning."
                )

    return interpolated


def resample_spectra(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    wn_min: float = 400.0,
    wn_max: float = 4000.0,
    resolution: float = 2.0,
    descending: bool = True,
    method: InterpolationMethod = "linear",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    show_progress: bool = True,
    n_jobs: int = 1,
    data_mode: DataMode = "auto",
) -> Tuple[Union[pd.DataFrame, "pl.DataFrame"], np.ndarray]:
    """
    Resample FT-IR spectra from a wide DataFrame to a common wavenumber grid.

    **IMPORTANT**: This function expects absorbance data (AU), not transmittance (%).
    If your data is in transmittance, convert it first using:
    convert_spectra(data, mode="to_absorbance") from trans_abs.py

    Automatically detects spectral columns (numeric column names) vs metadata columns
    (non-numeric column names like 'sample_name', 'group', etc.).

    Works with both pandas and polars DataFrames. Each row is a sample,
    numerical columns are wavenumbers. Applies resampling to all samples.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format DataFrame where rows are samples. Columns with numeric names
        are treated as wavenumbers; other columns are preserved as metadata.
    wn_min : float
        Minimum wavenumber for output grid (cm⁻¹).
    wn_max : float
        Maximum wavenumber for output grid (cm⁻¹).
    resolution : float
        Grid spacing (cm⁻¹). WARNING: Setting this finer than your input data
        resolution may amplify noise and create artificial detail. Match this
        to your lowest input resolution for best results.
    descending : bool
        If True, output columns run from wn_max to wn_min (FT-IR convention).
    method : str
        Interpolation method: "linear", "pchip", "akima", or "cubic".
        - "linear": Fast, no scipy needed. Good for densely sampled data.
        - "pchip": Shape-preserving, no overshoot. Recommended for FT-IR peaks.
        - "akima": Smooth, robust to outliers. Good for noisy data.
        - "cubic": Cubic spline. Smooth but can overshoot near sharp peaks.
    label_column : str, default "label"
        Name of the label/group column to exclude from resampling.
    exclude_columns : list[str], optional
        Additional column names to exclude from resampling (e.g., 'sample', 'id').
    show_progress : bool, default True
        If True, display a progress bar during processing.
    n_jobs : int, default 1
        Number of parallel jobs. -1 uses all CPU cores. 1 = no parallelization.
    data_mode : str, default "auto"
        Type of input data, controls validation:
        - "auto": Automatically detect transmittance and raise error if found
        - "absorbance": Skip transmittance check, warn on negative values
        - "normalized": Skip all validation (for SNV, derivatives, etc.)

    Returns
    -------
    tuple[pd.DataFrame | pl.DataFrame, np.ndarray]
        - DataFrame with metadata columns followed by resampled spectral columns
        - The wavenumber grid as a 1D array

    NaN Handling
    ------------
    Robustly handles NaN (missing) values in spectral data:
    - NaN values are preserved in output at same positions
    - Points outside the original wavenumber range are set to NaN (no extrapolation)
    - If entire spectrum is NaN, output remains NaN

    Performance
    -----------
    Optimized for large datasets using:
    - Robust wavenumber column detection (parses column names, not dtype)
    - Automatic column sorting to ensure monotonic wavenumber order
    - Vectorized numpy array access (no DataFrame.loc overhead)
    - Optional parallel processing with joblib (n_jobs > 1)
    - Progress tracking via tqdm

    Warnings
    --------
    - Warns if spectral columns are reordered during processing
    - Warns if wavenumber bounds are auto-expanded beyond defaults (200-8000 cm⁻¹)
    - Warns if resolution is finer than input data (risk of noise amplification)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'sample': ['A', 'B'],
    ...     'group': [1, 2],
    ...     '4000.0': [0.1, 0.2],
    ...     '3998.0': [0.15, 0.25],
    ...     '3996.0': [0.12, 0.22],
    ... })
    >>> df_out, grid = resample_spectra(df, wn_min=3990, wn_max=4000, resolution=2, method="pchip")
    """
    # Suppress specific scipy warnings that are expected during interpolation
    # (e.g., numerical warnings from cubic splines on edge cases)
    # We use context manager to avoid suppressing warnings globally
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')

    # Determine if input is polars or pandas (polars is optional)
    is_polars = (pl is not None) and isinstance(data, pl.DataFrame)

    # Convert to pandas for processing
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Generate target grid
    wn_grid = make_wn_grid(wn_min, wn_max, resolution, descending)

    # Identify columns to exclude from resampling
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
    numeric_cols, wavenumbers = _infer_spectral_columns(df, exclude_columns, wn_min=None, wn_max=None)
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(numeric_cols, wavenumbers)

    # Warn if columns will be reordered
    if not np.array_equal(sort_idx, np.arange(len(sort_idx))):
        logger.warning(
            "Spectral columns are not in ascending wavenumber order. "
            "Output DataFrame will have columns sorted by ascending wavenumber for standardization."
        )

    # Estimate input resolution and warn if output resolution is too fine
    if len(sorted_wavenumbers) > 1:
        input_spacing = np.diff(np.sort(sorted_wavenumbers))
        median_input_resolution = float(np.median(input_spacing))

        # Allow some tolerance (0.9x) to avoid warnings for minor rounding
        if resolution < 0.9 * median_input_resolution:
            logger.warning(
                f"Output resolution ({resolution:.2f} cm⁻¹) is finer than input data "
                f"(~{median_input_resolution:.2f} cm⁻¹). Over-interpolation may amplify noise "
                f"and create artificial detail. Consider using resolution={median_input_resolution:.2f} "
                f"or coarser to match your instrument's actual resolution."
            )

    # OPTIMIZATION: Extract numpy array and pre-allocate result
    spectral_data = df[sorted_cols].values.astype(np.float64)
    n_samples = spectral_data.shape[0]
    n_wavenumbers = len(wn_grid)

    # VALIDATION: Check if data appears to be transmittance instead of absorbance
    # Only perform in "auto" mode
    if data_mode == "auto":
        sample_size = min(100, n_samples)
        sample_data = spectral_data[:sample_size, :].flatten()
        sample_data_finite = sample_data[np.isfinite(sample_data)]

        if len(sample_data_finite) > 0:
            median_val = np.median(sample_data_finite)
            p95_val = np.percentile(sample_data_finite, 95)

            # Heuristic: if 95th percentile > 10.0 AND median > 1.0, likely transmittance (%)
            if p95_val > 10.0 and median_val > 1.0:
                raise ValueError(
                    f"Input data appears to be transmittance (%) rather than absorbance (AU). "
                    f"Detected: median={median_val:.2f}, 95th percentile={p95_val:.2f}. "
                    f"Interpolation must be performed on absorbance for physical validity. "
                    f"Please convert your data first using: "
                    f"convert_spectra(data, mode='to_absorbance') from trans_abs.py, "
                    f"or set data_mode='absorbance' or 'normalized' if this is intentional."
                )

    # Choose parallel or sequential processing
    # For per-sample calls:
    # - "auto" mode: transmittance already checked at batch level, skip it per-sample
    #   but still want negative value warnings, so use "absorbance"
    # - "absorbance" mode: no transmittance check, but want negative warnings
    # - "normalized" mode: no checks at all
    if data_mode == "normalized":
        per_sample_mode = "normalized"  # Skip all validation
    else:
        per_sample_mode = "absorbance"  # Skip transmittance check, keep negative warnings
    if n_jobs != 1 and n_samples > 1:
        # Parallel processing with joblib
        # Note: tqdm progress updates in multiprocessing are tricky because each worker
        # runs in a separate process. We use a simple approach here that updates on completion.
        def _interpolate_row(i):
            return interpolate_spectrum(
                sorted_wavenumbers, spectral_data[i, :], wn_grid, 
                method=method, data_mode=per_sample_mode
            )

        if show_progress:
            # Use tqdm with joblib - updates when each task completes
            # Not real-time due to multiprocessing, but better than nothing
            from tqdm.auto import tqdm as tqdm_auto
            results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
                joblib.delayed(_interpolate_row)(i)
                for i in tqdm_auto(range(n_samples), desc=f"Resampling ({method})", dynamic_ncols=True)
            )
            X_resampled = np.array(results)
        else:
            # No progress bar
            X_resampled = np.array(
                joblib.Parallel(n_jobs=n_jobs, backend="loky")(
                    joblib.delayed(_interpolate_row)(i) for i in range(n_samples)
                )
            )
    else:
        # Sequential processing with progress bar
        X_resampled = np.empty((n_samples, n_wavenumbers), dtype=np.float64)
        iterator = tqdm(
            range(n_samples),
            desc=f"Resampling ({method})",
            disable=not show_progress,
            dynamic_ncols=True
        )
        for i in iterator:
            X_resampled[i, :] = interpolate_spectrum(
                sorted_wavenumbers, spectral_data[i, :], wn_grid, 
                method=method, data_mode=per_sample_mode
            )

    # Build output DataFrame
    # Determine precision based on resolution to avoid rounding collisions
    # For high-resolution data (< 0.1 cm⁻¹), use more decimals
    if resolution < 0.1:
        precision = 6  # e.g., 4000.123456
    elif resolution < 1.0:
        precision = 5  # e.g., 4000.12345
    else:
        precision = 4  # e.g., 4000.1234 (default)

    spectral_colnames = [f"{wn:.{precision}f}" for wn in wn_grid]
    df_spectral = pd.DataFrame(X_resampled, columns=spectral_colnames, index=df.index)

    # Merge back with original metadata (columns not in sorted_cols)
    metadata_cols = [c for c in df.columns if c not in sorted_cols]
    if metadata_cols:
        df_final = pd.concat([df[metadata_cols], df_spectral], axis=1)
    else:
        df_final = df_spectral

    # Reorder columns to ensure metadata comes first
    final_cols = metadata_cols + spectral_colnames
    df_final = df_final[final_cols]

    # Convert back to polars if input was polars
    if is_polars:
        df_final = pl.from_pandas(df_final)

    return df_final, wn_grid


def combine_datasets(
    datasets: List[Union[pd.DataFrame, "pl.DataFrame"]],
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    resolution: float = 2.0,
    descending: bool = True,
    method: InterpolationMethod = "pchip",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    add_study_column: bool = True,
    study_names: Optional[List[str]] = None,
    show_progress: bool = True,
    n_jobs: int = 1,
    # =========== NEW PARAMETERS TO PREVENT SAMPLE LOSS ===========
    grid_mode: Literal["intersection", "union", "custom"] = "intersection",
    nan_threshold: float = 0.5,
    drop_nan_samples: bool = False,
    report_coverage: bool = True,
    data_mode: DataMode = "auto",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Combine FT-IR data from multiple studies onto a common wavenumber grid.

    **IMPORTANT**: This function expects absorbance data (AU), not transmittance (%).
    All datasets must be in absorbance before combining. If any dataset is in
    transmittance, convert it first using:
    convert_spectra(data, mode="to_absorbance") from trans_abs.py

    This function is designed for meta-analysis and data integration across studies
    with different instruments or measurement conditions that produce different
    wavenumber grids.

    Parameters
    ----------
    datasets : list of DataFrames
        List of DataFrames to combine. Each DataFrame should be in wide format
        with rows as samples and numeric column names as wavenumbers.
    wn_min : float, optional
        Minimum wavenumber for common grid. If None, determined by grid_mode.
    wn_max : float, optional
        Maximum wavenumber for common grid. If None, determined by grid_mode.
    resolution : float, default 2.0
        Grid spacing for common wavenumber grid (cm⁻¹).
    descending : bool, default True
        If True, output columns run from wn_max to wn_min (FT-IR convention).
    method : str, default "pchip"
        Interpolation method: "linear", "pchip", "akima", or "cubic".
        - "pchip": Recommended for multi-study combination (shape-preserving)
        - "linear": Faster but may lose peak details
        - "akima": Good for noisy data from diverse sources
        - "cubic": Smooth but can introduce artifacts
    label_column : str, default "label"
        Name of the label/group column to preserve.
    exclude_columns : list[str], optional
        Additional column names to exclude from interpolation.
    add_study_column : bool, default True
        If True, adds a 'study' column indicating source dataset.
    study_names : list[str], optional
        Names for each study. If None, uses "Study_1", "Study_2", etc.
        Must match length of datasets if provided.
    show_progress : bool, default True
        If True, display progress bars during processing.
    n_jobs : int, default 1
        Number of parallel jobs for interpolation. -1 uses all CPU cores.
    
    grid_mode : str, default "intersection"
        **NEW** How to determine the common wavenumber range:
        - "intersection": Use only overlapping range (conservative, no extrapolation)
        - "union": Use the full range of all datasets (may have NaN at edges)
        - "custom": Use provided wn_min/wn_max (must specify both)
    nan_threshold : float, default 0.5
        **NEW** Fraction of NaN values allowed per sample (0.0-1.0).
        Samples with more NaN than this threshold will be flagged.
    drop_nan_samples : bool, default False
        **NEW** If True, drop samples exceeding nan_threshold. If False, keep all
        samples but warn about quality issues.
    report_coverage : bool, default True
        **NEW** If True, print detailed coverage report showing how many samples
        have good/poor coverage.
    data_mode : str, default "auto"
        Type of input data, controls validation:
        - "auto": Automatically detect transmittance and raise error if found
        - "absorbance": Skip transmittance check, warn on negative values
        - "normalized": Skip all validation (for SNV, derivatives, mean-centered data)

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        - Combined DataFrame with all samples on common wavenumber grid
        - The common wavenumber grid as a 1D array

    Notes
    -----
    Common Grid Selection:
        - **Intersection approach** (default, grid_mode="intersection"):
          Uses only the overlapping wavenumber range across all datasets.

          **Advantages:**
            • Scientifically conservative: all interpolated values are within
              measured ranges (no extrapolation)
            • Minimizes interpolation errors
            • Ensures all output values are based on real measurements

          **Disadvantages:**
            • Discards non-overlapping regions that may contain valuable features
            • Can result in significant data loss if datasets have poor overlap

          **Example:** Three datasets covering 600-4000, 500-3800, 700-3900 cm⁻¹
          will produce a common grid of 700-3800 cm⁻¹ (intersection).

        - **Union approach** (grid_mode="union"):
          Uses the full range from all datasets (min of minimums, max of maximums).
          Samples without coverage in some regions will have NaN values there.

          **Advantages:**
            • Preserves all spectral regions
            • No data loss from range restriction

          **Disadvantages:**
            • May produce NaN-heavy outputs if overlap is small
            • Requires downstream NaN handling

        - **Custom approach** (grid_mode="custom"):
          Uses explicit wn_min/wn_max. Useful for matching a specific target grid.

    Resolution Choice:
        - Default is 2 cm⁻¹, but you should match the LOWEST input resolution
          to avoid over-interpolation
        - Over-interpolation (using finer resolution than input data) can:
          • Amplify noise
          • Create artificial spectral detail
          • Give false impression of higher precision
        - Use auto_detect_common_grid() to get recommended resolution based
          on median spacing across all datasets
        - The function will warn if resolution is finer than input data

    Performance:
        - Processes each dataset independently with optional parallelization
        - Efficient memory usage with pre-allocated arrays
        - Progress tracking for each dataset

    Examples
    --------
    >>> # Combine 3 studies - use union to preserve all samples
    >>> combined, grid = combine_datasets(
    ...     [df1, df2, df3],
    ...     grid_mode="union",          # Don't lose samples to narrow intersection
    ...     nan_threshold=0.3,          # Flag samples with >30% NaN
    ...     report_coverage=True        # Show coverage statistics
    ... )
    
    >>> # For normalized data (SNV, mean-centered, derivatives)
    >>> combined, grid = combine_datasets(
    ...     [df1_snv, df2_snv, df3_snv],
    ...     grid_mode="union",
    ...     data_mode="normalized",     # Skip transmittance check, allow negatives
    ... )
    
    >>> # Conservative intersection (original behavior)
    >>> combined, grid = combine_datasets(
    ...     [df1, df2, df3],
    ...     grid_mode="intersection",
    ...     drop_nan_samples=True       # Remove low-coverage samples
    ... )
    """
    if not datasets:
        raise ValueError("datasets list is empty")

    if study_names is not None and len(study_names) != len(datasets):
        raise ValueError(
            f"study_names length ({len(study_names)}) must match datasets length ({len(datasets)})"
        )

    if grid_mode == "custom" and (wn_min is None or wn_max is None):
        raise ValueError(
            "grid_mode='custom' requires both wn_min and wn_max to be specified"
        )

    # Convert all datasets to pandas
    dfs = []
    for i, ds in enumerate(datasets):
        is_polars = (pl is not None) and isinstance(ds, pl.DataFrame)
        dfs.append(ds.to_pandas() if is_polars else ds.copy())

    # Analyze wavenumber ranges across all datasets
    all_wn_min = []
    all_wn_max = []
    dataset_wn_ranges = []  # Store per-dataset ranges for coverage analysis
    dataset_sample_counts = []

    for i, df in enumerate(dfs):
        # Identify spectral columns
        exc_cols = list(exclude_columns) if exclude_columns else []
        if label_column in df.columns and label_column not in exc_cols:
            exc_cols.append(label_column)

        try:
            numeric_cols, wn = _infer_spectral_columns(df, exc_cols, wn_min=None, wn_max=None)
            if len(wn) > 0:
                ds_min = float(np.min(wn))
                ds_max = float(np.max(wn))
                all_wn_min.append(ds_min)
                all_wn_max.append(ds_max)
                dataset_wn_ranges.append((ds_min, ds_max))
                dataset_sample_counts.append(len(df))
            else:
                raise ValueError(f"Dataset {i} has no spectral columns")
        except Exception as e:
            raise ValueError(f"Error processing dataset {i}: {e}") from e

    # Determine common wavenumber range based on grid_mode
    if grid_mode == "intersection" or (wn_min is None and wn_max is None and grid_mode != "union"):
        # Intersection: use maximum of minimums and minimum of maximums
        auto_wn_min = max(all_wn_min)
        auto_wn_max = min(all_wn_max)

        if auto_wn_min >= auto_wn_max:
            raise ValueError(
                f"No overlapping wavenumber range across datasets. "
                f"Dataset ranges: min={all_wn_min}, max={all_wn_max}. "
                f"Consider using grid_mode='union' or specifying wn_min and wn_max manually."
            )
        
        if wn_min is None:
            wn_min = auto_wn_min
        if wn_max is None:
            wn_max = auto_wn_max

        logger.info(
            f"Using INTERSECTION range: {wn_min:.2f}-{wn_max:.2f} cm⁻¹ "
            f"(from {len(datasets)} datasets)"
        )

    elif grid_mode == "union":
        # Union: use full range
        auto_wn_min = min(all_wn_min)
        auto_wn_max = max(all_wn_max)
        
        if wn_min is None:
            wn_min = auto_wn_min
        if wn_max is None:
            wn_max = auto_wn_max

        logger.info(
            f"Using UNION range: {wn_min:.2f}-{wn_max:.2f} cm⁻¹ "
            f"(from {len(datasets)} datasets)"
        )

    # For custom mode, wn_min and wn_max are already set

    # Calculate coverage statistics for each dataset
    if report_coverage:
        print("\n" + "=" * 70)
        print("DATASET COVERAGE ANALYSIS")
        print("=" * 70)
        print(f"Target grid: {wn_min:.1f} - {wn_max:.1f} cm⁻¹ ({wn_max - wn_min:.1f} cm⁻¹ range)")
        print(f"Grid mode: {grid_mode}")
        print("-" * 70)
        
        total_samples = 0
        samples_with_full_coverage = 0
        
        for i, ((ds_min, ds_max), n_samples) in enumerate(zip(dataset_wn_ranges, dataset_sample_counts)):
            study_name = study_names[i] if study_names else f"Dataset {i+1}"
            
            # Calculate coverage
            overlap_min = max(wn_min, ds_min)
            overlap_max = min(wn_max, ds_max)
            
            if overlap_min < overlap_max:
                coverage = (overlap_max - overlap_min) / (wn_max - wn_min)
            else:
                coverage = 0.0
            
            coverage_status = "✓ FULL" if coverage >= 0.99 else f"⚠ {coverage*100:.1f}%"
            
            print(f"  {study_name}: {n_samples} samples, "
                  f"range {ds_min:.1f}-{ds_max:.1f} cm⁻¹, "
                  f"coverage: {coverage_status}")
            
            total_samples += n_samples
            if coverage >= 0.99:
                samples_with_full_coverage += n_samples
        
        print("-" * 70)
        print(f"Total: {total_samples} samples, "
              f"{samples_with_full_coverage} with full coverage "
              f"({100*samples_with_full_coverage/total_samples:.1f}%)")
        print("=" * 70 + "\n")

    # Warn if intersection approach discards significant data
    union_range = max(all_wn_max) - min(all_wn_min)
    target_range = wn_max - wn_min
    data_retention = target_range / union_range if union_range > 0 else 1.0

    if data_retention < 0.7 and grid_mode == "intersection":
        logger.warning(
            f"Intersection approach is discarding {(1-data_retention)*100:.1f}% of total spectral range. "
            f"Union range: {min(all_wn_min):.0f}-{max(all_wn_max):.0f} cm⁻¹ ({union_range:.0f} cm⁻¹), "
            f"Target range: {wn_min:.0f}-{wn_max:.0f} cm⁻¹ ({target_range:.0f} cm⁻¹). "
            f"Consider using grid_mode='union' to preserve all spectral regions."
        )

    # Generate common grid
    common_grid = make_wn_grid(wn_min, wn_max, resolution, descending)

    # Validate resolution against input data (analyze all datasets)
    all_resolutions = []
    for i, df in enumerate(dfs):
        exc_cols = list(exclude_columns) if exclude_columns else []
        if label_column in df.columns and label_column not in exc_cols:
            exc_cols.append(label_column)

        try:
            _, wn = _infer_spectral_columns(df, exc_cols, wn_min=None, wn_max=None)
            if len(wn) > 1:
                sorted_wn = np.sort(wn)
                spacing = np.diff(sorted_wn)
                all_resolutions.append(float(np.median(spacing)))
        except Exception:
            pass  # Skip if we can't determine resolution

    if all_resolutions:
        coarsest_input_resolution = max(all_resolutions)  # Worst-case (coarsest)

        # Warn if output resolution is finer than the coarsest input
        if resolution < 0.9 * coarsest_input_resolution:
            logger.warning(
                f"Output resolution ({resolution:.2f} cm⁻¹) is finer than some input datasets "
                f"(coarsest: {coarsest_input_resolution:.2f} cm⁻¹). Over-interpolation may amplify "
                f"noise. Recommended: use resolution={coarsest_input_resolution:.2f} cm⁻¹ or coarser "
                f"to match the lowest-quality input data."
            )

    # Process each dataset
    resampled_dfs = []
    for i, df in enumerate(dfs):
        study_name = study_names[i] if study_names else f"Study_{i+1}"

        if show_progress:
            logger.info(f"Processing {study_name} ({len(df)} samples)...")

        # Resample to common grid
        df_resampled, _ = resample_spectra(
            data=df,
            wn_min=wn_min,
            wn_max=wn_max,
            resolution=resolution,
            descending=descending,
            method=method,
            label_column=label_column,
            exclude_columns=exclude_columns,
            show_progress=show_progress,
            n_jobs=n_jobs,
            data_mode=data_mode,
        )

        # Add study identifier if requested
        if add_study_column:
            df_resampled.insert(0, 'study', study_name)

        resampled_dfs.append(df_resampled)

    # Concatenate all resampled datasets
    combined_df = pd.concat(resampled_dfs, axis=0, ignore_index=True)

    # Analyze NaN coverage in output
    # Identify spectral columns in combined output
    spectral_cols = [c for c in combined_df.columns if c not in 
                     (list(exclude_columns or []) + [label_column, 'study'])]
    try:
        # Filter to only numeric column names (wavenumbers)
        spectral_cols = [c for c in spectral_cols if _is_numeric_string(c)]
    except:
        pass

    if spectral_cols:
        nan_fractions = combined_df[spectral_cols].isna().mean(axis=1)
        
        # Report samples with high NaN fractions
        high_nan_mask = nan_fractions > nan_threshold
        n_high_nan = high_nan_mask.sum()
        
        if n_high_nan > 0 and report_coverage:
            print("\n" + "=" * 70)
            print("SAMPLE COVERAGE REPORT")
            print("=" * 70)
            print(f"Samples with >{nan_threshold*100:.0f}% NaN values: {n_high_nan} / {len(combined_df)}")
            
            # Group by study
            if 'study' in combined_df.columns:
                problem_samples = combined_df.loc[high_nan_mask, 'study'].value_counts()
                print("\nProblem samples by study:")
                for study, count in problem_samples.items():
                    print(f"  {study}: {count} samples")
            
            print("=" * 70 + "\n")
            
            if drop_nan_samples:
                original_count = len(combined_df)
                combined_df = combined_df.loc[~high_nan_mask].reset_index(drop=True)
                logger.warning(
                    f"Dropped {n_high_nan} samples with >{nan_threshold*100:.0f}% NaN values. "
                    f"Remaining: {len(combined_df)} / {original_count} samples."
                )
            else:
                logger.warning(
                    f"{n_high_nan} samples have >{nan_threshold*100:.0f}% NaN values. "
                    f"Set drop_nan_samples=True to remove them, or use grid_mode='intersection' "
                    f"with appropriate wn_min/wn_max to ensure better coverage."
                )

    logger.info(
        f"Combined {len(datasets)} datasets: {len(combined_df)} total samples "
        f"on {len(common_grid)}-point grid ({wn_min:.2f}-{wn_max:.2f} cm⁻¹)"
    )

    return combined_df, common_grid


def _is_numeric_string(s: str) -> bool:
    """Check if a string can be parsed as a number (wavenumber)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def auto_detect_common_grid(
    datasets: List[Union[pd.DataFrame, "pl.DataFrame"]],
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    grid_mode: Literal["intersection", "union"] = "intersection",
) -> Tuple[float, float, int]:
    """
    Analyze multiple datasets to recommend a common wavenumber grid.

    Useful for planning multi-study integration before actual resampling.

    Parameters
    ----------
    datasets : list of DataFrames
        List of DataFrames to analyze
    label_column : str, default "label"
        Name of the label column to exclude from analysis
    exclude_columns : list[str], optional
        Additional columns to exclude from analysis
    grid_mode : str, default "intersection"
        **NEW** How to determine the range:
        - "intersection": Overlapping range only (conservative)
        - "union": Full range from all datasets

    Returns
    -------
    tuple[float, float, int]
        - Recommended minimum wavenumber
        - Recommended maximum wavenumber  
        - Number of points in common grid at 1 cm⁻¹ resolution

    Examples
    --------
    >>> wn_min, wn_max, n_points = auto_detect_common_grid([df1, df2, df3])
    >>> print(f"Recommended range: {wn_min:.1f}-{wn_max:.1f} cm⁻¹ ({n_points} points)")
    """
    if not datasets:
        raise ValueError("datasets list is empty")

    # Convert all datasets to pandas
    dfs = []
    for ds in datasets:
        is_polars = (pl is not None) and isinstance(ds, pl.DataFrame)
        dfs.append(ds.to_pandas() if is_polars else ds.copy())

    # Analyze wavenumber ranges
    all_wn_min = []
    all_wn_max = []
    all_resolutions = []

    for i, df in enumerate(dfs):
        # Identify spectral columns
        exc_cols = list(exclude_columns) if exclude_columns else []
        if label_column in df.columns and label_column not in exc_cols:
            exc_cols.append(label_column)

        try:
            numeric_cols, wn = _infer_spectral_columns(df, exc_cols, wn_min=None, wn_max=None)
            if len(wn) > 1:
                all_wn_min.append(float(np.min(wn)))
                all_wn_max.append(float(np.max(wn)))
                # Estimate resolution from median spacing
                sorted_wn = np.sort(wn)
                spacing = np.diff(sorted_wn)
                all_resolutions.append(float(np.median(spacing)))
        except Exception as e:
            logger.warning(f"Could not analyze dataset {i}: {e}")

    if not all_wn_min:
        raise ValueError("No valid spectral data found in any dataset")

    # Determine range based on grid_mode
    if grid_mode == "intersection":
        common_wn_min = max(all_wn_min)
        common_wn_max = min(all_wn_max)

        if common_wn_min >= common_wn_max:
            raise ValueError(
                f"No overlapping wavenumber range. Dataset ranges: "
                f"min={all_wn_min}, max={all_wn_max}. "
                f"Try grid_mode='union' instead."
            )
    else:  # union
        common_wn_min = min(all_wn_min)
        common_wn_max = max(all_wn_max)

    # Recommend resolution (use finest resolution for best quality)
    recommended_resolution = min(all_resolutions)
    n_points = int((common_wn_max - common_wn_min) / recommended_resolution) + 1

    logger.info(
        f"Dataset analysis ({len(dfs)} datasets):\n"
        f"  Individual ranges: {[f'{mn:.1f}-{mx:.1f}' for mn, mx in zip(all_wn_min, all_wn_max)]}\n"
        f"  Common range ({grid_mode}): {common_wn_min:.1f}-{common_wn_max:.1f} cm⁻¹\n"
        f"  Resolutions: {[f'{r:.2f}' for r in all_resolutions]} cm⁻¹\n"
        f"  Recommended: {recommended_resolution:.2f} cm⁻¹ ({n_points} points)"
    )

    return common_wn_min, common_wn_max, n_points


def get_spectral_matrix(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract the spectral matrix and wavenumber axis from a resampled DataFrame.

    Compatible with output from resample_spectra() and combine_datasets().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spectral data in wide format.
    exclude_columns : list[str], optional
        Column names to exclude from spectral matrix (treated as metadata).

    Returns
    -------
    tuple
        - X: 2D array of shape (n_samples, n_wavenumbers)
        - wn: 1D array of wavenumber values
        - meta: DataFrame containing only metadata columns

    Examples
    --------
    >>> df_resampled, grid = resample_spectra(df, wn_min=400, wn_max=4000, resolution=2)
    >>> X, wn, metadata = get_spectral_matrix(df_resampled)
    >>> print(X.shape)  # (n_samples, n_wavenumbers)
    """
    # Identify spectral columns
    exc_cols = list(exclude_columns) if exclude_columns else []
    numeric_cols, wavenumbers = _infer_spectral_columns(df, exc_cols, wn_min=None, wn_max=None)
    sorted_cols, sorted_wavenumbers, _ = _sort_spectral_columns(numeric_cols, wavenumbers)

    # Extract spectral matrix and metadata
    X = df[sorted_cols].to_numpy(dtype=float)
    metadata_cols = [c for c in df.columns if c not in sorted_cols]
    meta = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

    return X, sorted_wavenumbers, meta


def validate_combined_dataset(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    nan_threshold: float = 0.1,
) -> dict:
    """
    Validate a combined dataset and report quality metrics.
    
    **NEW FUNCTION** - Use this to diagnose sample loss issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from combine_datasets()
    exclude_columns : list[str], optional
        Columns to exclude from analysis
    nan_threshold : float, default 0.1
        Threshold for flagging samples with too many NaN values
        
    Returns
    -------
    dict
        Validation report with keys:
        - 'total_samples': int
        - 'complete_samples': int (no NaN values)
        - 'partial_samples': int (some NaN, below threshold)
        - 'poor_samples': int (NaN above threshold)
        - 'nan_by_column': Series with NaN fraction per wavenumber
        - 'nan_by_sample': Series with NaN fraction per sample
        - 'problem_wavenumbers': list of wavenumbers with >50% NaN
        - 'problem_samples': list of sample indices with NaN > threshold
    """
    exc_cols = list(exclude_columns) if exclude_columns else []
    
    # Identify spectral columns
    spectral_cols = [c for c in df.columns if c not in exc_cols and _is_numeric_string(str(c))]
    
    if not spectral_cols:
        raise ValueError("No spectral columns found")
    
    spectral_data = df[spectral_cols]
    
    # Calculate NaN statistics
    nan_by_sample = spectral_data.isna().mean(axis=1)
    nan_by_column = spectral_data.isna().mean(axis=0)
    
    # Categorize samples
    complete_mask = nan_by_sample == 0
    poor_mask = nan_by_sample > nan_threshold
    partial_mask = ~complete_mask & ~poor_mask
    
    # Identify problem areas
    problem_wavenumbers = nan_by_column[nan_by_column > 0.5].index.tolist()
    problem_samples = nan_by_sample[poor_mask].index.tolist()
    
    report = {
        'total_samples': len(df),
        'complete_samples': complete_mask.sum(),
        'partial_samples': partial_mask.sum(),
        'poor_samples': poor_mask.sum(),
        'nan_by_column': nan_by_column,
        'nan_by_sample': nan_by_sample,
        'problem_wavenumbers': problem_wavenumbers,
        'problem_samples': problem_samples,
        'spectral_columns': len(spectral_cols),
        'wavenumber_range': (float(min(float(c) for c in spectral_cols)),
                            float(max(float(c) for c in spectral_cols))),
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Total samples: {report['total_samples']}")
    print(f"  ✓ Complete (0% NaN): {report['complete_samples']}")
    print(f"  ~ Partial (0-{nan_threshold*100:.0f}% NaN): {report['partial_samples']}")
    print(f"  ✗ Poor (>{nan_threshold*100:.0f}% NaN): {report['poor_samples']}")
    print(f"\nSpectral range: {report['wavenumber_range'][0]:.1f} - {report['wavenumber_range'][1]:.1f} cm⁻¹")
    print(f"Spectral columns: {report['spectral_columns']}")
    
    if problem_wavenumbers:
        print(f"\n⚠ Problem wavenumbers (>50% NaN): {len(problem_wavenumbers)}")
        if len(problem_wavenumbers) <= 10:
            print(f"  {problem_wavenumbers}")
    
    print("=" * 70 + "\n")
    
    return report
