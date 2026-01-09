"""
Denoising Module for FTIR Spectral Preprocessing
=================================================

**IMPORTANT**: This module expects absorbance data (AU), not transmittance (%).
Convert transmittance to absorbance first using convert_spectra() from trans_abs.py

Features:
- Single spectrum denoising via denoise()
- Batch DataFrame processing via apply_denoising()
- Automatic column detection and sorting by wavenumber
- Performance optimized for large datasets (vectorized operations)
- Pandas and Polars DataFrame support
- Method evaluation via evaluate_denoising_methods()
- Memory-safe evaluation via evaluate_denoising_methods_safe()
- Composite scoring for method selection via find_best_denoising_method()

Memory Management:
For systems with limited RAM or large datasets, use evaluate_denoising_methods_safe()
instead of evaluate_denoising_methods(). See MEMORY_MANAGEMENT_GUIDE.md for details.

Logging:
This module uses Python's logging module for warnings and informational messages.
Configure the logger to control output:

    import logging
    logging.getLogger('utils.denoise').setLevel(logging.INFO)  # Show all messages
    logging.getLogger('utils.denoise').setLevel(logging.ERROR)  # Only errors

Available Methods:
Run denoise_method_names() to see all available denoising algorithms.
Common methods: savgol, wavelet, moving_average, gaussian, median, whittaker, lowpass
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional
import logging
import numpy as np
import pandas as pd

# Optional polars support
try:
    import polars as pl  # type: ignore
except Exception:
    pl = None  # type: ignore
from scipy import signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

# Optional wavelet support
try:
    import pywt  # type: ignore
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    pywt = None  # type: ignore
import joblib
from tqdm import tqdm
from contextlib import contextmanager
import matplotlib.pyplot as plt

from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)


def denoise(
    intensities: Union[np.ndarray, list],
    wavenumbers: Optional[Union[np.ndarray, list]] = None,
    method: str = "savgol",
    **kwargs
) -> np.ndarray:
    """
    Denoise a 1-D FTIR spectrum using various filtering methods.

    Parameters
    ----------
    intensities : array-like
        Raw intensity values (1-D).
    wavenumbers : array-like, optional
        X-axis values (wavenumbers in cm⁻¹). If provided, validates that data
        is sorted in ascending order and warns if not. Ensures API consistency
        with other preprocessing modules (baseline, atmospheric).
    method : str, default "savgol"
        Denoising method. Options:
        - 'savgol': Savitzky-Golay filter (preserves peak shape)
        - 'wavelet': Discrete wavelet transform denoising
        - 'moving_average': Simple moving average
        - 'gaussian': Gaussian filter
        - 'median': Median filter (good for spike noise)
        - 'whittaker': Penalized least squares smoother
        - 'lowpass': Low-pass Butterworth filter

    **kwargs : method-specific parameters
        savgol: window_length (15), polyorder (3)
        wavelet: wavelet ('db4'), level (3), threshold_mode ('soft')
        moving_average: window (11)
        gaussian: sigma (2.0)
        median: kernel_size (5)
        whittaker: lam (1e4), d (2)
        lowpass: cutoff (0.1), order (4)

    Returns
    -------
    np.ndarray
        Denoised intensity values. NaN values in input are preserved at their
        original positions; denoising is applied only to finite values.
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

        # Check if wavenumbers are monotonic (required for proper denoising)
        from .spectral_utils import _is_monotonic_strict
        if not _is_monotonic_strict(x):
            logger.warning(
                "Wavenumbers are not strictly monotonic. Denoising methods assume "
                "uniform spacing or sorted data. Results may be incorrect."
            )

    # Handle NaN values: preserve positions but compute denoising on finite values only
    nan_mask = ~np.isfinite(y)
    has_nans = np.any(nan_mask)

    if has_nans:
        # If all values are NaN, return NaN array
        if np.all(nan_mask):
            return np.full_like(y, np.nan)

        # Store original array and extract finite values
        y_original = y.copy()
        y_finite = y[~nan_mask]

        # Need at least 2 points for denoising
        if len(y_finite) < 2:
            logger.warning(
                f"Insufficient finite data points ({len(y_finite)}) for denoising. "
                "Returning original spectrum with NaN preserved."
            )
            return y_original

    # Process finite values only if NaN present
    y_to_process = y_finite if has_nans else y

    try:
        if method == "savgol":
            denoised_finite = _denoise_savgol(y_to_process, **kwargs)
        elif method == "wavelet":
            denoised_finite = _denoise_wavelet(y_to_process, **kwargs)
        elif method == "moving_average":
            denoised_finite = _denoise_moving_average(y_to_process, **kwargs)
        elif method == "gaussian":
            denoised_finite = _denoise_gaussian(y_to_process, **kwargs)
        elif method == "median":
            denoised_finite = _denoise_median(y_to_process, **kwargs)
        elif method == "whittaker":
            denoised_finite = _denoise_whittaker(y_to_process, **kwargs)
        elif method == "lowpass":
            denoised_finite = _denoise_lowpass(y_to_process, **kwargs)
        else:
            raise ValueError(
                f"Unknown denoising method: '{method}'. "
                "Valid options: savgol, wavelet, moving_average, gaussian, "
                "median, whittaker, lowpass"
            )
    except Exception as e:
        if isinstance(e, ValueError) and "Unknown denoising method" in str(e):
            raise
        raise RuntimeError(
            f"Denoising failed for method '{method}'. "
            f"Error: {type(e).__name__}: {str(e)}. "
            f"Check parameter compatibility with method documentation."
        ) from e

    # Restore NaN positions if needed
    if has_nans:
        result = np.full_like(y, np.nan)
        result[~nan_mask] = denoised_finite
        return result
    else:
        return denoised_finite


def denoise_method_names() -> List[str]:
    """Return list of available denoising method names."""
    return sorted([
        "savgol", "wavelet", "moving_average", "gaussian",
        "median", "whittaker", "lowpass"
    ])


# ---------------------------------------------------------------------------
#                    DATAFRAME-COMPATIBLE BATCH DENOISING
# ---------------------------------------------------------------------------

def apply_denoising(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    method: str = "savgol",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    show_progress: bool = True,
    **kwargs
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """
    Apply denoising to a DataFrame of FTIR spectra (batch processing).

    Works with both pandas and polars DataFrames. Each row is a sample,
    numerical columns are wavenumbers. Applies denoising to all samples.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format DataFrame where rows = samples, columns = wavenumbers.
        Should contain numerical columns with spectral data and optional
        metadata columns (e.g., 'sample', 'label').
    method : str, default "savgol"
        Denoising method. Options:
        - 'savgol': Savitzky-Golay filter (preserves peak shape)
        - 'wavelet': Discrete wavelet transform denoising
        - 'moving_average': Simple moving average
        - 'gaussian': Gaussian filter
        - 'median': Median filter (good for spike noise)
        - 'whittaker': Penalized least squares smoother
        - 'lowpass': Low-pass Butterworth filter
    label_column : str, default "label"
        Name of the label/group column to exclude from denoising.
    exclude_columns : list[str], optional
        Additional column names to exclude from denoising (e.g., 'sample', 'id').
        If None, automatically excludes non-numeric columns.
    wn_min : float, optional
        Minimum wavenumber bound (cm⁻¹). If None, uses 200.0 cm⁻¹ as default,
        or auto-expands if no columns found within default range.
    wn_max : float, optional
        Maximum wavenumber bound (cm⁻¹). If None, uses 8000.0 cm⁻¹ as default,
        or auto-expands if no columns found within default range.
    show_progress : bool, default True
        If True, display a progress bar during processing.
    **kwargs : additional parameters
        Method-specific parameters forwarded to the denoising algorithm:
        - savgol: window_length (15), polyorder (3)
        - wavelet: wavelet ('db4'), level (3), threshold_mode ('soft')
        - moving_average: window (11)
        - gaussian: sigma (2.0)
        - median: kernel_size (5)
        - whittaker: lam (1e4), d (2)
        - lowpass: cutoff (0.1), order (4)

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Denoised DataFrame (same type as input) with spectral data
        denoised and metadata columns preserved. Columns are sorted
        by ascending wavenumber order.

    Examples
    --------
    >>> # Apply Savitzky-Golay denoising to all samples
    >>> df_denoised = apply_denoising(df_wide, method="savgol")

    >>> # Use wavelet denoising with custom parameters
    >>> df_denoised = apply_denoising(
    ...     df_wide,
    ...     method="wavelet",
    ...     wavelet="db4",
    ...     level=3,
    ...     threshold_mode="soft"
    ... )

    >>> # Use Gaussian smoothing
    >>> df_denoised = apply_denoising(
    ...     df_wide,
    ...     method="gaussian",
    ...     sigma=2.0
    ... )

    >>> # Works with both pandas and polars
    >>> df_pd_denoised = apply_denoising(df_pandas)
    >>> df_pl_denoised = apply_denoising(df_polars)

    >>> # Disable progress bar for cleaner output
    >>> df_denoised = apply_denoising(df_wide, show_progress=False)
    """
    # Check for polars support
    is_polars = False
    if pl is not None:
        is_polars = isinstance(data, pl.DataFrame)

    # Convert to pandas for processing
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Prepare exclude_columns list
    if exclude_columns is None:
        exclude_columns = []
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    else:
        exclude_columns = list(exclude_columns)

    # Always exclude the label column if it exists
    if label_column in df.columns and label_column not in exclude_columns:
        exclude_columns.append(label_column)

    # Use spectral_utils to infer and sort spectral columns
    numeric_cols, wavenumbers = _infer_spectral_columns(
        df, exclude_columns, wn_min, wn_max
    )
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(
        numeric_cols, wavenumbers
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
                f"Denoising should be performed on absorbance for physical validity. "
                f"Please convert your data first using: "
                f"convert_spectra(data, mode='to_absorbance') from trans_abs.py"
            )

    denoised_data = np.empty((n_samples, n_wavenumbers), dtype=np.float64)

    # Denoising loop with progress bar
    iterator = tqdm(
        range(n_samples),
        desc=f"Denoising ({method})",
        disable=not show_progress,
        dynamic_ncols=True
    )

    for i in iterator:
        intensities = spectral_data[i, :]
        denoised_data[i, :] = denoise(
            intensities=intensities,
            wavenumbers=sorted_wavenumbers,
            method=method,
            **kwargs
        )

    # PHYSICAL CONSTRAINT VALIDATION: Check for negative absorbance after denoising
    finite_mask = np.isfinite(denoised_data)
    if np.any(finite_mask):
        n_negative = np.sum(denoised_data[finite_mask] < 0)
        if n_negative > 0:
            min_negative = np.min(denoised_data[finite_mask])
            pct_negative = 100.0 * n_negative / np.sum(finite_mask)
            logger.warning(
                f"Denoising produced {n_negative} negative absorbance values "
                f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                f"This is physically invalid and may indicate: "
                f"(1) aggressive smoothing parameters, (2) baseline drift in input data, "
                f"or (3) input data already near zero. "
                f"Recommendations: Apply baseline correction before denoising, or adjust "
                f"denoising parameters (e.g., reduce window_length for savgol)."
            )

    # Create result DataFrame with metadata + denoised spectra
    metadata_cols = [c for c in df.columns if c not in numeric_cols]
    metadata_df = df[metadata_cols].copy()

    # Create spectral DataFrame from numpy array (avoids fragmentation)
    spectral_df = pd.DataFrame(denoised_data, columns=sorted_cols, index=df.index)

    # Concatenate metadata and spectral data
    result_df = pd.concat([metadata_df, spectral_df], axis=1)

    # Reorder columns to match original structure (metadata first, then spectra)
    final_cols = metadata_cols + sorted_cols
    result_df = result_df[final_cols]

    # Convert back to polars if input was polars
    if is_polars:
        result_df = pl.from_pandas(result_df)

    return result_df


# ---------------------------------------------------------------------------
#                           INDIVIDUAL METHODS
# ---------------------------------------------------------------------------

def _denoise_savgol(
    y: np.ndarray,
    window_length: int = 15,
    polyorder: int = 3
) -> np.ndarray:
    """
    Savitzky-Golay filter.

    Fits successive sub-sets of adjacent data points with a low-degree polynomial
    by the method of linear least squares. Excellent for preserving peak shapes.
    """
    original_window = window_length
    original_poly = polyorder

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
        logger.debug(f"Savitzky-Golay: Adjusted window_length from {original_window} to {window_length} (must be odd)")

    # Ensure polyorder < window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
        logger.debug(f"Savitzky-Golay: Adjusted polyorder from {original_poly} to {polyorder} (must be < window_length)")

    return signal.savgol_filter(y, window_length, polyorder)


def _denoise_wavelet(
    y: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold_mode: str = "soft"
) -> np.ndarray:
    """
    Wavelet denoising using thresholding.

    Decomposes signal into wavelet coefficients, thresholds small coefficients,
    and reconstructs. Good for multi-scale noise reduction.

    Requires pywt (PyWavelets) package. If not installed, raises ImportError.
    """
    if not HAS_PYWT:
        raise ImportError(
            "Wavelet denoising requires the 'pywt' (PyWavelets) package. "
            "Install it with: pip install PyWavelets"
        )

    # Decompose
    coeffs = pywt.wavedec(y, wavelet, level=level)

    # Estimate noise level from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(y)))

    # Threshold detail coefficients (keep approximation coefficients)
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:
        if threshold_mode == "soft":
            denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
        else:
            denoised_coeffs.append(pywt.threshold(c, threshold, mode='hard'))

    # Reconstruct
    return pywt.waverec(denoised_coeffs, wavelet)[:len(y)]


def _denoise_moving_average(
    y: np.ndarray,
    window: int = 11
) -> np.ndarray:
    """Simple moving average filter."""
    return uniform_filter1d(y, size=window, mode='nearest')


def _denoise_gaussian(
    y: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    """Gaussian smoothing filter."""
    return gaussian_filter1d(y, sigma=sigma, mode='nearest')


def _denoise_median(
    y: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Median filter.

    Non-linear filter excellent for removing impulse/spike noise
    while preserving edges.
    """
    original_kernel = kernel_size

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        logger.debug(f"Median filter: Adjusted kernel_size from {original_kernel} to {kernel_size} (must be odd)")

    return signal.medfilt(y, kernel_size=kernel_size)


def _denoise_whittaker(
    y: np.ndarray,
    lam: float = 1e4,
    d: int = 2
) -> np.ndarray:
    """
    Whittaker smoother (penalized least squares) with sparse matrices.

    Balances fidelity to data with smoothness penalty.
    Higher lambda = smoother result.

    Uses sparse matrix operations for O(n) performance on large spectra
    (vs O(n^3) for dense implementation). Efficient for n > 1000.

    Parameters
    ----------
    y : np.ndarray
        Input signal
    lam : float, default 1e4
        Smoothing parameter. Higher values = smoother result.
    d : int, default 2
        Order of differences for penalty term (2 = penalize curvature)

    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    n = len(y)

    # Create sparse difference matrix (d-th order differences)
    # For d=2: D is (n-2) x n matrix computing second differences
    E = eye(n, format='csr')
    D = E[1:, :] - E[:-1, :]  # First difference
    for _ in range(d - 1):
        D = D[1:, :] - D[:-1, :]  # Higher-order differences

    # Solve sparse system: (I + lam * D'D) z = y
    # D'D is (n x n), symmetric positive definite
    W = E + lam * (D.T @ D)
    z = spsolve(W, y)

    return z


def _denoise_lowpass(
    y: np.ndarray,
    cutoff: float = 0.1,
    order: int = 4
) -> np.ndarray:
    """
    Low-pass Butterworth filter.
    
    Removes high-frequency noise components.
    cutoff: normalized frequency (0 to 1, relative to Nyquist).
    """
    # Design filter
    b, a = signal.butter(order, cutoff, btype='low')
    # Apply forward-backward filtering (zero phase)
    return signal.filtfilt(b, a, y)


# ---------------------------------------------------------------------------
#                           EVALUATION UTILITIES
# ---------------------------------------------------------------------------

def estimate_snr(
    y_raw: np.ndarray,
    y_denoised: np.ndarray,
    flat_regions: Optional[Union[List[Tuple[int, int]], List[Tuple[float, float]]]] = None,
    wavenumbers: Optional[np.ndarray] = None
) -> float:
    """
    Estimate Signal-to-Noise Ratio improvement (NaN-aware).

    Parameters
    ----------
    y_raw : np.ndarray
        Original noisy spectrum.
    y_denoised : np.ndarray
        Denoised spectrum.
    flat_regions : list of tuples, optional
        Regions known to be baseline-only (for noise estimation).
        Can be either:
        - List of (start_idx, end_idx) integer index tuples (legacy)
        - List of (wn_min, wn_max) float wavenumber tuples (recommended)
        If wavenumbers provided, flat_regions interpreted as wavenumber ranges.
        If None, uses high-frequency residual estimation.
    wavenumbers : np.ndarray, optional
        Wavenumber array. Required if flat_regions specified as wavenumber ranges.

    Returns
    -------
    float
        Estimated SNR in dB. Returns np.nan if insufficient finite data.

    Notes
    -----
    - Uses NaN-aware statistics to handle missing values in spectra
    - Wavenumber-based regions (recommended): More robust across different spectral resolutions
    - Index-based regions (legacy): Faster but resolution-dependent

    Examples
    --------
    >>> # Index-based (legacy)
    >>> snr = estimate_snr(y_raw, y_denoised, flat_regions=[(10, 50), (200, 250)])

    >>> # Wavenumber-based (recommended)
    >>> wn = np.linspace(650, 4000, 1000)
    >>> snr = estimate_snr(y_raw, y_denoised, flat_regions=[(2500, 2600), (3200, 3500)], wavenumbers=wn)
    """
    if flat_regions:
        # Estimate noise from difference in flat regions (NaN-aware)
        noise_samples = []

        # Determine if flat_regions are wavenumber-based or index-based
        if wavenumbers is not None and len(flat_regions) > 0:
            # Check if first element looks like wavenumbers (floats > 100) vs indices (ints < data length)
            first_start, first_end = flat_regions[0]
            if isinstance(first_start, float) or first_start > len(y_raw):
                # Wavenumber-based: convert to indices
                for wn_min, wn_max in flat_regions:
                    mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)
                    if np.any(mask):
                        indices = np.where(mask)[0]
                        start_idx, end_idx = indices[0], indices[-1] + 1
                        region_diff = y_raw[start_idx:end_idx] - y_denoised[start_idx:end_idx]
                        finite_diff = region_diff[np.isfinite(region_diff)]
                        if len(finite_diff) > 0:
                            noise_samples.extend(finite_diff)
            else:
                # Index-based (legacy): use directly
                for start, end in flat_regions:
                    region_diff = y_raw[start:end] - y_denoised[start:end]
                    finite_diff = region_diff[np.isfinite(region_diff)]
                    if len(finite_diff) > 0:
                        noise_samples.extend(finite_diff)
        else:
            # No wavenumbers provided: assume index-based
            for start, end in flat_regions:
                region_diff = y_raw[start:end] - y_denoised[start:end]
                finite_diff = region_diff[np.isfinite(region_diff)]
                if len(finite_diff) > 0:
                    noise_samples.extend(finite_diff)

        if len(noise_samples) == 0:
            return np.nan
        noise_std = np.std(noise_samples)
    else:
        # Estimate noise from high-frequency residuals (NaN-aware)
        residual = y_raw - y_denoised
        finite_residual = residual[np.isfinite(residual)]
        if len(finite_residual) == 0:
            return np.nan
        noise_std = np.std(finite_residual)

    # Use NaN-aware variance for signal power
    finite_signal = y_denoised[np.isfinite(y_denoised)]
    if len(finite_signal) == 0:
        return np.nan
    signal_power = np.var(finite_signal)
    noise_power = noise_std ** 2

    if noise_power > 0 and np.isfinite(signal_power):
        return 10 * np.log10(signal_power / noise_power)
    elif noise_power == 0 and signal_power > 0:
        return np.inf  # Perfect denoising
    else:
        return np.nan


# ---------------------------------------------------------------------------
#                           PARALLEL PROCESSING HELPERS
# ---------------------------------------------------------------------------

def _evaluate_one_sample(
    sample_name: str,
    y_raw: np.ndarray,
    methods: List[str]
) -> List[dict]:
    """
    Worker function: evaluate all denoising methods for one sample (NaN-aware).

    Returns list of result dictionaries for each method.

    Notes
    -----
    Uses NaN-aware statistics to handle missing values in evaluation metrics.
    Includes computation time measurement for performance comparison.
    """
    import time

    results = []
    for method in methods:
        try:
            # Time the denoising operation
            start_time = time.perf_counter()
            y_denoised = denoise(y_raw, method=method)
            elapsed_time = time.perf_counter() - start_time

            snr_improvement = estimate_snr(y_raw, y_denoised)

            # Compute smoothness (inverse of 2nd derivative variance) - NaN-aware
            finite_denoised = y_denoised[np.isfinite(y_denoised)]
            if len(finite_denoised) >= 3:  # Need at least 3 points for 2nd derivative
                d2 = np.diff(finite_denoised, n=2)
                finite_d2 = d2[np.isfinite(d2)]
                if len(finite_d2) > 0:
                    smoothness = 1.0 / (np.var(finite_d2) + 1e-10)
                else:
                    smoothness = np.nan
            else:
                smoothness = np.nan

            # Compute fidelity (correlation with original) - NaN-aware
            # Use only positions where both signals are finite
            valid_mask = np.isfinite(y_raw) & np.isfinite(y_denoised)
            if np.sum(valid_mask) >= 2:  # Need at least 2 points for correlation
                y_raw_finite = y_raw[valid_mask]
                y_denoised_finite = y_denoised[valid_mask]
                # Check for constant arrays (correlation undefined)
                if np.std(y_raw_finite) > 0 and np.std(y_denoised_finite) > 0:
                    fidelity = np.corrcoef(y_raw_finite, y_denoised_finite)[0, 1]
                else:
                    fidelity = np.nan
            else:
                fidelity = np.nan

            results.append({
                'sample': sample_name,
                'method': method,
                'snr_db': snr_improvement,
                'smoothness': smoothness,
                'fidelity': fidelity,
                'time_ms': elapsed_time * 1000  # Convert to milliseconds
            })
        except Exception as e:
            # Log warning for debugging
            logger.debug(
                f"Evaluation failed for sample '{sample_name}' with method '{method}': "
                f"{type(e).__name__}: {str(e)}"
            )
            results.append({
                'sample': sample_name,
                'method': method,
                'snr_db': np.nan,
                'smoothness': np.nan,
                'fidelity': np.nan,
                'time_ms': np.nan
            })

    return results


@contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def evaluate_denoising_methods_safe(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    methods: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Memory-safe wrapper for evaluate_denoising_methods() with conservative defaults.

    This function automatically sets safe defaults to prevent memory issues:
    - n_samples=50 (instead of all samples)
    - n_jobs=2 (instead of all CPU cores)
    - methods=['savgol', 'gaussian', 'median'] if not specified

    Use this function for initial exploration, then switch to full
    evaluate_denoising_methods() with custom parameters if needed.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Spectral DataFrame
    methods : list of str, optional
        Denoising methods to test. If None, uses ['savgol', 'gaussian', 'median'].
    **kwargs : additional parameters
        Forwarded to evaluate_denoising_methods(). Note that n_samples and n_jobs
        will be overridden to safe defaults unless explicitly provided.

    Returns
    -------
    pd.DataFrame
        Evaluation results with columns: sample, method, snr_db, smoothness, fidelity, time_ms

    Examples
    --------
    >>> # Safe evaluation (won't cause memory issues)
    >>> results = evaluate_denoising_methods_safe(df)
    >>> recommendations = find_best_denoising_method(results)

    >>> # With custom methods but still safe
    >>> results = evaluate_denoising_methods_safe(df, methods=['savgol', 'wavelet'])
    """
    # Set safe defaults
    safe_defaults = {
        'n_samples': 50,
        'n_jobs': 2,
        'sample_selection': 'random',
        'random_state': 42
    }

    # Use default safe methods if not specified
    if methods is None:
        methods = ['savgol', 'gaussian', 'median']

    # Merge user kwargs with safe defaults (user kwargs take precedence)
    params = {**safe_defaults, **kwargs}

    logger.info(
        f"Running safe evaluation with: {len(methods)} methods, "
        f"n_samples={params['n_samples']}, n_jobs={params['n_jobs']}"
    )

    return evaluate_denoising_methods(data, methods=methods, **params)


def evaluate_denoising_methods(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    methods: Optional[List[str]] = None,
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    n_samples: Optional[int] = None,              # None = use all samples
    sample_selection: str = "random",   # "random", "first", "last"
    random_state: Optional[int] = None,           # for reproducibility
    n_jobs: int = -1,                # -1 → all CPU cores
) -> pd.DataFrame:
    """
    Compare denoising methods on a subset of spectra.

    **MEMORY WARNING**: Parallel processing can consume significant memory.
    For systems with <16 GB RAM or large datasets (>1000 samples):
    - Set n_jobs=2 (not -1)
    - Set n_samples=50 (not None)
    - Test with methods=['savgol', 'gaussian'] first
    See MEMORY_MANAGEMENT_GUIDE.md for detailed recommendations.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format spectral DataFrame where rows = samples, columns = wavenumbers.
        Should contain numerical columns with spectral data and optional
        metadata columns (e.g., 'sample', 'label').
    methods : list of str, optional
        Methods to evaluate. If None, evaluates all available methods.
        Available: 'savgol', 'wavelet', 'moving_average', 'gaussian',
        'median', 'whittaker', 'lowpass'.
        **Recommendation**: Start with 2-3 methods to test memory usage.
    label_column : str, default "label"
        Name of the label/group column to exclude from evaluation.
    exclude_columns : list[str], optional
        Additional column names to exclude from evaluation (e.g., 'sample', 'id').
        If None, automatically excludes non-numeric columns.
    wn_min : float, optional
        Minimum wavenumber bound (cm⁻¹). If None, uses 200.0 cm⁻¹ as default.
    wn_max : float, optional
        Maximum wavenumber bound (cm⁻¹). If None, uses 8000.0 cm⁻¹ as default.
    n_samples : int, optional
        Number of samples to evaluate. If None, uses all samples.
    sample_selection : str, default "random"
        How to select samples: "random", "first", or "last".
    random_state : int, optional
        Random seed for reproducibility when sample_selection="random".
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all CPU cores.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics for each (sample, method) combination with columns:
        - sample: Sample identifier
        - method: Denoising method name
        - snr_db: Signal-to-noise ratio improvement (dB)
        - smoothness: Inverse of 2nd derivative variance (higher = smoother)
        - fidelity: Correlation with original signal (0-1, higher = better)
        - time_ms: Computation time in milliseconds (for performance comparison)
    """
    # Check for polars support
    is_polars = False
    if pl is not None:
        is_polars = isinstance(data, pl.DataFrame)

    # Convert to pandas for processing
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Prepare exclude_columns list
    if exclude_columns is None:
        exclude_columns = []
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    else:
        exclude_columns = list(exclude_columns)

    # Always exclude the label column if it exists
    if label_column in df.columns and label_column not in exclude_columns:
        exclude_columns.append(label_column)

    # Use spectral_utils to infer and sort spectral columns
    numeric_cols, wavenumbers = _infer_spectral_columns(
        df, exclude_columns, wn_min, wn_max
    )
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(
        numeric_cols, wavenumbers
    )

    if methods is None:
        methods = denoise_method_names()

    # Get sample indices (up to n_samples)
    samples = df.index.tolist()

    # ---------------- Memory safety check ------------------------------------
    n_samples_actual = n_samples if n_samples is not None else len(samples)
    n_methods = len(methods)
    n_wavenumbers = len(sorted_cols)
    n_workers = n_jobs if n_jobs > 0 else joblib.cpu_count()

    # Estimate memory usage (rough heuristic)
    estimated_mb_per_worker = (n_samples_actual * n_wavenumbers * 8) / (1024 * 1024)  # 8 bytes per float64
    estimated_total_mb = estimated_mb_per_worker * n_workers * 2  # 2x for processing overhead

    # Warn if configuration is likely to cause memory issues
    if estimated_total_mb > 8192:
        logger.warning(
            f"Memory warning: Current configuration may use ~{estimated_total_mb:.0f} MB RAM.\n"
            f"  - Samples: {n_samples_actual}\n"
            f"  - Methods: {n_methods}\n"
            f"  - Wavenumbers: {n_wavenumbers}\n"
            f"  - Parallel workers: {n_workers}\n"
            f"Recommendations to reduce memory usage:\n"
            f"  1. Set n_jobs=2 (instead of {n_jobs})\n"
            f"  2. Set n_samples=50 (instead of {n_samples_actual})\n"
            f"  3. Reduce methods list to 2-3 methods\n"
            f"See MEMORY_MANAGEMENT_GUIDE.md for details."
        )

    # ---------------- sample selection ------------------------------------
    if n_samples is not None and n_samples < len(samples):
        if sample_selection == "random":
            rng = np.random.default_rng(random_state)
            samples = list(rng.choice(samples, size=n_samples, replace=False))
        elif sample_selection == "first":
            samples = samples[:n_samples]
        elif sample_selection == "last":
            samples = samples[-n_samples:]
        else:
            raise ValueError(
                f"Unknown sample_selection '{sample_selection}'. "
                "Choose from: 'random', 'first', 'last'."
            )

    # Extract spectra to NumPy once (parent process) using sorted columns
    spectra = {
        s: df.loc[s, sorted_cols].astype(float).values
        for s in samples
    }

    # ---------------- parallel loop --------------------------------------
    worker = joblib.delayed(_evaluate_one_sample)

    with _tqdm_joblib(tqdm(desc="denoise eval", total=len(samples), dynamic_ncols=True)) as _:
        sample_results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            worker(s, spectra[s], methods)
            for s in samples
        )

    # ---------------- assemble results -----------------------------------
    # Flatten the list of lists into a single list
    results = [item for sublist in sample_results for item in sublist]

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
#                           EVALUATION VISUALIZATION
# ---------------------------------------------------------------------------

def plot_denoising_evaluation(
    eval_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    show_mean_sd: bool = True,
    save_plot: Optional[bool] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot evaluation metrics from evaluate_denoising_methods as box plots.

    Creates box plots for each metric (SNR, smoothness, fidelity) across
    different denoising methods to help select the best method.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output from evaluate_denoising_methods() with columns:
        ['sample', 'method', 'snr_db', 'smoothness', 'fidelity']
    metrics : list of str, optional
        Metrics to plot. If None, plots all three: ['snr_db', 'smoothness', 'fidelity']
        Available: 'snr_db', 'smoothness', 'fidelity'
    figsize : tuple, default (14, 5)
        Figure size (width, height)
    show_mean_sd : bool, default True
        If True, overlay mean ± SD on box plots
    save_path : str, optional
        If provided, save the figure to this path (e.g., 'denoising_eval.pdf')
    save_path : str, optional
        If provided, save the figure to this path (e.g., 'denoising_eval.pdf')

    Returns
    -------
    None
        Displays matplotlib figure

    Examples
    --------
    >>> # Evaluate methods
    >>> eval_results = evaluate_denoising_methods(df_wide, n_samples=50)
    >>>
    >>> # Plot all metrics
    >>> plot_denoising_evaluation(eval_results)
    >>>
    >>> # Plot only SNR and fidelity
    >>> plot_denoising_evaluation(eval_results, metrics=['snr_db', 'fidelity'])
    >>>
    >>> # Save to file
    >>> plot_denoising_evaluation(eval_results, save_path='denoise_eval.pdf')
    """
    if metrics is None:
        metrics = ['snr_db', 'smoothness', 'fidelity']

    # Validate metrics
    available_metrics = ['snr_db', 'smoothness', 'fidelity']
    for metric in metrics:
        if metric not in available_metrics:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {available_metrics}")

    # Get unique methods
    methods = sorted(eval_df['method'].unique())
    n_metrics = len(metrics)

    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Metric display names and better labels
    metric_labels = {
        'snr_db': 'SNR (dB)',
        'smoothness': 'Smoothness',
        'fidelity': 'Fidelity (correlation)'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare data for box plot
        data = [eval_df[eval_df['method'] == m][metric].dropna().values
                for m in methods]

        # Create box plot
        bp = ax.boxplot(
            data,
            labels=methods,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor='lightblue', alpha=0.7)
        )

        # Overlay mean ± SD
        if show_mean_sd:
            means = [np.nanmean(d) for d in data]
            sds = [np.nanstd(d, ddof=1) for d in data]
            x_positions = np.arange(1, len(methods) + 1)

            ax.errorbar(
                x_positions,
                means,
                yerr=sds,
                fmt='o',
                color='red',
                ecolor='red',
                elinewidth=2,
                capsize=4,
                markersize=6,
                linewidth=0,
                label='mean ± SD',
                zorder=10
            )

        # Formatting
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_xlabel('Denoising Method', fontsize=10)
        ax.set_title(f'{metric_labels[metric]} Comparison', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if show_mean_sd and idx == n_metrics - 1:
            ax.legend(loc='best', frameon=False, fontsize=9)

    plt.tight_layout()

    # Save if path provided
    if save_plot:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_denoising_evaluation_summary(
    eval_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_plot: Optional[bool] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create a summary table showing mean ± SD for all metrics across methods.

    Displays a colored heatmap-style visualization to quickly identify
    the best performing denoising methods.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output from evaluate_denoising_methods()
    figsize : tuple, default (10, 6)
        Figure size
    save_plot : bool, optional
        If True, save the figure to save_path
    save_path : str, optional
        If provided, save the figure to this path

    Examples
    --------
    >>> eval_results = evaluate_denoising_methods(df_wide, n_samples=50)
    >>> plot_denoising_evaluation_summary(eval_results)
    """
    # Calculate summary statistics
    summary = eval_df.groupby('method').agg({
        'snr_db': ['mean', 'std'],
        'smoothness': ['mean', 'std'],
        'fidelity': ['mean', 'std']
    }).round(3)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Create table
    table_data = []
    headers = ['Method', 'SNR (dB)', 'Smoothness', 'Fidelity']

    for _, row in summary.iterrows():
        method = row['method']
        snr = f"{row['snr_db_mean']:.2f} ± {row['snr_db_std']:.2f}"
        smooth = f"{row['smoothness_mean']:.2e} ± {row['smoothness_std']:.2e}"
        fidelity = f"{row['fidelity_mean']:.3f} ± {row['fidelity_std']:.3f}"
        table_data.append([method, snr, smooth, fidelity])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Denoising Methods: Summary Statistics',
              fontsize=14, fontweight='bold', pad=20)

    # Save if path provided
    if save_plot:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to: {save_path}")

    plt.show()


def find_best_denoising_method(
    eval_df: pd.DataFrame,
    snr_min: float = 10.0,
    smoothness_min: float = 1e3,
    fidelity_min: float = 0.9,
    time_max_ms: float = 100.0,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend best denoising methods based on evaluation metrics.

    Analyzes SNR, smoothness, fidelity, and computation time across all samples
    to identify methods that consistently perform well. Methods are ranked by
    a composite score combining all metrics.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output from evaluate_denoising_methods() with columns:
        ['sample', 'method', 'snr_db', 'smoothness', 'fidelity', 'time_ms']
    snr_min : float, default 10.0
        Minimum acceptable SNR in dB (higher is better).
    smoothness_min : float, default 1e3
        Minimum acceptable smoothness (higher is better).
    fidelity_min : float, default 0.9
        Minimum acceptable fidelity correlation (0-1, higher is better).
    time_max_ms : float, default 100.0
        Maximum acceptable computation time in milliseconds (lower is better).
    top_n : int, default 5
        Number of top methods to return.

    Returns
    -------
    pd.DataFrame
        Ranked methods with columns:
        - method: Method name
        - median_snr_db: Median SNR across samples
        - median_smoothness: Median smoothness across samples
        - median_fidelity: Median fidelity across samples
        - median_time_ms: Median computation time across samples
        - pass_rate: Fraction of samples passing all thresholds (0-1)
        - composite_score: Weighted score (higher is better)
        Sorted by composite_score descending (best methods first).

    Notes
    -----
    Composite Score Calculation:
        - Normalizes each metric to [0, 1] range
        - SNR: Higher is better
        - Smoothness: Higher is better
        - Fidelity: Higher is better
        - Time: Lower is better (inverted for scoring)
        - Pass rate: Bonus for consistent performance
        - Composite = (0.3 * SNR_score) + (0.25 * smoothness_score) +
                      (0.3 * fidelity_score) + (0.05 * time_score) + (0.1 * pass_rate)

    Example
    -------
    >>> eval_results = evaluate_denoising_methods(df, n_samples=50)
    >>> recommendations = find_best_denoising_method(eval_results, top_n=3)
    >>> print(recommendations)
         method  median_snr_db  median_smoothness  median_fidelity  median_time_ms  pass_rate  composite_score
    0   savgol          18.5             2.5e4            0.985            12.3       0.94             0.87
    1  wavelet          16.2             1.8e4            0.972            45.2       0.88             0.82
    2 gaussian          15.8             2.1e4            0.968            8.5        0.86             0.80
    """
    import warnings

    # Suppress warnings in this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Compute median metrics across samples for each method
        grouped = eval_df.groupby('method')
        median_snr = grouped['snr_db'].median()
        median_smoothness = grouped['smoothness'].median()
        median_fidelity = grouped['fidelity'].median()
        median_time = grouped['time_ms'].median()

        # Compute pass rate: fraction of samples meeting all thresholds
        snr_pass = (eval_df.groupby('method')['snr_db'].apply(lambda x: (x >= snr_min).sum()) /
                    eval_df.groupby('method').size())
        smoothness_pass = (eval_df.groupby('method')['smoothness'].apply(lambda x: (x >= smoothness_min).sum()) /
                           eval_df.groupby('method').size())
        fidelity_pass = (eval_df.groupby('method')['fidelity'].apply(lambda x: (x >= fidelity_min).sum()) /
                         eval_df.groupby('method').size())
        time_pass = (eval_df.groupby('method')['time_ms'].apply(lambda x: (x <= time_max_ms).sum()) /
                     eval_df.groupby('method').size())

        # Average pass rate across all metrics
        pass_rate = (snr_pass + smoothness_pass + fidelity_pass + time_pass) / 4.0

        # Normalize metrics to [0, 1] for composite scoring
        # SNR, smoothness, fidelity: higher is better
        # Time: lower is better (invert for scoring)
        snr_min_val, snr_max = median_snr.min(), median_snr.max()
        smooth_min_val, smooth_max = median_smoothness.min(), median_smoothness.max()
        fid_min_val, fid_max = median_fidelity.min(), median_fidelity.max()
        time_min_val, time_max_val = median_time.min(), median_time.max()

        # Avoid division by zero
        snr_range = snr_max - snr_min_val if snr_max > snr_min_val else 1.0
        smooth_range = smooth_max - smooth_min_val if smooth_max > smooth_min_val else 1.0
        fid_range = fid_max - fid_min_val if fid_max > fid_min_val else 1.0
        time_range = time_max_val - time_min_val if time_max_val > time_min_val else 1.0

        # Normalize (higher is better for all scores)
        snr_score = (median_snr - snr_min_val) / snr_range
        smooth_score = (median_smoothness - smooth_min_val) / smooth_range
        fid_score = (median_fidelity - fid_min_val) / fid_range
        time_score = 1.0 - (median_time - time_min_val) / time_range  # Invert time

        # Composite score (weighted average)
        # Weights: SNR=0.3, Smoothness=0.25, Fidelity=0.3, Time=0.05, PassRate=0.1
        composite_score = (0.30 * snr_score +
                           0.25 * smooth_score +
                           0.30 * fid_score +
                           0.05 * time_score +
                           0.10 * pass_rate)

        # Build results DataFrame
        results = pd.DataFrame({
            'method': median_snr.index,
            'median_snr_db': median_snr.values,
            'median_smoothness': median_smoothness.values,
            'median_fidelity': median_fidelity.values,
            'median_time_ms': median_time.values,
            'pass_rate': pass_rate.values,
            'composite_score': composite_score.values
        })

        # Sort by composite score (best first) and return top_n
        results = results.sort_values('composite_score', ascending=False).reset_index(drop=True)
        return results.head(top_n)


def plot_denoising_comparison(
    y_raw: np.ndarray,
    wavenumbers: np.ndarray,
    methods: Optional[List[str]] = None,
    sample_name: str = "",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot comparison of multiple denoising methods.

    Parameters
    ----------
    y_raw : np.ndarray
        Raw spectrum.
    wavenumbers : np.ndarray
        Wavenumber axis.
    methods : list of str, optional
        Methods to compare. Default: all.
    sample_name : str
        Sample name for title.
    figsize : tuple
        Figure size.
    """

    if methods is None:
        methods = denoise_method_names()

    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=figsize, sharex=True)

    # Plot raw
    axes[0].plot(wavenumbers, y_raw, 'k-', lw=0.8)
    axes[0].set_ylabel('Raw')
    axes[0].set_title(f'Denoising Comparison: {sample_name}')

    # Plot each method
    for i, method in enumerate(methods, 1):
        y_denoised = denoise(y_raw, method=method)
        axes[i].plot(wavenumbers, y_denoised, 'b-', lw=0.8)
        axes[i].set_ylabel(method)

    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()
