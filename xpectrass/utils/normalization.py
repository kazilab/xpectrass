"""
Normalization Module for FTIR Spectral Preprocessing
=====================================================

**IMPORTANT**: This module expects absorbance data (AU), not transmittance (%).
Convert transmittance to absorbance first using convert_spectra() from trans_abs.py

Provides multiple normalization methods for FTIR spectra including
SNV, vector, area, min-max, and peak normalization.
Also includes mean centering for PCA/PLS preparation.
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional
import logging
import numpy as np
import pandas as pd
import polars as pl
from scipy import ndimage
from tqdm import tqdm

# Import shared spectral utilities
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


def normalize(
    intensities: Union[np.ndarray, list],
    wavenumbers: Optional[Union[np.ndarray, list]] = None,
    method: str = "snv",
    **kwargs
) -> np.ndarray:
    """
    Normalize a 1-D FTIR spectrum.

    Parameters
    ----------
    intensities : array-like
        Intensity values (1-D).
    wavenumbers : array-like, optional
        X-axis values (wavenumbers in cm⁻¹). Required for peak_wavenumber
        and adaptive_regional methods.
    method : str, default "snv"
        Normalization method. Options:
        - 'snv': Standard Normal Variate (mean=0, std=1 within spectrum)
        - 'vector': L2 vector normalization (unit length)
        - 'minmax': Min-Max scaling to [0, 1]
        - 'area': Area normalization (total area = 1)
        - 'peak': Normalize by peak intensity
        - 'range': Normalize by intensity range
        - 'max': Normalize by maximum value
        - 'detrend': Polynomial detrending
        - 'snv_detrend': SNV followed by detrending

        Novel methods (1D-compatible):
        - 'robust_snv': Robust SNV using median/MAD
        - 'curvature_weighted': Curvature-weighted normalization
        - 'peak_envelope': Peak envelope normalization
        - 'entropy_weighted': Entropy-weighted normalization
        - 'pqn': Probabilistic quotient normalization
        - 'total_variation': Total variation normalization
        - 'spectral_moments': Spectral moment normalization
        - 'adaptive_regional': Adaptive regional normalization (requires wavenumbers)
        - 'derivative_ratio': Derivative ratio normalization
        - 'signal_to_baseline': Signal-to-baseline ratio normalization

    **kwargs : method-specific parameters
        peak: peak_idx (index), peak_wavenumber (float, requires wavenumbers),
              peak_value (float), use_absolute (bool)
        minmax: feature_range (tuple, default (0, 1))
        detrend: order (int, default 1)
        adaptive_regional: regions (list of tuples), method_per_region (dict)

    Returns
    -------
    np.ndarray
        Normalized intensity values.

    Notes
    -----
    Methods that require multiple spectra (2D data) are NOT available here:
    - 'mean_center': Use mean_center() directly on 2D array
    - 'auto_scale': Use auto_scale() directly on 2D array
    - 'pareto': Use pareto_scale() directly on 2D array
    These methods compute column-wise statistics and should be applied via
    normalize_df() for batch processing.

    **PQN (Probabilistic Quotient Normalization):**
    - For single spectrum WITH reference: Provide `reference` kwarg for true PQN
      Example: normalize(y, method='pqn', reference=ref_spectrum)
    - For single spectrum WITHOUT reference: Falls back to median scaling (NOT true PQN!)
      A warning will be issued in this case.
    - For batch PQN: Use normalize_df() instead (auto-computes reference from dataset)
    """
    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("`intensities` must be a 1-D array-like object.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`intensities` contains non-finite values (NaN or Inf).")

    # Convert wavenumbers to numpy array if provided
    wn = np.asarray(wavenumbers, dtype=np.float64) if wavenumbers is not None else None

    if method == "snv":
        return _normalize_snv(y)
    elif method == "vector":
        return _normalize_vector(y)
    elif method == "minmax":
        feature_range = kwargs.get('feature_range', (0.0, 1.0))
        return _normalize_minmax(y, feature_range=feature_range)
    elif method == "area":
        return _normalize_area(y, wavenumbers=wn)
    elif method == "peak":
        # Handle peak_wavenumber parameter
        peak_wn = kwargs.get("peak_wavenumber", None)
        if peak_wn is not None:
            if wn is None:
                raise ValueError("`wavenumbers` must be provided if `peak_wavenumber` is used.")
            if len(wn) != len(y):
                raise ValueError("`wavenumbers` must have the same length as `intensities`.")
            peak_idx = int(np.argmin(np.abs(wn - float(peak_wn))))
        else:
            peak_idx = kwargs.get("peak_idx", None)
        peak_value = kwargs.get("peak_value", None)
        use_absolute = kwargs.get("use_absolute", True)

        return _normalize_peak(y, peak_idx=peak_idx, peak_value=peak_value, use_absolute=use_absolute)
    elif method == "range":
        return _normalize_range(y)
    elif method == "max":
        return _normalize_max(y)
    elif method == "detrend":
        order = kwargs.get('order', 1)
        return detrend(y, order=order, wavenumbers=wn)
    elif method == "snv_detrend":
        order = kwargs.get('order', 1)
        return snv_detrend(y, detrend_order=order, wavenumbers=wn)
    elif method == "robust_snv":
        return normalize_robust_snv(y)
    elif method == "curvature_weighted":
        sigma = kwargs.get('sigma', 3.0)
        min_weight = kwargs.get('min_weight', 0.01)
        return normalize_curvature_weighted(y, sigma=sigma, min_weight=min_weight)
    elif method == "peak_envelope":
        percentile = kwargs.get('percentile', 95)
        window_size = kwargs.get('window_size', 50)
        return normalize_peak_envelope(y, percentile=percentile, window_size=window_size)
    elif method == "entropy_weighted":
        n_bins = kwargs.get('n_bins', 50)
        window_size = kwargs.get('window_size', 30)
        return normalize_entropy_weighted(y, n_bins=n_bins, window_size=window_size)
    elif method == "pqn":
        reference = kwargs.get('reference', None)
        reference_type = kwargs.get('reference_type', 'median')
        return normalize_pqn(y, reference=reference, reference_type=reference_type)
    elif method == "total_variation":
        order = kwargs.get('order', 1)
        return normalize_total_variation(y, order=order)
    elif method == "spectral_moments":
        moment_order = kwargs.get('moment_order', 2)
        use_central = kwargs.get('use_central', True)
        return normalize_spectral_moments(y, moment_order=moment_order, use_central=use_central)
    elif method == "adaptive_regional":
        regions = kwargs.get('regions', None)
        method_per_region = kwargs.get('method_per_region', None)
        return normalize_adaptive_regional(y, wn, regions, method_per_region)
    elif method == "derivative_ratio":
        sigma = kwargs.get('sigma', 2.0)
        return normalize_derivative_ratio(y, sigma=sigma)
    elif method == "signal_to_baseline":
        baseline_percentile = kwargs.get('baseline_percentile', 10)
        signal_percentile = kwargs.get('signal_percentile', 90)
        return normalize_signal_to_baseline(y, baseline_percentile, signal_percentile)
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            "Valid 1D methods: snv, vector, minmax, area, peak, range, max, "
            "detrend, snv_detrend, robust_snv, curvature_weighted, peak_envelope, "
            "entropy_weighted, pqn, total_variation, spectral_moments, adaptive_regional, "
            "derivative_ratio, signal_to_baseline. "
            "For 2D methods (mean_center, auto_scale, pareto), use normalize_df() instead."
        )


def normalize_method_names() -> List[str]:
    """
    Return list of available 1D normalization method names.

    Note: Methods requiring 2D data (mean_center, auto_scale, pareto)
    are not included. Use normalize_df() for batch processing with those methods.
    """
    return sorted([
        "snv", "vector", "minmax", "area", "peak", "range", "max",
        "detrend", "snv_detrend", "robust_snv", "curvature_weighted",
        "peak_envelope", "entropy_weighted", "pqn", "total_variation",
        "spectral_moments", "adaptive_regional", "derivative_ratio",
        "signal_to_baseline"
    ])


# ---------------------------------------------------------------------------
#                           INDIVIDUAL METHODS
# ---------------------------------------------------------------------------

def _normalize_snv(y: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) normalization.
    
    Removes multiplicative scatter effects by centering and scaling
    each spectrum individually. Common preprocessing for NIR/IR.
    
    Formula: (x - mean(x)) / std(x)
    """
    mean = np.mean(y)
    std = np.std(y)
    if std == 0:
        return np.zeros_like(y)
    return (y - mean) / std


def _normalize_vector(y: np.ndarray) -> np.ndarray:
    """
    L2 Vector normalization.
    
    Scales spectrum to unit length (L2 norm = 1).
    Useful for comparing spectral shapes regardless of intensity.
    
    Formula: x / ||x||_2
    """
    norm = np.linalg.norm(y)
    if norm == 0:
        return np.zeros_like(y)
    return y / norm


def _normalize_minmax(
    y: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Min-Max normalization.
    
    Scales values to specified range (default [0, 1]).
    
    Formula: (x - min) / (max - min) * (new_max - new_min) + new_min
    """
    y_min, y_max = np.min(y), np.max(y)
    if y_max == y_min:
        return np.full_like(y, feature_range[0])
    
    scaled = (y - y_min) / (y_max - y_min)
    new_min, new_max = feature_range
    return scaled * (new_max - new_min) + new_min


def _normalize_area(
    y: np.ndarray,
    wavenumbers: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Area normalization using trapezoidal integration.

    Scales spectrum so total area under curve equals 1.
    Useful for comparing relative concentrations.

    Parameters
    ----------
    y : np.ndarray
        Spectrum intensities
    wavenumbers : np.ndarray, optional
        Wavenumber array for proper area integration.
        If provided, uses trapezoidal integration accounting for spacing.
        If None, falls back to sum of absolute values (uniform spacing assumption).

    Returns
    -------
    np.ndarray
        Area-normalized spectrum

    Notes
    -----
    - With wavenumbers: Uses np.trapezoid() for true area under curve,
      accounting for non-uniform spacing
    - Without wavenumbers: Uses sum(|y|) assuming uniform spacing
    - Works on absolute values to handle spectra with negative regions

    Formula:
        With wavenumbers: x / trapezoid(|x|, x=wavenumbers)
        Without wavenumbers: x / sum(|x|)
    """
    if wavenumbers is not None:
        if len(wavenumbers) != len(y):
            raise ValueError(
                f"Length mismatch: wavenumbers has {len(wavenumbers)} elements "
                f"but intensities has {len(y)} elements."
            )
        # Use trapezoidal integration for proper area calculation
        # This accounts for non-uniform wavenumber spacing
        x = np.asarray(wavenumbers, dtype=np.float64)
        yy = np.abs(y)
        # Ensure integration direction doesn't flip sign (common FT-IR: descending cm^-1)
        if x[0] > x[-1]:
            x = x[::-1]
            yy = yy[::-1]
        total_area = np.trapezoid(yy, x=x)
    else:
        # Fallback: sum of absolute values (uniform spacing assumption)
        total_area = np.sum(np.abs(y))

    if total_area == 0:
        return np.zeros_like(y)
    return y / total_area


def _normalize_peak(
    y: np.ndarray,
    peak_idx: Optional[int] = None,
    peak_value: Optional[float] = None,
    use_absolute: bool = True
) -> np.ndarray:
    """
    Peak normalization.

    Normalize by the intensity at a specific peak or maximum.
    Useful when reference peak intensity is known/expected.

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    peak_idx : int, optional
        Index of the normalization peak. If None, uses maximum absolute value.
        Must be within valid range [0, len(y)-1].
    peak_value : float, optional
        If provided, normalize to this value at peak_idx.
    use_absolute : bool, default True
        If True, use absolute value for intensity (recommended for robustness).
        If False, use signed value (can invert spectrum if peak is negative).

    Returns
    -------
    np.ndarray
        Peak-normalized spectrum.

    Raises
    ------
    IndexError
        If peak_idx is out of bounds.
    """
    if peak_idx is None:
        # Use maximum absolute value (default behavior)
        ref_intensity = np.max(np.abs(y))
    else:
        # Validate peak_idx bounds
        if peak_idx < 0 or peak_idx >= len(y):
            raise IndexError(
                f"peak_idx={peak_idx} is out of bounds for spectrum of length {len(y)}. "
                f"Valid range: [0, {len(y)-1}]"
            )

        # Get intensity at specified index
        if use_absolute:
            ref_intensity = np.abs(y[peak_idx])
        else:
            ref_intensity = y[peak_idx]

    if ref_intensity == 0:
        return np.zeros_like(y)

    if peak_value is not None:
        return y * (peak_value / ref_intensity)
    return y / ref_intensity


def _normalize_range(y: np.ndarray) -> np.ndarray:
    """
    Range normalization.
    
    Divides by the range (max - min).
    
    Formula: x / (max - min)
    """
    y_range = np.max(y) - np.min(y)
    if y_range == 0:
        return np.zeros_like(y)
    return y / y_range


def _normalize_max(y: np.ndarray) -> np.ndarray:
    """
    Maximum normalization.
    
    Scales so maximum value equals 1.
    
    Formula: x / max(|x|)
    """
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return np.zeros_like(y)
    return y / max_val


# ---------------------------------------------------------------------------
#                           BATCH OPERATIONS
# ---------------------------------------------------------------------------

def normalize_df(
    data: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
    method: str = "snv",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    show_progress: bool = True,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame, np.ndarray]:
    """
    Normalize multiple spectra (DataFrame or numpy array).

    Works with both pandas and polars DataFrames, or numpy arrays.
    For DataFrames: each row is a sample, numerical columns are wavenumbers.
    For numpy arrays: shape (n_samples, n_wavenumbers).

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | np.ndarray
        Wide-format DataFrame where rows = samples, columns = wavenumbers,
        OR numpy array of shape (n_samples, n_wavenumbers).
    method : str, default "snv"
        Normalization method. Options:

        Single-spectrum methods (row-wise, 1D):
        - 'snv': Standard Normal Variate (mean=0, std=1 within each spectrum)
        - 'vector': L2 vector normalization (unit length)
        - 'minmax': Min-Max scaling to [0, 1]
        - 'area': Area normalization (total area = 1)
        - 'peak': Normalize by peak intensity
        - 'range': Normalize by intensity range
        - 'max': Normalize by maximum value
        - 'detrend': Polynomial detrending
        - 'snv_detrend': SNV followed by detrending
        - Plus all novel methods (robust_snv, curvature_weighted, etc.)

        Multi-spectrum methods (column-wise, 2D - for PCA/PLS prep):
        - 'mean_center': Column-wise mean centering (mean=0 per wavenumber)
          **Requires ≥2 samples**
        - 'auto_scale': Column-wise auto-scaling (mean=0, std=1 per wavenumber)
          **Requires ≥2 samples**
        - 'pareto': Column-wise Pareto scaling (mean=0, scaled by sqrt(std))
          **Requires ≥2 samples**

        Dataset-level methods (require reference from entire dataset):
        - 'pqn': Probabilistic Quotient Normalization (computes reference spectrum
          from dataset median/mean, then normalizes each spectrum relative to it)
          **Requires ≥3 samples**
    label_column : str, default "label"
        Name of the label/group column to exclude from normalization.
        Only used for DataFrame inputs.
    exclude_columns : list[str], optional
        Additional column names to exclude from normalization (e.g., 'sample', 'id').
        If None, automatically excludes non-numeric columns.
        Only used for DataFrame inputs.
    wn_min : float, optional
        Minimum wavenumber bound (cm⁻¹). If None, uses 200.0 cm⁻¹ as default,
        or auto-expands if no columns found within default range.
    wn_max : float, optional
        Maximum wavenumber bound (cm⁻¹). If None, uses 8000.0 cm⁻¹ as default,
        or auto-expands if no columns found within default range.
    show_progress : bool, default True
        If True, display a progress bar during processing.
        Only used for DataFrame inputs.
    **kwargs : method-specific parameters
        peak: peak_idx (index) or peak_wavenumber (requires wavenumbers array)
        minmax: feature_range (tuple, default (0, 1))
        detrend: order (int, default 1)
        pqn: reference_type (str, default 'median') - 'median' or 'mean' for
             computing reference spectrum from dataset

    Returns
    -------
    pd.DataFrame | pl.DataFrame | np.ndarray
        Normalized data (same type as input).

    Raises
    ------
    ValueError
        If using 2D methods (mean_center, auto_scale, pareto) with <2 samples.
        If using PQN with <3 samples.

    Examples
    --------
    >>> # Normalize DataFrame with SNV
    >>> df_norm = normalize_batch(df_wide, method="snv")

    >>> # Normalize with vector normalization
    >>> df_norm = normalize_batch(df_wide, method="vector")

    >>> # Normalize numpy array
    >>> spectra_norm = normalize_batch(spectra_array, method="snv")

    >>> # Min-max normalization to [0, 1]
    >>> df_norm = normalize_batch(df_wide, method="minmax", feature_range=(0, 1))

    >>> # PQN with median reference (proper batch PQN)
    >>> df_norm = normalize_batch(df_wide, method="pqn", reference_type="median")

    >>> # PQN with mean reference
    >>> df_norm = normalize_batch(df_wide, method="pqn", reference_type="mean")

    >>> # Disable progress bar
    >>> df_norm = normalize_batch(df_wide, method="snv", show_progress=False)
    """
    # Handle numpy array input (only 1d methods)
    if isinstance(data, np.ndarray):
        return np.array([normalize(s, method=method, **kwargs) for s in data])

    # Handle DataFrame input (both 1d, 2d methods, with dataframes only)
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

    # Identify all parseable numeric columns (for metadata separation)
    all_numeric_cols = []
    for c in df.columns:
        if c in exclude_columns:
            continue
        try:
            float(c)
            all_numeric_cols.append(c)
        except Exception:
            continue

    # Metadata = all columns that are NOT parseable as wavenumbers
    metadata_cols = [c for c in df.columns if c not in all_numeric_cols]

    # VALIDATION: Check if data appears to be transmittance instead of absorbance
    # Sample spectral data for efficiency
    temp_spectral_data = df[sorted_cols].values.astype(np.float64)
    sample_size = min(100, temp_spectral_data.shape[0])
    sample_data = temp_spectral_data[:sample_size, :].flatten()
    sample_data_finite = sample_data[np.isfinite(sample_data)]

    if len(sample_data_finite) > 0:
        median_val = np.median(sample_data_finite)
        p95_val = np.percentile(sample_data_finite, 95)

        if p95_val > 10.0 and median_val > 1.0:
            raise ValueError(
                f"Input data appears to be transmittance (%) rather than absorbance (AU). "
                f"Detected: median={median_val:.2f}, 95th percentile={p95_val:.2f}. "
                f"Normalization should be performed on absorbance for physical validity. "
                f"Please convert your data first using: "
                f"convert_spectra(data, mode='to_absorbance') from trans_abs.py"
            )

    # Check if method requires dataset-level processing
    # - Column-wise 2D methods: mean_center, auto_scale, pareto
    # - Dataset reference methods: pqn (requires reference spectrum from dataset)
    methods_2d = ['mean_center', 'auto_scale', 'pareto']
    methods_dataset_ref = ['pqn']

    # Validate minimum sample size for dataset-level methods
    n_samples = len(df)
    MIN_SAMPLES_2D = 2  # Minimum for meaningful column-wise statistics
    MIN_SAMPLES_PQN = 3  # Minimum for meaningful reference spectrum

    if method in methods_2d and n_samples < MIN_SAMPLES_2D:
        raise ValueError(
            f"Method '{method}' requires at least {MIN_SAMPLES_2D} samples for "
            f"column-wise normalization, but only {n_samples} sample(s) provided. "
            f"For single-spectrum normalization, use a row-wise method like 'snv', "
            f"'vector', 'area', etc."
        )

    if method in methods_dataset_ref and n_samples < MIN_SAMPLES_PQN:
        raise ValueError(
            f"Method '{method}' requires at least {MIN_SAMPLES_PQN} samples to compute "
            f"a meaningful reference spectrum, but only {n_samples} sample(s) provided. "
            f"For single-spectrum normalization, use normalize() with a provided reference, "
            f"or use a different method."
        )

    if method in methods_2d:
        # Extract spectral data as 2D array (using sorted, filtered columns)
        spectral_data = df[sorted_cols].values.astype(np.float64)

        # Apply 2D method
        if method == 'mean_center':
            normalized_data = mean_center(spectral_data, axis=0, return_mean=False)
        elif method == 'auto_scale':
            normalized_data = auto_scale(spectral_data, return_params=False)
        elif method == 'pareto':
            normalized_data = pareto_scale(spectral_data, return_params=False)

        # Create result DataFrame with metadata + normalized spectra
        metadata_df = df[metadata_cols].copy()

        # Create spectral DataFrame from numpy array
        spectral_df = pd.DataFrame(normalized_data, columns=sorted_cols, index=df.index)

        # Concatenate metadata and spectral data
        result_df = pd.concat([metadata_df, spectral_df], axis=1)

        # Reorder columns to match original structure (metadata first, then spectra)
        final_cols = metadata_cols + sorted_cols
        df = result_df[final_cols]

    elif method in methods_dataset_ref:
        # PQN: Requires computing reference spectrum from dataset

        # Extract spectral data as 2D array
        spectral_data = df[sorted_cols].values.astype(np.float64)
        n_samples = spectral_data.shape[0]
        n_wavenumbers = spectral_data.shape[1]

        # Compute reference spectrum from dataset
        reference_type = kwargs.get('reference_type', 'median')

        if reference_type == 'median':
            reference_spectrum = np.median(spectral_data, axis=0)
        elif reference_type == 'mean':
            reference_spectrum = np.mean(spectral_data, axis=0)
        else:
            raise ValueError(
                f"Invalid reference_type '{reference_type}'. "
                f"Must be 'median' or 'mean'."
            )

        # Apply PQN to each spectrum using the reference
        normalized_data = np.empty((n_samples, n_wavenumbers), dtype=np.float64)

        iterator = tqdm(
            range(n_samples),
            desc=f"PQN normalization (ref={reference_type})",
            disable=not show_progress,
            dynamic_ncols=True
        )

        for i in iterator:
            normalized_data[i, :] = normalize_pqn(
                spectral_data[i, :],
                reference=reference_spectrum,
                reference_type=reference_type
            )

        # Create result DataFrame with metadata + normalized spectra
        metadata_df = df[metadata_cols].copy()

        # Create spectral DataFrame from numpy array
        spectral_df = pd.DataFrame(normalized_data, columns=sorted_cols, index=df.index)

        # Concatenate metadata and spectral data
        result_df = pd.concat([metadata_df, spectral_df], axis=1)

        # Reorder columns to match original structure (metadata first, then spectra)
        final_cols = metadata_cols + sorted_cols
        df = result_df[final_cols]

    else:
        # Apply 1D normalization to each sample (row-wise)

        # OPTIMIZATION: Extract numpy array and pre-allocate result
        spectral_data = df[sorted_cols].values.astype(np.float64)
        n_samples = spectral_data.shape[0]
        n_wavenumbers = spectral_data.shape[1]
        normalized_data = np.empty((n_samples, n_wavenumbers), dtype=np.float64)

        # Normalization loop with progress bar
        iterator = tqdm(
            range(n_samples),
            desc=f"Normalization ({method})",
            disable=not show_progress,
            dynamic_ncols=True
        )

        for i in iterator:
            intensities = spectral_data[i, :]
            normalized_data[i, :] = normalize(
                intensities=intensities,
                wavenumbers=sorted_wavenumbers,
                method=method,
                **kwargs
            )

        # Create result DataFrame with metadata + normalized spectra
        metadata_df = df[metadata_cols].copy()

        # Create spectral DataFrame from numpy array (avoids fragmentation)
        spectral_df = pd.DataFrame(normalized_data, columns=sorted_cols, index=df.index)

        # Concatenate metadata and spectral data
        result_df = pd.concat([metadata_df, spectral_df], axis=1)

        # Reorder columns to match original structure (metadata first, then spectra)
        final_cols = metadata_cols + sorted_cols
        df = result_df[final_cols]

    # PHYSICAL CONSTRAINT VALIDATION: Method-specific validation for negative values
    # Categorize methods by whether negative values are expected

    # Methods that EXPECT negative values (mean-centering, detrending, baseline-shifting)
    methods_expect_negative = [
        # Row-wise mean-centering methods
        'snv', 'snv_detrend', 'robust_snv',
        # Column-wise centering (for PCA/PLS prep)
        'mean_center', 'auto_scale', 'pareto',
        # Detrending methods (polynomial baseline removal)
        'detrend',
        # Novel methods that produce centered data or can produce negatives
        'spectral_moments',  # With use_central=True (default), returns mean-centered data
        'signal_to_baseline',  # Shifts by baseline_level, can produce negatives
        'derivative_ratio',  # Preserves sign of input (can be negative after baseline correction)
    ]

    # Methods that should ONLY produce non-negative values (if input is non-negative)
    methods_should_be_nonnegative = [
        # Simple scaling methods (preserve sign, but should be positive if input is positive)
        'minmax', 'area', 'peak', 'range', 'max', 'vector',
        # Ratio-based methods (should stay positive if input is positive)
        'pqn',
        # Novel methods that should preserve positivity
        'curvature_weighted', 'peak_envelope', 'entropy_weighted',
        'total_variation',
        # Note: adaptive_regional depends on methods used per region
    ]

    # Extract final normalized data for validation
    final_spectral_data = df[sorted_cols].values.astype(np.float64)
    finite_mask = np.isfinite(final_spectral_data)

    if np.any(finite_mask):
        n_negative = np.sum(final_spectral_data[finite_mask] < 0)

        if n_negative > 0:
            min_negative = np.min(final_spectral_data[finite_mask])
            pct_negative = 100.0 * n_negative / np.sum(finite_mask)

            # Decide whether to warn based on method type
            if method in methods_expect_negative:
                # Negative values are EXPECTED - only log at debug level
                logger.debug(
                    f"Normalization method '{method}' produced {n_negative} negative values "
                    f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                    f"This is EXPECTED for mean-centering/baseline-shifting methods."
                )
            elif method in methods_should_be_nonnegative:
                # Negative values are UNEXPECTED - warn strongly
                logger.warning(
                    f"Normalization method '{method}' produced {n_negative} negative values "
                    f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                    f"This is UNEXPECTED for this method and indicates a problem. "
                    f"Possible causes: (1) Input data already has negative absorbance (check baseline correction), "
                    f"(2) Numerical instability in normalization, or (3) Data contains artifacts. "
                    f"Recommendation: Inspect input data and consider baseline correction."
                )
            elif method == 'adaptive_regional':
                # Region-dependent method - give informational message
                logger.info(
                    f"Normalization method 'adaptive_regional' produced {n_negative} negative values "
                    f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                    f"This depends on the methods used per region (method_per_region parameter). "
                    f"If mean-centering methods (snv, robust_snv) are used, negatives are expected."
                )
            else:
                # Method not categorized - give general warning
                logger.info(
                    f"Normalization method '{method}' produced {n_negative} negative values "
                    f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                    f"Verify if this is expected for your normalization method."
                )

    # Convert back to polars if input was polars
    if is_polars:
        df = pl.from_pandas(df)

    return df


# ---------------------------------------------------------------------------
#                           MEAN CENTERING
# ---------------------------------------------------------------------------

def mean_center(
    spectra: np.ndarray,
    axis: int = 0,
    return_mean: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Mean-center spectra (essential preprocessing for PCA/PLS).
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    axis : int, default 0
        Axis along which to compute mean.
        - 0: Column-wise (feature/wavenumber centering) - standard for PCA
        - 1: Row-wise (sample centering)
    return_mean : bool, default False
        If True, return the mean array for later reconstruction.
    
    Returns
    -------
    centered : np.ndarray
        Mean-centered spectra.
    mean : np.ndarray, optional
        Mean values (returned if return_mean=True).
    """
    mean = np.mean(spectra, axis=axis, keepdims=True)
    centered = spectra - mean
    
    if return_mean:
        return centered, np.squeeze(mean)
    return centered


def auto_scale(
    spectra: np.ndarray,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Auto-scaling (mean centering + unit variance scaling).
    
    Each variable (wavenumber) is scaled to have mean=0 and std=1.
    Common preprocessing for PCA/PLS when variables have different scales.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    return_params : bool, default False
        If True, return mean and std for reconstruction.
    
    Returns
    -------
    scaled : np.ndarray
        Auto-scaled spectra.
    mean : np.ndarray, optional
    std : np.ndarray, optional
    """
    mean = np.mean(spectra, axis=0)
    std = np.std(spectra, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    scaled = (spectra - mean) / std
    
    if return_params:
        return scaled, mean, std
    return scaled


def pareto_scale(
    spectra: np.ndarray,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pareto scaling (mean centering + sqrt(std) scaling).
    
    Less aggressive than auto-scaling; preserves more of the 
    original data structure. Good for spectral data.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    return_params : bool, default False
        If True, return mean and std for reconstruction.
    
    Returns
    -------
    scaled : np.ndarray
        Pareto-scaled spectra.
    """
    mean = np.mean(spectra, axis=0)
    std = np.std(spectra, axis=0)
    std[std == 0] = 1
    
    scaled = (spectra - mean) / np.sqrt(std)
    
    if return_params:
        return scaled, mean, std
    return scaled


# ---------------------------------------------------------------------------
#                           DETRENDING
# ---------------------------------------------------------------------------

def detrend(
    intensities: np.ndarray,
    order: int = 1,
    wavenumbers: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove polynomial trend from spectrum.

    Often used after SNV to remove residual slope.

    Parameters
    ----------
    intensities : np.ndarray
        1-D spectrum.
    order : int, default 1
        Polynomial order (1 = linear detrending).
    wavenumbers : np.ndarray, optional
        Wavenumber array for physical slope calculation.
        If provided, polynomial fit uses actual wavenumber values (cm⁻¹).
        If None, uses array indices (0, 1, 2, ...).

    Returns
    -------
    np.ndarray
        Detrended spectrum.

    Notes
    -----
    - With wavenumbers: Fit uses physical x-axis (cm⁻¹), yielding physically
      meaningful slope coefficients. Important when comparing spectra on
      different grids or after resampling.
    - Without wavenumbers: Fit uses indices (0, 1, 2, ...), which is
      grid-dependent. Slope coefficients are in units of intensity/index.

    Examples
    --------
    >>> # Index-based detrending (grid-dependent)
    >>> detrended = detrend(spectrum)
    >>>
    >>> # Physical detrending (grid-independent)
    >>> detrended = detrend(spectrum, wavenumbers=wn, order=1)
    """
    if wavenumbers is not None:
        # Use physical wavenumber values for fitting
        if len(wavenumbers) != len(intensities):
            raise ValueError(
                f"Length mismatch: wavenumbers has {len(wavenumbers)} elements "
                f"but intensities has {len(intensities)} elements."
            )
        x = wavenumbers
    else:
        # Use array indices (original behavior)
        x = np.arange(len(intensities))

    coeffs = np.polyfit(x, intensities, order)
    trend = np.polyval(coeffs, x)
    return intensities - trend


def snv_detrend(
    intensities: np.ndarray,
    detrend_order: int = 1,
    wavenumbers: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    SNV followed by detrending.

    Common combined preprocessing for scatter correction.

    Parameters
    ----------
    intensities : np.ndarray
        1-D spectrum.
    detrend_order : int, default 1
        Polynomial order for detrending (1 = linear).
    wavenumbers : np.ndarray, optional
        Wavenumber array for physical slope calculation in detrending.
        If provided, uses actual wavenumber values (cm⁻¹).
        If None, uses array indices.

    Returns
    -------
    np.ndarray
        SNV-normalized and detrended spectrum.
    """
    snv_result = _normalize_snv(intensities)
    return detrend(snv_result, order=detrend_order, wavenumbers=wavenumbers)

# New methods

"""
Novel Normalization Methods for FT-IR Absorbance Spectra
=========================================================

These methods address limitations of traditional approaches by incorporating
spectral physics, robust statistics, and information-theoretic principles.
"""




# ---------------------------------------------------------------------------
#  1. ROBUST SNV (RSNV) - Median/MAD-based SNV
# ---------------------------------------------------------------------------

def normalize_robust_snv(y: np.ndarray, consistency_correction: bool = True, epsilon: float = 1e-10) -> np.ndarray:
    """
    Robust Standard Normal Variate using median and MAD.

    Traditional SNV uses mean and std, which are sensitive to:
    - Baseline artifacts (shifts the mean)
    - Outlier peaks (inflates std)
    - Asymmetric intensity distributions

    RSNV uses median (robust center) and MAD (robust scale).

    Formula: (x - median(x)) / MAD(x)

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    consistency_correction : bool, default True
        If True, scale MAD by 1.4826 to be consistent with std for normal data.
    epsilon : float, default 1e-10
        Small value added to MAD to avoid division by zero and prevent
        zero vectors (which cause issues with cosine-based methods).

    Returns
    -------
    np.ndarray
        Robustly normalized spectrum.

    Reference
    ---------
    Novel method - combines robust statistics with scatter correction.

    Notes
    -----
    When MAD is very small or zero (flat spectrum), epsilon prevents
    returning a zero vector, which would cause errors in downstream
    methods that use cosine similarity (e.g., clustering, PCA with
    cosine kernel).
    """
    median = np.median(y)
    mad = np.median(np.abs(y - median))

    if consistency_correction:
        mad *= 1.4826  # Makes MAD consistent with std for normal distribution

    # Add epsilon to prevent zero division and zero vectors
    # This ensures compatibility with cosine-based methods
    mad = max(mad, epsilon)

    return (y - median) / mad


# ---------------------------------------------------------------------------
#  2. CURVATURE-WEIGHTED NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_curvature_weighted(
    y: np.ndarray,
    sigma: float = 3.0,
    min_weight: float = 0.01
) -> np.ndarray:
    """
    Normalize with weights proportional to local curvature (2nd derivative).

    Physical motivation: In FT-IR, peaks have high curvature while baseline
    regions are flat. This method emphasizes peak regions during normalization,
    making it more representative of actual chemical information.

    The normalization factor is the curvature-weighted L2 norm.

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    sigma : float, default 3.0
        Gaussian smoothing sigma for curvature estimation (reduces noise).
    min_weight : float, default 0.01
        Minimum weight to avoid division issues in flat regions.

    Returns
    -------
    np.ndarray
        Curvature-weighted normalized spectrum.

    Notes
    -----
    Handles flat spectra (zero curvature) by falling back to uniform weighting.
    """
    # Smooth before computing curvature to reduce noise amplification
    y_smooth = ndimage.gaussian_filter1d(y, sigma=sigma)

    # Second derivative (curvature proxy)
    d2y = np.gradient(np.gradient(y_smooth))

    # Weights = |curvature|, normalized
    weights = np.abs(d2y)
    max_weight = np.max(weights)

    if max_weight > 0:
        # Normal case: normalize by max curvature
        weights = weights / max_weight + min_weight
    else:
        # Flat spectrum: use uniform weighting
        weights = np.ones_like(weights) * (1.0 + min_weight)

    # Weighted L2 norm
    weighted_norm = np.sqrt(np.sum(weights * y**2))

    if weighted_norm == 0:
        return np.zeros_like(y)

    return y / weighted_norm


# ---------------------------------------------------------------------------
#  3. PEAK-ENVELOPE NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_peak_envelope(
    y: np.ndarray,
    percentile: float = 95,
    window_size: int = 50
) -> np.ndarray:
    """
    Normalize by the upper envelope of the spectrum.

    Physical motivation: The upper envelope represents the maximum signal
    level across the spectrum, accounting for varying peak densities and
    intensities. This is more representative than single-peak normalization.

    Works on absolute values to handle negative/baseline-corrected spectra.

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    percentile : float, default 95
        Percentile for envelope estimation (avoids noise spikes).
    window_size : int, default 50
        Rolling window size for envelope estimation.

    Returns
    -------
    np.ndarray
        Envelope-normalized spectrum.

    Notes
    -----
    Uses absolute values to avoid NaN issues with negative/baseline-corrected spectra.
    """
    # Work with absolute values to handle negative spectra
    y_abs = np.abs(y)

    n = len(y_abs)
    envelope = np.zeros(n)
    half_win = window_size // 2

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        envelope[i] = np.percentile(y_abs[start:end], percentile)

    # Use median of envelope as normalization factor
    # Filter finite values > 0
    valid_envelope = envelope[np.isfinite(envelope) & (envelope > 0)]

    if len(valid_envelope) == 0:
        # Fallback: use max absolute value
        max_val = np.max(y_abs)
        norm_factor = max_val if max_val > 0 else 1.0
    else:
        norm_factor = np.median(valid_envelope)

    if norm_factor == 0:
        return np.zeros_like(y)

    return y / norm_factor


# ---------------------------------------------------------------------------
#  4. ENTROPY-WEIGHTED NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_entropy_weighted(
    y: np.ndarray,
    n_bins: int = 50,
    window_size: int = 30,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Normalize with weights based on local spectral entropy.
    
    Motivation: Regions with high entropy (high variability/information)
    should contribute more to normalization. Flat baseline regions have
    low entropy and contribute less.
    
    Local entropy is computed in sliding windows using histogram-based
    probability estimation.
    
    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    n_bins : int, default 50
        Number of bins for local histogram.
    window_size : int, default 30
        Window size for local entropy calculation.
    epsilon : float
        Small value to avoid log(0).
    
    Returns
    -------
    np.ndarray
        Entropy-weighted normalized spectrum.
    """
    n = len(y)
    local_entropy = np.zeros(n)
    half_win = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        window = y[start:end]
        
        # Compute local histogram and entropy
        hist, _ = np.histogram(window, bins=min(n_bins, len(window)//2 + 1))
        hist = hist / (hist.sum() + epsilon)
        hist = hist[hist > 0]
        local_entropy[i] = -np.sum(hist * np.log(hist + epsilon))
    
    # Normalize entropy to [0, 1] and use as weights
    weights = (local_entropy - local_entropy.min()) / (local_entropy.max() - local_entropy.min() + epsilon)
    weights = weights + 0.1  # Floor to avoid zero weights
    
    # Weighted normalization
    weighted_norm = np.sqrt(np.sum(weights * y**2))
    
    if weighted_norm == 0:
        return np.zeros_like(y)
    
    return y / weighted_norm


# ---------------------------------------------------------------------------
#  5. PROBABILISTIC QUOTIENT NORMALIZATION (PQN)
# ---------------------------------------------------------------------------

def normalize_pqn(
    y: np.ndarray,
    reference: Optional[np.ndarray] = None,
    reference_type: str = "median"
) -> np.ndarray:
    """
    Probabilistic Quotient Normalization (PQN) for FT-IR spectra.

    IMPORTANT: True PQN requires a reference spectrum from your dataset.
    For single-spectrum processing without a reference, this function
    performs "median scaling" (dividing by median intensity), which is
    NOT true PQN. Use normalize_df() for proper batch PQN.

    Originally from metabolomics (Dieterle et al., 2006), PQN uses
    median fold-change relative to a reference spectrum, which is robust
    to varying numbers/intensities of peaks.

    Physical motivation: Accounts for dilution effects and path length
    variations without being dominated by major peaks.

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    reference : np.ndarray, optional
        Reference spectrum from your dataset.
        - If provided: Performs true PQN using this reference
        - If None: Falls back to "median scaling" (divides by median
          of positive values). This is NOT true PQN!
    reference_type : str, default "median"
        Currently unused. Reserved for future use in batch processing
        via normalize_df() where reference can be computed from dataset.

    Returns
    -------
    np.ndarray
        Normalized spectrum.

    Notes
    -----
    **Single spectrum (reference=None):**
        Returns: y / median(y[y > 0])
        This is "median scaling", not true PQN.

    **With reference spectrum:**
        1. Compute quotients: q[i] = y[i] / reference[i] (where both > 0)
        2. Normalization factor: median(q)
        3. Returns: y / median(q)

    **For batch processing:**
        Use normalize_df() with method='pqn' to automatically compute
        a reference spectrum (median or mean) from your dataset.

    References
    ----------
    Dieterle et al. (2006) Probabilistic quotient normalization as robust
    method to account for dilution of complex biological mixtures.
    Anal Chem 78(13):4281-90.

    Examples
    --------
    >>> # Single spectrum: median scaling (NOT true PQN)
    >>> y_scaled = normalize_pqn(spectrum)  # reference=None
    >>>
    >>> # True PQN: provide reference from dataset
    >>> ref = np.median(all_spectra, axis=0)  # median spectrum
    >>> y_pqn = normalize_pqn(spectrum, reference=ref)
    >>>
    >>> # Batch PQN: use normalize_df() instead
    >>> df_pqn = normalize_df(df, method='pqn')
    """
    if reference is None:
        # FALLBACK: Median scaling for single spectrum
        # This is NOT true PQN - it's just dividing by median intensity
        import warnings
        warnings.warn(
            "PQN called without a reference spectrum. Falling back to median scaling, "
            "which is NOT true PQN. For proper PQN: (1) provide a reference spectrum "
            "via the 'reference' parameter, or (2) use normalize_df() for batch processing "
            "which automatically computes a reference from your dataset.",
            UserWarning,
            stacklevel=2
        )
        ref_value = np.median(y[y > 0]) if np.any(y > 0) else 1.0
        return y / ref_value

    # TRUE PQN: Compute quotients (fold changes) relative to reference
    # Only use positions where both spectra are positive
    mask = (y > 0) & (reference > 0)

    if not np.any(mask):
        return y  # Cannot normalize

    quotients = y[mask] / reference[mask]

    # Median quotient is the normalization factor
    norm_factor = np.median(quotients)

    if norm_factor == 0:
        return np.zeros_like(y)

    return y / norm_factor


# ---------------------------------------------------------------------------
#  6. TOTAL VARIATION NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_total_variation(
    y: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Normalize by total variation (sum of absolute differences).
    
    Physical motivation: Total variation captures the "roughness" or
    total signal content independent of baseline offset. It's related
    to the first derivative energy and is baseline-invariant.
    
    TV = sum(|y[i+1] - y[i]|) for first order
    
    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    order : int, default 1
        Order of differences (1 = first derivative, 2 = second derivative).
    
    Returns
    -------
    np.ndarray
        TV-normalized spectrum.
    """
    diff = y.copy()
    for _ in range(order):
        diff = np.diff(diff)
    
    tv = np.sum(np.abs(diff))
    
    if tv == 0:
        return np.zeros_like(y)
    
    return y / tv


# ---------------------------------------------------------------------------
#  7. SPECTRAL MOMENT NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_spectral_moments(
    y: np.ndarray,
    moment_order: int = 2,
    use_central: bool = True
) -> np.ndarray:
    """
    Normalize using spectral moments.
    
    Physical motivation: Higher-order moments capture distribution
    characteristics beyond simple mean/variance. The nth moment
    emphasizes larger deviations, useful for peak-rich spectra.
    
    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    moment_order : int, default 2
        Order of moment to use (2 = variance-like, 3 = skewness-like, etc.)
    use_central : bool, default True
        If True, use central moments (subtract mean first).
    
    Returns
    -------
    np.ndarray
        Moment-normalized spectrum.
    """
    if use_central:
        y_centered = y - np.mean(y)
    else:
        y_centered = y
    
    # Compute nth moment
    moment = np.mean(np.abs(y_centered) ** moment_order) ** (1.0 / moment_order)
    
    if moment == 0:
        return np.zeros_like(y)
    
    if use_central:
        return y_centered / moment
    return y / moment


# ---------------------------------------------------------------------------
#  8. ADAPTIVE REGIONAL NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_adaptive_regional(
    y: np.ndarray,
    wavenumbers: Optional[np.ndarray],
    regions: Optional[list] = None,
    method_per_region: Optional[dict] = None
) -> np.ndarray:
    """
    Apply different normalization to different spectral regions.

    Physical motivation: Different FT-IR regions have different
    characteristics:
    - 3600-2800 cm⁻¹: O-H, N-H, C-H stretching (often intense)
    - 1800-1500 cm⁻¹: C=O, C=C, amide bands
    - 1500-400 cm⁻¹: Fingerprint region (complex)

    Each region may benefit from different normalization.

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    wavenumbers : np.ndarray, optional
        Wavenumber axis. Required for this method.
    regions : list of tuples, optional
        [(start1, end1), (start2, end2), ...] defining regions.
        Default: standard FT-IR regions.
    method_per_region : dict, optional
        {"region_idx": "method_name"} mapping.
        Default: SNV for all regions.

    Returns
    -------
    np.ndarray
        Regionally-normalized spectrum.

    Raises
    ------
    ValueError
        If wavenumbers is None.
    """
    if wavenumbers is None:
        raise ValueError(
            "adaptive_regional normalization requires wavenumbers array. "
            "Provide wavenumbers to normalize() function."
        )

    if len(wavenumbers) != len(y):
        raise ValueError(
            f"Length mismatch: wavenumbers has {len(wavenumbers)} elements "
            f"but spectrum has {len(y)} elements."
        )

    if regions is None:
        # Default FT-IR regions
        regions = [
            (3600, 2800),  # X-H stretching
            (1800, 1500),  # Double bonds
            (1500, 400),   # Fingerprint
        ]
    
    if method_per_region is None:
        method_per_region = {i: "snv" for i in range(len(regions))}
    
    y_normalized = y.copy()
    
    for i, (start, end) in enumerate(regions):
        mask = (wavenumbers >= min(start, end)) & (wavenumbers <= max(start, end))
        
        if not np.any(mask):
            continue
        
        region_data = y[mask]
        method = method_per_region.get(i, "snv")
        
        if method == "snv":
            mean, std = np.mean(region_data), np.std(region_data)
            if std > 0:
                y_normalized[mask] = (region_data - mean) / std
        elif method == "minmax":
            rmin, rmax = np.min(region_data), np.max(region_data)
            if rmax > rmin:
                y_normalized[mask] = (region_data - rmin) / (rmax - rmin)
        elif method == "robust":
            y_normalized[mask] = normalize_robust_snv(region_data)
    
    return y_normalized


# ---------------------------------------------------------------------------
#  9. DERIVATIVE RATIO NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_derivative_ratio(
    y: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Normalize using the ratio of derivative energies.
    
    Physical motivation: The ratio of second to first derivative
    energy characterizes peak sharpness independent of intensity.
    This provides baseline-independent normalization.
    
    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    sigma : float
        Smoothing parameter before derivative computation.
    
    Returns
    -------
    np.ndarray
        Derivative-ratio normalized spectrum.
    """
    y_smooth = ndimage.gaussian_filter1d(y, sigma=sigma)
    
    d1 = np.gradient(y_smooth)
    d2 = np.gradient(d1)
    
    # Energy of derivatives
    e1 = np.sqrt(np.mean(d1**2))
    e2 = np.sqrt(np.mean(d2**2))
    
    # Combined normalization factor
    if e1 == 0:
        return np.zeros_like(y)
    
    # Use geometric mean of derivative energies
    norm_factor = np.sqrt(e1 * e2) if e2 > 0 else e1
    
    return y / norm_factor


# ---------------------------------------------------------------------------
#  10. SIGNAL-TO-BASELINE RATIO NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_signal_to_baseline(
    y: np.ndarray,
    baseline_percentile: float = 10,
    signal_percentile: float = 90
) -> np.ndarray:
    """
    Normalize by the ratio of signal to baseline levels.
    
    Physical motivation: This separates the "signal" (peaks) from
    "baseline" (background) and normalizes by their contrast. Useful
    when baseline levels vary between samples.
    
    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    baseline_percentile : float
        Percentile to estimate baseline level.
    signal_percentile : float
        Percentile to estimate signal level.
    
    Returns
    -------
    np.ndarray
        Signal-to-baseline normalized spectrum.
    """
    baseline_level = np.percentile(y, baseline_percentile)
    signal_level = np.percentile(y, signal_percentile)
    
    contrast = signal_level - baseline_level
    
    if contrast <= 0:
        return np.zeros_like(y)
    
    # Shift to baseline and normalize by contrast
    return (y - baseline_level) / contrast


# ---------------------------------------------------------------------------
#  CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def normalize_novel(
    y: np.ndarray,
    method: str = "robust_snv",
    **kwargs
) -> np.ndarray:
    """
    Apply novel normalization method.
    
    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    method : str
        One of: 'robust_snv', 'curvature', 'envelope', 'entropy',
        'pqn', 'total_variation', 'moments', 'derivative_ratio',
        'signal_baseline'
    **kwargs : method-specific parameters
    
    Returns
    -------
    np.ndarray
        Normalized spectrum.
    """
    methods = {
        'robust_snv': normalize_robust_snv,
        'curvature': normalize_curvature_weighted,
        'envelope': normalize_peak_envelope,
        'entropy': normalize_entropy_weighted,
        'pqn': normalize_pqn,
        'total_variation': normalize_total_variation,
        'moments': normalize_spectral_moments,
        'derivative_ratio': normalize_derivative_ratio,
        'signal_baseline': normalize_signal_to_baseline,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods.keys())}")
    
    return methods[method](y, **kwargs)


def novel_normalize_method_names() -> list:
    """Return list of novel normalization method names."""
    return [
        'robust_snv', 'curvature', 'envelope', 'entropy',
        'pqn', 'total_variation', 'moments', 'derivative_ratio',
        'signal_baseline'
    ]