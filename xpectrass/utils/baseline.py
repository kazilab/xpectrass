"""
Baseline Correction Module for FTIR Spectral Preprocessing
==========================================================

Provides baseline correction using 50+ algorithms from pybaselines library
plus custom windowed filters for FTIR and ToF-SIMS spectra.

**IMPORTANT**: This module expects absorbance data (AU), not transmittance (%).
Convert transmittance to absorbance first using convert_spectra() from trans_abs.py

Features:
- Single spectrum correction via baseline_correction()
- Batch DataFrame processing via apply_baseline_correction()
- Automatic column detection and sorting by wavenumber
- Performance optimized for large datasets (vectorized operations)
- Pandas and Polars DataFrame support
- Method evaluation via evaluate_baseline_correction_methods()

Logging:
This module uses Python's logging module for warnings and informational messages.
Configure the logger to control output:

    import logging
    logging.getLogger('utils.baseline').setLevel(logging.INFO)  # Show all messages
    logging.getLogger('utils.baseline').setLevel(logging.ERROR)  # Only errors

Available Methods:
Run baseline_method_names() to see all available correction algorithms.
Common methods: airpls, asls, arpls, iarpls, drpls, mor, snip, poly
"""

from __future__ import annotations   # for Python < 3.11 type-hinting
from typing import Tuple, Union, List, Optional
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from pybaselines import Baseline
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from contextlib import contextmanager

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

# --------------------------------------------------------------------------- #
#                    UNIVERSAL BASELINE-CORRECTION WRAPPER                    #
# --------------------------------------------------------------------------- #
def baseline_correction(
    intensities: Union[np.ndarray, list, tuple],
    wavenumbers: Optional[Union[np.ndarray, list, tuple]] = None,
    method: str = "airpls",
    window_size: int = 101,
    poly_order: int = 4,
    clip_negative: bool = False,
    return_baseline: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Baseline-correct a 1-D FT-IR or ToF-SIMS spectrum with >50 algorithms.

    Parameters
    ----------
    intensities : array-like
        Raw y-values (%T or absorbance); will be converted to ``float64``.
    wavenumbers : array-like, optional
        X-axis values (wavenumbers in cm⁻¹). If provided, passed to pybaselines
        for correct spacing. If None, assumes uniform spacing with dx=1.
        For FTIR spectra, should match the length of intensities.
    method : str, default "airpls"
        Name of the baseline algorithm.  All **pybaselines** methods plus
        two custom filters ("median_filter", "adaptive_window") are accepted.
    window_size : int, default 101
        Odd kernel width for the two custom windowed filters.
    poly_order : int, default 4
        Polynomial order for the `"poly"` baseline.
    clip_negative : bool, default False
        If *True*, set negative corrected values to 0 (useful for %T spectra).
    return_baseline : bool, default False
        If *True*, return ``(corrected, baseline)`` instead of just ``corrected``.
    **kwargs :
        Extra keyword arguments are forwarded verbatim to the selected
        **pybaselines** algorithm (e.g. ``lam=1e6, p=0.01`` for AsLS).

    Returns
    -------
    corrected : np.ndarray
        Baseline-subtracted intensities (same dtype & length as input).
    baseline : np.ndarray , optional
        Returned *only* if ``return_baseline=True``.

    Notes
    -----
    NaN Handling:
        - If input contains NaN values, they are temporarily removed for baseline estimation
        - Baseline is computed only on finite values
        - NaN positions are preserved in output (marked as NaN)
        - If all values are NaN, returns array of NaN
    """
    # Suppress warnings from pybaselines algorithms
    warnings.filterwarnings('ignore')

    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("`intensities` must be a 1-D array-like object.")

    # Handle wavenumbers (x-axis)
    if wavenumbers is not None:
        x = np.asarray(wavenumbers, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("`wavenumbers` must be a 1-D array-like object.")
        if len(x) != len(y):
            raise ValueError("`wavenumbers` and `intensities` must have the same length.")
    else:
        x = None

    # Handle NaN values: preserve positions but compute baseline on finite values only
    nan_mask = ~np.isfinite(y)
    has_nans = np.any(nan_mask)

    if has_nans:
        # If all values are NaN, return NaN array
        if np.all(nan_mask):
            baseline = np.full_like(y, np.nan)
            corrected = np.full_like(y, np.nan)
            return (corrected, baseline) if return_baseline else corrected

        # Store original array and work with finite values only
        y_original = y.copy()
        y_finite = y[~nan_mask]

        # Also filter wavenumbers if provided
        if x is not None:
            x_original = x.copy()
            x_finite = x[~nan_mask]
        else:
            x_finite = None
    else:
        y_finite = y
        x_finite = x

    # --------------------------------------------------------------------- #
    #                Dispatch table for pybaselines algorithms              #
    # --------------------------------------------------------------------- #

    # Pass x_data to Baseline if available
    bl = Baseline(x_data=x_finite) if x_finite is not None else Baseline()
    _skip = {"pentapy_solver", "banded_solver"}
    pyb_dispatch = {}
    for name in dir(bl):
        if name.startswith("_") or name in _skip:
            continue
        attr = getattr(bl, name)
        if callable(attr):
            pyb_dispatch[name] = attr
    # Add a convenience alias for polynomial fits
    pyb_dispatch["poly"] = lambda arr, *, poly_order=poly_order, **k: bl.poly(
        arr, poly_order=poly_order, **k
    )

    # --------------------------------------------------------------------- #
    #                          Custom windowed filters                      #
    # --------------------------------------------------------------------- #
    try:
        if method == "median_filter":
            baseline_finite = signal.medfilt(y_finite, kernel_size=window_size)
        elif method == "adaptive_window":
            baseline_finite = ndimage.minimum_filter1d(y_finite, size=window_size)
        elif method in pyb_dispatch:
            # pybaselines: -> (baseline, params_dict)
            baseline_finite, _ = pyb_dispatch[method](y_finite, **kwargs)
        else:
            raise ValueError(
                f"Unknown baseline method '{method}'. Check pybaselines docs "
                "or spell-check your custom method name."
            )
    except Exception as e:
        if isinstance(e, ValueError) and "Unknown baseline method" in str(e):
            raise  # Re-raise our own ValueError
        # Provide informative error message for pybaselines failures
        raise RuntimeError(
            f"Baseline correction failed for method '{method}'. "
            f"Error: {type(e).__name__}: {str(e)}. "
            f"Check parameter compatibility with pybaselines documentation."
        ) from e

    # Compute corrected spectrum on finite values
    corrected_finite = y_finite - baseline_finite
    if clip_negative:
        corrected_finite[corrected_finite < 0] = 0.0

    # If we had NaN values, restore them in the output
    if has_nans:
        # Create full-length arrays with NaN at original positions
        corrected = np.full_like(y_original, np.nan)
        baseline = np.full_like(y_original, np.nan)

        # Fill in computed values at finite positions
        corrected[~nan_mask] = corrected_finite
        baseline[~nan_mask] = baseline_finite
    else:
        corrected = corrected_finite
        baseline = baseline_finite

    return (corrected, baseline) if return_baseline else corrected


# --------------------------------------------------------------------------- #
#                    DATAFRAME-COMPATIBLE BATCH BASELINE CORRECTION           #
# --------------------------------------------------------------------------- #

def apply_baseline_correction(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    method: str = "airpls",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    window_size: int = 101,
    poly_order: int = 4,
    clip_negative: bool = False,
    show_progress: bool = True,
    **kwargs
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """
    Apply baseline correction to a DataFrame of FTIR spectra (batch processing).

    Works with both pandas and polars DataFrames. Each row is a sample,
    numerical columns are wavenumbers. Applies baseline correction to all samples.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format DataFrame where rows = samples, columns = wavenumbers.
        Should contain numerical columns with spectral data and optional
        metadata columns (e.g., 'sample', 'label').
    method : str, default "airpls"
        Baseline correction method. All pybaselines methods plus custom filters
        ("median_filter", "adaptive_window", "poly") are supported.
        Common methods: "airpls", "asls", "arpls", "iarpls", "drpls", "iasls",
        "aspls", "psalsa", "derpsalsa", "mpls", "mor", "imor", "amormol", "snip"
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
    window_size : int, default 101
        Odd kernel width for custom windowed filters ("median_filter", "adaptive_window").
    poly_order : int, default 4
        Polynomial order for the "poly" baseline method.
    clip_negative : bool, default False
        If True, set negative corrected values to 0 (useful for %T spectra).
    show_progress : bool, default True
        If True, display a progress bar during processing.
    **kwargs : additional parameters
        Extra keyword arguments forwarded to the baseline correction algorithm
        (e.g., lam=1e6, p=0.01 for AsLS/AirPLS methods).

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        Baseline-corrected DataFrame (same type as input) with spectral data
        corrected and metadata columns preserved. Output columns are sorted
        by ascending wavenumber for standardization.

    NaN Handling
    ------------
    Robustly handles NaN (missing) values in spectral data:
    - NaN values are temporarily removed before baseline estimation
    - Baseline is computed only on finite values
    - NaN positions are preserved in output
    - If an entire spectrum is NaN, it remains as NaN
    - Prevents baseline algorithms from failing on sparse/incomplete data

    Performance
    -----------
    Optimized for large datasets using:
    - Robust wavenumber column detection (parses column names, not dtype)
    - Automatic column sorting to ensure monotonic wavenumber order
    - Vectorized numpy array access (no DataFrame.loc overhead)
    - Pre-allocated output arrays (no dynamic list appending)
    - Progress tracking via tqdm

    Warnings
    --------
    - Warns if spectral columns are reordered during processing
    - Warns if wavenumber bounds are auto-expanded beyond defaults (200-8000 cm⁻¹)

    Examples
    --------
    >>> # Apply AirPLS baseline correction to all samples
    >>> df_corrected = apply_baseline_correction(df_wide, method="airpls")

    >>> # Use AsLS with custom parameters
    >>> df_corrected = apply_baseline_correction(
    ...     df_wide,
    ...     method="asls",
    ...     lam=1e6,
    ...     p=0.01
    ... )

    >>> # Use polynomial baseline
    >>> df_corrected = apply_baseline_correction(
    ...     df_wide,
    ...     method="poly",
    ...     poly_order=3
    ... )

    >>> # Works with both pandas and polars
    >>> df_pd_corrected = apply_baseline_correction(df_pandas)
    >>> df_pl_corrected = apply_baseline_correction(df_polars)

    >>> # Disable progress bar for cleaner output
    >>> df_corrected = apply_baseline_correction(df_wide, show_progress=False)

    >>> # Works correctly even with NaN values in spectra
    >>> df_with_nans = df_wide.copy()
    >>> df_with_nans.iloc[0, 10:20] = np.nan  # Introduce NaN values
    >>> df_corrected = apply_baseline_correction(df_with_nans, method="airpls")
    # NaN positions are preserved in output
    """
    # Determine if input is polars or pandas (polars is optional)
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
                f"Baseline correction should be performed on absorbance for physical validity. "
                f"Please convert your data first using: "
                f"convert_spectra(data, mode='to_absorbance') from trans_abs.py"
            )

    corrected_data = np.empty((n_samples, n_wavenumbers), dtype=np.float64)

    # Apply baseline correction to each sample with progress bar
    iterator = tqdm(
        range(n_samples),
        desc=f"Baseline correction ({method})",
        disable=not show_progress,
        dynamic_ncols=True
    )

    for i in iterator:
        intensities = spectral_data[i, :]

        # Apply baseline correction (pass wavenumbers for correct spacing)
        corrected_data[i, :] = baseline_correction(
            intensities=intensities,
            wavenumbers=sorted_wavenumbers,
            method=method,
            window_size=window_size,
            poly_order=poly_order,
            clip_negative=clip_negative,
            return_baseline=False,
            **kwargs
        )

    # PHYSICAL CONSTRAINT VALIDATION: Check for negative absorbance after correction
    # Note: clip_negative parameter may already handle this, but warn if it occurs before clipping
    if not clip_negative:
        finite_mask = np.isfinite(corrected_data)
        if np.any(finite_mask):
            n_negative = np.sum(corrected_data[finite_mask] < 0)
            if n_negative > 0:
                min_negative = np.min(corrected_data[finite_mask])
                pct_negative = 100.0 * n_negative / np.sum(finite_mask)
                logger.warning(
                    f"Baseline correction produced {n_negative} negative absorbance values "
                    f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                    f"This is physically invalid. The baseline may be over-estimated. "
                    f"Recommendations: (1) Set clip_negative=True to clip values to 0, "
                    f"(2) Try different baseline method (e.g., 'asls', 'arpls'), "
                    f"or (3) Adjust method parameters (e.g., increase lam, decrease p)."
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


# --------------------------------------------------------------------------- #
#                    CHECK AVAILABLE BASELINE-CORRECTION METHODS              #
# --------------------------------------------------------------------------- #


def baseline_method_names() -> list[str]:
    """
    Return a sorted list of method names that can be passed to
    `baseline_correction(method=...)`.

    The list is generated dynamically from `pybaselines.Baseline`,
    skipping the deprecated solver helpers, and then augmented with the
    two custom windowed filters plus the convenient 'poly' alias.
    """
    bl = Baseline()
    skip = {"pentapy_solver", "banded_solver"}           # solver helpers

    # collect every public callable attribute name
    methods = {
        name
        for name in dir(bl)
        if (
            not name.startswith("_")
            and name not in skip
            and callable(getattr(bl, name))
        )
    }

    # add our wrapper-specific extras
    methods.update({"median_filter", "adaptive_window", "poly"})

    # Remove experimental/unstable methods that may fail in 1-D context:
    # - 'collab_pls': Requires 2-D collaboration across multiple spectra
    # - 'interp_pts': May fail with automatic point selection on diverse spectra
    # - 'cwt_br': Wavelet-based method with inconsistent behavior on edge cases
    values_to_remove = ['collab_pls', 'interp_pts', 'cwt_br']
    methods = [x for x in methods if x not in values_to_remove]
    return sorted(methods)


# -------------------------------------------------------------------------
#  TEST BASELINE CORRECTION METHODS
#  WE WILL USE RELATIVE METHODS
#  CALCULATE Residual Flat-Zone Noise (RFZN) and Negative Area Ratio (NAR)
# -------------------------------------------------------------------------

def _make_mask(x_axis: np.ndarray, flat_windows: List[Tuple[float, float]]
               ) -> np.ndarray:
    """Boolean mask for baseline-only wavenumber regions."""
    mask = np.zeros_like(x_axis, dtype=bool)
    for lo, hi in flat_windows:
        mask |= (x_axis >= lo) & (x_axis <= hi)
    return mask


##### PLOT RFZN, NAR and SNR values #####

def plot_baseline_correction_metric_boxes(
    df: pd.DataFrame,
    metric_name: str,
    figsize: tuple[int, int] = (9, 5),
    mean_bar_width: float = 0.6,
    color_boxes: str | None = None,
    color_mean: str | None = None,
    plot_mean_sd: bool = False,
    save_plot: bool = False,
    save_path: str = '',
) -> None:
    """
    Box-plot of a baseline-quality metric (`RFZN` or `NAR`) across methods.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows = samples, columns = baseline-correction methods.
    metric_name : str
        Title for the y-axis and plot.
    figsize : (w, h), default (9, 5)
        Size of the figure in inches.
    mean_bar_width : float, default 0.6
        Width of the mean ± SD bar overlay (same units as box widths).
    color_boxes : str | None
        Matplotlib colour for the boxes.  *None* → Matplotlib default cycle.
    color_mean : str | None
        Colour for the mean ± SD bars.  *None* → Matplotlib default cycle.
    """
    # Suppress warnings in plotting function
    warnings.filterwarnings('ignore')

    # Prepare data ──────────────────────────────────────────────────────
    # keep numerical columns only, drop all-NaN cols (failed algorithms)
    df_num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if df_num.empty:
        raise ValueError("No numeric columns to plot.")

    methods = df_num.columns.to_list()
    data    = [df_num[m].dropna().values for m in methods]

    # Create boxplot ────────────────────────────────────────────────────
    plt.figure(figsize=figsize)
    box = plt.boxplot(
        data,
        vert=True,
        patch_artist=True,
        labels=methods,
        showfliers=False,        # hide outliers for clarity
        widths=mean_bar_width,
        medianprops=dict(color="black", lw=1.2),
        boxprops=dict(facecolor=color_boxes or "tab:blue", alpha=0.35),
    )
    
    # Overlay mean ± SD per method ──────────────────────────────────────
    if plot_mean_sd:
        x_positions = np.arange(1, len(methods) + 1)
        means = [np.mean(d) for d in data]
        sds   = [np.std(d, ddof=1) for d in data]
        
        plt.errorbar(
            x_positions,
            means,
            yerr=sds,
            fmt="o",
            color=color_mean or "tab:red",
            ecolor=color_mean or "tab:red",
            elinewidth=2,
            capsize=4,
            linewidth=2,
            label="mean ± 1 SD",
        )
    
    # Cosmetics ─────────────────────────────────────────────────────────
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} distribution across baseline methods")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.legend(frameon=False)
    if save_plot:
        plt.savefig(save_path+metric_name+'_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
# plot_metric_boxes(rfzn, metric_name="RFZN")
# plot_metric_boxes(nar,  metric_name="NAR")


#### Plot masked samples ####

def plot_baseline_correction_metric_boxes_masked(
    df: pd.DataFrame,
    metric_name: str,
    max_value: float,
    figsize: tuple[int, int] = (9, 5),
    mean_bar_width: float = 0.6,
    color_boxes: str | None = None,
    color_mean: str | None = None,
    plot_mean_sd: bool = False,
    save_plot: bool = False,
    save_path: str = 'masked_',
) -> None:

    good_mask = (df.fillna(np.inf) <= max_value).all(axis=0) # axis=0  ⇒  column-wise “all”
    good_methods = good_mask.index[good_mask].tolist()
    # print(f"Methods that always pass {metric_name} ≤ {max_value}:", good_methods)
    df_good = df[good_methods]
    plot_baseline_correction_metric_boxes(
                      df_good,
                      metric_name,
                      figsize,
                      mean_bar_width,
                      color_boxes,
                      color_mean,
                      plot_mean_sd,
                      save_plot,
                      save_path
                      )


# -------------------------------------------------------------------------
#  TEST BASELINE CORRECTION METHODS
#  ADD: Signal-to-Noise Ratio (SNR)
# -------------------------------------------------------------------------


def _make_mask(x_axis: np.ndarray,
               flat_windows: List[Tuple[float, float]]) -> np.ndarray:
    """Boolean mask for baseline-only wavenumber regions."""
    mask = np.zeros_like(x_axis, dtype=bool)
    for lo, hi in flat_windows:
        mask |= (x_axis >= lo) & (x_axis <= hi)
    return mask


def _score_one_sample(
    sample_name: str,
    y_raw: np.ndarray,
    methods: List[str],
    mask: np.ndarray,
    negative_clip: bool,
    wavenumbers: Optional[np.ndarray] = None,
    diagnostic_peaks: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker: returns RFZN, NAR, SNR vectors for one spectrum.

    Parameters
    ----------
    sample_name : str
        Sample identifier
    y_raw : np.ndarray
        Raw spectrum intensities
    methods : list[str]
        Baseline correction methods to evaluate
    mask : np.ndarray
        Boolean mask for flat-zone regions (noise calculation)
    negative_clip : bool
        Whether to clip negative values in corrected spectra
    wavenumbers : np.ndarray, optional
        Wavenumber array (required if diagnostic_peaks is specified)
    diagnostic_peaks : list of tuples, optional
        Specific wavenumber ranges for peak detection as (min, max) tuples.
        Example: [(2900, 2930)] for CH stretch region around 2916 cm⁻¹.
        If None, uses global maximum across entire spectrum.

    Returns
    -------
    sample_name : str
        Sample identifier (passed through)
    rfzn_row : np.ndarray
        RFZN values for each method (intensity units)
    nar_row : np.ndarray
        NAR values for each method (unitless, 0-1)
    snr_row : np.ndarray
        SNR values for each method (unitless ratio)

    Notes
    -----
    SNR Calculation:
        - SNR = peak_height / sigma_noise
        - peak_height: Maximum absolute intensity in diagnostic regions (or global)
        - sigma_noise: RMS noise in flat zones
    """
    # Suppress warnings in worker function (runs in parallel process)
    warnings.filterwarnings('ignore')

    n_methods = len(methods)
    rfzn_row = np.empty(n_methods, dtype=float)
    nar_row  = np.empty_like(rfzn_row)
    snr_row  = np.empty_like(rfzn_row)

    for j, m in enumerate(methods):
        try:
            y_corr = baseline_correction(
                y_raw, method=m, clip_negative=negative_clip
            )

            # ----------------- noise & peak (NaN-aware) ---------------------
            # Use nanmean for noise to handle NaN in flat zones
            flat_zone_values = y_corr[mask]
            if np.all(~np.isfinite(flat_zone_values)):
                # All flat zone values are NaN
                sigma_noise = np.nan
            else:
                sigma_noise = np.sqrt(np.nanmean(flat_zone_values ** 2))

            # Use nanmax for peak height (global or diagnostic regions)
            if diagnostic_peaks is not None and wavenumbers is not None:
                # Use specific diagnostic peak regions
                peak_heights = []
                for peak_min, peak_max in diagnostic_peaks:
                    peak_mask = (wavenumbers >= peak_min) & (wavenumbers <= peak_max)
                    if np.any(peak_mask):
                        peak_region = y_corr[peak_mask]
                        if np.any(np.isfinite(peak_region)):
                            peak_heights.append(np.nanmax(np.abs(peak_region)))

                if peak_heights:
                    peak_height = max(peak_heights)
                else:
                    peak_height = np.nan
            else:
                # Global maximum across entire spectrum
                if np.all(~np.isfinite(y_corr)):
                    peak_height = np.nan
                else:
                    peak_height = np.nanmax(np.abs(y_corr))

            # ----------------- RFZN ----------------------------------------
            rfzn_row[j] = sigma_noise            # identical definition
            # ----------------- NAR (NaN-aware) -----------------------------
            # Handle case where all values are NaN or zero
            finite_values = y_corr[np.isfinite(y_corr)]
            if len(finite_values) == 0:
                nar_row[j] = np.nan
            else:
                neg_area = np.sum(-finite_values[finite_values < 0.0])
                total_area = np.sum(np.abs(finite_values))
                nar_row[j] = neg_area / total_area if total_area > 0 else 0.0

            # ----------------- SNR (NaN-aware) -----------------------------
            if not np.isfinite(sigma_noise) or not np.isfinite(peak_height):
                snr_row[j] = np.nan
            elif sigma_noise > 0:
                snr_row[j] = peak_height / sigma_noise
            else:
                snr_row[j] = np.nan

        except Exception:
            rfzn_row[j] = np.nan
            nar_row[j]  = np.nan
            snr_row[j]  = np.nan

    return sample_name, rfzn_row, nar_row, snr_row


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


def evaluate_baseline_correction_methods(
    data: Union[pd.DataFrame, "pl.DataFrame"],
    flat_windows: List[Tuple[float, float]],
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    negative_clip: bool = False,
    diagnostic_peaks: Optional[List[Tuple[float, float]]] = None,
    baseline_methods: Optional[List[str]] = None,  # None = all methods
    n_samples: Optional[int] = None,               # None = use all samples
    sample_selection: str = "random",              # "random", "first", "last"
    random_state: Optional[int] = None,            # for reproducibility
    n_jobs: int = -1,                              # -1 → all CPU cores
):
    """
    Parallel computation of RFZN, NAR, SNR for every (sample, method) pair.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Wide-format DataFrame where rows = samples, columns = wavenumbers.
        Should contain numerical columns with spectral data and optional
        metadata columns (e.g., 'sample', 'label').
    flat_windows : list of tuples
        Wavenumber ranges to use for baseline noise evaluation.
        Each tuple is (min_wavenumber, max_wavenumber) for regions
        expected to contain only baseline (no peaks).
    label_column : str, default "label"
        Name of the label/group column to exclude from evaluation.
    exclude_columns : list[str], optional
        Additional column names to exclude from evaluation (e.g., 'sample', 'id').
    wn_min : float, optional
        Minimum wavenumber for column detection (default: 200.0 cm⁻¹).
    wn_max : float, optional
        Maximum wavenumber for column detection (default: 8000.0 cm⁻¹).
    negative_clip : bool, default False
        If True, clip negative values to 0 during baseline correction.
    diagnostic_peaks : list of tuples, optional
        Specific wavenumber ranges for peak detection in SNR calculation.
        Each tuple is (min_wavenumber, max_wavenumber) for diagnostic peaks.
        Example: [(2900, 2930), (2840, 2870)] for CH2/CH3 stretch regions.
        If None, uses global maximum across entire spectrum.
    baseline_methods : list of str, optional
        List of baseline correction methods to evaluate. If None (default),
        evaluates all available methods from baseline_method_names().
        Providing a subset significantly speeds up evaluation.
        Example: ['als', 'asls', 'arpls'] to test only ALS variants.
        Use baseline_method_names() to see all available methods.
    n_samples : int, optional
        Number of samples to evaluate. If None, evaluates all samples.
    sample_selection : str, default "random"
        How to select samples if n_samples < total samples.
        Options: "random", "first", "last".
    random_state : int, optional
        Random seed for reproducible sample selection when using "random".
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all CPU cores.

    Returns
    -------
    rfzn_tbl : pandas.DataFrame
        Residual Flat-Zone Noise (RFZN) values for each (sample, method) pair.
        Units: Same as input spectral intensities (e.g., absorbance, %T).
        Lower values indicate better baseline correction (less residual noise).
    nar_tbl : pandas.DataFrame
        Negative Area Ratio (NAR) values for each (sample, method) pair.
        Units: Unitless ratio in range [0, 1].
        Ratio of negative area to total area after baseline correction.
        Lower values indicate better correction (fewer negative artifacts).
    snr_tbl : pandas.DataFrame
        Signal-to-Noise Ratio (SNR) values for each (sample, method) pair.
        Units: Unitless ratio (peak_height / noise_level).
        Higher values indicate better correction (stronger signal relative to noise).

    Notes
    -----
    Metric Interpretations:
        - **RFZN**: RMS noise in flat zones. Good methods: < 0.01 (absorbance units)
        - **NAR**: Fraction of negative intensity. Good methods: < 0.05 (5%)
        - **SNR**: Peak/noise ratio. Good methods: > 10 (depends on sample)

    Example Usage:
        >>> flat_windows = [(2500, 2600), (3200, 3500)]  # Baseline-only regions
        >>>
        >>> # Evaluate all available methods
        >>> rfzn, nar, snr = evaluate_baseline_correction_methods(
        ...     df, flat_windows, diagnostic_peaks=[(2900, 2930)]
        ... )
        >>>
        >>> # Evaluate only specific methods (faster)
        >>> rfzn, nar, snr = evaluate_baseline_correction_methods(
        ...     df, flat_windows,
        ...     baseline_methods=['als', 'asls', 'arpls', 'rubberband']
        ... )
        >>>
        >>> # Find best method for each sample
        >>> best_methods = rfzn.idxmin(axis=1)  # Method with lowest noise per sample
    """
    # Suppress warnings in this function and all child processes
    warnings.filterwarnings('ignore')

    # Convert to pandas for processing
    is_polars = (pl is not None) and isinstance(data, pl.DataFrame)
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Identify columns to exclude
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

    # ---------------- preparation ----------------------------------------
    all_methods = baseline_method_names()  # All available baseline methods

    # Filter methods if baseline_methods parameter provided
    if baseline_methods is None:
        methods = all_methods  # Use all methods
    else:
        # Validate that all requested methods exist
        invalid_methods = [m for m in baseline_methods if m not in all_methods]
        if invalid_methods:
            raise ValueError(
                f"Invalid baseline method(s): {invalid_methods}. "
                f"Available methods: {all_methods}. "
                f"Use baseline_method_names() to see all valid options."
            )
        methods = baseline_methods  # Use only requested methods

    samples = df.index.tolist()

    # Validate index uniqueness (required for result DataFrame construction)
    if len(samples) != len(set(samples)):
        raise ValueError(
            "DataFrame index must be unique for evaluation. "
            "Found duplicate index values. Use df.reset_index() to create unique indices."
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


    # Use sorted wavenumbers for consistency
    mask = _make_mask(sorted_wavenumbers, flat_windows)

    # Extract spectra to NumPy once (parent process) - use sorted columns
    spectral_data = df[sorted_cols].values.astype(np.float64)
    spectra = {
        samples[i]: spectral_data[i, :]
        for i in range(len(samples))
    }

    # ---------------- parallel loop --------------------------------------
    worker = joblib.delayed(_score_one_sample)

    with _tqdm_joblib(tqdm(desc="baseline eval", total=len(samples), dynamic_ncols=True)) as _:
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            worker(
                s,
                spectra[s],
                methods,
                mask,
                negative_clip,
                sorted_wavenumbers,
                diagnostic_peaks
            )
            for s in samples
        )

    # ---------------- assemble -------------------------------------------
    rfzn_arr = np.vstack([row[1] for row in results])
    nar_arr  = np.vstack([row[2] for row in results])
    snr_arr  = np.vstack([row[3] for row in results])

    rfzn_tbl = pd.DataFrame(rfzn_arr, index=samples, columns=methods, dtype=float)
    nar_tbl  = pd.DataFrame(nar_arr,  index=samples, columns=methods, dtype=float)
    snr_tbl  = pd.DataFrame(snr_arr,  index=samples, columns=methods, dtype=float)

    return rfzn_tbl, nar_tbl, snr_tbl


def find_best_baseline_method(
    rfzn_tbl: pd.DataFrame,
    nar_tbl: pd.DataFrame,
    snr_tbl: pd.DataFrame,
    rfzn_threshold: float = 0.01,
    nar_threshold: float = 0.05,
    snr_min: float = 10.0,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend best baseline correction methods based on evaluation metrics.

    Analyzes RFZN, NAR, and SNR across all samples to identify methods that
    consistently perform well. Methods are ranked by a composite score combining
    all three metrics.

    Parameters
    ----------
    rfzn_tbl : pd.DataFrame
        RFZN values from evaluate_baseline_correction_methods()
    nar_tbl : pd.DataFrame
        NAR values from evaluate_baseline_correction_methods()
    snr_tbl : pd.DataFrame
        SNR values from evaluate_baseline_correction_methods()
    rfzn_threshold : float, default 0.01
        Maximum acceptable RFZN (lower is better). Default: 0.01 absorbance units.
    nar_threshold : float, default 0.05
        Maximum acceptable NAR (lower is better). Default: 0.05 (5%).
    snr_min : float, default 10.0
        Minimum acceptable SNR (higher is better). Default: 10.
    top_n : int, default 5
        Number of top methods to return.

    Returns
    -------
    pd.DataFrame
        Ranked methods with columns:
        - method: Method name
        - median_rfzn: Median RFZN across samples
        - median_nar: Median NAR across samples
        - median_snr: Median SNR across samples
        - pass_rate: Fraction of samples passing all thresholds (0-1)
        - composite_score: Weighted score (higher is better)
        Sorted by composite_score descending (best methods first).

    Notes
    -----
    Composite Score Calculation:
        - Normalizes each metric to [0, 1] range
        - RFZN: Lower is better (inverted for scoring)
        - NAR: Lower is better (inverted for scoring)
        - SNR: Higher is better
        - Pass rate: Bonus for consistent performance
        - Composite = (0.3 * RFZN_score) + (0.3 * NAR_score) + (0.3 * SNR_score) + (0.1 * pass_rate)

    Example
    -------
    >>> rfzn, nar, snr = evaluate_baseline_correction_methods(df, flat_windows)
    >>> recommendations = recommend_baseline_methods(rfzn, nar, snr, top_n=3)
    >>> print(recommendations)
         method  median_rfzn  median_nar  median_snr  pass_rate  composite_score
    0   airpls       0.0045        0.02        25.3       0.95             0.89
    1    arpls       0.0052        0.03        23.1       0.92             0.85
    2    drpls       0.0061        0.04        21.7       0.88             0.81
    """
    # Suppress warnings in this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Compute median metrics across samples
        median_rfzn = rfzn_tbl.median(axis=0)
        median_nar = nar_tbl.median(axis=0)
        median_snr = snr_tbl.median(axis=0)

        # Compute pass rate: fraction of samples meeting all thresholds
        rfzn_pass = (rfzn_tbl <= rfzn_threshold).sum(axis=0) / len(rfzn_tbl)
        nar_pass = (nar_tbl <= nar_threshold).sum(axis=0) / len(nar_tbl)
        snr_pass = (snr_tbl >= snr_min).sum(axis=0) / len(snr_tbl)
        pass_rate = (rfzn_pass + nar_pass + snr_pass) / 3.0  # Average pass rate

        # Normalize metrics to [0, 1] for composite scoring
        # RFZN and NAR: lower is better (invert for scoring)
        # SNR: higher is better
        rfzn_min, rfzn_max = median_rfzn.min(), median_rfzn.max()
        nar_min, nar_max = median_nar.min(), median_nar.max()
        snr_min_val, snr_max = median_snr.min(), median_snr.max()

        # Avoid division by zero
        rfzn_range = rfzn_max - rfzn_min if rfzn_max > rfzn_min else 1.0
        nar_range = nar_max - nar_min if nar_max > nar_min else 1.0
        snr_range = snr_max - snr_min_val if snr_max > snr_min_val else 1.0

        # Normalize (invert RFZN and NAR so higher scores are better)
        rfzn_score = 1.0 - (median_rfzn - rfzn_min) / rfzn_range
        nar_score = 1.0 - (median_nar - nar_min) / nar_range
        snr_score = (median_snr - snr_min_val) / snr_range

        # Composite score (weighted average)
        composite_score = (0.3 * rfzn_score) + (0.3 * nar_score) + (0.3 * snr_score) + (0.1 * pass_rate)

        # Build results DataFrame
        results = pd.DataFrame({
            'method': median_rfzn.index,
            'median_rfzn': median_rfzn.values,
            'median_nar': median_nar.values,
            'median_snr': median_snr.values,
            'pass_rate': pass_rate.values,
            'composite_score': composite_score.values
        })

        # Sort by composite score (best first) and return top_n
        results = results.sort_values('composite_score', ascending=False).reset_index(drop=True)
        return results.head(top_n)


# Run evaluation
# flat_windows = [(2500, 2600), (3200, 3500)]
# rfzn, nar, snr = evaluate_all_samples(final_df, flat_windows)
# recommendations = recommend_baseline_methods(rfzn, nar, snr)
