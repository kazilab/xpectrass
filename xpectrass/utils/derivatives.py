"""
Spectral Derivatives Module for FTIR Preprocessing
===================================================

Compute smoothed spectral derivatives for resolution enhancement
and baseline removal.
"""

from __future__ import annotations
from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import polars as pl
from scipy import signal

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)


def spectral_derivative(
    intensities: np.ndarray,
    order: int = 1,
    window_length: int = 15,
    polyorder: int = 3,
    delta: float = 1.0
) -> np.ndarray:
    """
    Compute smoothed spectral derivative using Savitzky-Golay.

    Parameters
    ----------
    intensities : np.ndarray
        1-D intensity array.
    order : int, default 1
        Derivative order (1 = first derivative, 2 = second derivative).
    window_length : int, default 15
        Savitzky-Golay window length (must be odd).
    polyorder : int, default 3
        Polynomial order for Savitzky-Golay filter.
    delta : float, default 1.0
        Spacing between samples (affects derivative scaling).
        **Important for FT-IR**: Set this to the actual wavenumber spacing
        (e.g., median of np.diff(wavenumbers)) to get physically meaningful
        derivatives in units of dI/d(cm⁻¹). Default of 1.0 assumes unit spacing.

    Returns
    -------
    np.ndarray
        Derivative spectrum.

    Notes
    -----
    - 1st derivative: Resolves overlapping peaks, removes constant baseline
    - 2nd derivative: Sharpens peaks, removes linear baseline
    - Higher derivatives increase noise; adjust window_length accordingly

    Warnings
    --------
    - Input must be 1-D array (will raise ValueError if multi-dimensional)
    - Large derivative orders may trigger automatic window expansion (logged as warning)

    Examples
    --------
    >>> import numpy as np
    >>> wn = np.linspace(400, 4000, 1000)
    >>> intensities = np.exp(-0.5 * ((wn - 1500) / 100) ** 2)
    >>>
    >>> # Correct: specify delta for physical units
    >>> delta = np.median(np.diff(wn))
    >>> deriv = spectral_derivative(intensities, order=1, delta=delta)
    >>>
    >>> # Warning: default delta=1.0 gives index-based units
    >>> deriv = spectral_derivative(intensities, order=1)  # Not recommended for FT-IR
    """
    import warnings

    y = np.asarray(intensities, dtype=np.float64)

    # Validate input is 1-D
    if y.ndim != 1:
        raise ValueError(
            f"Input must be 1-D array, got {y.ndim}-D array with shape {y.shape}. "
            "If you have multiple spectra, use derivative_batch() instead."
        )

    # Store original window_length for warning
    original_window = window_length

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure polyorder constraints
    polyorder = min(polyorder, window_length - 1)

    # Derivative order cannot exceed polyorder
    if order > polyorder:
        polyorder = order
        if polyorder >= window_length:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

            # Warn about automatic expansion (could over-smooth)
            if window_length > original_window * 2:
                warnings.warn(
                    f"Derivative order {order} required expanding window_length from "
                    f"{original_window} to {window_length}. This may over-smooth your data. "
                    f"Consider using a larger initial window_length or lower derivative order.",
                    UserWarning,
                    stacklevel=2
                )

    return signal.savgol_filter(
        y,
        window_length,
        polyorder,
        deriv=order,
        delta=delta
    )


def first_derivative(
    intensities: np.ndarray,
    window_length: int = 15,
    polyorder: int = 3
) -> np.ndarray:
    """
    Compute first derivative.
    
    Benefits:
    - Removes constant baseline offset
    - Resolves overlapping bands
    - Enhances small spectral differences
    """
    return spectral_derivative(intensities, order=1, 
                               window_length=window_length, 
                               polyorder=polyorder)


def second_derivative(
    intensities: np.ndarray,
    window_length: int = 15,
    polyorder: int = 4
) -> np.ndarray:
    """
    Compute second derivative.
    
    Benefits:
    - Removes linear baseline
    - Sharpens peaks (negative peaks in output)
    - Heavily used in FTIR for band identification
    
    Note: Peaks appear as negative minima in 2nd derivative.
    """
    return spectral_derivative(intensities, order=2, 
                               window_length=window_length, 
                               polyorder=polyorder)


def gap_derivative(
    intensities: np.ndarray,
    gap: int = 5,
    segment: int = 5,
    delta: float = 1.0,
    pad_mode: str = 'edge'
) -> np.ndarray:
    """
    Norris-Williams gap derivative.

    Averages points on either side of a gap, then takes difference.
    More noise-resistant than point-to-point derivatives.

    Parameters
    ----------
    intensities : np.ndarray
        1-D spectrum.
    gap : int, default 5
        Gap size (number of points to skip). Will be cast to int if float provided.
    segment : int, default 5
        Number of points to average on each side. Will be cast to int if float provided.
    delta : float, default 1.0
        Spacing between consecutive points (e.g., wavenumber spacing in cm⁻¹).
        The derivative is divided by (gap + segment) * delta to get proper units.
        For FT-IR with uniform 1 cm⁻¹ spacing, delta=1.0 is correct.
    pad_mode : str, default 'edge'
        Padding mode for edges. Options:
        - 'edge': Replicate edge values (default, simple but may create plateaus)
        - 'constant': Pad with zeros (pure approach, no artifacts)
        - None: Return unpadded array (length reduced by gap + 2*segment - 1)

    Returns
    -------
    np.ndarray
        Gap derivative. If pad_mode is None, array is shorter than input
        by (gap + 2*segment - 1). Otherwise, same length as input.

    Notes
    -----
    - Padding with 'edge' mode replicates edge values, which can introduce
      artificial plateaus at spectrum ends. For pure results, use pad_mode=None
      or 'constant'.
    - For FT-IR spectra, edges (<600 cm⁻¹, >4000 cm⁻¹) often have noise,
      so 'edge' padding is usually acceptable.
    - The derivative is scaled by (gap + segment) * delta to approximate
      dI/d(wavenumber). For uniform grids, this gives physically meaningful units.

    Examples
    --------
    >>> import numpy as np
    >>> wn = np.linspace(400, 4000, 1000)
    >>> y = np.exp(-0.5 * ((wn - 1500) / 100) ** 2)
    >>>
    >>> # With padding (same length as input)
    >>> deriv = gap_derivative(y, gap=5, segment=5, pad_mode='edge')
    >>> len(deriv) == len(y)  # True
    >>>
    >>> # Without padding (pure derivative, shorter)
    >>> deriv = gap_derivative(y, gap=5, segment=5, pad_mode=None)
    >>> len(deriv) == len(y) - 14  # True (lost gap + 2*segment - 1 points)
    >>>
    >>> # With delta scaling for physical units
    >>> delta = np.median(np.diff(wn))
    >>> deriv = gap_derivative(y, gap=5, segment=5, delta=delta)
    """
    y = np.asarray(intensities, dtype=np.float64)
    n = len(y)

    # Cast parameters to int if needed
    gap = int(gap)
    segment = int(segment)

    # Validate input is 1-D
    if y.ndim != 1:
        raise ValueError(
            f"Input must be 1-D array, got {y.ndim}-D array with shape {y.shape}."
        )

    result_len = n - gap - 2 * segment + 1
    if result_len <= 0:
        raise ValueError(
            f"Gap ({gap}) and segment ({segment}) too large for spectrum length ({n}). "
            f"Require: n > gap + 2*segment - 1 (i.e., {n} > {gap + 2*segment - 1})."
        )

    result = np.zeros(result_len)

    # Compute gap derivative
    for i in range(result_len):
        left_avg = np.mean(y[i:i + segment])
        right_avg = np.mean(y[i + segment + gap:i + 2 * segment + gap])
        result[i] = right_avg - left_avg

    # Scale by delta to get proper units
    # The effective spacing is (gap + segment) points
    effective_spacing = (gap + segment) * delta
    result = result / effective_spacing

    # Handle padding
    if pad_mode is None:
        # Return unpadded (shorter) array
        return result
    else:
        # Pad to original length
        pad_left = (gap + 2 * segment - 1) // 2
        pad_right = n - result_len - pad_left

        if pad_mode == 'constant':
            # Pad with zeros (pure, no artifacts)
            return np.pad(result, (pad_left, pad_right), mode='constant', constant_values=0)
        elif pad_mode == 'edge':
            # Pad with edge values (may create plateaus)
            return np.pad(result, (pad_left, pad_right), mode='edge')
        else:
            raise ValueError(
                f"Invalid pad_mode '{pad_mode}'. Choose 'edge', 'constant', or None."
            )



def derivative_with_smoothing(
    intensities: np.ndarray,
    order: int = 1,
    smooth_window: int = 11,
    deriv_window: int = 15,
    smooth_polyorder: int = 3,
    deriv_polyorder: int = None,
    delta: float = 1.0,
    smooth_first: bool = True
) -> np.ndarray:
    """
    Apply derivative with separate smoothing control.

    This function allows independent control over smoothing and derivative
    computation, useful for very noisy data or when you want to optimize
    smoothing separately from derivative calculation.

    Parameters
    ----------
    intensities : np.ndarray
        1-D spectrum.
    order : int, default 1
        Derivative order.
    smooth_window : int, default 11
        Window length for initial smoothing step (if smooth_first=True).
        Must be odd and > smooth_polyorder.
    deriv_window : int, default 15
        Window length for derivative calculation.
        Must be odd and > deriv_polyorder.
    smooth_polyorder : int, default 3
        Polynomial order for smoothing step (Savitzky-Golay).
        Must be less than smooth_window.
    deriv_polyorder : int, optional
        Polynomial order for derivative step. If None, defaults to order + 1
        (minimum required for the derivative order).
    delta : float, default 1.0
        Spacing between samples (affects derivative scaling).
        **Important for FT-IR**: Set to actual wavenumber spacing
        for physically meaningful derivatives in dI/d(cm⁻¹).
    smooth_first : bool, default True
        If True, smooth before taking derivative (recommended).
        If False, differentiate first then smooth (may distort peak shapes
        and amplify noise before smoothing—use with caution).

    Returns
    -------
    np.ndarray
        Derivative spectrum.

    Warnings
    --------
    - Setting smooth_first=False differentiates noisy data before smoothing,
      which can amplify noise and distort spectral features. Generally not
      recommended unless you have specific scientific reasons.
    - High smooth_polyorder with small smooth_window may under-smooth data.

    Notes
    -----
    - For most applications, use spectral_derivative() which combines
      smoothing and differentiation optimally in a single step.
    - Use this function only when you need separate control over smoothing
      and derivative parameters (e.g., aggressive pre-smoothing for very
      noisy data).
    - The two-step approach (smooth → derivative) may introduce edge artifacts
      at both ends of the spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> wn = np.linspace(400, 4000, 1000)
    >>> y = np.exp(-0.5 * ((wn - 1500) / 100) ** 2)
    >>> y_noisy = y + np.random.normal(0, 0.01, len(y))
    >>>
    >>> # Recommended: smooth first (default)
    >>> delta = np.median(np.diff(wn))
    >>> deriv = derivative_with_smoothing(
    ...     y_noisy,
    ...     order=1,
    ...     smooth_window=25,  # Heavy smoothing
    ...     deriv_window=15,
    ...     delta=delta,
    ...     smooth_first=True
    ... )
    >>>
    >>> # Not recommended: differentiate noisy data first
    >>> deriv = derivative_with_smoothing(
    ...     y_noisy,
    ...     order=1,
    ...     smooth_first=False  # Amplifies noise before smoothing
    ... )
    """
    import warnings

    y = np.asarray(intensities, dtype=np.float64)

    # Validate input is 1-D
    if y.ndim != 1:
        raise ValueError(
            f"Input must be 1-D array, got {y.ndim}-D array with shape {y.shape}."
        )

    # Set default deriv_polyorder if not provided
    if deriv_polyorder is None:
        deriv_polyorder = order + 1  # Minimum required for derivative

    # Ensure windows are odd
    if smooth_window % 2 == 0:
        smooth_window += 1
    if deriv_window % 2 == 0:
        deriv_window += 1

    # Validate polyorder constraints
    if smooth_polyorder >= smooth_window:
        raise ValueError(
            f"smooth_polyorder ({smooth_polyorder}) must be less than "
            f"smooth_window ({smooth_window})"
        )
    if deriv_polyorder >= deriv_window:
        raise ValueError(
            f"deriv_polyorder ({deriv_polyorder}) must be less than "
            f"deriv_window ({deriv_window})"
        )
    if order > deriv_polyorder:
        raise ValueError(
            f"Derivative order ({order}) cannot exceed deriv_polyorder ({deriv_polyorder}). "
            f"Set deriv_polyorder >= {order}"
        )

    # Warn about smooth_first=False
    if not smooth_first:
        warnings.warn(
            "smooth_first=False differentiates noisy data before smoothing, which "
            "amplifies noise and may distort peak shapes. This is generally not "
            "recommended. Consider smooth_first=True (default) instead.",
            UserWarning,
            stacklevel=2
        )

    # Apply smoothing and derivative
    if smooth_first:
        # Recommended: smooth first, then differentiate
        y = signal.savgol_filter(y, smooth_window, smooth_polyorder, deriv=0)
        return signal.savgol_filter(y, deriv_window, deriv_polyorder, deriv=order, delta=delta)
    else:
        # Not recommended: differentiate first, then smooth
        y = signal.savgol_filter(y, deriv_window, deriv_polyorder, deriv=order, delta=delta)
        return signal.savgol_filter(y, smooth_window, smooth_polyorder, deriv=0)



# ---------------------------------------------------------------------------
#                           BATCH OPERATIONS
# ---------------------------------------------------------------------------

def derivative_batch(
    data: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    order: int = 1,
    window_length: int = 15,
    polyorder: int = 3,
    delta: float = 1.0,
    show_progress: bool = True
) -> Union[pd.DataFrame, pl.DataFrame, np.ndarray]:
    """
    Compute spectral derivatives for multiple spectra (DataFrame or numpy array).

    Works with both pandas and polars DataFrames, or numpy arrays.
    For DataFrames: each row is a sample, numerical columns are wavenumbers.
    For numpy arrays: shape (n_samples, n_wavenumbers).

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | np.ndarray
        Wide-format DataFrame where rows = samples, columns = wavenumbers,
        OR numpy array of shape (n_samples, n_wavenumbers).
    label_column : str, default "label"
        Name of the label/group column to exclude from derivative computation.
        Only used for DataFrame inputs.
    exclude_columns : list[str], optional
        Additional column names to exclude from derivative computation (e.g., 'sample', 'id').
        If None, automatically excludes non-numeric columns.
        Only used for DataFrame inputs.
    wn_min : float, optional
        Minimum wavenumber bound (cm⁻¹). If None, uses 200.0 cm⁻¹ as default,
        or auto-expands if no columns found within default range.
    wn_max : float, optional
        Maximum wavenumber bound (cm⁻¹). If None, uses 8000.0 cm⁻¹ as default,
        or auto-expands if no columns found within default range.
    order : int, default 1
        Derivative order:
        - 1: First derivative (resolves overlapping peaks, removes constant baseline)
        - 2: Second derivative (sharpens peaks, removes linear baseline)
        - 3+: Higher derivatives (increases noise sensitivity)
    window_length : int, default 15
        Savitzky-Golay filter window length (must be odd).
        Larger values = more smoothing but less detail.
    polyorder : int, default 3
        Polynomial order for Savitzky-Golay filter.
        Must be less than window_length.
    delta : float, default 1.0
        Spacing between samples (affects derivative scaling).
        For DataFrame inputs, this parameter is automatically computed from
        wavenumber spacing and the provided value is ignored.
        For numpy array inputs, uses the provided delta value.
    show_progress : bool, default True
        If True, display a progress bar during processing.
        Only used for DataFrame inputs.

    Returns
    -------
    pd.DataFrame | pl.DataFrame | np.ndarray
        Derivative spectra (same type as input).

    Examples
    --------
    >>> # First derivative of DataFrame
    >>> df_d1 = derivative_batch(df_wide, order=1)

    >>> # Second derivative with larger smoothing window
    >>> df_d2 = derivative_batch(df_wide, order=2, window_length=21)

    >>> # Third derivative (highly sensitive to noise)
    >>> df_d3 = derivative_batch(df_wide, order=3, window_length=25, polyorder=4)

    >>> # Numpy array processing (legacy)
    >>> spectra_d1 = derivative_batch(spectra_array, order=1)

    >>> # Disable progress bar
    >>> df_d1 = derivative_batch(df_wide, order=1, show_progress=False)

    Notes
    -----
    - 1st derivative: Removes constant baseline, enhances spectral differences
    - 2nd derivative: Removes linear baseline, sharpens peaks (peaks appear as negative minima)
    - Higher derivatives amplify noise; increase window_length for smoother results
    - Savitzky-Golay filtering preserves peak shapes better than simple numerical derivatives
    """
    # Handle numpy array input (legacy behavior)
    if isinstance(data, np.ndarray):
        return np.array([
            spectral_derivative(s, order, window_length, polyorder, delta)
            for s in data
        ])

    # Handle DataFrame input (new behavior)
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

    # OPTIMIZATION: Extract numpy array for vectorized processing
    spectral_data = df[sorted_cols].values.astype(np.float64)
    n_samples = spectral_data.shape[0]

    # Compute proper delta from wavenumber spacing for correct derivative scaling
    # This is critical for mathematically correct derivatives
    if len(sorted_wavenumbers) > 1:
        # Use median spacing for robustness against non-uniform grids
        computed_delta = np.median(np.abs(np.diff(sorted_wavenumbers)))
    else:
        computed_delta = 1.0

    # Use computed delta (overrides user-provided delta for DataFrames)
    actual_delta = computed_delta

    # Validate and adjust parameters once (instead of per-spectrum)
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure polyorder constraints
    polyorder = min(polyorder, window_length - 1)

    # Derivative order cannot exceed polyorder
    if order > polyorder:
        polyorder = order
        if polyorder >= window_length:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

    # VECTORIZED: Apply savgol_filter to entire 2D array at once (10-100x faster)
    # axis=1 means apply filter along wavenumber dimension (columns)
    if show_progress:
        desc = f"Computing {order}{'st' if order==1 else 'nd' if order==2 else 'rd' if order==3 else 'th'} derivative"
        print(f"{desc} for {n_samples} samples...")

    derivative_data = signal.savgol_filter(
        spectral_data,
        window_length,
        polyorder,
        deriv=order,
        delta=actual_delta,
        axis=1  # Apply along wavenumber axis
    )

    # Create result DataFrame with metadata + derivative spectra
    metadata_df = df[metadata_cols].copy()

    # Create spectral DataFrame from numpy array (avoids fragmentation)
    spectral_df = pd.DataFrame(derivative_data, columns=sorted_cols, index=df.index)

    # Concatenate metadata and spectral data
    result_df = pd.concat([metadata_df, spectral_df], axis=1)

    # Reorder columns to match original structure (metadata first, then spectra)
    final_cols = metadata_cols + sorted_cols
    df = result_df[final_cols]

    # Convert back to polars if input was polars
    if is_polars:
        df = pl.from_pandas(df)

    return df


# ---------------------------------------------------------------------------
#                           VISUALIZATION
# ---------------------------------------------------------------------------

def plot_derivatives(
    data: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    orders: List[int] = [0, 1, 2],
    sample: Union[str, int] = None,
    wavenumbers: Optional[np.ndarray] = None,
    window_length: int = 15,
    polyorder: int = 3,
    figsize: Tuple[int, int] = (10, 8),
    invert_x: bool = True
) -> None:
    """
    Plot spectrum and its derivatives for DataFrame or numpy array.

    Works with both pandas/polars DataFrames and numpy arrays.
    For DataFrames: automatically extracts wavenumbers from column names.
    For numpy arrays: wavenumbers parameter is required.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | np.ndarray
        Wide-format DataFrame (rows=samples, columns=wavenumbers) OR
        1-D numpy array of intensities.
    label_column : str, default "label"
        Name of the label/group column to exclude.
        Only used for DataFrame inputs.
    exclude_columns : list[str], optional
        Additional column names to exclude (e.g., 'sample', 'id').
        Only used for DataFrame inputs.
    wn_min : float, optional
        Minimum wavenumber bound (cm⁻¹). If None, uses 200.0 cm⁻¹ as default.
    wn_max : float, optional
        Maximum wavenumber bound (cm⁻¹). If None, uses 8000.0 cm⁻¹ as default.
    orders : list of int, default [0, 1, 2]
        Derivative orders to plot:
        - 0: Original spectrum
        - 1: First derivative
        - 2: Second derivative
        - 3+: Higher derivatives
    sample : str | int, optional
        For DataFrames: sample name (index) to plot.
        If None, plots the first sample.
        For numpy arrays: ignored (plots the provided array).
    wavenumbers : np.ndarray, optional
        Wavenumber axis. Required only for numpy array input.
        For DataFrames, automatically extracted from column names.
    window_length : int, default 15
        Savitzky-Golay window length for derivative computation.
    polyorder : int, default 3
        Polynomial order for Savitzky-Golay filter.
    figsize : tuple, default (10, 8)
        Figure size (width, height).
    invert_x : bool, default True
        If True, invert x-axis (higher wavenumbers on left).

    Examples
    --------
    >>> # Plot derivatives from DataFrame
    >>> plot_derivatives(df_wide, orders=[0, 1, 2], sample="PP225")

    >>> # Plot original and 2nd derivative only
    >>> plot_derivatives(df_wide, orders=[0, 2], sample="HDPE1")

    >>> # Plot from first sample (default)
    >>> plot_derivatives(df_wide, orders=[0, 1, 2, 3])

    >>> # Plot from numpy array
    >>> plot_derivatives(spectrum, wavenumbers=wn_array, orders=[0, 1, 2])

    >>> # Custom window for smoother derivatives
    >>> plot_derivatives(df_wide, sample="PET1", window_length=25, polyorder=4)

    Notes
    -----
    - 1st derivative: Removes constant baseline, enhances differences
    - 2nd derivative: Removes linear baseline, sharpens peaks (negative minima)
    - Higher orders increase noise sensitivity; use larger window_length
    """
    import matplotlib.pyplot as plt

    # Handle numpy array input (legacy behavior)
    if isinstance(data, np.ndarray):
        if wavenumbers is None:
            raise ValueError("wavenumbers parameter is required for numpy array input")
        intensities = data
        x = wavenumbers
        sample_name = "Spectrum"

    # Handle DataFrame input
    else:
        is_polars = isinstance(data, pl.DataFrame)

        # Convert to pandas
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
        numeric_cols, wn_values = _infer_spectral_columns(
            df, exclude_columns, wn_min, wn_max
        )
        sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(
            numeric_cols, wn_values
        )

        if len(sorted_cols) == 0:
            raise ValueError("No numeric columns found for plotting!")

        # Select sample to plot
        if sample is None:
            sample = df.index[0]
            sample_name = str(sample)
        else:
            sample_name = str(sample)

        # Extract spectrum and wavenumbers (sorted)
        intensities = df.loc[sample, sorted_cols].astype(float).values
        x = sorted_wavenumbers

    # Create subplots
    n_plots = len(orders)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Derivative labels
    labels = {
        0: 'Original',
        1: '1st Derivative',
        2: '2nd Derivative',
        3: '3rd Derivative',
        4: '4th Derivative'
    }

    # Compute proper delta for derivatives
    # delta = spacing between consecutive x values (for correct derivative scaling)
    if len(x) > 1:
        # Use median spacing for robustness against non-uniform grids
        delta = np.median(np.abs(np.diff(x)))
    else:
        delta = 1.0

    # Plot each derivative order
    for ax, order in zip(axes, orders):
        if order == 0:
            y = intensities
        else:
            y = spectral_derivative(
                intensities,
                order=order,
                window_length=window_length,
                polyorder=polyorder,
                delta=delta
            )

        ax.plot(x, y, 'b-', lw=0.8)
        ax.set_ylabel(labels.get(order, f'{order}th Derivative'))
        ax.grid(alpha=0.3)

    # Formatting
    axes[0].set_title(f'Spectral Derivatives: {sample_name}')
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')

    if invert_x:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.show()
