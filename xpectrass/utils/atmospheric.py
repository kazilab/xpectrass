"""
Atmospheric Correction Module for FTIR Spectral Preprocessing
==============================================================

Corrects for CO₂ and H₂O vapor interference in FTIR spectra
that result from atmospheric absorption during measurement.

**IMPORTANT**: This module expects absorbance data (AU), not transmittance (%).
Convert transmittance to absorbance first using convert_spectra() from trans_abs.py

Default atmospheric regions:
- CO₂: 2300-2400 cm⁻¹ (asymmetric stretch) and 650-690 cm⁻¹ (bending)
- H₂O: 1350-1900 cm⁻¹ (bending) and 3550-3900 cm⁻¹ (stretching)

Logging:
This module uses Python's logging module for warnings and informational messages.
Configure the logger to control output:

    import logging
    logging.getLogger('utils.atmospheric').setLevel(logging.INFO)  # Show all messages
    logging.getLogger('utils.atmospheric').setLevel(logging.ERROR)  # Only errors

Auto-Detection:
Use auto_detect=True in atmospheric_correction() to automatically check for
atmospheric interference and receive warnings if detected.
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Union
import logging

import numpy as np
import pandas as pd

import tqdm
from scipy import interpolate

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

# Optional dependency: the algorithms work with pandas; polars support is best-effort.
try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore


# Default atmospheric interference regions (wavenumbers in cm⁻¹)
CO2_REGIONS = [
    (2300.0, 2400.0),  # CO₂ asymmetric stretch 2350
    (650.0, 690.0),    # CO₂ bending mode 667
]
H2O_REGIONS = [
    (1350.0, 1900.0),  # H₂O bending mode 1640
    (3550.0, 3900.0),  # H₂O stretching modes 3650
]


def _atmospheric_correction(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    method: str = "interpolate",
    co2_ranges: Optional[List[Tuple[float, float]]] = None,
    h2o_ranges: Optional[List[Tuple[float, float]]] = None,
    reference_spectrum: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Correct for atmospheric CO₂ and H₂O interference on a single spectrum array.
    """
    y = np.asarray(intensities, dtype=np.float64).copy()
    x = np.asarray(wavenumbers, dtype=np.float64)

    if len(y) != len(x):
        raise ValueError("intensities and wavenumbers must have same length")

    if np.unique(x).size != x.size:
        raise ValueError("wavenumbers contains duplicates; cannot correct reliably")

    # The correction logic assumes regions are contiguous in *index* space.
    # If x is non-monotonic, sort by wavenumber, apply correction, then un-sort.
    order: Optional[np.ndarray] = None
    if not _is_monotonic_strict(x):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    # Use defaults if not provided
    if co2_ranges is None:
        co2_ranges = CO2_REGIONS
    if h2o_ranges is None:
        h2o_ranges = H2O_REGIONS

    # Combine all atmospheric regions
    all_regions = list(co2_ranges) + list(h2o_ranges)

    if method == "interpolate":
        corrected = _correct_interpolate(y, x, all_regions, kind='linear')
    elif method == "spline":
        corrected = _correct_interpolate(y, x, all_regions, kind='cubic')
    elif method == "reference":
        if reference_spectrum is None:
            raise ValueError("reference_spectrum required for 'reference' method")
        ref = np.asarray(reference_spectrum, dtype=np.float64)
        if order is not None:
            if ref.shape[0] != order.shape[0]:
                raise ValueError("reference_spectrum must match intensities length")
            ref = ref[order]
        corrected = _correct_reference(y, x, all_regions, ref, **kwargs)
    elif method == "zero":
        corrected = _correct_zero(y, x, all_regions)
    elif method == "exclude":
        corrected = _correct_exclude(y, x, all_regions)
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            "Valid options: interpolate, spline, reference, zero, exclude"
        )

    if order is not None:
        # Invert the argsort permutation
        inv = np.argsort(order)
        corrected = corrected[inv]

    return corrected


def atmospheric_correction_spectrum(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    method: str = "interpolate",
    co2_ranges: Optional[List[Tuple[float, float]]] = None,
    h2o_ranges: Optional[List[Tuple[float, float]]] = None,
    reference_spectrum: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Public wrapper for correcting a *single* spectrum (numpy arrays)."""
    return _atmospheric_correction(
        intensities=intensities,
        wavenumbers=wavenumbers,
        method=method,
        co2_ranges=co2_ranges,
        h2o_ranges=h2o_ranges,
        reference_spectrum=reference_spectrum,
        **kwargs,
    )


def _make_atmospheric_mask(
    wavenumbers: np.ndarray,
    regions: List[Tuple[float, float]]
) -> np.ndarray:
    """Create boolean mask for atmospheric interference regions."""
    mask = np.zeros(len(wavenumbers), dtype=bool)
    for lo, hi in regions:
        mask |= (wavenumbers >= lo) & (wavenumbers <= hi)
    return mask


def _correct_interpolate(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]],
    kind: str = 'linear',
    n_side: int = 8,
) -> np.ndarray:
    """Interpolate across atmospheric regions using boundary values.

    If boundary regions contain only NaN values, the atmospheric region
    is left as NaN rather than attempting poor extrapolation.
    """
    result = y.copy()

    for lo, hi in regions:
        # Identify affected points
        mask = (x >= lo) & (x <= hi)
        affected_idx = np.flatnonzero(mask)
        if affected_idx.size == 0:
            continue

        left_idx = affected_idx[0]
        right_idx = affected_idx[-1]

        # Control points from both sides of the region (in index space).
        left_ctrl = np.arange(max(0, left_idx - n_side), left_idx)
        right_ctrl = np.arange(right_idx + 1, min(len(y), right_idx + 1 + n_side))

        # Can't safely interpolate if the region touches an edge or lacks data.
        if left_ctrl.size == 0 or right_ctrl.size == 0:
            continue

        ctrl_idx = np.concatenate([left_ctrl, right_ctrl])
        ctrl_x = x[ctrl_idx]
        ctrl_y = y[ctrl_idx]

        # Filter to finite values only
        finite = np.isfinite(ctrl_x) & np.isfinite(ctrl_y)
        ctrl_x = ctrl_x[finite]
        ctrl_y = ctrl_y[finite]

        # If insufficient finite data, leave region as NaN instead of poor extrapolation
        if ctrl_x.size < 2:
            result[mask] = np.nan
            continue

        order = np.argsort(ctrl_x)
        ctrl_x = ctrl_x[order]
        ctrl_y = ctrl_y[order]

        use_cubic = (kind.lower() in {"cubic", "spline"}) and (ctrl_x.size >= 4)
        interp_kind = "cubic" if use_cubic else "linear"

        f = interpolate.interp1d(
            ctrl_x,
            ctrl_y,
            kind=interp_kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        result[mask] = f(x[mask])

    return result


def _correct_reference(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]],
    reference: np.ndarray,
    reference_scale: Optional[float] = None
) -> np.ndarray:
    """Subtract scaled reference atmospheric spectrum."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != y.shape:
        raise ValueError("reference_spectrum must have the same shape as intensities")

    mask = _make_atmospheric_mask(x, regions)

    if reference_scale is None:
        # Auto-fit scale factor using least squares in the masked regions
        ref_region = reference[mask]
        y_region = y[mask]

        # Filter out NaN values before computing scale factor
        finite = np.isfinite(y_region) & np.isfinite(ref_region)
        if np.sum(finite) < 2:
            raise ValueError(
                "Insufficient finite data in atmospheric regions for reference scaling. "
                "Need at least 2 valid points."
            )

        ref_region = ref_region[finite]
        y_region = y_region[finite]

        denom = float(np.dot(ref_region, ref_region))
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError(
                "reference_spectrum has zero/invalid energy in the atmospheric regions"
            )
        reference_scale = float(np.dot(y_region, ref_region) / denom)

    return y - float(reference_scale) * reference


def _correct_zero(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]]
) -> np.ndarray:
    """Set atmospheric regions to local baseline (average of boundaries)."""
    result = y.copy()
    
    for lo, hi in regions:
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        
        affected_idx = np.where(mask)[0]
        left_idx = max(0, affected_idx[0] - 10)
        right_idx = min(len(y), affected_idx[-1] + 11)
        
        # Use average of boundary regions (guard against empty slices)
        left_slice = y[left_idx:affected_idx[0]]
        right_slice = y[affected_idx[-1] + 1:right_idx]
        if left_slice.size == 0 or right_slice.size == 0:
            continue

        left_mean = float(np.nanmean(left_slice))
        right_mean = float(np.nanmean(right_slice))
        # Skip if either side has no valid (non-NaN) data
        if np.isnan(left_mean) or np.isnan(right_mean):
            continue
        baseline = (left_mean + right_mean) / 2
        
        result[mask] = baseline
    
    return result


def _correct_exclude(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]]
) -> np.ndarray:
    """Mark atmospheric regions as NaN (for exclusion from analysis)."""
    result = y.copy()
    mask = _make_atmospheric_mask(x, regions)
    result[mask] = np.nan
    return result


# ---------------------------------------------------------------------------
#                           UTILITIES
# ---------------------------------------------------------------------------

def _get_atmospheric_regions() -> dict:
    """Return standard atmospheric interference regions."""
    return {
        'co2': CO2_REGIONS,
        'h2o': H2O_REGIONS,
        'all': CO2_REGIONS + H2O_REGIONS
    }


def identify_atmospheric_features(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    threshold: float = 0.1
) -> dict:
    """Check for presence of atmospheric interference.

    Args:
        intensities: Spectral intensity values
        wavenumbers: Wavenumber grid (cm⁻¹)
        threshold: Sensitivity threshold as a fraction of total spectral variation.
                   Higher values (e.g., 0.2) reduce false positives but may miss
                   weak interference. Lower values (e.g., 0.05) are more sensitive
                   but may flag noise. Default of 0.1 works well for typical FTIR spectra.

    Returns:
        Dictionary with 'co2_detected', 'h2o_detected' (bool), and 'recommendations' (list)
    """
    regions = _get_atmospheric_regions()
    report = {'co2_detected': False, 'h2o_detected': False, 'recommendations': []}

    # Compute overall spectral variation (use nanstd to handle NaN values)
    overall_std = np.nanstd(intensities)
    if not np.isfinite(overall_std) or overall_std == 0:
        # Cannot detect features in invalid or flat spectrum
        return report

    # Check CO2 (loop over default CO2 regions)
    for lo, hi in regions['co2']:
        co2_mask = (wavenumbers >= lo) & (wavenumbers <= hi)
        if not np.any(co2_mask):
            continue
        co2_region = intensities[co2_mask]

        # Use nanstd for consistent NaN handling
        region_std = np.nanstd(co2_region)
        if np.isfinite(region_std) and region_std > threshold * overall_std:
            report['co2_detected'] = True
            report['recommendations'].append(
                f"Apply CO₂ correction ({lo:.0f}-{hi:.0f} cm⁻¹)"
            )
            break

    # Check H2O (use same metric as CO2 for consistency)
    for lo, hi in regions['h2o']:
        h2o_mask = (wavenumbers >= lo) & (wavenumbers <= hi)
        if not np.any(h2o_mask):
            continue
        h2o_region = intensities[h2o_mask]

        # Use nanstd for consistent NaN handling
        region_std = np.nanstd(h2o_region)
        if np.isfinite(region_std) and region_std > threshold * overall_std:
            report['h2o_detected'] = True
            report['recommendations'].append(
                f"Apply H₂O correction ({lo:.0f}-{hi:.0f} cm⁻¹)"
            )

    return report


def _exclude_and_interpolate_regions(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    interpolate_ranges: Optional[List[Tuple[float, float]]] = None,
    method: str = "interpolate",
    reference_spectrum: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Exclude specific wavenumber ranges and interpolate others.

    Args:
        intensities: Spectral intensity values
        wavenumbers: Wavenumber grid (cm⁻¹)
        exclude_ranges: Wavenumber ranges to physically remove from output
        interpolate_ranges: Wavenumber ranges to process using specified method
        method: Method for interpolate_ranges ('interpolate', 'spline', 'reference', 'zero', 'exclude')
        reference_spectrum: Reference spectrum for 'reference' method
        **kwargs: Additional arguments for specific methods (e.g., reference_scale)

    Returns:
        Tuple of (corrected_intensities, corrected_wavenumbers)
    """
    y = np.asarray(intensities, dtype=np.float64).copy()
    x = np.asarray(wavenumbers, dtype=np.float64).copy()

    if np.unique(x).size != x.size:
        raise ValueError("wavenumbers contains duplicates; cannot process reliably")

    # Validate that exclude and interpolate ranges don't overlap
    if exclude_ranges and interpolate_ranges:
        for ex_lo, ex_hi in exclude_ranges:
            for int_lo, int_hi in interpolate_ranges:
                # Check if ranges overlap
                if not (ex_hi < int_lo or ex_lo > int_hi):
                    raise ValueError(
                        f"Exclude range ({ex_lo:.1f}-{ex_hi:.1f}) overlaps with "
                        f"interpolate range ({int_lo:.1f}-{int_hi:.1f}). "
                        "Ranges must not overlap."
                    )

    # Sort non-monotonic grids to keep the region logic meaningful.
    order: Optional[np.ndarray] = None
    if not _is_monotonic_strict(x):
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        # Also sort reference_spectrum if provided
        if reference_spectrum is not None:
            ref = np.asarray(reference_spectrum, dtype=np.float64)
            if ref.shape[0] != order.shape[0]:
                raise ValueError("reference_spectrum must match intensities length")
            reference_spectrum = ref[order]

    # Step 1: Process interpolate_ranges using the specified method
    if interpolate_ranges:
        if method == "interpolate":
            y = _correct_interpolate(y, x, interpolate_ranges, kind='linear')
        elif method == "spline":
            y = _correct_interpolate(y, x, interpolate_ranges, kind='cubic')
        elif method == "reference":
            if reference_spectrum is None:
                raise ValueError("reference_spectrum required for 'reference' method")
            y = _correct_reference(y, x, interpolate_ranges, reference_spectrum, **kwargs)
        elif method == "zero":
            y = _correct_zero(y, x, interpolate_ranges)
        elif method == "exclude":
            y = _correct_exclude(y, x, interpolate_ranges)
        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                "Valid options: interpolate, spline, reference, zero, exclude"
            )

    # Step 2: Exclude specified ranges (physically remove from data)
    if exclude_ranges:
        # Create mask for regions to KEEP (inverse of exclude)
        keep_mask = np.ones(len(x), dtype=bool)
        for min_wn, max_wn in exclude_ranges:
            exclude_mask = (x >= min_wn) & (x <= max_wn)
            keep_mask &= ~exclude_mask

        # Apply mask
        y = y[keep_mask]
        x = x[keep_mask]

    return y, x


def exclude_and_interpolate_spectrum(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    interpolate_ranges: Optional[List[Tuple[float, float]]] = None,
    method: str = "interpolate",
    reference_spectrum: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Public wrapper for excluding/interpolating regions on a *single* spectrum.

    Args:
        intensities: Spectral intensity values
        wavenumbers: Wavenumber grid (cm⁻¹)
        exclude_ranges: Wavenumber ranges to physically remove from output
        interpolate_ranges: Wavenumber ranges to process using specified method
        method: Method for interpolate_ranges ('interpolate', 'spline', 'reference', 'zero', 'exclude')
        reference_spectrum: Reference spectrum for 'reference' method
        **kwargs: Additional arguments for specific methods (e.g., reference_scale)

    Returns:
        Tuple of (corrected_intensities, corrected_wavenumbers)
    """
    return _exclude_and_interpolate_regions(
        intensities=intensities,
        wavenumbers=wavenumbers,
        exclude_ranges=exclude_ranges,
        interpolate_ranges=interpolate_ranges,
        method=method,
        reference_spectrum=reference_spectrum,
        **kwargs
    )


# ---------------------------------------------------------------------------
#                      DATAFRAME-COMPATIBLE FUNCTIONS
# ---------------------------------------------------------------------------

def atmospheric_correction(
    data: Union[pd.DataFrame, "pl.DataFrame"],  # String type hint avoids import error
    method: str = "interpolate",
    co2_ranges: Optional[List[Tuple[float, float]]] = None,
    h2o_ranges: Optional[List[Tuple[float, float]]] = None,
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    auto_detect: bool = False,
    **kwargs
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """
    Apply atmospheric correction to a DataFrame of FTIR spectra.

    Works with both pandas and polars DataFrames. Each row is a sample,
    numerical columns are wavenumbers. Applies correction to all samples.

    NaN Handling:
        All correction methods robustly handle NaN values in spectral data:
        - 'interpolate'/'spline': Filters NaN from boundary regions; sets atmospheric
          regions to NaN if insufficient finite boundary data exists
        - 'reference': Filters NaN before computing scale factor
        - 'zero': Uses nanmean for baselines; skips regions with all-NaN boundaries
        - 'exclude': Marks regions as NaN

    Performance:
        Optimized for large datasets using:
        - Vectorized numpy array access (no DataFrame.loc overhead)
        - Pre-allocated output arrays (no list appending)
        - Progress tracking via tqdm
        For maximum performance on very large datasets (100k+ spectra), consider
        using atmospheric_correction_spectrum() on extracted numpy arrays directly.

    Args:
        data: Input DataFrame (pandas or polars)
        method: Correction method ('interpolate', 'spline', 'reference', 'zero', 'exclude')
        co2_ranges: Custom CO₂ regions to correct (default: standard FTIR regions)
        h2o_ranges: Custom H₂O regions to correct (default: standard FTIR regions)
        label_column: Name of label/metadata column to preserve
        exclude_columns: Additional columns to exclude from processing
        wn_min: Minimum wavenumber for column detection (default: auto-detect)
        wn_max: Maximum wavenumber for column detection (default: auto-detect)
        auto_detect: If True, automatically check first spectrum for atmospheric interference
                     and warn if detected but no custom ranges provided (default: False)
        **kwargs: Additional arguments passed to correction methods

    Returns:
        Corrected DataFrame in same format as input (columns sorted by ascending wavenumber)

    Warnings:
        - Warns if spectral columns are reordered during processing
        - Warns if wavenumber bounds are auto-expanded
        - Warns if auto_detect=True and atmospheric interference detected without custom ranges
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

    # Identify spectral columns by parsing column names as wavenumbers.
    numeric_cols, wavenumbers = _infer_spectral_columns(df, exclude_columns, wn_min, wn_max)
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(numeric_cols, wavenumbers)

    # Warn if columns will be reordered
    if not np.array_equal(sort_idx, np.arange(len(sort_idx))):
        logger.warning(
            "Spectral columns are not in ascending wavenumber order. "
            "Output DataFrame will have columns sorted by ascending wavenumber for standardization."
        )

    # Auto-detection: Check first spectrum for atmospheric interference
    if auto_detect:
        # Extract first spectrum for analysis
        first_spectrum = df[sorted_cols].iloc[0].values.astype(np.float64)
        detection_result = identify_atmospheric_features(
            intensities=first_spectrum,
            wavenumbers=sorted_wavenumbers,
            threshold=0.1
        )

        # Warn if interference detected but user hasn't provided custom ranges
        if (detection_result['co2_detected'] or detection_result['h2o_detected']):
            if co2_ranges is None and h2o_ranges is None:
                interference_types = []
                if detection_result['co2_detected']:
                    interference_types.append('CO₂')
                if detection_result['h2o_detected']:
                    interference_types.append('H₂O')

                logger.warning(
                    f"Atmospheric interference detected ({', '.join(interference_types)}) "
                    f"in first spectrum. Using default correction regions. "
                    f"Recommendations: {'; '.join(detection_result['recommendations'])}. "
                    "Provide custom co2_ranges/h2o_ranges if defaults are incorrect."
                )

    # If a reference spectrum is provided, ensure it aligns with the sorted grid
    local_kwargs = dict(kwargs)
    if "reference_spectrum" in local_kwargs and local_kwargs["reference_spectrum"] is not None:
        ref = np.asarray(local_kwargs["reference_spectrum"], dtype=np.float64)
        if ref.shape[0] != len(numeric_cols):
            raise ValueError(
                "reference_spectrum length must match number of spectral columns"
            )
        local_kwargs["reference_spectrum"] = ref[sort_idx]

    # OPTIMIZATION: Extract numpy array and pre-allocate result
    spectral_data = df[sorted_cols].values.astype(np.float64)
    n_samples = spectral_data.shape[0]
    n_wavenumbers = spectral_data.shape[1]
    corrected_data = np.empty((n_samples, n_wavenumbers), dtype=np.float64)

    # Iterate with progress bar
    for i in tqdm.tqdm(range(n_samples), desc="Correcting Atmosphere"):
        intensities = spectral_data[i, :]

        # Apply atmospheric correction
        corrected_data[i, :] = _atmospheric_correction(
            intensities=intensities,
            wavenumbers=sorted_wavenumbers,
            method=method,
            co2_ranges=co2_ranges,
            h2o_ranges=h2o_ranges,
            **local_kwargs
        )

    # Reconstruct Spectral Data
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

    # Reorder columns to ensure metadata comes first (optional but nice)
    final_cols = metadata_cols + sorted_cols
    df_final = df_final[final_cols]

    # Convert back to polars if input was polars
    if is_polars:
        df_final = pl.from_pandas(df_final)

    return df_final


def exclude_and_interpolate_regions(
    data: Union[pd.DataFrame, "pl.DataFrame"],  # String type hint
    exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    interpolate_ranges: Optional[List[Tuple[float, float]]] = None,
    method: str = "interpolate",
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    reference_spectrum: Optional[np.ndarray] = None,
    **kwargs
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """
    Exclude wavenumber ranges and process interpolate regions for DataFrame of spectra.

    Args:
        data: Input DataFrame (pandas or polars)
        exclude_ranges: Wavenumber ranges to physically remove from output
        interpolate_ranges: Wavenumber ranges to process using specified method
        method: Method for interpolate_ranges ('interpolate', 'spline', 'reference', 'zero', 'exclude')
        label_column: Name of label/metadata column to preserve
        exclude_columns: Additional columns to exclude from processing
        wn_min: Minimum wavenumber for column detection (default: auto-detect)
        wn_max: Maximum wavenumber for column detection (default: auto-detect)
        reference_spectrum: Reference spectrum for 'reference' method
        **kwargs: Additional arguments for specific methods (e.g., reference_scale)

    Returns:
        Processed DataFrame in same format as input
    """
    # Determine if input is polars or pandas (polars is optional)
    is_polars = (pl is not None) and isinstance(data, pl.DataFrame)

    # Convert to pandas for processing
    if is_polars:
        df = data.to_pandas()
    else:
        df = data.copy()

    # Identify columns to exclude from processing
    if exclude_columns is None:
        exclude_columns = []
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    else:
        exclude_columns = list(exclude_columns)

    # Always exclude the label column if it exists
    if label_column in df.columns and label_column not in exclude_columns:
        exclude_columns.append(label_column)

    # Identify spectral columns by parsing column names as wavenumbers.
    numeric_cols, wavenumbers = _infer_spectral_columns(df, exclude_columns, wn_min, wn_max)
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(numeric_cols, wavenumbers)

    # Warn if columns will be reordered
    if not np.array_equal(sort_idx, np.arange(len(sort_idx))):
        logger.warning(
            "Spectral columns are not in ascending wavenumber order. "
            "Output DataFrame will have columns sorted by ascending wavenumber for standardization."
        )

    # If a reference spectrum is provided, ensure it aligns with the sorted grid
    local_kwargs = dict(kwargs)
    if reference_spectrum is not None:
        ref = np.asarray(reference_spectrum, dtype=np.float64)
        if ref.shape[0] != len(numeric_cols):
            raise ValueError(
                "reference_spectrum length must match number of spectral columns"
            )
        # Sort reference spectrum to match sorted wavenumbers
        local_kwargs["reference_spectrum"] = ref[sort_idx]
    else:
        local_kwargs["reference_spectrum"] = None

    # Process first sample to get the new wavenumber grid
    spectral_data = df[sorted_cols].values.astype(np.float64)

    # VALIDATION: Check if data appears to be transmittance instead of absorbance
    sample_size = min(100, spectral_data.shape[0])
    sample_data = spectral_data[:sample_size, :].flatten()
    sample_data_finite = sample_data[np.isfinite(sample_data)]

    if len(sample_data_finite) > 0:
        median_val = np.median(sample_data_finite)
        p95_val = np.percentile(sample_data_finite, 95)

        if p95_val > 10.0 and median_val > 1.0:
            raise ValueError(
                f"Input data appears to be transmittance (%) rather than absorbance (AU). "
                f"Detected: median={median_val:.2f}, 95th percentile={p95_val:.2f}. "
                f"Atmospheric correction should be performed on absorbance for physical validity. "
                f"Please convert your data first using: "
                f"convert_spectra(data, mode='to_absorbance') from trans_abs.py"
            )

    first_sample = spectral_data[0, :]
    _, new_wavenumbers = _exclude_and_interpolate_regions(
        intensities=first_sample,
        wavenumbers=sorted_wavenumbers,
        exclude_ranges=exclude_ranges,
        interpolate_ranges=interpolate_ranges,
        method=method,
        **local_kwargs
    )

    # OPTIMIZATION: Pre-allocate result array
    n_samples = spectral_data.shape[0]
    n_new_wavenumbers = len(new_wavenumbers)
    corrected_array = np.empty((n_samples, n_new_wavenumbers), dtype=np.float64)
    metadata_cols = [c for c in df.columns if c not in numeric_cols]

    for i in tqdm.tqdm(range(n_samples), desc="Processing Regions"):
        intensities = spectral_data[i, :]

        # Apply processing
        corrected_intensities, _ = _exclude_and_interpolate_regions(
            intensities=intensities,
            wavenumbers=sorted_wavenumbers,
            exclude_ranges=exclude_ranges,
            interpolate_ranges=interpolate_ranges,
            method=method,
            **local_kwargs
        )

        corrected_array[i, :] = corrected_intensities

    # INFO: Check for negative absorbance after correction (informational only)
    # Note: Negative values are preserved and will be handled by baseline correction
    finite_mask = np.isfinite(corrected_array)
    if np.any(finite_mask):
        n_negative = np.sum(corrected_array[finite_mask] < 0)
        if n_negative > 0:
            min_negative = np.min(corrected_array[finite_mask])
            pct_negative = 100.0 * n_negative / np.sum(finite_mask)
            logger.info(
                f"Atmospheric correction resulted in {n_negative} negative absorbance values "
                f"({pct_negative:.1f}% of valid points, min={min_negative:.4f}). "
                f"These values are preserved and will be handled by subsequent baseline correction."
            )

    new_numeric_cols = [str(wn) for wn in new_wavenumbers]

    df_corrected = pd.DataFrame(
        corrected_array,
        index=df.index,
        columns=new_numeric_cols
    )

    # Add back metadata columns
    for col in metadata_cols:
        df_corrected[col] = df[col].values

    # Reorder columns: metadata first, then spectral
    final_cols = metadata_cols + new_numeric_cols
    df_corrected = df_corrected[final_cols]

    # Convert back to polars if input was polars
    if is_polars:
        df_corrected = pl.from_pandas(df_corrected)

    return df_corrected
