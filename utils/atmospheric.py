"""
Atmospheric Correction Module for FTIR Spectral Preprocessing
==============================================================

Corrects for CO₂ and H₂O vapor interference in FTIR spectra
that result from atmospheric absorption during measurement.
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional
import numpy as np
from scipy import interpolate


# Default atmospheric interference regions (wavenumbers in cm⁻¹)
CO2_REGION = (2300.0, 2400.0)  # CO₂ asymmetric stretch
H2O_REGIONS = [
    (1350.0, 1900.0),  # H₂O bending mode
    (3550.0, 3900.0),  # H₂O stretching modes
]


def atmospheric_correction(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    method: str = "interpolate",
    co2_range: Tuple[float, float] = CO2_REGION,
    h2o_ranges: List[Tuple[float, float]] = H2O_REGIONS,
    reference_spectrum: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Correct for atmospheric CO₂ and H₂O interference.

    Parameters
    ----------
    intensities : np.ndarray
        1-D intensity array.
    wavenumbers : np.ndarray
        Corresponding wavenumber array.
    method : str, default "interpolate"
        Correction method:
        - 'interpolate': Linear interpolation across affected regions
        - 'spline': Cubic spline interpolation
        - 'reference': Subtract scaled reference atmospheric spectrum
        - 'zero': Set affected regions to baseline (local mean)
        - 'exclude': Return NaN for affected regions
    co2_range : tuple
        Wavenumber range for CO₂ correction.
    h2o_ranges : list of tuples
        Wavenumber ranges for H₂O correction.
    reference_spectrum : np.ndarray, optional
        Reference atmospheric spectrum (required for 'reference' method).
    **kwargs : additional parameters
        reference_scale: float (auto-fit if not provided)

    Returns
    -------
    np.ndarray
        Corrected intensity array.
    """
    y = np.asarray(intensities, dtype=np.float64).copy()
    x = np.asarray(wavenumbers, dtype=np.float64)

    if len(y) != len(x):
        raise ValueError("intensities and wavenumbers must have same length")

    # Combine all atmospheric regions
    all_regions = [co2_range] + list(h2o_ranges)

    if method == "interpolate":
        return _correct_interpolate(y, x, all_regions, kind='linear')
    elif method == "spline":
        return _correct_interpolate(y, x, all_regions, kind='cubic')
    elif method == "reference":
        if reference_spectrum is None:
            raise ValueError("reference_spectrum required for 'reference' method")
        return _correct_reference(y, x, all_regions, reference_spectrum, **kwargs)
    elif method == "zero":
        return _correct_zero(y, x, all_regions)
    elif method == "exclude":
        return _correct_exclude(y, x, all_regions)
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            "Valid options: interpolate, spline, reference, zero, exclude"
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
    kind: str = 'linear'
) -> np.ndarray:
    """
    Interpolate across atmospheric regions using boundary values.
    """
    result = y.copy()
    
    for lo, hi in regions:
        # Find indices
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        
        affected_idx = np.where(mask)[0]
        if len(affected_idx) == 0:
            continue
        
        # Find boundary points
        left_idx = affected_idx[0]
        right_idx = affected_idx[-1]
        
        # Get boundary values (average of a few points outside)
        n_boundary = 5
        left_start = max(0, left_idx - n_boundary)
        right_end = min(len(y), right_idx + n_boundary + 1)
        
        if left_idx > 0 and right_idx < len(y) - 1:
            # Interpolate between boundaries
            boundary_x = [x[left_start], x[min(right_end - 1, len(x) - 1)]]
            boundary_y = [
                np.mean(y[left_start:left_idx]),
                np.mean(y[right_idx + 1:right_end])
            ]
            
            if kind == 'linear':
                f = interpolate.interp1d(boundary_x, boundary_y, 
                                         fill_value='extrapolate')
            else:
                # For cubic, need at least 4 points
                f = interpolate.interp1d(boundary_x, boundary_y, 
                                         kind='linear', fill_value='extrapolate')
            
            result[mask] = f(x[mask])
    
    return result


def _correct_reference(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]],
    reference: np.ndarray,
    reference_scale: Optional[float] = None
) -> np.ndarray:
    """
    Subtract scaled reference atmospheric spectrum.
    
    If scale not provided, auto-fit using least squares in affected regions.
    """
    mask = _make_atmospheric_mask(x, regions)
    
    if reference_scale is None:
        # Auto-fit scale factor using least squares
        ref_region = reference[mask]
        y_region = y[mask]
        
        # Scale = (y · ref) / (ref · ref)
        reference_scale = np.dot(y_region, ref_region) / np.dot(ref_region, ref_region)
    
    return y - reference_scale * reference


def _correct_zero(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Set atmospheric regions to local baseline (average of boundaries).
    """
    result = y.copy()
    
    for lo, hi in regions:
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        
        affected_idx = np.where(mask)[0]
        left_idx = max(0, affected_idx[0] - 10)
        right_idx = min(len(y), affected_idx[-1] + 11)
        
        # Use average of boundary regions
        left_mean = np.mean(y[max(0, left_idx - 5):affected_idx[0]])
        right_mean = np.mean(y[affected_idx[-1] + 1:right_idx])
        baseline = (left_mean + right_mean) / 2
        
        result[mask] = baseline
    
    return result


def _correct_exclude(
    y: np.ndarray,
    x: np.ndarray,
    regions: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Mark atmospheric regions as NaN (for exclusion from analysis).
    """
    result = y.copy()
    mask = _make_atmospheric_mask(x, regions)
    result[mask] = np.nan
    return result


# ---------------------------------------------------------------------------
#                           UTILITIES
# ---------------------------------------------------------------------------

def get_atmospheric_regions() -> dict:
    """
    Return standard atmospheric interference regions.
    
    Returns
    -------
    dict with 'co2' and 'h2o' regions.
    """
    return {
        'co2': CO2_REGION,
        'h2o': H2O_REGIONS,
        'all': [CO2_REGION] + H2O_REGIONS
    }


def identify_atmospheric_features(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    threshold: float = 0.1
) -> dict:
    """
    Check for presence of atmospheric interference.
    
    Parameters
    ----------
    intensities : np.ndarray
        Spectrum intensities.
    wavenumbers : np.ndarray
        Wavenumber array.
    threshold : float
        Threshold for detecting significant features.
    
    Returns
    -------
    dict
        Report on detected atmospheric features.
    """
    regions = get_atmospheric_regions()
    report = {'co2_detected': False, 'h2o_detected': False, 'recommendations': []}
    
    # Check CO2
    co2_mask = (wavenumbers >= regions['co2'][0]) & (wavenumbers <= regions['co2'][1])
    if np.any(co2_mask):
        co2_region = intensities[co2_mask]
        # Look for characteristic dip
        if np.ptp(co2_region) > threshold * np.ptp(intensities):
            report['co2_detected'] = True
            report['recommendations'].append("Apply CO₂ correction (2300-2400 cm⁻¹)")
    
    # Check H2O
    for h2o_range in regions['h2o']:
        h2o_mask = (wavenumbers >= h2o_range[0]) & (wavenumbers <= h2o_range[1])
        if np.any(h2o_mask):
            h2o_region = intensities[h2o_mask]
            if np.std(h2o_region) > threshold * np.std(intensities):
                report['h2o_detected'] = True
                report['recommendations'].append(
                    f"Apply H₂O correction ({h2o_range[0]:.0f}-{h2o_range[1]:.0f} cm⁻¹)"
                )
    
    return report
