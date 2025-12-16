"""
Normalization Module for FTIR Spectral Preprocessing
=====================================================

Provides multiple normalization methods for FTIR spectra including
SNV, vector, area, min-max, and peak normalization.
Also includes mean centering for PCA/PLS preparation.
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional
import numpy as np
import pandas as pd
import polars as pl


def normalize(
    intensities: Union[np.ndarray, list],
    method: str = "snv",
    **kwargs
) -> np.ndarray:
    """
    Normalize a 1-D FTIR spectrum.

    Parameters
    ----------
    intensities : array-like
        Intensity values (1-D).
    method : str, default "snv"
        Normalization method. Options:
        - 'snv': Standard Normal Variate (mean=0, std=1)
        - 'vector': L2 vector normalization (unit length)
        - 'minmax': Min-Max scaling to [0, 1]
        - 'area': Area normalization (total area = 1)
        - 'peak': Normalize by peak intensity
        - 'range': Normalize by intensity range
        - 'max': Normalize by maximum value

    **kwargs : method-specific parameters
        peak: peak_idx (index) or peak_wavenumber (requires wavenumbers array)
        minmax: feature_range (tuple, default (0, 1))

    Returns
    -------
    np.ndarray
        Normalized intensity values.
    """
    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("`intensities` must be a 1-D array-like object.")

    if method == "snv":
        return _normalize_snv(y)
    elif method == "vector":
        return _normalize_vector(y)
    elif method == "minmax":
        return _normalize_minmax(y, **kwargs)
    elif method == "area":
        return _normalize_area(y)
    elif method == "peak":
        return _normalize_peak(y, **kwargs)
    elif method == "range":
        return _normalize_range(y)
    elif method == "max":
        return _normalize_max(y)
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            "Valid options: snv, vector, minmax, area, peak, range, max"
        )


def normalize_method_names() -> List[str]:
    """Return list of available normalization method names."""
    return sorted(["snv", "vector", "minmax", "area", "peak", "range", "max"])


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


def _normalize_area(y: np.ndarray) -> np.ndarray:
    """
    Area normalization.
    
    Scales spectrum so total area (sum of absolute values) equals 1.
    Useful for comparing relative concentrations.
    
    Formula: x / sum(|x|)
    """
    total_area = np.sum(np.abs(y))
    if total_area == 0:
        return np.zeros_like(y)
    return y / total_area


def _normalize_peak(
    y: np.ndarray,
    peak_idx: Optional[int] = None,
    peak_value: Optional[float] = None
) -> np.ndarray:
    """
    Peak normalization.
    
    Normalize by the intensity at a specific peak or maximum.
    Useful when reference peak intensity is known/expected.
    
    Parameters
    ----------
    peak_idx : int, optional
        Index of the normalization peak. If None, uses maximum.
    peak_value : float, optional
        If provided, normalize to this value at peak_idx.
    """
    if peak_idx is None:
        # Use maximum value
        ref_intensity = np.max(np.abs(y))
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

def normalize_batch(
    spectra: np.ndarray,
    method: str = "snv",
    **kwargs
) -> np.ndarray:
    """
    Normalize multiple spectra.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    method : str
        Normalization method.
    **kwargs : method-specific parameters
    
    Returns
    -------
    np.ndarray
        Normalized spectra matrix.
    """
    return np.array([normalize(s, method=method, **kwargs) for s in spectra])


# ---------------------------------------------------------------------------
#                           MEAN CENTERING
# ---------------------------------------------------------------------------

def mean_center(
    spectra: np.ndarray,
    axis: int = 0,
    return_mean: bool = True
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
    return_mean : bool, default True
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
    return_params: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Auto-scaling (mean centering + unit variance scaling).
    
    Each variable (wavenumber) is scaled to have mean=0 and std=1.
    Common preprocessing for PCA/PLS when variables have different scales.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    return_params : bool, default True
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
    return_params: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pareto scaling (mean centering + sqrt(std) scaling).
    
    Less aggressive than auto-scaling; preserves more of the 
    original data structure. Good for spectral data.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    return_params : bool, default True
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
    order: int = 1
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
    
    Returns
    -------
    np.ndarray
        Detrended spectrum.
    """
    x = np.arange(len(intensities))
    coeffs = np.polyfit(x, intensities, order)
    trend = np.polyval(coeffs, x)
    return intensities - trend


def snv_detrend(intensities: np.ndarray, detrend_order: int = 1) -> np.ndarray:
    """
    SNV followed by detrending.
    
    Common combined preprocessing for scatter correction.
    """
    snv_result = _normalize_snv(intensities)
    return detrend(snv_result, order=detrend_order)


# ---------------------------------------------------------------------------
#                           POLARS INTEGRATION
# ---------------------------------------------------------------------------

def normalize_dataframe(
    df: pl.DataFrame,
    method: str = "snv",
    **kwargs
) -> pl.DataFrame:
    """
    Apply normalization to all spectra in a Polars DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        Wide-format DataFrame with 'sample', 'label', and wavenumber columns.
    method : str
        Normalization method.
    
    Returns
    -------
    pl.DataFrame
        Normalized DataFrame with same structure.
    """
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    
    # Extract spectra as numpy array
    spectra = df.select(wavenumber_cols).to_numpy().astype(float)
    
    # Normalize each spectrum
    normalized = normalize_batch(spectra, method=method, **kwargs)
    
    # Reconstruct DataFrame
    result = df.select(['sample', 'label'])
    for i, col in enumerate(wavenumber_cols):
        result = result.with_columns(pl.lit(normalized[:, i]).alias(col))
    
    return result


def mean_center_dataframe(
    df: pl.DataFrame
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Mean-center a Polars DataFrame of spectra.
    
    Parameters
    ----------
    df : pl.DataFrame
        Wide-format DataFrame.
    
    Returns
    -------
    pl.DataFrame
        Mean-centered DataFrame.
    np.ndarray
        Mean spectrum for reconstruction.
    """
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    
    spectra = df.select(wavenumber_cols).to_numpy().astype(float)
    centered, mean = mean_center(spectra, axis=0)
    
    result = df.select(['sample', 'label'])
    for i, col in enumerate(wavenumber_cols):
        result = result.with_columns(pl.lit(centered[:, i]).alias(col))
    
    return result, mean
