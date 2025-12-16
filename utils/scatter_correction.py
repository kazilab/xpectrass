"""
Scatter Correction Module for FTIR Spectral Preprocessing
==========================================================

Provides multiplicative scatter correction (MSC), extended MSC (EMSC),
and related methods for correcting light scattering effects.
"""

from __future__ import annotations
from typing import Union, Tuple, Optional
import numpy as np


def scatter_correction(
    spectra: np.ndarray,
    method: str = "msc",
    reference: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply scatter correction to spectra matrix.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra (each row is one spectrum).
    method : str, default "msc"
        Correction method:
        - 'msc': Multiplicative Scatter Correction
        - 'emsc': Extended MSC (includes polynomial baseline terms)
        - 'snv': Standard Normal Variate (per-spectrum)
        - 'snv_detrend': SNV followed by detrending
    reference : np.ndarray, optional
        Reference spectrum for MSC/EMSC. If None, uses mean spectrum.
    **kwargs : method-specific parameters
        emsc: poly_order (default 2)
        snv_detrend: detrend_order (default 1)

    Returns
    -------
    np.ndarray
        Corrected spectra matrix.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)
    
    if method == "msc":
        return _msc(spectra, reference)
    elif method == "emsc":
        return _emsc(spectra, reference, **kwargs)
    elif method == "snv":
        return _snv_batch(spectra)
    elif method == "snv_detrend":
        return _snv_detrend_batch(spectra, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            "Valid options: msc, emsc, snv, snv_detrend"
        )


def scatter_method_names() -> list:
    """Return list of available scatter correction method names."""
    return sorted(["msc", "emsc", "snv", "snv_detrend"])


# ---------------------------------------------------------------------------
#                           INDIVIDUAL METHODS
# ---------------------------------------------------------------------------

def _msc(
    spectra: np.ndarray,
    reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Multiplicative Scatter Correction (MSC).
    
    Corrects for additive (baseline offset) and multiplicative 
    (path length) scatter effects by regressing each spectrum 
    against a reference spectrum.
    
    Model: spectrum = a + b * reference
    Corrected: (spectrum - a) / b
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
    reference : np.ndarray, optional
        Reference spectrum. If None, uses mean of spectra.
    
    Returns
    -------
    np.ndarray
        MSC-corrected spectra.
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    
    n_samples = spectra.shape[0]
    corrected = np.zeros_like(spectra)
    
    for i in range(n_samples):
        # Fit linear regression: spectrum_i = a + b * reference
        # Using least squares: [1, ref] @ [a, b]^T = spectrum_i
        X = np.column_stack([np.ones_like(reference), reference])
        coeffs = np.linalg.lstsq(X, spectra[i], rcond=None)[0]
        a, b = coeffs[0], coeffs[1]
        
        # Avoid division by zero
        if abs(b) < 1e-10:
            b = 1.0
        
        corrected[i] = (spectra[i] - a) / b
    
    return corrected


def _emsc(
    spectra: np.ndarray,
    reference: Optional[np.ndarray] = None,
    poly_order: int = 2
) -> np.ndarray:
    """
    Extended Multiplicative Scatter Correction (EMSC).
    
    Extends MSC by including polynomial baseline terms to handle
    more complex scatter patterns.
    
    Model: spectrum = a + b * reference + c1*x + c2*x² + ...
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
    reference : np.ndarray, optional
        Reference spectrum.
    poly_order : int, default 2
        Order of polynomial baseline terms.
    
    Returns
    -------
    np.ndarray
        EMSC-corrected spectra.
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)
    
    n_samples, n_points = spectra.shape
    corrected = np.zeros_like(spectra)
    
    # Create normalized x values for polynomial
    x = np.linspace(-1, 1, n_points)
    
    # Build design matrix: [1, reference, x, x², ...]
    X = [np.ones(n_points), reference]
    for p in range(1, poly_order + 1):
        X.append(x ** p)
    X = np.column_stack(X)
    
    for i in range(n_samples):
        coeffs = np.linalg.lstsq(X, spectra[i], rcond=None)[0]
        
        # Reconstruct baseline (polynomial terms only)
        baseline = coeffs[0]  # intercept
        for p in range(1, poly_order + 1):
            baseline += coeffs[2 + p - 1] * (x ** p)
        
        b = coeffs[1]  # multiplicative term
        if abs(b) < 1e-10:
            b = 1.0
        
        corrected[i] = (spectra[i] - baseline) / b
    
    return corrected


def _snv_batch(spectra: np.ndarray) -> np.ndarray:
    """
    Apply SNV (Standard Normal Variate) to each spectrum.
    
    SNV normalizes each spectrum to have mean=0 and std=1.
    """
    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    return (spectra - means) / stds


def _snv_detrend_batch(
    spectra: np.ndarray,
    detrend_order: int = 1
) -> np.ndarray:
    """
    Apply SNV followed by polynomial detrending.
    
    Removes residual baseline slope after SNV correction.
    """
    snv_spectra = _snv_batch(spectra)
    
    n_samples, n_points = snv_spectra.shape
    x = np.arange(n_points)
    
    corrected = np.zeros_like(snv_spectra)
    for i in range(n_samples):
        coeffs = np.polyfit(x, snv_spectra[i], detrend_order)
        trend = np.polyval(coeffs, x)
        corrected[i] = snv_spectra[i] - trend
    
    return corrected


# ---------------------------------------------------------------------------
#                           SINGLE SPECTRUM FUNCTIONS
# ---------------------------------------------------------------------------

def msc_single(
    spectrum: np.ndarray,
    reference: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Apply MSC to a single spectrum.
    
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


# ---------------------------------------------------------------------------
#                           EVALUATION UTILITIES
# ---------------------------------------------------------------------------

def evaluate_scatter_correction(
    spectra: np.ndarray,
    method: str = "msc"
) -> dict:
    """
    Evaluate scatter correction effectiveness.
    
    Parameters
    ----------
    spectra : np.ndarray
        Original spectra matrix.
    method : str
        Correction method.
    
    Returns
    -------
    dict
        Evaluation metrics:
        - 'variance_reduction': ratio of variance before/after
        - 'correlation_improvement': mean pairwise correlation change
    """
    corrected = scatter_correction(spectra, method=method)
    
    # Variance across samples at each wavenumber
    var_original = np.mean(np.var(spectra, axis=0))
    var_corrected = np.mean(np.var(corrected, axis=0))
    
    # Mean pairwise correlation
    n = spectra.shape[0]
    
    if n > 1:
        corr_matrix_orig = np.corrcoef(spectra)
        corr_matrix_corr = np.corrcoef(corrected)
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix_orig), k=1).astype(bool)
        mean_corr_orig = np.mean(corr_matrix_orig[mask])
        mean_corr_corr = np.mean(corr_matrix_corr[mask])
    else:
        mean_corr_orig = mean_corr_corr = 1.0
    
    return {
        'variance_ratio': var_corrected / var_original if var_original > 0 else 1.0,
        'mean_correlation_original': mean_corr_orig,
        'mean_correlation_corrected': mean_corr_corr,
        'correlation_improvement': mean_corr_corr - mean_corr_orig
    }
