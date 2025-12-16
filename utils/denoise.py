"""
Denoising Module for FTIR Spectral Preprocessing
=================================================

Provides multiple noise reduction methods for FTIR spectra with
unified interface and evaluation utilities.
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional
import numpy as np
import pandas as pd
import polars as pl
from scipy import signal, ndimage
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
import pywt


def denoise(
    intensities: Union[np.ndarray, list],
    method: str = "savgol",
    **kwargs
) -> np.ndarray:
    """
    Denoise a 1-D FTIR spectrum using various filtering methods.

    Parameters
    ----------
    intensities : array-like
        Raw intensity values (1-D).
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
        Denoised intensity values.
    """
    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("`intensities` must be a 1-D array-like object.")

    if method == "savgol":
        return _denoise_savgol(y, **kwargs)
    elif method == "wavelet":
        return _denoise_wavelet(y, **kwargs)
    elif method == "moving_average":
        return _denoise_moving_average(y, **kwargs)
    elif method == "gaussian":
        return _denoise_gaussian(y, **kwargs)
    elif method == "median":
        return _denoise_median(y, **kwargs)
    elif method == "whittaker":
        return _denoise_whittaker(y, **kwargs)
    elif method == "lowpass":
        return _denoise_lowpass(y, **kwargs)
    else:
        raise ValueError(
            f"Unknown denoising method: '{method}'. "
            "Valid options: savgol, wavelet, moving_average, gaussian, "
            "median, whittaker, lowpass"
        )


def denoise_method_names() -> List[str]:
    """Return list of available denoising method names."""
    return sorted([
        "savgol", "wavelet", "moving_average", "gaussian",
        "median", "whittaker", "lowpass"
    ])


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
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    # Ensure polyorder < window_length
    polyorder = min(polyorder, window_length - 1)
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
    """
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
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return signal.medfilt(y, kernel_size=kernel_size)


def _denoise_whittaker(
    y: np.ndarray,
    lam: float = 1e4,
    d: int = 2
) -> np.ndarray:
    """
    Whittaker smoother (penalized least squares).
    
    Balances fidelity to data with smoothness penalty.
    Higher lambda = smoother result.
    """
    n = len(y)
    # Create difference matrix
    E = np.eye(n)
    D = np.diff(E, n=d, axis=0)
    # Solve (I + lam * D'D) z = y
    W = E + lam * D.T @ D
    z = np.linalg.solve(W, y)
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
    flat_regions: Optional[List[Tuple[int, int]]] = None
) -> float:
    """
    Estimate Signal-to-Noise Ratio improvement.
    
    Parameters
    ----------
    y_raw : np.ndarray
        Original noisy spectrum.
    y_denoised : np.ndarray
        Denoised spectrum.
    flat_regions : list of (start, end) index tuples, optional
        Index regions known to be baseline-only (for noise estimation).
        If None, uses a simple high-frequency component estimation.
    
    Returns
    -------
    float
        Estimated SNR in dB.
    """
    if flat_regions:
        # Estimate noise from difference in flat regions
        noise_samples = []
        for start, end in flat_regions:
            noise_samples.extend(y_raw[start:end] - y_denoised[start:end])
        noise_std = np.std(noise_samples)
    else:
        # Estimate noise from high-frequency residuals
        residual = y_raw - y_denoised
        noise_std = np.std(residual)
    
    signal_power = np.var(y_denoised)
    noise_power = noise_std ** 2
    
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    return np.inf


def evaluate_denoising(
    df: pl.DataFrame,
    methods: Optional[List[str]] = None,
    n_samples: int = 50
) -> pd.DataFrame:
    """
    Compare denoising methods on a subset of spectra.
    
    Parameters
    ----------
    df : pl.DataFrame
        Wide-format spectral DataFrame.
    methods : list of str, optional
        Methods to evaluate. Default: all available methods.
    n_samples : int, default 50
        Number of samples to evaluate.
    
    Returns
    -------
    pd.DataFrame
        Evaluation metrics (SNR improvement, smoothness) for each method.
    """
    if methods is None:
        methods = denoise_method_names()
    
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    samples = df['sample'].to_list()[:n_samples]
    
    results = []
    for sample in samples:
        row = df.filter(pl.col('sample') == sample).select(wavenumber_cols)
        y = row.to_numpy().ravel().astype(float)
        
        for method in methods:
            try:
                y_denoised = denoise(y, method=method)
                snr_improvement = estimate_snr(y, y_denoised)
                
                # Compute smoothness (inverse of 2nd derivative variance)
                d2 = np.diff(y_denoised, n=2)
                smoothness = 1.0 / (np.var(d2) + 1e-10)
                
                # Compute fidelity (correlation with original)
                fidelity = np.corrcoef(y, y_denoised)[0, 1]
                
                results.append({
                    'sample': sample,
                    'method': method,
                    'snr_db': snr_improvement,
                    'smoothness': smoothness,
                    'fidelity': fidelity
                })
            except Exception as e:
                results.append({
                    'sample': sample,
                    'method': method,
                    'snr_db': np.nan,
                    'smoothness': np.nan,
                    'fidelity': np.nan
                })
    
    return pd.DataFrame(results)


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
    import matplotlib.pyplot as plt
    
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
