"""
Spectral Derivatives Module for FTIR Preprocessing
===================================================

Compute smoothed spectral derivatives for resolution enhancement
and baseline removal.
"""

from __future__ import annotations
from typing import Union, List, Tuple
import numpy as np
from scipy import signal


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

    Returns
    -------
    np.ndarray
        Derivative spectrum.
    
    Notes
    -----
    - 1st derivative: Resolves overlapping peaks, removes constant baseline
    - 2nd derivative: Sharpens peaks, removes linear baseline
    - Higher derivatives increase noise; adjust window_length accordingly
    """
    y = np.asarray(intensities, dtype=np.float64)
    
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
    segment: int = 5
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
        Gap size (number of points to skip).
    segment : int, default 5
        Number of points to average on each side.
    
    Returns
    -------
    np.ndarray
        Gap derivative (shorter than input by gap + 2*segment).
    """
    y = np.asarray(intensities, dtype=np.float64)
    n = len(y)
    
    result_len = n - gap - 2 * segment + 1
    if result_len <= 0:
        raise ValueError("Gap and segment too large for spectrum length")
    
    result = np.zeros(result_len)
    
    for i in range(result_len):
        left_avg = np.mean(y[i:i + segment])
        right_avg = np.mean(y[i + segment + gap:i + 2 * segment + gap])
        result[i] = right_avg - left_avg
    
    # Pad to original length
    pad_left = (gap + 2 * segment - 1) // 2
    pad_right = n - result_len - pad_left
    return np.pad(result, (pad_left, pad_right), mode='edge')


def derivative_with_smoothing(
    intensities: np.ndarray,
    order: int = 1,
    smooth_window: int = 11,
    deriv_window: int = 15,
    smooth_first: bool = True
) -> np.ndarray:
    """
    Apply derivative with separate smoothing control.
    
    Parameters
    ----------
    intensities : np.ndarray
        1-D spectrum.
    order : int
        Derivative order.
    smooth_window : int
        Window for initial smoothing (if smooth_first=True).
    deriv_window : int
        Window for derivative calculation.
    smooth_first : bool, default True
        If True, smooth before taking derivative.
    
    Returns
    -------
    np.ndarray
        Derivative spectrum.
    """
    y = np.asarray(intensities, dtype=np.float64)
    
    # Ensure windows are odd
    if smooth_window % 2 == 0:
        smooth_window += 1
    if deriv_window % 2 == 0:
        deriv_window += 1
    
    if smooth_first:
        y = signal.savgol_filter(y, smooth_window, 3, deriv=0)
    
    return signal.savgol_filter(y, deriv_window, order + 1, deriv=order)


# ---------------------------------------------------------------------------
#                           BATCH OPERATIONS
# ---------------------------------------------------------------------------

def derivative_batch(
    spectra: np.ndarray,
    order: int = 1,
    window_length: int = 15,
    polyorder: int = 3
) -> np.ndarray:
    """
    Compute derivatives for multiple spectra.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Matrix of spectra.
    order : int
        Derivative order.
    window_length : int
        Savitzky-Golay window.
    polyorder : int
        Polynomial order.
    
    Returns
    -------
    np.ndarray
        Derivative spectra matrix.
    """
    return np.array([
        spectral_derivative(s, order, window_length, polyorder) 
        for s in spectra
    ])


# ---------------------------------------------------------------------------
#                           VISUALIZATION
# ---------------------------------------------------------------------------

def plot_derivatives(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    orders: List[int] = [0, 1, 2],
    sample_name: str = "",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot spectrum and its derivatives.
    
    Parameters
    ----------
    intensities : np.ndarray
        Original spectrum.
    wavenumbers : np.ndarray
        Wavenumber axis.
    orders : list of int
        Derivative orders to plot (0 = original).
    sample_name : str
        Sample name for title.
    figsize : tuple
        Figure size.
    """
    import matplotlib.pyplot as plt
    
    n_plots = len(orders)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    labels = {0: 'Original', 1: '1st Derivative', 2: '2nd Derivative', 
              3: '3rd Derivative', 4: '4th Derivative'}
    
    for ax, order in zip(axes, orders):
        if order == 0:
            y = intensities
        else:
            y = spectral_derivative(intensities, order=order)
        
        ax.plot(wavenumbers, y, 'b-', lw=0.8)
        ax.set_ylabel(labels.get(order, f'{order}th Derivative'))
        ax.grid(alpha=0.3)
    
    axes[0].set_title(f'Spectral Derivatives: {sample_name}')
    axes[-1].set_xlabel('Wavenumber (cm⁻¹)')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()
