"""
Region Selection Module for FTIR Spectral Preprocessing
========================================================

Provides utilities for selecting, excluding, and extracting
spectral regions based on wavenumber ranges.
"""

from __future__ import annotations
from typing import Union, List, Tuple, Optional
import numpy as np
import polars as pl
import pandas as pd


# ---------------------------------------------------------------------------
#                    COMMON FTIR REGIONS FOR PLASTICS
# ---------------------------------------------------------------------------

FTIR_REGIONS = {
    # Full spectrum
    'full': (400, 4000),
    
    # Main regions
    'fingerprint': (400, 1500),
    'functional': (1500, 4000),
    
    # Specific functional groups
    'ch_stretch': (2800, 3100),       # C-H stretching
    'ch_bend': (1350, 1480),          # C-H bending
    'carbonyl': (1650, 1800),          # C=O stretch (PET, etc.)
    'aromatic': (1400, 1600),          # Aromatic ring (PS, PET)
    'oh_stretch': (3200, 3600),        # O-H stretching
    'ether': (1000, 1300),             # C-O-C stretch
    
    # Plastic-specific diagnostic regions
    'hdpe_ldpe': (700, 750),           # CH2 rocking (PE identification)
    'pp_methyl': (1370, 1380),         # CH3 symmetric deformation
    'ps_aromatic': (690, 760),         # Aromatic out-of-plane bending
    'pet_ester': (1710, 1730),         # Ester C=O stretch
    'pvc_ccl': (600, 700),             # C-Cl stretch
    
    # Atmospheric interference (for exclusion)
    'co2': (2300, 2400),
    'h2o_bend': (1350, 1900),
    'h2o_stretch': (3550, 3900),
}


def get_region_names() -> List[str]:
    """Return list of predefined region names."""
    return list(FTIR_REGIONS.keys())


def get_region_range(name: str) -> Tuple[float, float]:
    """Get wavenumber range for a named region."""
    if name not in FTIR_REGIONS:
        raise ValueError(f"Unknown region: '{name}'. Available: {get_region_names()}")
    return FTIR_REGIONS[name]


# ---------------------------------------------------------------------------
#                           SELECTION FUNCTIONS
# ---------------------------------------------------------------------------

def select_region(
    df: pl.DataFrame,
    regions: Union[Tuple[float, float], List[Tuple[float, float]], str]
) -> pl.DataFrame:
    """
    Select spectral regions by wavenumber ranges.

    Parameters
    ----------
    df : pl.DataFrame
        Wide-format DataFrame with 'sample', 'label', and wavenumber columns.
    regions : tuple, list of tuples, or str
        - tuple: (start, end) wavenumber range
        - list of tuples: multiple ranges to include
        - str: predefined region name (e.g., 'fingerprint', 'ch_stretch')

    Returns
    -------
    pl.DataFrame
        DataFrame with only selected wavenumber columns.
    
    Examples
    --------
    >>> select_region(df, (400, 1500))  # Fingerprint region
    >>> select_region(df, 'ch_stretch')  # Named region
    >>> select_region(df, [(400, 1500), (2800, 3100)])  # Multiple regions
    """
    # Convert string to range
    if isinstance(regions, str):
        regions = [get_region_range(regions)]
    elif isinstance(regions, tuple):
        regions = [regions]
    
    # Get wavenumber columns
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    wavenumbers = np.array([float(c) for c in wavenumber_cols])
    
    # Create mask for selected regions
    mask = np.zeros(len(wavenumbers), dtype=bool)
    for start, end in regions:
        mask |= (wavenumbers >= start) & (wavenumbers <= end)
    
    # Select columns
    selected_cols = ['sample', 'label'] + [c for c, m in zip(wavenumber_cols, mask) if m]
    
    return df.select(selected_cols)


def exclude_regions(
    df: pl.DataFrame,
    regions: Union[Tuple[float, float], List[Tuple[float, float]], str]
) -> pl.DataFrame:
    """
    Exclude spectral regions (opposite of select_region).

    Parameters
    ----------
    df : pl.DataFrame
        Wide-format DataFrame.
    regions : tuple, list of tuples, or str
        Regions to exclude.

    Returns
    -------
    pl.DataFrame
        DataFrame with excluded regions removed.
    """
    # Convert string to range
    if isinstance(regions, str):
        regions = [get_region_range(regions)]
    elif isinstance(regions, tuple):
        regions = [regions]
    
    # Get wavenumber columns
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    wavenumbers = np.array([float(c) for c in wavenumber_cols])
    
    # Create mask for excluded regions
    exclude_mask = np.zeros(len(wavenumbers), dtype=bool)
    for start, end in regions:
        exclude_mask |= (wavenumbers >= start) & (wavenumbers <= end)
    
    # Select columns NOT in excluded regions
    selected_cols = ['sample', 'label'] + [
        c for c, excluded in zip(wavenumber_cols, exclude_mask) if not excluded
    ]
    
    return df.select(selected_cols)


def exclude_atmospheric(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convenience function to exclude atmospheric interference regions.
    
    Excludes CO2 (2300-2400 cm⁻¹) and H2O (1350-1900, 3550-3900 cm⁻¹).
    """
    atmospheric_regions = [
        FTIR_REGIONS['co2'],
        FTIR_REGIONS['h2o_bend'],
        FTIR_REGIONS['h2o_stretch']
    ]
    return exclude_regions(df, atmospheric_regions)


# ---------------------------------------------------------------------------
#                           NUMPY ARRAY FUNCTIONS
# ---------------------------------------------------------------------------

def select_region_np(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    start: float,
    end: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select region from 1-D spectrum arrays.

    Parameters
    ----------
    intensities : np.ndarray
        Intensity values.
    wavenumbers : np.ndarray
        Wavenumber values.
    start, end : float
        Wavenumber range.

    Returns
    -------
    selected_intensities : np.ndarray
    selected_wavenumbers : np.ndarray
    """
    mask = (wavenumbers >= start) & (wavenumbers <= end)
    return intensities[mask], wavenumbers[mask]


def select_regions_np(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    regions: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select multiple regions from 1-D spectrum arrays.
    
    Regions are concatenated in the order provided.
    """
    mask = np.zeros(len(wavenumbers), dtype=bool)
    for start, end in regions:
        mask |= (wavenumbers >= start) & (wavenumbers <= end)
    
    return intensities[mask], wavenumbers[mask]


# ---------------------------------------------------------------------------
#                           REGION ANALYSIS
# ---------------------------------------------------------------------------

def analyze_regions(
    df: pl.DataFrame,
    regions: Optional[List[Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Analyze intensity statistics across different spectral regions.

    Parameters
    ----------
    df : pl.DataFrame
        Wide-format spectral DataFrame.
    regions : list of tuples, optional
        Regions to analyze. If None, analyzes predefined plastic regions.

    Returns
    -------
    pd.DataFrame
        Statistics for each region (mean, std, min, max, peak location).
    """
    if regions is None:
        regions = [
            ('fingerprint', FTIR_REGIONS['fingerprint']),
            ('ch_stretch', FTIR_REGIONS['ch_stretch']),
            ('carbonyl', FTIR_REGIONS['carbonyl']),
            ('aromatic', FTIR_REGIONS['aromatic']),
        ]
    else:
        regions = [(f"{r[0]}-{r[1]}", r) for r in regions]
    
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    wavenumbers = np.array([float(c) for c in wavenumber_cols])
    spectra = df.select(wavenumber_cols).to_numpy().astype(float)
    
    results = []
    for name, (start, end) in regions:
        mask = (wavenumbers >= start) & (wavenumbers <= end)
        if not np.any(mask):
            continue
        
        region_data = spectra[:, mask]
        region_wn = wavenumbers[mask]
        
        # Find peak location for each spectrum
        peak_indices = np.argmax(np.abs(region_data), axis=1)
        peak_wavenumbers = region_wn[peak_indices]
        
        results.append({
            'region': name,
            'start_cm': start,
            'end_cm': end,
            'n_points': np.sum(mask),
            'mean_intensity': np.mean(region_data),
            'std_intensity': np.std(region_data),
            'min_intensity': np.min(region_data),
            'max_intensity': np.max(region_data),
            'mean_peak_location': np.mean(peak_wavenumbers),
        })
    
    return pd.DataFrame(results)


def get_wavenumbers(df: pl.DataFrame) -> np.ndarray:
    """Extract wavenumber array from DataFrame columns."""
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    return np.array([float(c) for c in wavenumber_cols])


def get_spectra_matrix(df: pl.DataFrame) -> np.ndarray:
    """Extract spectra as numpy matrix (n_samples, n_wavenumbers)."""
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    return df.select(wavenumber_cols).to_numpy().astype(float)
