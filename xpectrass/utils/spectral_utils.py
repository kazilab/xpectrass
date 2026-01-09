"""
Shared Spectral Data Utilities
===============================

Common utilities for spectral data processing across modules.
Provides robust column detection, sorting, and validation for FTIR spectra.
"""

from __future__ import annotations

from typing import Tuple, List, Any, Sequence, Optional
import logging

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)


def _infer_spectral_columns(
    df: pd.DataFrame,
    exclude_columns: Sequence[str],
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
) -> Tuple[List[Any], np.ndarray]:
    """Infer spectral columns by parsing column names as wavenumbers.

    This is safer than selecting by dtype because real datasets often contain
    numeric metadata columns (IDs, temperatures, timestamps) that should NOT be
    treated as spectra, and spectral columns sometimes arrive as object dtype.

    Args:
        df: Input DataFrame
        exclude_columns: Column names to exclude from spectral detection
        wn_min: Minimum wavenumber bound. If None, uses 200.0 cm⁻¹ as default,
                or auto-expands if no columns found within default range.
        wn_max: Maximum wavenumber bound. If None, uses 8000.0 cm⁻¹ as default,
                or auto-expands if no columns found within default range.

    Returns:
        Tuple of (column_list, wavenumber_array)

    Raises:
        ValueError: If no spectral columns detected or duplicates found
    """
    cols: List[Any] = []
    wns: List[float] = []
    exclude_set = set(exclude_columns)

    # Default bounds for typical FTIR spectra
    default_wn_min = 200.0
    default_wn_max = 8000.0

    # Use provided bounds or defaults
    use_wn_min = wn_min if wn_min is not None else default_wn_min
    use_wn_max = wn_max if wn_max is not None else default_wn_max

    # First pass: collect all parseable numeric columns
    all_numeric_cols: List[Any] = []
    all_numeric_wns: List[float] = []

    for c in df.columns:
        if c in exclude_set:
            continue
        try:
            wn = float(c)
            all_numeric_cols.append(c)
            all_numeric_wns.append(wn)
        except Exception:
            continue

    if not all_numeric_cols:
        raise ValueError(
            "No spectral columns detected. Expected column names parseable as "
            "wavenumbers (e.g., '4000', '3998.0')."
        )

    # Second pass: filter by bounds
    for col, wn in zip(all_numeric_cols, all_numeric_wns):
        if use_wn_min <= wn <= use_wn_max:
            cols.append(col)
            wns.append(wn)

    # If no columns within bounds and user didn't specify custom bounds,
    # auto-expand to include all numeric columns
    if not cols and wn_min is None and wn_max is None:
        cols = all_numeric_cols
        wns = all_numeric_wns
        actual_min = min(wns)
        actual_max = max(wns)
        logger.warning(
            f"No columns found in default wavenumber range ({default_wn_min}-{default_wn_max} cm⁻¹). "
            f"Auto-expanded to include all numeric columns ({actual_min:.1f}-{actual_max:.1f} cm⁻¹). "
            "Specify wn_min/wn_max explicitly if this is incorrect."
        )
    elif not cols:
        raise ValueError(
            f"No spectral columns detected within specified bounds "
            f"({use_wn_min}-{use_wn_max} cm⁻¹). "
            f"Found {len(all_numeric_cols)} numeric columns with range "
            f"({min(all_numeric_wns):.1f}-{max(all_numeric_wns):.1f} cm⁻¹)."
        )

    wn_arr = np.asarray(wns, dtype=np.float64)
    # Duplicates can break sorting/interpolation logic
    if np.unique(wn_arr).size != wn_arr.size:
        raise ValueError("Detected duplicate wavenumber columns after parsing.")

    return cols, wn_arr


def _sort_spectral_columns(
    spectral_cols: Sequence[Any],
    wavenumbers: np.ndarray,
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    """Return columns/wavenumbers sorted by wavenumber (ascending).

    Args:
        spectral_cols: List of column names
        wavenumbers: Array of wavenumber values

    Returns:
        Tuple of (sorted_columns, sorted_wavenumbers, sort_indices)
    """
    sort_idx = np.argsort(wavenumbers)
    sorted_cols = [spectral_cols[i] for i in sort_idx]
    sorted_wn = wavenumbers[sort_idx]
    return sorted_cols, sorted_wn, sort_idx


def _is_monotonic_strict(x: np.ndarray) -> bool:
    """Return True if x is strictly increasing or strictly decreasing.

    Args:
        x: Input array

    Returns:
        True if array is strictly monotonic, False otherwise
    """
    if x.size < 2:
        return True
    d = np.diff(x)
    return bool(np.all(d > 0) or np.all(d < 0))
