"""
Data Validation Module for FTIR Spectral Preprocessing
======================================================

Provides comprehensive validation checks for FTIR spectral data to ensure
data quality before preprocessing and analysis.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path


def validate_spectra(
    df: pl.DataFrame,
    expected_samples_per_class: int = 500,
    expected_classes: Optional[List[str]] = None,
    wavenumber_range: Tuple[float, float] = (399.0, 4000.0),
    intensity_range: Tuple[float, float] = (0.0, 150.0),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive validation of FTIR spectral data.

    Parameters
    ----------
    df : pl.DataFrame
        Wide-format DataFrame with columns: 'sample', 'label', and wavenumber columns.
    expected_samples_per_class : int, default 500
        Expected number of samples per plastic type.
    expected_classes : list of str, optional
        Expected class labels. Default: ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']
    wavenumber_range : tuple, default (399.0, 4000.0)
        Expected (min, max) wavenumber range in cm⁻¹.
    intensity_range : tuple, default (0.0, 150.0)
        Valid intensity range for %T values.
    verbose : bool, default True
        Print validation report to console.

    Returns
    -------
    dict
        Validation report with keys:
        - 'valid': bool - overall pass/fail
        - 'n_samples': int - total samples
        - 'n_wavenumbers': int - spectral points
        - 'class_counts': dict - samples per label
        - 'missing_values': int - count of NaN/Inf
        - 'out_of_range': dict - samples with intensities outside range
        - 'wavenumber_check': dict - actual vs expected range
        - 'duplicates': list - duplicate sample names
        - 'issues': list - list of issue descriptions
    """
    if expected_classes is None:
        expected_classes = ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']

    report = {
        'valid': True,
        'n_samples': 0,
        'n_wavenumbers': 0,
        'class_counts': {},
        'missing_values': 0,
        'out_of_range': {},
        'wavenumber_check': {},
        'duplicates': [],
        'issues': []
    }

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------
    required_cols = {'sample', 'label'}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        report['issues'].append(f"Missing required columns: {missing}")
        report['valid'] = False
        return report

    # -------------------------------------------------------------------------
    # Basic counts
    # -------------------------------------------------------------------------
    report['n_samples'] = df.height
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    report['n_wavenumbers'] = len(wavenumber_cols)

    # -------------------------------------------------------------------------
    # Class distribution
    # -------------------------------------------------------------------------
    class_counts = df.group_by('label').agg(pl.count().alias('count'))
    report['class_counts'] = dict(
        zip(class_counts['label'].to_list(), class_counts['count'].to_list())
    )

    # Check for expected classes
    actual_classes = set(report['class_counts'].keys())
    missing_classes = set(expected_classes) - actual_classes
    if missing_classes:
        report['issues'].append(f"Missing classes: {missing_classes}")
        report['valid'] = False

    # Check sample counts per class
    for label, count in report['class_counts'].items():
        if count != expected_samples_per_class:
            report['issues'].append(
                f"Class '{label}' has {count} samples (expected {expected_samples_per_class})"
            )

    # -------------------------------------------------------------------------
    # Wavenumber range check
    # -------------------------------------------------------------------------
    try:
        wavenumbers = np.array([float(c) for c in wavenumber_cols])
        actual_min, actual_max = wavenumbers.min(), wavenumbers.max()
        report['wavenumber_check'] = {
            'actual_range': (actual_min, actual_max),
            'expected_range': wavenumber_range,
            'n_points': len(wavenumbers),
            'spacing': np.mean(np.diff(np.sort(wavenumbers)))
        }
        if actual_min > wavenumber_range[0] + 10 or actual_max < wavenumber_range[1] - 10:
            report['issues'].append(
                f"Wavenumber range [{actual_min:.1f}, {actual_max:.1f}] differs from "
                f"expected [{wavenumber_range[0]}, {wavenumber_range[1]}]"
            )
    except ValueError as e:
        report['issues'].append(f"Cannot parse wavenumber columns: {e}")
        report['valid'] = False

    # -------------------------------------------------------------------------
    # Missing values check
    # -------------------------------------------------------------------------
    intensity_df = df.select(wavenumber_cols)
    null_count = intensity_df.null_count().sum_horizontal()[0]
    
    # Check for inf values
    inf_count = 0
    for col in wavenumber_cols:
        col_data = df[col].to_numpy()
        inf_count += np.sum(~np.isfinite(col_data))
    
    report['missing_values'] = int(null_count) + int(inf_count)
    if report['missing_values'] > 0:
        report['issues'].append(f"Found {report['missing_values']} missing/infinite values")

    # -------------------------------------------------------------------------
    # Intensity range check
    # -------------------------------------------------------------------------
    out_of_range_samples = []
    samples = df['sample'].to_list()
    
    for i, row in enumerate(intensity_df.iter_rows()):
        intensities = np.array(row, dtype=float)
        valid_intensities = intensities[np.isfinite(intensities)]
        if len(valid_intensities) > 0:
            min_int, max_int = valid_intensities.min(), valid_intensities.max()
            if min_int < intensity_range[0] or max_int > intensity_range[1]:
                out_of_range_samples.append({
                    'sample': samples[i],
                    'min': float(min_int),
                    'max': float(max_int)
                })

    report['out_of_range'] = {
        'count': len(out_of_range_samples),
        'samples': out_of_range_samples[:10]  # Show first 10
    }
    if len(out_of_range_samples) > 0:
        report['issues'].append(
            f"{len(out_of_range_samples)} samples have intensities outside "
            f"[{intensity_range[0]}, {intensity_range[1]}]"
        )

    # -------------------------------------------------------------------------
    # Duplicate check
    # -------------------------------------------------------------------------
    sample_counts = df.group_by('sample').agg(pl.count().alias('count'))
    duplicates = sample_counts.filter(pl.col('count') > 1)['sample'].to_list()
    report['duplicates'] = duplicates
    if duplicates:
        report['issues'].append(f"Found {len(duplicates)} duplicate sample names")
        report['valid'] = False

    # -------------------------------------------------------------------------
    # Final validation status
    # -------------------------------------------------------------------------
    if report['missing_values'] > 0:
        report['valid'] = False

    # -------------------------------------------------------------------------
    # Print report
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print("FTIR DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Total samples:      {report['n_samples']}")
        print(f"Wavenumber points:  {report['n_wavenumbers']}")
        print(f"Classes:            {list(report['class_counts'].keys())}")
        print(f"Samples per class:  {report['class_counts']}")
        print(f"Missing values:     {report['missing_values']}")
        print(f"Out of range:       {report['out_of_range']['count']}")
        print(f"Duplicates:         {len(report['duplicates'])}")
        print("-" * 60)
        if report['issues']:
            print("ISSUES FOUND:")
            for issue in report['issues']:
                print(f"  ⚠ {issue}")
        else:
            print("✓ All validation checks passed")
        print("-" * 60)
        print(f"VALIDATION STATUS:  {'PASSED ✓' if report['valid'] else 'FAILED ✗'}")
        print("=" * 60)

    return report


def detect_outlier_spectra(
    df: pl.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect outlier spectra based on overall intensity statistics.

    Parameters
    ----------
    df : pl.DataFrame
        Wide-format spectral DataFrame.
    method : str, default "zscore"
        Detection method: 'zscore', 'iqr', or 'mad'.
    threshold : float, default 3.0
        Threshold for outlier detection.

    Returns
    -------
    dict
        - 'outlier_samples': list of sample names flagged as outliers
        - 'outlier_indices': list of row indices
        - 'statistics': dict with mean/std/median per sample
    """
    wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
    samples = df['sample'].to_list()

    # Compute statistics per spectrum
    stats = []
    for row in df.select(wavenumber_cols).iter_rows():
        arr = np.array(row, dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            stats.append({
                'mean': np.mean(valid),
                'std': np.std(valid),
                'median': np.median(valid),
                'range': np.ptp(valid)
            })
        else:
            stats.append({'mean': np.nan, 'std': np.nan, 'median': np.nan, 'range': np.nan})

    means = np.array([s['mean'] for s in stats])

    if method == "zscore":
        global_mean = np.nanmean(means)
        global_std = np.nanstd(means)
        z_scores = np.abs((means - global_mean) / global_std)
        outlier_mask = z_scores > threshold

    elif method == "iqr":
        q1, q3 = np.nanpercentile(means, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        outlier_mask = (means < lower) | (means > upper)

    elif method == "mad":
        median = np.nanmedian(means)
        mad = np.nanmedian(np.abs(means - median))
        outlier_mask = np.abs(means - median) / (mad * 1.4826) > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_samples = [samples[i] for i in outlier_indices]

    return {
        'outlier_samples': outlier_samples,
        'outlier_indices': outlier_indices,
        'n_outliers': len(outlier_samples),
        'method': method,
        'threshold': threshold,
        'statistics': stats
    }


def check_wavenumber_consistency(
    file_paths: List[str],
    skiprows: int = 15,
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Check if all files have consistent wavenumber grids.

    Parameters
    ----------
    file_paths : list of str
        Paths to CSV spectral files.
    skiprows : int, default 15
        Header rows to skip.
    tolerance : float, default 0.1
        Maximum allowed difference in wavenumber values.

    Returns
    -------
    dict
        - 'consistent': bool
        - 'reference_shape': tuple
        - 'mismatched_files': list
    """
    reference_wn = None
    reference_shape = None
    mismatched = []

    for fp in file_paths[:min(50, len(file_paths))]:  # Check first 50
        try:
            data = pd.read_csv(fp, skiprows=skiprows, header=None)
            wn = data.iloc[:, 0].values

            if reference_wn is None:
                reference_wn = wn
                reference_shape = wn.shape
            else:
                if wn.shape != reference_shape:
                    mismatched.append({'file': fp, 'issue': 'shape_mismatch'})
                elif np.max(np.abs(wn - reference_wn)) > tolerance:
                    mismatched.append({'file': fp, 'issue': 'wavenumber_mismatch'})

        except Exception as e:
            mismatched.append({'file': fp, 'issue': str(e)})

    return {
        'consistent': len(mismatched) == 0,
        'reference_shape': reference_shape,
        'files_checked': min(50, len(file_paths)),
        'mismatched_files': mismatched
    }
