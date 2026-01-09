# Import libraries
from __future__ import annotations
from typing import Union, List, Optional
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import false_discovery_control
import warnings
warnings.filterwarnings('ignore')

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)

# Statistical analysis
from scipy.stats import f_oneway

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def perform_anova_analysis(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        p_threshold: float = 0.05,
        correction: str = 'fdr',
        figsize: tuple = (14, 10),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform one-way ANOVA for each wavenumber to find discriminative features.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input spectral data with samples as rows and wavenumbers as columns.
    label_column : str, default "label"
        Column name containing class labels.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include in analysis.
    wn_max : float, optional
        Maximum wavenumber to include in analysis.
    dataset_name : str, default "c8"
        Dataset identifier for title.
    p_threshold : float, default 0.05
        Significance threshold for p-values (applied to corrected p-values if correction is used).
    correction : str, default 'fdr'
        Multiple testing correction method:
        - 'none': No correction (use raw p-values)
        - 'bonferroni': Bonferroni correction (most conservative)
        - 'fdr': False Discovery Rate (Benjamini-Hochberg method, recommended)
    figsize : tuple, default (14, 10)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved.

    Returns
    -------
    dict
        Dictionary containing:
        - 'wavenumbers': Array of wavenumber values
        - 'f_statistics': F-statistic for each wavenumber
        - 'p_values': Raw p-values
        - 'p_values_corrected': Corrected p-values (same as p_values if correction='none')
        - 'significant': Boolean array indicating significance after correction

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Multiple testing correction helps control false positives when testing many features
    • FDR (Benjamini-Hochberg) is recommended as it balances sensitivity and specificity
    """
    # Convert to pandas for processing
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    else:
        df = data.copy()

    # Build exclusion list
    exclude = [label_column] if label_column in df.columns else []
    if "sample" in df.columns:
        exclude.append("sample")
    if exclude_columns:
        exclude.extend(exclude_columns)

    # Infer and sort spectral columns using shared utilities
    spectral_cols, wn_values = _infer_spectral_columns(
        df,
        exclude_columns=exclude,
        wn_min=wn_min,
        wn_max=wn_max
    )

    spectral_cols_sorted, wn_sorted, _ = _sort_spectral_columns(
        spectral_cols, wn_values
    )

    # Get unique labels
    unique_labels = sorted(df[label_column].unique())

    # Prepare data for ANOVA
    f_statistics = []
    p_values = []

    for wn_col in spectral_cols_sorted:
        # Group data by label
        groups = [df[df[label_column] == label][wn_col].astype(float).values
                 for label in unique_labels]

        # Perform ANOVA
        f_stat, p_val = f_oneway(*groups)
        f_statistics.append(f_stat)
        p_values.append(p_val)

    f_statistics = np.array(f_statistics)
    p_values = np.array(p_values)

    # Handle NaN or invalid p-values
    # Replace NaN with 1.0 (not significant) and clip to valid range [0, 1]
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=0.0)
    p_values = np.clip(p_values, 0.0, 1.0)

    # Apply multiple testing correction
    if correction.lower() == 'none':
        p_values_corrected = p_values.copy()
        correction_label = "No correction"
    elif correction.lower() == 'bonferroni':
        p_values_corrected = np.minimum(p_values * len(p_values), 1.0)
        correction_label = "Bonferroni"
    elif correction.lower() == 'fdr':
        p_values_corrected = false_discovery_control(p_values, method='bh')
        correction_label = "FDR (BH)"
    else:
        raise ValueError(f"Invalid correction method: {correction}. Must be 'none', 'bonferroni', or 'fdr'.")

    # Determine significance based on corrected p-values
    significant = p_values_corrected < p_threshold

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # F-statistics
    axes[0].plot(wn_sorted, f_statistics, linewidth=1.5, color='blue')
    axes[0].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[0].set_ylabel('F-statistic', fontsize=12)
    axes[0].set_title(f'ANOVA F-statistics - {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    # P-values (log scale) - plot both raw and corrected if correction is applied
    if correction.lower() == 'none':
        axes[1].plot(wn_sorted, -np.log10(p_values), linewidth=1.5, color='red', label='p-value')
    else:
        # Plot both raw and corrected p-values for comparison
        axes[1].plot(wn_sorted, -np.log10(p_values), linewidth=1.2, color='lightcoral',
                    alpha=0.6, label='Raw p-value')
        axes[1].plot(wn_sorted, -np.log10(p_values_corrected), linewidth=1.5, color='red',
                    label=f'Corrected p-value ({correction_label})')

    axes[1].axhline(y=-np.log10(p_threshold), color='black', linestyle='--',
                   label=f'p={p_threshold} threshold')
    axes[1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[1].set_ylabel('-log₁₀(p-value)', fontsize=12)

    title_suffix = f' ({correction_label})' if correction.lower() != 'none' else ''
    axes[1].set_title(f'ANOVA p-values (log scale) - {dataset_name}{title_suffix}',
                     fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"anova_analysis_{dataset_name}.pdf"
        else:
            file_path = f"anova_analysis_{dataset_name}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_path}")

    plt.show()

    # Print summary
    significant_features_raw = np.sum(p_values < p_threshold)
    significant_features_corrected = np.sum(significant)

    print(f"\nANOVA Summary for {dataset_name}:")
    print(f"  Total wavenumbers: {len(wn_sorted)}")
    print(f"  Mean F-statistic: {np.mean(f_statistics):.2f}")
    print(f"  Max F-statistic: {np.max(f_statistics):.2f} at {wn_sorted[np.argmax(f_statistics)]:.1f} cm⁻¹")
    print(f"\n  Multiple testing correction: {correction_label}")
    if correction.lower() != 'none':
        print(f"  Significant features (raw p < {p_threshold}): {significant_features_raw} ({significant_features_raw/len(wn_sorted)*100:.1f}%)")
        print(f"  Significant features (corrected p < {p_threshold}): {significant_features_corrected} ({significant_features_corrected/len(wn_sorted)*100:.1f}%)")
    else:
        print(f"  Significant features (p < {p_threshold}): {significant_features_raw} ({significant_features_raw/len(wn_sorted)*100:.1f}%)")

    return {
        'wavenumbers': wn_sorted,
        'f_statistics': f_statistics,
        'p_values': p_values,
        'p_values_corrected': p_values_corrected,
        'significant': significant
    }
    
    

def plot_correlation_matrix(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        sample_size: int = 100,
        figsize: tuple = (12, 10),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Plot correlation matrix of spectral features (using subset for visualization).

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input spectral data with samples as rows and wavenumbers as columns.
    label_column : str, default "label"
        Column name containing class labels.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include in analysis.
    wn_max : float, optional
        Maximum wavenumber to include in analysis.
    dataset_name : str, default "c8"
        Dataset identifier for title.
    sample_size : int, default 100
        Target number of wavenumbers to sample for correlation matrix.
    figsize : tuple, default (12, 10)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Samples wavenumbers for computational efficiency
    """
    # Convert to pandas for processing
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    else:
        df = data.copy()

    # Build exclusion list
    exclude = [label_column] if label_column in df.columns else []
    if "sample" in df.columns:
        exclude.append("sample")
    if exclude_columns:
        exclude.extend(exclude_columns)

    # Infer and sort spectral columns using shared utilities
    spectral_cols, wn_values = _infer_spectral_columns(
        df,
        exclude_columns=exclude,
        wn_min=wn_min,
        wn_max=wn_max
    )

    spectral_cols_sorted, wn_sorted, _ = _sort_spectral_columns(
        spectral_cols, wn_values
    )

    # Sample wavenumbers for visualization (every nth wavenumber)
    step = max(1, len(spectral_cols_sorted) // sample_size)
    sampled_wn_cols = spectral_cols_sorted[::step]

    # Calculate correlation matrix
    corr_matrix = df[sampled_wn_cols].astype(float).corr()

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
               square=True, linewidths=0, cbar_kws={"shrink": 0.8})
    plt.title(f'Spectral Feature Correlation Matrix - {dataset_name}\n(Every {step}th wavenumber)',
             fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"correlation_matrix_{dataset_name}.pdf"
        else:
            file_path = f"correlation_matrix_{dataset_name}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_path}")

    plt.show()

    print(f"\nCorrelation Summary for {dataset_name}:")
    print(f"  Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    print(f"  Max correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
    print(f"  Min correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f}")
