# Import libraries
from __future__ import annotations
from typing import Union, List, Optional
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def _canonical_mode(mode: str) -> str:
    """Normalize mode string to canonical form."""
    m = (mode or "auto").lower().strip()
    if m in {"auto", "a"}:
        return "auto"
    if m in {"transmittance", "trans", "t"}:
        return "transmittance"
    if m in {"absorbance", "abs", "a"}:
        return "absorbance"
    raise ValueError(f"Invalid mode='{mode}'. Use 'auto'/'a', 'transmittance'/'trans'/'t', or 'absorbance'/'abs'/'a'.")


def plot_mean_spectra_by_class(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        title: Optional[str] = "Mean Spectra by Type",
        dataset_name: Optional[str] = "c8",
        figsize: tuple = (16, 12),
        mode: str = "auto",
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Plot mean spectra for each plastic type with confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input spectral data with samples as rows and wavenumbers as columns.
    label_column : str, default "label"
        Column name containing class labels.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include in plot.
    wn_max : float, optional
        Maximum wavenumber to include in plot.
    title : str, default "Mean Spectra by Type"
        Plot title.
    dataset_name : str, optional, default "c8"
        Dataset identifier for title.
    figsize : tuple, default (16, 12)
        Figure size in inches (width, height).
    mode : str, default "auto"
        Plot mode:
        - "transmittance": Display as transmittance (%). Converts if data is absorbance.
        - "absorbance": Display as absorbance (AU). Converts if data is transmittance.
        - "auto": Auto-detect data type (threshold: 10.0) and plot as-is without conversion.
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved. If None, saves in current directory.
        The directory will be created automatically if it doesn't exist.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Automatically handles both 'pd' and 'pl' DataFrame formats
    • Supports automatic transmittance/absorbance detection and conversion
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

    # Detect data type using robust statistics
    if len(df) > 0 and len(spectral_cols_sorted) > 0:
        sample_size = min(100, len(df))
        sample_data = df[spectral_cols_sorted].iloc[:sample_size].values.flatten()
        sample_data = sample_data[np.isfinite(sample_data)]

        if len(sample_data) > 0:
            median_val = np.median(sample_data)
            p95_val = np.percentile(sample_data, 95)
            data_type = "transmittance" if (p95_val > 10.0 and median_val > 1.0) else "absorbance"
        else:
            data_type = "transmittance"
    else:
        data_type = "transmittance"

    # Determine display mode and whether conversion is needed
    mode_c = _canonical_mode(mode)
    if mode_c == "auto":
        display_mode = data_type
        needs_conversion = False
        print(f"Auto-detected: {data_type.capitalize()}")
    else:
        display_mode = mode_c
        needs_conversion = (display_mode != data_type)

    # Get unique labels
    unique_labels = sorted(df[label_column].unique())
    plot_rows = len(unique_labels)//2
    # Create figure
    fig, axes = plt.subplots(plot_rows, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, label in enumerate(unique_labels):
        # Filter data for this label
        label_data = df[df[label_column] == label]

        # Extract spectra matrix
        spectra_matrix = label_data[spectral_cols_sorted].astype(float).values

        # Convert if needed
        if needs_conversion:
            if display_mode == "absorbance" and data_type == "transmittance":
                # Convert transmittance to absorbance: A = -log10(T/100)
                eps = 1e-12
                invalid_mask = spectra_matrix <= 0
                spectra_matrix_clipped = np.where(spectra_matrix > eps, spectra_matrix, eps)
                spectra_matrix = -np.log10(spectra_matrix_clipped / 100)
                spectra_matrix[invalid_mask] = np.nan
            elif display_mode == "transmittance" and data_type == "absorbance":
                # Convert absorbance to transmittance: T = 100 * 10^(-A)
                spectra_matrix = 100.0 * np.power(10, -spectra_matrix)

        # Calculate mean and std
        mean_spectrum = np.mean(spectra_matrix, axis=0)
        std_spectrum = np.std(spectra_matrix, axis=0)

        # Plot
        axes[idx].plot(wn_sorted, mean_spectrum, linewidth=2, label=f'{label} (n={len(label_data)})')
        axes[idx].fill_between(wn_sorted,
                               mean_spectrum - std_spectrum,
                               mean_spectrum + std_spectrum,
                               alpha=0.3)

        axes[idx].set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        ylabel = "Absorbance (AU)" if display_mode == "absorbance" else "Transmittance (%)"
        axes[idx].set_ylabel(ylabel, fontsize=10)
        axes[idx].set_title(f'{label}', fontsize=12, fontweight='bold')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].invert_xaxis()

    plt.suptitle(f'{title} - {dataset_name}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_plot:
        # Create save directory if it doesn't exist
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"mean_spectra_by_class_{dataset_name}.pdf"
        else:
            file_path = f"mean_spectra_by_class_{dataset_name}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_path}")

    plt.show()


def plot_overlay_mean_spectra(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        title: str = "Overlay of Mean Spectra",
        dataset_name: Optional[str] = "c8",
        figsize: tuple = (14, 6),
        mode: str = "auto",
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Plot all mean spectra overlaid on one plot.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input spectral data with samples as rows and wavenumbers as columns.
    label_column : str, default "label"
        Column name containing class labels.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include in plot.
    wn_max : float, optional
        Maximum wavenumber to include in plot.
    title : str, default "Overlay of Mean Spectra"
        Plot title.
    dataset_name : str, optional, default "c8"
        Dataset identifier for title.
    figsize : tuple, default (14, 6)
        Figure size in inches (width, height).
    mode : str, default "auto"
        Plot mode:
        - "transmittance": Display as transmittance (%). Converts if data is absorbance.
        - "absorbance": Display as absorbance (AU). Converts if data is transmittance.
        - "auto": Auto-detect data type (threshold: 10.0) and plot as-is without conversion.
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved. If None, saves in current directory.
        The directory will be created automatically if it doesn't exist.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Automatically handles both 'pd' and 'pl' DataFrame formats
    • Supports automatic transmittance/absorbance detection and conversion
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

    # Detect data type using robust statistics
    if len(df) > 0 and len(spectral_cols_sorted) > 0:
        sample_size = min(100, len(df))
        sample_data = df[spectral_cols_sorted].iloc[:sample_size].values.flatten()
        sample_data = sample_data[np.isfinite(sample_data)]

        if len(sample_data) > 0:
            median_val = np.median(sample_data)
            p95_val = np.percentile(sample_data, 95)
            data_type = "transmittance" if (p95_val > 10.0 and median_val > 1.0) else "absorbance"
        else:
            data_type = "transmittance"
    else:
        data_type = "transmittance"

    # Determine display mode and whether conversion is needed
    mode_c = _canonical_mode(mode)
    if mode_c == "auto":
        display_mode = data_type
        needs_conversion = False
        print(f"Auto-detected: {data_type.capitalize()}")
    else:
        display_mode = mode_c
        needs_conversion = (display_mode != data_type)

    # Get unique labels
    unique_labels = sorted(df[label_column].unique())

    # Create figure
    plt.figure(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        # Filter data for this label
        label_data = df[df[label_column] == label]

        # Extract spectra matrix
        spectra_matrix = label_data[spectral_cols_sorted].astype(float).values

        # Convert if needed
        if needs_conversion:
            if display_mode == "absorbance" and data_type == "transmittance":
                # Convert transmittance to absorbance: A = -log10(T/100)
                eps = 1e-12
                invalid_mask = spectra_matrix <= 0
                spectra_matrix_clipped = np.where(spectra_matrix > eps, spectra_matrix, eps)
                spectra_matrix = -np.log10(spectra_matrix_clipped / 100)
                spectra_matrix[invalid_mask] = np.nan
            elif display_mode == "transmittance" and data_type == "absorbance":
                # Convert absorbance to transmittance: T = 100 * 10^(-A)
                spectra_matrix = 100.0 * np.power(10, -spectra_matrix)

        # Calculate mean
        mean_spectrum = np.mean(spectra_matrix, axis=0)

        # Plot
        plt.plot(wn_sorted, mean_spectrum, linewidth=2,
                label=f'{label} (n={len(label_data)})', color=colors[idx])

    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ylabel = "Absorbance (AU)" if display_mode == "absorbance" else "Transmittance (%)"
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title} - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()

    if save_plot:
        # Create save directory if it doesn't exist
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"overlay_mean_spectra_{dataset_name}.pdf"
        else:
            file_path = f"overlay_mean_spectra_{dataset_name}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_path}")

    plt.show()
    


def plot_coefficient_of_variation(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        title: str = "Spectral Variability by Type",
        dataset_name: Optional[str] = "c8",
        figsize: tuple = (14, 6),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Plot coefficient of variation (CV) across wavenumbers for each class.

    CV = std / mean, shows relative variability.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input spectral data with samples as rows and wavenumbers as columns.
    label_column : str, default "label"
        Column name containing class labels.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include in plot.
    wn_max : float, optional
        Maximum wavenumber to include in plot.
    dataset_name : str, optional, default "c8"
        Dataset identifier for title.
    figsize : tuple, default (14, 6)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved. If None, saves in current directory.
        The directory will be created automatically if it doesn't exist.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Automatically handles both 'pd' and 'pl' DataFrame formats
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

    # Create figure
    plt.figure(figsize=figsize)

    for label in unique_labels:
        # Filter data for this label
        label_data = df[df[label_column] == label]

        # Extract spectra matrix
        spectra_matrix = label_data[spectral_cols_sorted].astype(float).values

        # Calculate CV
        mean_spectrum = np.mean(spectra_matrix, axis=0)
        std_spectrum = np.std(spectra_matrix, axis=0)
        cv = np.divide(std_spectrum, mean_spectrum,
                      out=np.zeros_like(std_spectrum),
                      where=mean_spectrum!=0) * 100

        # Plot
        plt.plot(wn_sorted, cv, linewidth=1.5, label=label, alpha=0.8)

    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    plt.ylabel('Coefficient of Variation (%)', fontsize=12)
    plt.title(f'{title} - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()

    if save_plot:
        # Create save directory if it doesn't exist
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"coefficient_of_variation_{dataset_name}.pdf"
        else:
            file_path = f"coefficient_of_variation_{dataset_name}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_path}")

    plt.show()


def plot_spectral_heatmap(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: Optional[str] = "c8",
        n_samples: int = 50,
        figsize: tuple = (16, 10),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Plot heatmap of spectra (random subset for visualization).

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input spectral data with samples as rows and wavenumbers as columns.
    label_column : str, default "label"
        Column name containing class labels.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include in plot.
    wn_max : float, optional
        Maximum wavenumber to include in plot.
    dataset_name : str, optional, default "c8"
        Dataset identifier for title.
    n_samples : int, default 50
        Maximum number of samples per class to include in heatmap.
    figsize : tuple, default (16, 10)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved. If None, saves in current directory.
        The directory will be created automatically if it doesn't exist.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Automatically handles both 'pd' and 'pl' DataFrame formats
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

    # Sample random spectra from each class
    unique_labels = sorted(df[label_column].unique())
    sampled_data = []
    labels_list = []

    for label in unique_labels:
        label_data = df[df[label_column] == label]
        # Sample n_samples from each class
        if len(label_data) > n_samples:
            label_data = label_data.sample(n=n_samples, random_state=42)

        spectra_matrix = label_data[spectral_cols_sorted].astype(float).values
        sampled_data.append(spectra_matrix)
        labels_list.extend([label] * len(label_data))

    # Concatenate all sampled data
    all_spectra = np.vstack(sampled_data)

    # Create heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(all_spectra, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Absorbance')

    # Set labels
    plt.xlabel('Wavenumber Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.title(f'Spectral Heatmap ({n_samples} samples per class) - {dataset_name}',
             fontsize=14, fontweight='bold')

    # Add horizontal lines to separate classes
    y_positions = np.cumsum([len(sd) for sd in sampled_data])[:-1]
    for y_pos in y_positions:
        plt.axhline(y=y_pos - 0.5, color='red', linewidth=2, linestyle='--')

    # Add class labels on the right
    y_centers = np.concatenate([[0], y_positions]) + np.diff(np.concatenate([[0], y_positions, [len(all_spectra)]]))/2
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_centers)
    ax2.set_yticklabels(unique_labels)

    plt.tight_layout()

    if save_plot:
        # Create save directory if it doesn't exist
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"spectral_heatmap_{dataset_name}.pdf"
        else:
            file_path = f"spectral_heatmap_{dataset_name}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_path}")

    plt.show()



