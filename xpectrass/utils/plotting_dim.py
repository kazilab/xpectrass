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

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
# UMAP is imported inside perform_umap_analysis() to avoid import-time warnings

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

def perform_pca_analysis(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        n_components: int = 10,
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (16, 5),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform PCA and visualize results.

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
    n_components : int, default 10
        Number of principal components to compute.
    standardize : bool, default True
        If True, standardize features using StandardScaler (zero mean, unit variance).
        Set to False if data is already standardized/normalized.
    handle_missing : str, default "zero"
        How to handle missing values (NaN):
        - "drop": Remove samples with any NaN values
        - "mean": Impute NaN with column mean
        - "zero": Replace NaN with 0
        - "raise": Raise an error if NaN values are found
    figsize : tuple, default (16, 5)
        Figure size in inches (width, height) for variance plots.
    save_plot : bool, default False
        If True, save the plots to disk as PDF.
    save_path : str, optional
        Directory path where plots will be saved.

    Returns
    -------
    dict
        Dictionary containing 'pca', 'X_pca', 'scaler', and 'wavenumbers'.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Handles missing values and provides optional standardization
    • PCA is sensitive to scale, so standardization is recommended unless data is already normalized
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

    # Convert to numpy
    X = df[spectral_cols_sorted].astype(float).values
    y = df[label_column].values

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            # Remove samples with any NaN
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask]
            n_dropped = n_samples_before - X.shape[0]
            print(f"  Dropped {n_dropped} samples with NaN values ({n_samples_before} -> {X.shape[0]} samples)")
        elif handle_missing == "mean":
            # Impute with column mean
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            print(f"  Imputed NaN values with column means")
        elif handle_missing == "zero":
            # Replace with zero
            X = np.nan_to_num(X, nan=0.0)
            print(f"  Replaced NaN values with 0")
        elif handle_missing == "raise":
            raise ValueError(f"Found {n_missing} NaN values in data. Use handle_missing='drop', 'mean', or 'zero' to handle them.")
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}. Must be 'drop', 'mean', 'zero', or 'raise'.")

    # Standardize (optional)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Data standardized (mean=0, std=1)")
    else:
        X_scaled = X.copy()
        scaler = None
        print(f"Using data without standardization (assumes already normalized)")

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scree plot
    axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_ * 100)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance (%)', fontsize=12)
    axes[0].set_title(f'PCA Scree Plot - {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_ * 100)
    axes[1].plot(range(1, n_components+1), cumsum_var, marker='o', linewidth=2)
    axes[1].axhline(y=95, color='r', linestyle='--', label='95% variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    axes[1].set_title(f'Cumulative Explained Variance - {dataset_name}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"pca_variance_{dataset_name}.pdf"
        else:
            file_path = f"pca_variance_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Variance plot saved to: {file_path}")

    plt.show()

    # Print summary
    print(f"\nPCA Summary for {dataset_name}:")
    print(f"  PC1 explains: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  PC2 explains: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"  First 2 PCs explain: {sum(pca.explained_variance_ratio_[:2])*100:.2f}%")
    print(f"  First 3 PCs explain: {sum(pca.explained_variance_ratio_[:3])*100:.2f}%")
    n_for_95 = np.argmax(cumsum_var >= 95) + 1
    print(f"  Components needed for 95% variance: {n_for_95}")

    # Plot 2D scatter
    fig = plt.figure(figsize=(12, 8))
    unique_labels = np.unique(y)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[idx]], label=label, alpha=0.6, s=50)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    plt.title(f'PCA Scatter Plot - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"pca_scatter_{dataset_name}.pdf"
        else:
            file_path = f"pca_scatter_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {file_path}")

    plt.show()

    # Plot 3D scatter
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=[colors[idx]], label=label, alpha=0.6, s=50)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)', fontsize=12)
    ax.set_title(f'PCA 3D Scatter Plot - {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"pca_scatter_3d_{dataset_name}.pdf"
        else:
            file_path = f"pca_scatter_3d_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"3D scatter plot saved to: {file_path}")

    plt.show()

    # Plot loadings for PC1 and PC2
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(wn_sorted, pca.components_[0], linewidth=1.5, color='blue')
    axes[0].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[0].set_ylabel('Loading', fontsize=12)
    axes[0].set_title(f'PC1 Loadings ({pca.explained_variance_ratio_[0]*100:.2f}% variance) - {dataset_name}',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    axes[1].plot(wn_sorted, pca.components_[1], linewidth=1.5, color='red')
    axes[1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[1].set_ylabel('Loading', fontsize=12)
    axes[1].set_title(f'PC2 Loadings ({pca.explained_variance_ratio_[1]*100:.2f}% variance) - {dataset_name}',
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"pca_loadings_{dataset_name}.pdf"
        else:
            file_path = f"pca_loadings_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Loadings plot saved to: {file_path}")

    plt.show()

    return {'pca': pca, 'X_pca': X_pca, 'scaler': scaler, 'wavenumbers': wn_sorted}


def perform_tsne_analysis(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        perplexity: int = 30,
        n_iter: int = 1000,
        pca_components: int = 50,
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (12, 8),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform t-SNE dimensionality reduction.

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
    perplexity : int, default 30
        t-SNE perplexity parameter (5-50 typical range).
    n_iter : int, default 1000
        Number of iterations for optimization.
    pca_components : int, default 50
        Number of PCA components to use for pre-processing.
    standardize : bool, default True
        If True, standardize features using StandardScaler (zero mean, unit variance).
        Set to False if data is already standardized/normalized.
    handle_missing : str, default "zero"
        How to handle missing values (NaN):
        - "drop": Remove samples with any NaN values
        - "mean": Impute NaN with column mean
        - "zero": Replace NaN with 0
        - "raise": Raise an error if NaN values are found
    figsize : tuple, default (12, 8)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved.

    Returns
    -------
    dict
        Dictionary containing 'tsne' and 'X_tsne'.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Applies PCA pre-processing for denoising
    • Handles missing values and provides optional standardization
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

    # Convert to numpy
    X = df[spectral_cols_sorted].astype(float).values
    y = df[label_column].values

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask]
            n_dropped = n_samples_before - X.shape[0]
            print(f"  Dropped {n_dropped} samples with NaN values ({n_samples_before} -> {X.shape[0]} samples)")
        elif handle_missing == "mean":
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            print(f"  Imputed NaN values with column means")
        elif handle_missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
            print(f"  Replaced NaN values with 0")
        elif handle_missing == "raise":
            raise ValueError(f"Found {n_missing} NaN values in data. Use handle_missing='drop', 'mean', or 'zero' to handle them.")
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}. Must be 'drop', 'mean', 'zero', or 'raise'.")

    # Standardize (optional)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Data standardized (mean=0, std=1)")
    else:
        X_scaled = X.copy()
        print(f"Using data without standardization (assumes already normalized)")

    # Apply PCA first for denoising (common practice)
    pca = PCA(n_components=min(pca_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"Performing t-SNE for {dataset_name}...")
    print(f"  Using {X_pca.shape[1]} PCA components as input")
    print(f"  Perplexity: {perplexity}")
    print(f"  Iterations: {n_iter}")

    # Perform t-SNE (using max_iter for newer scikit-learn versions)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
               random_state=42, verbose=1)
    X_tsne = tsne.fit_transform(X_pca)

    # Plot
    fig = plt.figure(figsize=figsize)
    unique_labels = np.unique(y)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors[idx]], label=label, alpha=0.6, s=50)

    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title(f't-SNE Visualization - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"tsne_{dataset_name}.pdf"
        else:
            file_path = f"tsne_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to: {file_path}")

    plt.show()

    return {'tsne': tsne, 'X_tsne': X_tsne}


def perform_umap_analysis(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        pca_components: int = 50,
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (12, 8),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform UMAP dimensionality reduction.

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
    n_neighbors : int, default 15
        UMAP n_neighbors parameter (controls local vs global structure).
    min_dist : float, default 0.1
        UMAP min_dist parameter (controls cluster tightness).
    pca_components : int, default 50
        Number of PCA components to use for pre-processing.
    standardize : bool, default True
        If True, standardize features using StandardScaler (zero mean, unit variance).
        Set to False if data is already standardized/normalized.
    handle_missing : str, default "zero"
        How to handle missing values (NaN):
        - "drop": Remove samples with any NaN values
        - "mean": Impute NaN with column mean
        - "zero": Replace NaN with 0
        - "raise": Raise an error if NaN values are found
    figsize : tuple, default (12, 8)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved.

    Returns
    -------
    dict or None
        Dictionary containing 'umap' and 'X_umap', or None if UMAP not installed.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Applies PCA pre-processing for denoising and dimensionality reduction
    • Handles missing values and provides optional standardization
    • Requires umap-learn package: pip install umap-learn
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Please run: pip install umap-learn")
        return None

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

    # Convert to numpy
    X = df[spectral_cols_sorted].astype(float).values
    y = df[label_column].values

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask]
            n_dropped = n_samples_before - X.shape[0]
            print(f"  Dropped {n_dropped} samples with NaN values ({n_samples_before} -> {X.shape[0]} samples)")
        elif handle_missing == "mean":
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            print(f"  Imputed NaN values with column means")
        elif handle_missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
            print(f"  Replaced NaN values with 0")
        elif handle_missing == "raise":
            raise ValueError(f"Found {n_missing} NaN values in data. Use handle_missing='drop', 'mean', or 'zero' to handle them.")
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}. Must be 'drop', 'mean', 'zero', or 'raise'.")

    # Standardize (optional)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Data standardized (mean=0, std=1)")
    else:
        X_scaled = X.copy()
        print(f"Using data without standardization (assumes already normalized)")

    # Apply PCA first for denoising (common practice)
    pca = PCA(n_components=min(pca_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"Performing UMAP for {dataset_name}...")
    print(f"  Using {X_pca.shape[1]} PCA components as input")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")

    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                       n_components=2, random_state=42, verbose=True, n_jobs=-1)
    X_umap = reducer.fit_transform(X_pca)

    # Plot
    fig = plt.figure(figsize=figsize)
    unique_labels = np.unique(y)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=[colors[idx]], label=label, alpha=0.6, s=50)

    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.title(f'UMAP Visualization - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"umap_{dataset_name}.pdf"
        else:
            file_path = f"umap_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"UMAP plot saved to: {file_path}")

    plt.show()

    return {'umap': reducer, 'X_umap': X_umap}


def perform_plsda_analysis(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        n_components: int = 10,
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (12, 8),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform PLS-DA (Partial Least Squares Discriminant Analysis) for multiclass classification.

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
    n_components : int, default 10
        Number of PLS components to compute.
    standardize : bool, default True
        If True, standardize features using StandardScaler (zero mean, unit variance).
        Set to False if data is already standardized/normalized.
    handle_missing : str, default "zero"
        How to handle missing values (NaN):
        - "drop": Remove samples with any NaN values
        - "mean": Impute NaN with column mean
        - "zero": Replace NaN with 0
        - "raise": Raise an error if NaN values are found
    figsize : tuple, default (12, 8)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plots to disk as PDF.
    save_path : str, optional
        Directory path where plots will be saved.

    Returns
    -------
    dict
        Dictionary containing 'pls', 'X_pls', 'Y_encoded', 'label_encoder', 'scaler', and 'wavenumbers'.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Handles multiclass via one-hot encoding of labels
    • Handles missing values and provides optional standardization
    • PLS-DA maximizes covariance between X (spectra) and Y (class labels)
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

    # Convert to numpy
    X = df[spectral_cols_sorted].astype(float).values
    y = df[label_column].values

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask]
            n_dropped = n_samples_before - X.shape[0]
            print(f"  Dropped {n_dropped} samples with NaN values ({n_samples_before} -> {X.shape[0]} samples)")
        elif handle_missing == "mean":
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            print(f"  Imputed NaN values with column means")
        elif handle_missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
            print(f"  Replaced NaN values with 0")
        elif handle_missing == "raise":
            raise ValueError(f"Found {n_missing} NaN values in data. Use handle_missing='drop', 'mean', or 'zero' to handle them.")
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}. Must be 'drop', 'mean', 'zero', or 'raise'.")

    # Encode labels for multiclass (one-hot encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Create one-hot encoded Y matrix
    Y = np.zeros((len(y_encoded), n_classes))
    Y[np.arange(len(y_encoded)), y_encoded] = 1

    # Standardize X (optional)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Data standardized (mean=0, std=1)")
    else:
        X_scaled = X.copy()
        scaler = None
        print(f"Using data without standardization (assumes already normalized)")

    # Perform PLS-DA
    print(f"Performing PLS-DA for {dataset_name}...")
    print(f"  Number of classes: {n_classes}")
    print(f"  Number of components: {n_components}")

    pls = PLSRegression(n_components=min(n_components, X_scaled.shape[1], n_classes))
    X_pls = pls.fit_transform(X_scaled, Y)[0]

    # Calculate R2X and R2Y
    X_reconstructed = pls.inverse_transform(X_pls)
    r2x = 1 - np.sum((X_scaled - X_reconstructed) ** 2) / np.sum(X_scaled ** 2)

    Y_pred = pls.predict(X_scaled)
    r2y = 1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)

    print(f"  R²X (variance explained in X): {r2x:.4f}")
    print(f"  R²Y (variance explained in Y): {r2y:.4f}")

    # Plot 2D scores plot
    fig = plt.figure(figsize=figsize)
    unique_labels = le.classes_
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_pls[mask, 0], X_pls[mask, 1],
                   c=[colors[idx]], label=label, alpha=0.6, s=50)

    plt.xlabel(f'LV1 (Latent Variable 1)', fontsize=12)
    plt.ylabel(f'LV2 (Latent Variable 2)', fontsize=12)
    plt.title(f'PLS-DA Scores Plot - {dataset_name}\nR²X={r2x:.3f}, R²Y={r2y:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"plsda_scores_{dataset_name}.pdf"
        else:
            file_path = f"plsda_scores_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Scores plot saved to: {file_path}")

    plt.show()

    # Plot 3D scores if we have at least 3 components
    if X_pls.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for idx, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(X_pls[mask, 0], X_pls[mask, 1], X_pls[mask, 2],
                       c=[colors[idx]], label=label, alpha=0.6, s=50)

        ax.set_xlabel('LV1', fontsize=12)
        ax.set_ylabel('LV2', fontsize=12)
        ax.set_zlabel('LV3', fontsize=12)
        ax.set_title(f'PLS-DA 3D Scores Plot - {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f"plsda_scores_3d_{dataset_name}.pdf"
            else:
                file_path = f"plsda_scores_3d_{dataset_name}.pdf"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"3D scores plot saved to: {file_path}")

        plt.show()

    # Plot loadings for LV1 and LV2
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(wn_sorted, pls.x_loadings_[:, 0], linewidth=1.5, color='blue')
    axes[0].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[0].set_ylabel('Loading', fontsize=12)
    axes[0].set_title(f'LV1 Loadings - {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    axes[1].plot(wn_sorted, pls.x_loadings_[:, 1], linewidth=1.5, color='red')
    axes[1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[1].set_ylabel('Loading', fontsize=12)
    axes[1].set_title(f'LV2 Loadings - {dataset_name}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"plsda_loadings_{dataset_name}.pdf"
        else:
            file_path = f"plsda_loadings_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Loadings plot saved to: {file_path}")

    plt.show()

    return {
        'pls': pls,
        'X_pls': X_pls,
        'Y_encoded': Y,
        'label_encoder': le,
        'scaler': scaler,
        'wavenumbers': wn_sorted,
        'r2x': r2x,
        'r2y': r2y
    }


def perform_oplsda_analysis(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        n_components: int = 1,
        n_orthogonal: int = 2,
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (12, 8),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform OPLS-DA (Orthogonal Partial Least Squares Discriminant Analysis) for multiclass classification.

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
    n_components : int, default 1
        Number of predictive components (typically 1 for OPLS-DA).
    n_orthogonal : int, default 2
        Number of orthogonal components to remove.
    standardize : bool, default True
        If True, standardize features using StandardScaler (zero mean, unit variance).
        Set to False if data is already standardized/normalized.
    handle_missing : str, default "zero"
        How to handle missing values (NaN):
        - "drop": Remove samples with any NaN values
        - "mean": Impute NaN with column mean
        - "zero": Replace NaN with 0
        - "raise": Raise an error if NaN values are found
    figsize : tuple, default (12, 8)
        Figure size in inches (width, height).
    save_plot : bool, default False
        If True, save the plots to disk as PDF.
    save_path : str, optional
        Directory path where plots will be saved.

    Returns
    -------
    dict
        Dictionary containing OPLS-DA results including scores, loadings, and model metrics.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Handles multiclass via one-hot encoding
    • Handles missing values and provides optional standardization
    • OPLS-DA separates predictive and orthogonal (non-predictive) variation
    • Implementation uses iterative deflation algorithm
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

    # Convert to numpy
    X = df[spectral_cols_sorted].astype(float).values
    y = df[label_column].values

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask]
            n_dropped = n_samples_before - X.shape[0]
            print(f"  Dropped {n_dropped} samples with NaN values ({n_samples_before} -> {X.shape[0]} samples)")
        elif handle_missing == "mean":
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            print(f"  Imputed NaN values with column means")
        elif handle_missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
            print(f"  Replaced NaN values with 0")
        elif handle_missing == "raise":
            raise ValueError(f"Found {n_missing} NaN values in data. Use handle_missing='drop', 'mean', or 'zero' to handle them.")
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}. Must be 'drop', 'mean', 'zero', or 'raise'.")

    # Encode labels for multiclass (one-hot encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Create one-hot encoded Y matrix
    Y = np.zeros((len(y_encoded), n_classes))
    Y[np.arange(len(y_encoded)), y_encoded] = 1

    # Standardize X (optional)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Data standardized (mean=0, std=1)")
    else:
        X_scaled = X.copy()
        scaler = None
        print(f"Using data without standardization (assumes already normalized)")

    print(f"Performing OPLS-DA for {dataset_name}...")
    print(f"  Number of classes: {n_classes}")
    print(f"  Predictive components: {n_components}")
    print(f"  Orthogonal components: {n_orthogonal}")

    # OPLS-DA implementation using iterative deflation
    X_work = X_scaled.copy()

    # Storage for components
    T_pred = np.zeros((X_work.shape[0], n_components))  # Predictive scores
    P_pred = np.zeros((X_work.shape[1], n_components))  # Predictive loadings
    W_pred = np.zeros((X_work.shape[1], n_components))  # Predictive weights

    T_ortho = np.zeros((X_work.shape[0], n_orthogonal))  # Orthogonal scores
    P_ortho = np.zeros((X_work.shape[1], n_orthogonal))  # Orthogonal loadings
    W_ortho = np.zeros((X_work.shape[1], n_orthogonal))  # Orthogonal weights

    # Step 1: Extract predictive component(s)
    for i in range(n_components):
        # PLS regression to get predictive direction
        pls_temp = PLSRegression(n_components=1)
        pls_temp.fit(X_work, Y)

        # Extract weights, scores, and loadings
        w = pls_temp.x_weights_[:, 0]
        t = X_work @ w
        p = (X_work.T @ t) / (t.T @ t)

        W_pred[:, i] = w
        T_pred[:, i] = t
        P_pred[:, i] = p

    # Step 2: Remove orthogonal variation
    X_orth = X_scaled.copy()
    for i in range(n_orthogonal):
        # Get residuals after removing predictive component
        X_resid = X_orth - T_pred[:, 0:1] @ P_pred[:, 0:1].T

        # Find orthogonal component
        w_ortho = X_resid.T @ (X_resid @ X_resid.T @ T_pred[:, 0])
        w_ortho = w_ortho / np.linalg.norm(w_ortho)

        t_ortho = X_resid @ w_ortho
        p_ortho = (X_resid.T @ t_ortho) / (t_ortho.T @ t_ortho)

        W_ortho[:, i] = w_ortho
        T_ortho[:, i] = t_ortho
        P_ortho[:, i] = p_ortho

        # Deflate
        X_orth = X_orth - t_ortho[:, np.newaxis] @ p_ortho[np.newaxis, :]

    # Calculate model metrics
    X_reconstructed = T_pred @ P_pred.T + T_ortho @ P_ortho.T
    r2x = 1 - np.sum((X_scaled - X_reconstructed) ** 2) / np.sum(X_scaled ** 2)

    # Predict Y from predictive scores
    Y_pred_model = np.linalg.lstsq(T_pred, Y, rcond=None)[0]
    Y_pred = T_pred @ Y_pred_model
    r2y = 1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)

    print(f"  R²X (variance explained in X): {r2x:.4f}")
    print(f"  R²Y (variance explained in Y): {r2y:.4f}")

    # Plot scores: Predictive vs First Orthogonal
    fig = plt.figure(figsize=figsize)
    unique_labels = le.classes_
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(T_pred[mask, 0], T_ortho[mask, 0],
                   c=[colors[idx]], label=label, alpha=0.6, s=50)

    plt.xlabel(f'Predictive Component (t_pred)', fontsize=12)
    plt.ylabel(f'Orthogonal Component 1 (t_ortho[1])', fontsize=12)
    plt.title(f'OPLS-DA Scores Plot - {dataset_name}\nR²X={r2x:.3f}, R²Y={r2y:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"oplsda_scores_{dataset_name}.pdf"
        else:
            file_path = f"oplsda_scores_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Scores plot saved to: {file_path}")

    plt.show()

    # Plot loadings: Predictive and Orthogonal
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(wn_sorted, P_pred[:, 0], linewidth=1.5, color='blue')
    axes[0].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[0].set_ylabel('Loading', fontsize=12)
    axes[0].set_title(f'Predictive Component Loadings - {dataset_name}',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    axes[1].plot(wn_sorted, P_ortho[:, 0], linewidth=1.5, color='red')
    axes[1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    axes[1].set_ylabel('Loading', fontsize=12)
    axes[1].set_title(f'Orthogonal Component 1 Loadings - {dataset_name}',
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"oplsda_loadings_{dataset_name}.pdf"
        else:
            file_path = f"oplsda_loadings_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Loadings plot saved to: {file_path}")

    plt.show()

    return {
        'T_pred': T_pred,
        'P_pred': P_pred,
        'W_pred': W_pred,
        'T_ortho': T_ortho,
        'P_ortho': P_ortho,
        'W_ortho': W_ortho,
        'Y_encoded': Y,
        'label_encoder': le,
        'scaler': scaler,
        'wavenumbers': wn_sorted,
        'r2x': r2x,
        'r2y': r2y
    }
