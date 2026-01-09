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

# Statistical analysis
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Dimensionality reduction
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix


# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

def perform_kmeans_clustering(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        n_clusters: int = 6,
        pca_components: int = 50,
        n_components_clustering: int = 10,
        k_range: tuple = (2, 11),
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (12, 8),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform K-means clustering on PCA-reduced spectral data.

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
    n_clusters : int, default 6
        Number of clusters for final K-means model.
    pca_components : int, default 50
        Number of PCA components to extract for preprocessing.
    n_components_clustering : int, default 10
        Number of PCA components to use for clustering.
    k_range : tuple, default (2, 11)
        Range of K values to test (start, end).
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
        Dictionary containing 'kmeans', 'cluster_labels', 'X_pca', 'pca', and 'scaler'.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Applies PCA preprocessing before clustering
    • Handles missing values and provides optional standardization
    • Provides elbow plot and silhouette score analysis
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
    y = df[label_column].values if label_column in df.columns else None

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            if y is not None:
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
        scaler = None
        print(f"Using data without standardization (assumes already normalized)")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(pca_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"Performing K-Means Clustering for {dataset_name}...")
    print(f"  Using {n_components_clustering} PCA components for clustering")

    # Elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(k_range[0], k_range[1])

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca[:, :n_components_clustering])
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca[:, :n_components_clustering], kmeans.labels_))

    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(K_range, inertias, marker='o', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title(f'K-Means Elbow Curve - {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, color='orange')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title(f'Silhouette Score vs K - {dataset_name}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"kmeans_elbow_{dataset_name}.pdf"
        else:
            file_path = f"kmeans_elbow_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Elbow plot saved to: {file_path}")

    plt.show()

    # Perform final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca[:, :n_components_clustering])

    # Evaluate clustering
    silhouette = silhouette_score(X_pca[:, :n_components_clustering], cluster_labels)
    davies_bouldin = davies_bouldin_score(X_pca[:, :n_components_clustering], cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_pca[:, :n_components_clustering], cluster_labels)

    print(f"\nK-Means Clustering Summary for {dataset_name}:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.3f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")

    # Plot clusters
    fig = plt.figure(figsize=figsize)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    plt.title(f'K-Means Clustering (K={n_clusters}) - {dataset_name}',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"kmeans_clusters_{dataset_name}.pdf"
        else:
            file_path = f"kmeans_clusters_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Cluster plot saved to: {file_path}")

    plt.show()

    # Confusion matrix between clusters and true labels (if labels available)
    if y is not None:
        # Encode string labels to integers for confusion matrix
        le_labels = LabelEncoder()
        y_encoded = le_labels.fit_transform(y)

        cm = confusion_matrix(y_encoded, cluster_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                   yticklabels=le_labels.classes_)
        plt.xlabel('Predicted Cluster', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Cluster vs True Label Confusion Matrix - {dataset_name}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_plot:
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f"kmeans_confusion_{dataset_name}.pdf"
            else:
                file_path = f"kmeans_confusion_{dataset_name}.pdf"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {file_path}")

        plt.show()

    return {
        'kmeans': kmeans,
        'cluster_labels': cluster_labels,
        'X_pca': X_pca,
        'pca': pca,
        'scaler': scaler,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }
    
    

def perform_hierarchical_clustering(
        data: Union[pd.DataFrame, pl.DataFrame],
        label_column: str = "label",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        dataset_name: str = "c8",
        n_clusters: int = 6,
        pca_components: int = 50,
        n_components_clustering: int = 10,
        n_samples_dendro: int = 100,
        linkage_method: str = 'ward',
        standardize: bool = True,
        handle_missing: str = "zero",
        figsize: tuple = (12, 8),
        save_plot: bool = False,
        save_path: Optional[str] = None
    ):
    """
    Perform hierarchical clustering with dendrogram visualization.

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
    n_clusters : int, default 6
        Number of clusters for agglomerative clustering.
    pca_components : int, default 50
        Number of PCA components to extract for preprocessing.
    n_components_clustering : int, default 10
        Number of PCA components to use for clustering.
    n_samples_dendro : int, default 100
        Number of samples to use for dendrogram visualization.
    linkage_method : str, default 'ward'
        Linkage method ('ward', 'complete', 'average', 'single').
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
        Dictionary containing 'hc', 'cluster_labels', 'X_pca', 'pca', and 'scaler'.

    Notes
    -----
    • Works with both pandas and polars DataFrames
    • Uses shared spectral utilities for robust column detection
    • Applies PCA preprocessing before clustering
    • Handles missing values and provides optional standardization
    • Samples data for dendrogram to avoid overcrowding
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
    y = df[label_column].values if label_column in df.columns else None

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask] if y is not None else None
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
        scaler = None
        print(f"Using data without standardization (assumes already normalized)")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(pca_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"Performing Hierarchical Clustering for {dataset_name}...")
    print(f"  Using {n_components_clustering} PCA components for clustering")
    print(f"  Linkage method: {linkage_method}")

    # Sample for dendrogram visualization
    n_samples_dendro = min(n_samples_dendro, len(X_pca))
    sample_idx = np.random.choice(len(X_pca), n_samples_dendro, replace=False)
    X_sample = X_pca[sample_idx, :n_components_clustering]
    y_sample = y[sample_idx] if y is not None else None

    # Compute linkage
    Z = linkage(X_sample, method=linkage_method)

    # Plot dendrogram
    plt.figure(figsize=(16, 8))
    if y_sample is not None:
        dendrogram(Z, labels=y_sample, leaf_font_size=8)
    else:
        dendrogram(Z, leaf_font_size=8)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title(f'Hierarchical Clustering Dendrogram ({n_samples_dendro} samples) - {dataset_name}',
             fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"hierarchical_dendrogram_{dataset_name}.pdf"
        else:
            file_path = f"hierarchical_dendrogram_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved to: {file_path}")

    plt.show()

    # Perform hierarchical clustering on full dataset
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    cluster_labels = hc.fit_predict(X_pca[:, :n_components_clustering])

    # Evaluate
    silhouette = silhouette_score(X_pca[:, :n_components_clustering], cluster_labels)
    davies_bouldin = davies_bouldin_score(X_pca[:, :n_components_clustering], cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_pca[:, :n_components_clustering], cluster_labels)

    print(f"\nHierarchical Clustering Summary for {dataset_name}:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.3f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")

    # Plot clusters in PCA space
    fig = plt.figure(figsize=figsize)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    plt.title(f'Hierarchical Clustering (K={n_clusters}) - {dataset_name}',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"hierarchical_clusters_{dataset_name}.pdf"
        else:
            file_path = f"hierarchical_clusters_{dataset_name}.pdf"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Cluster plot saved to: {file_path}")

    plt.show()

    # Confusion matrix between clusters and true labels (if labels available)
    if y is not None:
        # Encode string labels to integers for confusion matrix
        le_labels = LabelEncoder()
        y_encoded = le_labels.fit_transform(y)

        cm = confusion_matrix(y_encoded, cluster_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                   yticklabels=le_labels.classes_)
        plt.xlabel('Predicted Cluster', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Cluster vs True Label Confusion Matrix - {dataset_name}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_plot:
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f"hierarchical_confusion_{dataset_name}.pdf"
            else:
                file_path = f"hierarchical_confusion_{dataset_name}.pdf"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {file_path}")

        plt.show()

    return {
        'hc': hc,
        'cluster_labels': cluster_labels,
        'X_pca': X_pca,
        'pca': pca,
        'scaler': scaler,
        'linkage_matrix': Z,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }
