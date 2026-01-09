# Data Analysis and Visualization

The `FTIRdataanalysis` class provides comprehensive tools for exploratory data analysis, statistical testing, dimensionality reduction, and clustering of preprocessed FTIR spectra.

## Overview

After preprocessing your FTIR data with `FTIRdataprocessing`, use `FTIRdataanalysis` to:

- Visualize spectral patterns and trends
- Perform statistical analysis
- Reduce dimensionality for visualization
- Cluster samples based on spectral similarity
- Prepare data for machine learning

```python
from xpectrass import FTIRdataanalysis

# Initialize with preprocessed data
analysis = FTIRdataanalysis(
    df=processed_df,
    dataset_name="MyDataset",
    label_column="type",
    random_state=42,
    n_jobs=-1
)
```

## Initialization Parameters

### FTIRdataanalysis.__init__()

```python
FTIRdataanalysis(
    df,                      # Preprocessed DataFrame
    dataset_name=None,       # Dataset identifier for plots
    label_column="type",     # Label column name
    exclude_columns=None,    # Additional non-spectral columns
    random_state=None,       # Random seed for reproducibility
    n_jobs=-1               # Parallel processing cores
)
```

## Spectral Visualization

### Plot Mean Spectra by Class

Visualize average spectra for each polymer type:

```python
# Plot mean spectra for all classes
analysis.plot_mean_spectra(
    by_class=True,           # Separate by class
    show_std=True,           # Show standard deviation bands
    figsize=(12, 6),
    save_plot=False,
    save_path=None
)
```

**Output:**
- Mean spectrum for each class with different colors
- Optional standard deviation shading
- Legend showing all polymer types

### Plot Overlay of Mean Spectra

Compare mean spectra across classes:

```python
# Overlay all class means
analysis.plot_overlay_spectra(
    normalize=True,          # Normalize for comparison
    offset=0.0,              # Vertical offset between spectra
    figsize=(12, 8)
)
```

### Plot Spectral Heatmap

Visualize all spectra as a heatmap:

```python
# Create heatmap ordered by class
analysis.plot_heatmap(
    cluster=False,           # Don't cluster samples
    cmap='viridis',          # Colormap
    figsize=(14, 10),
    save_plot=False
)
```

**Features:**
- Samples as rows, wavenumbers as columns
- Color intensity represents absorbance
- Optional hierarchical clustering
- Class annotations

### Plot Coefficient of Variation

Identify variable and stable spectral regions:

```python
# Plot CV by class
analysis.plot_cv(
    by_class=True,           # Separate CV for each class
    figsize=(12, 6)
)
```

**Interpretation:**
- High CV = high variability (potential noise or real variation)
- Low CV = stable peaks (good for classification)

## Statistical Analysis

### ANOVA Analysis

Test for significant differences between classes:

```python
# Perform ANOVA at each wavenumber
anova_results = analysis.perform_anova(
    plot=True,               # Plot significant regions
    alpha=0.05,              # Significance level
    figsize=(12, 6)
)

print(f"Significant wavenumbers: {anova_results['n_significant']}")
print(f"P-values shape: {anova_results['p_values'].shape}")
```

**Output:**
- `p_values`: P-value at each wavenumber
- `n_significant`: Number of significant wavenumbers
- `significant_wn`: List of significant wavenumber positions

**Plot shows:**
- -log10(p-value) across spectrum
- Horizontal line at significance threshold
- Regions where classes differ significantly

### Correlation Matrix

Visualize correlations between wavenumbers:

```python
# Plot correlation matrix
analysis.plot_correlation(
    method='pearson',        # 'pearson' or 'spearman'
    figsize=(10, 8),
    cmap='coolwarm'
)
```

**Use cases:**
- Identify correlated spectral regions
- Detect redundancy in features
- Guide feature selection

## Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
# Perform PCA
pca_results = analysis.plot_pca(
    n_components=3,          # Number of PCs to plot
    plot_loadings=True,      # Show PC loadings
    plot_scree=True,         # Show scree plot
    figsize=(15, 10),
    save_plot=False
)
```

**Returns:**
- `pca_model`: Fitted PCA model
- `scores`: PC scores for all samples
- `loadings`: PC loadings (wavenumber contributions)
- `explained_variance`: Variance explained by each PC

**Plots generated:**
1. **2D scatter**: PC1 vs PC2 colored by class
2. **3D scatter**: PC1 vs PC2 vs PC3 (if n_components ≥ 3)
3. **Loadings plot**: Shows which wavenumbers contribute to each PC
4. **Scree plot**: Variance explained by each component

**Interpretation:**
- Well-separated clusters = good class discrimination
- PC loadings show important spectral features
- Explained variance indicates information retention

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction for visualization:

```python
# Perform t-SNE
tsne_results = analysis.plot_tsne(
    perplexity=30,           # Balance local vs global structure (5-50)
    n_iter=1000,             # Number of iterations
    learning_rate=200,       # Step size
    figsize=(10, 8)
)
```

**Parameters:**
- `perplexity`: Higher values focus on global structure
- Typical range: 5-50
- Recommended: 30 for most datasets

**Best for:**
- Visualizing complex, non-linear patterns
- Revealing cluster structure
- Publication-quality figures

### UMAP (Uniform Manifold Approximation and Projection)

Modern non-linear dimensionality reduction:

```python
# Perform UMAP
umap_results = analysis.plot_umap(
    n_neighbors=15,          # Local neighborhood size (2-100)
    min_dist=0.1,            # Minimum distance between points (0.0-0.99)
    metric='euclidean',      # Distance metric
    figsize=(10, 8)
)
```

**Parameters:**
- `n_neighbors`: Controls balance between local and global structure
  - Low (5-15): Emphasizes local structure
  - High (50-100): Emphasizes global structure
- `min_dist`: Controls cluster tightness
  - Low (0.0-0.1): Tight clusters
  - High (0.5-0.99): Spread out clusters

**Advantages over t-SNE:**
- Faster computation
- Better preserves global structure
- More consistent results
- Supports supervised projections

### PLS-DA (Partial Least Squares Discriminant Analysis)

Supervised dimensionality reduction:

```python
# Perform PLS-DA
plsda_results = analysis.plot_plsda(
    n_components=3,          # Number of latent variables
    cv_folds=5,              # Cross-validation folds
    plot_scores=True,        # Plot score plot
    plot_loadings=True,      # Plot loadings
    figsize=(15, 10)
)
```

**Returns:**
- `model`: Fitted PLS-DA model
- `scores`: LV scores
- `loadings`: Variable loadings
- `cv_scores`: Cross-validation scores
- `vip_scores`: Variable Importance in Projection

**Best for:**
- Supervised classification
- Feature importance analysis
- Regression tasks

### OPLS-DA (Orthogonal PLS-DA)

Enhanced PLS-DA with orthogonal signal correction:

```python
# Perform OPLS-DA
oplsda_results = analysis.plot_oplsda(
    n_components=1,          # Predictive components
    n_orthogonal=2,          # Orthogonal components
    figsize=(12, 8)
)
```

**Advantages:**
- Separates predictive and orthogonal variation
- Easier interpretation than PLS-DA
- Better for biomarker discovery

## Clustering Analysis

### K-means Clustering

Partition spectra into K clusters:

```python
# Perform K-means clustering
kmeans_results = analysis.plot_kmeans_clus(
    n_clusters=5,            # Number of clusters
    plot_elbow=True,         # Show elbow plot for K selection
    max_k=10,                # Maximum K for elbow plot
    figsize=(15, 8)
)
```

**Returns:**
- `model`: Fitted K-means model
- `labels`: Cluster assignments
- `inertia`: Sum of squared distances
- `silhouette_score`: Cluster quality metric

**Plots:**
1. **Cluster scatter**: PCA projection colored by cluster
2. **Elbow plot**: Helps choose optimal K
3. **Silhouette plot**: Shows cluster quality

**Choosing K:**
- Look for "elbow" in inertia plot
- Check silhouette scores
- Consider domain knowledge

### Hierarchical Clustering

Build dendrogram showing sample relationships:

```python
# Perform hierarchical clustering
hclust_results = analysis.plot_hierarchical_clus(
    method='ward',           # Linkage method
    metric='euclidean',      # Distance metric
    figsize=(15, 10),
    dendrogram_only=False    # Also show clustered heatmap
)
```

**Linkage methods:**
- `ward`: Minimizes variance (recommended)
- `average`: Average linkage
- `complete`: Maximum linkage
- `single`: Minimum linkage

**Plots:**
1. **Dendrogram**: Tree showing sample relationships
2. **Clustered heatmap**: Spectra reordered by clustering

## Complete Analysis Example

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_jung_2018

# 1. Load and preprocess data
df = load_jung_2018()
ftir = FTIRdataprocessing(df, label_column="type")
ftir.run()
processed_df = ftir.df_norm

# 2. Initialize analysis
analysis = FTIRdataanalysis(
    processed_df,
    dataset_name="Jung_2018",
    label_column="type",
    random_state=42
)

# 3. Exploratory visualization
print("Plotting mean spectra...")
analysis.plot_mean_spectra(by_class=True, show_std=True)

print("Creating spectral heatmap...")
analysis.plot_heatmap(cluster=True)

print("Calculating coefficient of variation...")
analysis.plot_cv(by_class=True)

# 4. Statistical analysis
print("\nPerforming ANOVA...")
anova_results = analysis.perform_anova(plot=True)
print(f"Found {anova_results['n_significant']} significant wavenumbers")

# 5. Dimensionality reduction
print("\nDimensionality reduction analysis:")

print("  - PCA...")
pca_results = analysis.plot_pca(n_components=3, plot_loadings=True)
print(f"    Variance explained: {pca_results['explained_variance'][:3]}")

print("  - t-SNE...")
analysis.plot_tsne(perplexity=30)

print("  - UMAP...")
analysis.plot_umap(n_neighbors=15)

print("  - PLS-DA...")
plsda_results = analysis.plot_plsda(n_components=3, cv_folds=5)

# 6. Clustering
print("\nClustering analysis:")

print("  - K-means...")
kmeans_results = analysis.plot_kmeans_clus(n_clusters=5, plot_elbow=True)
print(f"    Silhouette score: {kmeans_results['silhouette_score']:.3f}")

print("  - Hierarchical...")
analysis.plot_hierarchical_clus(method='ward')

print("\n✓ Analysis complete!")
```

## Saving Figures

All plotting methods support saving:

```python
# Save individual plots
analysis.plot_pca(
    n_components=3,
    save_plot=True,
    save_path="figures/pca_analysis.png"
)

# Save with custom format
analysis.plot_mean_spectra(
    save_plot=True,
    save_path="figures/mean_spectra.pdf",  # Supports: png, pdf, svg, eps
    dpi=300
)
```

## Tips and Best Practices

1. **Always visualize first**: Start with mean spectra and heatmaps to understand your data
2. **Check statistical significance**: Use ANOVA to identify discriminative wavenumbers
3. **Try multiple methods**: Compare PCA, t-SNE, and UMAP for different perspectives
4. **Use appropriate parameters**:
   - t-SNE perplexity: 5-50 (typically 30)
   - UMAP n_neighbors: 5-100 (typically 15)
   - K-means: Use elbow plot to choose K
5. **Save your figures**: Use high DPI (300) for publication-quality images
6. **Cross-validate**: Use PLS-DA cross-validation to assess model quality
7. **Interpret loadings**: PCA/PLS loadings show which peaks drive separation

## Next Steps

- See [Machine Learning](machine_learning.md) for classification workflows
- See [Preprocessing Pipeline](preprocessing_pipeline.md) for data preparation
- See [Examples](../examples.md) for complete workflows
