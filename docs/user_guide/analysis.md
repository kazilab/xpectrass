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
    sample_id_column="sample_id",  # Sample ID column name
    exclude_columns=None,    # Additional non-spectral columns
    start_wn=None,           # Minimum wavenumber (not yet implemented)
    end_wn=None,             # Maximum wavenumber (not yet implemented)
    drop_region=None,        # Wavenumber region to drop (not yet implemented)
    random_state=None,       # Random seed for reproducibility
    n_jobs=-1                # Parallel processing cores
)
```

## Spectral Visualization

### Plot Mean Spectra by Class

Visualize average spectra for each polymer type:

```python
# Plot mean spectra for all classes
analysis.plot_mean_spectra(
    title="Mean Spectra by Type",  # Plot title
    figsize=(16, 12),
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
    title="Mean Spectra overlay",  # Plot title
    figsize=(16, 12),
    save_plot=False,
    save_path=None
)
```

### Plot Spectral Heatmap

Visualize all spectra as a heatmap:

```python
# Create heatmap ordered by class
analysis.plot_heatmap(
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
    title="Spectral Variability by Type",  # Plot title
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
analysis.perform_anova(
    figsize=(16, 12),
    save_plot=False,
    save_path=None
)
```

**Plot shows:**
- -log10(p-value) across spectrum
- Significance threshold line
- Regions where classes differ significantly

### Correlation Matrix

Visualize correlations between wavenumbers:

```python
# Plot correlation matrix
analysis.plot_correlation(
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
analysis.plot_pca(
    standardize=True,        # Standardize features before PCA
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
)
```

**Plots generated:**
1. **2D scatter**: PC1 vs PC2 colored by class
2. **Explained variance plot**: Variance explained by each component

**Interpretation:**
- Well-separated clusters = good class discrimination
- Explained variance indicates information retention

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Non-linear dimensionality reduction for visualization:

```python
# Perform t-SNE
analysis.plot_tsne(
    pca_components=20,       # Number of PCA components for pre-reduction
    standardize=True,        # Standardize features
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
analysis.plot_umap(
    pca_components=20,       # Number of PCA components for pre-reduction
    standardize=True,        # Standardize features
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
analysis.plot_plsda(
    n_components=20,         # Number of latent variables
    standardize=True,        # Standardize features
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
)
```

**Best for:**
- Supervised classification and dimensionality reduction
- Feature importance analysis
- Maximizing class separation

### OPLS-DA (Orthogonal PLS-DA)

Enhanced PLS-DA with orthogonal signal correction:

```python
# Perform OPLS-DA
analysis.plot_oplsda(
    n_components=1,          # Predictive components
    n_orthogonal=2,          # Orthogonal components
    standardize=True,        # Standardize features
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
analysis.plot_kmeans_clus(
    n_components_clustering=10,  # PCA components for clustering
    k_range=(2, 11),         # Range of K values to evaluate
    standardize=True,        # Standardize features
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
)
```

**Plots:**
1. **Cluster scatter**: PCA projection colored by cluster
2. **Elbow plot**: Helps choose optimal K
3. **Silhouette analysis**: Shows cluster quality

**Choosing K:**
- Look for "elbow" in inertia plot
- Check silhouette scores
- Consider domain knowledge

### Hierarchical Clustering

Build dendrogram showing sample relationships:

```python
# Perform hierarchical clustering
analysis.plot_hierarchical_clus(
    n_samples_dendro=100,    # Max samples in dendrogram
    standardize=True,        # Standardize features
    handle_missing="zero",   # How to handle missing values
    figsize=(16, 12),
    save_plot=False,
    save_path=None
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
analysis.plot_mean_spectra()

print("Creating spectral heatmap...")
analysis.plot_heatmap()

print("Calculating coefficient of variation...")
analysis.plot_cv()

# 4. Statistical analysis
print("\nPerforming ANOVA...")
analysis.perform_anova()

# 5. Dimensionality reduction
print("\nDimensionality reduction analysis:")

print("  - PCA...")
analysis.plot_pca()

print("  - t-SNE...")
analysis.plot_tsne()

print("  - UMAP...")
analysis.plot_umap()

print("  - PLS-DA...")
analysis.plot_plsda()

# 6. Clustering
print("\nClustering analysis:")

print("  - K-means...")
analysis.plot_kmeans_clus()

print("  - Hierarchical...")
analysis.plot_hierarchical_clus()

print("\n✓ Analysis complete!")
```

## Saving Figures

All plotting methods support saving:

```python
# Save individual plots
analysis.plot_pca(
    save_plot=True,
    save_path="figures/pca_analysis.png"
)

# Save with custom format
analysis.plot_mean_spectra(
    save_plot=True,
    save_path="figures/mean_spectra.pdf"  # Supports: png, pdf, svg, eps
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
