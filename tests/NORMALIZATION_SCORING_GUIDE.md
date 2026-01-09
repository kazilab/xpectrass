# Normalization Evaluation Scoring System

## Overview

The `normalization_eval.py` module now includes a **comprehensive scoring system** that properly handles all 12 computed metrics with correct directionality and offers 4 different scoring schemes to match different priorities.

## Improvements Made

### 1. **Fixed Metric Directionality**
Previously, the scoring system ignored that some metrics are "lower is better". Now all metrics are properly handled:

- **Higher is Better** (9 metrics):
  - `supervised_macro_f1` - F1 score for classification
  - `supervised_bal_acc` - Balanced accuracy
  - `cluster_ARI_kmeans_vs_label` - Adjusted Rand Index (K-means)
  - `cluster_NMI_kmeans_vs_label` - Normalized Mutual Information (K-means)
  - `cluster_ARI_agglomerative_cosine_vs_label` - ARI (Agglomerative)
  - `cluster_NMI_agglomerative_cosine_vs_label` - NMI (Agglomerative)
  - `cluster_stability_ARI_vs_ref` - Clustering stability
  - `internal_silhouette_cosine_kmeans` - Silhouette score
  - `internal_calinski_harabasz_kmeans` - Calinski-Harabasz index

- **Lower is Better** (3 metrics - **negated before z-scoring**):
  - `internal_davies_bouldin_kmeans` - Davies-Bouldin index (lower = tighter clusters)
  - `within_group_mean_SAM` - Spectral angle (lower = more similar spectra)
  - `compute_time_sec` - Computational time (faster is better)

### 2. **Expanded from 3 to 12 Metrics**
The old `score_combined` only used 3 metrics. Now we compute z-scores for all 12 metrics and offer multiple scoring schemes.

### 3. **Four Scoring Schemes**

#### **`score_combined`** (Default - Recommended)
Balanced score emphasizing supervised performance and clustering quality.

**Weights:**
- 40% Supervised (20% F1 + 20% Balanced Accuracy)
- 30% Clustering (10% ARI K-means + 10% NMI K-means + 10% ARI Agglomerative)
- 20% Stability
- 10% Consistency (SAM)

**Use when:** You have labeled data and want a well-rounded evaluation.

#### **`score_unsupervised`**
Optimized for scenarios with unlabeled data.

**Weights:**
- 20% ARI K-means
- 15% NMI K-means
- 15% ARI Agglomerative
- 25% Stability
- 15% Silhouette
- 10% SAM

**Use when:** You don't have reliable labels or want to focus on clustering quality.

#### **`score_comprehensive`**
Equal-weighted average of all 11 quality metrics (excludes compute time).

**Weights:** Each metric contributes 1/11 ≈ 9.1%

**Use when:** You want to give equal importance to all aspects of normalization quality.

#### **`score_efficient`**
Balances normalization quality with computational cost.

**Weights:**
- 85% Combined score
- 15% Speed (negated compute time)

**Use when:** Computational efficiency matters (e.g., real-time applications, large datasets).

## Usage Examples

### Basic Usage

```python
from xpectrass_v002.utils.normalization_eval import evaluate_norm_methods, print_scoring_summary
import pandas as pd

# Load your spectral data
df = pd.read_parquet("your_spectra.parquet")

# Evaluate normalization methods
results = evaluate_norm_methods(
    df=df,
    methods=["snv", "vector", "area", "minmax", "pqn", "robust_snv"],
    label_column="label",
    n_splits=5,
    random_state=42
)

# Display summary with all scoring schemes
print_scoring_summary(results, top_n=5)
```

### Viewing Specific Scoring Schemes

```python
# Sort by different scores
by_combined = results.sort_values("score_combined", ascending=False)
by_unsupervised = results.sort_values("score_unsupervised", ascending=False)
by_comprehensive = results.sort_values("score_comprehensive", ascending=False)
by_efficient = results.sort_values("score_efficient", ascending=False)

# Get top method for each scheme
print("Best by combined:", by_combined.iloc[0]["method"])
print("Best by unsupervised:", by_unsupervised.iloc[0]["method"])
print("Best by comprehensive:", by_comprehensive.iloc[0]["method"])
print("Best by efficient:", by_efficient.iloc[0]["method"])
```

### Detailed Metrics View

```python
# View all metrics and scores for top methods
print(results[[
    "method",
    "supervised_macro_f1",
    "cluster_ARI_kmeans_vs_label",
    "cluster_stability_ARI_vs_ref",
    "within_group_mean_SAM",
    "compute_time_sec",
    "score_combined",
    "score_unsupervised",
    "score_comprehensive",
    "score_efficient"
]].head(10))
```

### Export Results

```python
# Save complete results to CSV
results.to_csv("normalization_evaluation_results.csv", index=False)
```

## Understanding Z-Scores

All individual metrics are converted to **z-scores** (standardized scores):
- Mean = 0
- Standard deviation = 1
- Higher z-score = better performance (even for metrics where raw lower is better)

This allows fair comparison and combination of metrics with different scales.

## Column Naming Convention

- **Raw metrics**: Original metric names (e.g., `supervised_macro_f1`)
- **Z-scored metrics**: Same name with `_z` suffix (e.g., `score_supervised_f1_z`)
- **Combined scores**: Start with `score_` (e.g., `score_combined`)

## Interpreting Results

### Raw Metrics
- **F1 & Balanced Accuracy**: 0-1, higher is better
- **ARI**: -1 to 1, higher is better (0 = random, 1 = perfect agreement)
- **NMI**: 0-1, higher is better
- **Silhouette**: -1 to 1, higher is better
- **Davies-Bouldin**: ≥0, **lower is better**
- **Calinski-Harabasz**: ≥0, higher is better
- **SAM**: 0 to π radians, **lower is better** (smaller angle = more similar)
- **Compute time**: seconds, **lower is better**

### Combined Scores
All combined scores are z-score based:
- Positive values = above average
- Negative values = below average
- Typical range: -3 to +3

## Recommendations

1. **Start with `score_combined`** - It's the most balanced for labeled data
2. **Check consistency** - Look at top methods across different scoring schemes
3. **Examine raw metrics** - Don't rely solely on combined scores; inspect individual metrics
4. **Consider your priorities** - Choose the scoring scheme that matches your use case
5. **Use `print_scoring_summary()`** - Get a quick overview of all perspectives

## Migration from Old Scoring

**Old behavior:**
```python
# Only 3 metrics, no proper handling of directionality
res["score_combined"] = (
    res["score_supervised_z"] +
    res["score_cluster_z"] +
    res["score_stability_z"]
)
```

**New behavior:**
```python
# 12 metrics total, 4 scoring schemes, proper directionality
# Weighted combination with all major aspects considered
res["score_combined"] = (
    0.20 * res["score_supervised_f1_z"] +
    0.20 * res["score_supervised_bal_acc_z"] +
    # ... (7 more components)
)
```

The new system is **backward compatible** - existing code will still work, but you now have access to much richer evaluation information.
