# normalization_eval.py
# Refactored version without EvalConfig class for pipeline integration

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import time
import warnings

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    balanced_accuracy_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import contextlib

from .normalization import normalize


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.
    
    Usage:
        with tqdm_joblib(tqdm(desc="Processing", total=n)) as pbar:
            results = Parallel(n_jobs=-1)(delayed(func)(x) for x in items)
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)



# ----------------------------
# Helpers
# ----------------------------
def spectral_angle(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Spectral angle mapper (SAM) in radians; lower => more similar shape."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(cos))


def within_group_mean_sam(X: np.ndarray, groups: np.ndarray) -> float:
    """Mean SAM across all pairs within each group (technical replicates)."""
    groups = np.asarray(groups)
    vals: List[float] = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < 2:
            continue
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                vals.append(spectral_angle(X[idx[i]], X[idx[j]]))
    return float(np.mean(vals)) if vals else np.nan


def infer_wavenumbers_from_columns(columns: Sequence[Any]) -> np.ndarray:
    """
    If your spectral columns are strings like '399.19', parse them to floats.
    If already numeric, this returns float array.
    """
    wn = []
    for c in columns:
        wn.append(float(c))
    return np.asarray(wn, dtype=float)


def zscore_robust(series: pd.Series) -> pd.Series:
    """
    Z-score normalization with small epsilon to avoid division by zero.

    Parameters
    ----------
    series : pd.Series
        Series of metric values to normalize.

    Returns
    -------
    pd.Series
        Z-scored values.
    """
    return (series - series.mean()) / (series.std(ddof=0) + 1e-12)


def make_agglomerative_cosine(n_clusters: int) -> AgglomerativeClustering:
    """
    sklearn changed 'affinity' -> 'metric'. This tries metric first, then falls back.
    """
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="cosine")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity="cosine")


# ----------------------------
# Normalization as a Transformer (leakage-safe)
# ----------------------------
class FTIRNormalizer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer wrapping your normalize() 1D function.

    Key feature:
    - PQN is implemented as a fit/transform: reference spectrum computed on fit()
      (train fold) and used in transform() (test fold), preventing leakage.

    Notes:
    - For methods that need wavenumbers (e.g. adaptive_regional), pass wavenumbers.
    """
    def __init__(
        self,
        method: str = "snv",
        wavenumbers: Optional[np.ndarray] = None,
        pqn_reference_type: str = "median",
        **kwargs: Any,
    ):
        self.method = method
        self.wavenumbers = None if wavenumbers is None else np.asarray(wavenumbers, dtype=float)
        self.pqn_reference_type = pqn_reference_type
        self.kwargs = kwargs
        self.reference_: Optional[np.ndarray] = None  # for PQN

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FTIRNormalizer":
        X = np.asarray(X, dtype=float)
        if self.method == "pqn":
            if self.pqn_reference_type not in ("median", "mean"):
                raise ValueError("pqn_reference_type must be 'median' or 'mean'")
            self.reference_ = (
                np.median(X, axis=0) if self.pqn_reference_type == "median" else np.mean(X, axis=0)
            ).astype(float)
        else:
            self.reference_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X, dtype=float)

        if self.method == "pqn":
            if self.reference_ is None:
                raise RuntimeError("PQN reference not fit. Call fit() first.")
            ref = self.reference_
            for i in range(X.shape[0]):
                out[i] = normalize(
                    intensities=X[i],
                    wavenumbers=self.wavenumbers,
                    method="pqn",
                    reference=ref,  # critical: train-fold reference
                    reference_type=self.pqn_reference_type,
                    **self.kwargs,
                )
            return out

        for i in range(X.shape[0]):
            out[i] = normalize(
                intensities=X[i],
                wavenumbers=self.wavenumbers,
                method=self.method,
                **self.kwargs,
            )
        return out


# ----------------------------
# Evaluation Functions
# ----------------------------
def evaluate_one_method(
    X: np.ndarray,
    y: np.ndarray,
    sam_groups: np.ndarray,
    wavenumbers: Optional[np.ndarray],
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 0,
    n_clusters: Optional[int] = None,
    cluster_bootstrap_rounds: int = 30,
    cluster_bootstrap_frac: float = 0.8,
    compute_internal_cluster_metrics: bool = True,
    clf: Any = None,
) -> Dict[str, Any]:
    """
    Evaluate a single normalization method.

    Parameters
    ----------
    X : np.ndarray
        Spectral data matrix (n_samples, n_features).
    y : np.ndarray
        Class labels for each sample.
    sam_groups : np.ndarray
        Group labels for within-class SAM (spectral angle) calculation.
        Typically the same as y (class labels).
    wavenumbers : np.ndarray or None
        Wavenumber values for spectral columns.
    method : str
        Normalization method name.
    method_kwargs : dict or None
        Additional keyword arguments for the normalization method.
    n_splits : int, default=5
        Number of cross-validation splits.
    random_state : int, default=0
        Random seed for reproducibility.
    n_clusters : int or None
        Number of clusters. If None, uses number of unique labels.
    cluster_bootstrap_rounds : int, default=30
        Number of bootstrap rounds for cluster stability.
    cluster_bootstrap_frac : float, default=0.8
        Fraction of data to subsample per bootstrap round.
    compute_internal_cluster_metrics : bool, default=True
        Whether to compute internal clustering metrics.
    clf : Any or None
        Classifier for supervised evaluation. If None, uses LogisticRegression.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    method_kwargs = method_kwargs or {}

    # Suppress all warnings during evaluation (including RuntimeWarnings in parallel workers)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # ---- supervised CV (leakage-safe via Pipeline) ----
        if clf is None:
            clf = LogisticRegression(max_iter=2000, n_jobs=None)

        norm = FTIRNormalizer(method=method, wavenumbers=wavenumbers, **method_kwargs)

        pipe = Pipeline([
            ("norm", norm),
            ("clf", clf),
        ])

        # Use StratifiedKFold for cross-validation (consistent with other modules)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y)

        f1s: List[float] = []
        bals: List[float] = []
        for tr, te in splits:
            pipe_fold = clone(pipe)
            pipe_fold.fit(X[tr], y[tr])
            pred = pipe_fold.predict(X[te])
            f1s.append(f1_score(y[te], pred, average="macro"))
            bals.append(balanced_accuracy_score(y[te], pred))

        sup_macro_f1 = float(np.mean(f1s))
        sup_bal_acc = float(np.mean(bals))

        # ---- normalize once on full data for unsupervised evaluation ----
        # Track computational time
        start_time = time.time()
        Xn = FTIRNormalizer(method=method, wavenumbers=wavenumbers, **method_kwargs).fit(X).transform(X)
        compute_time_sec = time.time() - start_time

        # Within-class consistency using spectral angle
        rep_sam = within_group_mean_sam(Xn, sam_groups)

        # ---- clustering (external validity vs labels + stability) ----
        n_clusters_actual = n_clusters if n_clusters is not None else int(np.unique(y).size)

        # Reference clustering on full data
        km_ref = KMeans(n_clusters=n_clusters_actual, n_init="auto", random_state=random_state)
        lab_ref = km_ref.fit_predict(Xn)

        ari_ref = adjusted_rand_score(y, lab_ref)
        nmi_ref = normalized_mutual_info_score(y, lab_ref)

        # Stability: subsample, recluster, compare to reference on same indices
        rng = np.random.default_rng(random_state)
        stab_scores: List[float] = []
        n = Xn.shape[0]
        m = max(2, int(round(cluster_bootstrap_frac * n)))
        for _ in range(cluster_bootstrap_rounds):
            idx = rng.choice(n, size=m, replace=False)
            km = KMeans(n_clusters=n_clusters_actual, n_init="auto", random_state=rng.integers(0, 2**31 - 1))
            lab_sub = km.fit_predict(Xn[idx])
            stab_scores.append(adjusted_rand_score(lab_ref[idx], lab_sub))
        cluster_stability = float(np.mean(stab_scores))

        # Optional: add a cosine-linkage clustering too (often good for spectra)
        agg = make_agglomerative_cosine(n_clusters=n_clusters_actual)
        lab_agg = agg.fit_predict(Xn)
        ari_agg = adjusted_rand_score(y, lab_agg)
        nmi_agg = normalized_mutual_info_score(y, lab_agg)

        # Internal clustering metrics (can be misleading; treat as secondary)
        sil = db = ch = np.nan
        if compute_internal_cluster_metrics:
            # cosine silhouette is often more meaningful for spectra
            try:
                sil = float(silhouette_score(Xn, lab_ref, metric="cosine"))
            except Exception:
                sil = np.nan
            try:
                db = float(davies_bouldin_score(Xn, lab_ref))
            except Exception:
                db = np.nan
            try:
                ch = float(calinski_harabasz_score(Xn, lab_ref))
            except Exception:
                ch = np.nan

        return {
            "method": method,
            "supervised_macro_f1": sup_macro_f1,
            "supervised_bal_acc": sup_bal_acc,
            "cluster_ARI_kmeans_vs_label": float(ari_ref),
            "cluster_NMI_kmeans_vs_label": float(nmi_ref),
            "cluster_stability_ARI_vs_ref": cluster_stability,
            "cluster_ARI_agglomerative_cosine_vs_label": float(ari_agg),
            "cluster_NMI_agglomerative_cosine_vs_label": float(nmi_agg),
            "internal_silhouette_cosine_kmeans": sil,
            "internal_davies_bouldin_kmeans": db,
            "internal_calinski_harabasz_kmeans": ch,
            "within_group_mean_SAM": float(rep_sam),
            "compute_time_sec": compute_time_sec,
        }


def print_scoring_summary(res: pd.DataFrame, top_n: int = 5) -> None:
    """
    Print a summary of normalization evaluation results with different scoring perspectives.

    Parameters
    ----------
    res : pd.DataFrame
        Results DataFrame from evaluate_norm_methods().
    top_n : int, default=5
        Number of top methods to display for each scoring scheme.
    """
    print("\n" + "=" * 80)
    print(f"NORMALIZATION EVALUATION SUMMARY (Top {top_n} Methods)")
    print("=" * 80)

    # Main combined score
    print("\n--- COMBINED SCORE (Recommended) ---")
    print("Weights: 40% supervised, 30% clustering, 20% stability, 10% consistency")
    top_combined = res.nlargest(top_n, "score_combined")
    for i, row in top_combined.iterrows():
        print(f"  {i+1}. {row['method']:20s} | Score: {row['score_combined']:6.3f} | "
              f"F1: {row['supervised_macro_f1']:.3f} | ARI: {row['cluster_ARI_kmeans_vs_label']:.3f} | "
              f"Stability: {row['cluster_stability_ARI_vs_ref']:.3f}")

    # Comprehensive score
    print("\n--- COMPREHENSIVE SCORE (All Metrics Equal) ---")
    top_comp = res.nlargest(top_n, "score_comprehensive")
    for i, row in top_comp.iterrows():
        print(f"  {i+1}. {row['method']:20s} | Score: {row['score_comprehensive']:6.3f}")

    # Unsupervised focus
    print("\n--- UNSUPERVISED SCORE (For Unlabeled Data) ---")
    print("Emphasizes clustering quality and stability")
    top_unsup = res.nlargest(top_n, "score_unsupervised")
    for i, row in top_unsup.iterrows():
        print(f"  {i+1}. {row['method']:20s} | Score: {row['score_unsupervised']:6.3f}")

    # Efficiency-aware
    print("\n--- EFFICIENT SCORE (Balances Quality & Speed) ---")
    top_eff = res.nlargest(top_n, "score_efficient")
    for i, row in top_eff.iterrows():
        print(f"  {i+1}. {row['method']:20s} | Score: {row['score_efficient']:6.3f} | "
              f"Time: {row['compute_time_sec']:.4f}s")

    print("\n" + "=" * 80 + "\n")


def evaluate_norm_methods(
    df: pd.DataFrame,
    methods: Optional[Sequence[str]] = None,
    method_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 0,
    n_clusters: Optional[int] = None,
    cluster_bootstrap_rounds: int = 30,
    cluster_bootstrap_frac: float = 0.8,
    compute_internal_cluster_metrics: bool = True,
    clf: Any = None,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple normalization methods on spectral data.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame: rows=samples, columns=wavenumbers + metadata.
        Spectral columns are auto-detected as numeric column names.
    methods : list of str, optional
        Normalization method names to evaluate.
        Default: ["snv", "vector", "area", "minmax", "max", "robust_snv", "pqn"]
    method_kwargs_map : dict, optional
        Dict mapping method name -> kwargs (e.g., {"minmax": {"feature_range": (0,1)}})
    label_column : str, default="label"
        Column name containing class labels for classification and within-class
        consistency metrics (SAM).
    exclude_columns : list of str, optional
        Non-spectral columns to exclude from processing (e.g., metadata columns).
    wn_min : float, optional
        Minimum wavenumber for filtering spectral range.
    wn_max : float, optional
        Maximum wavenumber for filtering spectral range.
    n_splits : int, default=5
        Number of cross-validation splits for StratifiedKFold.
    random_state : int, default=0
        Random seed for reproducibility.
    n_clusters : int, optional
        Number of clusters for clustering evaluation. If None, uses number of unique labels.
    cluster_bootstrap_rounds : int, default=30
        Number of bootstrap rounds for cluster stability evaluation.
    cluster_bootstrap_frac : float, default=0.8
        Fraction of data to subsample per bootstrap round.
    compute_internal_cluster_metrics : bool, default=True
        Whether to compute internal clustering metrics (silhouette, etc.).
    clf : Any, optional
        Classifier to use for supervised evaluation. If None, uses LogisticRegression.
    n_jobs : int, default=-1
        Number of parallel jobs for evaluating methods.
        -1 means using all processors (recommended). 1 means no parallelization.
    show_progress : bool, default=True
        Whether to show progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per method containing evaluation metrics,
        sorted by combined score (descending).

        The DataFrame includes:
        - Raw metrics (12 total): All evaluation metrics as computed
        - Z-scored metrics: Normalized versions where higher is always better
        - Combined scores (4 variants):
          * score_combined: Balanced score (40% supervised, 30% clustering, 20% stability, 10% consistency)
          * score_unsupervised: For unlabeled scenarios (emphasizes clustering & stability)
          * score_comprehensive: Equal-weighted average of all 11 metrics
          * score_efficient: Includes computational cost consideration

    Notes
    -----
    Metric directions (before z-scoring):
    - Higher is better: F1, balanced accuracy, ARI, NMI, silhouette, Calinski-Harabasz
    - Lower is better: Davies-Bouldin, SAM (spectral angle), compute time

    All z-scored metrics have been normalized so that higher values indicate better performance.
    """
    method_kwargs_map = method_kwargs_map or {}

    # Build list of columns to exclude from spectral data
    meta_columns = [label_column]
    if exclude_columns is not None:
        meta_columns.extend(exclude_columns)

    # Use label column for within-class SAM calculation
    # No GroupKFold - always use StratifiedKFold for consistency with other modules
    sam_groups = df.loc[:, label_column].to_numpy()

    # Use spectral_utils to infer and sort spectral columns (like baseline.py and denoise.py)
    spectral_columns, wavenumbers = _infer_spectral_columns(
        df, meta_columns, wn_min, wn_max
    )
    sorted_cols, sorted_wavenumbers, sort_idx = _sort_spectral_columns(
        spectral_columns, wavenumbers
    )

    X = df.loc[:, sorted_cols].to_numpy(dtype=float)
    y = df.loc[:, label_column].to_numpy()
    wavenumbers = sorted_wavenumbers

    # Edge case: check for sufficient classes
    n_unique_classes = len(np.unique(y))
    if n_unique_classes < 2:
        raise ValueError(
            f"Need at least 2 classes for meaningful evaluation, but only {n_unique_classes} found. "
            f"Check your '{label_column}' column."
        )

    if methods is None:
        # common, sane shortlist to start:
        methods = ["snv", "vector", "area", "minmax", "max", "robust_snv", "pqn"]

    # Parallel or sequential evaluation
    if n_jobs == 1:
        # Sequential execution with progress bar
        rows = []
        iterator = tqdm(methods, desc="Evaluating normalization methods") if show_progress else methods
        for m in iterator:
            rows.append(
                evaluate_one_method(
                    X=X,
                    y=y,
                    sam_groups=sam_groups,
                    wavenumbers=wavenumbers,
                    method=m,
                    method_kwargs=method_kwargs_map.get(m, {}),
                    n_splits=n_splits,
                    random_state=random_state,
                    n_clusters=n_clusters,
                    cluster_bootstrap_rounds=cluster_bootstrap_rounds,
                    cluster_bootstrap_frac=cluster_bootstrap_frac,
                    compute_internal_cluster_metrics=compute_internal_cluster_metrics,
                    clf=clf,
                )
            )
    else:
        # Parallel execution with progress bar using tqdm_joblib context manager
        if show_progress:
            with tqdm_joblib(tqdm(total=len(methods), desc="Evaluating normalization methods")):
                rows = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(evaluate_one_method)(
                        X=X,
                        y=y,
                        sam_groups=sam_groups,
                        wavenumbers=wavenumbers,
                        method=m,
                        method_kwargs=method_kwargs_map.get(m, {}),
                        n_splits=n_splits,
                        random_state=random_state,
                        n_clusters=n_clusters,
                        cluster_bootstrap_rounds=cluster_bootstrap_rounds,
                        cluster_bootstrap_frac=cluster_bootstrap_frac,
                        compute_internal_cluster_metrics=compute_internal_cluster_metrics,
                        clf=clf,
                    )
                    for m in methods
                )
        else:
            rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(evaluate_one_method)(
                    X=X,
                    y=y,
                    sam_groups=sam_groups,
                    wavenumbers=wavenumbers,
                    method=m,
                    method_kwargs=method_kwargs_map.get(m, {}),
                    n_splits=n_splits,
                    random_state=random_state,
                    n_clusters=n_clusters,
                    cluster_bootstrap_rounds=cluster_bootstrap_rounds,
                    cluster_bootstrap_frac=cluster_bootstrap_frac,
                    compute_internal_cluster_metrics=compute_internal_cluster_metrics,
                    clf=clf,
                )
                for m in methods
            )

    res = pd.DataFrame(rows)

    # ========================================================================
    # SCORING SYSTEM
    # ========================================================================
    # All metrics are converted to z-scores where HIGHER is BETTER.
    # For metrics where lower values are better (Davies-Bouldin, SAM, time),
    # we negate them before z-scoring.
    #
    # Raw metrics computed (12 total):
    #   HIGHER = BETTER:
    #   - supervised_macro_f1, supervised_bal_acc
    #   - cluster_ARI_kmeans_vs_label, cluster_NMI_kmeans_vs_label
    #   - cluster_ARI_agglomerative_cosine_vs_label, cluster_NMI_agglomerative_cosine_vs_label
    #   - cluster_stability_ARI_vs_ref
    #   - internal_silhouette_cosine_kmeans, internal_calinski_harabasz_kmeans
    #
    #   LOWER = BETTER (negated for z-scoring):
    #   - internal_davies_bouldin_kmeans (lower DB = better clustering)
    #   - within_group_mean_SAM (smaller angle = more similar spectra)
    #   - compute_time_sec (faster is better)
    #
    # Combined scoring variants:
    #   1. score_combined: Balanced (40% supervised, 30% clustering, 20% stability, 10% consistency)
    #   2. score_unsupervised: For unlabeled data (emphasizes clustering & stability)
    #   3. score_comprehensive: Equal-weighted average of all 11 quality metrics
    #   4. score_efficient: Includes computational cost (balance quality vs speed)
    # ========================================================================

    # ---- Compute z-scores for all metrics ----
    # For metrics where LOWER is better, negate before z-scoring so higher z-score = better

    # Supervised metrics (higher is better)
    res["score_supervised_f1_z"] = zscore_robust(res["supervised_macro_f1"])
    res["score_supervised_bal_acc_z"] = zscore_robust(res["supervised_bal_acc"])

    # Clustering external validity (higher is better)
    res["score_cluster_ARI_kmeans_z"] = zscore_robust(res["cluster_ARI_kmeans_vs_label"])
    res["score_cluster_NMI_kmeans_z"] = zscore_robust(res["cluster_NMI_kmeans_vs_label"])
    res["score_cluster_ARI_agg_z"] = zscore_robust(res["cluster_ARI_agglomerative_cosine_vs_label"])
    res["score_cluster_NMI_agg_z"] = zscore_robust(res["cluster_NMI_agglomerative_cosine_vs_label"])

    # Clustering stability (higher is better)
    res["score_stability_z"] = zscore_robust(res["cluster_stability_ARI_vs_ref"])

    # Internal clustering metrics
    res["score_silhouette_z"] = zscore_robust(res["internal_silhouette_cosine_kmeans"])  # higher is better
    # Davies-Bouldin: LOWER is better, so negate it
    res["score_davies_bouldin_z"] = zscore_robust(-res["internal_davies_bouldin_kmeans"])
    res["score_calinski_harabasz_z"] = zscore_robust(res["internal_calinski_harabasz_kmeans"])  # higher is better

    # Within-group consistency: LOWER SAM is better (smaller angle = more similar), so negate it
    res["score_sam_z"] = zscore_robust(-res["within_group_mean_SAM"])

    # Computational efficiency: LOWER time is better, so negate it
    res["score_compute_time_z"] = zscore_robust(-res["compute_time_sec"])

    # ---- Combined scores with different weighting schemes ----

    # Combined score (primary): Emphasizes supervised performance and clustering quality
    # Weights: supervised (40%), clustering (30%), stability (20%), consistency (10%)
    res["score_combined"] = (
        0.20 * res["score_supervised_f1_z"] +
        0.20 * res["score_supervised_bal_acc_z"] +
        0.10 * res["score_cluster_ARI_kmeans_z"] +
        0.10 * res["score_cluster_NMI_kmeans_z"] +
        0.10 * res["score_cluster_ARI_agg_z"] +
        0.20 * res["score_stability_z"] +
        0.10 * res["score_sam_z"]
    )

    # Alternative: Unsupervised-focused score (for unlabeled data scenarios)
    res["score_unsupervised"] = (
        0.20 * res["score_cluster_ARI_kmeans_z"] +
        0.15 * res["score_cluster_NMI_kmeans_z"] +
        0.15 * res["score_cluster_ARI_agg_z"] +
        0.25 * res["score_stability_z"] +
        0.15 * res["score_silhouette_z"] +
        0.10 * res["score_sam_z"]
    )

    # Alternative: Comprehensive score (uses all metrics equally)
    res["score_comprehensive"] = (
        res["score_supervised_f1_z"] +
        res["score_supervised_bal_acc_z"] +
        res["score_cluster_ARI_kmeans_z"] +
        res["score_cluster_NMI_kmeans_z"] +
        res["score_cluster_ARI_agg_z"] +
        res["score_cluster_NMI_agg_z"] +
        res["score_stability_z"] +
        res["score_silhouette_z"] +
        res["score_davies_bouldin_z"] +
        res["score_calinski_harabasz_z"] +
        res["score_sam_z"]
    ) / 11  # Average of 11 metrics (excluding compute time)

    # Alternative: Efficiency-aware score (includes computational cost)
    # Reweights the combined score to include 15% for speed
    res["score_efficient"] = (
        0.85 * res["score_combined"] +
        0.15 * res["score_compute_time_z"]
    )

    return res.sort_values("score_combined", ascending=False).reset_index(drop=True)


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    """
    Example usage with a DataFrame that has:
    - "label" column: class/group names (e.g., "PET", "HDPE", "PP")
    - Spectral columns named as wavenumbers (e.g., "680.0", "690.0", "700.0", ...)
    - Optional: other metadata columns (use exclude_columns to exclude them)
    """
    # Example: Load your data
    # df = pd.read_parquet("your_spectra_wide.parquet")
    # or
    # df = pd.read_csv("your_spectra.csv")

    # Define methods to evaluate
    methods = [
        "snv", "vector", "area", "minmax", "max",
        "robust_snv", "pqn",
        # Add more if available:
        # "curvature_weighted", "entropy_weighted",
        # "derivative_ratio", "signal_to_baseline",
    ]

    # Define method-specific parameters
    method_kwargs_map = {
        "minmax": {"feature_range": (0.0, 1.0)},
        "pqn": {"pqn_reference_type": "median"},
        # "curvature_weighted": {"sigma": 3.0, "min_weight": 0.01},
        # "entropy_weighted": {"n_bins": 50, "window_size": 30},
        # "derivative_ratio": {"sigma": 2.0},
        # "signal_to_baseline": {"baseline_percentile": 10, "signal_percentile": 90},
    }

    # Run evaluation
    # res = evaluate_norm_methods(
    #     df=df,
    #     methods=methods,
    #     method_kwargs_map=method_kwargs_map,
    #     label_column="label",      # Column with class labels
    #     exclude_columns=["sample"],  # Exclude sample ID column from spectral data
    #     n_splits=5,
    #     random_state=42,
    #     n_clusters=None,           # Auto-detect from unique labels
    #     cluster_bootstrap_rounds=30,
    # )

    # Display results using the helper function
    # print_scoring_summary(res, top_n=5)
    #
    # # Or view detailed results in a DataFrame
    # print("\n=== Detailed Metrics for Top Methods ===")
    # print(res[[
    #     "method",
    #     "supervised_macro_f1",
    #     "supervised_bal_acc",
    #     "cluster_ARI_kmeans_vs_label",
    #     "cluster_stability_ARI_vs_ref",
    #     "within_group_mean_SAM",
    #     "score_combined",
    #     "score_comprehensive",
    # ]].head(10))
    #
    # # Export results to CSV for further analysis
    # res.to_csv("normalization_evaluation_results.csv", index=False)
    # print("\nFull results saved to normalization_evaluation_results.csv")