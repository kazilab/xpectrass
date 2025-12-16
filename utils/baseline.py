from __future__ import annotations   # for Python < 3.11 type-hinting
from pathlib import Path
from typing import Literal, Tuple, Union, Callable, Dict, List
import warnings
import numpy as np
import pandas as pd
import polars as pl
from scipy import signal, ndimage
from pybaselines import Baseline
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
#                    UNIVERSAL BASELINE-CORRECTION WRAPPER                    #
# --------------------------------------------------------------------------- #
def baseline_correction(
    intensities: Union[np.ndarray, list, tuple],
    method: str = "airpls",
    window_size: int = 101,
    poly_order: int = 4,
    clip_negative: bool = True,
    return_baseline: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Baseline-correct a 1-D FT-IR or ToF-SIMS spectrum with >50 algorithms.

    Parameters
    ----------
    intensities : array-like
        Raw y-values (%T or absorbance); will be converted to ``float64``.
    method : str, default "airpls"
        Name of the baseline algorithm.  All **pybaselines** methods plus
        two custom filters ("median_filter", "adaptive_window") are accepted.
    window_size : int, default 101
        Odd kernel width for the two custom windowed filters.
    poly_order : int, default 4
        Polynomial order for the `"poly"` baseline.
    clip_negative : bool, default True
        If *True*, set negative corrected values to 0 (useful for %T spectra).
    return_baseline : bool, default False
        If *True*, return ``(corrected, baseline)`` instead of just ``corrected``.
    **kwargs :
        Extra keyword arguments are forwarded verbatim to the selected
        **pybaselines** algorithm (e.g. ``lam=1e6, p=0.01`` for AsLS).

    Returns
    -------
    corrected : np.ndarray
        Baseline-subtracted intensities (same dtype & length as input).
    baseline : np.ndarray , optional
        Returned *only* if ``return_baseline=True``.
    """
    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("`intensities` must be a 1-D array-like object.")

    # --------------------------------------------------------------------- #
    #                Dispatch table for pybaselines algorithms              #
    # --------------------------------------------------------------------- #
    
    bl = Baseline()
    _skip = {"pentapy_solver", "banded_solver"}
    pyb_dispatch = {}
    for name in dir(bl):
        if name.startswith("_") or name in _skip:
            continue
        attr = getattr(bl, name)
        if callable(attr):
            pyb_dispatch[name] = attr
    # Add a convenience alias for polynomial fits
    pyb_dispatch["poly"] = lambda arr, *, poly_order=poly_order, **k: bl.poly(
        arr, poly_order=poly_order, **k
    )

    # --------------------------------------------------------------------- #
    #                          Custom windowed filters                      #
    # --------------------------------------------------------------------- #
    if method == "median_filter":
        baseline = signal.medfilt(y, kernel_size=window_size)
    elif method == "adaptive_window":
        baseline = ndimage.minimum_filter1d(y, size=window_size)
    elif method in pyb_dispatch:
        # pybaselines: -> (baseline, params_dict)
        baseline, _ = pyb_dispatch[method](y, **kwargs)
    else:
        raise ValueError(
            f"Unknown baseline method '{method}'. Check pybaselines docs "
            "or spell-check your custom method name."
        )

    corrected = y - baseline
    if clip_negative:
        corrected[corrected < 0] = 0.0

    return (corrected, baseline) if return_baseline else corrected


# --------------------------------------------------------------------------- #
#                    CHECK AVAILABLE BASELINE-CORRECTION METHODS              #
# --------------------------------------------------------------------------- #


def baseline_method_names() -> list[str]:
    """
    Return a sorted list of method names that can be passed to
    `baseline_correction(method=...)`.

    The list is generated dynamically from `pybaselines.Baseline`,
    skipping the deprecated solver helpers, and then augmented with the
    two custom windowed filters plus the convenient 'poly' alias.
    """
    bl = Baseline()
    skip = {"pentapy_solver", "banded_solver"}           # solver helpers

    # collect every public callable attribute name
    methods = {
        name
        for name in dir(bl)
        if (
            not name.startswith("_")
            and name not in skip
            and callable(getattr(bl, name))
        )
    }

    # add our wrapper-specific extras
    methods.update({"median_filter", "adaptive_window", "poly"})
    values_to_remove = ['collab_pls', 'interp_pts', 'cwt_br']
    methods = [x for x in methods if x not in values_to_remove]
    return sorted(methods)


# --------------------------------------------------------------------------- #
#   CHECK PERFORMANCE OF AVAILABLE BASELINE-CORRECTION METHODS BY PLOTING     #
# --------------------------------------------------------------------------- #

def plot_corrected_spectrum(
    df: pd.DataFrame | pl.DataFrame,
    sample_name: str,
    method: str
) -> None:
    """
    Plot raw vs. baseline-corrected FT-IR spectrum for one sample.

    Parameters
    ----------
    df : pandas.DataFrame | polars.DataFrame
        Wide-format matrix (rows = samples; columns = wavenumbers) plus:
            • a 'label' column,               and
            • either an index named 'sample' (pandas) *or*
              a column 'sample'   (Polars).
    sample_name : str
        Name in the index/column identifying the row to plot.
    method : str
        Baseline algorithm recognised by `baseline(...)`.
    lam : float, default 1e6
        Example parameter forwarded to `baseline` for Whittaker-type methods.
    **baseline_kwargs :
        Extra keyword arguments forwarded verbatim to `baseline(...)`.
    """
    # ------------------------------------------------------------------ #
    # DataFrame-type autodetection & spectrum extraction              #
    # ------------------------------------------------------------------ #
    if isinstance(df, pd.DataFrame):
        if "label" not in df.columns:
            raise KeyError("'label' column missing in pandas DataFrame")
        if sample_name not in df.index:
            raise KeyError(f"sample '{sample_name}' not found in index")
        series = df.loc[sample_name].drop("label")          # 1-D Series
        x      = series.index.astype(float).to_numpy()
        y_pct  = series.to_numpy(dtype=float)

    elif isinstance(df, pl.DataFrame):
        if not {"sample", "label"}.issubset(df.columns):
            raise KeyError("'sample' and/or 'label' column(s) missing in Polars DataFrame")
        row_pl = (df.filter(pl.col("sample") == sample_name)
                     .select(pl.exclude(["sample", "label"])))
        if row_pl.height == 0:
            raise KeyError(f"sample '{sample_name}' not found in 'sample' column")
        x      = np.array([float(c) for c in row_pl.columns], dtype=float)
        y_pct  = row_pl.to_numpy().ravel().astype(float)

    else:
        raise TypeError("df must be a pandas or polars DataFrame")

    # ------------------------------------------------------------------ #
    # %T   → absorbance (optional, but needed for Beer–Lambert)       #
    # ------------------------------------------------------------------ #
    y_abs = -np.log10(y_pct / 100.0)

    # ------------------------------------------------------------------ #
    # Baseline correction                                             #
    # ------------------------------------------------------------------ #
    y_corr_abs = baseline_correction(y_abs, method=method)

    # ------------------------------------------------------------------ #
    # Convert back to %T for visual comparison                        #
    # ------------------------------------------------------------------ #
    y_corr_pct = 100.0 * 10.0 ** (-y_corr_abs)

    # ------------------------------------------------------------------ #
    # Plot                                                            #
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(7, 4))
    plt.plot(x, y_pct,       label="raw %T",                linewidth=1.2)
    plt.plot(x, y_corr_pct,  label=f"{method}-corrected %T", linewidth=1.2)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Transmittance (%)")
    plt.title(f"{sample_name}: baseline correction via '{method}'")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3, lw=0.5)
    plt.tight_layout()
    plt.show()


# Usage
# plot_corrected_spectrum(df=df_wide, sample_name='PP7', method = 'airpls')


import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def eval_baseline(y_raw, x_axis, flat_windows):
    """
    Rank baseline algorithms by three unsupervised proxy metrics.

    Parameters
    ----------
    y_raw : np.ndarray            # one spectrum
    x_axis : np.ndarray           # wavenumbers (same length)
    flat_windows : list[tuple]    # e.g. [(1850, 1900), (3200, 3500)]

    Returns
    -------
    pandas.DataFrame  (rows = method, cols = metrics)
    """
    results = []
    methods = baseline_methods()
    for m in methods:
        y_corr = baseline(y_raw, method=m, clip_negative=False)
        # --- metric 1: RFZN ---------------------------------------------------
        mask = np.zeros_like(x_axis, dtype=bool)
        for lo, hi in flat_windows:
            mask |= (x_axis >= lo) & (x_axis <= hi)
        rfzn = np.sqrt(np.mean(y_corr[mask]**2))
        # --- metric 2: NAR ----------------------------------------------------
        nar  = np.sum(-y_corr[y_corr < 0]) / np.sum(np.abs(y_corr))
        results.append((m, rfzn, nar))
    cols = ["method", "RFZN", "NAR"]
    return pd.DataFrame(results, columns=cols).set_index("method")\
             .sort_values(["RFZN", "NAR"], ascending=[True, True])


# -------------------------------------------------------------------------
#  TEST BASELINE CORRECTION METHODS
#  WE WILL USE RELATIVE METHODS
#  CALCULATE Residual Flat-Zone Noise (RFZN) and Negative Area Ratio (NAR)
# -------------------------------------------------------------------------

def _make_mask(x_axis: np.ndarray, flat_windows: List[Tuple[float, float]]
               ) -> np.ndarray:
    """Boolean mask for baseline-only wavenumber regions."""
    mask = np.zeros_like(x_axis, dtype=bool)
    for lo, hi in flat_windows:
        mask |= (x_axis >= lo) & (x_axis <= hi)
    return mask


def _score_one_sample(
    sample_name: str,
    y_raw: np.ndarray,
    methods: List[str],
    mask: np.ndarray,
    negative_clip: bool,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Worker function: returns RFZN and NAR vectors for one spectrum."""
    n_methods = len(methods)
    rfzn_row = np.empty(n_methods, dtype=float)
    nar_row  = np.empty(n_methods, dtype=float)

    for j, m in enumerate(methods):
        try:
            y_corr = baseline_correction(
                y_raw, method=m, clip_negative=negative_clip
            )
            # RFZN
            rfzn_row[j] = np.sqrt(np.mean(y_corr[mask] ** 2))
            # NAR
            neg_area = np.sum(-y_corr[y_corr < 0.0])
            nar_row[j] = neg_area / np.sum(np.abs(y_corr))
        except Exception as e:
            # algorithm failed (shape mismatch, optimiser error, …)
            rfzn_row[j] = np.nan
            nar_row[j]  = np.nan
    return sample_name, rfzn_row, nar_row


def evaluate_all_samples(
    df_wide: pl.DataFrame,
    flat_windows: List[Tuple[float, float]],
    negative_clip: bool = False,
    n_jobs: int = -1,                # -1 → all CPU cores
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute RFZN & NAR for every (sample, method) pair in parallel.

    Parameters
    ----------
    df_wide : Polars DataFrame
        Rows = samples, columns = wavenumbers, plus 'sample' and 'label'.
    flat_windows : list[(low, high)]
        Wavenumber intervals known to contain baseline only.
    negative_clip : bool, default False
        Forwarded to `baseline_correction`.
    n_jobs : int, default -1
        How many worker processes; -1 = all cores.

    Returns
    -------
    rfzn_tbl, nar_tbl : pandas.DataFrame
        index = sample, columns = method
    """
    # ---------------- preparation ----------------------------------------
    methods  = baseline_method_names()            # safe 1-D algorithms
    samples  = df_wide.get_column("sample").to_list()
    wavenums = np.array(
        [float(c) for c in df_wide.columns if c not in ("sample", "label")]
    )
    mask     = _make_mask(wavenums, flat_windows)

    # extract all spectra to NumPy in parent process (zero-copy)
    spectra = {
        s: (df_wide
             .filter(pl.col("sample") == s)
             .select(pl.exclude(["sample", "label"]))
             .to_numpy()
             .ravel())
        for s in samples
    }

    # ---------------- parallel execution ---------------------------------
    worker = delayed(_score_one_sample)
    iterator = (worker(s, spectra[s], methods, mask, negative_clip) for s in samples)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        tqdm(iterator, total=len(samples), desc="baseline eval", ncols=88)
    )

    # ---------------- assemble results -----------------------------------
    rfzn_arr = np.vstack([row[1] for row in results])
    nar_arr  = np.vstack([row[2] for row in results])

    rfzn_tbl = pd.DataFrame(rfzn_arr, index=samples, columns=methods, dtype=float)
    nar_tbl  = pd.DataFrame(nar_arr,  index=samples, columns=methods, dtype=float)

    return rfzn_tbl, nar_tbl

# Run evaluation
# flat_windows = [(2500, 2600), (3200, 3500)]
# rfzn, nar = evaluate_all_samples_parallel(final_df, flat_windows)


##### PLOT RFZN and NAR values #####

def plot_metric_boxes(
    df: pd.DataFrame,
    metric_name: str,
    figsize: tuple[int, int] = (9, 5),
    mean_bar_width: float = 0.6,
    color_boxes: str | None = None,
    color_mean: str | None = None,
    save_path: str = '',
) -> None:
    """
    Box-plot of a baseline-quality metric (`RFZN` or `NAR`) across methods.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows = samples, columns = baseline-correction methods.
    metric_name : str
        Title for the y-axis and plot.
    figsize : (w, h), default (9, 5)
        Size of the figure in inches.
    mean_bar_width : float, default 0.6
        Width of the mean ± SD bar overlay (same units as box widths).
    color_boxes : str | None
        Matplotlib colour for the boxes.  *None* → Matplotlib default cycle.
    color_mean : str | None
        Colour for the mean ± SD bars.  *None* → Matplotlib default cycle.
    """
    # Prepare data ──────────────────────────────────────────────────────
    # keep numerical columns only, drop all-NaN cols (failed algorithms)
    df_num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if df_num.empty:
        raise ValueError("No numeric columns to plot.")

    methods = df_num.columns.to_list()
    data    = [df_num[m].dropna().values for m in methods]

    # Create boxplot ────────────────────────────────────────────────────
    plt.figure(figsize=figsize)
    box = plt.boxplot(
        data,
        vert=True,
        patch_artist=True,
        labels=methods,
        showfliers=False,        # hide outliers for clarity
        widths=0.8,
        medianprops=dict(color="black", lw=1.2),
        boxprops=dict(facecolor=color_boxes or "tab:blue", alpha=0.35),
    )
    '''
    # Overlay mean ± SD per method ──────────────────────────────────────
    x_positions = np.arange(1, len(methods) + 1)
    means = [np.mean(d) for d in data]
    sds   = [np.std(d, ddof=1) for d in data]
    
    plt.errorbar(
        x_positions,
        means,
        yerr=sds,
        fmt="o",
        color=color_mean or "tab:red",
        ecolor=color_mean or "tab:red",
        elinewidth=2,
        capsize=4,
        linewidth=2,
        label="mean ± 1 SD",
    )
    '''
    # Cosmetics ─────────────────────────────────────────────────────────
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} distribution across baseline methods")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.savefig(metric_name+'_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
# plot_metric_boxes(rfzn, metric_name="RFZN")
# plot_metric_boxes(nar,  metric_name="NAR")


#### Plot masked samples ####

def plot_metric_boxes_masked(
    df: pd.DataFrame,
    metric_name: str,
    max_value: double,
    figsize: tuple[int, int] = (9, 5),
    mean_bar_width: float = 0.6,
    color_boxes: str | None = None,
    color_mean: str | None = None,
    save_path: str = 'masked_',
) -> None:

    good_mask = (df.fillna(np.inf) <= max_value).all(axis=0) # axis=0  ⇒  column-wise “all”
    good_methods = good_mask.index[good_mask].tolist()
    # print(f"Methods that always pass {metric_name} ≤ {max_value}:", good_methods)
    df_good = df[good_methods]
    plot_metric_boxes(
                      df_good,
                      metric_name,
                      figsize,
                      mean_bar_width,
                      color_boxes,
                      color_mean,
                      save_path
                      )
    

def snr_single(y_corr, x_axis, peak_center, window=10, noise_mask=None):
    """
    SNR from baseline-corrected spectrum.

    Parameters
    ----------
    y_corr      : 1-D np.ndarray  # baseline-corrected intensity
    x_axis      : 1-D np.ndarray  # matching wavenumbers
    peak_center : float           # centre of peak of interest (cm-1)
    window      : float           # ± half-width for peak height (cm-1)
    noise_mask  : bool array or None
        True where baseline-only → compute RMS there.
        If None, construct mask from predefined RFZN flat windows.

    Returns
    -------
    snr : float
    """
    if noise_mask is None:
        raise ValueError("Provide a flat-zone mask for RMS noise.")
    # peak height above baseline (max within ±window)
    idx_peak = (x_axis >= peak_center - window) & (x_axis <= peak_center + window)
    peak_height = y_corr[idx_peak].max()
    # RMS noise in flat zone
    sigma_noise = np.sqrt(np.mean(y_corr[noise_mask]**2))
    return peak_height / sigma_noise


# -------------------------------------------------------------------------
#  TEST BASELINE CORRECTION METHODS
#  ADD: Signal-to-Noise Ratio (SNR)
# -------------------------------------------------------------------------


def _make_mask(x_axis: np.ndarray,
               flat_windows: List[Tuple[float, float]]) -> np.ndarray:
    """Boolean mask for baseline-only wavenumber regions."""
    mask = np.zeros_like(x_axis, dtype=bool)
    for lo, hi in flat_windows:
        mask |= (x_axis >= lo) & (x_axis <= hi)
    return mask


def _score_one_sample(
    sample_name: str,
    y_raw: np.ndarray,
    methods: List[str],
    mask: np.ndarray,
    negative_clip: bool,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker: returns RFZN, NAR, SNR vectors for one spectrum.
    SNR = |peak| / RMS(noise); peak = global max(|y_corr|).
    """
    n_methods = len(methods)
    rfzn_row = np.empty(n_methods, dtype=float)
    nar_row  = np.empty_like(rfzn_row)
    snr_row  = np.empty_like(rfzn_row)

    for j, m in enumerate(methods):
        try:
            y_corr = baseline_correction(
                y_raw, method=m, clip_negative=negative_clip
            )

            # ----------------- noise & peak ---------------------------------
            sigma_noise = np.sqrt(np.mean(y_corr[mask] ** 2))
            peak_height = np.max(np.abs(y_corr))  # global |peak|

            # If you want specific peaks (e.g. 2916 cm-1), replace the line above with:
            # peak_height = max(
            #     np.max(np.abs(y_corr[(x_axis >= c - 10) & (x_axis <= c + 10)]))
            #     for c in diag_peaks_cm1
            # )

            # ----------------- RFZN ----------------------------------------
            rfzn_row[j] = sigma_noise            # identical definition
            # ----------------- NAR  ----------------------------------------
            neg_area = np.sum(-y_corr[y_corr < 0.0])
            nar_row[j] = neg_area / np.sum(np.abs(y_corr))
            # ----------------- SNR  ----------------------------------------
            snr_row[j] = peak_height / sigma_noise if sigma_noise > 0 else np.nan

        except Exception:
            rfzn_row[j] = np.nan
            nar_row[j]  = np.nan
            snr_row[j]  = np.nan

    return sample_name, rfzn_row, nar_row, snr_row


def evaluate_all_samples(
    df_wide: pl.DataFrame,
    flat_windows: List[Tuple[float, float]],
    negative_clip: bool = False,
    n_jobs: int = -1,                # -1 → all CPU cores
):
    """
    Parallel computation of RFZN, NAR, SNR for every (sample, method) pair.

    Returns
    -------
    rfzn_tbl, nar_tbl, snr_tbl : pandas.DataFrame
        index = sample, columns = method
    """
    # ---------------- preparation ----------------------------------------
    methods  = baseline_method_names()            # safe 1-D algorithms
    samples  = df_wide.get_column("sample").to_list()
    wavenums = np.array(
        [float(c) for c in df_wide.columns if c not in ("sample", "label")]
    )
    mask     = _make_mask(wavenums, flat_windows)

    # extract spectra to NumPy once (parent process)
    spectra = {
        s: (df_wide
             .filter(pl.col("sample") == s)
             .select(pl.exclude(["sample", "label"]))
             .to_numpy()
             .ravel())
        for s in samples
    }

    # ---------------- parallel loop --------------------------------------
    worker = delayed(_score_one_sample)
    iterator = (worker(s, spectra[s], methods, mask, negative_clip)
                for s in samples)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        tqdm(iterator, total=len(samples), desc="baseline eval", ncols=88)
    )

    # ---------------- assemble -------------------------------------------
    rfzn_arr = np.vstack([row[1] for row in results])
    nar_arr  = np.vstack([row[2] for row in results])
    snr_arr  = np.vstack([row[3] for row in results])

    rfzn_tbl = pd.DataFrame(rfzn_arr, index=samples, columns=methods, dtype=float)
    nar_tbl  = pd.DataFrame(nar_arr,  index=samples, columns=methods, dtype=float)
    snr_tbl  = pd.DataFrame(snr_arr,  index=samples, columns=methods, dtype=float)

    return rfzn_tbl, nar_tbl, snr_tbl
