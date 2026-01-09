from __future__ import annotations
import re
from typing import Dict, Any, Optional, List, Tuple, Union, Sequence
import warnings
import numpy as np
import polars as pl
import pandas as pd

from .utils.plotting import plot_ftir_spectra, compare_ftir_spectra
from .utils.trans_abs import convert_spectra
from .utils.atmospheric import exclude_and_interpolate_regions
from .utils.baseline import (
                    evaluate_baseline_correction_methods,
                    plot_baseline_correction_metric_boxes,
                    baseline_method_names,
                    apply_baseline_correction,
                    find_best_baseline_method
                    )
from .utils.denoise import (
    denoise_method_names,
    apply_denoising,
    evaluate_denoising_methods,
    plot_denoising_evaluation,
    plot_denoising_evaluation_summary,
    find_best_denoising_method,
)
from .utils.normalization import (
    normalize_method_names,
    normalize_df
)
from .utils.normalization_eval import evaluate_norm_methods

from .utils.derivatives import (
    derivative_batch,
    plot_derivatives
)

from .utils.plottingx import (
    plot_mean_spectra_by_class,
    plot_overlay_mean_spectra,
    plot_coefficient_of_variation,
    plot_spectral_heatmap
)
from .utils.plotting_stats import (
    perform_anova_analysis,
    plot_correlation_matrix
)
from .utils.plotting_dim import (
    perform_pca_analysis,
    perform_tsne_analysis,
    perform_umap_analysis,
    perform_plsda_analysis,
    perform_oplsda_analysis
)
from .utils.plotting_clus import (
    perform_kmeans_clustering,
    perform_hierarchical_clustering
)
from .utils.ml import (
    prepare_data,
    get_all_models,
    evaluate_model,
    evaluate_all_models,
    plot_confusion_matrix,
    calculate_multiclass_metrics,
    plot_model_comparison,
    plot_family_comparison,
    plot_efficiency_analysis,
    plot_overfitting_analysis,
    tune_top_models,
    explain_model_shap,
    plot_shap_decision
)

# CO2 and H2O removal
# ====================================================================
# Define regions
EXCLUDE_REGIONS = [
    (0, 680),       # Exclude everything below 680, CO₂ bending mode, 670 cm-1
    # (1350, 1450),   # Exclude H2O bend region
    # (1250, 1900),  # Exclude H2O bend region
    # (2300, 2400),   # Exclude CO2 stretch region, 2350 cm-1
    (3500, 5000)    # Exclude everything above 3500, O–H stretch region
]

INTERPOLATE_REGIONS = [
    (1250, 2400)    # Interpolate over H2O region
]

FLAT_WINDOWS = [(1800, 1900), (2400, 2700)]

FTIR_BASELINE_METHODS = [
    # Core PLS methods
    'asls', 'aspls', 'arpls', 'airpls', 'iarpls', 'drpls', 'brpls',
    # Classic methods
    'rubberband', 'snip', 'rolling_ball',
    # Morphological
    'mor', 'imor', 'mormol',
    # Polynomial
    'modpoly', 'imodpoly', 'penalized_poly'
]

FTIR_DNOISING_METHODS = [
'savgol', 'wavelet', 'moving_average', 'gaussian', 'median', 'whittaker', 'lowpass'
]

FTIR_NORMALIZATION_METHODS = [
    'adaptive_regional', 'area', 'curvature_weighted', 'derivative_ratio',
    'detrend', 'max', 'minmax', 'peak', 'pqn', 'range', 'snv', 'robust_snv',
    'snv_detrend', 'signal_to_baseline', 'spectral_moments', 'total_variation',
    'vector',
    # 'entropy_weighted', # slow 98 sec
    # 'peak_envelope', # slow 78 sec
]


class FTIRdataprocessing:
    """
    FTIR Data Processing Pipeline Class

    A comprehensive wrapper class for FTIR spectral data preprocessing and analysis workflows.
    Provides a unified interface for common FTIR data processing operations including:
    - Transmittance/Absorbance conversion
    - Atmospheric interference removal (CO2, H2O)
    - Baseline correction
    - Denoising
    - Normalization
    - Spectral derivatives
    - Visualization and comparison

    This class streamlines multi-step preprocessing pipelines by maintaining state between
    operations and providing sensible defaults for FTIR-specific workflows.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input FTIR spectral data with samples as rows and wavenumbers as columns.
        Must include a label column for sample identification.

    label_column : str, default="label"
        Name of the column containing sample labels/identifiers.

    exclude_columns : list of str, optional
        Additional non-spectral columns to exclude from processing (e.g., metadata).

    wn_min : float, optional
        Minimum wavenumber (cm⁻¹) for filtering spectral range.

    wn_max : float, optional
        Maximum wavenumber (cm⁻¹) for filtering spectral range.

    exclude_regions : list of tuples, default=EXCLUDE_REGIONS
        Wavenumber ranges to completely exclude from analysis.
        Default removes regions affected by CO2 and H2O.

    interpolate_regions : list of tuples, default=INTERPOLATE_REGIONS
        Wavenumber ranges to interpolate over (e.g., H2O bending region).

    flat_windows : list of tuples, default=FLAT_WINDOWS
        Wavenumber windows expected to be flat (used for baseline quality metrics).

    baseline_methods : list of str, default=FTIR_BASELINE_METHODS
        Baseline correction methods to evaluate.

    denoising_methods : list of str, default=FTIR_DNOISING_METHODS
        Denoising methods to evaluate.

    sample_selection : str, default="random"
        Strategy for selecting samples during evaluation ("random", "stratified", etc.).

    random_state : int, optional
        Random seed for reproducible sample selection.

    n_jobs : int, default=-1
        Number of parallel jobs for multi-sample processing (-1 uses all cores).

    Attributes
    ----------
    df : DataFrame
        Original input data
    converted_df : DataFrame
        Data after transmittance/absorbance conversion
    df_atm : DataFrame
        Data after atmospheric interference removal
    df_corr : DataFrame
        Data after baseline correction
    df_denoised : DataFrame
        Data after denoising
    df_norm : DataFrame
        Data after normalization
    df_deriv : DataFrame
        Data after derivative calculation
    rfzn_tbl : DataFrame
        Baseline correction evaluation: Residual Flatness in Zero Noise regions
    nar_tbl : DataFrame
        Baseline correction evaluation: Negative Absorbance Ratio
    snr_tbl : DataFrame
        Baseline correction evaluation: Signal-to-Noise Ratio
    denoising_results : DataFrame
        Denoising method evaluation results
    norm_eval_results : DataFrame
        Normalization method evaluation results

    Examples
    --------
    >>> # Basic FTIR preprocessing workflow
    >>> import pandas as pd
    >>> from xpectrass import FTIRdataprocessing
    >>>
    >>> # Load FTIR data
    >>> df = pd.read_csv("ftir_data.csv", index_col=0)
    >>>
    >>> # Initialize processing pipeline
    >>> ftir = FTIRdataprocessing(
    ...     df,
    ...     label_column="label",
    ...     wn_min=400,
    ...     wn_max=4000
    ... )
    >>>
    >>> # Step 1: Convert to absorbance
    >>> ftir.convert(mode="to_absorbance", plot=True)
    >>>
    >>> # Step 2: Remove atmospheric interference
    >>> ftir.exclude_interpolate(method="spline", plot=True)
    >>>
    >>> # Step 3: Find best baseline method
    >>> ftir.find_baseline_method(n_samples=50, plot=True)
    >>>
    >>> # Step 4: Apply baseline correction
    >>> ftir.correct_baseline(method="asls", plot=True)
    >>>
    >>> # Step 5: Find best denoising method
    >>> ftir.find_denoising_method(n_samples=50, plot=True)
    >>>
    >>> # Step 6: Apply denoising
    >>> ftir.denoise_spect(method="savgol")
    >>>
    >>> # Step 7: Normalize
    >>> ftir.normalize(method="snv")
    >>>
    >>> # Step 8: Compare all processing stages
    >>> ftir.plot_multiple_spec(sample="Sample1")

    Notes
    -----
    - All intermediate results are stored as class attributes for easy access
    - Most methods have a `plot` parameter for immediate visualization
    - The pipeline is flexible - steps can be skipped or reordered as needed
    - All methods use spectral_utils for robust wavenumber column handling
    """

    def __init__(
        self,
        df,
        label_column: str = "type",
        exclude_columns: Optional[List[str]] = None,
        wn_min: Optional[float] = None,
        wn_max: Optional[float] = None,
        denoising_methods: Optional[List[str]] = FTIR_DNOISING_METHODS,
        flat_windows: List[Tuple[float, float]] = FLAT_WINDOWS,
        baseline_methods: Optional[List[str]] = FTIR_BASELINE_METHODS,
        exclude_regions = EXCLUDE_REGIONS,
        interpolate_regions = INTERPOLATE_REGIONS,
        normalization_methods: Optional[List[str]] = FTIR_NORMALIZATION_METHODS,
        sample_selection: str = "random",
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        self.df = df
        self.label_column = label_column
        self.exclude_columns = exclude_columns
        self.wn_min = wn_min
        self.wn_max = wn_max
        self.denoising_methods = denoising_methods
        self.flat_windows = flat_windows
        self.baseline_methods = baseline_methods
        self.exclude_regions = exclude_regions
        self.interpolate_regions = interpolate_regions
        self.normalization_methods = normalization_methods
        self.sample_selection = sample_selection
        self.random_state = random_state
        self.n_jobs = n_jobs

    def plot(
        self,
        data: Union[pd.DataFrame, pl.DataFrame] = None,
        samples: Union[str, Sequence[str]] = None,
        invert_x: bool = True,
        figsize: tuple = (7, 4),
        show_legend: bool = True,
        color_by_group: bool = False,
        x_min: float = None,
        x_max: float = None,
        mode: str = "auto",
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot FTIR spectra from the current dataset.

        Convenience wrapper around plot_ftir_spectra() that uses the instance's
        configuration (label_column, exclude_columns, wn_min/wn_max).

        Parameters
        ----------
        samples : str or list of str, optional
            Sample name(s) to plot. If None, plots all samples.

        invert_x : bool, default=True
            If True, inverts x-axis (4000 on left, 400 on right) for standard FTIR display.

        figsize : tuple, default=(7, 4)
            Figure size in inches (width, height).

        show_legend : bool, default=True
            Whether to display the legend.

        color_by_group : bool, default=False
            If True, colors spectra by their group (label_column value).

        x_min : float, optional
            Minimum wavenumber for display zoom (does not filter data).

        x_max : float, optional
            Maximum wavenumber for display zoom (does not filter data).

        mode : str, default="auto"
            Display mode: "auto", "transmittance", or "absorbance".
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot. Required if save_plot is True.

        Examples
        --------
        >>> # Plot all samples
        >>> ftir.plot()
        >>>
        >>> # Plot specific sample
        >>> ftir.plot(samples="Sample1")
        >>>
        >>> # Plot multiple samples with zoom
        >>> ftir.plot(samples=["Sample1", "Sample2"], x_min=1000, x_max=2000)
        """
        if data is None:
            data = self.df
        
        
        plot_ftir_spectra(
            data = data,
            samples = samples,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max,
            invert_x = invert_x,
            figsize = figsize,
            show_legend = show_legend,
            color_by_group = color_by_group,
            x_min = x_min,
            x_max = x_max,
            mode = mode,
            save_plot = save_plot,
            save_path = save_path,
        )

    def convert(
        self,
        data: Union[pd.DataFrame, pl.DataFrame] = None,
        mode: str = "auto",
        plot: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None
    ):
        """
        Convert between transmittance and absorbance spectra.

        Applies Beer-Lambert law conversion: A = -log₁₀(T/100)
        Results are stored in self.converted_df.

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data to convert. If None, uses self.df.

        mode : str, default="auto"
            Conversion mode:
            - "auto": Auto-detect current mode and leave as-is
            - "to_absorbance": Convert to absorbance
            - "to_transmittance": Convert to transmittance

        plot : bool, default=True
            If True, plots the converted spectra.
        
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot. Required if save_plot is True.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Converted spectral data (also stored in self.converted_df).

        Examples
        --------
        >>> # Convert transmittance to absorbance
        >>> ftir.convert(mode="to_absorbance")
        >>>
        >>> # Convert without plotting
        >>> df_abs = ftir.convert(mode="to_absorbance", plot=False)

        Notes
        -----
        - Transmittance values should be in range [0, 100]
        - Absorbance values are typically in range [0, ~3]
        - Invalid values (T <= 0 or T > 100) are handled gracefully
        """
        # Use self.df if no data provided
        if data is None:
            data = self.df

        # Apply conversion using spectral_utils
        converted_df = convert_spectra(
            data = data,
            mode = mode,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
        )

        # Store result for later use
        self.converted_df = converted_df

        # Optionally plot the converted spectra
        if plot:
            print(f'{"#"*10} Plotting Converted Spectra! {"#"*10}')
            plot_ftir_spectra(
                data = self.converted_df,
                samples = None,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                invert_x = True,
                figsize = (7, 4),
                show_legend = True,
                color_by_group = True,
                x_min = None,
                x_max = None,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path,
            )
        return self.converted_df


    def find_denoising_method(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame"] = None,
        methods: Optional[List[str]] = "FTIR",
        n_samples: Optional[int] = 50,
        sample_selection: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        plot: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        """
        Evaluate multiple denoising methods.

        Tests different denoising algorithms and computes quality metrics including:
        - SNR: Signal-to-Noise Ratio
        - RMSE: Root Mean Squared Error
        - Peak preservation metrics

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_corr.

        methods : list of str or "FTIR", default="FTIR"
            Denoising methods to evaluate. "FTIR" uses self.denoising_methods_.

        n_samples : int, default=50
            Number of samples to evaluate.

        sample_selection : str, optional
            Sample selection strategy. If None, uses self.sample_selection.

        random_state : int, optional
            Random seed. If None, uses self.random_state.

        n_jobs : int, optional
            Parallel jobs. If None, uses self.n_jobs.

        plot : bool, default=True
            If True, plots evaluation results and summary.
        save_plot : bool, default=False
            If True, saves the plots to file.
        save_path : str, optional
            File path to save the plots. Required if save_plot is True.

        Returns
        -------
        pd.DataFrame
            Evaluation results for each method (also stored in self.denoising_results).

        Examples
        --------
        >>> # Evaluate all FTIR denoising methods
        >>> ftir.find_denoising_method(n_samples=50)
        >>>
        >>> # Evaluate specific methods
        >>> ftir.find_denoising_method(methods=['savgol', 'wavelet', 'gaussian'])

        Notes
        -----
        - Higher SNR = better noise reduction (better)
        - Lower RMSE = less signal distortion (better)
        - Balance between noise reduction and feature preservation is key
        """
        # Use baseline-corrected data by default
        if data is None:
            data = self.converted_df

        # Use instance defaults
        if methods == "FTIR":
            methods = self.denoising_methods
        else:
            methods = None
        if sample_selection is None:
            sample_selection = self.sample_selection
        if random_state is None:
            random_state = self.random_state
        if n_jobs is None:
            n_jobs = self.n_jobs

        # Evaluate all denoising methods
        denoising_results = evaluate_denoising_methods(
            data = data,
            methods = methods,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max,
            n_samples = n_samples,
            sample_selection = sample_selection,
            random_state = random_state,
            n_jobs = n_jobs,
        )

        # Store results
        self.denoising_results = denoising_results

        # Plot evaluation results
        if plot:
            print(f'{"#"*10} Plotting Denoising Evaluation! {"#"*10}')
            plot_denoising_evaluation(
                eval_df = self.denoising_results,
                metrics = None,
                figsize = (14, 5),
                show_mean_sd = False,
                save_plot = save_plot,
                save_path = save_path
            )
            print(f'{"#"*10} Plotting Denoising Evaluation Summary! {"#"*10}')
            plot_denoising_evaluation_summary(
                eval_df = self.denoising_results,
                figsize = (10, 6),
                save_plot = save_plot,
                save_path = save_path
            )

        return self.denoising_results

    def best_denoising_methods(
            self,
            eval_df: pd.DataFrame = None,
            snr_min: float = 10,
            smoothness_min: float = 1e3,
            fidelity_min: float = 0.9,
            time_max_ms: float = 100.0,
            top_n: int = 7
        ):
        if eval_df is None:
            eval_df = self.denoising_results
        result = find_best_denoising_method(
                    eval_df = eval_df,
                    snr_min = snr_min,
                    smoothness_min = smoothness_min,
                    fidelity_min = fidelity_min,
                    time_max_ms = time_max_ms,
                    top_n = top_n,
            )
        return result

    def plot_denoising_eval(
        self,
        eval_df: pd.DataFrame = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 5),
        show_mean_sd: bool = False,
        plot_summary: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot denoising method evaluation results.

        Creates visualizations of denoising performance metrics across methods.

        Parameters
        ----------
        eval_df : pd.DataFrame, optional
            Evaluation results. If None, uses self.denoising_results.

        metrics : list of str, optional
            Specific metrics to plot. If None, plots all available metrics.

        figsize : tuple, default=(14, 5)
            Figure size in inches (width, height).

        show_mean_sd : bool, default=False
            If True, shows mean and standard deviation in plots.

        plot_summary : bool, default=True
            If True, also creates a summary plot.

        save_path : str, default=""
            Path to save figures. If empty, doesn't save.

        Examples
        --------
        >>> # Plot all metrics
        >>> ftir.plot_denoising_eval()
        >>>
        >>> # Plot specific metrics only
        >>> ftir.plot_denoising_eval(metrics=['SNR', 'RMSE'])
        """
        # Use stored results if not provided
        if eval_df is None:
            eval_df = self.denoising_results

        # Plot detailed evaluation
        plot_denoising_evaluation(
            eval_df = eval_df,
            metrics = metrics,
            figsize = figsize,
            show_mean_sd = show_mean_sd,
            save_plot= save_plot,
            save_path = save_path
        )

        # Optionally plot summary
        if plot_summary:
            plot_denoising_evaluation_summary(
                eval_df = eval_df,
                figsize = (10, 6),
                save_plot= save_plot,
                save_path = save_path
            )

    def denoising_methods_available(self):
        """
        Print available denoising methods.

        Displays a list of all supported denoising algorithms.

        Examples
        --------
        >>> ftir.denoising_methods()
        Available methods for denoising: ['savgol', 'wavelet', ...]
        """
        denoise_names = denoise_method_names()
        print("Available methods for denoising: ", denoise_names)


    def denoise_spect(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame"] = None,
        method: str = "savgol",
        plot: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Apply denoising to FTIR spectra.

        Reduces noise while preserving spectral features. Results are
        stored in self.df_denoised.

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_corr.

        method : str, default="savgol"
            Denoising method. See denoising_methods() for options.
            Common choices:
            - "savgol": Savitzky-Golay filter (preserves peaks well)
            - "wavelet": Wavelet denoising (good for complex spectra)
            - "gaussian": Gaussian smoothing (simple, fast)
            - "median": Median filter (robust to outliers)

        **kwargs
            Method-specific parameters. Examples:
            - savgol: window_length=11, polyorder=3
            - wavelet: wavelet='db4', level=3
            - gaussian: sigma=2

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Denoised data (also stored in self.df_denoised).

        Examples
        --------
        >>> # Apply Savitzky-Golay denoising
        >>> ftir.denoise_spect(method="savgol", window_length=11)
        >>>
        >>> # Apply wavelet denoising
        >>> ftir.denoise_spect(method="wavelet", wavelet="db4", level=3)

        Notes
        -----
        Use find_denoising_method() first to identify the best method for your data.
        """
        # Use baseline-corrected data by default
        if data is None:
            data = self.converted_df

        # Apply denoising
        df_denoised = apply_denoising(
            data = data,
            method = method,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max,
            **kwargs,
        )

        # Store result
        self.df_denoised = df_denoised

        if plot:
            print(f'{"#"*10} Plotting Denoised Spectra! {"#"*10}')
            plot_ftir_spectra(
                data = self.df_denoised,
                samples = None,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                invert_x = True,
                figsize = (7, 4),
                show_legend = True,
                color_by_group = True,
                x_min = None,
                x_max = None,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path,
            )
        return self.df_denoised


    def find_baseline_method(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame"] = None,
        flat_windows: List[Tuple[float, float]] = None,
        negative_clip: bool = False,
        diagnostic_peaks: Optional[List[Tuple[float, float]]] = None,
        baseline_methods: Optional[List[str]] = "FTIR",
        n_samples: Optional[int] = 50,
        sample_selection: str = None,
        random_state: Optional[int] = None,
        n_jobs: int = None,
        plot: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        """
        Evaluate multiple baseline correction methods.

        Tests different baseline correction algorithms and computes quality metrics:
        - RFZN: Residual Flatness in Zero-Noise regions
        - NAR: Negative Absorbance Ratio (lower is better)
        - SNR: Signal-to-Noise Ratio (higher is better)

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_atm.

        flat_windows : list of tuples, optional
            Wavenumber ranges expected to be flat (for RFZN metric).
            If None, uses self.flat_windows.

        negative_clip : bool, default=False
            If True, clips negative values to zero after baseline correction.

        diagnostic_peaks : list of tuples, optional
            Peak regions for SNR calculation.

        baseline_methods : list of str or "FTIR", default="FTIR"
            Methods to evaluate. "FTIR" uses self.baseline_methods.

        n_samples : int, default=50
            Number of samples to evaluate (for computational efficiency).

        sample_selection : str, optional
            Sample selection strategy. If None, uses self.sample_selection.

        random_state : int, optional
            Random seed. If None, uses self.random_state.

        n_jobs : int, optional
            Parallel jobs. If None, uses self.n_jobs.

        plot : bool, default=True
            If True, plots boxplots for all three metrics.
        save_plot : bool, default=False
            If True, saves the plots to file.
        save_path : str, optional
            File path to save the plots. Required if save_plot is True.

        Returns
        -------
        tuple of (rfzn_tbl, nar_tbl, snr_tbl)
            DataFrames containing evaluation metrics for each method.

        Examples
        --------
        >>> # Evaluate all FTIR baseline methods
        >>> ftir.find_baseline_method(n_samples=50)
        >>>
        >>> # Evaluate specific methods
        >>> ftir.find_baseline_method(
        ...     baseline_methods=['asls', 'arpls', 'rubberband'],
        ...     n_samples=100
        ... )

        Notes
        -----
        - Lower RFZN = flatter baseline in noise-free regions (better)
        - Lower NAR = fewer negative absorbance values (better)
        - Higher SNR = better signal preservation (better)
        - Use best_baseline_method() to get recommendations based on these metrics
        """
        # Suppress warnings in this function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Use atmospheric-corrected data by default
            if data is None:
                data = self.df_denoised

            # Use instance defaults if not provided
            if flat_windows is None:
                flat_windows = self.flat_windows
            if sample_selection is None:
                sample_selection = self.sample_selection
            if baseline_methods == "FTIR":
                baseline_methods = self.baseline_methods
            if random_state is None:
                random_state = self.random_state
            if n_jobs is None:
                n_jobs = self.n_jobs

            # Evaluate all baseline methods
            rfzn_tbl, nar_tbl, snr_tbl = evaluate_baseline_correction_methods(
                data= data,
                flat_windows = flat_windows,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                negative_clip = negative_clip,
                diagnostic_peaks = diagnostic_peaks,
                baseline_methods = baseline_methods,
                n_samples = n_samples,
                sample_selection = sample_selection,
                random_state = random_state,
                n_jobs = n_jobs,
            )

            # Store results for later analysis
            self.rfzn_tbl, self.nar_tbl, self.snr_tbl = rfzn_tbl, nar_tbl, snr_tbl

            # Plot all three metrics
            if plot:
                print(f'{"#"*5} Plotting Residual Flatness in Zero-Noise (RFZN) - lower is better ! {"#"*5}')
                plot_baseline_correction_metric_boxes(
                    df = self.rfzn_tbl,
                    metric_name = "RZFN",
                    figsize = (15, 5),
                    mean_bar_width = 0.5,
                    save_plot = save_plot,
                    save_path = save_path,
                )
                print(f'{"#"*5} Plotting Negative Absorbance Ratio (NAR) - lower is better! {"#"*5}')
                plot_baseline_correction_metric_boxes(
                    df = self.nar_tbl,
                    metric_name = "NAR",
                    figsize = (15, 5),
                    mean_bar_width = 0.5,
                    save_plot = save_plot,
                    save_path = save_path,
                )
                print(f'{"#"*5} Plotting Signal-to-Noise Ratio (SNR) - higher is better! {"#"*5}')
                plot_baseline_correction_metric_boxes(
                    df = self.snr_tbl,
                    metric_name = "SNR",
                    figsize = (15, 5),
                    mean_bar_width = 0.5,
                    save_plot = save_plot,
                    save_path = save_path,
                )
            return self.rfzn_tbl, self.nar_tbl, self.snr_tbl

    def plot_rfzn_nar_snr(
        self,
        df: pd.DataFrame = None,
        metric_name: str = "RFZN",
        figsize: tuple[int, int] = (15, 5),
        mean_bar_width: float = 0.5,
        save_plot: bool = False,
        save_path: str = "",
    ):
        """
        Plot baseline correction evaluation metrics.

        Creates boxplot visualizations for RFZN, NAR, or SNR metrics
        across different baseline correction methods.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Metric data to plot. If None, automatically selects based on metric_name.

        metric_name : str, default="RFZN"
            Metric to plot: "RZFN", "NAR", or "SNR".

        figsize : tuple, default=(15, 5)
            Figure size in inches (width, height).

        mean_bar_width : float, default=0.5
            Width of mean value bars in the plot.

        save_path : str, default=""
            Path to save figure. If empty, doesn't save.

        Examples
        --------
        >>> # Plot RFZN metric
        >>> ftir.plot_rfzn_nar_snr(metric_name="RZFN")
        >>>
        >>> # Plot NAR metric with custom size
        >>> ftir.plot_rfzn_nar_snr(metric_name="NAR", figsize=(20, 6))
        """
        # Validate metric name
        if metric_name not in ["RZFN", "NAR", "SNR"]:
            raise ValueError(" only allowed matric names are 'RZFN', 'NAR' and S'NR', please enter the correct one!")

        # Auto-select the correct table if not provided
        if df is None and metric_name == "RZFN":
            df = self.rfzn_tbl
        if df is None and metric_name == "NAR":
            df = self.nar_tbl
        if df is None and metric_name == "SNR":
            df = self.snr_tbl

        # Create the plot
        plot_baseline_correction_metric_boxes(
            df = df,
            metric_name = metric_name,
            figsize = figsize,
            mean_bar_width = mean_bar_width,
            save_plot = save_plot,
            save_path = save_path,
        )

    def best_baseline_method(
        self,
        rfzn_tbl: pd.DataFrame = None,
        nar_tbl: pd.DataFrame = None,
        snr_tbl: pd.DataFrame = None,
        rfzn_threshold: float = 0.01,
        nar_threshold: float = 0.05,
        snr_min: float = 10.0,
        top_n: int = 5,
    ):
        """
        Recommend best baseline correction methods based on evaluation metrics.

        Analyzes RFZN, NAR, and SNR metrics to identify the best-performing
        baseline correction methods for your data.

        Parameters
        ----------
        rfzn_tbl : pd.DataFrame, optional
            RFZN metric table. If None, uses self.rfzn_tbl.

        nar_tbl : pd.DataFrame, optional
            NAR metric table. If None, uses self.nar_tbl.

        snr_tbl : pd.DataFrame, optional
            SNR metric table. If None, uses self.snr_tbl.

        rfzn_threshold : float, default=0.01
            Maximum acceptable RFZN value (flatness requirement).

        nar_threshold : float, default=0.05
            Maximum acceptable NAR value (negative absorbance tolerance).

        snr_min : float, default=10.0
            Minimum acceptable SNR value.

        top_n : int, default=5
            Number of top methods to recommend.

        Returns
        -------
        DataFrame
            Ranked recommendations with scores for each method.

        Examples
        --------
        >>> # Get top 5 recommendations
        >>> recommendations = ftir.best_baseline_method()
        >>>
        >>> # Get top 3 with stricter criteria
        >>> recommendations = ftir.best_baseline_method(
        ...     rfzn_threshold=0.005,
        ...     nar_threshold=0.01,
        ...     top_n=3
        ... )
        """
        # Use stored tables if not provided
        if rfzn_tbl is None:
            rfzn_tbl = self.rfzn_tbl

        if nar_tbl is None:
            nar_tbl = self.nar_tbl

        if snr_tbl is None:
            snr_tbl = self.snr_tbl

        # Get recommendations based on all three metrics
        results = find_best_baseline_method(
            rfzn_tbl = rfzn_tbl,
            nar_tbl = nar_tbl,
            snr_tbl = snr_tbl,
            rfzn_threshold = rfzn_threshold,
            nar_threshold = nar_threshold,
            snr_min = snr_min,
            top_n = top_n,
        )
        return results


    def baseline_correction_methods_available(self):
        """
        Print available baseline correction methods.

        Displays a list of all supported baseline correction algorithms.

        Examples
        --------
        >>> ftir.baselines()
        Available methods for baseline correction: ['asls', 'arpls', ...]
        """
        base_names = baseline_method_names()
        print("Available methods for baseline correction: ", base_names)


    def correct_baseline(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame"] = None,
        method: str = "asls",
        window_size: int = 101,
        poly_order: int = 4,
        clip_negative: bool = False,
        plot: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Apply baseline correction to FTIR spectra.

        Removes baseline drift using the specified algorithm. Results are
        stored in self.df_corr.

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_atm.

        method : str, default="asls"
            Baseline correction method. See baselines() for options.
            Common choices:
            - "asls": Asymmetric Least Squares (fast, robust)
            - "arpls": Adaptive iteratively Reweighted Penalized Least Squares
            - "rubberband": Rubberband baseline

        window_size : int, default=101
            Window size for windowed methods (must be odd).

        poly_order : int, default=4
            Polynomial order for polynomial-based methods.

        clip_negative : bool, default=False
            If True, clips negative values to zero after correction.

        plot : bool, default=True
            If True, plots the corrected spectra.
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot. Required if save_plot is True.

        **kwargs
            Additional method-specific parameters.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Baseline-corrected data (also stored in self.df_corr).

        Examples
        --------
        >>> # Apply ASLS baseline correction
        >>> ftir.correct_baseline(method="asls")
        >>>
        >>> # Apply arPLS with custom parameters
        >>> ftir.correct_baseline(method="arpls", lam=1e5)

        Notes
        -----
        Use find_baseline_method() first to identify the best method for your data.
        """
        # Use atmospheric-corrected data by default
        if data is None:
            data = self.df_denoised

        # Apply baseline correction
        df_corr = apply_baseline_correction(
            data = data,
            method = method,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max,
            window_size = window_size,
            poly_order = poly_order,
            clip_negative = clip_negative,
            **kwargs
        )

        # Store result
        self.df_corr = df_corr

        # Optionally plot the corrected spectra
        if plot:
            print(f'{"#"*10} Plotting Baseline Corrected Spectra! {"#"*10}')
            plot_ftir_spectra(
                data = self.df_corr,
                samples = None,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                invert_x = True,
                figsize = (7, 4),
                show_legend = True,
                color_by_group = True,
                x_min = None,
                x_max = None,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path,
            )
        return self.df_corr


    def exclude_interpolate(
            self,
            data: Union[pd.DataFrame, "pl.DataFrame"] = None,
            method: str = "interpolate",
            plot: bool = True,
            save_plot: Optional[bool] = False,
            save_path: Optional[str] = None,
    ):
        """
        Remove atmospheric interference regions (CO2, H2O).

        Excludes problematic wavenumber ranges and interpolates over others.
        This is critical for FTIR data as atmospheric CO2 and H2O create strong
        interference peaks that can mask sample features.

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.converted_df.

        method : str, default="interpolate"
            Interpolation method for INTERPOLATE_REGIONS.
            Options: "interpolate", "spline", "reference", "zero", "exclude"

        plot : bool, default=True
            If True, plots the processed spectra.
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot. Required if save_plot is True.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Data with atmospheric regions removed (also stored in self.df_atm).

        Examples
        --------
        >>> # Remove atmospheric interference
        >>> ftir.exclude_interpolate(method="spline")
        >>>
        >>> # Use linear interpolation
        >>> ftir.exclude_interpolate(method="linear")

        Notes
        -----
        Default settings remove:
        - Below 680 cm⁻¹ (CO2 bending)
        - 2300-2400 cm⁻¹ (CO2 stretching)
        - Above 3500 cm⁻¹ (O-H stretching)

        Default settings interpolate:
        - 1250-1900 cm⁻¹ (H2O bending region)
        """
        # Use converted data if available, otherwise use raw data
        if data is None:
            data = self.df_corr

        # Apply atmospheric correction
        df_atm = exclude_and_interpolate_regions(
            data = data,
            exclude_ranges = self.exclude_regions,
            interpolate_ranges = self.interpolate_regions,
            method = method,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max
        )

        # Store result
        self.df_atm = df_atm

        # Optionally plot the result
        if plot:
            print(f'{"#"*10} Plotting Atmospheric Interference Corrected Spectra! {"#"*10}')
            plot_ftir_spectra(
                data = self.df_atm,
                samples = None,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                invert_x = True,
                figsize = (7, 4),
                show_legend = True,
                color_by_group = True,
                x_min = None,
                x_max = None,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path,
            )
        return self.df_atm


    def normalization_methods_available(self):
        """
        Print available normalization methods.

        Displays a list of all supported normalization algorithms.

        Examples
        --------
        >>> ftir.normalization_methods()
        Available methods for normalization: ['minmax', 'snv', 'mean_center', ...]
        """
        norm = normalize_method_names()
        print("Available methods for normalization: ", norm)

    def find_normalization_method(
        self,
        data: Union[pd.DataFrame, pl.DataFrame] = None,
        methods: Optional[List[str]] = "FTIR",
        method_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        n_splits: int = 5,
        n_clusters: Optional[int] = None,
        cluster_bootstrap_rounds: int = 30,
        cluster_bootstrap_frac: float = 0.8,
        compute_internal_cluster_metrics: bool = True,
        n_jobs: int = -1,
    ):
        """
        Evaluate multiple normalization methods.

        Tests different normalization algorithms and computes quality metrics including:
        - Supervised metrics (F1 score, balanced accuracy via cross-validation)
        - Clustering metrics (ARI, NMI vs labels, stability)
        - Internal clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
        - Within-class spectral angle (using label column)
        - Computation time

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_denoised (or self.df_corr if denoised not available).
            Spectral columns are auto-detected as numeric column names.

        methods : list of str, optional
            Normalization methods to evaluate. If None, uses common methods:
            ['snv', 'vector', 'area', 'minmax', 'max', 'robust_snv', 'pqn']

        method_kwargs_map : dict, optional
            Method-specific kwargs. Example:
            {"minmax": {"feature_range": (0.0, 1.0)}, "pqn": {"pqn_reference_type": "median"}}

        n_splits : int, default=5
            Number of cross-validation splits for StratifiedKFold.

        n_clusters : int, optional
            Number of clusters for clustering evaluation. If None, uses number of unique labels.

        cluster_bootstrap_rounds : int, default=30
            Number of bootstrap rounds for cluster stability evaluation.

        cluster_bootstrap_frac : float, default=0.8
            Fraction of data to subsample per bootstrap round.

        compute_internal_cluster_metrics : bool, default=True
            Whether to compute internal clustering metrics.

        n_jobs : int, default=-1
            Number of parallel jobs for evaluating methods.
            -1 means using all processors (recommended). 1 means no parallelization.

        Returns
        -------
        pd.DataFrame
            Evaluation results for each method, sorted by combined score (also stored in self.norm_eval_results).

        Examples
        --------
        >>> # Evaluate common normalization methods
        >>> ftir.find_normalization_method()
        >>>
        >>> # Evaluate specific methods with custom parameters
        >>> ftir.find_normalization_method(
        ...     methods=['snv', 'vector', 'minmax', 'pqn'],
        ...     method_kwargs_map={'minmax': {'feature_range': (0, 1)}}
        ... )

        Notes
        -----
        - Higher supervised_macro_f1 and supervised_bal_acc = better classification performance
        - Higher cluster_ARI and cluster_NMI = better agreement with true labels
        - Higher cluster_stability = more consistent clustering across subsamples
        - Lower within_group_mean_SAM = more consistent technical replicates (if applicable)
        - Combined score is computed as z-score sum of key metrics
        """
        # Use atmosphere corrected data if available, otherwise baseline-corrected
        if data is None:
            if hasattr(self, 'df_denoised') and self.df_atm is not None:
                data = self.df_atm
            elif hasattr(self, 'df_corr') and self.df_corr is not None:
                data = self.df_corr
            else:
                raise ValueError("No processed data available. Please run baseline correction or denoising first.")

        # Convert polars to pandas if needed
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        if methods == "FTIR":
            methods = self.normalization_methods
    
        # Evaluate normalization methods
        norm_eval_results = evaluate_norm_methods(
            df=data,
            methods=methods,
            method_kwargs_map=method_kwargs_map,
            label_column=self.label_column,
            exclude_columns=self.exclude_columns,
            wn_min=self.wn_min,
            wn_max=self.wn_max,
            n_splits=n_splits,
            random_state=self.random_state,
            n_clusters=n_clusters,
            cluster_bootstrap_rounds=cluster_bootstrap_rounds,
            cluster_bootstrap_frac=cluster_bootstrap_frac,
            compute_internal_cluster_metrics=compute_internal_cluster_metrics,
            n_jobs=n_jobs,
        )

        # Store results
        self.norm_eval_results = norm_eval_results
        # print(self.norm_eval_results.head(5))
        return self.norm_eval_results

    def normalize(
        self,
        data: Union[pd.DataFrame, pl.DataFrame, np.ndarray] = None,
        method: str = "mean_center",
        plot: bool = True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """
        Normalize FTIR spectra.

        Applies normalization/scaling to make spectra comparable. Results are
        stored in self.df_norm.

        Parameters
        ----------
        data : pd.DataFrame, pl.DataFrame, or np.ndarray, optional
            Input data. If None, uses self.df_denoised.

        method : str, default="mean_center"
            Normalization method. See normalization_methods() for options.
            Common choices:
            - "mean_center": Center to zero mean (for PCA)
            - "snv": Standard Normal Variate (removes scatter)
            - "minmax": Scale to [0, 1] range
            - "vector": Vector normalization (unit length)
            - "area": Area normalization (constant integral)

        **kwargs
            Method-specific parameters.

        Returns
        -------
        pd.DataFrame, pl.DataFrame, or np.ndarray
            Normalized data (also stored in self.df_norm).

        Examples
        --------
        >>> # Apply SNV normalization
        >>> ftir.normalize(method="snv")
        >>>
        >>> # Apply min-max scaling
        >>> ftir.normalize(method="minmax")

        Notes
        -----
        - SNV is recommended for FTIR to remove multiplicative scatter
        - Mean centering is required before PCA
        - Area normalization is useful for quantitative analysis
        """
       # Use atmosphere corrected data if available, otherwise baseline-corrected
        if data is None:
            if hasattr(self, 'df_denoised') and self.df_atm is not None:
                data = self.df_atm
            elif hasattr(self, 'df_corr') and self.df_corr is not None:
                data = self.df_corr
            else:
                raise ValueError("No processed data available. Please run baseline correction or denoising first.")


        # Apply normalization
        df_norm = normalize_df(
            data = data,
            method = method,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max,
            show_progress = True,
            **kwargs
        )
        # Store result
        self.df_norm = df_norm

        if plot:
            print(f'{"#"*10} Plotting Normalized Spectra! {"#"*10}')
            plot_ftir_spectra(
                data = self.df_norm,
                samples = None,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                invert_x = True,
                figsize = (7, 4),
                show_legend = True,
                color_by_group = True,
                x_min = None,
                x_max = None,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path,
            )

        return self.df_norm

    def derivatives(
        self,
        data: Union[pd.DataFrame, pl.DataFrame, np.ndarray] = None,
        order: int = 1,
        window_length: int = 15,
        polyorder: int = 3,
        delta: float = 1.0,
        plot: bool= True,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        """
        Calculate spectral derivatives.

        Computes 1st or 2nd derivative spectra using Savitzky-Golay differentiation.
        Derivatives enhance spectral resolution and can reveal overlapping peaks.

        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_norm.

        order : int, default=1
            Derivative order (1 for first derivative, 2 for second derivative).

        window_length : int, default=15
            Window size for Savitzky-Golay filter (must be odd).

        polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter.

        delta : float, default=1.0
            Spacing between data points (wavenumber step).

        plot : bool, default=True
            If True, plots the derivative spectra.
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot. Required if save_plot is True.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Derivative spectra (also stored in self.df_deriv).

        Examples
        --------
        >>> # Calculate 1st derivative
        >>> ftir.derivatives(order=1)
        >>>
        >>> # Calculate 2nd derivative with custom window
        >>> ftir.derivatives(order=2, window_length=21)

        Notes
        -----
        - 1st derivative: Enhances slope changes, reveals peak positions
        - 2nd derivative: Enhances curvature, reveals hidden peaks
        - Larger window_length = more smoothing but less detail
        """
        # Use normalized data by default
        if data is None:
            data = self.df_norm

        # Calculate derivatives
        df_deriv = derivative_batch(
            data = data,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            order = order,
            window_length = window_length,
            polyorder = polyorder,
            delta = delta,
            show_progress = True
        )
        labels = {0: "Zero-derivative", 1: "First-derivative", 2: "Second-derivative"}
        deriv = labels.get(order, f"{order}th-derivative") 

        # Optionally plot results
        if plot:
            print(f'{"#"*10} Plotting {deriv} Spectra! {"#"*10}')
            plot_ftir_spectra(
                data = self.df_deriv,
                samples = None,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                invert_x = True,
                figsize = (7, 4),
                show_legend = True,
                color_by_group = True,
                x_min = None,
                x_max = None,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path,
            )

        # Store result
        self.df_deriv = df_deriv
        return self.df_deriv

    def plot_deriv(
        self,
        df = None,
        orders = [0, 1, 2],
        sample = None,
        wavenumber = None,
        window_length = 15,
        polyorder = 3,
        figsize = (10, 8),
        invert_x = True
    ):
        """
        Plot multiple derivative orders for comparison.

        Creates subplots showing original spectrum (order=0), 1st derivative,
        and 2nd derivative side-by-side.

        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame, optional
            Input data. If None, uses self.df_deriv.

        orders : list of int, default=[0, 1, 2]
            Derivative orders to plot (0=original, 1=first, 2=second).

        sample : str, optional
            Sample name to plot. If None, uses first sample.

        wavenumber : array-like, optional
            Wavenumber values. If None, inferred from column names.

        window_length : int, default=15
            Window size for Savitzky-Golay filter.

        polyorder : int, default=3
            Polynomial order for Savitzky-Golay filter.

        figsize : tuple, default=(10, 8)
            Figure size in inches (width, height).

        invert_x : bool, default=True
            If True, inverts x-axis for standard FTIR display.

        Examples
        --------
        >>> # Plot all derivatives
        >>> ftir.plot_deriv(sample="Sample1")
        >>>
        >>> # Plot only 1st and 2nd derivatives
        >>> ftir.plot_deriv(orders=[1, 2], sample="Sample1")
        """
        # Use derivative data by default
        if df is None:
            df = self.df_deriv

        # Create derivative comparison plot
        plot_derivatives(
            data = df,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            orders = orders,
            sample = sample,
            wavenumbers = wavenumber,
            window_length = window_length,
            polyorder = polyorder,
            figsize = figsize,
            invert_x = invert_x
        )


    def plot_multiple_spec(
        self,
        data_list: List[Union[pd.DataFrame, pl.DataFrame]] = None,
        labels: List[str] = None,
        sample: str = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
    ):
        """
        Compare spectra across multiple processing stages.

        Visualizes the same sample at different stages of the processing pipeline
        (e.g., raw → atmospheric corrected → baseline corrected → denoised → normalized).

        Parameters
        ----------
        data_list : list of DataFrames, optional
            List of datasets to compare. If None, uses:
            [self.df_atm, self.df_corr, self.df_denoised, self.df_norm]

        labels : list of str, optional
            Labels for each dataset. If None, uses:
            ['Atmosphere removed', 'Baseline corrected', 'Denoised', 'Normalized']

        sample : str, optional
            Sample name to plot. If None, uses first sample from df_atm.

        x_min : float, optional
            Minimum wavenumber for display zoom.

        x_max : float, optional
            Maximum wavenumber for display zoom.

        Examples
        --------
        >>> # Compare all processing stages for Sample1
        >>> ftir.plot_multiple_spec(sample="Sample1")
        >>>
        >>> # Compare specific stages
        >>> ftir.plot_multiple_spec(
        ...     data_list=[ftir.df_raw, ftir.df_norm],
        ...     labels=["Raw", "Fully Processed"],
        ...     sample="Sample1"
        ... )
        >>>
        >>> # Zoom into specific region
        >>> ftir.plot_multiple_spec(sample="Sample1", x_min=1000, x_max=2000)

        Notes
        -----
        Uses the compare_ftir_spectra() function for overlay visualization.
        """
        # Use default processing stages if not provided
        if data_list is None:
            data_list = [self.df_atm, self.df_corr, self.df_denoised, self.df_norm]

        if labels is None:
            labels = ['Atmosphare removed', 'Baseline corrected', 'Denoised', 'Normalized']

        # Use first sample if not specified
        if sample is None:
            sample = self.df_atm[:1,:1].values

        # Create comparison plot
        compare_ftir_spectra(
            data_list = data_list,
            labels = labels,
            sample = sample,
            label_column = self.label_column,
            exclude_columns = self.exclude_columns,
            wn_min = self.wn_min,
            wn_max = self.wn_max,
            layout = "overlay",
            offset = None,
            invert_x = True,
            figsize = (10, 6),
            show_legend = True,
            colors = None,
            x_min = x_min,
            x_max = x_max,
            mode = "auto",
        )

    def run(
        self,
        denoising_method: str = "savgol",
        baseline_method: str = "asls",
        exclude_interpolate_method: str = "spline",
        normalization_method: str = "vector",
        plot: bool = False,
        plot_compare: str = None,
        plot_sample: str = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        save_plot: Optional[bool] = False,
        save_path: Optional[str] = None,
    ):
        """
        Run the full FTIR preprocessing pipeline.

        Executes all preprocessing steps in sequence:
        1. Transmittance/Absorbance conversion
        2. Denoising
        3. Baseline correction
        4. Atmospheric correction
        5. Normalization
        6. Derivatives (0th, 1st, 2nd, 3rd order)

        Parameters
        ----------
        exclude_interpolate_method : str, default="spline"
            Method for atmospheric region interpolation.

        baseline_method : str, default="asls"
            Baseline correction method.

        denoising_method : str, default="savgol"
            Denoising method.

        normalization_method : str, default="mean_center"
            Normalization method.

        plot : str, optional
            Plot type: "deriv" for derivative comparison, "compare" for pipeline comparison.

        plot_sample : str, optional
            Sample name to plot. If None, uses first sample from dataset.

        x_min : float, optional
            Minimum wavenumber for display zoom.

        x_max : float, optional
            Maximum wavenumber for display zoom.

        save_plot : bool, default False
            If True, save the plot to disk as PDF.

        save_path : str, optional
            Directory path where plot will be saved. If None, saves in current directory.
            The directory will be created automatically if it doesn't exist.

        Returns
        -------
        tuple of (df_norm, df_0deriv, df_1deriv, df_2deriv, df_3deriv)
            Normalized data and derivatives.

        Examples
        --------
        >>> # Run full preprocessing pipeline
        >>> ftir = FTIRdataprocessing(df, label_column="label")
        >>> df_norm, d0, d1, d2, d3 = ftir.run()
        >>>
        >>> # Run with plotting
        >>> ftir.run(plot="compare", plot_sample="Sample1")
        >>>
        >>> # Run with plot saving (directory created automatically)
        >>> ftir.run(plot="deriv", plot_sample="Sample1",
        ...          save_plot=True, save_path="./output/plots/")
        """
        # Step 1: Convert to absorbance
        converted_df = self.convert(
            data=self.df,
            mode="to_absorbance",
            plot=plot
            )
        
        # Step 2: Denoising
        df_denoised = self.denoise_spect(
            data=converted_df,
            method=denoising_method,
            plot=plot
            )

        # Step 3: Baseline correction
        df_corr = self.correct_baseline(
            data=df_denoised,
            method=baseline_method,
            plot=plot
            )

        # Step 4: Remove atmospheric interference
        df_atm = self.exclude_interpolate(
            data=df_corr,
            method=exclude_interpolate_method,
            plot=plot
            )

        # Step 5: Normalization
        df_norm = self.normalize(
            data=df_atm,
            method=normalization_method,
            plot=plot
            )

        # Step 6: Calculate derivatives
        df_0deriv = self.derivatives(
            data=df_norm,
            order=0,
            plot=plot
            )
        df_1deriv = self.derivatives(
            data=df_norm,
            order=1,
            plot=plot
            )
        df_2deriv = self.derivatives(
            data=df_norm,
            order=2,
            plot=plot
            )
        df_3deriv = self.derivatives(
            data=df_norm,
            order=3,
            plot=plot
            )

        # Determine sample for plotting
        if plot_sample is None:
            # Use first sample from the dataframe (handle both Polars and Pandas)
            if isinstance(df_atm, pl.DataFrame):
                plot_sample = df_atm.to_pandas().index[0]
            else:
                plot_sample = df_atm.index[0]
        if plot_compare is not None:
            print(f"Plotting sample: {plot_sample}")
        # Optional plotting
        if plot_compare == "deriv":
            compare_ftir_spectra(
                data_list = [df_0deriv, df_1deriv, df_2deriv, df_3deriv],
                labels = ['Zero', 'First', 'Second', 'Third'],
                sample = plot_sample,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                layout = "overlay",
                offset = None,
                invert_x = True,
                figsize = (10, 6),
                show_legend = True,
                colors = None,
                x_min = x_min,
                x_max = x_max,
                mode = "absorbance",
                save_plot = save_plot,
                save_path = save_path,
                )
        elif plot_compare == "compare":
            atmspheric_removed = f"Atmosphere removed: {exclude_interpolate_method}"
            baseline_corrected = f"Baseline corrected: {baseline_method}"
            denoised = f"Denoised: {denoising_method}"
            normalized = f"Normalized: {normalization_method}"
            
            cols = df_atm.columns
            wn_vals = []
            for c in cols:
                m = re.search(r'(\d+(?:\.\d+)?)', str(c))  # first number in the column label
                if m:
                    v = float(m.group(1))
                    if 100 <= v <= 8000:
                        wn_vals.append(v)
            w_min = min(wn_vals) if wn_vals else None
            w_max = max(wn_vals) if wn_vals else None

            if x_min is None or x_min < w_min:
                x_min = w_min
            
            if x_max is None or x_max > w_max:
                x_max = w_max

            compare_ftir_spectra(
                data_list = [df_denoised, df_corr, df_atm, df_norm],
                labels = [denoised, baseline_corrected, atmspheric_removed, normalized],
                sample = plot_sample,
                label_column = self.label_column,
                exclude_columns = self.exclude_columns,
                wn_min = self.wn_min,
                wn_max = self.wn_max,
                layout = "overlay",
                offset = None,
                invert_x = True,
                figsize = (10, 6),
                show_legend = True,
                colors = None,
                x_min = x_min,
                x_max = x_max,
                mode = "auto",
                save_plot = save_plot,
                save_path = save_path
                )
        else:
            pass

        return df_norm, df_0deriv, df_1deriv, df_2deriv, df_3deriv
        

    
    def _get_denoised_data(
                self,
                denoising_method: str = "savgol",
                plot: bool = False,
            ):
        df_abs = self.convert(
            plot=plot
            )
        df_denoise = self.denoise_spect(
            data=df_abs,
            method=denoising_method,
            plot=plot
            )
        return df_denoise
    
    def _get_baseline_corrected_data(
                self,
                denoising_method: str = "savgol",
                baseline_correction_method: str = "asls",
                plot: bool = False,
            ):
        df_abs = self.convert(
            plot=plot
            )
        df_denoise = self.denoise_spect(
            data=df_abs,
            method=denoising_method,
            plot=plot
            )
        df_corr = self.correct_baseline(
            data=df_denoise,
            method=baseline_correction_method,
            plot=plot
            )
        return df_corr

    def _get_atmosphere_corrected_data(
                self,
                denoising_method: str = "savgol",
                baseline_correction_method: str = "asls",
                interpolate_method: str = "interpolate",
                plot: bool = False,
            ):
        df_abs = self.convert(
            plot=plot
            )
        df_denoise = self.denoise_spect(
            data=df_abs,
            method=denoising_method,
            plot=plot
            )
        df_corr = self.correct_baseline(
            data=df_denoise,
            method=baseline_correction_method,
            plot=plot
            )
        df_atm = self.exclude_interpolate(
            data=df_corr,
            method = interpolate_method,
            plot=plot
            )
        return df_atm
    
    
    def _get_normalized_data(
                self,
                denoising_method: str = "savgol",
                baseline_correction_method: str = "asls",
                interpolate_method: str = "interpolate",
                normalization_method: str = "vector",
                plot: bool = False,
            ):
        df_abs = self.convert(
            plot=plot
            )
        df_denoise = self.denoise_spect(
            data=df_abs,
            method=denoising_method,
            plot=plot
            )
        df_corr = self.correct_baseline(
            data=df_denoise,
            method=baseline_correction_method,
            plot=plot
            )
        df_atm = self.exclude_interpolate(
            data=df_corr,
            method = interpolate_method,
            plot=plot
            )
        df_normalized = self.normalize(
            data=df_atm,
            method=normalization_method,
            plot=plot
            )
        return df_normalized


class FTIRdataanalysis:
    """
    FTIR Data Analysis and Machine Learning Pipeline Class

    A comprehensive wrapper class for FTIR spectral data analysis, visualization, and
    machine learning workflows. Provides a unified interface for:
    - Exploratory data analysis and visualization
    - Statistical analysis (ANOVA, correlation)
    - Dimensionality reduction (PCA, t-SNE, UMAP, PLS-DA, OPLS-DA)
    - Clustering analysis (K-means, hierarchical)
    - Machine learning model evaluation and comparison
    - Model interpretation (SHAP analysis)

    This class focuses on downstream analysis of preprocessed FTIR data, complementing
    the FTIRdataprocessing class which handles preprocessing steps.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Preprocessed FTIR spectral data with samples as rows and wavenumbers as columns.
        Must include a label column for sample classification.

    dataset_name : str, optional
        Name identifier for the dataset (used in plot titles and file names).

    label_column : str, default="type"
        Name of the column containing sample class labels/categories.

    exclude_columns : list of str, optional
        Additional non-spectral columns to exclude from analysis (e.g., metadata).

    start_wn : float, optional
        Minimum wavenumber (cm⁻¹) for analysis (currently not implemented).

    end_wn : float, optional
        Maximum wavenumber (cm⁻¹) for analysis (currently not implemented).

    drop_region : float, optional
        Wavenumber region to drop from analysis (currently not implemented).

    random_state : int, optional
        Random seed for reproducible train/test splits and cross-validation.

    n_jobs : int, default=-1
        Number of parallel jobs for multi-processing (-1 uses all cores).

    Attributes
    ----------
    df : DataFrame
        Input spectral data
    models : dict
        Dictionary of available machine learning models
    x_train_scaled, x_test_scaled : ndarray
        Scaled training and test feature matrices
    y_train, y_test : ndarray
        Training and test labels
    class_names : list
        Unique class labels
    wavenumbers : ndarray
        Wavenumber values from spectral columns
    results_all : DataFrame
        Machine learning model comparison results
    shap_results : dict
        SHAP explainability analysis results

    Examples
    --------
    >>> # Basic FTIR analysis workflow
    >>> import pandas as pd
    >>> from xpectrass import FTIRdataanalysis
    >>>
    >>> # Load preprocessed FTIR data
    >>> df_preprocessed = pd.read_csv("preprocessed_ftir.csv", index_col=0)
    >>>
    >>> # Initialize analysis pipeline
    >>> ftir_analysis = FTIRdataanalysis(
    ...     df=df_preprocessed,
    ...     dataset_name="MyDataset",
    ...     label_column="type",
    ...     random_state=42
    ... )
    >>>
    >>> # Exploratory analysis
    >>> ftir_analysis.plot_mean_spectra()
    >>> ftir_analysis.plot_pca()
    >>>
    >>> # Machine learning
    >>> results = ftir_analysis.run_all_models()
    >>> ftir_analysis.explain_by_shap(model_name='XGBoost (100)')

    Notes
    -----
    - This class assumes data is already preprocessed (baseline corrected, normalized, etc.)
    - Use FTIRdataprocessing class for preprocessing steps
    - All plotting methods have save_plot and save_path parameters for saving figures
    """

    def __init__(
        self,
        df,
        dataset_name: str = None,
        label_column: str = "type",
        exclude_columns: Optional[List[str]] = None,
        start_wn: Optional[float] = None,
        end_wn: Optional[float] = None,
        drop_region: Optional[float] = None,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        self.df = df
        self.dataset_name = dataset_name
        self.label_column = label_column
        self.exclude_columns = exclude_columns
        self.start_wn = start_wn
        self.end_wn = end_wn
        self.drop_region = drop_region
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = get_all_models()

    def plot_mean_spectra(
            self,
            title: str = "Mean Spectra by Type",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Plot mean spectra for each sample class.

        Creates separate subplots showing the mean spectrum for each class/type
        in the dataset, useful for visualizing class-specific spectral signatures.

        Parameters
        ----------
        title : str, default="Mean Spectra by Type"
            Title for the figure.
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_mean_spectra()
        >>> ftir_analysis.plot_mean_spectra(title="Average Spectra", save_plot=True)
        """
        plot_mean_spectra_by_class(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        title=title,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_overlay_spectra(
            self,
            title: str = "Mean Spectra overlay",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Plot overlaid mean spectra for all classes.

        Creates a single plot with overlaid mean spectra for each class,
        facilitating direct comparison of spectral differences between classes.

        Parameters
        ----------
        title : str, default="Mean Spectra overlay"
            Title for the figure.
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_overlay_spectra()
        >>> ftir_analysis.plot_overlay_spectra(save_plot=True, save_path="overlay.pdf")
        """
        plot_overlay_mean_spectra(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        title=title,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_cv(
            self,
            title: str = "Spectral Variability by Type",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Plot coefficient of variation (CV) across classes.

        Visualizes spectral variability within each class by plotting the
        coefficient of variation at each wavenumber, helping identify regions
        with high or low reproducibility.

        Parameters
        ----------
        title : str, default="Spectral Variability by Type"
            Title for the figure.
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_cv()
        >>> ftir_analysis.plot_cv(title="Within-Class Variability")
        """
        plot_coefficient_of_variation(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        title=title,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_heatmap(
            self,
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Plot spectral heatmap of all samples.

        Creates a heatmap visualization of the entire dataset, with samples as rows
        and wavenumbers as columns, useful for identifying patterns and outliers.

        Parameters
        ----------
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_heatmap()
        >>> ftir_analysis.plot_heatmap(figsize=(20, 15), save_plot=True)
        """
        plot_spectral_heatmap(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def perform_anova(
            self,
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform ANOVA analysis across classes.

        Conducts one-way ANOVA at each wavenumber to identify spectral regions
        with statistically significant differences between classes.

        Parameters
        ----------
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Returns
        -------
        results : DataFrame
            ANOVA results including F-statistics and p-values for each wavenumber.

        Examples
        --------
        >>> results = ftir_analysis.perform_anova()
        >>> significant_regions = results[results['p_value'] < 0.05]
        """
        results = perform_anova_analysis(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )
        return results

    def plot_correlation(
            self,
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Plot correlation matrix of spectral features.

        Creates a correlation heatmap showing relationships between different
        wavenumber regions in the spectral data.

        Parameters
        ----------
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_correlation()
        >>> ftir_analysis.plot_correlation(figsize=(18, 14), save_plot=True)
        """
        plot_correlation_matrix(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )
        
    def plot_pca(
            self,
            standardize: bool = True,
            handle_missing: str = "zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize Principal Component Analysis (PCA).

        Reduces dimensionality using PCA and creates scatter plots showing sample
        separation in principal component space, along with explained variance plots.

        Parameters
        ----------
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_pca()
        >>> ftir_analysis.plot_pca(save_plot=True, save_path="pca_results.pdf")
        """
        perform_pca_analysis(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )
    def plot_tsne(
            self,
            perplexity=50,
            n_iter=1000,
            pca_components=20,
            standardize: bool = True,
            handle_missing: str = "zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize t-SNE analysis.

        Applies t-distributed Stochastic Neighbor Embedding for nonlinear
        dimensionality reduction, useful for visualizing complex cluster structures.

        Parameters
        ----------
        perplexity : float, default=50
            t-SNE perplexity parameter (roughly, number of nearest neighbors).
        n_iter : int, default=1000
            Number of optimization iterations.
        pca_components : int, default=20
            Number of PCA components to apply before t-SNE (for efficiency).
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_tsne()
        >>> ftir_analysis.plot_tsne(perplexity=30, n_iter=1500)
        """
        perform_tsne_analysis(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        perplexity=perplexity,
                        n_iter=n_iter,
                        pca_components=pca_components,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_umap(
            self,
            n_neighbors=100,
            min_dist=0.5,
            pca_components=20,
            standardize: bool = True,
            handle_missing: str = "zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize UMAP analysis.

        Applies Uniform Manifold Approximation and Projection for dimensionality
        reduction, preserving both local and global data structure.

        Parameters
        ----------
        n_neighbors : int, default=100
            Number of neighbors to consider for local structure.
        min_dist : float, default=0.5
            Minimum distance between points in low-dimensional space.
        pca_components : int, default=20
            Number of PCA components to apply before UMAP (for efficiency).
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_umap()
        >>> ftir_analysis.plot_umap(n_neighbors=50, min_dist=0.1)
        """
        perform_umap_analysis(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        pca_components=pca_components,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_plsda(
            self,
            n_components=20,
            standardize: bool = True,
            handle_missing: str = "zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize PLS-DA (Partial Least Squares Discriminant Analysis).

        Applies supervised dimensionality reduction that maximizes class separation
        using partial least squares regression.

        Parameters
        ----------
        n_components : int, default=20
            Number of PLS components to compute.
        standardize : bool, default=True
            If True, standardize features before PLS-DA.
        handle_missing : str, default="zero"
            How to handle missing values ("drop", "mean", "zero", "raise").
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_plsda()
        >>> ftir_analysis.plot_plsda(n_components=30, save_plot=True)
        >>> ftir_analysis.plot_plsda(standardize=False, handle_missing="mean")
        """
        perform_plsda_analysis(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        n_components=n_components,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )
    def plot_oplsda(
            self,
            n_components=1,
            n_orthogonal=2,
            standardize: bool = True,
            handle_missing: str = "zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize OPLS-DA (Orthogonal Partial Least Squares Discriminant Analysis).

        Applies supervised dimensionality reduction that separates variation into
        predictive and orthogonal (non-predictive) components.

        Parameters
        ----------
        n_components : int, default=1
            Number of predictive components.
        n_orthogonal : int, default=2
            Number of orthogonal components.
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_oplsda()
        >>> ftir_analysis.plot_oplsda(n_components=2, n_orthogonal=1)
        """
        perform_oplsda_analysis(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        n_components=n_components,
                        n_orthogonal=n_orthogonal,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_kmeans_clus(
            self,
            n_clusters=8,
            pca_components=20,
            n_components_clustering=10,
            k_range=(2,11),
            standardize=True,
            handle_missing="zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize K-means clustering analysis.

        Applies K-means clustering with elbow method and silhouette analysis
        to identify optimal number of clusters and visualize clustering results.

        Parameters
        ----------
        n_clusters : int, default=8
            Number of clusters for final K-means.
        pca_components : int, default=20
            Number of PCA components to apply before clustering.
        n_components_clustering : int, default=10
            Number of components to use for clustering (after PCA).
        k_range : tuple, default=(2, 11)
            Range of k values to test for elbow/silhouette analysis.
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_kmeans_clus()
        >>> ftir_analysis.plot_kmeans_clus(n_clusters=5, k_range=(2, 15))
        """
        perform_kmeans_clustering(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        n_clusters=n_clusters,
                        pca_components=pca_components,
                        n_components_clustering=n_components_clustering,
                        k_range=k_range,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )

    def plot_hierarchical_clus(
            self,
            n_clusters=8,
            pca_components=20,
            n_components_clustering=10,
            n_samples_dendro=100,
            standardize=True,
            handle_missing="zero",
            figsize: Tuple[int] = (16,12),
            save_plot: bool = False,
            save_path: str = None,
    ):
        """
        Perform and visualize hierarchical clustering analysis.

        Applies agglomerative hierarchical clustering with dendrogram visualization
        to reveal hierarchical relationships between samples.

        Parameters
        ----------
        n_clusters : int, default=8
            Number of clusters to identify in the dendrogram.
        pca_components : int, default=20
            Number of PCA components to apply before clustering.
        n_components_clustering : int, default=10
            Number of components to use for clustering (after PCA).
        n_samples_dendro : int, default=100
            Number of samples to include in dendrogram (for readability).
        figsize : tuple of int, default=(16, 12)
            Figure size in inches (width, height).
        save_plot : bool, default=False
            If True, saves the plot to file.
        save_path : str, optional
            File path to save the plot.

        Examples
        --------
        >>> ftir_analysis.plot_hierarchical_clus()
        >>> ftir_analysis.plot_hierarchical_clus(n_clusters=6, n_samples_dendro=150)
        """
        perform_hierarchical_clustering(
                        data=self.df,
                        dataset_name=self.dataset_name,
                        label_column=self.label_column,
                        exclude_columns=self.exclude_columns,
                        n_clusters=n_clusters,
                        pca_components=pca_components,
                        n_components_clustering=n_components_clustering,
                        n_samples_dendro=n_samples_dendro,
                        standardize=standardize,
                        handle_missing=handle_missing,
                        figsize=figsize,
                        save_plot=save_plot,
                        save_path=save_path
                        )
    
    def ml_prepare_data(
            self,
            test_size=0.2,
    ):
        """
        Prepare data for machine learning.

        Splits data into training and test sets, applies standard scaling,
        and prepares all necessary components for model training.

        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of dataset to include in the test split (0.0 to 1.0).

        Returns
        -------
        dict
            Dictionary containing:
            - X_train, X_test: Scaled feature matrices
            - X_train_raw, X_test_raw: Unscaled feature matrices
            - y_train, y_test: Encoded labels
            - scaler: Fitted StandardScaler object
            - label_encoder: Fitted LabelEncoder object
            - class_names: List of class names
            - wavenumbers: Array of wavenumber values

        Examples
        --------
        >>> data_dict = ftir_analysis.ml_prepare_data(test_size=0.3)
        >>> print(data_dict['X_train'].shape)
        """
        if self.random_state is None:
            random_state = 42
        else:
            random_state = self.random_state
        dir = prepare_data(
            data=self.df,
            label_column=self.label_column,
            exclude_columns=self.exclude_columns,
            test_size=test_size,
            random_state=random_state
        )
        self.x_train_scaled = dir['X_train']
        self.x_test_scaled = dir['X_test']
        self.x_train_raw = dir['X_train_raw']
        self.x_test_raw = dir['X_test_raw']
        self.y_train = dir['y_train']
        self.y_test = dir['y_test']
        self.scaler = dir['scaler']
        self.label_encoder = dir['label_encoder']
        self.class_names = dir['class_names']
        self.wavenumbers = dir['wavenumbers']
        self.dir = dir
        return self.dir

    def available_models(
            self,
    ):
        """
        Get dictionary of all available machine learning models.

        Returns
        -------
        dict
            Dictionary mapping model names to configured sklearn model objects.

        Examples
        --------
        >>> models = ftir_analysis.available_models()
        >>> print(list(models.keys()))
        """
        return get_all_models()

    def run_a_model(
            self,
            model_name='XGBoost (100)',
            model=None,
            cv_folds=5,
            plot_confusion=True,
            save_plot_path=None,
            print_test_result=True,
    ):
        """
        Train and evaluate a single machine learning model.

        Trains a specified model using cross-validation and evaluates performance
        on the test set with comprehensive metrics and confusion matrix visualization.

        Parameters
        ----------
        model_name : str, default='XGBoost (100)'
            Name of the model to train (must be in available_models()).
        model : sklearn estimator, optional
            Custom model object. If None, uses model from available_models().
        cv_folds : int, default=5
            Number of cross-validation folds.
        plot_confusion : bool, default=True
            If True, plots confusion matrix.
        save_plot_path : str, optional
            File path to save the confusion matrix plot.
        print_test_result : bool, default=True
            If True, prints detailed test set metrics.

        Returns
        -------
        tuple of (overall, per_class, confusion_matrix)
            - overall: Dict of overall classification metrics
            - per_class: Dict of per-class metrics
            - confusion_matrix: Confusion matrix array

        Examples
        --------
        >>> overall, per_class, cm = ftir_analysis.run_a_model(model_name='XGBoost (100)')
        >>> print(f"Test Accuracy: {overall['accuracy']:.3f}")
        """
        if model is None:
            model = self.models[model_name]
        if model_name not in self.models:
            available_models = ', '.join(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        self.ml_prepare_data()
        res, m = evaluate_model(
            name=model_name,
            model=model,
            X_train=self.x_train_scaled, 
            X_test=self.x_test_scaled,
            y_train=self.y_train,
            y_test=self.y_test,
            cv_folds=cv_folds, 
            X_train_raw=self.x_train_raw, 
            X_test_raw=self.x_test_raw
            )
        self.single_result = res
        self.trained_model = m
        if plot_confusion:
            plot_confusion_matrix(
                y_test=self.y_test,
                y_pred=res['y_pred'],
                model=None,
                X_test=None,
                class_names=self.class_names,
                data_dict=None,
                label_encoder=self.label_encoder,
                dataset_name=self.dataset_name,
                normalize=False,
                save_plot=save_plot_path
            )
        if print_test_result:
            results_metrics = calculate_multiclass_metrics(
                y_test=self.y_test,
                y_pred=res['y_pred'],
                model=None,
                X_test=None,
                class_names=self.class_names,
                data_dict=None,
                label_encoder=self.label_encoder,
                y_proba=res['y_pred'],
            )
            
            overall = results_metrics['overall_metrics']
            per_class = results_metrics['per_class_metrics']
            cf = results_metrics['confusion_matrix']
            
            # Overall metrics
            print("OVERALL METRICS:")
            print("-" * 80)
            print(f"  Accuracy:              {overall['accuracy']:.4f}")
            print(f"  Precision (macro):     {overall['precision_macro']:.4f}")
            print(f"  Precision (weighted):  {overall['precision_weighted']:.4f}")
            print(f"  Recall (macro):        {overall['recall_macro']:.4f}")
            print(f"  Recall (weighted):     {overall['recall_weighted']:.4f}")
            print(f"  F1 (macro):            {overall['f1_macro']:.4f}")
            print(f"  F1 (weighted):         {overall['f1_weighted']:.4f}")
            print(f"  MCC:                   {overall['mcc']:.4f}")
            print(f"  Cohen's Kappa:         {overall['cohens_kappa']:.4f}")
            print(f"  Jaccard (macro):       {overall['jaccard_macro']:.4f}")
            print(f"  Jaccard (weighted):    {overall['jaccard_weighted']:.4f}")

            if not np.isnan(overall['roc_auc_ovr_macro']):
                print(f"  ROC-AUC (OvR, macro):  {overall['roc_auc_ovr_macro']:.4f}")
                print(f"  ROC-AUC (OvR, wtd):    {overall['roc_auc_ovr_weighted']:.4f}")
            else:
                print(f"  ROC-AUC:               N/A (no probabilities)")
            print(f'Overall Results:\n {overall}')
            print(f'Per Class Results:\n {per_class}')
            print(f'Confusion Matrix:\n {cf}')
        else:
            overall=None
            per_class=None
            cf=None
        return overall, per_class, cf

    def run_all_models(
            self,
            test_size=0.2,
            plot_comparision=True,
            accuracy_threshold=0.9,
            top_n_methods=20,
            save_plot_path=None
            ):
        """
        Train and evaluate all available machine learning models.

        Performs comprehensive model comparison by training and evaluating all
        models in the available_models() dictionary, with optional visualization
        of performance comparisons.

        Parameters
        ----------
        plot_comparision : bool, default=True
            If True, creates multiple comparison plots (model comparison,
            family comparison, efficiency analysis, overfitting analysis).
        accuracy_threshold : float, default=0.9
            Minimum accuracy threshold for efficiency analysis plots.
        top_n_methods : int, default=20
            Number of top-performing models to highlight in plots.
        save_plot_path : str, optional
            Base path for saving plots.

        Returns
        -------
        pd.DataFrame
            Results dataframe with metrics for all models, sorted by performance.

        Examples
        --------
        >>> results = ftir_analysis.run_all_models()
        >>> print(results.head())
        >>> best_model = results.iloc[0]['model_name']
        """
        dir = self.ml_prepare_data(test_size=test_size)
        results_all = evaluate_all_models(
            models=get_all_models(),
            data_dict=dir,
            dataset_name=self.dataset_name
        )
        if plot_comparision:
            plot_model_comparison(
                results_df=results_all,
                dataset_name=self.dataset_name,
                top_n=top_n_methods,
                save_plot=save_plot_path
            )
            plot_family_comparison(
                results_df=results_all,
                dataset_name=self.dataset_name,
                save_plot=save_plot_path
            )
            plot_efficiency_analysis(
                results_df=results_all,
                dataset_name=self.dataset_name,
                accuracy_threshold=accuracy_threshold,
                save_plot=save_plot_path
            )
            plot_overfitting_analysis(
                results_df=results_all,
                dataset_name=self.dataset_name,
                top_n=top_n_methods,
                save_plot=save_plot_path
            )
        self.results_all = results_all
        return self.results_all

    def model_parameter_tuning(
            self,
            number_of_models=2

    ):
        """
        Perform hyperparameter tuning on top-performing models.

        Uses GridSearchCV or RandomizedSearchCV to optimize hyperparameters
        for the top N models from run_all_models() results.

        Parameters
        ----------
        number_of_models : int, default=2
            Number of top models to tune.

        Returns
        -------
        dict
            Dictionary containing tuning results for each model, including
            best parameters, best scores, and tuned model objects.

        Examples
        --------
        >>> results = ftir_analysis.run_all_models()
        >>> tuned_results = ftir_analysis.model_parameter_tuning(number_of_models=3)
        >>> print(tuned_results[0]['best_params'])
        """
        tuning_results = tune_top_models(
            data_dict=self.dir,
            results_df=self.results_all,
            top_n=number_of_models
            )
        return tuning_results

    def explain_by_shap(
            self,
            model_name = 'XGBoost (100)',
            max_display = 20,
            sample_size=100,
            test_size=0.2,
            cv_folds=5,
            save_plot_path=None
    ):
        """
        Explain model predictions using SHAP (SHapley Additive exPlanations).

        Trains a model and generates SHAP values to explain feature importance
        and model predictions at both global and local levels.

        Parameters
        ----------
        model_name : str, default='XGBoost (100)'
            Name of the model to explain (must be in available_models()).
        max_display : int, default=20
            Maximum number of features to display in SHAP plots.
        sample_size : int, default=100
            Number of background samples for SHAP value calculation.
        save_plot_path : str, optional
            Base path for saving SHAP plots.

        Notes
        -----
        Results are stored in self.shap_results for use with local_shap_plot().
        This method automatically trains the model before generating explanations.

        Examples
        --------
        >>> ftir_analysis.explain_by_shap(model_name='XGBoost (100)')
        >>> ftir_analysis.explain_by_shap(model_name='Random Forest', max_display=30)
        """
        if model_name not in self.models:
            available_models = ', '.join(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        models = get_all_models()
        model = models[model_name]
        dir = self.ml_prepare_data(test_size=test_size)
        x_train_scaled = dir['X_train']
        x_test_scaled = dir['X_test']
        x_train_raw = dir['X_train_raw']
        x_test_raw = dir['X_test_raw']
        y_train = dir['y_train']
        y_test = dir['y_test']
        scaler = dir['scaler']
        label_encoder = dir['label_encoder']
        class_names = dir['class_names']
        wavenumbers = dir['wavenumbers']

        _, trained_model = evaluate_model(
            name=model_name,
            model=model,
            X_train=x_train_scaled,
            X_test=x_test_scaled,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            X_train_raw=x_train_raw,
            X_test_raw=x_test_raw
            )

        shap_results = explain_model_shap(
            model=trained_model,
            X_train=x_train_scaled,
            X_test=x_test_scaled,
            y_test=y_test,
            class_names=class_names,
            data_dict=dir,
            wavenumbers=wavenumbers,
            max_display=max_display,
            sample_size=sample_size,
            dataset_name=self.dataset_name,
            X_train_raw=x_train_raw,
            X_test_raw=x_test_raw,
            save_plot=save_plot_path
            )
        self.shap_results = shap_results
        return self.shap_results

    def local_shap_plot(
            self,
            sample_index=0,
            figsize=(10,8),
            save_plot_path=None
        ):
        """
        Plot SHAP decision plot for a specific test sample.

        Creates a detailed visualization showing how individual features
        contribute to the model's prediction for a single sample.

        Parameters
        ----------
        sample_index : int, default=0
            Index of the test sample to explain.
        figsize : tuple, default=(10, 8)
            Figure size in inches (width, height).
        save_plot_path : str, optional
            File path to save the plot.

        Notes
        -----
        Must call explain_by_shap() first to generate SHAP values.

        Examples
        --------
        >>> ftir_analysis.explain_by_shap()
        >>> ftir_analysis.local_shap_plot(sample_index=0)
        >>> ftir_analysis.local_shap_plot(sample_index=5, save_plot_path="shap_sample5.pdf")
        """
        plot_shap_decision(
            shap_results=self.shap_results,
            sample_idx=sample_index,
            figsize=figsize,
            save_plot=save_plot_path
        )
