import matplotlib.pyplot as plt
from typing import Union, Sequence, Optional, List
import pandas as pd
import polars as pl
import numpy as np
import tqdm
from pathlib import Path

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)

def plot_ftir_spectra(
    data: Union[pd.DataFrame, pl.DataFrame],
    samples: Union[str, Sequence[str]] = None,
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    invert_x: bool = True,
    figsize: tuple = (7, 4),
    show_legend: bool = True,
    color_by_group: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    mode: str = "auto",
    save_plot: Optional[bool] = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot one or more FT-IR spectra stored row-wise (samples × wavenumbers).

    Parameters
    ----------
    data : pandas.DataFrame | polars.DataFrame
        Rows = samples, columns = wavenumbers (numerical), plus an optional
        `label_column` for grouping information.
    samples : str | list[str] | None, default None
        Which sample names (from "sample" column) to plot.  *None* ⇒ plot *all* rows.
    label_column : str, default "label"
        Name of the non-spectral column to ignore when plotting.
    exclude_columns : list[str] | None, default None
        Additional column names to exclude from spectral data (e.g., metadata
        like 'temperature', 'batch_id'). The label_column is always excluded.
    wn_min : float, optional
        Minimum wavenumber to include in plot (filters columns before plotting).
        Use this to restrict which spectral data is processed (more efficient).
        If None, uses full spectrum range. Consistent with other modules
        (denoise.py, baseline.py, etc.).
    wn_max : float, optional
        Maximum wavenumber to include in plot (filters columns before plotting).
        Use this to restrict which spectral data is processed (more efficient).
        If None, uses full spectrum range. Consistent with other modules
        (denoise.py, baseline.py, etc.).
    invert_x : bool, default True
        If True, reverse the x-axis so that 4000 cm⁻¹ is at the left.
    figsize : tuple(int, int), default (7, 4)
        Size of the matplotlib figure.
    show_legend : bool, default True
        If True, display legend with sample labels. Set to False for large datasets.
    color_by_group : bool, default False
        If True, color spectra by their group from the label_column.
    x_min : float, optional
        Minimum wavenumber for x-axis display (zoom). All data is plotted,
        but the view is zoomed to this range. If None, uses full spectrum range.
    x_max : float, optional
        Maximum wavenumber for x-axis display (zoom). All data is plotted,
        but the view is zoomed to this range. If None, uses full spectrum range.
    mode : str, default "auto"
        Plot mode:
        - "transmittance": Display as transmittance (%). Converts if data is absorbance.
        - "absorbance": Display as absorbance (AU). Converts if data is transmittance.
        - "auto": Auto-detect data type (threshold: 10.0) and plot as-is without conversion.

    Notes
    -----
    • Uses shared spectral utilities for robust column detection and sorting
    • Spectral columns are automatically sorted by wavenumber (ascending)
    • Handles unsorted or non-contiguous wavenumber columns correctly
    • **wn_min/wn_max**: Filter which columns to process (efficient, reduces data)
    • **x_min/x_max**: Control display zoom only (all data plotted, view zoomed)
    • For best performance: use wn_min/wn_max to filter, then x_min/x_max to zoom
    • Works with output from `import_data_pl` / `import_data_polars`
    """
    # --- Convert to pandas for plotting convenience --------------------------
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    else:
        df = data.copy()

    # --- Determine if we should use "sample" column or index ------------------
    # If "sample" column exists, use it for sample identification
    use_sample_column = "sample" in df.columns

    # --- Select rows to plot --------------------------------------------------
    if samples is None:
        if use_sample_column:
            rows = df["sample"].tolist()
        else:
            rows = df.index.tolist()
    elif isinstance(samples, str):
        rows = [samples]
    else:
        rows = list(samples)

    # --- Identify and sort spectral columns using shared utilities -----------
    # Build exclusion list
    exclude = [label_column] if label_column in df.columns else []
    if use_sample_column:
        exclude.append("sample")
    if exclude_columns:
        exclude.extend(exclude_columns)

    # Infer spectral columns (filter by wn_min/wn_max if provided)
    spectral_cols, wn_values = _infer_spectral_columns(
        df,
        exclude_columns=exclude,
        wn_min=wn_min,  # Filter columns to this range (efficient)
        wn_max=wn_max   # x_min/x_max will be used for display zoom
    )

    # Sort columns by wavenumber (ascending: 400 → 4000)
    spectral_cols_sorted, wn_sorted, _ = _sort_spectral_columns(
        spectral_cols, wn_values
    )

    # --- Detect what the data currently is ------------------------------------
    # Detect the actual data type using robust statistics
    if len(df) > 0 and len(spectral_cols_sorted) > 0:
        # Sample up to 100 rows for robust statistics (avoid slow computation on large datasets)
        sample_size = min(100, len(df))
        sample_data = df[spectral_cols_sorted].iloc[:sample_size].values.flatten()

        # Remove NaN/inf values for robust statistics
        sample_data = sample_data[np.isfinite(sample_data)]

        if len(sample_data) > 0:
            median_val = np.median(sample_data)
            p95_val = np.percentile(sample_data, 95)

            # Heuristic: if 95th percentile > 10 and median > 1, likely transmittance (%)
            # If values are mostly small (median < 5 and p95 < 10), likely absorbance
            if p95_val > 10.0 and median_val > 1.0:
                data_type = "transmittance"
            else:
                data_type = "absorbance"
        else:
            # Fallback if all values are NaN/inf
            data_type = "transmittance"
    else:
        # Fallback if no data
        data_type = "transmittance"

    # --- Determine display mode and whether conversion is needed --------------
    mode_c = _canonical_mode(mode)
    if mode_c == "auto":
        # Auto mode: display data as-is (no conversion)
        display_mode = data_type
        needs_conversion = False
        print(f"Auto-detected: {data_type.capitalize()}")
    else:
        # User specified a mode: display in that mode
        display_mode = mode_c
        # Only convert if requested mode differs from actual data type
        needs_conversion = (display_mode != data_type)

    # --- Prepare figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # --- Prepare data for efficient plotting (avoid O(N²) masking) ------------
    if use_sample_column:
        # Check for duplicate sample names ONCE before plotting
        duplicates = df[df["sample"].isin(rows)]["sample"].value_counts()
        duplicates = duplicates[duplicates > 1]
        if len(duplicates) > 0:
            dup_samples = ", ".join([f"'{s}' ({count}x)" for s, count in duplicates.items()])
            raise ValueError(
                f"Duplicate sample names found: {dup_samples}. "
                f"Sample names must be unique. Please ensure each sample has a unique identifier, "
                f"or use row indices instead of the 'sample' column."
            )

        # Filter to requested samples and set index ONCE (O(N) instead of O(N²))
        df_subset = df[df["sample"].isin(rows)].set_index("sample")

        # Check for missing samples
        missing_samples = set(rows) - set(df_subset.index)
        if missing_samples:
            print(f"Warning: Samples not found in data, skipping: {missing_samples}")
            rows = [s for s in rows if s in df_subset.index]

    # --- Set up color mapping if needed ---------------------------------------
    if color_by_group and label_column in df.columns:
        if use_sample_column:
            unique_groups = df_subset.loc[rows, label_column].unique()
        else:
            unique_groups = df.loc[rows, label_column].unique()
        color_map = plt.cm.get_cmap('tab10', len(unique_groups))
        group_colors = {group: color_map(i) for i, group in enumerate(unique_groups)}

    # --- Plot spectra ---------------------------------------------------------
    # Optimization: use vectorized plotting when labels not needed (much faster)
    if not show_legend and not color_by_group:
        # Extract all spectra at once as 2D array (rows x wavenumbers)
        if use_sample_column:
            y_matrix = df_subset.loc[rows, spectral_cols_sorted].astype(float).values
        else:
            y_matrix = df.loc[rows, spectral_cols_sorted].astype(float).values

        # Convert if needed (vectorized operations on entire matrix)
        if needs_conversion:
            if display_mode == "absorbance" and data_type == "transmittance":
                eps = 1e-12
                invalid_mask = y_matrix <= 0
                if invalid_mask.any():
                    n_invalid = invalid_mask.sum()
                    print(f"Warning: {n_invalid} nonpositive transmittance values found across all samples. "
                          f"Setting to NaN (physically invalid for log conversion).")
                y_matrix_clipped = np.where(y_matrix > eps, y_matrix, eps)
                y_matrix = -np.log10(y_matrix_clipped / 100)
                y_matrix[invalid_mask] = np.nan
            elif display_mode == "transmittance" and data_type == "absorbance":
                y_matrix = 100.0 * np.power(10, -y_matrix)

        # Plot all spectra in one call (much faster than loop)
        ax.plot(wn_sorted, y_matrix.T, alpha=0.7)
    else:
        # Need individual labels or colors - plot one by one
        for s in tqdm.tqdm(rows):
            # Extract intensities in sorted wavenumber order
            if use_sample_column:
                # Direct index lookup (O(1) with indexed DataFrame)
                y = df_subset.loc[s, spectral_cols_sorted].astype(float).values
            else:
                y = df.loc[s, spectral_cols_sorted].astype(float).values

            # Use sorted wavenumbers as x-axis
            x = wn_sorted

            # Convert if needed (when display_mode differs from data_type)
            if needs_conversion:
                if display_mode == "absorbance" and data_type == "transmittance":
                    # Convert transmittance to absorbance: A = -log10(T/100)
                    # Handle nonpositive values: clip to small epsilon and set invalid to NaN
                    eps = 1e-12
                    invalid_mask = y <= 0
                    if invalid_mask.any():
                        n_invalid = invalid_mask.sum()
                        print(f"Warning: {n_invalid} nonpositive transmittance values found for sample '{s}'. "
                              f"Setting to NaN (physically invalid for log conversion).")
                    y_clipped = np.where(y > eps, y, eps)
                    y = -np.log10(y_clipped / 100)
                    y[invalid_mask] = np.nan
                elif display_mode == "transmittance" and data_type == "absorbance":
                    # Convert absorbance to transmittance: T = 100 * 10^(-A)
                    y = 100.0 * np.power(10, -y)

            if color_by_group and label_column in df.columns:
                if use_sample_column:
                    group = df_subset.loc[s, label_column]
                else:
                    group = df.loc[s, label_column]
                ax.plot(x, y, label=str(group) if show_legend else None,
                       color=group_colors.get(group), alpha=0.7)
            else:
                ax.plot(x, y, label=str(s) if show_legend else None)

    # --- Formatting -----------------------------------------------------------
    # Set x-axis limits for zoom first (if specified), then invert
    if x_min is not None or x_max is not None:
        current_xlim = ax.get_xlim()
        new_x_min = x_min if x_min is not None else current_xlim[0]
        new_x_max = x_max if x_max is not None else current_xlim[1]
        ax.set_xlim(new_x_min, new_x_max)

    # Invert x-axis after setting limits
    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    # Set y-axis label based on display mode
    if display_mode == "absorbance":
        ax.set_ylabel("Absorbance (AU)")
    else:
        ax.set_ylabel("Transmittance (%)")

    ax.set_title("FT-IR spectra")

    if show_legend:
        if color_by_group and label_column in df.columns:
            # Remove duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title="Group", frameon=False)
        else:
            ax.legend(title="Sample", frameon=False)

    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    if save_plot:
        if samples is None:
            sample_name = "NA"
        elif isinstance(samples, str):
            sample_name = samples
        else:
            sample_name = "_"
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"{sample_name}_plot.pdf"
        else:
            file_path = f"{sample_name}_plot.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_ftir_spectra(
    data_list: List[Union[pd.DataFrame, pl.DataFrame]],
    labels: List[str],
    sample: str,
    label_column: str = "label",
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    layout: str = "overlay",
    offset: Optional[float] = None,
    invert_x: bool = True,
    figsize: tuple = (10, 6),
    show_legend: bool = True,
    colors: Optional[List[str]] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    mode: str = "auto",
    save_plot: Optional[bool] = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare a single spectrum across multiple processing stages or datasets.

    Perfect for visualizing before/after comparisons (e.g., raw vs denoised vs
    baseline-corrected vs normalized spectra).

    Parameters
    ----------
    data_list : list of pd.DataFrame or pl.DataFrame
        List of DataFrames to compare (e.g., [raw, denoised, baseline, normalized]).
        Each DataFrame should have the same structure (rows=samples, columns=wavenumbers).
    labels : list of str
        Labels for each dataset (e.g., ["Raw", "Denoised", "Baseline Corrected"]).
        Must have same length as data_list.
    sample : str
        Sample name (from "sample" column or row index) to compare across all datasets.
    label_column : str, default "label"
        Name of the non-spectral column to ignore when plotting.
    exclude_columns : list[str] | None, default None
        Additional column names to exclude from spectral data.
    wn_min : float, optional
        Minimum wavenumber to include (filters columns before plotting).
    wn_max : float, optional
        Maximum wavenumber to include (filters columns before plotting).
    layout : str, default "overlay"
        Plot layout:
        - "overlay": All spectra on same axes (default)
        - "subplots": Separate subplot for each dataset (stacked vertically)
        - "stacked": All on same axes with vertical offset for visibility
    offset : float, optional
        Vertical offset between spectra for "stacked" layout.
        If None, automatically calculated based on data range.
    invert_x : bool, default True
        If True, reverse the x-axis so that 4000 cm⁻¹ is at the left.
    figsize : tuple(int, int), default (10, 6)
        Size of the matplotlib figure.
    show_legend : bool, default True
        If True, display legend with dataset labels.
    colors : list of str, optional
        Custom colors for each dataset. If None, uses default color cycle.
    x_min : float, optional
        Minimum wavenumber for x-axis display (zoom).
    x_max : float, optional
        Maximum wavenumber for x-axis display (zoom).
    mode : str, default "auto"
        Plot mode:
        - "transmittance": Display as transmittance (%)
        - "absorbance": Display as absorbance (AU)
        - "auto": Auto-detect and plot as-is
    save_plot : bool, default False
        If True, save the plot to disk as PDF.
    save_path : str, optional
        Directory path where plot will be saved. If None, saves in current directory.
        **The directory will be created automatically if it doesn't exist.**

    Notes
    -----
    • Uses shared spectral utilities for robust column detection
    • All datasets are aligned by wavenumber (handles different column orders)
    • Perfect for before/after comparison workflows
    • Works with output from any processing function (denoise, baseline, normalize, etc.)
    • When saving plots, the directory structure is automatically created

    Examples
    --------
    >>> # Processing pipeline
    >>> df_raw = import_data_pl("data.csv")
    >>> df_denoised = apply_denoising(df_raw, method="savgol")
    >>> df_baseline = apply_baseline_correction(df_denoised, method="asls")
    >>> df_normalized = apply_normalization(df_baseline, method="minmax")
    >>>
    >>> # Compare all stages (overlay)
    >>> compare_ftir_spectra(
    ...     data_list=[df_raw, df_denoised, df_baseline, df_normalized],
    ...     labels=["Raw", "Denoised", "Baseline", "Normalized"],
    ...     sample="Sample1",
    ...     layout="overlay"
    ... )
    >>>
    >>> # Compare with subplots
    >>> compare_ftir_spectra(
    ...     data_list=[df_raw, df_baseline],
    ...     labels=["Before", "After"],
    ...     sample="Sample1",
    ...     layout="subplots",
    ...     wn_min=1000,
    ...     wn_max=2000
    ... )
    >>>
    >>> # Compare with stacked offset
    >>> compare_ftir_spectra(
    ...     data_list=[df_raw, df_denoised, df_baseline],
    ...     labels=["Raw", "Denoised", "Baseline"],
    ...     sample="Sample1",
    ...     layout="stacked",
    ...     offset=0.5
    ... )
    """
    # Validate inputs
    if len(data_list) != len(labels):
        raise ValueError(f"data_list ({len(data_list)}) and labels ({len(labels)}) must have same length")

    if len(data_list) < 2:
        raise ValueError("Need at least 2 datasets to compare")

    if layout not in ["overlay", "subplots", "stacked"]:
        raise ValueError(f"Invalid layout: '{layout}'. Choose 'overlay', 'subplots', or 'stacked'")

    # Setup colors
    if colors is None:
        # Use tab10 colormap for up to 10 datasets
        cmap = plt.cm.get_cmap('tab10', len(data_list))
        colors = [cmap(i) for i in range(len(data_list))]
    elif len(colors) != len(data_list):
        raise ValueError(f"colors ({len(colors)}) must match data_list ({len(data_list)})")

    # Process each dataset and extract spectrum for the sample
    spectra_data = []

    for i, (data, label) in enumerate(zip(data_list, labels)):
        # Convert to pandas
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()

        # Determine if we should use "sample" column or index
        use_sample_column = "sample" in df.columns

        # Prepare data for efficient extraction
        if use_sample_column:
            # Check for duplicate sample names
            sample_counts = df["sample"].value_counts()
            if sample in sample_counts and sample_counts[sample] > 1:
                raise ValueError(
                    f"Duplicate sample name '{sample}' found in dataset '{label}' ({sample_counts[sample]} occurrences). "
                    f"Sample names must be unique. Please ensure each sample has a unique identifier, "
                    f"or use row indices instead of the 'sample' column."
                )

            # Set index for O(1) lookup
            df = df.set_index("sample")

            # Check if sample exists
            if sample not in df.index:
                raise ValueError(f"Sample '{sample}' not found in dataset '{label}'")
        else:
            if sample not in df.index:
                raise ValueError(f"Sample '{sample}' not found in dataset '{label}'")

        # Build exclusion list (sample column already moved to index if applicable)
        exclude = [label_column] if label_column in df.columns else []
        if exclude_columns:
            exclude.extend(exclude_columns)

        # Infer and sort spectral columns
        spectral_cols, wn_values = _infer_spectral_columns(
            df,
            exclude_columns=exclude,
            wn_min=wn_min,
            wn_max=wn_max
        )

        spectral_cols_sorted, wn_sorted, _ = _sort_spectral_columns(
            spectral_cols, wn_values
        )

        # Extract spectrum (direct O(1) index lookup, duplicates already checked)
        y = df.loc[sample, spectral_cols_sorted].astype(float).values
        x = wn_sorted

        # Detect data type using robust statistics
        y_finite = y[np.isfinite(y)]
        if len(y_finite) > 0:
            median_val = np.median(y_finite)
            p95_val = np.percentile(y_finite, 95)

            # Heuristic: if 95th percentile > 10 and median > 1, likely transmittance (%)
            if p95_val > 10.0 and median_val > 1.0:
                data_type = "transmittance"
            else:
                data_type = "absorbance"
        else:
            # Fallback if all values are NaN/inf
            data_type = "transmittance"

        # Handle mode conversion
        if mode.lower() == "auto":
            # No conversion, use as-is
            pass
        elif mode.lower() in ["transmittance", "t"]:
            if data_type == "absorbance":
                # Convert absorbance to transmittance
                y = 100.0 * np.power(10, -y)
        elif mode.lower() in ["absorbance", "a"]:
            if data_type == "transmittance":
                # Convert transmittance to absorbance: A = -log10(T/100)
                # Handle nonpositive values: clip to small epsilon and set invalid to NaN
                eps = 1e-12
                invalid_mask = y <= 0
                if invalid_mask.any():
                    n_invalid = invalid_mask.sum()
                    print(f"Warning: {n_invalid} nonpositive transmittance values found for sample '{sample}' "
                          f"in dataset '{label}'. Setting to NaN (physically invalid for log conversion).")
                y_clipped = np.where(y > eps, y, eps)
                y = -np.log10(y_clipped / 100)
                y[invalid_mask] = np.nan

        spectra_data.append({
            'x': x,
            'y': y,
            'label': label,
            'color': colors[i],
            "data_type": data_type
        })

    # Determine y-axis label based on mode
    mode_c = _canonical_mode(mode)
    if mode_c == "transmittance":
        ylabel = "Transmittance (%)"
    elif mode_c == "absorbance":
        ylabel = "Absorbance (AU)"
    else:
        # Auto - use first dataset's type
        ylabel = "Transmittance (%)" if spectra_data[0]["data_type"] == "transmittance" else "Absorbance (AU)"

    # Create plot based on layout
    if layout == "overlay":
        # All spectra on same axes
        fig, ax = plt.subplots(figsize=figsize)

        for spec in spectra_data:
            ax.plot(spec['x'], spec['y'], label=spec['label'], color=spec['color'], linewidth=1.5)

        # Set x-axis limits first, then invert
        if x_min is not None or x_max is not None:
            current_xlim = ax.get_xlim()
            new_x_min = x_min if x_min is not None else current_xlim[0]
            new_x_max = x_max if x_max is not None else current_xlim[1]
            ax.set_xlim(new_x_min, new_x_max)

        if invert_x:
            ax.invert_xaxis()

        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Comparison: {sample}")

        if show_legend:
            ax.legend(frameon=False, loc='best')

        ax.grid(alpha=0.3, linewidth=0.5)

    elif layout == "stacked":
        # All on same axes with vertical offset
        fig, ax = plt.subplots(figsize=figsize)

        # Auto-calculate offset if not provided
        if offset is None:
            # Use 1.5x the max range of any dataset
            max_range = max([spec['y'].max() - spec['y'].min() for spec in spectra_data])
            offset = max_range * 1.5

        for i, spec in enumerate(spectra_data):
            y_offset = spec['y'] + (i * offset)
            ax.plot(spec['x'], y_offset, label=spec['label'], color=spec['color'], linewidth=1.5)

        # Set x-axis limits first, then invert
        if x_min is not None or x_max is not None:
            current_xlim = ax.get_xlim()
            new_x_min = x_min if x_min is not None else current_xlim[0]
            new_x_max = x_max if x_max is not None else current_xlim[1]
            ax.set_xlim(new_x_min, new_x_max)

        if invert_x:
            ax.invert_xaxis()

        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel(f"{ylabel} (offset)")
        ax.set_title(f"Comparison: {sample} (stacked)")

        if show_legend:
            ax.legend(frameon=False, loc='best')

        ax.grid(alpha=0.3, linewidth=0.5)

    elif layout == "subplots":
        # Separate subplot for each dataset
        n_plots = len(spectra_data)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]

        for i, (ax, spec) in enumerate(zip(axes, spectra_data)):
            ax.plot(spec['x'], spec['y'], color=spec['color'], linewidth=1.5)

            ax.set_ylabel(ylabel)
            ax.set_title(spec['label'], fontsize=10)
            ax.grid(alpha=0.3, linewidth=0.5)

        # Set x-label only on bottom plot
        axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")

        # Apply x-axis limits first, then invert all subplots
        if x_min is not None or x_max is not None:
            current_xlim = axes[0].get_xlim()
            new_x_min = x_min if x_min is not None else current_xlim[0]
            new_x_max = x_max if x_max is not None else current_xlim[1]
            for ax in axes:
                ax.set_xlim(new_x_min, new_x_max)

        if invert_x:
            for ax in axes:
                ax.invert_xaxis()

        # Add overall title
        fig.suptitle(f"Comparison: {sample}", fontsize=12, y=0.995)

    plt.tight_layout()
    if save_plot:
        if sample is None:
            sample_name = "NA"
        elif isinstance(sample, str):
            sample_name = sample
        else:
            sample_name = "_"

        # Create save directory if it doesn't exist
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"{sample_name}_plot.pdf"
        else:
            file_path = f"{sample_name}_plot.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()


def _canonical_mode(mode: str) -> str:
    m = (mode or "auto").lower().strip()
    if m in {"auto", "a"}:
        return "auto"
    if m in {"transmittance", "trans", "t"}:
        return "transmittance"
    if m in {"absorbance", "abs", "a"}:
        return "absorbance"
    raise ValueError(f"Invalid mode='{mode}'. Use 'auto'/'a', 'transmittance'/'trans'/'t', or 'absorbance'/'abs'/'a'.")

'''
# `df_wide` is what you obtained from import_data_pl(...)
plot_ftir_spectra(df_wide)                                      # plot every sample
plot_ftir_spectra(df_wide, samples="PP225")                     # plot just one
plot_ftir_spectra(df_wide, samples=["PP225", "HDPE1"])          # plot multiple
plot_ftir_spectra(df_wide, show_legend=False)                   # hide legend for large datasets
plot_ftir_spectra(df_wide, color_by_group=True)                 # color by group labels
plot_ftir_spectra(df_wide, color_by_group=True, show_legend=True)  # color by group with legend

# Wavenumber filtering (efficient - filters columns before plotting)
plot_ftir_spectra(df_wide, wn_min=1000, wn_max=2000)            # process only 1000-2000 cm⁻¹

# Display zoom (all data plotted, view zoomed)
plot_ftir_spectra(df_wide, x_min=1000, x_max=2000)              # zoom view to 1000-2000 cm⁻¹

# Combined: filter + zoom (best performance)
plot_ftir_spectra(df_wide, wn_min=500, wn_max=3500, x_min=1000, x_max=2000)  # filter 500-3500, zoom to 1000-2000

plot_ftir_spectra(df_wide, mode="absorbance")                   # plot as absorbance instead of transmittance
plot_ftir_spectra(df_wide, mode="auto")                         # auto-detect transmittance/absorbance
plot_ftir_spectra(df_wide, color_by_group=True, mode="absorbance", wn_min=500, wn_max=4000)  # combined options

# ==================== compare_ftir_spectra() examples ====================

# Processing pipeline
df_raw = import_data_pl("data.csv")
df_denoised = apply_denoising(df_raw, method="savgol")
df_baseline = apply_baseline_correction(df_denoised, method="asls")
df_normalized = apply_normalization(df_baseline, method="minmax")

# Compare all stages (overlay) - see evolution on same plot
compare_ftir_spectra(
    data_list=[df_raw, df_denoised, df_baseline, df_normalized],
    labels=["Raw", "Denoised", "Baseline", "Normalized"],
    sample="Sample1",
    layout="overlay"
)

# Compare before/after (subplots) - side-by-side comparison
compare_ftir_spectra(
    data_list=[df_raw, df_normalized],
    labels=["Before Processing", "After Processing"],
    sample="Sample1",
    layout="subplots",
    wn_min=1000,
    wn_max=2000
)

# Compare with stacked offset - see all details with vertical separation
compare_ftir_spectra(
    data_list=[df_raw, df_denoised, df_baseline],
    labels=["Raw", "Denoised", "Baseline Corrected"],
    sample="Sample1",
    layout="stacked",
    offset=0.3  # Custom offset
)

# Custom colors for specific comparison
compare_ftir_spectra(
    data_list=[df_raw, df_baseline],
    labels=["Before Baseline", "After Baseline"],
    sample="Sample1",
    colors=['red', 'blue'],
    wn_min=500,
    wn_max=3500
)
'''
