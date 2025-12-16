import matplotlib.pyplot as plt
from typing import Union, Sequence
import pandas as pd
import polars as pl

def plot_ftir_spectra(
    data: Union[pd.DataFrame, pl.DataFrame],
    samples: Union[str, Sequence[str]] = None,
    label_column: str = "label",
    invert_x: bool = True,
    figsize: tuple = (7, 4),
) -> None:
    """
    Plot one or more FT-IR spectra stored row-wise (samples × wavenumbers).

    Parameters
    ----------
    data : pandas.DataFrame | polars.DataFrame
        Rows = samples, columns = wavenumbers (numerical), plus an optional
        `label_column` for grouping information.
    samples : str | list[str] | None, default None
        Which sample names (row index) to plot.  *None* ⇒ plot *all* rows.
    label_column : str, default "label"
        Name of the non-spectral column to ignore when plotting.
    invert_x : bool, default True
        If True, reverse the x-axis so that 4000 cm⁻¹ is at the left.
    figsize : tuple(int, int), default (7, 4)
        Size of the matplotlib figure.

    Notes
    -----
    • Works with output from `import_data_pl` / `import_data_polars`
      (each sample in one row, wavenumber columns numeric).  
    • Any column whose name is not numeric or cannot be converted
      to float is skipped automatically.
    """
    # --- Convert to pandas for plotting convenience --------------------------
    if isinstance(data, pl.DataFrame):
        df = data.to_pandas()
    else:
        df = data.copy()

    # --- Select rows to plot --------------------------------------------------
    if samples is None:
        rows = df.index.tolist()
    elif isinstance(samples, str):
        rows = [samples]
    else:
        rows = list(samples)

    # --- Identify spectral (numeric) columns ---------------------------------
    numeric_cols = [c for c in df.columns
                    if c != label_column
                    and pd.api.types.is_numeric_dtype(df[c])]

    # --- Prepare figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    for s in rows:
        y = df.loc[s, numeric_cols].astype(float).values
        x = pd.Series(numeric_cols, dtype=float).values
        ax.plot(x, y, label=str(s))

    # --- Formatting -----------------------------------------------------------
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.set_title("FT-IR spectra")
    ax.legend(title="Sample", frameon=False)
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.show()


'''
# `df_wide` is what you obtained from import_data_pl(...)
plot_ftir_spectra(df_wide)                     # plot every sample
plot_ftir_spectra(df_wide, samples="PP225")    # plot just one
plot_ftir_spectra(df_wide, samples=["PP225", "HDPE1"])
'''
