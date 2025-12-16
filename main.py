# Data import function

import os
import pywt
import tqdm
import polars as pl
import pandas as pd
import numpy as np
from scipy import signal, ndimage, optimize, stats, special
from pybaselines import Baseline
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import (import_data, baseline_correction, noise_filtering, 
                    handle_missing_values, tic_normalization, detect_peaks_with_area,
                    detect_peaks_cwt_with_area, robust_peak_detection, align_peaks)

def data_preprocessing(
        file_path,
        mz_min=None,
        mz_max=None,
        baseline_method='airpls',
        noise_method='wavelet',
        missing_value_method='interpolation',
        normalization_target=1e7,
        verbose=False,
        return_all=False
    ):
    """
    Import and preprocess ToF-SIMS data from a text file.

    Parameters:
    -----------
    file_path : str
        Path to the ToF-SIMS data file
    mz_min, mz_max : float, optional
        m/z range to import
    baseline_method : str or None
        Method for baseline correction ('airpls', etc.), or None to skip
    noise_method : str or None
        Method for noise filtering ('wavelet', etc.), or None to skip
    missing_value_method : str or None
        Method for handling missing values, or None to skip
    normalization_target : float or None
        Target TIC for normalization, or None to skip
    verbose : bool
        Print progress if True
    return_all : bool
        If True, return all intermediate arrays

    Returns:
    --------
    mz_values : numpy.ndarray
    normalized_intensities : numpy.ndarray
    sample_name : str
    group : str
    (optionally: intermediate arrays)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mz_values, intensity, sample_name, group = import_data(file_path, mz_min, mz_max)
    if verbose: print("Imported data:",sample_name, mz_values.shape, intensity.shape)

    # Baseline correction
    if baseline_method:
        intensity_base = baseline_correction(intensity, method=baseline_method)
        if verbose: print("Baseline corrected.")
    else:
        intensity_base = intensity

    # Noise filtering
    if noise_method:
        intensity_base_noise = noise_filtering(intensity_base, method=noise_method)
        if verbose: print("Noise filtered.")
    else:
        intensity_base_noise = intensity_base

    # Handle missing values
    if missing_value_method:
        _, intensity_base_noise_mis = handle_missing_values(mz_values, intensity_base_noise, method=missing_value_method)
        if verbose: print("Missing values handled.")
    else:
        intensity_base_noise_mis = intensity_base_noise

    # TIC normalization
    if normalization_target:
        normalized_intensities = tic_normalization(intensity_base_noise_mis, target_tic=normalization_target)
        if verbose: print("TIC normalized.")
    else:
        normalized_intensities = intensity_base_noise_mis

    if return_all:
        return (sample_name, group, mz_values,  
                intensity, intensity_base, intensity_base_noise, 
                intensity_base_noise_mis, normalized_intensities)
    else:
        return sample_name, group, mz_values, normalized_intensities


def plot_peaks(
    spectrum1, 
    spectrum2,
    mz_min=None,
    mz_max=None,
    pre_processing=True,
    normalization_target=1e7,
    log_scale=False,
    show_peaks=True,
    show_peaks_both=False,
    peak_height=0.1,
    peak_prominence=0.1,
    min_peak_width=0.1,
    max_peak_width=1.0,
    figsize=(10, 6),
    colors=('red', 'navy'),
    alpha=(0.6, 0.8),
    legend_loc='best',
    save_path=None,
    save_figure=False
    ):
    """
    Plot two ToF-SIMS spectra, optionally with peak annotation.

    Parameters
    ----------
    spectrum1, spectrum2 : str
        File paths to the spectra.
    mz_min, mz_max : float or None
        m/z range to plot.
    pre_processing : bool
        Whether to preprocess spectra.
    show_peaks : bool
        Annotate peaks for the first spectrum.
    show_peaks_both : bool
        Annotate peaks for both spectra.
    peak_height, peak_prominence, min_peak_width, max_peak_width : float
        Parameters for peak finding.
    figsize : tuple
        Figure size.
    colors : tuple
        Colors for the two spectra.
    alpha : tuple
        Alpha values for the two spectra.
    legend_loc : str
        Legend location.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # Preprocess or import spectra
    if pre_processing:
        name1, group1, mz_values1, intensities1 = data_preprocessing(
            file_path=spectrum1, 
            mz_min=mz_min, 
            mz_max=mz_max,
            baseline_method='airpls', 
            noise_method='wavelet',
            missing_value_method='interpolation', 
            normalization_target=normalization_target
        )
        name2, group2, mz_values2, intensities2 = data_preprocessing(
            file_path=spectrum2, 
            mz_min=mz_min, 
            mz_max=mz_max,
            baseline_method='airpls', 
            noise_method='wavelet',
            missing_value_method='interpolation', 
            normalization_target=normalization_target
        )
    else:
        name1, group1, mz_values1, intensities1 = data_preprocessing(
            file_path=spectrum1, 
            mz_min=mz_min, 
            mz_max=mz_max,
            baseline_method=None, 
            noise_method=None,
            missing_value_method=None, 
            normalization_target=None
        )
        name2, group2, mz_values2, intensities2 = data_preprocessing(
            file_path=spectrum2, 
            mz_min=mz_min, 
            mz_max=mz_max,
            baseline_method=None, 
            noise_method=None,
            missing_value_method=None, 
            normalization_target=None
        )

    # Apply m/z region filter
    mask1 = np.ones_like(mz_values1, dtype=bool)
    mask2 = np.ones_like(mz_values2, dtype=bool)
    if mz_min is not None:
        mask1 &= mz_values1 >= mz_min
        mask2 &= mz_values2 >= mz_min
    if mz_max is not None:
        mask1 &= mz_values1 <= mz_max
        mask2 &= mz_values2 <= mz_max

    mz_region1 = mz_values1[mask1]
    intensities_region1 = intensities1[mask1]
    mz_region2 = mz_values2[mask2]
    intensities_region2 = intensities2[mask2]

    # Plot
    if log_scale:
        intensities_region1 = np.log10(intensities_region1 + 1)
        intensities_region2 = np.log10(intensities_region2 + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mz_region1, intensities_region1, label=group1, color=colors[0], alpha=alpha[0])
    ax.plot(mz_region2, intensities_region2, label=group2, color=colors[1], alpha=alpha[1])

    # Peak finding and annotation
    def annotate_peaks(mz, intensities, color, label):
        peaks, properties = signal.find_peaks(
            intensities,
            height=peak_height,
            prominence=peak_prominence,
            width=(min_peak_width, max_peak_width)
        )
        ax.plot(mz[peaks], intensities[peaks], "x", color=color, label=label)
        for peak_idx in peaks:
            ax.text(
                mz[peak_idx], intensities[peak_idx],
                f"{mz[peak_idx]:.1f}", color=color, fontsize=8,
                rotation=90, va='bottom'
            )

    if show_peaks:
        annotate_peaks(mz_region1, intensities_region1, color=colors[0], label=f"{group1} Peaks")
    if show_peaks_both:
        annotate_peaks(mz_region2, intensities_region2, color=colors[1], label=f"{group2} Peaks")

    ax.set_xlabel('m/z')
    if log_scale:
        ax.set_ylabel('Log10(Intensity + 1)')
    else:
        ax.set_ylabel('Intensity')
    ax.set_title(f'Spectra: {group1} and {group2}')
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    if save_path and save_figure:
        # Compose base filename with parameters
        suffix = f"_mz_{mz_min}_{mz_max}_TIC_norm_{normalization_target}"
        png_path = save_path + suffix + ".png"
        pdf_path = save_path + suffix + ".pdf"
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path, format="pdf")
    plt.show()
    return fig, ax


# Identify normalization target for ITC values across multiple files.

def normalization_target(
        files,
        mz_min=None,
        mz_max=None,
        baseline_method='airpls',
        noise_method='wavelet',
        missing_value_method='interpolation'
    ):
    """
    Normalize peak intensities or areas to a target value.

    Parameters
    ----------
    files : list of str
        List of file paths to process.
    mz_min, mz_max : float or None
        m/z window for data import (if supported).
    baseline_method : str
        Method for baseline correction.
    noise_method : str
        Noise filtering method.
    missing_value_method : str
        Method for handling missing values.
    
    Returns
    -------
    normalized_df : pd.DataFrame
        Normalized DataFrame.
    """
    tic_values = []
    for file_path in tqdm.tqdm(files):
        try:
            sample_name, group, mz_values, normalized_intensities = data_preprocessing(
                file_path=file_path,
                mz_min=mz_min,
                mz_max=mz_max,
                baseline_method=baseline_method,
                noise_method=noise_method,
                missing_value_method=missing_value_method,
                normalization_target=None,  # No normalization here
                verbose=False,
                return_all=False
            )
            tic = np.sum(normalized_intensities)
            tic_values.append((sample_name, group, (tic/1e6).round(1)))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    # Create a DataFrame with TIC values
    tic_df = pd.DataFrame(tic_values, columns=['SampleName', 'Group', 'TIC-Million'])
    print(f"\n Mean TIC: {tic_df['TIC-Million'].mean().round(1)} Million", 
          f"\n Median TIC: {tic_df['TIC-Million'].median().round(1)}  Million",
          f"\n Max TIC: {tic_df['TIC-Million'].max().round(1)}  Million", 
          f"\n Min TIC: {tic_df['TIC-Million'].min().round(1)}  Million")

    return tic_df



# Process one file in parallel using ProcessPoolExecutor


def _process_one_file(file_path,
                      mz_min, 
                      mz_max,
                      baseline_method, 
                      noise_method, 
                      missing_value_method,
                      normalization_target,
                      method,
                      min_intensity, 
                      min_snr, 
                      min_distance, 
                      window_size,
                      peak_height, 
                      prominence,
                      min_peak_width, 
                      max_peak_width, 
                      width_rel_height,
                      distance_threshold, 
                      combined):
    """
    Return a list[dict] with peak records extracted from `file_path`.
    Designed to be *pickle-safe* for ProcessPoolExecutor.
    """
    try:
        # -------- preprocessing --------
        sample_name, group, mz_vals, norm_int = data_preprocessing(
            file_path=file_path,
            mz_min=mz_min, 
            mz_max=mz_max,
            baseline_method=baseline_method,
            noise_method=noise_method,
            missing_value_method=missing_value_method,
            normalization_target=normalization_target,
            verbose=False,
            return_all=False
        )

        # -------- peak detection --------
        if method is None:
            peak_props = detect_peaks_with_area(
                mz_values=mz_vals, 
                intensities=norm_int, 
                sample_name=sample_name, 
                group=group,
                min_intensity=min_intensity, 
                min_snr=min_snr, 
                min_distance=min_distance,
                window_size=window_size, 
                peak_height=peak_height, 
                prominence=prominence,
                min_peak_width=min_peak_width, 
                max_peak_width=max_peak_width, 
                width_rel_height=width_rel_height
                )
        elif method == "cwt":
            peak_props = detect_peaks_cwt_with_area(
                mz_values=mz_vals, 
                intensities=norm_int, 
                sample_name=sample_name, 
                group=group,
                min_intensity=min_intensity, 
                min_snr=min_snr, 
                min_distance=min_distance,
                window_size=window_size, 
                peak_height=peak_height, 
                prominence=prominence,
                min_peak_width=min_peak_width, 
                max_peak_width=max_peak_width, 
                width_rel_height=width_rel_height
                )
        else:
            peak_props = robust_peak_detection(
                mz_values=mz_vals, 
                intensities=norm_int, 
                sample_name=sample_name, 
                group=group, 
                method=method,
                min_intensity=min_intensity, 
                min_snr=min_snr, 
                min_distance=min_distance,
                window_size=window_size, 
                peak_height=peak_height, 
                prominence=prominence,
                min_peak_width=min_peak_width, 
                max_peak_width=max_peak_width, 
                width_rel_height=width_rel_height,
                distance_threshold=distance_threshold, 
                combined=combined
            )

        # -------- flatten to records --------
        recs = []
        for i in range(len(peak_props["Group"])):
            recs.append({
                "SampleName":  str(peak_props["SampleName"][i]),
                "Group":       str(peak_props["Group"][i]),
                "PeakCenter":  float(peak_props["PeakCenter"][i]),
                "PeakWidth":   float(peak_props.get("PeakWidth", [None])[i]),
                "PeakArea":    float(peak_props.get("PeakArea", [None])[i]),
                "Amplitude":   float(peak_props.get("Amplitude", [None])[i]),
                "DetectedBy":  str(peak_props["DetectedBy"][i]),
                "Deconvoluted": str(peak_props["Deconvoluted"][i])
            })
        return recs

    except Exception as exc:
        # Bubble the full traceback up to the parent so it is not silently lost
        raise RuntimeError(f"{file_path}: {exc}") from exc



# Process all peaks to get peak properties


def batch_processing(
        files,
        *,
        max_workers=None,
        mz_min=None, 
        mz_max=None,
        baseline_method='airpls',
        noise_method='wavelet',
        missing_value_method='interpolation',
        normalization_target=1e8,
        method='Gaussian',
        min_intensity=1,
        min_snr=3,
        min_distance=5,
        window_size=10,
        peak_height=50,
        prominence=50,
        min_peak_width=1,
        max_peak_width=75,
        width_rel_height=0.5,
        distance_threshold=0.01,
        combined=False,
        mz_tolerance=0.2,
        mz_rounding_precision=1
    ):
    """
    Parallel version of batch_processing.
    Returns (peaks_df, intensity_df, area_df) just like the serial code.
    """
    # Freeze constant parameters so each worker only receives the file path
    worker = partial(
        _process_one_file,
        mz_min=mz_min, 
        mz_max=mz_max,
        baseline_method=baseline_method,
        noise_method=noise_method,
        missing_value_method=missing_value_method,
        normalization_target=normalization_target,
        method=method,
        min_intensity=min_intensity, 
        min_snr=min_snr, 
        min_distance=min_distance,
        window_size=window_size, 
        peak_height=peak_height, 
        prominence=prominence,
        min_peak_width=min_peak_width, 
        max_peak_width=max_peak_width,
        width_rel_height=width_rel_height,
        distance_threshold=distance_threshold, 
        combined=combined
    )

    all_peak_records = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, fp): fp for fp in files}

        for fut in tqdm.tqdm(as_completed(futures),
                             total=len(futures),
                             desc="Processing ToF-SIMS files"):
            all_peak_records.extend(fut.result())  # may raise RuntimeError

    # Convert the list of records to a DataFrame
    if not all_peak_records:
        raise ValueError("No peaks detected in any file. Check parameters or data quality.")

    peaks_df = pd.DataFrame(all_peak_records)

    intensity_df = align_peaks(
        peaks_df,
        mz_tolerance=mz_tolerance,
        mz_rounding_precision=mz_rounding_precision,
        output='intensity'
    )
    area_df = align_peaks(
        peaks_df,
        mz_tolerance=mz_tolerance,
        mz_rounding_precision=mz_rounding_precision,
        output='area'
    )

    return peaks_df, intensity_df, area_df
