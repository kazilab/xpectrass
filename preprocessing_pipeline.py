"""
FTIR Preprocessing Pipeline
============================

Master orchestrator for FTIR spectral preprocessing.
Provides a configurable pipeline that chains preprocessing steps
in the optimal order.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import polars as pl
import pandas as pd
from copy import deepcopy

# Import preprocessing modules
from .utils.data_validation import validate_spectra, detect_outlier_spectra
from .utils.baseline import baseline_correction, baseline_method_names
from .utils.denoise import denoise, denoise_method_names
from .utils.normalization import (
    normalize, normalize_method_names, mean_center, 
    auto_scale, pareto_scale, snv_detrend
)
from .utils.atmospheric import atmospheric_correction, get_atmospheric_regions
from .utils.derivatives import spectral_derivative, first_derivative, second_derivative
from .utils.scatter_correction import scatter_correction, scatter_method_names
from .utils.region_selection import (
    select_region, exclude_regions, exclude_atmospheric,
    get_wavenumbers, get_spectra_matrix, FTIR_REGIONS
)


@dataclass
class PreprocessingConfig:
    """
    Configuration for FTIR preprocessing pipeline.
    
    Each preprocessing step can be enabled/disabled and configured
    with method-specific parameters.
    """
    # Data validation
    validate: bool = True
    validation_params: Dict[str, Any] = field(default_factory=lambda: {
        'expected_samples_per_class': 500,
        'verbose': False
    })
    
    # Baseline correction
    baseline: bool = True
    baseline_params: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'airpls',
        'lam': 1e6
    })
    
    # Denoising
    denoise: bool = True
    denoise_params: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'savgol',
        'window_length': 15,
        'polyorder': 3
    })
    
    # Normalization
    normalize: bool = True
    normalize_params: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'snv'
    })
    
    # Atmospheric correction
    atmospheric: bool = False  # Often not needed for ATR
    atmospheric_params: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'interpolate'
    })
    
    # Spectral derivatives
    derivatives: bool = False  # Optional
    derivative_params: Dict[str, Any] = field(default_factory=lambda: {
        'order': 1,
        'window_length': 15,
        'polyorder': 3
    })
    
    # Scatter correction
    scatter: bool = False  # Can replace individual SNV
    scatter_params: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'msc'
    })
    
    # Region selection
    region_selection: bool = False
    regions: List[Tuple[float, float]] = field(default_factory=lambda: [(400, 4000)])
    exclude_atmospheric_regions: bool = False
    
    # Mean centering (for PCA/PLS)
    mean_center: bool = True
    
    # Processing order
    order: List[str] = field(default_factory=lambda: [
        'validate',
        'region_selection',
        'atmospheric',
        'baseline',
        'denoise',
        'scatter',
        'normalize',
        'derivatives',
        'mean_center'
    ])


class FTIRPreprocessor:
    """
    Configurable preprocessing pipeline for FTIR spectra.
    
    Parameters
    ----------
    config : PreprocessingConfig or dict, optional
        Pipeline configuration. If dict, converted to PreprocessingConfig.
    
    Attributes
    ----------
    config : PreprocessingConfig
        Current configuration.
    fitted_params_ : dict
        Parameters learned during fit (e.g., mean for centering, reference for MSC).
    validation_report_ : dict
        Validation report from last fit call.
    
    Examples
    --------
    >>> # Default preprocessing
    >>> pipe = FTIRPreprocessor()
    >>> processed = pipe.fit_transform(df)
    
    >>> # Custom configuration
    >>> config = PreprocessingConfig(
    ...     baseline_params={'method': 'asls', 'lam': 1e7},
    ...     normalize_params={'method': 'vector'},
    ...     derivatives=True
    ... )
    >>> pipe = FTIRPreprocessor(config)
    >>> processed = pipe.fit_transform(df)
    """
    
    def __init__(
        self, 
        config: Optional[Union[PreprocessingConfig, dict]] = None
    ):
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, dict):
            self.config = PreprocessingConfig(**config)
        else:
            self.config = config
        
        self.fitted_params_: Dict[str, Any] = {}
        self.validation_report_: Dict[str, Any] = {}
        self._is_fitted = False
    
    def fit(self, df: pl.DataFrame) -> 'FTIRPreprocessor':
        """
        Fit preprocessing pipeline (learn parameters like mean, reference spectrum).
        
        Parameters
        ----------
        df : pl.DataFrame
            Training spectral data.
        
        Returns
        -------
        self
        """
        wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
        self.fitted_params_['wavenumber_cols'] = wavenumber_cols
        self.fitted_params_['wavenumbers'] = np.array([float(c) for c in wavenumber_cols])
        
        # Validation
        if self.config.validate:
            self.validation_report_ = validate_spectra(df, **self.config.validation_params)
        
        # Extract spectra for fitting
        spectra = get_spectra_matrix(df)
        
        # Learn mean for centering
        if self.config.mean_center:
            self.fitted_params_['mean'] = np.mean(spectra, axis=0)
        
        # Learn reference spectrum for MSC
        if self.config.scatter and self.config.scatter_params.get('method') in ('msc', 'emsc'):
            self.fitted_params_['scatter_reference'] = np.mean(spectra, axis=0)
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply preprocessing pipeline to spectra.
        
        Parameters
        ----------
        df : pl.DataFrame
            Spectral data to preprocess.
        
        Returns
        -------
        pl.DataFrame
            Preprocessed spectra.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform. Call fit() first.")
        
        result_df = df
        wavenumbers = self.fitted_params_['wavenumbers']
        
        # Process according to configured order
        for step in self.config.order:
            result_df = self._apply_step(result_df, step, wavenumbers)
        
        return result_df
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fit and transform in one call.
        
        Parameters
        ----------
        df : pl.DataFrame
            Spectral data.
        
        Returns
        -------
        pl.DataFrame
            Preprocessed spectra.
        """
        return self.fit(df).transform(df)
    
    def _apply_step(
        self, 
        df: pl.DataFrame, 
        step: str,
        wavenumbers: np.ndarray
    ) -> pl.DataFrame:
        """Apply a single preprocessing step."""
        
        if step == 'validate':
            # Validation is done in fit, just pass through
            return df
        
        elif step == 'region_selection':
            if self.config.region_selection:
                df = select_region(df, self.config.regions)
                # Update wavenumbers for subsequent steps
                wavenumbers = get_wavenumbers(df)
            if self.config.exclude_atmospheric_regions:
                df = exclude_atmospheric(df)
                wavenumbers = get_wavenumbers(df)
            return df
        
        elif step == 'atmospheric':
            if not self.config.atmospheric:
                return df
            return self._apply_per_spectrum(
                df, wavenumbers,
                lambda y, wn: atmospheric_correction(y, wn, **self.config.atmospheric_params)
            )
        
        elif step == 'baseline':
            if not self.config.baseline:
                return df
            return self._apply_per_spectrum(
                df, wavenumbers,
                lambda y, wn: baseline_correction(y, **self.config.baseline_params)
            )
        
        elif step == 'denoise':
            if not self.config.denoise:
                return df
            return self._apply_per_spectrum(
                df, wavenumbers,
                lambda y, wn: denoise(y, **self.config.denoise_params)
            )
        
        elif step == 'scatter':
            if not self.config.scatter:
                return df
            # Scatter correction needs all spectra together
            spectra = get_spectra_matrix(df)
            reference = self.fitted_params_.get('scatter_reference')
            corrected = scatter_correction(spectra, reference=reference, 
                                           **self.config.scatter_params)
            return self._matrix_to_df(df, corrected)
        
        elif step == 'normalize':
            if not self.config.normalize:
                return df
            return self._apply_per_spectrum(
                df, wavenumbers,
                lambda y, wn: normalize(y, **self.config.normalize_params)
            )
        
        elif step == 'derivatives':
            if not self.config.derivatives:
                return df
            return self._apply_per_spectrum(
                df, wavenumbers,
                lambda y, wn: spectral_derivative(y, **self.config.derivative_params)
            )
        
        elif step == 'mean_center':
            if not self.config.mean_center:
                return df
            spectra = get_spectra_matrix(df)
            mean = self.fitted_params_.get('mean', np.mean(spectra, axis=0))
            centered = spectra - mean
            return self._matrix_to_df(df, centered)
        
        else:
            return df
    
    def _apply_per_spectrum(
        self,
        df: pl.DataFrame,
        wavenumbers: np.ndarray,
        func
    ) -> pl.DataFrame:
        """Apply a function to each spectrum individually."""
        wavenumber_cols = [c for c in df.columns if c not in ('sample', 'label')]
        spectra = df.select(wavenumber_cols).to_numpy().astype(float)
        
        processed = np.array([func(s, wavenumbers) for s in spectra])
        
        return self._matrix_to_df(df, processed)
    
    def _matrix_to_df(
        self,
        original_df: pl.DataFrame,
        spectra: np.ndarray
    ) -> pl.DataFrame:
        """Convert numpy matrix back to Polars DataFrame."""
        wavenumber_cols = [c for c in original_df.columns if c not in ('sample', 'label')]
        
        # Start with sample and label columns
        result = original_df.select(['sample', 'label'])
        
        # Add spectral columns
        for i, col in enumerate(wavenumber_cols):
            result = result.with_columns(pl.lit(spectra[:, i]).alias(col))
        
        return result
    
    def get_params(self) -> dict:
        """Get current configuration as dictionary."""
        return {
            'validate': self.config.validate,
            'baseline': self.config.baseline,
            'baseline_params': self.config.baseline_params,
            'denoise': self.config.denoise,
            'denoise_params': self.config.denoise_params,
            'normalize': self.config.normalize,
            'normalize_params': self.config.normalize_params,
            'atmospheric': self.config.atmospheric,
            'atmospheric_params': self.config.atmospheric_params,
            'derivatives': self.config.derivatives,
            'derivative_params': self.config.derivative_params,
            'scatter': self.config.scatter,
            'scatter_params': self.config.scatter_params,
            'mean_center': self.config.mean_center,
            'order': self.config.order
        }
    
    def set_params(self, **params) -> 'FTIRPreprocessor':
        """Set configuration parameters."""
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self


# ---------------------------------------------------------------------------
#                           QUICK PRESETS
# ---------------------------------------------------------------------------

def get_preset_config(name: str) -> PreprocessingConfig:
    """
    Get predefined preprocessing configurations.
    
    Parameters
    ----------
    name : str
        Preset name:
        - 'minimal': Only baseline + normalization
        - 'standard': Full pipeline (baseline, denoise, normalize, center)
        - 'classification': Optimized for plastic classification
        - 'pca': Standard + derivatives for PCA analysis
        - 'raw': No preprocessing (validation only)
    
    Returns
    -------
    PreprocessingConfig
    """
    presets = {
        'minimal': PreprocessingConfig(
            validate=True,
            baseline=True,
            baseline_params={'method': 'airpls'},
            denoise=False,
            normalize=True,
            normalize_params={'method': 'snv'},
            mean_center=False
        ),
        
        'standard': PreprocessingConfig(
            validate=True,
            baseline=True,
            baseline_params={'method': 'airpls', 'lam': 1e6},
            denoise=True,
            denoise_params={'method': 'savgol', 'window_length': 15, 'polyorder': 3},
            normalize=True,
            normalize_params={'method': 'snv'},
            mean_center=True
        ),
        
        'classification': PreprocessingConfig(
            validate=True,
            baseline=True,
            baseline_params={'method': 'airpls'},
            denoise=True,
            denoise_params={'method': 'savgol', 'window_length': 11, 'polyorder': 2},
            normalize=True,
            normalize_params={'method': 'vector'},
            region_selection=True,
            regions=[(400, 1800), (2700, 3100)],  # Key plastic regions
            mean_center=True
        ),
        
        'pca': PreprocessingConfig(
            validate=True,
            baseline=True,
            denoise=True,
            normalize=True,
            normalize_params={'method': 'snv'},
            derivatives=True,
            derivative_params={'order': 1, 'window_length': 15},
            mean_center=True
        ),
        
        'raw': PreprocessingConfig(
            validate=True,
            baseline=False,
            denoise=False,
            normalize=False,
            mean_center=False
        )
    }
    
    if name not in presets:
        raise ValueError(f"Unknown preset: '{name}'. Available: {list(presets.keys())}")
    
    return presets[name]


def create_preprocessor(preset: str = 'standard', **overrides) -> FTIRPreprocessor:
    """
    Create preprocessor from preset with optional overrides.
    
    Parameters
    ----------
    preset : str
        Preset name.
    **overrides : keyword arguments
        Override specific configuration options.
    
    Returns
    -------
    FTIRPreprocessor
    
    Examples
    --------
    >>> pipe = create_preprocessor('standard')
    >>> pipe = create_preprocessor('minimal', baseline_params={'method': 'asls'})
    """
    config = get_preset_config(preset)
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return FTIRPreprocessor(config)
