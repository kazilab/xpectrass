# Utils Module

## data_validation

```{eval-rst}
.. automodule:: xpectrass.utils.data_validation
   :members:
   :undoc-members:
```

### validate_spectra

```python
validate_spectra(
    df: pl.DataFrame,
    expected_samples_per_class: int = 500,
    expected_classes: List[str] = None,
    wavenumber_range: Tuple[float, float] = (399.0, 4000.0),
    intensity_range: Tuple[float, float] = (0.0, 150.0),
    verbose: bool = True
) -> Dict[str, Any]
```

### detect_outlier_spectra

```python
detect_outlier_spectra(
    df: pl.DataFrame,
    method: str = "zscore",  # 'zscore', 'iqr', 'mad'
    threshold: float = 3.0
) -> Dict[str, Any]
```

---

## baseline

```{eval-rst}
.. automodule:: xpectrass.utils.baseline
   :members:
   :undoc-members:
```

### baseline_correction

```python
baseline_correction(
    intensities: np.ndarray,
    method: str = "airpls",
    window_size: int = 101,
    poly_order: int = 4,
    clip_negative: bool = True,
    return_baseline: bool = False,
    **kwargs
) -> np.ndarray
```

### baseline_method_names

```python
baseline_method_names() -> List[str]
```

Returns list of 50+ available baseline correction methods.

---

## denoise

```{eval-rst}
.. automodule:: xpectrass.utils.denoise
   :members:
   :undoc-members:
```

### denoise

```python
denoise(
    intensities: np.ndarray,
    method: str = "savgol",
    **kwargs
) -> np.ndarray
```

**Methods:** savgol, wavelet, moving_average, gaussian, median, whittaker, lowpass

---

## normalization

```{eval-rst}
.. automodule:: xpectrass.utils.normalization
   :members:
   :undoc-members:
```

### normalize

```python
normalize(intensities: np.ndarray, method: str = "snv", **kwargs) -> np.ndarray
```

**Methods:** snv, vector, minmax, area, peak, range, max

### mean_center

```python
mean_center(
    spectra: np.ndarray,
    axis: int = 0,
    return_mean: bool = True
) -> Tuple[np.ndarray, np.ndarray]
```

### auto_scale

```python
auto_scale(spectra: np.ndarray, return_params: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

---

## atmospheric

```{eval-rst}
.. automodule:: xpectrass.utils.atmospheric
   :members:
   :undoc-members:
```

### atmospheric_correction

```python
atmospheric_correction(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    method: str = "interpolate",
    co2_range: Tuple[float, float] = (2300, 2400),
    h2o_ranges: List[Tuple[float, float]] = [(1350, 1900), (3550, 3900)],
    **kwargs
) -> np.ndarray
```

**Methods:** interpolate, spline, reference, zero, exclude

---

## derivatives

```{eval-rst}
.. automodule:: xpectrass.utils.derivatives
   :members:
   :undoc-members:
```

### spectral_derivative

```python
spectral_derivative(
    intensities: np.ndarray,
    order: int = 1,
    window_length: int = 15,
    polyorder: int = 3,
    delta: float = 1.0
) -> np.ndarray
```

### first_derivative / second_derivative

```python
first_derivative(intensities, window_length=15, polyorder=3) -> np.ndarray
second_derivative(intensities, window_length=15, polyorder=4) -> np.ndarray
```

---

## scatter_correction

```{eval-rst}
.. automodule:: xpectrass.utils.scatter_correction
   :members:
   :undoc-members:
```

### scatter_correction

```python
scatter_correction(
    spectra: np.ndarray,  # (n_samples, n_wavenumbers)
    method: str = "msc",
    reference: np.ndarray = None,
    **kwargs
) -> np.ndarray
```

**Methods:** msc, emsc, snv, snv_detrend

---

## region_selection

```{eval-rst}
.. automodule:: xpectrass.utils.region_selection
   :members:
   :undoc-members:
```

### select_region

```python
select_region(
    df: pl.DataFrame,
    regions: Union[str, Tuple, List[Tuple]]
) -> pl.DataFrame
```

### exclude_regions

```python
exclude_regions(df: pl.DataFrame, regions: Union[str, Tuple, List[Tuple]]) -> pl.DataFrame
```

### FTIR_REGIONS

```python
FTIR_REGIONS = {
    'fingerprint': (400, 1500),
    'ch_stretch': (2800, 3100),
    'carbonyl': (1650, 1800),
    # ... and more
}
```

---

## file_management

```{eval-rst}
.. automodule:: xpectrass.utils.file_management
   :members:
   :undoc-members:
```

### process_batch_files

```python
process_batch_files(
    files: Iterable[str],
    skiprows: int = 15,
    separator: str = ',',
    engine: str = "pl",
    show_progress: bool = True
) -> pl.DataFrame
```

### import_data

```python
import_data(file_path: str, engine: str = 'pl', skiprows: int = 15) -> pl.DataFrame
```
