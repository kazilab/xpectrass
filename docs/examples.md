# Examples

This page provides complete examples for common FTIR preprocessing workflows.

## Example 1: Notebook-Style Method Selection

This mirrors the method-selection workflow used in the notebooks.

```python
from xpectrass import FTIRdataprocessing, load_villegas_camacho_2024_c4

dataset = load_villegas_camacho_2024_c4()
fdp = FTIRdataprocessing(
    df=dataset,
    label_column="type",
    flat_windows=[(1880, 1900), (2400, 2700)],
)

# Convert first (transmittance -> absorbance)
df_abs = fdp.convert(plot=False)

# 1) Pick denoising method
denoise_eval = fdp.find_denoising_method(
    data=df_abs,
    methods="FTIR",
    n_samples=50,
    plot=False,
)
best_denoise = fdp.best_denoising_methods(eval_df=denoise_eval, top_n=5)
denoise_method = best_denoise.iloc[0]["method"]

# 2) Pick baseline correction method
df_denoised = fdp._get_denoised_data(denoising_method=denoise_method, plot=False)
rfzn, nar, snr = fdp.find_baseline_method(
    data=df_denoised,
    flat_windows=[(1880, 1900), (2400, 2700)],
    baseline_methods="FTIR",
    n_samples=50,
    plot=False,
)
best_baseline = fdp.best_baseline_method(
    rfzn_tbl=rfzn,
    nar_tbl=nar,
    snr_tbl=snr,
    top_n=5,
)
baseline_method = best_baseline.iloc[0]["method"]

print("Best denoising:", denoise_method)
print("Best baseline:", baseline_method)
```

---

## Example 2 Atmospheric + Normalization Workflow

```python
# Continue from Example 8
df_atm = fdp._get_atmosphere_corrected_data(
    denoising_method=denoise_method,
    baseline_correction_method=baseline_method,
    interpolate_method="zero",
    plot=False,
)

# Rank normalization methods by classification-oriented metrics
norm_scores = fdp.find_normalization_method(
    data=df_atm,
    methods="FTIR",
    n_splits=5,
)
normalization_method = norm_scores.iloc[0]["method"]

# Run full preprocessing with selected methods
df_norm = fdp._get_normalized_data(
    denoising_method=denoise_method,
    baseline_correction_method=baseline_method,
    interpolate_method="zero",
    normalization_method=normalization_method,
    plot=False,
)

df_norm.to_excel("DenoisedBaselineAtmosphericCorrectedNormalizedData.xlsx", index=False)
print("Saved normalized data.")
```

---

## Example 3: Derivatives + Multi-Dataset Combination

```python
from xpectrass import FTIRdataprocessing, load_all_datasets, combine_datasets

all_sets = load_all_datasets()

common_kwargs = dict(
    label_column="type",
    exclude_regions=[(0, 680), (3500, 5000)],
    interpolate_regions=[(1250, 2700)],
    flat_windows=[(1880, 1900), (2400, 2700)],
)

fdp_jung = FTIRdataprocessing(df=all_sets["jung_2018"], **common_kwargs)
jung_norm = fdp_jung._get_normalized_data(
    denoising_method="wavelet",
    baseline_correction_method="aspls",
    interpolate_method="zero",
    normalization_method="snv_detrend",
    plot=False,
)

fdp_frond = FTIRdataprocessing(df=all_sets["frond_2021"], **common_kwargs)
frond_norm = fdp_frond._get_normalized_data(
    denoising_method="wavelet",
    baseline_correction_method="aspls",
    interpolate_method="zero",
    normalization_method="snv_detrend",
    plot=False,
)

# Derivatives
jung_d1 = fdp_jung.derivatives(data=jung_norm, order=1, window_length=15, polyorder=3, delta=1.0, plot=False)
jung_d2 = fdp_jung.derivatives(data=jung_norm, order=2, window_length=15, polyorder=3, delta=1.0, plot=False)

# Combine normalized datasets on a common grid
combined_norm, grid = combine_datasets(
    datasets=[jung_norm, frond_norm],
    wn_min=680,
    wn_max=3000,
    resolution=2.0,
    descending=True,
    method="pchip",
    label_column="type",
    add_study_column=True,
    study_names=["jung_2018", "frond_2021"],
    show_progress=True,
    n_jobs=4,
    data_mode="normalized",
)

combined_norm.to_csv("processed_data/combined_norm_data.csv.xz", compression="xz", index=None)
print("Combined shape:", combined_norm.shape)
```

---

## Example 4: Notebook-Style Data Analysis Class

```python
import pandas as pd
from xpectrass import FTIRdataanalysis

df = pd.read_csv("processed_data/combined_deriv1_data.csv.xz", compression="xz")
df = df[df["study"] != "kedzierski_2019_u"]

fda = FTIRdataanalysis(
    df=df,
    dataset_name="Combined dataset",
    label_column="type",
    exclude_columns=["study", "sample_id", "environmental", "resolution"],
    random_state=42,
    n_jobs=-1,
)

fda.plot_pca(standardize=True, handle_missing="zero", figsize=(12, 8))
fda.plot_umap(
    n_neighbors=100,
    min_dist=0.5,
    pca_components=20,
    standardize=True,
    handle_missing="zero",
    figsize=(12, 8),
)

data_dict = fda.ml_prepare_data(test_size=0.2)
single = fda.run_a_model(model_name="XGBoost (100)", cv_folds=5, plot_confusion=True)
all_results = fda.run_all_models(plot_comparison=True, accuracy_threshold=0.9, top_n_methods=20)
```
