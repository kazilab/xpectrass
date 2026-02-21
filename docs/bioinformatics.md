# Xpectrass for Bioinformatics

FTIR spectroscopy is increasingly used in bioinformatics-adjacent workflows, especially where rapid, non-destructive biochemical fingerprinting is useful. Typical examples include strain-level microbial typing, tissue-state classification, and biomaterial characterization. In these settings, the bottleneck is often not model selection, but reliable preprocessing: small differences in denoising, baseline correction, atmospheric handling, and normalization can dominate downstream classification performance.

`xpectrass` addresses this by providing an end-to-end and evaluation-first workflow for FTIR spectral data. Instead of forcing one fixed preprocessing recipe, it lets you compare methods at each stage and choose approaches that are empirically better for your dataset.

## Why this matters in bioinformatics

Biological spectra are noisy and heterogeneous. Batch effects, sample handling differences, hydration state, and instrument drift can all change spectral shape before any true biological signal is considered. A practical bioinformatics pipeline therefore needs:

- strict data handling and validation
- robust preprocessing with method comparisons
- reproducible dimensionality-reduction and clustering views
- explainable classification outputs

`xpectrass` covers these requirements with two main classes:

- `FTIRdataprocessing`: conversion, atmospheric correction, baseline correction, denoising, normalization, derivatives
- `FTIRdataanalysis`: visualization, statistics, dimensionality reduction, clustering, machine learning, SHAP-based explainability

## A practical workflow

For a typical classification task, the recommended sequence is:

1. Convert spectra into a consistent representation (usually absorbance).
2. Reduce high-frequency noise (denoising).
3. Correct baseline artifacts.
4. Handle atmospheric regions.
5. Normalize intensities.
6. Optionally compute derivatives for peak enhancement.
7. Run exploratory analysis and train classification models.

This progression helps separate spectral cleanup from biological interpretation, which reduces accidental overfitting to acquisition artifacts.

## Example: from spectra to model comparison

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_jung_2018

# Load data
df = load_jung_2018()

# Preprocess
fdp = FTIRdataprocessing(df=df, label_column="type")
df_abs = fdp.convert(plot=False)

denoise_eval = fdp.find_denoising_method(data=df_abs, methods="FTIR", n_samples=50, plot=False)
denoise_method = fdp.best_denoising_methods(denoise_eval, top_n=1).iloc[0]["method"]

df_denoised = fdp._get_denoised_data(denoising_method=denoise_method, plot=False)
rfzn, nar, snr = fdp.find_baseline_method(data=df_denoised, baseline_methods="FTIR", n_samples=50, plot=False)
baseline_method = fdp.best_baseline_method(rfzn, nar, snr, top_n=1).iloc[0]["method"]

df_atm = fdp._get_atmosphere_corrected_data(
    denoising_method=denoise_method,
    baseline_correction_method=baseline_method,
    interpolate_method="zero",
    plot=False,
)

norm_scores = fdp.find_normalization_method(data=df_atm, methods="FTIR", n_splits=5)
norm_method = norm_scores.iloc[0]["method"]

df_processed = fdp._get_normalized_data(
    denoising_method=denoise_method,
    baseline_correction_method=baseline_method,
    interpolate_method="zero",
    normalization_method=norm_method,
    plot=False,
)

# Analyze
fda = FTIRdataanalysis(df_processed, label_column="type")
fda.plot_pca()
fda.ml_prepare_data(test_size=0.2)
results = fda.run_all_models(plot_comparison=False)
print(results.sort_values("test_f1", ascending=False).head())
```

## Bioinformatics use cases

While `xpectrass` includes plastic-focused example datasets, the same workflow is directly applicable to biological FTIR data when schema and preprocessing assumptions are satisfied:

- microbial phenotype or strain classification
- disease-state or treatment-response stratification
- biofluid fingerprinting
- biomaterial and biopolymer spectral profiling

The key advantage is not a single model, but a reproducible decision trail from raw spectra to selected preprocessing methods and final model metrics.

## Explainability and auditability

In bioinformatics environments, traceability is often as important as raw accuracy. `xpectrass` supports this by combining:

- method-ranking tables for preprocessing stages
- comparative model performance outputs
- SHAP-based global and local feature contribution analysis

This helps teams defend model behavior in technical review and makes it easier to detect when predictions are driven by plausible spectral regions versus spurious patterns.

## Practical considerations

- Keep train/test separation strict before optimization and tuning.
- Use consistent acquisition settings across batches when possible.
- Treat derivatives as optional: they can improve class separation but can also amplify noise when smoothing is inadequate.
- Validate preprocessing choices on representative subsets before scaling to full cohorts.

## Conclusion

For bioinformatics teams using FTIR data, `xpectrass` provides a structured and reproducible path from raw spectra to interpretable machine-learning results. Its evaluation-first preprocessing design is especially useful in biological datasets where signal quality and sample variability can otherwise obscure true biological patterns.
