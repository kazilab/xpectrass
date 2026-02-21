# Xpectrass: A Comprehensive Python Library for FTIR Spectral Preprocessing, Analysis, and Machine Learning Classification

---

## Authors

**Author 1** [Corresponding Author]
Data Analysis Team, KaziLab.se
Email: xpectrass@kazilab.se

---

## Abstract

Fourier-Transform Infrared (FTIR) spectroscopy is a powerful analytical technique widely used for material identification, quality control, and environmental monitoring. However, the preprocessing and analysis of FTIR spectral data require significant expertise and involve multiple complex steps including baseline correction, denoising, normalization, and classification. We present **Xpectrass**, an open-source Python library that provides a comprehensive, end-to-end pipeline for FTIR spectral data preprocessing, statistical analysis, and machine learning classification. Xpectrass implements an evaluation-first methodology that enables researchers to systematically compare and select optimal preprocessing parameters using quantitative metrics. The library includes over 50 baseline correction methods, 7 denoising algorithms, 17+ normalization techniques, and 20+ machine learning classifiers with model explainability via SHAP values. Additionally, Xpectrass bundles six published FTIR datasets for benchmarking and reproducible research. The software is designed to be accessible to both novice users through sensible defaults and expert users through extensive customization options. Xpectrass is freely available under the MIT license and aims to democratize access to advanced FTIR spectral analysis methods.

**Keywords:** FTIR spectroscopy, spectral preprocessing, baseline correction, machine learning, Python, plastic classification, chemometrics

---

## 1. Introduction

### 1.1 Background

Fourier-Transform Infrared (FTIR) spectroscopy is a fundamental analytical technique that measures the absorption of infrared radiation by chemical bonds in materials [1]. The resulting spectral fingerprints enable identification and characterization of polymers, organic compounds, pharmaceuticals, biological samples, and environmental contaminants [2]. In recent years, FTIR spectroscopy has gained particular importance in microplastics research, where rapid and accurate identification of polymer types is essential for understanding plastic pollution in marine and terrestrial environments [3,4].

The raw spectral data obtained from FTIR instruments typically require extensive preprocessing before meaningful analysis or classification can be performed. Common preprocessing steps include atmospheric correction (removal of CO₂ and H₂O absorption bands), baseline correction (compensation for instrumental drift and scattering effects), denoising (reduction of random measurement noise), and normalization (standardization across samples) [5]. Each preprocessing step involves selecting from multiple available algorithms, each with its own parameters and assumptions. The choice of preprocessing methods significantly impacts downstream analysis results, yet optimal method selection often relies on trial-and-error or expert knowledge [6].

### 1.2 Statement of Need

Despite the widespread use of FTIR spectroscopy, researchers face several challenges in spectral data processing:

1. **Fragmented software landscape:** Existing tools for spectral preprocessing are scattered across multiple packages, each implementing only a subset of available methods, making comprehensive comparison difficult.

2. **Parameter optimization:** Selecting appropriate preprocessing parameters often requires extensive domain expertise and manual experimentation.

3. **Reproducibility barriers:** Lack of standardized datasets and workflows limits reproducibility across studies.

4. **Integration gaps:** Preprocessing tools are typically separate from downstream analysis and machine learning pipelines, requiring researchers to bridge multiple software ecosystems.

5. **Accessibility:** Advanced chemometric methods are often implemented in commercial software with prohibitive licensing costs, limiting access for researchers in resource-constrained settings.

### 1.3 Purpose and Scope

Xpectrass addresses these challenges by providing a unified, open-source Python library that integrates all stages of FTIR spectral analysis—from raw data preprocessing to machine learning classification with model explainability. The library is specifically designed with an **evaluation-first methodology** that enables researchers to systematically compare preprocessing methods using quantitative metrics before applying them to their data. This approach replaces ad-hoc parameter selection with evidence-based decision making.

---

## 2. Software Description

### 2.1 Architecture Overview

Xpectrass is organized into two main classes that represent the complete analytical workflow:

1. **FTIRdataprocessing:** Implements a modular 9-step preprocessing pipeline with built-in evaluation capabilities at each step.

2. **FTIRdataanalysis:** Provides statistical analysis, dimensionality reduction, clustering, and machine learning classification with model interpretation.

The library follows a state-preserving design pattern where intermediate results at each preprocessing step are stored and accessible, enabling quality control, comparison, and selective re-processing (Figure 1).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Xpectrass Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    FTIRdataprocessing                             │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Step 1: Data Loading & Validation                          │  │  │
│  │  │  Step 2: Transmittance ↔ Absorbance Conversion              │  │  │
│  │  │  Step 3: Atmospheric Correction (CO₂/H₂O removal)           │  │  │
│  │  │  Step 4: Baseline Correction (50+ methods)                  │  │  │
│  │  │  Step 5: Denoising (7 methods)                              │  │  │
│  │  │  Step 6: Normalization (17+ methods)                        │  │  │
│  │  │  Step 7: Spectral Derivatives (1st, 2nd, gap)               │  │  │
│  │  │  Step 8: Region Selection                                   │  │  │
│  │  │  Step 9: Export & Visualization                             │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                         ↓ Processed Data                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     FTIRdataanalysis                              │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Exploratory Analysis: Mean spectra, heatmaps, overlays     │  │  │
│  │  │  Dimensionality Reduction: PCA, t-SNE, UMAP, PLS-DA, OPLS-DA│  │  │
│  │  │  Statistical Analysis: ANOVA, correlation analysis          │  │  │
│  │  │  Clustering: K-means, hierarchical                          │  │  │
│  │  │  Classification: 20+ ML models with hyperparameter tuning   │  │  │
│  │  │  Explainability: SHAP values, feature importance            │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
*Figure 1: Xpectrass architectural overview showing the two-stage pipeline design.*

### 2.2 Preprocessing Pipeline (FTIRdataprocessing)

The preprocessing pipeline implements the following steps:

#### 2.2.1 Data Loading and Validation

Xpectrass accepts spectral data in tabular format (Pandas or Polars DataFrames) where rows represent individual spectra and columns represent wavenumber measurements plus a label column. The library performs automatic detection of wavenumber columns and validates data integrity.

#### 2.2.2 Transmittance/Absorbance Conversion

FTIR instruments may output data as transmittance (%T) or absorbance (A) units. Xpectrass provides bidirectional conversion using the Beer-Lambert law relationship: A = -log₁₀(T/100). The library automatically detects the current data type and converts as needed.

#### 2.2.3 Atmospheric Correction

Ambient CO₂ and water vapor produce characteristic absorption bands that can interfere with sample spectra. Xpectrass implements atmospheric correction through interpolation-based removal of affected regions, with support for spline, linear, and polynomial interpolation methods.

#### 2.2.4 Baseline Correction

Baseline distortions arise from instrumental drift, sample scattering, and fluorescence. Xpectrass provides access to over 50 baseline correction algorithms through integration with the PyBaselines library [7], including:

- **Polynomial methods:** Polynomial, Modified polynomial, I-ModPoly
- **Whittaker-based:** ALS, airPLS, arPLS, drPLS, iALS, psalsa
- **Morphological:** MOR, IMOR, Top-hat, Rolling ball
- **Spline-based:** Mixture Model, PSPLINE, Corner-cutting
- **Window-based:** SNIP, noise-median
- **Optimization-based:** Custom cost function minimization

##### Baseline Correction Evaluation Framework

A critical innovation in Xpectrass is the evaluation-first approach to baseline correction. Rather than arbitrarily selecting a method, researchers can systematically compare algorithms using three complementary metrics computed across user-defined "flat zones" (spectral regions known to contain only baseline, no sample absorption peaks):

**1. Residual Flat-Zone Noise (RFZN)**

RFZN quantifies the residual noise level in baseline-only regions after correction. It is computed as the root-mean-square (RMS) of corrected intensities within flat zones:

$$\text{RFZN} = \sqrt{\frac{1}{N_{\text{flat}}} \sum_{i \in \text{flat zones}} y_{\text{corr},i}^2}$$

where $y_{\text{corr},i}$ represents corrected intensity values and $N_{\text{flat}}$ is the number of points in flat zones. **Lower RFZN values indicate better baseline correction** with less residual structure in regions that should be flat. Typical acceptable values are < 0.01 absorbance units.

**2. Negative Area Ratio (NAR)**

NAR measures the proportion of negative absorbance values introduced by over-correction. Since absorbance cannot physically be negative, this metric detects baseline over-estimation:

$$\text{NAR} = \frac{\sum_{i: y_{\text{corr},i} < 0} |y_{\text{corr},i}|}{\sum_{i} |y_{\text{corr},i}|}$$

NAR ranges from 0 (no negative values) to 1 (all negative). **Lower NAR values indicate better correction** without physically invalid negative absorbance. Acceptable values are typically < 0.05 (5%).

**3. Signal-to-Noise Ratio (SNR)**

SNR assesses the preservation of spectral information relative to noise after correction:

$$\text{SNR} = \frac{h_{\text{peak}}}{\sigma_{\text{noise}}}$$

where $h_{\text{peak}}$ is the maximum peak height (either global or within user-specified diagnostic regions) and $\sigma_{\text{noise}}$ is the RMS noise estimated from flat zones. **Higher SNR values indicate better signal preservation**. Values > 10 are generally considered acceptable.

**Composite Scoring for Method Selection**

To facilitate automated method selection, Xpectrass computes a composite score combining all three metrics. Each metric is first normalized to a [0, 1] range across all evaluated methods, with directionality adjusted so that higher scores are always better:

$$\text{RFZN}_{\text{score}} = 1 - \frac{\text{RFZN} - \text{RFZN}_{\min}}{\text{RFZN}_{\max} - \text{RFZN}_{\min}}$$

$$\text{NAR}_{\text{score}} = 1 - \frac{\text{NAR} - \text{NAR}_{\min}}{\text{NAR}_{\max} - \text{NAR}_{\min}}$$

$$\text{SNR}_{\text{score}} = \frac{\text{SNR} - \text{SNR}_{\min}}{\text{SNR}_{\max} - \text{SNR}_{\min}}$$

The composite score is then computed as a weighted average, with an additional "pass rate" component that rewards methods performing consistently across all samples:

$$S_{\text{composite}} = 0.3 \cdot \text{RFZN}_{\text{score}} + 0.3 \cdot \text{NAR}_{\text{score}} + 0.3 \cdot \text{SNR}_{\text{score}} + 0.1 \cdot \text{PassRate}$$

where PassRate is the fraction of samples meeting all quality thresholds (RFZN < 0.01, NAR < 0.05, SNR > 10).

#### 2.2.5 Denoising

Seven denoising algorithms are implemented:

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| Savitzky-Golay | Polynomial smoothing preserving spectral features | window_length, polyorder |
| Wavelet | Multi-resolution decomposition with thresholding | wavelet type, level, threshold_mode |
| Median | Non-linear smoothing robust to outliers | kernel_size |
| Gaussian | Linear smoothing with Gaussian kernel | sigma |
| Whittaker | Penalized least squares smoother | lambda, d |
| Moving Average | Simple uniform filter | window size |
| Low-pass (FFT) | Frequency-domain Butterworth filtering | cutoff, order |

##### Denoising Evaluation Framework

Selecting the optimal denoising method requires balancing noise reduction against signal distortion. Xpectrass evaluates denoising performance using three complementary metrics:

**1. Signal-to-Noise Ratio Improvement (SNR)**

SNR quantifies the improvement in signal quality by comparing signal power to residual noise power:

$$\text{SNR}_{\text{dB}} = 10 \cdot \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

where signal power is estimated from the variance of the denoised spectrum and noise power from the residual (difference between raw and denoised):

$$P_{\text{signal}} = \text{Var}(y_{\text{denoised}})$$

$$P_{\text{noise}} = \text{Var}(y_{\text{raw}} - y_{\text{denoised}})$$

Alternatively, when flat regions (baseline-only zones) are specified, noise is estimated directly from residuals in those regions. **Higher SNR values (in dB) indicate better noise reduction**. Values > 10 dB typically indicate effective denoising.

**2. Smoothness**

Smoothness measures the reduction in high-frequency oscillations, computed as the inverse variance of the second derivative:

$$\text{Smoothness} = \frac{1}{\text{Var}(\Delta^2 y_{\text{denoised}}) + \epsilon}$$

where $\Delta^2 y$ represents the discrete second derivative (second-order finite differences):

$$\Delta^2 y_i = y_{i+1} - 2y_i + y_{i-1}$$

and $\epsilon = 10^{-10}$ prevents division by zero. **Higher smoothness values indicate more effective noise reduction** with fewer high-frequency oscillations. However, excessively high smoothness may indicate over-smoothing and loss of spectral detail.

**3. Fidelity**

Fidelity measures the preservation of the original spectral shape using Pearson correlation:

$$\text{Fidelity} = \rho(y_{\text{raw}}, y_{\text{denoised}}) = \frac{\text{Cov}(y_{\text{raw}}, y_{\text{denoised}})}{\sigma_{y_{\text{raw}}} \cdot \sigma_{y_{\text{denoised}}}}$$

Fidelity ranges from -1 to 1, where 1 indicates perfect preservation of spectral shape. **Higher fidelity values (closer to 1) indicate better preservation of original features**. Values > 0.9 are generally acceptable.

**Wavelet Denoising: Universal Threshold**

For wavelet denoising, Xpectrass implements universal thresholding based on robust noise estimation from the finest wavelet coefficients:

$$\hat{\sigma} = \frac{\text{MAD}(d_J)}{0.6745}$$

where MAD is the median absolute deviation of the finest detail coefficients $d_J$, and 0.6745 is the scaling factor for Gaussian noise. The universal threshold is then:

$$\tau = \hat{\sigma} \sqrt{2 \ln(N)}$$

where $N$ is the signal length. Coefficients below $\tau$ are attenuated (soft thresholding) or zeroed (hard thresholding).

**Composite Scoring for Denoising Method Selection**

Similar to baseline correction, Xpectrass computes a composite score for denoising methods. Each metric is normalized to [0, 1] using min-max scaling across methods, with time efficiency also considered:

$$S_{\text{composite}} = 0.30 \cdot \text{SNR}_{\text{score}} + 0.25 \cdot \text{Smoothness}_{\text{score}} + 0.30 \cdot \text{Fidelity}_{\text{score}} + 0.05 \cdot \text{Time}_{\text{score}} + 0.10 \cdot \text{PassRate}$$

where Time_score is inverted (faster is better) and PassRate rewards methods consistently meeting quality thresholds across samples.

#### 2.2.6 Normalization

Xpectrass implements 17+ normalization methods across several categories:

| Category | Methods | Description |
|----------|---------|-------------|
| Standard | Min-max, Z-score, Vector | Common statistical normalizations |
| Spectroscopy-specific | SNV, MSC, EMSC | Scatter correction methods |
| Area-based | Area, Peak | Intensity-preserving normalizations |
| Robust | Median-MAD, IQR, Robust-SNV | Outlier-resistant methods |
| Advanced | PQN, Entropy | Probabilistic and information-theoretic |

**Key Normalization Methods:**

*Standard Normal Variate (SNV):*
$$y_{\text{SNV},i} = \frac{y_i - \bar{y}}{\sigma_y}$$

where $\bar{y}$ is the mean and $\sigma_y$ is the standard deviation of the spectrum.

*Vector Normalization (L2):*
$$y_{\text{vec},i} = \frac{y_i}{\|y\|_2} = \frac{y_i}{\sqrt{\sum_j y_j^2}}$$

*Probabilistic Quotient Normalization (PQN):*
$$y_{\text{PQN},i} = \frac{y_i}{\text{median}_j\left(\frac{y_j}{r_j}\right)}$$

where $r$ is a reference spectrum (typically median or mean across training samples).

##### Normalization Evaluation Framework

Normalization evaluation in Xpectrass employs a comprehensive multi-metric framework combining supervised classification, unsupervised clustering, and spectral consistency measures. This evaluation is designed to identify methods that improve downstream analysis performance while preserving meaningful spectral information.

**1. Supervised Classification Metrics**

Xpectrass evaluates normalization methods using stratified k-fold cross-validation with a logistic regression classifier. Critically, normalization is applied *within* each fold to prevent data leakage:

*Macro F1-Score:*
$$F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

where $P_c$ and $R_c$ are precision and recall for class $c$, and $C$ is the number of classes.

*Balanced Accuracy:*
$$\text{BalAcc} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{N_c}$$

where $TP_c$ is true positives and $N_c$ is total samples for class $c$.

**2. Clustering Quality Metrics**

Normalization quality is also assessed through unsupervised clustering agreement with known labels:

*Adjusted Rand Index (ARI):*
$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

ARI measures agreement between cluster assignments and true labels, adjusted for chance. Values range from -1 to 1, where 1 indicates perfect agreement. ARI is computed for both K-means and hierarchical clustering with cosine distance.

*Normalized Mutual Information (NMI):*
$$\text{NMI} = \frac{2 \cdot I(Y; \hat{Y})}{H(Y) + H(\hat{Y})}$$

where $I(Y; \hat{Y})$ is mutual information between true labels $Y$ and cluster assignments $\hat{Y}$, and $H(\cdot)$ is entropy.

**3. Cluster Stability**

Cluster stability measures the reproducibility of clustering results under data perturbation:

$$\text{Stability} = \frac{1}{B} \sum_{b=1}^{B} \text{ARI}(\hat{Y}_{\text{ref}}, \hat{Y}_b)$$

where $B$ bootstrap samples (80% of data) are clustered and compared against a reference clustering on the full dataset. Higher stability indicates more robust normalization.

**4. Within-Group Spectral Consistency (SAM)**

The Spectral Angle Mapper (SAM) measures angular similarity between spectra from the same class:

$$\text{SAM}(a, b) = \arccos\left(\frac{a \cdot b}{\|a\|_2 \|b\|_2}\right)$$

Mean within-group SAM is computed across all pairs of spectra within each class:

$$\overline{\text{SAM}}_{\text{within}} = \frac{1}{|P|} \sum_{(i,j) \in P} \text{SAM}(y_i, y_j)$$

where $P$ is the set of within-class pairs. **Lower SAM values indicate better spectral consistency** within classes, suggesting the normalization preserves class-specific features.

**5. Internal Clustering Metrics**

Additional metrics assess cluster quality independent of labels:

*Silhouette Score (cosine):*
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance for sample $i$.

*Davies-Bouldin Index:* Lower values indicate better-defined clusters.

*Calinski-Harabasz Index:* Higher values indicate denser, well-separated clusters.

**Multi-Perspective Composite Scoring**

Xpectrass provides four scoring schemes to accommodate different analysis scenarios:

1. **Combined Score** (recommended for labeled data):
$$S_{\text{combined}} = 0.20 \cdot F1_z + 0.20 \cdot \text{BalAcc}_z + 0.20 \cdot \text{ARI}_z + 0.20 \cdot \text{Stability}_z + 0.10 \cdot \text{SAM}_z + 0.10 \cdot \text{NMI}_z$$

2. **Unsupervised Score** (for unlabeled data):
$$S_{\text{unsup}} = 0.20 \cdot \text{ARI}_z + 0.25 \cdot \text{Stability}_z + 0.15 \cdot \text{Silhouette}_z + 0.15 \cdot \text{NMI}_z + 0.15 \cdot \text{ARI}_{\text{agg},z} + 0.10 \cdot \text{SAM}_z$$

3. **Comprehensive Score**: Equal-weighted average of all 11 metrics.

4. **Efficient Score**: Incorporates computational time as 15% of the score.

All metrics are converted to z-scores before combination, with directionality adjusted so higher values always indicate better performance (e.g., SAM and Davies-Bouldin are negated before z-scoring).

#### 2.2.7 Spectral Derivatives

First and second derivatives can enhance spectral features and reduce baseline effects. Xpectrass implements:
- Standard Savitzky-Golay derivatives
- Gap derivatives for noisy data
- Configurable window size and polynomial order

#### 2.2.8 Region Selection

Thirteen predefined spectral regions relevant to plastic/polymer analysis are included, with support for custom region definition.

### 2.3 Analysis and Machine Learning (FTIRdataanalysis)

#### 2.3.1 Exploratory Visualization

- Mean spectra with confidence intervals
- Spectral overlays by class
- Heatmaps and waterfall plots
- Sample distribution visualizations

#### 2.3.2 Dimensionality Reduction

| Method | Description |
|--------|-------------|
| PCA | Principal Component Analysis |
| t-SNE | t-Distributed Stochastic Neighbor Embedding |
| UMAP | Uniform Manifold Approximation and Projection |
| PLS-DA | Partial Least Squares Discriminant Analysis |
| OPLS-DA | Orthogonal PLS-DA for improved class separation |

#### 2.3.3 Statistical Analysis

- One-way ANOVA for spectral region comparison
- Correlation analysis between spectral features
- Peak detection and assignment

#### 2.3.4 Clustering

- K-means clustering with elbow method optimization
- Hierarchical clustering with dendrogram visualization
- Silhouette analysis for cluster quality assessment

#### 2.3.5 Machine Learning Classification

Xpectrass includes 20+ classification models across multiple families:

| Family | Models |
|--------|--------|
| Ensemble | Random Forest, Extra Trees, AdaBoost, Gradient Boosting |
| Boosting | XGBoost (3 configurations), LightGBM (3 configurations) |
| SVM | Linear, RBF, Polynomial kernels |
| Linear | Logistic Regression, Ridge, SGD Classifier |
| Neighbors | KNN (k=3, 5, 7, 9) |
| Neural Network | MLP (various architectures) |
| Naive Bayes | Gaussian, Multinomial |
| Discriminant | LDA, QDA |

Each model is evaluated using:
- Accuracy, precision, recall, F1-score (weighted)
- Stratified k-fold cross-validation
- Training and inference time benchmarks
- Overfitting analysis (train vs. test performance)
- Confusion matrix visualization

#### 2.3.6 Hyperparameter Tuning

Random search optimization with customizable parameter grids enables model optimization without exhaustive grid search.

#### 2.3.7 Model Explainability

SHAP (SHapley Additive exPlanations) values provide both global and local interpretability:
- Feature importance rankings
- Summary plots showing feature effects
- Individual prediction explanations
- Decision plots for sample-level interpretation

---

## 3. Bundled Datasets

To promote reproducibility and provide benchmarking capabilities, Xpectrass bundles six published FTIR datasets from peer-reviewed studies:

| Dataset | Reference | Description | Samples |
|---------|-----------|-------------|---------|
| Jung et al. 2018 | [8] | Baseline plastic identification | ~1,000 |
| Kedzierski et al. 2019 (A) | [9] | Microplastics analysis (configuration A) | ~2,500 |
| Kedzierski et al. 2019 (B) | [9] | Microplastics analysis (configuration B) | ~3,500 |
| Frond et al. 2021 | [10] | Extended polymer types | ~800 |
| Villegas-Camacho et al. 2024 (C4) | [11] | Comprehensive analysis (4-class) | ~5,000 |
| Villegas-Camacho et al. 2024 (C8) | [11] | Comprehensive analysis (8-class) | ~4,000 |

All datasets are provided in compressed CSV format (XZ compression) and can be loaded with single function calls.

---

## 4. Illustrative Examples

### 4.1 Minimal Working Example

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_jung_2018

# Load bundled dataset
df = load_jung_2018()

# Initialize and run preprocessing pipeline
ftir = FTIRdataprocessing(df, label_column="type")
ftir.run()  # Applies default preprocessing chain

# Initialize analysis with preprocessed data
analysis = FTIRdataanalysis(ftir.df_norm, label_column="type")

# Run all classification models
results = analysis.run_all_models()

# Display top 5 performing models
print(results.nlargest(5, 'f1_score')[['model', 'f1_score', 'accuracy']])
```

### 4.2 Evaluation-First Preprocessing

```python
from xpectrass import FTIRdataprocessing
from xpectrass.data import load_jung_2018

df = load_jung_2018()
ftir = FTIRdataprocessing(df, label_column="type")

# Evaluate baseline correction methods on representative sample
ftir.evaluate_baseline_methods(
    sample_index=0,
    methods=['airpls', 'arpls', 'asls', 'mor', 'snip'],
    plot=True
)

# Select best method based on metrics and apply
ftir.apply_baseline_correction(method='airpls', lam=1e6)

# Evaluate denoising methods
ftir.evaluate_denoising_methods(
    sample_index=0,
    methods=['savgol', 'wavelet', 'gaussian'],
    plot=True
)

# Apply selected method
ftir.apply_denoising(method='savgol', window_length=11, polyorder=3)
```

### 4.3 Complete Classification Pipeline with SHAP

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_villegas_2024_c4

# Load and preprocess
df = load_villegas_2024_c4()
ftir = FTIRdataprocessing(df, label_column="label")
ftir.run(
    baseline_method='airpls',
    denoise_method='savgol',
    normalize_method='snv'
)

# Analysis
analysis = FTIRdataanalysis(ftir.df_norm, label_column="label")

# Dimensionality reduction visualization
analysis.plot_pca(n_components=3)
analysis.plot_tsne(perplexity=30)
analysis.plot_umap(n_neighbors=15)

# Train best model with hyperparameter tuning
best_model = analysis.tune_model('random_forest', n_iter=50)

# Generate SHAP explanations
analysis.explain_model_shap(best_model, plot_type='summary')
analysis.explain_model_shap(best_model, plot_type='bar')
```

---

## 5. Implementation Details

### 5.1 Technology Stack

Xpectrass is implemented in Python 3.8+ and leverages the following key dependencies:

| Category | Packages |
|----------|----------|
| Data handling | NumPy, Pandas, Polars |
| Scientific computing | SciPy, PyBaselines, PyWavelets |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Dimensionality reduction | UMAP-learn |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |

### 5.2 Performance Considerations

- **Parallel processing:** Multi-core support via joblib (configurable `n_jobs` parameter)
- **Memory efficiency:** Lazy loading of bundled datasets; XZ compression reduces storage requirements by ~75%
- **Vectorized operations:** NumPy-based implementations for computational efficiency

### 5.3 Code Quality

- Type hints throughout for IDE support and static analysis
- Comprehensive docstrings following NumPy conventions
- Test suite with pytest framework
- Pre-commit hooks for code formatting (Black, isort) and linting (flake8, mypy)

---

## 6. Impact and Applications

### 6.1 Target Applications

Xpectrass is designed for:

1. **Microplastics research:** Rapid identification of polymer types in environmental samples
2. **Quality control:** Verification of material identity in manufacturing settings
3. **Pharmaceutical analysis:** Identification and quality assessment of drug substances
4. **Food science:** Detection of adulterants and contaminants
5. **Forensic science:** Material identification in criminal investigations
6. **Education:** Teaching spectral analysis and chemometrics concepts

### 6.2 Comparison with Existing Tools

| Feature | Xpectrass | OPUS | Spectragryph | Orange Spectroscopy |
|---------|-----------|------|--------------|---------------------|
| Open source | ✓ | ✗ | ✗ | ✓ |
| Python API | ✓ | ✗ | ✗ | ✓ |
| Baseline methods | 50+ | ~10 | ~5 | ~10 |
| ML classification | 20+ models | Limited | ✗ | Basic |
| SHAP explainability | ✓ | ✗ | ✗ | ✗ |
| Bundled datasets | 6 | ✗ | ✗ | ✗ |
| Evaluation metrics | ✓ | Limited | Limited | Limited |

### 6.3 Community Adoption

Xpectrass is designed to lower barriers to entry for FTIR analysis by:
- Providing sensible defaults that work for most use cases
- Including comprehensive documentation and tutorials
- Bundling reference datasets for learning and benchmarking
- Supporting interactive exploration via Jupyter notebooks

---

## 7. Quality Control

### 7.1 Testing

Xpectrass includes a test suite covering:
- Unit tests for individual preprocessing functions
- Integration tests for complete pipelines
- Validation against known reference implementations

### 7.2 Documentation

Documentation is provided through multiple channels:
- **User guides:** 13 detailed tutorials covering all major features
- **API reference:** Auto-generated from docstrings
- **Example notebooks:** 6 Jupyter notebooks for interactive learning
- **README:** Quick-start guide and installation instructions

### 7.3 Continuous Integration

The project uses GitHub Actions for:
- Automated testing on multiple Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
- Code quality checks (linting, type checking)
- Documentation building and deployment

---

## 8. Availability and Requirements

### 8.1 Software Availability

- **Repository:** https://github.com/kazilab/xpectrass
- **Package:** https://pypi.org/project/xpectrass/
- **Documentation:** https://xpectrass.readthedocs.io
- **License:** MIT License

### 8.2 System Requirements

| Requirement | Specification |
|-------------|---------------|
| Python version | 3.8 or higher |
| Operating system | Windows, macOS, Linux |
| Memory | 4 GB minimum, 8 GB recommended |
| Disk space | ~100 MB (including bundled datasets) |

### 8.3 Installation

```bash
pip install xpectrass
```

For development installation:
```bash
git clone https://github.com/kazilab/xpectrass.git
cd xpectrass
pip install -e ".[dev,docs]"
```

---

## 9. Conclusions

Xpectrass provides a comprehensive, open-source solution for FTIR spectral data preprocessing, analysis, and machine learning classification. By implementing an evaluation-first methodology, the library enables researchers to make evidence-based decisions about preprocessing parameters rather than relying on trial-and-error. The integration of 50+ baseline correction methods, multiple denoising and normalization algorithms, and 20+ machine learning classifiers with SHAP-based explainability provides a complete analytical workflow in a single package.

The inclusion of bundled reference datasets, extensive documentation, and example notebooks makes Xpectrass accessible to researchers across experience levels while maintaining the flexibility needed for advanced applications. We anticipate that Xpectrass will facilitate reproducible research in FTIR spectroscopy and contribute to standardization of preprocessing and analysis workflows in the field.

Future development will focus on expanding the bundled dataset collection, implementing additional machine learning architectures (including deep learning models), and developing graphical user interface components for users who prefer non-programmatic interaction.

---

## Acknowledgments

[To be completed by authors]

---

## Author Contributions

**CRediT Author Statement:**

[To be completed by authors - example format below]

**Author 1:** Conceptualization, Methodology, Software, Writing – Original Draft, Writing – Review & Editing

---

## Declaration of Competing Interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## References

[1] Griffiths, P.R., de Haseth, J.A. (2007). Fourier Transform Infrared Spectrometry (2nd ed.). Wiley-Interscience.

[2] Stuart, B.H. (2004). Infrared Spectroscopy: Fundamentals and Applications. Wiley.

[3] Araujo, C.F., Nolasco, M.M., Ribeiro, A.M.P., Ribeiro-Claro, P.J.A. (2018). Identification of microplastics using Raman spectroscopy: Latest developments and future prospects. Water Research, 142, 426-440.

[4] Xu, J.L., Thomas, K.V., Luo, Z., Gowen, A.A. (2019). FTIR and Raman imaging for microplastics analysis: State of the art, challenges and prospects. TrAC Trends in Analytical Chemistry, 119, 115629.

[5] Rinnan, Å., van den Berg, F., Engelsen, S.B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. TrAC Trends in Analytical Chemistry, 28(10), 1201-1222.

[6] Gerretzen, J., Szymańska, E., Jansen, J.J., Bart, J., van Manen, H.J., van den Heuvel, E.R., Buydens, L.M.C. (2015). Simple and effective way for data preprocessing selection based on design of experiments. Analytical Chemistry, 87(24), 12096-12103.

[7] Erb, D. (2022). pybaselines: A Python library of algorithms for the baseline correction of experimental data. GitHub repository. https://github.com/derb12/pybaselines

[8] Jung, M.R., Horgen, F.D., Orski, S.V., Rodriguez, V., Beers, K.L., Balazs, G.H., Jones, T.T., Work, T.M., Brignac, K.C., Royer, S.J., Hyrenbach, K.D., Jensen, B.A., Lynch, J.M. (2018). Validation of ATR FT-IR to identify polymers of plastic marine debris, including those ingested by marine organisms. Marine Pollution Bulletin, 127, 704-716.

[9] Kedzierski, M., Falcou-Préfol, M., Kerber, M., Henry, M., Pedrotti, M.L., Bruzaud, S. (2019). A machine learning algorithm for high throughput identification of FTIR spectra: Application on microplastics collected in the Mediterranean Sea. Chemosphere, 234, 242-251.

[10] De Frond, H., Rubin, E., Chelazzi, D., Cincinelli, A., Mejanelle, L., Liboiron, M., Prata, J.C., Scholz-Böttcher, B.M., Shen, M., Sintes, M., Völker, C., Amaral-Zettler, L.A., Bank, M.S., Brown, D., Brander, S.M., Coffin, S., Cole, M., Dunaev, T., Fechner, L.C., Heydebreck, F., Holland, E.R., Koelmans, A.A., Mintenig, S., Munier, B., Munno, K., Napper, I.E., Orro, K., Palardy, J.E., Pham, C.K., Primpke, S., Quarino, L., Rochman, C.M., Weiss, L., Zettler, E.R., Hammoud, S.A. (2021). Standardized protocol for ATR-FTIR spectroscopy verification of microplastic polymer types. Marine Pollution Bulletin, 165, 112134.

[11] Villegas-Camacho, A., Ramirez-Nunez, A.L., Medina-Ramirez, I.E., Silva-Martinez, S., Cardenas-Chavez, D.L. (2024). Machine Learning approach for microplastic FTIR identification in aquatic environments. Chemosphere, 348, 140772.

---

## Supplementary Material

### Table S1: Complete List of Baseline Correction Methods

| Method | Category | Key Parameters |
|--------|----------|----------------|
| airPLS | Whittaker | lambda, differences |
| arPLS | Whittaker | lambda, ratio |
| asLS | Whittaker | lambda, p |
| drPLS | Whittaker | lambda, eta |
| iALS | Whittaker | lambda |
| psalsa | Whittaker | lambda, p |
| modpoly | Polynomial | poly_order, tol |
| imodpoly | Polynomial | poly_order, tol |
| penalized_poly | Polynomial | poly_order |
| mor | Morphological | half_window |
| imor | Morphological | half_window |
| rolling_ball | Morphological | half_window |
| tophat | Morphological | half_window |
| snip | Window | max_half_window |
| noise_median | Window | half_window |
| mixture_model | Spline | half_window, lam |
| pspline_asls | Spline | lam, p |
| corner_cutting | Spline | max_iter |
| ... | ... | ... |

### Table S2: Machine Learning Model Configurations

| Model | Default Parameters |
|-------|-------------------|
| Random Forest | n_estimators=100, max_depth=None |
| XGBoost | n_estimators=100, learning_rate=0.1 |
| LightGBM | n_estimators=100, learning_rate=0.1 |
| SVM (RBF) | C=1.0, gamma='scale' |
| KNN | n_neighbors=5, weights='uniform' |
| MLP | hidden_layers=(100,), activation='relu' |
| ... | ... |

---

*Manuscript prepared for submission to [Journal Name]*

*Word count: ~5,500 (main text, excluding equations and tables)*

