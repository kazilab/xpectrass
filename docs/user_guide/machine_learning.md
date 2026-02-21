# Machine Learning

The `FTIRdataanalysis` class provides comprehensive machine learning capabilities for FTIR plastic classification, including model evaluation, hyperparameter tuning, and explainability analysis.

## Overview

The machine learning workflow includes:

1. **Data preparation**: Train/test split and scaling
2. **Model evaluation**: Test 20+ classification algorithms
3. **Model comparison**: Visualize performance metrics
4. **Hyperparameter tuning**: Optimize top models
5. **Model interpretation**: SHAP explainability analysis

```python
from xpectrass import FTIRdataanalysis

# Initialize with preprocessed data
analysis = FTIRdataanalysis(processed_df, label_column="type", random_state=42)
```

## Data Preparation

### Train/Test Split

```python
# Prepare data for machine learning
analysis.ml_prepare_data(
    test_size=0.2           # 20% for testing
)

print(f"Training samples: {len(analysis.y_train)}")
print(f"Test samples: {len(analysis.y_test)}")
print(f"Features: {analysis.x_train_scaled.shape[1]}")
print(f"Classes: {analysis.class_names}")
```

**Attributes created:**
- `x_train_scaled`, `x_test_scaled`: Standardized feature matrices
- `x_train_raw`, `x_test_raw`: Raw (unscaled) feature matrices
- `y_train`, `y_test`: Labels
- `class_names`: Unique class labels
- `scaler`: Fitted StandardScaler
- `label_encoder`: Fitted LabelEncoder
- `wavenumbers`: Array of wavenumber values
- `dir_`: Dictionary containing all of the above

## Available Models

### View All Models

```python
# See available classification models
models = analysis.available_models()
print(f"Total models: {len(models)}")
for model_name in models:
    print(f"  - {model_name}")
```

### Model Categories

The library includes **20+ classification algorithms** across multiple families:

**Linear Models:**
- Logistic Regression
- Ridge Classifier
- SGD Classifier

**Tree-Based Models:**
- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting

**Ensemble Models:**
- XGBoost (multiple configs)
- LightGBM (multiple configs)

**Naive Bayes:**
- Gaussian Naive Bayes
- Multinomial Naive Bayes

**Support Vector Machines:**
- Linear SVM
- RBF SVM
- Poly SVM

**Nearest Neighbors:**
- K-Nearest Neighbors (multiple K values)

**Neural Networks:**
- Multi-Layer Perceptron (multiple architectures)

**Discriminant Analysis:**
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis

## Running Models

### Run a Single Model

```python
# Run specific model
results = analysis.run_a_model(
    model_name='XGBoost (100)',
    model=None,
    cv_folds=5,
    plot_confusion=True,
    save_plot_path=None,
    print_test_result=True
)
overall, per_class, confusion_matrix = results
print(f"Accuracy: {overall['accuracy']:.3f}")
print(f"F1 (weighted): {overall['f1_weighted']:.3f}")
```

### Run All Models

Evaluate all available models with cross-validation:

```python
# Run all models
results = analysis.run_all_models(
    test_size=0.2,
    plot_comparison=True,
    accuracy_threshold=0.9,
    top_n_methods=20,
    save_plot_path=None
)

# View results sorted by F1 score
print(results.sort_values('test_f1', ascending=False))

# Save results
analysis.results_all = results  # Stored for later use
```

**Results DataFrame includes:**
- `model_name`: Model name
- `test_accuracy`: Test set accuracy
- `test_precision`: Weighted precision
- `test_recall`: Weighted recall
- `test_f1`: Weighted F1 score
- `y_pred`: Predicted labels
- `y_proba`: Predicted probabilities
- `train_accuracy`: Training accuracy
- `cv_mean`: Mean cross-validation score
- `cv_std`: CV standard deviation
- `overfit_gap`: Difference between train and test accuracy
- `train_time`: Training time (seconds)
- `pred_time`: Prediction time (seconds)

### View Top Models

```python
# Get top 10 models by F1 score
top_models = results.nlargest(10, 'test_f1')
print(top_models[['model_name', 'test_accuracy', 'test_f1', 'cv_mean']])
```

## Model Comparison Visualization

When `plot_comparison=True` (the default), `run_all_models()` automatically generates four visualization plots:

### Model Comparison Plot
- Bar plot of model performance sorted by test accuracy
- Shows top N models (controlled by `top_n_methods` parameter)
- Error bars showing CV standard deviation
- Color-coded by performance

### Family Comparison Plot
- Average performance per model family (e.g., tree-based vs ensemble vs linear)
- Violin plots showing distribution within each family
- Best model in each family highlighted

### Efficiency Analysis Plot
- Scatter plot: Training time vs F1 score
- Helps identify fast, accurate models
- Models above the `accuracy_threshold` are highlighted

### Overfitting Analysis Plot
- Train score vs test score comparison
- Diagonal line = perfect generalization
- Points above line indicate overfitting
- Shows top N models (controlled by `top_n_methods` parameter)

## Hyperparameter Tuning

### Tune Top Models

Optimize hyperparameters for best-performing models:

```python
# Tune top 2 models (must run run_all_models() first)
results = analysis.run_all_models()
tuned_results = analysis.model_parameter_tuning(
    number_of_models=2       # Number of top models to tune
)
```

> **Note:** You must call `run_all_models()` before `model_parameter_tuning()`, as it relies on `self.results_all` and `self.dir_` being set.

**Tuning Search Spaces:**

Each model has a predefined search space covering important hyperparameters:

- **Random Forest**: n_estimators, max_depth, min_samples_split, max_features
- **XGBoost**: learning_rate, max_depth, n_estimators, subsample, colsample_bytree
- **LightGBM**: learning_rate, num_leaves, max_depth, feature_fraction
- **SVM**: C, gamma, kernel
- **KNN**: n_neighbors, weights, metric

## Model Interpretation

### SHAP Explainability

Understand which spectral features drive predictions:

```python
# Explain model predictions with SHAP
shap_results = analysis.explain_by_shap(
    model_name='XGBoost (100)',  # Model to explain
    max_display=20,              # Max features to display in SHAP plots
    sample_size=100,             # Background samples for SHAP values
    test_size=0.2,               # Train/test split ratio
    cv_folds=5,                  # Cross-validation folds
    save_plot_path=None          # Path to save SHAP plots
)

# Results are stored in analysis.shap_results
```

**Plots generated:**
1. **Summary plot**: Shows global feature importance
2. **Beeswarm plot**: Feature importance with value distributions

### Local SHAP Interpretation

Explain individual predictions:

```python
# Plot decision plot for a single test sample
# (must call explain_by_shap() first)
analysis.local_shap_plot(
    sample_index=0,              # Index of test sample to explain
    figsize=(10, 8),
    save_plot_path=None
)
```

**Decision plot shows:**
- How features push prediction from base value to final prediction
- Feature contributions for specific samples
- Comparison across multiple samples

### Feature Importance by Wavenumber

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance as DataFrame
importance_df = pd.DataFrame({
    'wavenumber': analysis.wavenumbers,
    'importance': np.abs(shap_results['shap_values']).mean(0)
})

# Plot top important wavenumbers
top_features = importance_df.nlargest(20, 'importance')
plt.figure(figsize=(10, 6))
plt.bar(top_features['wavenumber'], top_features['importance'])
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Mean |SHAP value|')
plt.title('Top 20 Discriminative Wavenumbers')
plt.show()
```

## Complete Machine Learning Workflow

```python
from xpectrass import FTIRdataprocessing, FTIRdataanalysis
from xpectrass.data import load_jung_2018

# 1. Load and preprocess data
print("Loading and preprocessing data...")
df = load_jung_2018()
ftir = FTIRdataprocessing(df, label_column="type")
ftir.run()
processed_df = ftir.df_norm

# 2. Initialize analysis
analysis = FTIRdataanalysis(
    processed_df,
    dataset_name="Jung_2018",
    label_column="type",
    random_state=42
)

# 3. Run all models (automatically prepares data)
print("\nEvaluating all models...")
results = analysis.run_all_models(
    test_size=0.2,
    plot_comparison=True,
    accuracy_threshold=0.9,
    top_n_methods=20
)

# 4. Display top models
print("\n" + "="*60)
print("TOP 10 MODELS")
print("="*60)
top10 = results.nlargest(10, 'test_f1')
print(top10[['model_name', 'test_accuracy', 'test_f1', 'cv_mean', 'train_time']])

# 5. Tune top models
print("\nTuning top 2 models...")
tuned = analysis.model_parameter_tuning(number_of_models=2)

# 6. Explain best model with SHAP
print("\nExplaining with SHAP...")
analysis.explain_by_shap(
    model_name='XGBoost (100)',
    max_display=20,
    sample_size=100
)

# 7. Local explanations
print("\nGenerating local explanations...")
analysis.local_shap_plot(sample_index=0)

# 8. Save results
results.to_csv("model_comparison_results.csv", index=False)

print("\n✓ Machine learning workflow complete!")
```

## Cross-Dataset Validation

Test model generalization across different datasets:

```python
from xpectrass.data import load_jung_2018, load_frond_2021

# Train on one dataset
df_train = load_jung_2018()
# ... preprocess ...
analysis_train = FTIRdataanalysis(df_train, label_column="type")
analysis_train.ml_prepare_data(test_size=0.2)

# Test on another dataset
df_test = load_frond_2021()
# ... preprocess with same pipeline ...
analysis_test = FTIRdataanalysis(df_test, label_column="type")

# Evaluate cross-dataset performance
# (Note: Requires manual model training and prediction)
```

## Tips and Best Practices

### Data Preparation

1. **Always preprocess first**: Baseline correction, denoising, normalization
2. **Use stratified split**: Maintains class balance in train/test sets
3. **Set random_state**: For reproducible results
4. **Check class balance**: Imbalanced classes may require special handling

### Model Selection

1. **Start with all models**: Run `run_all_models()` to get baseline
2. **Consider speed vs accuracy**: Use efficiency analysis plot
3. **Check cross-validation scores**: Models with low CV std are more stable
4. **Don't overfit**: Monitor overfitting analysis plot

### Hyperparameter Tuning

1. **Tune top 3-5 models**: No need to tune everything
2. **Use cross-validation**: Prevents overfitting to test set
3. **Increase n_iter for better results**: 50-100 iterations recommended
4. **Be patient**: Tuning can take time for complex models

### Model Interpretation

1. **Use SHAP for final model**: Understand what features matter
2. **Check if important wavenumbers make sense**: Should align with chemistry
3. **Validate with domain knowledge**: Peak assignments should be reasonable
4. **Use local explanations**: Understand individual predictions

### Performance Metrics

Choose metrics appropriate for your problem:

- **Accuracy**: Overall correctness (good for balanced datasets)
- **F1 Score**: Harmonic mean of precision and recall (better for imbalanced data)
- **Precision**: Minimize false positives
- **Recall**: Minimize false negatives

## Common Issues and Solutions

### Issue: Poor Model Performance

**Solutions:**
1. Check preprocessing: Baseline correction, normalization
2. Remove outliers: Use data validation
3. Try feature selection: Remove noisy wavenumber regions
4. Increase training data: Combine multiple datasets
5. Check class balance: Use class weights or resampling

### Issue: Overfitting

**Solutions:**
1. Use cross-validation consistently
2. Reduce model complexity
3. Increase training data
4. Apply regularization
5. Use ensemble methods

### Issue: Slow Training

**Solutions:**
1. Use `n_jobs=-1` for parallelization
2. Reduce sample size for initial testing
3. Start with fast models (Logistic Regression, Linear SVM)
4. Reduce n_iter for hyperparameter tuning

### Issue: SHAP Takes Too Long

**Solutions:**
1. Reduce `sample_size` parameter (50-100 is usually sufficient)
2. Use TreeExplainer for tree-based models (faster)
3. Use KernelExplainer sample size parameter
4. Explain fewer samples

## Model Export and Deployment

### Save Trained Model

```python
import joblib

# Train your best model
best_model_name = tuned.iloc[0]['model']
# ... train model ...

# Save model
joblib.dump(model, 'best_ftir_classifier.pkl')

# Save preprocessing parameters
joblib.dump({
    'scaler': analysis.scaler,
    'class_names': analysis.class_names,
    'wavenumbers': analysis.wavenumbers
}, 'preprocessing_params.pkl')
```

### Load and Use Model

```python
import joblib
import pandas as pd

# Load model and parameters
model = joblib.load('best_ftir_classifier.pkl')
params = joblib.load('preprocessing_params.pkl')

# Preprocess new data
# ... apply same preprocessing pipeline ...

# Predict
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## Next Steps

- See [Analysis](analysis.md) for exploratory data analysis
- See [Preprocessing Pipeline](preprocessing_pipeline.md) for data preparation
- See [Examples](../examples.md) for complete workflows
- Check [API Reference](../api/index.md) for detailed function documentation
