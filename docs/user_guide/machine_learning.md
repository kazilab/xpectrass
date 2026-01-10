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
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=True            # Maintain class balance
)

print(f"Training samples: {len(analysis.y_train)}")
print(f"Test samples: {len(analysis.y_test)}")
print(f"Features: {analysis.X_train.shape[1]}")
print(f"Classes: {analysis.class_names}")
```

**Attributes created:**
- `X_train`, `X_test`: Feature matrices
- `y_train`, `y_test`: Labels
- `X_train_scaled`, `X_test_scaled`: Standardized features
- `class_names`: Unique class labels

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
result = analysis.run_a_model(
    model_name='RandomForest',
    X_train=analysis.X_train_scaled,
    y_train=analysis.y_train,
    X_test=analysis.X_test_scaled,
    y_test=analysis.y_test
)

print(f"Accuracy: {result['accuracy']:.3f}")
print(f"F1 Score: {result['f1_score']:.3f}")
print(f"Training time: {result['train_time']:.2f}s")
```

### Run All Models

Evaluate all available models with cross-validation:

```python
# Run all models
results = analysis.run_all_models(
    cv_folds=5,              # 5-fold cross-validation
    scoring='f1_weighted',   # Scoring metric
    n_jobs=-1                # Use all CPU cores
)

# View results sorted by F1 score
print(results.sort_values('f1_score', ascending=False))

# Save results
analysis.results_all = results  # Stored for later use
```

**Results DataFrame includes:**
- `model`: Model name
- `accuracy`: Test set accuracy
- `precision`: Weighted precision
- `recall`: Weighted recall
- `f1_score`: Weighted F1 score
- `cv_score_mean`: Mean cross-validation score
- `cv_score_std`: CV standard deviation
- `train_time`: Training time (seconds)
- `predict_time`: Prediction time (seconds)

### View Top Models

```python
# Get top 10 models by F1 score
top_models = results.nlargest(10, 'f1_score')
print(top_models[['model', 'accuracy', 'f1_score', 'cv_score_mean']])
```

## Model Comparison Visualization

### Plot Model Comparison

```python
# Compare all models
analysis.plot_model_comparison(
    results=analysis.results_all,
    metric='f1_score',       # Metric to display
    top_n=10,                # Show top 10 models
    figsize=(12, 8)
)
```

**Features:**
- Bar plot of model performance
- Error bars showing CV standard deviation
- Sorted by selected metric
- Color-coded by performance

### Plot Family Comparison

Compare model families (e.g., tree-based vs ensemble vs linear):

```python
# Compare model families
analysis.plot_family_comparison(
    results=analysis.results_all,
    figsize=(10, 6)
)
```

**Shows:**
- Average performance per model family
- Violin plots showing distribution
- Best model in each family

### Plot Efficiency Analysis

Analyze model trade-offs between accuracy and speed:

```python
# Efficiency analysis: accuracy vs speed
analysis.plot_efficiency_analysis(
    results=analysis.results_all,
    figsize=(10, 8)
)
```

**Features:**
- Scatter plot: Training time vs F1 score
- Bubble size represents model complexity
- Helps identify fast, accurate models

### Plot Overfitting Analysis

Identify overfitting by comparing train and test performance:

```python
# Overfitting analysis
analysis.plot_overfitting_analysis(
    results=analysis.results_all,
    figsize=(10, 8)
)
```

**Shows:**
- Train score vs test score
- Diagonal line = perfect generalization
- Points above line = overfitting
- Points below line = underfitting

## Hyperparameter Tuning

### Tune Top Models

Optimize hyperparameters for best-performing models:

```python
# Tune top 3 models
tuned_results = analysis.model_parameter_tuning(
    top_n=3,                 # Number of top models to tune
    cv_folds=5,              # Cross-validation folds
    n_iter=50,               # Iterations for random search
    scoring='f1_weighted',   # Optimization metric
    n_jobs=-1                # Use all cores
)

print("\nTuned model results:")
print(tuned_results[['model', 'best_f1', 'improvement', 'best_params']])
```

**Returns:**
- `model`: Model name
- `best_f1`: Best F1 score after tuning
- `improvement`: Improvement over default
- `best_params`: Optimal hyperparameters
- `cv_scores`: Cross-validation scores

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
    X=analysis.X_test_scaled,     # Data to explain
    sample_size=100,               # Samples for SHAP values
    plot=True                      # Create summary plot
)

# Access SHAP results
analysis.shap_results = shap_results
```

**Returns:**
- `shap_values`: SHAP values for each prediction
- `explainer`: SHAP explainer object
- `base_value`: Expected value (baseline)
- `feature_names`: Wavenumber labels

**Plots generated:**
1. **Summary plot**: Shows global feature importance
2. **Beeswarm plot**: Feature importance with value distributions

### Local SHAP Interpretation

Explain individual predictions:

```python
# Plot decision plots for individual samples
analysis.local_shap_plot(
    sample_indices=[0, 1, 2],    # Which samples to explain
    figsize=(12, 8)
)
```

**Decision plot shows:**
- How features push prediction from base value to final prediction
- Feature contributions for specific samples
- Comparison across multiple samples

### Feature Importance by Wavenumber

```python
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

# 3. Prepare data for ML
print("\nPreparing data...")
analysis.ml_prepare_data(test_size=0.2, random_state=42)
print(f"Training: {len(analysis.y_train)}, Test: {len(analysis.y_test)}")
print(f"Classes: {analysis.class_names}")

# 4. Run all models
print("\nEvaluating all models...")
results = analysis.run_all_models(cv_folds=5)

# 5. Display top models
print("\n" + "="*60)
print("TOP 10 MODELS")
print("="*60)
top10 = results.nlargest(10, 'f1_score')
print(top10[['model', 'accuracy', 'f1_score', 'cv_score_mean', 'train_time']])

# 6. Visualize model comparison
print("\nGenerating comparison plots...")
analysis.plot_model_comparison(results, top_n=15)
analysis.plot_family_comparison(results)
analysis.plot_efficiency_analysis(results)
analysis.plot_overfitting_analysis(results)

# 7. Tune top models
print("\nTuning top 3 models...")
tuned = analysis.model_parameter_tuning(top_n=3, n_iter=50)
print("\nTuned Results:")
print(tuned[['model', 'best_f1', 'improvement']])

# 8. Explain best model
best_model = tuned.iloc[0]['model']
print(f"\nExplaining {best_model} with SHAP...")
shap_results = analysis.explain_by_shap(
    model_name=best_model,
    X=analysis.X_test_scaled,
    sample_size=100,
    plot=True
)

# 9. Local explanations
print("\nGenerating local explanations...")
analysis.local_shap_plot(sample_indices=[0, 5, 10])

# 10. Save results
results.to_csv("model_comparison_results.csv", index=False)
tuned.to_csv("tuned_model_results.csv", index=False)

print("\n✓ Machine learning workflow complete!")
print(f"Best model: {best_model}")
print(f"Best F1 score: {tuned.iloc[0]['best_f1']:.4f}")
```

## Cross-Dataset Validation

Test model generalization across different datasets:

```python
from xpectrass.data import load_jung_2018, load_frond_2021

# Train on one dataset
df_train = load_jung_2018()
# ... preprocess ...
analysis_train = FTIRdataanalysis(df_train, label_column="type")
analysis_train.ml_prepare_data(test_size=0.0)  # Use all for training

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
