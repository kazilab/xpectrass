# Import libraries
from __future__ import annotations
from typing import Union, Tuple, List, Optional, Dict, Any
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Import shared spectral utilities
from .spectral_utils import (
    _infer_spectral_columns,
    _sort_spectral_columns
)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, jaccard_score
)

# Linear Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Tree-based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier
)

# Boosting libraries (silently check availability)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# SVM
from sklearn.svm import SVC, LinearSVC

# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def model_needs_scaling(model: Any) -> bool:
    """
    Determine if a model requires feature scaling.

    Tree-based models are scale-invariant, while distance/gradient-based models
    benefit from scaling.

    Parameters
    ----------
    model : sklearn estimator
        The machine learning model to check.

    Returns
    -------
    bool
        True if model needs scaling, False otherwise.
    """
    # Get model class name
    model_name = model.__class__.__name__.lower()

    # Tree-based models that DON'T need scaling
    tree_based = [
        'decisiontree', 'randomforest', 'extratrees',
        'xgb', 'lgb', 'catboost', 'gradientboosting',
        'adaboost', 'bagging'
    ]

    # Check if it's a tree-based model
    for tree_type in tree_based:
        if tree_type in model_name:
            return False

    # All other models (SVM, kNN, LogReg, Neural Nets, LDA, etc.) need scaling
    return True


def get_class_names(
    data_dict: Optional[Dict[str, Any]] = None,
    label_encoder: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
    n_classes: Optional[int] = None,
    warn_on_fallback: bool = True
) -> List[str]:
    """
    Extract class names from various sources, prioritizing explicit names.
    
    This helper function resolves class names from multiple possible sources,
    ensuring that plots display meaningful labels instead of numeric indices.
    
    Parameters
    ----------
    data_dict : dict, optional
        Data dictionary from prepare_data() containing 'class_names' and 'label_encoder'
    label_encoder : LabelEncoder, optional
        Fitted sklearn LabelEncoder with classes_ attribute
    class_names : list of str, optional
        Explicit class names (highest priority)
    n_classes : int, optional
        Number of classes (for generating fallback names like 'Class_0', 'Class_1')
    warn_on_fallback : bool, default True
        If True, print a warning when falling back to generic class names
    
    Returns
    -------
    list of str
        Class names in order corresponding to encoded indices [0, 1, 2, ...]
    
    Examples
    --------
    >>> # From data_dict (most common usage)
    >>> names = get_class_names(data_dict=data_dict)
    
    >>> # From explicit list
    >>> names = get_class_names(class_names=['PE', 'PP', 'PS', 'PVC'])
    
    >>> # From label encoder
    >>> names = get_class_names(label_encoder=le)
    """
    # Priority 1: Explicit class_names
    if class_names is not None:
        return list(class_names)
    
    # Priority 2: From data_dict
    if data_dict is not None:
        if 'class_names' in data_dict and data_dict['class_names'] is not None:
            # Handle numpy arrays
            names = data_dict['class_names']
            if hasattr(names, 'tolist'):
                return names.tolist()
            return list(names)
        if 'label_encoder' in data_dict and data_dict['label_encoder'] is not None:
            le = data_dict['label_encoder']
            if hasattr(le, 'classes_'):
                classes = le.classes_
                if hasattr(classes, 'tolist'):
                    return classes.tolist()
                return list(classes)
    
    # Priority 3: From label_encoder directly
    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        classes = label_encoder.classes_
        if hasattr(classes, 'tolist'):
            return classes.tolist()
        return list(classes)
    
    # Priority 4: Generate default names (with warning)
    if n_classes is not None:
        if warn_on_fallback:
            print(
                f"WARNING: No class names provided. Using generic names (Class_0, Class_1, ...). "
                f"To show actual class names (e.g., 'HDPE', 'PP'), pass data_dict=data_dict "
                f"or class_names=['HDPE', 'PP', ...] to this function."
            )
        return [f'Class_{i}' for i in range(n_classes)]
    
    # Cannot determine class names
    return None


def prepare_data(
    data: Union[pd.DataFrame, pl.DataFrame],
    label_column: str = 'type',
    exclude_columns: Optional[List[str]] = None,
    wn_min: Optional[float] = None,
    wn_max: Optional[float] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    handle_missing: str = "zero"
) -> Dict[str, Any]:
    """
    Prepare FTIR spectral data for machine learning classification.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input DataFrame containing spectral data (wavenumber columns) and labels.
        Spectral columns should be numeric (wavenumbers in cm⁻¹).
    label_column : str, default='type'
        Name of the column containing class labels for classification.
    exclude_columns : list of str, optional
        Additional columns to exclude from spectral data (e.g., 'sample', 'batch').
        The label_column is automatically excluded.
    wn_min : float, optional
        Minimum wavenumber (cm⁻¹) to include. If None, uses all available wavenumbers.
    wn_max : float, optional
        Maximum wavenumber (cm⁻¹) to include. If None, uses all available wavenumbers.
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0 to 1.0).
    random_state : int, default=42
        Random seed for reproducible train/test splits.
    scale_features : bool, default=True
        If True, apply StandardScaler to features. If False, return unscaled data.
        Note: For preprocessed FTIR data (already normalized), scaling may not be
        necessary for tree-based models but is recommended for distance/gradient-based models.
    handle_missing : str, default="zero"
        How to handle missing values (NaN):
        - "drop": Remove samples with any NaN values
        - "mean": Impute NaN with column mean
        - "zero": Replace NaN with 0
        - "raise": Raise an error if NaN values are found

    Returns
    -------
    dict
        Dictionary containing:
        - 'X_train': Training features (scaled if scale_features=True, else raw)
        - 'X_test': Test features (scaled if scale_features=True, else raw)
        - 'X_train_raw': Raw unscaled training features (always included)
        - 'X_test_raw': Raw unscaled test features (always included)
        - 'y_train': Training labels (encoded)
        - 'y_test': Test labels (encoded)
        - 'scaler': Fitted StandardScaler object (if scale_features=True, else None)
        - 'label_encoder': Fitted LabelEncoder object
        - 'class_names': Original class label names
        - 'wavenumbers': Selected wavenumber values

    Examples
    --------
    >>> # Default: returns scaled data (backward compatible)
    >>> data_dict = prepare_data(df, label_column='label', test_size=0.2)
    >>> X_train, y_train = data_dict['X_train'], data_dict['y_train']

    >>> # Return unscaled data for tree-based models
    >>> data_dict = prepare_data(df, scale_features=False)

    >>> # Always have access to raw data
    >>> X_train_raw = data_dict['X_train_raw']
    """
    # Convert polars to pandas if necessary
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()

    # Build list of columns to exclude
    if exclude_columns is None:
        exclude_columns = []
    exclude_columns = list(exclude_columns) + [label_column]

    # Infer and sort spectral columns using shared utilities
    spectral_cols, wn_values = _infer_spectral_columns(
        data,
        exclude_columns=exclude_columns,
        wn_min=wn_min,
        wn_max=wn_max
    )

    spectral_cols_sorted, wavenumbers, _ = _sort_spectral_columns(
        spectral_cols, wn_values
    )

    # Extract features and labels
    X = data[spectral_cols_sorted].values
    y = data[label_column].values

    # Handle missing values
    n_samples_before = X.shape[0]
    if np.any(np.isnan(X)):
        n_missing = np.sum(np.isnan(X))
        print(f"Warning: Found {n_missing} NaN values in data")

        if handle_missing == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
            y = y[mask]
            n_dropped = n_samples_before - X.shape[0]
            print(f"  Dropped {n_dropped} samples with NaN values ({n_samples_before} -> {X.shape[0]} samples)")
        elif handle_missing == "mean":
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            print(f"  Imputed NaN values with column means")
        elif handle_missing == "zero":
            X = np.nan_to_num(X, nan=0.0)
            print(f"  Replaced NaN values with 0")
        elif handle_missing == "raise":
            raise ValueError(f"Found {n_missing} NaN values in data. Use handle_missing='drop', 'mean', or 'zero' to handle them.")
        else:
            raise ValueError(f"Invalid handle_missing option: {handle_missing}. Must be 'drop', 'mean', 'zero', or 'raise'.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Optionally standardize features
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_train_raw': X_train,  # Always include raw data
        'X_test_raw': X_test,    # Always include raw data
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': le,
        'class_names': le.classes_,
        'wavenumbers': wavenumbers
    }


def get_all_models():
    """
    Define all models to test
    """
    models = {}
    
    # ========== LINEAR MODELS ==========
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    models['Ridge Classifier'] = RidgeClassifier(random_state=42)
    models['Linear Discriminant Analysis'] = LinearDiscriminantAnalysis()
    models['Quadratic Discriminant Analysis'] = QuadraticDiscriminantAnalysis()
    models['SGD Classifier'] = SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1)
    
    # ========== TREE-BASED MODELS ==========
    models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
    models['Random Forest (50)'] = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    models['Random Forest (100)'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    models['Random Forest (200)'] = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    models['Extra Trees (100)'] = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    models['Extra Trees (200)'] = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    # ========== BOOSTING MODELS ==========
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
    models['Gradient Boosting (50)'] = GradientBoostingClassifier(n_estimators=50, random_state=42)
    models['Gradient Boosting (100)'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    models['Gradient Boosting (200)'] = GradientBoostingClassifier(n_estimators=200, random_state=42)
    
    if XGBOOST_AVAILABLE:
        models['XGBoost (50)'] = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1, eval_metric='mlogloss')
        models['XGBoost (100)'] = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss')
        models['XGBoost (200)'] = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM (50)'] = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        models['LightGBM (100)'] = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        models['LightGBM (200)'] = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
    
    if CATBOOST_AVAILABLE:
        models['CatBoost (50)'] = cb.CatBoostClassifier(iterations=50, random_state=42, verbose=0)
        models['CatBoost (100)'] = cb.CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        models['CatBoost (200)'] = cb.CatBoostClassifier(iterations=200, random_state=42, verbose=0)
    
    # ========== SUPPORT VECTOR MACHINES ==========
    models['Linear SVM'] = LinearSVC(max_iter=2000, random_state=42)
    models['SVM (RBF)'] = SVC(kernel='rbf', random_state=42, probability=True)
    models['SVM (Poly)'] = SVC(kernel='poly', degree=3, random_state=42, probability=True)
    models['SVM (Sigmoid)'] = SVC(kernel='sigmoid', random_state=42, probability=True)
    
    # ========== NEAREST NEIGHBORS ==========
    models['KNN (k=3)'] = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    models['KNN (k=5)'] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    models['KNN (k=7)'] = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    models['KNN (k=10)'] = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    models['KNN (k=15)'] = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    
    # ========== NAIVE BAYES ==========
    models['Gaussian Naive Bayes'] = GaussianNB()
    
    # ========== NEURAL NETWORKS ==========
    models['MLP (64)'] = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    models['MLP (128)'] = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
    models['MLP (128,64)'] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    models['MLP (256,128)'] = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    models['MLP (256,128,64)'] = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
    
    # ========== BAGGING ==========
    models['Bagging (DT)'] = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100, random_state=42, n_jobs=-1
    )
    
    return models


def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv_folds=5, X_train_raw=None, X_test_raw=None):
    """
    Comprehensive model evaluation with proper cross-validation.

    Parameters
    ----------
    name : str
        Name of the model
    model : sklearn estimator
        The model to evaluate
    X_train : np.ndarray
        Training features (may be scaled)
    X_test : np.ndarray
        Test features (may be scaled)
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    cv_folds : int, default=5
        Number of cross-validation folds
    X_train_raw : np.ndarray, optional
        Raw (unscaled) training features for proper CV.
        If None, assumes X_train is already raw.
    X_test_raw : np.ndarray, optional
        Raw (unscaled) test features (for consistency).
        If None, assumes X_test is already raw.

    Returns
    -------
    dict
        Results dictionary with metrics and predictions
    """
    results = {'model_name': name}

    try:
        # Determine if we need to scale for this model
        needs_scaling = model_needs_scaling(model)

        # Use raw data if provided, otherwise use the provided data
        X_train_for_cv = X_train_raw if X_train_raw is not None else X_train

        # Training
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        results['train_time'] = train_time

        # Prediction
        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred
        results['pred_time'] = pred_time
        results['pred_time_per_sample'] = pred_time / len(X_test)

        # Metrics
        results['test_accuracy'] = accuracy_score(y_test, y_pred)
        results['test_precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        results['test_recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        results['test_f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Store predictions for reuse
        results['y_pred'] = y_pred

        # Training accuracy
        y_train_pred = model.predict(X_train)
        results['train_accuracy'] = accuracy_score(y_train, y_train_pred)

        # Get probability predictions if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None
        results['y_proba'] = y_proba

        # Cross-validation (if not too slow) - FIX LEAKAGE
        if train_time < 30:  # Only do CV if training is reasonably fast
            # Create a pipeline for CV to avoid data leakage
            # Filter out nested parameters (e.g., 'estimator__ccp_alpha') that can't be passed to __init__
            params = model.get_params()
            base_params = {k: v for k, v in params.items() if '__' not in k}

            if needs_scaling:
                # For models that need scaling, use pipeline with scaler + model
                cv_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model.__class__(**base_params))
                ])
            else:
                # For tree-based models, no scaling needed
                cv_pipeline = model.__class__(**base_params)

            # Perform CV on RAW data to avoid leakage
            cv_scores = cross_val_score(cv_pipeline, X_train_for_cv, y_train,
                                       cv=cv_folds, scoring='accuracy', n_jobs=-1)
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
        else:
            results['cv_mean'] = np.nan
            results['cv_std'] = np.nan

        # Overfitting check
        results['overfit_gap'] = results['train_accuracy'] - results['test_accuracy']

        results['status'] = 'success'
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)
        # Fill with NaN for failed models
        for key in ['train_time', 'pred_time', 'pred_time_per_sample', 'test_accuracy',
                   'test_precision', 'test_recall', 'test_f1', 'train_accuracy',
                   'cv_mean', 'cv_std', 'overfit_gap']:
            if key not in results:
                results[key] = np.nan
    
    return results, model


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    X_test: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    data_dict: Optional[Dict[str, Any]] = None,
    label_encoder: Optional[Any] = None,
    dataset_name: str = 'dataset',
    normalize: bool = False,
    save_plot: Optional[str] = None
) -> np.ndarray:
    """
    Plot confusion matrix for classification predictions.

    Flexible usage - provide either:
    1. Predictions directly via y_pred, OR
    2. Model + X_test to generate predictions

    Parameters
    ----------
    y_test : np.ndarray
        True labels for test set (encoded as integers).
    y_pred : np.ndarray, optional
        Predicted labels. If None, must provide model and X_test.
    model : sklearn model, optional
        Trained classification model with predict() method.
        Required if y_pred is None.
    X_test : np.ndarray, optional
        Test feature matrix of shape (n_samples, n_features).
        Required if y_pred is None.
    class_names : list of str, optional
        Names of classes corresponding to label indices.
        If None, attempts to extract from data_dict or label_encoder.
    data_dict : dict, optional
        Data dictionary from prepare_data() - will extract class_names automatically.
    label_encoder : LabelEncoder, optional
        Fitted LabelEncoder to extract class names from.
    dataset_name : str, default='dataset'
        Name of dataset for plot title.
    normalize : bool, default=False
        If True, normalize confusion matrix by row (true label).
    save_plot : str or Path, optional
        If provided, save the plot to this file path.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes).

    Examples
    --------
    >>> # Method 1: Pass data_dict (RECOMMENDED - automatically gets class names)
    >>> cm = plot_confusion_matrix(
    ...     y_test=data_dict['y_test'],
    ...     y_pred=y_pred,
    ...     data_dict=data_dict,  # Automatically extracts class_names
    ...     dataset_name='C4'
    ... )
    >>>
    >>> # Method 2: Pass class_names explicitly
    >>> cm = plot_confusion_matrix(
    ...     y_test=data_dict['y_test'],
    ...     y_pred=y_pred,
    ...     class_names=['PE', 'PP', 'PS', 'PVC'],
    ...     dataset_name='C4'
    ... )
    >>>
    >>> # Method 3: Let function generate predictions
    >>> cm = plot_confusion_matrix(
    ...     y_test=data_dict['y_test'],
    ...     model=model,
    ...     X_test=data_dict['X_test'],
    ...     data_dict=data_dict,
    ...     save_plot='confusion_matrix.png'
    ... )
    """
    # Generate predictions if not provided
    if y_pred is None:
        if model is None or X_test is None:
            raise ValueError(
                'If y_pred is not provided, both model and X_test must be provided!'
            )
        y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    n_classes = cm.shape[0]

    # Get class names from various sources
    resolved_names = get_class_names(
        data_dict=data_dict,
        label_encoder=label_encoder,
        class_names=class_names,
        n_classes=n_classes,
        warn_on_fallback=True  # Warn user if falling back to generic names
    )

    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    else:
        cm_display = cm

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot heatmap
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=resolved_names if resolved_names else range(n_classes),
        yticklabels=resolved_names if resolved_names else range(n_classes),
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    # Labels and title
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    title = f'Confusion Matrix - {dataset_name}'
    if normalize:
        title += ' (Normalized)'
    plt.title(title, fontsize=14, fontweight='bold')

    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    plt.text(
        0.5, -0.1,
        f'Accuracy: {accuracy:.3f}',
        ha='center',
        transform=plt.gca().transAxes,
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()

    # Save if requested
    if save_plot:
        dir = save_plot + '_confusion_matrix.pdf'
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_plot}")

    plt.show()

    return cm


def calculate_multiclass_metrics(
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    X_test: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    data_dict: Optional[Dict[str, Any]] = None,
    label_encoder: Optional[Any] = None,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive multiclass classification metrics with per-class breakdown.

    Flexible usage - provide either:
    1. Predictions directly via y_pred (and optionally y_proba), OR
    2. Model + X_test to generate predictions

    Parameters
    ----------
    y_test : np.ndarray
        True labels for test set (encoded as integers).
    y_pred : np.ndarray, optional
        Predicted labels. If None, must provide model and X_test.
    model : sklearn model, optional
        Trained classification model with predict() method.
        Required if y_pred is None.
    X_test : np.ndarray, optional
        Test feature matrix of shape (n_samples, n_features).
        Required if y_pred is None.
    class_names : list of str, optional
        Names of classes corresponding to label indices.
        If None, attempts to extract from data_dict or label_encoder.
    data_dict : dict, optional
        Data dictionary from prepare_data() - will extract class_names automatically.
    label_encoder : LabelEncoder, optional
        Fitted LabelEncoder to extract class names from.
    y_proba : np.ndarray, optional
        Predicted probabilities of shape (n_samples, n_classes).
        If None and model has predict_proba(), will compute it.
        Required for ROC-AUC calculation.

    Returns
    -------
    dict
        Dictionary containing:
        - 'overall_metrics': Dict with accuracy, precision, recall, F1, MCC, Kappa, Jaccard, ROC-AUC
        - 'per_class_metrics': DataFrame with metrics for each class (uses class names, not numbers)
        - 'confusion_matrix': Confusion matrix array

    Examples
    --------
    >>> # Method 1: Pass data_dict (RECOMMENDED - automatically gets class names)
    >>> metrics = calculate_multiclass_metrics(
    ...     y_test=data_dict['y_test'],
    ...     y_pred=y_pred,
    ...     data_dict=data_dict  # Automatically extracts class_names
    ... )
    >>>
    >>> # Method 2: Pass class_names explicitly
    >>> metrics = calculate_multiclass_metrics(
    ...     y_test=data_dict['y_test'],
    ...     y_pred=y_pred,
    ...     class_names=['PE', 'PP', 'PS', 'PVC']
    ... )
    >>>
    >>> print("Overall Metrics:")
    >>> print(metrics['overall_metrics'])
    >>> print("\\nPer-Class Metrics:")
    >>> print(metrics['per_class_metrics'])
    """
    # Generate predictions if not provided
    if y_pred is None:
        if model is None or X_test is None:
            raise ValueError(
                'If y_pred is not provided, both model and X_test must be provided!'
            )
        y_pred = model.predict(X_test)

    # Get predicted probabilities if not provided
    if y_proba is None and model is not None and hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
        except:
            y_proba = None

    # Initialize results
    overall_metrics = {}

    # === OVERALL METRICS ===

    # 1. Accuracy
    overall_metrics['accuracy'] = accuracy_score(y_test, y_pred)

    # 2. Precision (macro average)
    overall_metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    overall_metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)

    # 3. Recall/Sensitivity (macro average)
    overall_metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    overall_metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # 4. F1 Score (macro average)
    overall_metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    overall_metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # 5. Matthews Correlation Coefficient (MCC)
    overall_metrics['mcc'] = matthews_corrcoef(y_test, y_pred)

    # 6. Cohen's Kappa
    overall_metrics['cohens_kappa'] = cohen_kappa_score(y_test, y_pred)

    # 7. Jaccard Score (macro average)
    overall_metrics['jaccard_macro'] = jaccard_score(y_test, y_pred, average='macro', zero_division=0)
    overall_metrics['jaccard_weighted'] = jaccard_score(y_test, y_pred, average='weighted', zero_division=0)

    # 8. ROC-AUC (if probabilities available)
    if y_proba is not None:
        try:
            overall_metrics['roc_auc_ovr_macro'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            overall_metrics['roc_auc_ovr_weighted'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            overall_metrics['roc_auc_ovr_macro'] = np.nan
            overall_metrics['roc_auc_ovr_weighted'] = np.nan
    else:
        overall_metrics['roc_auc_ovr_macro'] = np.nan
        overall_metrics['roc_auc_ovr_weighted'] = np.nan

    # === PER-CLASS METRICS ===

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    n_classes = cm.shape[0]

    # Get class names from various sources
    resolved_names = get_class_names(
        data_dict=data_dict,
        label_encoder=label_encoder,
        class_names=class_names,
        n_classes=n_classes,
        warn_on_fallback=True  # Warn user if falling back to generic names
    )

    # Calculate per-class metrics
    per_class_data = []

    for i in range(n_classes):
        class_metrics = {'class': resolved_names[i]}

        # True Positives, False Positives, False Negatives, True Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        # Support (number of true instances)
        support = cm[i, :].sum()
        class_metrics['support'] = support

        # Precision (for this class)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_metrics['precision'] = precision

        # Recall/Sensitivity (for this class)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_metrics['recall'] = recall
        class_metrics['sensitivity'] = recall  # Same as recall

        # Specificity (for this class)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        class_metrics['specificity'] = specificity

        # F1 Score (for this class)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics['f1'] = f1

        # Negative Predictive Value (NPV)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        class_metrics['npv'] = npv

        # Jaccard (for this class)
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        class_metrics['jaccard'] = jaccard

        per_class_data.append(class_metrics)

    # Convert to DataFrame for easy viewing
    per_class_df = pd.DataFrame(per_class_data)

    # Sort by F1 score (descending) to see which classes perform best
    per_class_df = per_class_df.sort_values('f1', ascending=False).reset_index(drop=True)

    return {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_df,
        'confusion_matrix': cm
    }


def print_multiclass_metrics(
    metrics_dict: Dict[str, Any],
    dataset_name: str = 'dataset'
) -> None:
    """
    Print comprehensive multiclass metrics in a formatted way.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary returned by calculate_multiclass_metrics().
    dataset_name : str, default='dataset'
        Name of dataset for display.

    Examples
    --------
    >>> metrics = calculate_multiclass_metrics(model, X_test, y_test, class_names)
    >>> print_multiclass_metrics(metrics, dataset_name='C4')
    """
    print(f"\n{'='*80}")
    print(f"MULTICLASS CLASSIFICATION METRICS - {dataset_name}")
    print(f"{'='*80}\n")

    # Overall metrics
    print("OVERALL METRICS:")
    print("-" * 80)
    overall = metrics_dict['overall_metrics']

    print(f"  Accuracy:              {overall['accuracy']:.4f}")
    print(f"  Precision (macro):     {overall['precision_macro']:.4f}")
    print(f"  Precision (weighted):  {overall['precision_weighted']:.4f}")
    print(f"  Recall (macro):        {overall['recall_macro']:.4f}")
    print(f"  Recall (weighted):     {overall['recall_weighted']:.4f}")
    print(f"  F1 (macro):            {overall['f1_macro']:.4f}")
    print(f"  F1 (weighted):         {overall['f1_weighted']:.4f}")
    print(f"  MCC:                   {overall['mcc']:.4f}")
    print(f"  Cohen's Kappa:         {overall['cohens_kappa']:.4f}")
    print(f"  Jaccard (macro):       {overall['jaccard_macro']:.4f}")
    print(f"  Jaccard (weighted):    {overall['jaccard_weighted']:.4f}")

    if not np.isnan(overall['roc_auc_ovr_macro']):
        print(f"  ROC-AUC (OvR, macro):  {overall['roc_auc_ovr_macro']:.4f}")
        print(f"  ROC-AUC (OvR, wtd):    {overall['roc_auc_ovr_weighted']:.4f}")
    else:
        print(f"  ROC-AUC:               N/A (no probabilities)")

    # Per-class metrics
    print(f"\n{'='*80}")
    print("PER-CLASS METRICS (sorted by F1 score):")
    print("-" * 80)

    per_class = metrics_dict['per_class_metrics']
    print(per_class.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Identify best and worst performing classes
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY:")
    print("-" * 80)

    best_class = per_class.iloc[0]
    worst_class = per_class.iloc[-1]

    print(f"  Best performing class:  {best_class['class']} (F1: {best_class['f1']:.4f})")
    print(f"  Worst performing class: {worst_class['class']} (F1: {worst_class['f1']:.4f})")

    # Classes with low metrics
    low_precision = per_class[per_class['precision'] < 0.80]
    low_recall = per_class[per_class['recall'] < 0.80]

    if len(low_precision) > 0:
        print(f"\n  Classes with precision < 80%:")
        for _, row in low_precision.iterrows():
            print(f"    - {row['class']}: {row['precision']:.4f}")

    if len(low_recall) > 0:
        print(f"\n  Classes with recall < 80%:")
        for _, row in low_recall.iterrows():
            print(f"    - {row['class']}: {row['recall']:.4f}")

    print(f"\n{'='*80}\n")


def evaluate_all_models(
    models: Dict[str, Any],
    data_dict: Dict[str, Any],
    dataset_name: str = "dataset"
) -> pd.DataFrame:
    """
    Evaluate all classification models on prepared data.

    Parameters
    ----------
    models : dict
        Dictionary of model_name -> model_instance pairs to evaluate.
        Use get_all_models() to get a comprehensive set of models.
    data_dict : dict
        Prepared data dictionary from prepare_data() containing:
        'X_train', 'X_test', 'y_train', 'y_test', etc.
    dataset_name : str, default='dataset'
        Name of dataset for display purposes in progress messages.

    Returns
    -------
    pd.DataFrame
        Results dataframe sorted by test accuracy (descending), containing:
        - model_name: Name of the model
        - test_accuracy: Test set accuracy
        - test_precision, test_recall, test_f1: Classification metrics
        - train_accuracy: Training set accuracy
        - cv_mean, cv_std: Cross-validation results
        - train_time, pred_time: Timing information
        - overfit_gap: Difference between train and test accuracy
        - status: 'success' or 'failed'

    Examples
    --------
    >>> models = get_all_models()
    >>> data_dict = prepare_data(df)
    >>> results = evaluate_all_models(models, data_dict, dataset_name="c8")
    >>> print(results.head())
    """
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    # Get raw data if available (for proper CV without leakage)
    X_train_raw = data_dict.get('X_train_raw', None)
    X_test_raw = data_dict.get('X_test_raw', None)

    results_list = []

    print(f"\n{'='*80}")
    print(f"EVALUATING ALL MODELS - {dataset_name}")
    print(f"{'='*80}\n")

    total_models = len(models)

    for idx, (name, model) in enumerate(models.items(), 1):
        print(f"[{idx}/{total_models}] Evaluating {name}...")

        results, trained_model = evaluate_model(name, model, X_train, X_test, y_train, y_test,
                                                X_train_raw=X_train_raw, X_test_raw=X_test_raw)
        results_list.append(results)

        if results['status'] == 'success':
            print(f"  ✓ Accuracy: {results['test_accuracy']:.4f} | Train time: {results['train_time']:.2f}s")
        else:
            print(f"  ✗ Failed")

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"{'='*80}\n")

    # Create dataframe
    results_df = pd.DataFrame(results_list)

    # Sort by test accuracy
    results_df = results_df.sort_values('test_accuracy', ascending=False).reset_index(drop=True)

    return results_df
    

def plot_model_comparison(
    results_df: pd.DataFrame,
    dataset_name: str = "dataset",
    top_n: int = 20,
    save_plot: Optional[str] = None
) -> None:
    """
    Plot comprehensive comparison of model performance metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_all_models().
    dataset_name : str, default='dataset'
        Name of dataset for plot titles.
    top_n : int, default=20
        Number of top-performing models to display.
    save_plot : str, optional
        If provided, saves the plot to this filepath instead of displaying.
        Supports formats: .png, .pdf, .svg, .jpg

    Examples
    --------
    >>> results = evaluate_all_models(models, data_dict)
    >>> plot_model_comparison(results, dataset_name="c8", top_n=20)
    >>> plot_model_comparison(results, save_plot="model_comparison.png")
    """
    # Filter successful models and top N
    df = results_df[results_df['status'] == 'success'].head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['test_accuracy'], alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['model_name'], fontsize=9)
    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Test Accuracy Comparison - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # 2. Training time vs Accuracy
    ax = axes[0, 1]
    scatter = ax.scatter(df['train_time'], df['test_accuracy'],
                        s=100, alpha=0.6, c=df['test_accuracy'],
                        cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Annotate top 5
    for idx in range(min(5, len(df))):
        ax.annotate(df.iloc[idx]['model_name'],
                   (df.iloc[idx]['train_time'], df.iloc[idx]['test_accuracy']),
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs Training Time - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Accuracy')
    
    # 3. F1-Score comparison
    ax = axes[1, 0]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['test_f1'], alpha=0.7, color='coral')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['model_name'], fontsize=9)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title(f'F1-Score Comparison - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # 4. Metrics heatmap (top 15)
    ax = axes[1, 1]
    top_15 = df.head(15)
    metrics_data = top_15[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']].values
    
    im = ax.imshow(metrics_data, cmap='YlGn', aspect='auto', vmin=0.8, vmax=1.0)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(top_15)))
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'], fontsize=10)
    ax.set_yticklabels(top_15['model_name'], fontsize=9)
    ax.set_title(f'Top 15 Models - All Metrics - {dataset_name}', fontsize=14, fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(top_15)):
        for j in range(4):
            ax.text(j, i, f'{metrics_data[i, j]:.3f}',
                   ha="center", va="center", color="black", fontsize=7)

    plt.colorbar(im, ax=ax, label='Score')

    plt.tight_layout()

    if save_plot:
        dir = save_plot + '_model_comparison.pdf'
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_plot}")
        plt.close()
    else:
        plt.show()
    

def categorize_model(model_name):
    """
    Categorize models into families
    """
    model_name_lower = model_name.lower()
    
    if any(x in model_name_lower for x in ['logistic', 'ridge', 'discriminant', 'sgd']):
        return 'Linear'
    elif any(x in model_name_lower for x in ['tree', 'forest', 'extra']):
        return 'Tree-Based'
    elif any(x in model_name_lower for x in ['boost', 'xgb', 'lgbm', 'lightgbm', 'catboost', 'adaboost']):
        return 'Boosting'
    elif 'svm' in model_name_lower or 'svc' in model_name_lower:
        return 'SVM'
    elif 'knn' in model_name_lower:
        return 'KNN'
    elif 'naive' in model_name_lower or 'bayes' in model_name_lower:
        return 'Naive Bayes'
    elif 'mlp' in model_name_lower:
        return 'Neural Network'
    elif 'bagging' in model_name_lower:
        return 'Bagging'
    else:
        return 'Other'

def plot_family_comparison(
    results_df: pd.DataFrame,
    dataset_name: str = "dataset",
    save_plot: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare performance of different model families.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_all_models().
    dataset_name : str, default='dataset'
        Name of dataset for plot titles.
    save_plot : str, optional
        If provided, saves the plot to this filepath instead of displaying.

    Returns
    -------
    pd.DataFrame
        Family-level statistics including mean, max, min accuracy and timing.

    Examples
    --------
    >>> family_stats = plot_family_comparison(results, dataset_name="c8")
    >>> print(family_stats)
    """
    # Add family column
    df = results_df[results_df['status'] == 'success'].copy()
    df['family'] = df['model_name'].apply(categorize_model)
    
    # Group by family
    family_stats = df.groupby('family').agg({
        'test_accuracy': ['mean', 'max', 'min', 'std'],
        'train_time': ['mean', 'median'],
        'pred_time': ['mean', 'median'],
        'model_name': 'count'
    }).round(4)
    
    family_stats.columns = ['_'.join(col).strip() for col in family_stats.columns.values]
    family_stats = family_stats.reset_index()
    family_stats = family_stats.sort_values('test_accuracy_max', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"MODEL FAMILY COMPARISON - {dataset_name}")
    print(f"{'='*80}\n")
    print(family_stats.to_string(index=False))
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Best accuracy per family
    ax = axes[0]
    ax.barh(family_stats['family'], family_stats['test_accuracy_max'], alpha=0.7, color='steelblue')
    ax.set_xlabel('Best Test Accuracy', fontsize=12)
    ax.set_title(f'Best Model per Family - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # 2. Average accuracy per family
    ax = axes[1]
    ax.barh(family_stats['family'], family_stats['test_accuracy_mean'],
           xerr=family_stats['test_accuracy_std'], alpha=0.7, color='coral')
    ax.set_xlabel('Mean Test Accuracy', fontsize=12)
    ax.set_title(f'Average Performance per Family - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # 3. Training time per family
    ax = axes[2]
    ax.barh(family_stats['family'], family_stats['train_time_mean'], alpha=0.7, color='lightgreen')
    ax.set_xlabel('Mean Training Time (seconds)', fontsize=12)
    ax.set_title(f'Training Time per Family - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_plot:
        dir = save_plot + '_family_comparison.pdf'
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_plot}")
        plt.close()
    else:
        plt.show()

    return family_stats
    

def plot_efficiency_analysis(
    results_df: pd.DataFrame,
    dataset_name: str = "dataset",
    accuracy_threshold: float = 0.95,
    save_plot: Optional[str] = None
) -> None:
    """
    Analyze computational efficiency of models (accuracy vs time trade-off).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_all_models().
    dataset_name : str, default='dataset'
        Name of dataset for plot titles.
    accuracy_threshold : float, default=0.95
        Accuracy threshold for highlighting high-performing models.
    save_plot : str, optional
        If provided, saves the plot to this filepath instead of displaying.

    Examples
    --------
    >>> plot_efficiency_analysis(results, dataset_name="c8", accuracy_threshold=0.95)
    """
    df = results_df[results_df['status'] == 'success'].copy()
    
    # Filter high-performing models
    high_perf = df[df['test_accuracy'] >= accuracy_threshold]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Accuracy vs Total Time (Train + Predict)
    ax = axes[0]
    df['total_time'] = df['train_time'] + df['pred_time']
    
    scatter = ax.scatter(df['total_time'], df['test_accuracy'],
                        s=100, alpha=0.6, c=df['test_accuracy'],
                        cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Highlight high performers
    if len(high_perf) > 0:
        high_perf_total_time = high_perf['train_time'] + high_perf['pred_time']
        ax.scatter(high_perf_total_time, high_perf['test_accuracy'],
                  s=200, alpha=0.8, edgecolors='red', linewidth=2, facecolors='none')
    
    # Annotate best performers
    for idx in range(min(5, len(df))):
        ax.annotate(df.iloc[idx]['model_name'],
                   (df.iloc[idx]['total_time'], df.iloc[idx]['test_accuracy']),
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Total Time: Train + Predict (seconds)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Efficiency Analysis - {dataset_name}', fontsize=14, fontweight='bold')
    ax.axhline(y=accuracy_threshold, color='red', linestyle='--', alpha=0.5,
              label=f'{accuracy_threshold} threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Accuracy')
    
    # 2. Top models by efficiency score
    ax = axes[1]
    # Efficiency score: accuracy / log(time + 1)
    df['efficiency_score'] = df['test_accuracy'] / np.log10(df['total_time'] + 1)
    df_sorted_eff = df.sort_values('efficiency_score', ascending=False).head(20)
    
    y_pos = np.arange(len(df_sorted_eff))
    ax.barh(y_pos, df_sorted_eff['efficiency_score'], alpha=0.7, color='purple')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted_eff['model_name'], fontsize=9)
    ax.set_xlabel('Efficiency Score (Accuracy / log(Time))', fontsize=12)
    ax.set_title(f'Top 20 Most Efficient Models - {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_plot:
        dir = save_plot + '_efficiency_analysis.pdf'
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_plot}")
        plt.close()
    else:
        plt.show()

    # Print efficiency leaders
    print(f"\n{'='*80}")
    print(f"TOP 10 MOST EFFICIENT MODELS - {dataset_name}")
    print(f"{'='*80}\n")
    eff_cols = ['model_name', 'test_accuracy', 'total_time', 'efficiency_score']
    print(df_sorted_eff[eff_cols].head(10).to_string(index=False))


def plot_overfitting_analysis(
    results_df: pd.DataFrame,
    dataset_name: str = "dataset",
    top_n: int = 20,
    save_plot: Optional[str] = None
) -> None:
    """
    Analyze overfitting tendency across models.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_all_models().
    dataset_name : str, default='dataset'
        Name of dataset for plot titles.
    top_n : int, default=20
        Number of top-performing models to analyze.
    save_plot : str, optional
        If provided, saves the plot to this filepath instead of displaying.

    Examples
    --------
    >>> plot_overfitting_analysis(results, dataset_name="c8", top_n=20)
    """
    df = results_df[results_df['status'] == 'success'].head(top_n).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Train vs Test Accuracy
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['train_accuracy'], width, label='Train', alpha=0.7)
    ax.bar(x + width/2, df['test_accuracy'], width, label='Test', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Train vs Test Accuracy - {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=90, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Overfitting gap
    ax = axes[1]
    df_sorted = df.sort_values('overfit_gap')
    y_pos = np.arange(len(df_sorted))
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red'
             for gap in df_sorted['overfit_gap']]
    
    ax.barh(y_pos, df_sorted['overfit_gap'], alpha=0.7, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['model_name'], fontsize=9)
    ax.set_xlabel('Overfitting Gap (Train - Test Accuracy)', fontsize=12)
    ax.set_title(f'Overfitting Analysis - {dataset_name}', fontsize=14, fontweight='bold')
    ax.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, label='5% gap')
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='10% gap')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()

    if save_plot:
        dir = save_plot + '_overfitting_analysis.pdf'
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_plot}")
        plt.close()
    else:
        plt.show()

    # Print models with least overfitting
    print(f"\n{'='*80}")
    print(f"MODELS WITH LEAST OVERFITTING - {dataset_name}")
    print(f"{'='*80}\n")
    overfit_cols = ['model_name', 'train_accuracy', 'test_accuracy', 'overfit_gap']
    print(df.sort_values('overfit_gap')[overfit_cols].head(10).to_string(index=False))


def tune_top_models(data_dict, results_df, top_n=3):
    """
    Perform hyperparameter tuning for top N models
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    # Get top models
    top_models = results_df.head(top_n)
    
    tuning_results = []
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING FOR TOP {top_n} MODELS")
    print(f"{'='*80}\n")
    
    for idx, row in top_models.iterrows():
        model_name = row['model_name']
        print(f"\nTuning {model_name}...")
        
        # Define parameter grids based on model type
        if 'Random Forest' in model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif 'XGBoost' in model_name and XGBOOST_AVAILABLE:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
            
        elif 'LightGBM' in model_name and LIGHTGBM_AVAILABLE:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 70]
            }
            model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            
        elif 'CatBoost' in model_name and CATBOOST_AVAILABLE:
            param_grid = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 7]
            }
            model = cb.CatBoostClassifier(random_state=42, verbose=0)
            
        elif 'Gradient Boosting' in model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        else:
            print(f"  Skipping {model_name} - no tuning grid defined")
            continue
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        tuned_accuracy = accuracy_score(y_test, y_pred)
        tuned_f1 = f1_score(y_test, y_pred, average='weighted')
        
        result = {
            'model_name': model_name,
            'original_accuracy': row['test_accuracy'],
            'tuned_accuracy': tuned_accuracy,
            'improvement': tuned_accuracy - row['test_accuracy'],
            'tuned_f1': tuned_f1,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        tuning_results.append(result)
        
        print(f"  Original Accuracy: {row['test_accuracy']:.4f}")
        print(f"  Tuned Accuracy: {tuned_accuracy:.4f}")
        print(f"  Improvement: {result['improvement']:+.4f}")
        print(f"  Best Parameters: {grid_search.best_params_}")
    
    return pd.DataFrame(tuning_results)


def compare_datasets_performance(
    results_c4_df: pd.DataFrame,
    results_c8_df: pd.DataFrame,
    save_plot: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare model performance across two datasets (e.g., c4 vs c8).

    Parameters
    ----------
    results_c4_df : pd.DataFrame
        Results dataframe for first dataset.
    results_c8_df : pd.DataFrame
        Results dataframe for second dataset.
    save_plot : str, optional
        If provided, saves the plot to this filepath instead of displaying.

    Returns
    -------
    pd.DataFrame
        Merged comparison dataframe with accuracy and time differences.

    Examples
    --------
    >>> comparison = compare_datasets_performance(results_c4, results_c8)
    """
    # Merge results
    c4_subset = results_c4_df[['model_name', 'test_accuracy', 'train_time']].copy()
    c8_subset = results_c8_df[['model_name', 'test_accuracy', 'train_time']].copy()
    
    c4_subset.columns = ['model_name', 'c4_accuracy', 'c4_train_time']
    c8_subset.columns = ['model_name', 'c8_accuracy', 'c8_train_time']
    
    merged = pd.merge(c4_subset, c8_subset, on='model_name', how='inner')
    merged['accuracy_diff'] = merged['c4_accuracy'] - merged['c8_accuracy']
    merged['time_ratio'] = merged['c4_train_time'] / merged['c8_train_time']
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Accuracy comparison
    ax = axes[0]
    top_20 = merged.sort_values('c4_accuracy', ascending=False).head(20)
    
    x = np.arange(len(top_20))
    width = 0.35
    
    ax.bar(x - width/2, top_20['c4_accuracy'], width, label='c4 (High Res)', alpha=0.7)
    ax.bar(x + width/2, top_20['c8_accuracy'], width, label='c8 (Low Res)', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison: c4 vs c8 (Top 20 Models)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_20['model_name'], rotation=90, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Accuracy difference
    ax = axes[1]
    diff_sorted = merged.sort_values('accuracy_diff', ascending=False).head(20)
    y_pos = np.arange(len(diff_sorted))
    colors = ['green' if d > 0 else 'red' for d in diff_sorted['accuracy_diff']]
    
    ax.barh(y_pos, diff_sorted['accuracy_diff'], alpha=0.7, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diff_sorted['model_name'], fontsize=9)
    ax.set_xlabel('Accuracy Difference (c4 - c8)', fontsize=12)
    ax.set_title('Which Dataset Performs Better?', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()

    if save_plot:
        dir = save_plot + '_compare_dataset_performance.pdf'
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_plot}")
        plt.close()
    else:
        plt.show()
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total models compared: {len(merged)}")
    print(f"\nAccuracy Statistics:")
    print(f"  c4 mean accuracy: {merged['c4_accuracy'].mean():.4f}")
    print(f"  c8 mean accuracy: {merged['c8_accuracy'].mean():.4f}")
    print(f"  c4 best accuracy: {merged['c4_accuracy'].max():.4f}")
    print(f"  c8 best accuracy: {merged['c8_accuracy'].max():.4f}")
    
    better_c4 = (merged['accuracy_diff'] > 0).sum()
    better_c8 = (merged['accuracy_diff'] < 0).sum()
    print(f"\nModels performing better on c4: {better_c4} ({better_c4/len(merged)*100:.1f}%)")
    print(f"Models performing better on c8: {better_c8} ({better_c8/len(merged)*100:.1f}%)")
    
    print(f"\nTraining Time Statistics:")
    print(f"  c4 mean train time: {merged['c4_train_time'].mean():.2f}s")
    print(f"  c8 mean train time: {merged['c8_train_time'].mean():.2f}s")
    print(f"  c4/c8 time ratio: {merged['time_ratio'].mean():.2f}x")
    
    # Top models for each dataset
    print(f"\n{'='*80}")
    print("TOP 5 MODELS PER DATASET")
    print(f"{'='*80}\n")
    
    print("c4 (High Resolution):")
    top5_c4 = merged.nlargest(5, 'c4_accuracy')[['model_name', 'c4_accuracy']]
    print(top5_c4.to_string(index=False))
    
    print("\nc8 (Low Resolution):")
    top5_c8 = merged.nlargest(5, 'c8_accuracy')[['model_name', 'c8_accuracy']]
    print(top5_c8.to_string(index=False))
    
    return merged


def generate_recommendations(results_c8_df, results_c4_df, comparison_df):
    """
    Generate final recommendations
    """
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS FOR FTIR PLASTIC CLASSIFICATION")
    print("="*80 + "\n")
    
    # Best overall model
    best_c8 = results_c8_df.iloc[0]
    best_c4 = results_c4_df.iloc[0]
    
    print("1. BEST PERFORMING MODELS:")
    print("-" * 80)
    print(f"   c8 (Low Resolution): {best_c8['model_name']}")
    print(f"      - Accuracy: {best_c8['test_accuracy']:.4f}")
    print(f"      - F1-Score: {best_c8['test_f1']:.4f}")
    print(f"      - Training time: {best_c8['train_time']:.2f}s")
    print(f"\n   c4 (High Resolution): {best_c4['model_name']}")
    print(f"      - Accuracy: {best_c4['test_accuracy']:.4f}")
    print(f"      - F1-Score: {best_c4['test_f1']:.4f}")
    print(f"      - Training time: {best_c4['train_time']:.2f}s")
    
    # Most efficient models
    print("\n2. MOST EFFICIENT MODELS (Accuracy + Speed):")
    print("-" * 80)
    
    # Filter high accuracy (>0.95) and fast (<10s)
    efficient_c8 = results_c8_df[
        (results_c8_df['test_accuracy'] > 0.95) &
        (results_c8_df['train_time'] < 10)
    ].head(3)
    
    if len(efficient_c8) > 0:
        print("   c8 dataset:")
        for idx, row in efficient_c8.iterrows():
            print(f"      - {row['model_name']}: {row['test_accuracy']:.4f} accuracy in {row['train_time']:.2f}s")
    
    # Best model families
    print("\n3. RECOMMENDED MODEL FAMILIES:")
    print("-" * 80)
    print("   Based on overall performance:")
    
    # Add family categorization
    results_c8_df['family'] = results_c8_df['model_name'].apply(categorize_model)
    family_best = results_c8_df.groupby('family')['test_accuracy'].max().sort_values(ascending=False)
    
    for family, acc in family_best.head(5).items():
        print(f"      - {family}: up to {acc:.4f} accuracy")
    
    # Deployment recommendations
    print("\n4. DEPLOYMENT RECOMMENDATIONS:")
    print("-" * 80)
    print("   For production deployment, consider:")
    print("\n   High Accuracy Priority:")
    print(f"      → {results_c8_df.iloc[0]['model_name']}")
    print(f"        (Accuracy: {results_c8_df.iloc[0]['test_accuracy']:.4f})")
    
    # Find fast and accurate
    fast_accurate = results_c8_df[
        (results_c8_df['test_accuracy'] > 0.97) &
        (results_c8_df['train_time'] < 5)
    ]
    
    if len(fast_accurate) > 0:
        print("\n   Speed + Accuracy Balance:")
        print(f"      → {fast_accurate.iloc[0]['model_name']}")
        print(f"        (Accuracy: {fast_accurate.iloc[0]['test_accuracy']:.4f}, ")
        print(f"         Train time: {fast_accurate.iloc[0]['train_time']:.2f}s)")
    
    # Dataset recommendation
    print("\n5. DATASET SELECTION:")
    print("-" * 80)
    
    avg_diff = comparison_df['accuracy_diff'].mean()
    avg_time_ratio = comparison_df['time_ratio'].mean()
    
    if abs(avg_diff) < 0.01:
        print("   → Recommendation: Use c8 (Low Resolution)")
        print(f"     - Similar accuracy to c4 (diff: {avg_diff:+.4f})")
        print(f"     - Faster training ({avg_time_ratio:.2f}x speedup)")
        print(f"     - Smaller file size and memory footprint")
    elif avg_diff > 0.01:
        print("   → Recommendation: Use c4 (High Resolution)")
        print(f"     - Better accuracy ({avg_diff:+.4f} improvement)")
        print(f"     - Worth the extra training time")
    else:
        print("   → Recommendation: Use c8 (Low Resolution)")
        print(f"     - Actually performs better than c4 ({avg_diff:+.4f})")
        print(f"     - Also faster to train")
    
    # Key insights
    print("\n6. KEY INSIGHTS:")
    print("-" * 80)
    print(f"   - {len(results_c8_df[results_c8_df['test_accuracy'] > 0.95])} models achieve >95% accuracy")
    print(f"   - {len(results_c8_df[results_c8_df['test_accuracy'] > 0.99])} models achieve >99% accuracy")
    print(f"   - Best accuracy achieved: {results_c8_df['test_accuracy'].max():.4f}")
    print(f"   - Fastest high-performer (<95% acc): {results_c8_df[results_c8_df['test_accuracy']>0.95]['train_time'].min():.2f}s")
    
    overfit_ok = results_c8_df[results_c8_df['overfit_gap'] < 0.05]
    print(f"   - {len(overfit_ok)} models show minimal overfitting (<5% gap)")
    
    print("\n" + "="*80)



def explain_model_shap(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    data_dict: Optional[Dict[str, Any]] = None,
    wavenumbers: Optional[np.ndarray] = None,
    max_display: int = 20,
    sample_size: int = 100,
    dataset_name: str = 'dataset',
    X_train_raw: Optional[np.ndarray] = None,
    X_test_raw: Optional[np.ndarray] = None,
    save_plot: Optional[str] = None,
    plot_layout: str = 'auto',
    yaxis_fontsize: int = 6,
    title_fontsize: int = 8,
    xlabel_fontsize: int = 7
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a trained classification model.

    SHAP (SHapley Additive exPlanations) provides insight into how each feature contributes
    to the prediction of each class. For multiclass problems, SHAP generates separate
    importance values for each class.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model (e.g., RandomForest, XGBoost, etc.).
    X_train : np.ndarray
        Training features (may be scaled) - used as background data for SHAP.
    X_test : np.ndarray
        Test features (may be scaled) to explain.
    y_test : np.ndarray, optional
        True test labels. Only used for display/validation.
    class_names : list of str, optional
        Names of each class (e.g., ['PE', 'PP', 'PS', 'PVC']).
    data_dict : dict, optional
        Data dictionary from prepare_data() - will extract class_names and wavenumbers.
    wavenumbers : np.ndarray, optional
        Wavenumber values for feature names. If None, uses feature indices.
    max_display : int, default=20
        Maximum number of top features to display in plots
    sample_size : int, default=100
        Number of background samples to use for SHAP computation.
    dataset_name : str, default='dataset'
        Name of dataset for plot titles
    X_train_raw : np.ndarray, optional
        Raw (unscaled) training features for interpretability.
    X_test_raw : np.ndarray, optional
        Raw (unscaled) test features for interpretability.
    save_plot : str, optional
        Directory path where plots will be saved as PDFs.
    plot_layout : str, default='auto'
        Layout strategy for multiclass plots:
        - 'auto': Automatically choose based on number of classes
        - 'grid': Use 2-column grid layout (good for 4-12 classes)
        - 'vertical': Stack vertically (good for 2-4 classes)
        - 'individual': Create separate figure per class (good for 8+ classes)
    yaxis_fontsize : int, default=9
        Font size for y-axis tick labels (feature names/wavenumbers)
    title_fontsize : int, default=12
        Font size for subplot titles
    xlabel_fontsize : int, default=10
        Font size for x-axis labels

    Returns
    -------
    dict
        Dictionary containing SHAP results.
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Please install it with: pip install shap"
        )

    # Extract from data_dict if provided
    if data_dict is not None:
        if class_names is None and 'class_names' in data_dict:
            class_names = list(data_dict['class_names'])
        if wavenumbers is None and 'wavenumbers' in data_dict:
            wavenumbers = data_dict['wavenumbers']
        if X_train_raw is None and 'X_train_raw' in data_dict:
            X_train_raw = data_dict['X_train_raw']
        if X_test_raw is None and 'X_test_raw' in data_dict:
            X_test_raw = data_dict['X_test_raw']
        if y_test is None and 'y_test' in data_dict:
            y_test = data_dict['y_test']

    # Validate class_names
    if class_names is None:
        raise ValueError(
            "class_names must be provided either directly or via data_dict."
        )

    # Determine which data to use for display
    X_train_display = X_train_raw if X_train_raw is not None else X_train
    X_test_display = X_test_raw if X_test_raw is not None else X_test

    print("\n" + "="*80)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    if X_train_raw is not None:
        print("\nData mode: Using SCALED data for SHAP computation, RAW data for visualization")
    else:
        print("\nData mode: Using provided data for both SHAP computation and visualization")

    # Prepare feature names
    if wavenumbers is not None:
        feature_names = [f"{wn:.1f}" for wn in wavenumbers]
    else:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    # Sample background data
    if X_train.shape[0] > sample_size:
        print(f"\nSampling {sample_size} background samples from {X_train.shape[0]} training samples...")
        bg_idx = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_background = X_train[bg_idx]
    else:
        X_background = X_train
        print(f"\nUsing all {X_train.shape[0]} training samples as background...")

    # Determine which explainer to use
    model_name = model.__class__.__name__
    tree_models = [
        'RandomForestClassifier', 'ExtraTreesClassifier',
        'GradientBoostingClassifier', 'XGBClassifier',
        'LGBMClassifier', 'CatBoostClassifier',
        'DecisionTreeClassifier'
    ]

    print(f"\nModel: {model_name}")

    # Create SHAP explainer
    start_time = time.time()

    if model_name in tree_models:
        print("Using TreeExplainer (fast for tree-based models)...")
        explainer = shap.TreeExplainer(model, X_background)
    else:
        print("Using KernelExplainer (model-agnostic, may be slower)...")
        def model_predict(X):
            return model.predict_proba(X)
        explainer = shap.KernelExplainer(model_predict, X_background)

    # Calculate SHAP values
    print(f"Computing SHAP values for {X_test.shape[0]} test samples...")
    shap_values = explainer.shap_values(X_test)

    compute_time = time.time() - start_time
    print(f"SHAP computation completed in {compute_time:.2f}s")

    # Handle different SHAP value formats
    n_classes = len(class_names)

    if isinstance(shap_values, list):
        print(f"\nMulticlass problem detected: {len(shap_values)} classes")
        if len(shap_values) != n_classes:
            raise ValueError(
                f"Mismatch: SHAP returned {len(shap_values)} arrays but "
                f"class_names has {n_classes} classes"
            )
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            print(f"\nMulticlass problem detected: {shap_values.shape[2]} classes")
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        elif shap_values.ndim == 2:
            shap_values = [shap_values]
            n_classes = 1
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    else:
        raise ValueError(f"Unexpected SHAP values type: {type(shap_values)}")

    # ==========================================================================
    # FIXED: Improved layout calculation for multiclass SHAP plots
    # ==========================================================================
    
    # Determine optimal layout based on number of classes
    if plot_layout == 'auto':
        if n_classes <= 3:
            plot_layout = 'vertical'
        elif n_classes <= 12:
            plot_layout = 'grid'
        else:
            plot_layout = 'individual'
    
    print(f"\nUsing '{plot_layout}' layout for {n_classes} classes")

    # ==========================================================================
    # Plot 1: Summary plot (dot plot) for each class
    # ==========================================================================
    print("\nGenerating SHAP summary plots for each class...")
    
    if plot_layout == 'vertical':
        # Stack vertically for few classes
        n_cols = 1
        n_rows = n_classes
        fig_width = 12
        fig_height = 4.5 * n_classes
        
    elif plot_layout == 'grid':
        # Use 2-column grid for moderate number of classes
        n_cols = 2
        n_rows = int(np.ceil(n_classes / n_cols))
        fig_width = 14
        fig_height = 4 * n_rows
        
    elif plot_layout == 'individual':
        # Individual plots for many classes
        n_cols = 1
        n_rows = 1
        fig_width = 12
        fig_height = 6
    
    if plot_layout != 'individual':
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = np.atleast_1d(axes).flatten()
        
        for i, class_name in enumerate(class_names):
            if i < len(axes):
                plt.sca(axes[i])
                shap.summary_plot(
                    shap_values[i],
                    X_test_display,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                    plot_type='dot'
                )
                axes[i].set_title(f'SHAP Feature Importance - Class: {class_name}',
                                 fontsize=title_fontsize, fontweight='bold')
                axes[i].set_xlabel('SHAP Value (impact on model output)', fontsize=xlabel_fontsize)
                axes[i].tick_params(axis='y', labelsize=yaxis_fontsize)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            save_dir = Path(save_plot)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f'shap_summary_{dataset_name}.pdf'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"   → Saved: {file_path}")
            plt.close()
        else:
            plt.show()
    else:
        # Individual plots for each class
        for i, class_name in enumerate(class_names):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.sca(ax)
            shap.summary_plot(
                shap_values[i],
                X_test_display,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                plot_type='dot'
            )
            ax.set_title(f'SHAP Feature Importance - Class: {class_name}',
                        fontsize=title_fontsize + 2, fontweight='bold')
            ax.set_xlabel('SHAP Value (impact on model output)', fontsize=xlabel_fontsize + 1)
            ax.tick_params(axis='y', labelsize=yaxis_fontsize)
            plt.tight_layout()
            
            if save_plot:
                save_dir = Path(save_plot)
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f'shap_summary_{dataset_name}_{class_name}.pdf'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"   → Saved: {file_path}")
                plt.close()
            else:
                plt.show()

    # ==========================================================================
    # Plot 2: Bar plot showing mean absolute SHAP values
    # ==========================================================================
    print("\nGenerating SHAP bar plots (mean absolute impact)...")
    
    if plot_layout == 'vertical':
        n_cols = 1
        n_rows = n_classes
        fig_width = 10
        fig_height = 4 * n_classes
        
    elif plot_layout == 'grid':
        n_cols = 2
        n_rows = int(np.ceil(n_classes / n_cols))
        fig_width = 14
        fig_height = 3.5 * n_rows
        
    elif plot_layout == 'individual':
        n_cols = 1
        n_rows = 1
        fig_width = 10
        fig_height = 5
    
    if plot_layout != 'individual':
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = np.atleast_1d(axes).flatten()
        
        for i, class_name in enumerate(class_names):
            if i < len(axes):
                plt.sca(axes[i])
                shap.summary_plot(
                    shap_values[i],
                    X_test_display,
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False,
                    plot_type='bar'
                )
                axes[i].set_title(f'Top Features - {class_name}', fontsize=title_fontsize - 1, fontweight='bold')
                axes[i].set_xlabel('Mean |SHAP value|', fontsize=xlabel_fontsize)
                axes[i].tick_params(axis='y', labelsize=yaxis_fontsize)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            save_dir = Path(save_plot)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f'shap_bar_{dataset_name}.pdf'
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"   → Saved: {file_path}")
            plt.close()
        else:
            plt.show()
    else:
        # Individual plots for each class
        for i, class_name in enumerate(class_names):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.sca(ax)
            shap.summary_plot(
                shap_values[i],
                X_test_display,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                plot_type='bar'
            )
            ax.set_title(f'Top Features - {class_name}', fontsize=title_fontsize + 1, fontweight='bold')
            ax.set_xlabel('Mean |SHAP value|', fontsize=xlabel_fontsize + 1)
            ax.tick_params(axis='y', labelsize=yaxis_fontsize)
            plt.tight_layout()
            
            if save_plot:
                save_dir = Path(save_plot)
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f'shap_bar_{dataset_name}_{class_name}.pdf'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"   → Saved: {file_path}")
                plt.close()
            else:
                plt.show()

    # ==========================================================================
    # BONUS: Combined bar plot showing global feature importance
    # ==========================================================================
    print("\nGenerating combined global feature importance plot...")
    
    # Calculate mean absolute SHAP across all classes
    global_importance = np.zeros(len(feature_names))
    for sv in shap_values:
        global_importance += np.abs(sv).mean(axis=0)
    global_importance /= n_classes
    
    # Get top features
    top_idx = np.argsort(global_importance)[::-1][:max_display]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_idx))
    ax.barh(y_pos, global_importance[top_idx][::-1], color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=yaxis_fontsize)
    ax.set_xlabel('Mean |SHAP value| (averaged across classes)', fontsize=xlabel_fontsize + 1)
    ax.set_title(f'Global Feature Importance - {dataset_name}', fontsize=title_fontsize + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_plot:
        save_dir = Path(save_plot)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f'shap_global_{dataset_name}.pdf'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"   → Saved: {file_path}")
        plt.close()
    else:
        plt.show()

    # Print top features for each class
    print("\n" + "="*80)
    print("TOP 10 FEATURES PER CLASS (by mean absolute SHAP value)")
    print("="*80)

    for i, class_name in enumerate(class_names):
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]

        print(f"\n{class_name}:")
        print("-" * 40)
        for rank, idx in enumerate(top_indices, 1):
            feature_name = feature_names[idx]
            importance = mean_abs_shap[idx]
            print(f"  {rank:2d}. {feature_name:>10s}  |  Impact: {importance:.4f}")

    print("\n" + "="*80)

    # Return results
    results = {
        'explainer': explainer,
        'shap_values': shap_values,
        'expected_values': explainer.expected_value,
        'feature_names': feature_names,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names
    }

    return results


def plot_shap_decision(
    shap_results: Dict[str, Any],
    sample_idx: int,
    figsize: Tuple[int, int] = None,
    save_plot: Optional[str] = None,
    yaxis_fontsize: int = 6,
    title_fontsize: int = 8
) -> None:
    """
    Plot SHAP decision plot for a specific test sample.

    Fixed to handle multiclass with proper sizing.

    Parameters
    ----------
    shap_results : dict
        Results dictionary from explain_model_shap()
    sample_idx : int
        Index of the test sample to explain
    figsize : tuple, optional
        Figure size (width, height)
    save_plot : str, optional
        Directory path to save the plot
    yaxis_fontsize : int, default=8
        Font size for y-axis tick labels (feature names)
    title_fontsize : int, default=11
        Font size for subplot titles
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Please install it with: pip install shap"
        )

    shap_values = shap_results['shap_values']
    X_test = shap_results['X_test']
    y_test = shap_results['y_test']
    class_names = shap_results['class_names']
    feature_names = shap_results['feature_names']

    true_label = y_test[sample_idx]
    true_class_name = class_names[true_label]
    n_classes = len(class_names)

    print(f"\nSHAP Decision Plot for Sample {sample_idx}")
    print(f"True Label: {true_class_name}")
    print("="*60)

    # Auto-size based on number of classes
    if figsize is None:
        if n_classes <= 4:
            figsize = (4 * n_classes, 8)
        else:
            # Use grid layout for many classes
            n_cols = min(4, n_classes)
            n_rows = int(np.ceil(n_classes / n_cols))
            figsize = (4 * n_cols, 6 * n_rows)
    
    # Determine layout
    if n_classes <= 4:
        n_cols = n_classes
        n_rows = 1
    else:
        n_cols = min(4, n_classes)
        n_rows = int(np.ceil(n_classes / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, class_name in enumerate(class_names):
        if i < len(axes):
            plt.sca(axes[i])
            
            shap.decision_plot(
                shap_results['expected_values'][i],
                shap_values[i][sample_idx],
                feature_names=feature_names,
                show=False,
                highlight=0
            )

            if i == true_label:
                axes[i].set_title(f'{class_name}\n(TRUE LABEL)',
                                fontsize=title_fontsize, fontweight='bold', color='green')
            else:
                axes[i].set_title(class_name, fontsize=title_fontsize)

            axes[i].tick_params(axis='y', labelsize=yaxis_fontsize)
    
    # Hide unused subplots
    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_plot:
        save_dir = Path(save_plot)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f'shap_decision_sample_{sample_idx}.pdf'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"   → Saved: {file_path}")
        plt.close()
    else:
        plt.show()



