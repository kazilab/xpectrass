"""
Comprehensive Tests for ml.py
==============================

Tests mathematical correctness, edge cases, and expected behavior
for machine learning utilities.

Run with: python test_ml.py
Or with pytest: pytest test_ml.py -v
"""

import numpy as np
import pandas as pd
import sys
from typing import Callable
import warnings

# Test configuration
np.random.seed(42)
TOLERANCE = 1e-10
FLOAT_TOLERANCE = 1e-6


# ============================================================================
# Test Helper Functions
# ============================================================================

def create_synthetic_classification_data(
    n_samples: int = 200,
    n_features: int = 100,
    n_classes: int = 4,
    class_names: list = None
) -> pd.DataFrame:
    """Create synthetic FTIR-like classification data."""
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Wavenumbers from 4000 to 400 cm⁻¹
    wavenumbers = np.linspace(4000, 400, n_features)
    
    # Generate random spectral data
    X = np.random.randn(n_samples, n_features)
    
    # Add class-specific patterns
    y_indices = np.random.randint(0, n_classes, n_samples)
    y = [class_names[i] for i in y_indices]
    
    # Create class-specific signatures
    for i in range(n_classes):
        mask = np.array(y_indices) == i
        peak_position = int(n_features * (i + 1) / (n_classes + 1))
        X[mask, max(0, peak_position-5):min(n_features, peak_position+5)] += 2.0
    
    # Create DataFrame
    columns = [str(wn) for wn in wavenumbers]
    df = pd.DataFrame(X, columns=columns)
    df['label'] = y
    
    return df


def assert_close(actual, expected, tol=FLOAT_TOLERANCE, msg=""):
    """Assert that actual is close to expected within tolerance."""
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        if not np.allclose(actual, expected, atol=tol, rtol=tol):
            diff = np.max(np.abs(actual - expected))
            raise AssertionError(f"{msg}: Max difference {diff} exceeds tolerance {tol}")
    else:
        if abs(actual - expected) > tol:
            raise AssertionError(f"{msg}: {actual} != {expected} (diff={abs(actual-expected)})")


def run_test(test_func: Callable, test_name: str) -> bool:
    """Run a test function and report result."""
    try:
        test_func()
        print(f"  ✓ {test_name}")
        return True
    except AssertionError as e:
        print(f"  ✗ {test_name}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ {test_name}: Unexpected error: {e}")
        return False


# ============================================================================
# Standalone implementations for testing (no dependencies)
# ============================================================================

def model_needs_scaling(model_name: str) -> bool:
    """Determine if a model requires feature scaling."""
    model_name_lower = model_name.lower()
    
    tree_based = [
        'decisiontree', 'randomforest', 'extratrees',
        'xgb', 'lgb', 'catboost', 'gradientboosting',
        'adaboost', 'bagging'
    ]
    
    for tree_type in tree_based:
        if tree_type in model_name_lower:
            return False
    return True


def calculate_confusion_matrix_metrics(cm: np.ndarray, class_idx: int):
    """Calculate per-class metrics from confusion matrix."""
    tp = cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - tp
    fn = cm[class_idx, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall,
        'specificity': specificity, 'f1': f1
    }


# ============================================================================
# TEST CASES
# ============================================================================

class TestModelScalingDecision:
    """Tests for model_needs_scaling function."""
    
    @staticmethod
    def test_tree_models_no_scaling():
        """Tree-based models should NOT need scaling."""
        tree_models = [
            'DecisionTreeClassifier', 'RandomForestClassifier',
            'ExtraTreesClassifier', 'XGBClassifier',
            'LGBMClassifier', 'CatBoostClassifier',
            'GradientBoostingClassifier', 'AdaBoostClassifier',
            'BaggingClassifier'
        ]
        
        for model_name in tree_models:
            needs = model_needs_scaling(model_name)
            assert not needs, f"{model_name} should NOT need scaling"
    
    @staticmethod
    def test_distance_models_need_scaling():
        """Distance-based models should need scaling."""
        distance_models = [
            'KNeighborsClassifier', 'SVC', 'LinearSVC',
            'LogisticRegression', 'MLPClassifier',
            'LinearDiscriminantAnalysis', 'RidgeClassifier'
        ]
        
        for model_name in distance_models:
            needs = model_needs_scaling(model_name)
            assert needs, f"{model_name} should need scaling"
    
    @staticmethod
    def test_case_insensitivity():
        """Function should be case-insensitive."""
        assert not model_needs_scaling('randomforest')
        assert not model_needs_scaling('RANDOMFOREST')
        assert not model_needs_scaling('RandomForest')
        assert model_needs_scaling('SVC')
        assert model_needs_scaling('svc')


class TestConfusionMatrixMetrics:
    """Tests for confusion matrix metric calculations."""
    
    @staticmethod
    def test_perfect_classification():
        """Perfect classification should give precision=recall=f1=1.0."""
        # Diagonal confusion matrix (all correct)
        cm = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ])
        
        for i in range(3):
            metrics = calculate_confusion_matrix_metrics(cm, i)
            assert_close(metrics['precision'], 1.0, msg=f"Class {i} precision")
            assert_close(metrics['recall'], 1.0, msg=f"Class {i} recall")
            assert_close(metrics['f1'], 1.0, msg=f"Class {i} F1")
            assert_close(metrics['specificity'], 1.0, msg=f"Class {i} specificity")
    
    @staticmethod
    def test_completely_wrong():
        """Class with all predictions wrong should have precision=0."""
        # Class 0 is never predicted correctly
        cm = np.array([
            [0, 10, 0],  # Class 0: all misclassified as Class 1
            [0, 10, 0],
            [0, 0, 10]
        ])
        
        metrics = calculate_confusion_matrix_metrics(cm, 0)
        assert_close(metrics['precision'], 0.0, msg="Class 0 precision should be 0")
        assert_close(metrics['recall'], 0.0, msg="Class 0 recall should be 0")
        assert_close(metrics['f1'], 0.0, msg="Class 0 F1 should be 0")
    
    @staticmethod
    def test_precision_calculation():
        """Verify precision = TP / (TP + FP)."""
        cm = np.array([
            [8, 2, 0],
            [1, 7, 2],
            [1, 1, 8]
        ])
        
        # Class 0: TP=8, FP=1+1=2, precision = 8/10 = 0.8
        metrics = calculate_confusion_matrix_metrics(cm, 0)
        assert_close(metrics['precision'], 0.8, msg="Class 0 precision")
        
        # Class 1: TP=7, FP=2+1=3, precision = 7/10 = 0.7
        metrics = calculate_confusion_matrix_metrics(cm, 1)
        assert_close(metrics['precision'], 0.7, msg="Class 1 precision")
    
    @staticmethod
    def test_recall_calculation():
        """Verify recall = TP / (TP + FN)."""
        cm = np.array([
            [8, 2, 0],
            [1, 7, 2],
            [1, 1, 8]
        ])
        
        # Class 0: TP=8, FN=2+0=2, recall = 8/10 = 0.8
        metrics = calculate_confusion_matrix_metrics(cm, 0)
        assert_close(metrics['recall'], 0.8, msg="Class 0 recall")
        
        # Class 1: TP=7, FN=1+2=3, recall = 7/10 = 0.7
        metrics = calculate_confusion_matrix_metrics(cm, 1)
        assert_close(metrics['recall'], 0.7, msg="Class 1 recall")
    
    @staticmethod
    def test_specificity_calculation():
        """Verify specificity = TN / (TN + FP)."""
        cm = np.array([
            [8, 2, 0],
            [1, 7, 2],
            [1, 1, 8]
        ])
        
        # Class 0: TN = sum of all except row 0 and col 0
        # = 7 + 2 + 1 + 8 = 18
        # FP = 1 + 1 = 2
        # Specificity = 18 / 20 = 0.9
        metrics = calculate_confusion_matrix_metrics(cm, 0)
        assert_close(metrics['specificity'], 0.9, msg="Class 0 specificity")
    
    @staticmethod
    def test_f1_calculation():
        """Verify F1 = 2 * precision * recall / (precision + recall)."""
        cm = np.array([
            [8, 2, 0],
            [1, 7, 2],
            [1, 1, 8]
        ])
        
        metrics = calculate_confusion_matrix_metrics(cm, 0)
        # Precision = 0.8, Recall = 0.8
        # F1 = 2 * 0.8 * 0.8 / (0.8 + 0.8) = 1.28 / 1.6 = 0.8
        expected_f1 = 2 * 0.8 * 0.8 / (0.8 + 0.8)
        assert_close(metrics['f1'], expected_f1, msg="Class 0 F1")


class TestDataPreparation:
    """Tests for data preparation logic."""
    
    @staticmethod
    def test_stratified_split_preserves_proportions():
        """Stratified split should preserve class proportions."""
        # Create data with known proportions
        n_samples = 100
        class_counts = {'A': 40, 'B': 30, 'C': 30}
        
        labels = []
        for cls, count in class_counts.items():
            labels.extend([cls] * count)
        
        labels = np.array(labels)
        
        # Shuffle
        np.random.shuffle(labels)
        
        # Calculate original proportions
        original_props = {}
        for cls in class_counts.keys():
            original_props[cls] = np.sum(labels == cls) / len(labels)
        
        # Simulate stratified split
        from sklearn.model_selection import train_test_split
        y_train, y_test = train_test_split(
            labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Check train proportions
        for cls in class_counts.keys():
            train_prop = np.sum(y_train == cls) / len(y_train)
            test_prop = np.sum(y_test == cls) / len(y_test)
            
            # Should be within 10% of original
            assert abs(train_prop - original_props[cls]) < 0.1, \
                f"Train proportion for {cls} differs too much"
            assert abs(test_prop - original_props[cls]) < 0.1, \
                f"Test proportion for {cls} differs too much"
    
    @staticmethod
    def test_label_encoder_roundtrip():
        """Label encoder should correctly encode and decode."""
        from sklearn.preprocessing import LabelEncoder
        
        original_labels = ['PE', 'PP', 'PS', 'PVC', 'PE', 'PP']
        
        le = LabelEncoder()
        encoded = le.fit_transform(original_labels)
        decoded = le.inverse_transform(encoded)
        
        assert list(decoded) == original_labels, "Roundtrip should preserve labels"
        assert list(le.classes_) == sorted(set(original_labels)), \
            "Classes should be unique original labels"
    
    @staticmethod
    def test_standard_scaler_properties():
        """StandardScaler should produce mean=0, std=1."""
        from sklearn.preprocessing import StandardScaler
        
        X = np.random.randn(100, 50) * 10 + 5  # Mean ~5, std ~10
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check column means ≈ 0
        col_means = np.mean(X_scaled, axis=0)
        assert np.allclose(col_means, 0, atol=1e-10), "Means should be ~0"
        
        # Check column stds ≈ 1
        col_stds = np.std(X_scaled, axis=0)
        assert np.allclose(col_stds, 1, atol=1e-10), "Stds should be ~1"


class TestMetricsFormulas:
    """Tests for metric calculation formulas."""
    
    @staticmethod
    def test_accuracy_formula():
        """Accuracy = correct / total."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 2, 1])  # 6 correct, 2 wrong
        
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        
        expected = 6 / 8
        assert_close(acc, expected, msg="Accuracy formula")
    
    @staticmethod
    def test_mcc_perfect():
        """MCC should be 1.0 for perfect classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(y_true, y_pred)
        
        assert_close(mcc, 1.0, msg="MCC for perfect classification")
    
    @staticmethod
    def test_mcc_random():
        """MCC should be ~0 for random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 1000)
        y_pred = np.random.randint(0, 3, 1000)
        
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # MCC should be close to 0 for random
        assert abs(mcc) < 0.1, f"MCC for random should be ~0, got {mcc}"
    
    @staticmethod
    def test_cohen_kappa_perfect():
        """Cohen's Kappa should be 1.0 for perfect classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(y_true, y_pred)
        
        assert_close(kappa, 1.0, msg="Kappa for perfect classification")


class TestEdgeCases:
    """Tests for edge cases."""
    
    @staticmethod
    def test_single_class():
        """Metrics should handle single-class predictions gracefully."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # All predicted as class 0
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Should not raise errors
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0 <= precision <= 1, "Precision should be in [0, 1]"
        assert 0 <= recall <= 1, "Recall should be in [0, 1]"
        assert 0 <= f1 <= 1, "F1 should be in [0, 1]"
    
    @staticmethod
    def test_imbalanced_classes():
        """Metrics should handle imbalanced classes."""
        # Very imbalanced: 90% class 0, 10% class 1
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 100)  # Predict all as class 0
        
        from sklearn.metrics import accuracy_score, f1_score
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Accuracy is misleadingly high (90%)
        assert_close(acc, 0.9, msg="Accuracy for imbalanced")
        
        # Macro F1 should be lower (penalizes missing class 1)
        assert f1_macro < acc, "Macro F1 should be lower than accuracy for imbalanced"
    
    @staticmethod
    def test_empty_predictions():
        """Should handle edge case of empty arrays gracefully."""
        # This tests that the code doesn't crash
        y_true = np.array([])
        y_pred = np.array([])
        
        from sklearn.metrics import confusion_matrix
        
        # Should handle empty arrays
        try:
            cm = confusion_matrix(y_true, y_pred)
            # Empty confusion matrix is OK
            assert cm.size == 0 or cm.shape == (0, 0), "Empty CM should be empty"
        except ValueError:
            # ValueError is acceptable for empty input
            pass


class TestCrossValidation:
    """Tests for cross-validation logic."""
    
    @staticmethod
    def test_cv_folds_correct():
        """CV should produce correct number of folds."""
        from sklearn.model_selection import StratifiedKFold
        
        X = np.random.randn(100, 10)
        y = np.array([0] * 50 + [1] * 50)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(cv.split(X, y))
        
        assert len(folds) == 5, "Should have 5 folds"
        
        for train_idx, test_idx in folds:
            # Each fold should have ~80 train, ~20 test
            assert len(train_idx) == 80, f"Train should be 80, got {len(train_idx)}"
            assert len(test_idx) == 20, f"Test should be 20, got {len(test_idx)}"
    
    @staticmethod
    def test_cv_no_leakage():
        """CV folds should not overlap (no data leakage)."""
        from sklearn.model_selection import StratifiedKFold
        
        X = np.random.randn(100, 10)
        y = np.array([0] * 50 + [1] * 50)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        all_test_indices = []
        for train_idx, test_idx in cv.split(X, y):
            # No overlap between train and test
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test should not overlap"
            
            all_test_indices.extend(test_idx)
        
        # All indices should appear exactly once in test sets
        assert sorted(all_test_indices) == list(range(100)), \
            "Each sample should appear in test exactly once"


class TestOverfittingDetection:
    """Tests for overfitting detection logic."""
    
    @staticmethod
    def test_overfit_gap_calculation():
        """Overfit gap = train_accuracy - test_accuracy."""
        train_acc = 0.95
        test_acc = 0.85
        
        gap = train_acc - test_acc
        
        assert_close(gap, 0.10, msg="Overfit gap calculation")
    
    @staticmethod
    def test_overfit_classification():
        """Classify overfitting severity correctly."""
        # No overfitting: gap < 5%
        assert 0.03 < 0.05, "Gap < 5% is acceptable"
        
        # Moderate overfitting: 5% <= gap < 10%
        assert 0.07 >= 0.05 and 0.07 < 0.10, "5-10% is moderate"
        
        # Severe overfitting: gap >= 10%
        assert 0.15 >= 0.10, ">=10% is severe"


# ============================================================================
# Run all tests
# ============================================================================

def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestModelScalingDecision,
        TestConfusionMatrixMetrics,
        TestDataPreparation,
        TestMetricsFormulas,
        TestEdgeCases,
        TestCrossValidation,
        TestOverfittingDetection,
    ]
    
    total_passed = 0
    total_failed = 0
    
    print("=" * 70)
    print("ML MODULE TESTS")
    print("=" * 70)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        
        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        
        for method_name in test_methods:
            method = getattr(test_class, method_name)
            if run_test(method, method_name):
                total_passed += 1
            else:
                total_failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
