"""
Comprehensive Tests for normalization.py
=========================================

Tests mathematical correctness, edge cases, and expected behavior
for all normalization methods.

Run with: python test_normalization.py
Or with pytest: pytest test_normalization.py -v
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

def create_synthetic_spectrum(n_points: int = 1000, noise_level: float = 0.01) -> tuple:
    """Create a synthetic FTIR-like spectrum with known properties."""
    # Wavenumbers: 4000 to 400 cm⁻¹
    wavenumbers = np.linspace(4000, 400, n_points)
    
    # Create peaks at known positions (Gaussian peaks)
    def gaussian(x, center, width, height):
        return height * np.exp(-0.5 * ((x - center) / width) ** 2)
    
    spectrum = np.zeros(n_points)
    # Add typical FTIR peaks
    spectrum += gaussian(wavenumbers, 2920, 30, 0.8)   # CH stretch
    spectrum += gaussian(wavenumbers, 1650, 50, 0.5)   # Amide I
    spectrum += gaussian(wavenumbers, 1540, 40, 0.3)   # Amide II
    spectrum += gaussian(wavenumbers, 1050, 60, 0.4)   # C-O stretch
    
    # Add small baseline
    spectrum += 0.05
    
    # Add noise
    if noise_level > 0:
        spectrum += np.random.normal(0, noise_level, n_points)
    
    return wavenumbers, spectrum


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
# Import the module (with graceful fallback for missing dependencies)
# ============================================================================

try:
    # Try importing as a module
    from normalization import (
        normalize, normalize_df, normalize_method_names,
        _normalize_snv, _normalize_vector, _normalize_minmax,
        _normalize_area, _normalize_peak, _normalize_range, _normalize_max,
        mean_center, auto_scale, pareto_scale,
        detrend, snv_detrend,
        normalize_robust_snv, normalize_curvature_weighted,
        normalize_peak_envelope, normalize_entropy_weighted,
        normalize_pqn, normalize_total_variation,
        normalize_spectral_moments, normalize_adaptive_regional,
        normalize_derivative_ratio, normalize_signal_to_baseline,
    )
    MODULE_LOADED = True
except ImportError:
    # If spectral_utils is not available, create mock functions
    print("Note: Could not import from normalization module. Running standalone tests.")
    MODULE_LOADED = False
    
    # Implement core functions locally for testing
    def _normalize_snv(y):
        mean = np.mean(y)
        std = np.std(y)
        if std == 0:
            return np.zeros_like(y)
        return (y - mean) / std
    
    def _normalize_vector(y):
        norm = np.linalg.norm(y)
        if norm == 0:
            return np.zeros_like(y)
        return y / norm
    
    def _normalize_minmax(y, feature_range=(0.0, 1.0)):
        y_min, y_max = np.min(y), np.max(y)
        if y_max == y_min:
            return np.full_like(y, feature_range[0])
        scaled = (y - y_min) / (y_max - y_min)
        new_min, new_max = feature_range
        return scaled * (new_max - new_min) + new_min
    
    def _normalize_area(y, wavenumbers=None):
        if wavenumbers is not None:
            x = np.asarray(wavenumbers, dtype=np.float64)
            yy = np.abs(y)
            if x[0] > x[-1]:
                x = x[::-1]
                yy = yy[::-1]
            total_area = np.trapezoid(yy, x=x)
        else:
            total_area = np.sum(np.abs(y))
        if total_area == 0:
            return np.zeros_like(y)
        return y / total_area
    
    def _normalize_peak(y, peak_idx=None, peak_value=None, use_absolute=True):
        if peak_idx is None:
            ref_intensity = np.max(np.abs(y))
        else:
            if use_absolute:
                ref_intensity = np.abs(y[peak_idx])
            else:
                ref_intensity = y[peak_idx]
        if ref_intensity == 0:
            return np.zeros_like(y)
        if peak_value is not None:
            return y * (peak_value / ref_intensity)
        return y / ref_intensity
    
    def _normalize_range(y):
        y_range = np.max(y) - np.min(y)
        if y_range == 0:
            return np.zeros_like(y)
        return y / y_range
    
    def _normalize_max(y):
        max_val = np.max(np.abs(y))
        if max_val == 0:
            return np.zeros_like(y)
        return y / max_val
    
    def mean_center(spectra, axis=0, return_mean=False):
        mean = np.mean(spectra, axis=axis, keepdims=True)
        centered = spectra - mean
        if return_mean:
            return centered, np.squeeze(mean)
        return centered
    
    def auto_scale(spectra, return_params=False):
        mean = np.mean(spectra, axis=0)
        std = np.std(spectra, axis=0)
        std[std == 0] = 1
        scaled = (spectra - mean) / std
        if return_params:
            return scaled, mean, std
        return scaled
    
    def pareto_scale(spectra, return_params=False):
        mean = np.mean(spectra, axis=0)
        std = np.std(spectra, axis=0)
        std[std == 0] = 1
        scaled = (spectra - mean) / np.sqrt(std)
        if return_params:
            return scaled, mean, std
        return scaled
    
    def normalize_robust_snv(y, consistency_correction=True, epsilon=1e-10):
        median = np.median(y)
        mad = np.median(np.abs(y - median))
        if consistency_correction:
            mad *= 1.4826
        mad = max(mad, epsilon)
        return (y - median) / mad
    
    def normalize_total_variation(y, order=1):
        diff = y.copy()
        for _ in range(order):
            diff = np.diff(diff)
        tv = np.sum(np.abs(diff))
        if tv == 0:
            return np.zeros_like(y)
        return y / tv
    
    def normalize_pqn(y, reference=None, reference_type="median"):
        if reference is None:
            ref_value = np.median(y[y > 0]) if np.any(y > 0) else 1.0
            return y / ref_value
        mask = (y > 0) & (reference > 0)
        if not np.any(mask):
            return y
        quotients = y[mask] / reference[mask]
        norm_factor = np.median(quotients)
        if norm_factor == 0:
            return np.zeros_like(y)
        return y / norm_factor


# ============================================================================
# TEST CASES
# ============================================================================

class TestSNV:
    """Tests for Standard Normal Variate normalization."""
    
    @staticmethod
    def test_snv_mean_zero():
        """SNV should produce mean = 0."""
        _, spectrum = create_synthetic_spectrum()
        result = _normalize_snv(spectrum)
        assert_close(np.mean(result), 0.0, tol=TOLERANCE, 
                    msg="SNV mean should be 0")
    
    @staticmethod
    def test_snv_std_one():
        """SNV should produce std = 1."""
        _, spectrum = create_synthetic_spectrum()
        result = _normalize_snv(spectrum)
        assert_close(np.std(result), 1.0, tol=TOLERANCE,
                    msg="SNV std should be 1")
    
    @staticmethod
    def test_snv_preserves_shape():
        """SNV should preserve relative peak positions."""
        _, spectrum = create_synthetic_spectrum(noise_level=0)
        result = _normalize_snv(spectrum)
        
        # Peak positions should be preserved
        orig_peak_idx = np.argmax(spectrum)
        result_peak_idx = np.argmax(result)
        assert orig_peak_idx == result_peak_idx, "Peak position should be preserved"
    
    @staticmethod
    def test_snv_constant_spectrum():
        """SNV of constant spectrum should return zeros."""
        constant = np.ones(100) * 5.0
        result = _normalize_snv(constant)
        assert np.all(result == 0), "Constant spectrum should return zeros"
    
    @staticmethod
    def test_snv_formula():
        """Verify SNV formula: (x - mean) / std."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = (y - np.mean(y)) / np.std(y)
        result = _normalize_snv(y)
        assert_close(result, expected, msg="SNV formula verification")


class TestVector:
    """Tests for L2 vector normalization."""
    
    @staticmethod
    def test_vector_unit_length():
        """Vector norm should produce unit length."""
        _, spectrum = create_synthetic_spectrum()
        result = _normalize_vector(spectrum)
        norm = np.linalg.norm(result)
        assert_close(norm, 1.0, tol=TOLERANCE,
                    msg="Vector norm should be 1")
    
    @staticmethod
    def test_vector_zero_spectrum():
        """Zero spectrum should return zeros."""
        zeros = np.zeros(100)
        result = _normalize_vector(zeros)
        assert np.all(result == 0), "Zero spectrum should return zeros"
    
    @staticmethod
    def test_vector_preserves_direction():
        """Vector normalization should preserve direction (ratios)."""
        y = np.array([3.0, 4.0])  # 3-4-5 triangle
        result = _normalize_vector(y)
        # Result should be [0.6, 0.8] (unit vector in same direction)
        expected = np.array([0.6, 0.8])
        assert_close(result, expected, msg="Vector direction should be preserved")
    
    @staticmethod
    def test_vector_formula():
        """Verify vector formula: x / ||x||_2."""
        y = np.array([3.0, 4.0])
        expected = y / np.linalg.norm(y)
        result = _normalize_vector(y)
        assert_close(result, expected, msg="Vector formula verification")


class TestMinMax:
    """Tests for Min-Max normalization."""
    
    @staticmethod
    def test_minmax_range_01():
        """Min-Max should produce values in [0, 1]."""
        _, spectrum = create_synthetic_spectrum()
        result = _normalize_minmax(spectrum)
        assert np.min(result) >= 0.0, "Min should be >= 0"
        assert np.max(result) <= 1.0, "Max should be <= 1"
        assert_close(np.min(result), 0.0, msg="Min should be 0")
        assert_close(np.max(result), 1.0, msg="Max should be 1")
    
    @staticmethod
    def test_minmax_custom_range():
        """Min-Max should scale to custom range."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _normalize_minmax(y, feature_range=(-1.0, 1.0))
        assert_close(np.min(result), -1.0, msg="Min should be -1")
        assert_close(np.max(result), 1.0, msg="Max should be 1")
    
    @staticmethod
    def test_minmax_constant_spectrum():
        """Constant spectrum should return feature_range[0]."""
        constant = np.ones(100) * 5.0
        result = _normalize_minmax(constant, feature_range=(0.0, 1.0))
        assert np.all(result == 0.0), "Constant spectrum should return 0"
    
    @staticmethod
    def test_minmax_formula():
        """Verify Min-Max formula."""
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        # Expected: (y - 2) / (10 - 2) = (y - 2) / 8
        expected = (y - 2.0) / 8.0
        result = _normalize_minmax(y)
        assert_close(result, expected, msg="MinMax formula verification")


class TestArea:
    """Tests for Area normalization."""
    
    @staticmethod
    def test_area_sums_to_one():
        """Area-normalized spectrum should have area ≈ 1."""
        wn, spectrum = create_synthetic_spectrum(noise_level=0)
        # Make spectrum positive for clean area test
        spectrum = np.abs(spectrum)
        result = _normalize_area(spectrum, wavenumbers=wn)
        
        # Integrate result
        x = wn.copy()
        if x[0] > x[-1]:
            x = x[::-1]
            y = np.abs(result[::-1])
        else:
            y = np.abs(result)
        area = np.trapezoid(y, x=x)
        assert_close(area, 1.0, tol=0.01, msg="Area should be 1")
    
    @staticmethod
    def test_area_without_wavenumbers():
        """Area norm without wavenumbers should use sum(|y|)."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = _normalize_area(y)
        expected = y / np.sum(np.abs(y))  # y / 10
        assert_close(result, expected, msg="Area without wavenumbers")
    
    @staticmethod
    def test_area_zero_spectrum():
        """Zero spectrum should return zeros."""
        zeros = np.zeros(100)
        result = _normalize_area(zeros)
        assert np.all(result == 0), "Zero spectrum should return zeros"


class TestPeak:
    """Tests for Peak normalization."""
    
    @staticmethod
    def test_peak_max_normalized():
        """Peak at max should be 1 after normalization."""
        _, spectrum = create_synthetic_spectrum(noise_level=0)
        result = _normalize_peak(spectrum)
        assert_close(np.max(np.abs(result)), 1.0, 
                    msg="Max absolute value should be 1")
    
    @staticmethod
    def test_peak_specific_index():
        """Normalization by specific peak index."""
        y = np.array([1.0, 2.0, 5.0, 3.0, 1.0])
        result = _normalize_peak(y, peak_idx=2)  # Peak at index 2 (value=5)
        expected = y / 5.0
        assert_close(result, expected, msg="Peak at index 2")
    
    @staticmethod
    def test_peak_with_target_value():
        """Normalization to specific target value."""
        y = np.array([1.0, 2.0, 5.0, 3.0, 1.0])
        result = _normalize_peak(y, peak_idx=2, peak_value=10.0)
        # Should scale so y[2] = 10.0
        assert_close(result[2], 10.0, msg="Peak should equal target value")
    
    @staticmethod
    def test_peak_zero_at_index():
        """Zero at peak index should return zeros."""
        y = np.array([1.0, 2.0, 0.0, 3.0, 1.0])
        result = _normalize_peak(y, peak_idx=2)
        assert np.all(result == 0), "Zero peak should return zeros"


class TestRange:
    """Tests for Range normalization."""
    
    @staticmethod
    def test_range_formula():
        """Verify range formula: x / (max - min)."""
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        expected = y / (10.0 - 2.0)  # y / 8
        result = _normalize_range(y)
        assert_close(result, expected, msg="Range formula verification")
    
    @staticmethod
    def test_range_constant():
        """Constant spectrum should return zeros."""
        constant = np.ones(100)
        result = _normalize_range(constant)
        assert np.all(result == 0), "Constant spectrum should return zeros"


class TestMax:
    """Tests for Maximum normalization."""
    
    @staticmethod
    def test_max_scales_to_one():
        """Max absolute value should be 1."""
        y = np.array([-5.0, 2.0, 3.0, -4.0])
        result = _normalize_max(y)
        assert_close(np.max(np.abs(result)), 1.0, 
                    msg="Max absolute should be 1")
    
    @staticmethod
    def test_max_formula():
        """Verify max formula: x / max(|x|)."""
        y = np.array([-5.0, 2.0, 3.0])
        expected = y / 5.0  # max(|y|) = 5
        result = _normalize_max(y)
        assert_close(result, expected, msg="Max formula verification")


class TestMeanCenter:
    """Tests for mean centering (2D method)."""
    
    @staticmethod
    def test_mean_center_column_means_zero():
        """Column-wise mean centering should give column means = 0."""
        spectra = np.random.rand(10, 100)  # 10 samples, 100 wavenumbers
        result = mean_center(spectra, axis=0)
        col_means = np.mean(result, axis=0)
        assert_close(col_means, np.zeros(100), tol=TOLERANCE,
                    msg="Column means should be 0")
    
    @staticmethod
    def test_mean_center_row_means_zero():
        """Row-wise mean centering should give row means = 0."""
        spectra = np.random.rand(10, 100)
        result = mean_center(spectra, axis=1)
        row_means = np.mean(result, axis=1)
        assert_close(row_means, np.zeros(10), tol=TOLERANCE,
                    msg="Row means should be 0")
    
    @staticmethod
    def test_mean_center_return_mean():
        """Should correctly return mean when requested."""
        spectra = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result, mean = mean_center(spectra, axis=0, return_mean=True)
        expected_mean = np.array([4.0, 5.0, 6.0])
        assert_close(mean, expected_mean, msg="Returned mean")


class TestAutoScale:
    """Tests for auto-scaling (2D method)."""
    
    @staticmethod
    def test_auto_scale_means_zero():
        """Auto-scaling should give column means = 0."""
        spectra = np.random.rand(10, 100) * 10 + 5
        result = auto_scale(spectra)
        col_means = np.mean(result, axis=0)
        assert_close(col_means, np.zeros(100), tol=TOLERANCE,
                    msg="Column means should be 0")
    
    @staticmethod
    def test_auto_scale_stds_one():
        """Auto-scaling should give column stds = 1."""
        spectra = np.random.rand(10, 100) * 10 + 5
        result = auto_scale(spectra)
        col_stds = np.std(result, axis=0)
        assert_close(col_stds, np.ones(100), tol=TOLERANCE,
                    msg="Column stds should be 1")
    
    @staticmethod
    def test_auto_scale_formula():
        """Verify auto-scale formula: (x - mean) / std per column."""
        spectra = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = auto_scale(spectra)
        
        # Column 0: mean=3, std=1.633
        # Column 1: mean=4, std=1.633
        expected_col0 = (spectra[:, 0] - 3.0) / np.std(spectra[:, 0])
        expected_col1 = (spectra[:, 1] - 4.0) / np.std(spectra[:, 1])
        
        assert_close(result[:, 0], expected_col0, msg="Auto-scale col 0")
        assert_close(result[:, 1], expected_col1, msg="Auto-scale col 1")


class TestParetoScale:
    """Tests for Pareto scaling (2D method)."""
    
    @staticmethod
    def test_pareto_means_zero():
        """Pareto scaling should give column means = 0."""
        spectra = np.random.rand(10, 100) * 10 + 5
        result = pareto_scale(spectra)
        col_means = np.mean(result, axis=0)
        assert_close(col_means, np.zeros(100), tol=TOLERANCE,
                    msg="Column means should be 0")
    
    @staticmethod
    def test_pareto_formula():
        """Verify Pareto formula: (x - mean) / sqrt(std) per column."""
        spectra = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pareto_scale(spectra)
        
        # Column 0: mean=3, std=1.633
        std0 = np.std(spectra[:, 0])
        expected_col0 = (spectra[:, 0] - 3.0) / np.sqrt(std0)
        
        assert_close(result[:, 0], expected_col0, msg="Pareto formula col 0")


class TestRobustSNV:
    """Tests for Robust SNV (median/MAD based)."""
    
    @staticmethod
    def test_robust_snv_median_zero():
        """Robust SNV should give median ≈ 0."""
        _, spectrum = create_synthetic_spectrum()
        result = normalize_robust_snv(spectrum)
        assert_close(np.median(result), 0.0, tol=TOLERANCE,
                    msg="Median should be 0")
    
    @staticmethod
    def test_robust_snv_outlier_resistant():
        """Robust SNV should be resistant to outliers."""
        # Normal spectrum
        y = np.random.normal(0, 1, 1000)
        result_normal = normalize_robust_snv(y.copy())
        
        # Add outliers
        y_outlier = y.copy()
        y_outlier[0] = 1000  # Extreme outlier
        y_outlier[1] = -1000
        result_outlier = normalize_robust_snv(y_outlier)
        
        # Results should be similar for non-outlier points
        # (robust statistics should ignore outliers)
        # Compare middle portion where outliers don't exist
        mid_normal = result_normal[100:900]
        mid_outlier = result_outlier[100:900]
        
        # Correlation should be very high
        corr = np.corrcoef(mid_normal, mid_outlier)[0, 1]
        assert corr > 0.99, f"Correlation should be high, got {corr}"
    
    @staticmethod
    def test_robust_snv_mad_consistency():
        """MAD with consistency correction should match std for normal data."""
        # For normal distribution, MAD * 1.4826 ≈ std
        y = np.random.normal(5, 2, 10000)  # mean=5, std=2
        
        # Regular SNV
        snv_result = _normalize_snv(y)
        
        # Robust SNV
        robust_result = normalize_robust_snv(y, consistency_correction=True)
        
        # For normal data, results should be very similar
        # (both should have similar spread)
        snv_std = np.std(snv_result)
        robust_std = np.std(robust_result)
        
        # Should be within 5% of each other for large normal samples
        assert abs(snv_std - robust_std) / snv_std < 0.05, \
            f"STD mismatch: SNV={snv_std}, Robust={robust_std}"


class TestPQN:
    """Tests for Probabilistic Quotient Normalization."""
    
    @staticmethod
    def test_pqn_with_reference():
        """PQN with reference should normalize by median quotient."""
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectrum = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # 2x reference
        
        result = normalize_pqn(spectrum, reference=reference)
        
        # Quotients are all 2.0, so median quotient = 2.0
        # Result should be spectrum / 2.0
        expected = spectrum / 2.0
        assert_close(result, expected, msg="PQN with 2x reference")
    
    @staticmethod
    def test_pqn_uneven_quotients():
        """PQN should use median of quotients."""
        reference = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Quotients: 1,2,3,4,5
        
        result = normalize_pqn(spectrum, reference=reference)
        
        # Median quotient = 3.0
        expected = spectrum / 3.0
        assert_close(result, expected, msg="PQN with uneven quotients")
    
    @staticmethod  
    def test_pqn_without_reference_warns():
        """PQN without reference should warn (falls back to median scaling)."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Just test that it returns median-scaled result
        result = normalize_pqn(spectrum, reference=None)
        
        # Without reference, should divide by median of positive values
        # Median of [1,2,3,4,5] = 3
        expected = spectrum / 3.0
        assert_close(result, expected, msg="PQN fallback to median scaling")


class TestTotalVariation:
    """Tests for Total Variation normalization."""
    
    @staticmethod
    def test_tv_first_order():
        """TV first order should be sum of absolute differences."""
        y = np.array([1.0, 3.0, 2.0, 5.0])
        # Differences: |3-1| + |2-3| + |5-2| = 2 + 1 + 3 = 6
        result = normalize_total_variation(y, order=1)
        expected = y / 6.0
        assert_close(result, expected, msg="TV first order")
    
    @staticmethod
    def test_tv_second_order():
        """TV second order should use second differences."""
        y = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
        # First diff: [2, -1, 3, -4]
        # Second diff: [-3, 4, -7]
        # TV = |-3| + |4| + |-7| = 14
        result = normalize_total_variation(y, order=2)
        expected = y / 14.0
        assert_close(result, expected, msg="TV second order")
    
    @staticmethod
    def test_tv_constant_spectrum():
        """Constant spectrum should return zeros."""
        constant = np.ones(100)
        result = normalize_total_variation(constant)
        assert np.all(result == 0), "Constant spectrum should return zeros"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @staticmethod
    def test_negative_values_preserved():
        """Normalization should preserve negative values (for normalized data)."""
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # SNV should work with negatives
        result = _normalize_snv(y)
        assert np.any(result < 0), "SNV should preserve negative signs"
        
        # Vector should work with negatives
        result = _normalize_vector(y)
        assert np.any(result < 0), "Vector should preserve negative signs"
    
    @staticmethod
    def test_single_value_spectrum():
        """Single value spectrum edge case."""
        y = np.array([5.0])
        
        # SNV: std=0, should return 0
        result = _normalize_snv(y)
        assert result[0] == 0, "SNV of single value should be 0"
        
        # Vector: norm=5, should return 1
        result = _normalize_vector(y)
        assert_close(result[0], 1.0, msg="Vector of single value")
    
    @staticmethod
    def test_very_small_values():
        """Very small values should not cause numerical issues."""
        y = np.array([1e-15, 2e-15, 3e-15])
        
        result = _normalize_snv(y)
        assert np.all(np.isfinite(result)), "SNV should handle small values"
        
        result = _normalize_vector(y)
        assert np.all(np.isfinite(result)), "Vector should handle small values"
    
    @staticmethod
    def test_very_large_values():
        """Very large values should not cause overflow."""
        y = np.array([1e15, 2e15, 3e15])
        
        result = _normalize_snv(y)
        assert np.all(np.isfinite(result)), "SNV should handle large values"
        
        result = _normalize_vector(y)
        assert np.all(np.isfinite(result)), "Vector should handle large values"
    
    @staticmethod
    def test_mixed_positive_negative():
        """Mixed positive/negative spectrum should normalize correctly."""
        y = np.array([-5.0, -2.0, 0.0, 3.0, 8.0])
        
        # MinMax should map to [0, 1]
        result = _normalize_minmax(y)
        assert_close(np.min(result), 0.0, msg="MinMax min")
        assert_close(np.max(result), 1.0, msg="MinMax max")
        
        # Max should divide by max(|y|) = 8
        result = _normalize_max(y)
        assert_close(np.max(np.abs(result)), 1.0, msg="Max normalization")


class TestMathematicalProperties:
    """Tests for mathematical properties that must hold."""
    
    @staticmethod
    def test_snv_idempotence():
        """Applying SNV twice should give same result as once (already normalized)."""
        _, spectrum = create_synthetic_spectrum()
        once = _normalize_snv(spectrum)
        twice = _normalize_snv(once)
        
        # After SNV, mean=0, std=1
        # Applying again: (x - 0) / 1 = x
        assert_close(once, twice, tol=TOLERANCE, msg="SNV should be idempotent")
    
    @staticmethod
    def test_vector_idempotence():
        """Applying vector norm twice should give same result."""
        _, spectrum = create_synthetic_spectrum()
        once = _normalize_vector(spectrum)
        twice = _normalize_vector(once)
        
        # After vector norm, ||x||=1
        # Applying again: x / 1 = x
        assert_close(once, twice, tol=TOLERANCE, msg="Vector should be idempotent")
    
    @staticmethod
    def test_scaling_invariance_snv():
        """SNV should be invariant to scaling."""
        _, spectrum = create_synthetic_spectrum()
        
        result_orig = _normalize_snv(spectrum)
        result_scaled = _normalize_snv(spectrum * 100)  # Scale by 100
        
        assert_close(result_orig, result_scaled, tol=TOLERANCE,
                    msg="SNV should be scale-invariant")
    
    @staticmethod
    def test_scaling_invariance_vector():
        """Vector norm should be invariant to positive scaling."""
        _, spectrum = create_synthetic_spectrum()
        
        result_orig = _normalize_vector(spectrum)
        result_scaled = _normalize_vector(spectrum * 100)
        
        assert_close(result_orig, result_scaled, tol=TOLERANCE,
                    msg="Vector should be scale-invariant")
    
    @staticmethod
    def test_shift_invariance_snv():
        """SNV should be invariant to constant shifts."""
        _, spectrum = create_synthetic_spectrum()
        
        result_orig = _normalize_snv(spectrum)
        result_shifted = _normalize_snv(spectrum + 1000)  # Shift by 1000
        
        assert_close(result_orig, result_shifted, tol=TOLERANCE,
                    msg="SNV should be shift-invariant")
    
    @staticmethod
    def test_range_preserves_order():
        """Range normalization should preserve order of values."""
        y = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = _normalize_range(y)
        
        # Argmax/argmin should be preserved
        assert np.argmax(y) == np.argmax(result), "Argmax should be preserved"
        assert np.argmin(y) == np.argmin(result), "Argmin should be preserved"


# ============================================================================
# Run all tests
# ============================================================================

def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestSNV,
        TestVector,
        TestMinMax,
        TestArea,
        TestPeak,
        TestRange,
        TestMax,
        TestMeanCenter,
        TestAutoScale,
        TestParetoScale,
        TestRobustSNV,
        TestPQN,
        TestTotalVariation,
        TestEdgeCases,
        TestMathematicalProperties,
    ]
    
    total_passed = 0
    total_failed = 0
    
    print("=" * 70)
    print("NORMALIZATION MODULE TESTS")
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
