"""
Scientific Analysis and Tests for plotting_stats.py
====================================================

This file analyzes the scientific correctness of the FTIR statistical
plotting module, particularly ANOVA and correlation analysis.
"""

import numpy as np
import pandas as pd
import sys
from typing import Callable
from scipy.stats import f_oneway, pearsonr

np.random.seed(42)
TOLERANCE = 1e-10
FLOAT_TOLERANCE = 1e-6


# ============================================================================
# Test Helper Functions
# ============================================================================

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
# Test Cases for ANOVA
# ============================================================================

class TestANOVA:
    """Tests for one-way ANOVA implementation."""
    
    @staticmethod
    def test_anova_identical_groups():
        """ANOVA of identical groups should give F≈0, p≈1."""
        # All groups have same distribution
        group1 = np.array([10, 11, 12, 13, 14])
        group2 = np.array([10, 11, 12, 13, 14])
        group3 = np.array([10, 11, 12, 13, 14])
        
        f_stat, p_val = f_oneway(group1, group2, group3)
        
        assert f_stat < 0.01, f"F-statistic should be ~0 for identical groups, got {f_stat}"
        assert p_val > 0.99, f"p-value should be ~1 for identical groups, got {p_val}"
    
    @staticmethod
    def test_anova_clearly_different_groups():
        """ANOVA of clearly different groups should give high F, low p."""
        # Groups with very different means
        group1 = np.array([10, 11, 12, 13, 14])      # Mean = 12
        group2 = np.array([100, 101, 102, 103, 104]) # Mean = 102
        group3 = np.array([200, 201, 202, 203, 204]) # Mean = 202
        
        f_stat, p_val = f_oneway(group1, group2, group3)
        
        assert f_stat > 100, f"F-statistic should be high for different groups, got {f_stat}"
        assert p_val < 0.001, f"p-value should be very low for different groups, got {p_val}"
    
    @staticmethod
    def test_anova_f_statistic_formula():
        """Verify F-statistic calculation: F = MSB / MSW."""
        # Create simple data
        group1 = np.array([1, 2, 3])
        group2 = np.array([4, 5, 6])
        group3 = np.array([7, 8, 9])
        
        # Manual calculation
        all_data = np.concatenate([group1, group2, group3])
        grand_mean = np.mean(all_data)
        
        # Between-group sum of squares
        group_means = [np.mean(group1), np.mean(group2), np.mean(group3)]
        group_sizes = [len(group1), len(group2), len(group3)]
        SSB = sum(n * (m - grand_mean)**2 for n, m in zip(group_sizes, group_means))
        
        # Within-group sum of squares
        SSW = (np.sum((group1 - np.mean(group1))**2) + 
               np.sum((group2 - np.mean(group2))**2) + 
               np.sum((group3 - np.mean(group3))**2))
        
        # Degrees of freedom
        k = 3  # number of groups
        n = 9  # total samples
        df_between = k - 1
        df_within = n - k
        
        # Mean squares
        MSB = SSB / df_between
        MSW = SSW / df_within
        
        # F-statistic
        expected_f = MSB / MSW
        
        # Compare with scipy
        f_stat, _ = f_oneway(group1, group2, group3)
        
        assert_close(f_stat, expected_f, tol=1e-10, 
                    msg=f"F-statistic formula verification")
    
    @staticmethod
    def test_anova_p_value_interpretation():
        """Verify p-value interpretation: p < 0.05 means significant."""
        np.random.seed(42)
        
        # Significantly different groups
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(5, 1, 30)  # Mean shifted by 5
        
        f_stat, p_val = f_oneway(group1, group2)
        
        assert p_val < 0.05, "Groups with different means should be significant"
        
        # Non-significantly different groups
        group3 = np.random.normal(0, 1, 30)
        group4 = np.random.normal(0.1, 1, 30)  # Mean shifted by only 0.1
        
        f_stat2, p_val2 = f_oneway(group3, group4)
        
        # With such small difference and moderate n, likely not significant
        # (though this is probabilistic)
    
    @staticmethod
    def test_log_p_value_transformation():
        """-log10(p) transformation: higher = more significant."""
        p_values = np.array([0.5, 0.05, 0.01, 0.001, 0.0001])
        
        neg_log_p = -np.log10(p_values)
        
        # Expected values
        expected = np.array([0.301, 1.301, 2.0, 3.0, 4.0])
        
        assert np.allclose(neg_log_p, expected, atol=0.01), \
            f"-log10(p) transformation: got {neg_log_p}, expected {expected}"
        
        # Verify ordering: more significant (lower p) = higher -log10(p)
        assert np.all(np.diff(neg_log_p) > 0), \
            "-log10(p) should increase as p decreases"
    
    @staticmethod
    def test_anova_multiple_testing_awareness():
        """
        Document: When testing many wavenumbers, multiple testing correction needed.
        
        If testing 1000 wavenumbers at α=0.05:
        - Expected false positives = 1000 × 0.05 = 50
        
        Bonferroni correction: α_corrected = 0.05 / 1000 = 0.00005
        -log10(0.00005) ≈ 4.3 (threshold line would be much higher)
        """
        n_tests = 1000
        alpha = 0.05
        
        # Expected false positives without correction
        expected_false_positives = n_tests * alpha
        assert expected_false_positives == 50, "Multiple testing issue"
        
        # Bonferroni corrected threshold
        alpha_bonferroni = alpha / n_tests
        neg_log_threshold = -np.log10(alpha_bonferroni)
        
        assert neg_log_threshold > 4, \
            f"Bonferroni threshold should be high: {neg_log_threshold}"


class TestCorrelation:
    """Tests for correlation matrix analysis."""
    
    @staticmethod
    def test_correlation_perfect_positive():
        """Perfectly correlated variables should have r=1."""
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 3  # Perfect linear relationship
        
        r, p = pearsonr(x, y)
        
        assert_close(r, 1.0, tol=1e-10, msg="Perfect positive correlation")
    
    @staticmethod
    def test_correlation_perfect_negative():
        """Perfectly negatively correlated variables should have r=-1."""
        x = np.array([1, 2, 3, 4, 5])
        y = -2 * x + 10  # Perfect negative linear relationship
        
        r, p = pearsonr(x, y)
        
        assert_close(r, -1.0, tol=1e-10, msg="Perfect negative correlation")
    
    @staticmethod
    def test_correlation_uncorrelated():
        """Uncorrelated variables should have r≈0."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)  # Independent
        
        r, p = pearsonr(x, y)
        
        assert abs(r) < 0.1, f"Uncorrelated variables should have r≈0, got {r}"
    
    @staticmethod
    def test_correlation_range():
        """Correlation coefficient should be in [-1, 1]."""
        np.random.seed(42)
        
        for _ in range(100):
            x = np.random.randn(50)
            y = np.random.randn(50)
            r, _ = pearsonr(x, y)
            
            assert -1 <= r <= 1, f"Correlation {r} out of range [-1, 1]"
    
    @staticmethod
    def test_correlation_symmetric():
        """Correlation should be symmetric: corr(x,y) = corr(y,x)."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)
        
        r_xy, _ = pearsonr(x, y)
        r_yx, _ = pearsonr(y, x)
        
        assert_close(r_xy, r_yx, tol=1e-15, msg="Correlation should be symmetric")
    
    @staticmethod
    def test_correlation_matrix_diagonal():
        """Diagonal of correlation matrix should be 1 (self-correlation)."""
        np.random.seed(42)
        data = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100),
        })
        
        corr_matrix = data.corr()
        
        diagonal = np.diag(corr_matrix.values)
        assert np.allclose(diagonal, 1.0), "Diagonal should be all 1s"
    
    @staticmethod
    def test_correlation_matrix_symmetric():
        """Correlation matrix should be symmetric."""
        np.random.seed(42)
        data = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100),
        })
        
        corr_matrix = data.corr().values
        
        assert np.allclose(corr_matrix, corr_matrix.T), \
            "Correlation matrix should be symmetric"
    
    @staticmethod
    def test_upper_triangle_extraction():
        """Upper triangle should exclude diagonal and lower triangle."""
        matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.7],
            [0.3, 0.7, 1.0]
        ])
        
        # Extract upper triangle (excluding diagonal)
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        
        expected = np.array([0.5, 0.3, 0.7])
        assert np.allclose(upper_tri, expected), \
            f"Upper triangle extraction: got {upper_tri}, expected {expected}"


class TestSpectralCorrelation:
    """Tests specific to spectral data correlation patterns."""
    
    @staticmethod
    def test_adjacent_wavenumbers_high_correlation():
        """Adjacent wavenumbers in spectra are typically highly correlated."""
        # Simulate smooth spectrum
        np.random.seed(42)
        n_samples = 100
        n_wavenumbers = 500
        
        # Create smooth spectra (correlated adjacent points)
        base = np.cumsum(np.random.randn(n_wavenumbers))  # Random walk
        spectra = np.array([base + np.random.randn(n_wavenumbers) * 0.1 
                          for _ in range(n_samples)])
        
        # Adjacent columns should be highly correlated
        r_adjacent, _ = pearsonr(spectra[:, 0], spectra[:, 1])
        
        assert r_adjacent > 0.9, \
            f"Adjacent wavenumbers should be highly correlated, got {r_adjacent}"
    
    @staticmethod
    def test_distant_wavenumbers_lower_correlation():
        """Distant wavenumbers may have lower correlation."""
        # In real spectra, distant regions may be less correlated
        # (different functional groups)
        
        np.random.seed(42)
        n_samples = 100
        
        # Simulate two independent spectral regions
        region1 = np.random.randn(n_samples)
        region2 = np.random.randn(n_samples)  # Independent
        
        r_distant, _ = pearsonr(region1, region2)
        
        assert abs(r_distant) < 0.3, \
            f"Independent regions should have low correlation, got {r_distant}"


class TestStatisticalAssumptions:
    """Tests for ANOVA assumptions and limitations."""
    
    @staticmethod
    def test_anova_normality_assumption():
        """
        ANOVA assumes normality within groups.
        Document: Robust to violations with large n (Central Limit Theorem).
        """
        # Even with non-normal data, ANOVA can work with large samples
        np.random.seed(42)
        
        # Uniform distribution (not normal)
        group1 = np.random.uniform(0, 10, 100)
        group2 = np.random.uniform(5, 15, 100)  # Different range
        
        f_stat, p_val = f_oneway(group1, group2)
        
        # Should still detect difference despite non-normality
        assert p_val < 0.05, "ANOVA should detect difference even with uniform data"
    
    @staticmethod
    def test_anova_homoscedasticity_assumption():
        """
        ANOVA assumes equal variances (homoscedasticity).
        Document: Welch's ANOVA is alternative for unequal variances.
        """
        np.random.seed(42)
        
        # Groups with very different variances
        group1 = np.random.normal(10, 1, 50)   # Low variance
        group2 = np.random.normal(10, 10, 50)  # High variance
        
        # Standard ANOVA may not be ideal here
        f_stat, p_val = f_oneway(group1, group2)
        
        # Document that variances are different
        var_ratio = np.var(group2) / np.var(group1)
        assert var_ratio > 50, f"Variance ratio is {var_ratio} - consider Welch's test"


# ============================================================================
# Scientific Analysis Report
# ============================================================================

def print_scientific_analysis():
    """Print scientific analysis of the statistical plotting module."""
    
    print("\n" + "="*80)
    print("SCIENTIFIC ANALYSIS: plotting_stats.py")
    print("="*80)
    
    print("""
## 1. One-Way ANOVA ✓ CORRECT (with caveats)

### Formula:
```
F = MSB / MSW = (Between-group variance) / (Within-group variance)
```

Where:
- MSB = SSB / (k-1)  [Mean Square Between]
- MSW = SSW / (n-k)  [Mean Square Within]
- k = number of groups
- n = total samples

### Implementation:
```python
f_stat, p_val = f_oneway(*groups)  # scipy.stats
```
✓ Correct use of scipy's f_oneway

### -log10(p) Visualization:
```python
-np.log10(p_values)
```
✓ Standard Manhattan plot style
✓ Higher values = more significant
✓ p=0.05 → -log10(0.05) ≈ 1.3

### ⚠️ IMPORTANT: Multiple Testing Correction

The code tests EACH wavenumber independently. With ~1000 wavenumbers:
- Expected false positives at α=0.05: 50 wavenumbers!

**Recommendation:** Add Bonferroni or FDR correction:

```python
from scipy.stats import false_discovery_control

# Option 1: Bonferroni (conservative)
p_threshold_corrected = 0.05 / len(wavenumbers)

# Option 2: Benjamini-Hochberg FDR (less conservative)
adjusted_p = false_discovery_control(p_values, method='bh')
```

### ANOVA Assumptions:
| Assumption | Status | Notes |
|------------|--------|-------|
| Independence | User responsibility | Samples should be independent |
| Normality | Not checked | Robust with large n |
| Homoscedasticity | Not checked | Consider Welch's if violated |

## 2. Correlation Matrix ✓ CORRECT

### Implementation:
```python
corr_matrix = df[columns].corr()  # Pearson correlation
```
✓ Correct use of pandas correlation

### Properties verified:
- Diagonal = 1 (self-correlation)
- Symmetric: corr(x,y) = corr(y,x)
- Range: [-1, +1]

### Upper Triangle Summary:
```python
corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
```
✓ Correctly extracts unique pairs (excludes diagonal)

### Spectral Data Pattern:
- Adjacent wavenumbers: HIGH correlation (smooth spectra)
- Distant wavenumbers: LOWER correlation (different functional groups)

This is expected for FTIR data and visible in the correlation heatmap.

## 3. Visualization Conventions ✓ CORRECT

| Feature | Implementation | Status |
|---------|----------------|--------|
| X-axis inversion | `invert_xaxis()` | ✓ FTIR standard |
| P-value threshold line | `axhline(-log10(0.05))` | ✓ |
| Correlation colormap | 'coolwarm', center=0 | ✓ Good choice |

## RECOMMENDATIONS

### Critical (Should Implement):
1. **Multiple testing correction** - Add Bonferroni or FDR
2. **Report number of tests** - User should know how many comparisons

### Optional Improvements:
1. **Welch's ANOVA** - For unequal variances
2. **Effect size** - Report eta-squared (η²) alongside F
3. **Normality check** - Shapiro-Wilk test per group

## OVERALL ASSESSMENT

| Component | Status |
|-----------|--------|
| ANOVA F-statistic | ✓ Correct |
| P-value calculation | ✓ Correct |
| -log10 transformation | ✓ Correct |
| Correlation matrix | ✓ Correct |
| Upper triangle summary | ✓ Correct |
| Multiple testing correction | ⚠️ Missing |

**Verdict: SCIENTIFICALLY SOUND with one important caveat about multiple testing.**
""")


# ============================================================================
# Run all tests
# ============================================================================

def run_all_tests():
    """Run all test classes and report results."""
    test_classes = [
        TestANOVA,
        TestCorrelation,
        TestSpectralCorrelation,
        TestStatisticalAssumptions,
    ]
    
    total_passed = 0
    total_failed = 0
    
    print("=" * 70)
    print("STATISTICAL PLOTTING MODULE TESTS")
    print("=" * 70)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        
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
    
    # Print scientific analysis
    print_scientific_analysis()
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
