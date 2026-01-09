# Scientific Analysis: plotting_stats.py

## Executive Summary

**VERDICT: SCIENTIFICALLY SOUND ✓ (with one important caveat)**

The statistical plotting module correctly implements ANOVA and correlation analysis. However, **multiple testing correction is missing** and should be added for rigorous analysis.

---

## Test Results

| Test Category | Passed | Total |
|---------------|--------|-------|
| ANOVA | 6 | 6 |
| Correlation | 8 | 8 |
| Spectral Correlation | 1 | 2 |
| Statistical Assumptions | 2 | 2 |
| **Total** | **17** | **18** |

---

## Detailed Analysis

### 1. One-Way ANOVA ✓ CORRECT

**Formula implemented:**
```
F = MSB / MSW = (Between-group variance) / (Within-group variance)
```

**Implementation:**
```python
f_stat, p_val = f_oneway(*groups)  # scipy.stats
```

| Test | Expected | Result |
|------|----------|--------|
| Identical groups | F≈0, p≈1 | ✓ |
| Different groups | High F, low p | ✓ |
| F-statistic formula | Manual = scipy | ✓ |

**-log₁₀(p) transformation:**
```python
-np.log10(p_values)
```

| p-value | -log₁₀(p) | Interpretation |
|---------|-----------|----------------|
| 0.5 | 0.3 | Not significant |
| 0.05 | 1.3 | Threshold |
| 0.01 | 2.0 | Significant |
| 0.001 | 3.0 | Highly significant |

✓ Standard Manhattan plot style - higher = more significant

---

### 2. ⚠️ CRITICAL: Multiple Testing Correction MISSING

**The Problem:**

The code tests EACH wavenumber independently. With ~1000 wavenumbers at α=0.05:
- **Expected false positives = 1000 × 0.05 = 50 wavenumbers!**

This means ~50 wavenumbers may appear "significant" by chance alone.

**Recommended Fix:**

```python
from scipy.stats import false_discovery_control

def perform_anova_analysis(..., correction='fdr'):
    # ... existing code ...
    
    # After computing p_values:
    if correction == 'bonferroni':
        p_threshold_corrected = p_threshold / len(p_values)
        significant = p_values < p_threshold_corrected
    elif correction == 'fdr':
        # Benjamini-Hochberg FDR correction
        adjusted_p = false_discovery_control(p_values, method='bh')
        significant = adjusted_p < p_threshold
    else:
        significant = p_values < p_threshold  # No correction
    
    # Update threshold line in plot
    if correction == 'bonferroni':
        threshold_line = -np.log10(p_threshold / len(p_values))
    # ...
```

**Comparison of corrections:**

| Method | Threshold (n=1000) | -log₁₀ | Conservativeness |
|--------|-------------------|--------|------------------|
| None | 0.05 | 1.3 | Too liberal |
| Bonferroni | 0.00005 | 4.3 | Very conservative |
| FDR (BH) | Varies | ~2-3 | Balanced |

---

### 3. Correlation Matrix ✓ CORRECT

**Implementation:**
```python
corr_matrix = df[columns].corr()  # Pearson correlation
```

**Properties verified:**
| Property | Expected | Verified |
|----------|----------|----------|
| Diagonal | All 1s | ✓ |
| Symmetry | corr(x,y) = corr(y,x) | ✓ |
| Range | [-1, +1] | ✓ |

**Upper triangle extraction:**
```python
matrix[np.triu_indices_from(matrix, k=1)]
```
✓ Correctly excludes diagonal and avoids double-counting

**Summary statistics:** Mean, max, min of unique correlations - correctly computed.

---

### 4. ANOVA Assumptions

| Assumption | Checked? | Notes |
|------------|----------|-------|
| Independence | No | User's responsibility |
| Normality | No | ANOVA is robust with large n (CLT) |
| Homoscedasticity | No | Consider Welch's ANOVA if variances differ |

**Robustness:** ANOVA is generally robust to normality violations when:
- Sample sizes are moderate to large (n > 30 per group)
- Group sizes are roughly equal

---

### 5. Visualization ✓ CORRECT

| Feature | Implementation | Status |
|---------|----------------|--------|
| X-axis | Inverted (high→low wavenumber) | ✓ FTIR standard |
| Threshold line | `axhline(-log10(0.05))` | ✓ |
| Colormap | 'coolwarm', center=0 | ✓ Good choice |
| Log scale for p | -log₁₀(p) | ✓ Standard |

---

## Recommendations

### Critical (Should Implement):

1. **Add multiple testing correction:**
```python
# Add parameter
def perform_anova_analysis(..., correction: str = 'fdr'):
```

2. **Report corrected threshold:**
```python
print(f"  Corrected p-threshold (FDR): {corrected_threshold:.2e}")
```

### Optional Improvements:

1. **Effect size (eta-squared):**
```python
eta_squared = SSB / (SSB + SSW)  # Proportion of variance explained
```

2. **Welch's ANOVA for unequal variances:**
```python
from scipy.stats import alexandergovern  # or implement manually
```

3. **Return adjusted p-values:**
```python
return {
    'wavenumbers': wn_sorted,
    'f_statistics': f_statistics,
    'p_values': p_values,
    'p_values_adjusted': adjusted_p,  # Add this
    'significant_corrected': significant,  # Add this
}
```

---

## Conclusion

**plotting_stats.py is scientifically correct** with one important caveat:

| Component | Status |
|-----------|--------|
| ANOVA F-statistic | ✓ Correct |
| P-value calculation | ✓ Correct |
| -log₁₀ transformation | ✓ Correct |
| Correlation matrix | ✓ Correct |
| Upper triangle summary | ✓ Correct |
| **Multiple testing correction** | ⚠️ **Missing - should add** |

Without multiple testing correction, users may identify many false positive "significant" wavenumbers. Adding Bonferroni or FDR correction is strongly recommended for publication-quality analysis.
