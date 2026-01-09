"""
Example: Memory-Safe Denoising Method Evaluation

This script demonstrates how to safely evaluate denoising methods
without causing memory issues, even on systems with limited RAM.
"""

import numpy as np
import pandas as pd
from utils.denoise import (
    evaluate_denoising_methods_safe,
    find_best_denoising_method,
    plot_denoising_evaluation
)

# Generate synthetic FTIR spectra (replace with your actual data)
np.random.seed(42)
n_samples = 200  # Even with 200 samples, safe wrapper handles it
n_wavenumbers = 1000

wavenumbers = np.linspace(650, 4000, n_wavenumbers)

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

spectra = []
for i in range(n_samples):
    spectrum = np.zeros(n_wavenumbers)
    spectrum += gaussian(wavenumbers, 2916, 30, 0.8)
    spectrum += gaussian(wavenumbers, 2850, 25, 0.6)
    spectrum += gaussian(wavenumbers, 1740, 20, 0.5)
    spectrum += np.random.normal(0, 0.02, n_wavenumbers)
    spectra.append(spectrum)

df = pd.DataFrame(spectra, columns=wavenumbers.astype(str))
df['label'] = [f'sample_{i}' for i in range(n_samples)]

print("=" * 80)
print("MEMORY-SAFE DENOISING EVALUATION EXAMPLE")
print("=" * 80)
print(f"\nDataset: {n_samples} samples, {n_wavenumbers} wavenumbers")

# ============================================================================
# METHOD 1: Use the safe wrapper (RECOMMENDED)
# ============================================================================
print("\n" + "=" * 80)
print("Method 1: Using evaluate_denoising_methods_safe() [RECOMMENDED]")
print("=" * 80)

# This will automatically use safe settings:
# - n_samples=50 (not 200)
# - n_jobs=2 (not all cores)
# - methods=['savgol', 'gaussian', 'median'] (not all 7 methods)
results_safe = evaluate_denoising_methods_safe(df)

print(f"\n✓ Evaluation complete!")
print(f"  Evaluated: {results_safe['method'].nunique()} methods")
print(f"  On: {results_safe['sample'].nunique()} samples")
print(f"  Total evaluations: {len(results_safe)} rows")

# Get recommendations
recommendations = find_best_denoising_method(results_safe, top_n=3)
print("\n" + "-" * 80)
print("TOP 3 RECOMMENDED METHODS:")
print("-" * 80)
print(recommendations.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ============================================================================
# METHOD 2: Manual safe configuration (for more control)
# ============================================================================
print("\n\n" + "=" * 80)
print("Method 2: Manual safe configuration [MORE CONTROL]")
print("=" * 80)

from utils.denoise import evaluate_denoising_methods

# Manually specify safe parameters
results_manual = evaluate_denoising_methods(
    df,
    methods=['savgol', 'whittaker'],  # Just 2 methods
    n_samples=30,                      # Very conservative
    n_jobs=1,                          # Single-threaded (safest)
    sample_selection='random',
    random_state=42
)

print(f"\n✓ Manual evaluation complete!")
print(f"  Evaluated: {results_manual['method'].nunique()} methods")
print(f"  On: {results_manual['sample'].nunique()} samples")

recommendations_manual = find_best_denoising_method(results_manual, top_n=2)
print("\n" + "-" * 80)
print("RECOMMENDATIONS (manual config):")
print("-" * 80)
print(recommendations_manual.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ============================================================================
# METHOD 3: Customize safe wrapper (override specific parameters)
# ============================================================================
print("\n\n" + "=" * 80)
print("Method 3: Customize safe wrapper [BEST OF BOTH WORLDS]")
print("=" * 80)

# Use safe wrapper but test more methods
results_custom = evaluate_denoising_methods_safe(
    df,
    methods=['savgol', 'gaussian', 'median', 'whittaker', 'lowpass'],  # 5 methods
    n_samples=75  # Override default (50) but still safe
    # n_jobs=2 is still used from safe defaults
)

print(f"\n✓ Custom evaluation complete!")
print(f"  Evaluated: {results_custom['method'].nunique()} methods")
print(f"  On: {results_custom['sample'].nunique()} samples")

recommendations_custom = find_best_denoising_method(results_custom, top_n=3)
print("\n" + "-" * 80)
print("RECOMMENDATIONS (custom config):")
print("-" * 80)
print(recommendations_custom.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ============================================================================
# VISUALIZATION (optional)
# ============================================================================
print("\n\n" + "=" * 80)
print("Generating comparison plots...")
print("=" * 80)

try:
    # Plot evaluation results
    plot_denoising_evaluation(
        results_safe,
        metrics=['snr_db', 'fidelity'],  # Just 2 metrics for speed
        show_mean_sd=True,
        save_plot=False
    )
    print("✓ Plots generated successfully!")
except Exception as e:
    print(f"Plotting skipped (non-interactive environment or missing display): {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✓ All three methods completed without memory issues!

Key Takeaways:
1. evaluate_denoising_methods_safe() is the safest option
2. You can customize it while keeping safe defaults
3. 50 samples is usually enough to get stable metrics
4. n_jobs=2 prevents memory issues while still being faster than n_jobs=1

Best Practice Workflow:
1. Start with evaluate_denoising_methods_safe(df)
2. Get recommendations with find_best_denoising_method()
3. If you need to test more methods/samples, increase gradually
4. Never use n_jobs=-1 on laptops or systems with <16 GB RAM

For more details, see:
- MEMORY_MANAGEMENT_GUIDE.md (detailed explanations)
- QUICK_REFERENCE_MEMORY.md (quick lookup table)
""")

print("=" * 80)
print("Example completed successfully!")
print("=" * 80)
