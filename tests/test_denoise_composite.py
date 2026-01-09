"""
Test script to demonstrate composite scoring for denoising evaluation.

This script shows how to use the find_best_denoising_method() function
to get a composite score similar to baseline.py's approach.
"""

import numpy as np
import pandas as pd
from utils.denoise import evaluate_denoising_methods, find_best_denoising_method

# Create synthetic test data (mimicking FTIR spectra)
np.random.seed(42)
n_samples = 20
n_wavenumbers = 500

# Generate wavenumbers
wavenumbers = np.linspace(650, 4000, n_wavenumbers)

# Generate synthetic spectra with noise
# Gaussian peaks at different positions
def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

spectra = []
for i in range(n_samples):
    # Create baseline spectrum with multiple peaks
    spectrum = np.zeros(n_wavenumbers)

    # Add several Gaussian peaks at typical FTIR positions
    spectrum += gaussian(wavenumbers, 2916, 30, 0.8)  # CH2 asymmetric stretch
    spectrum += gaussian(wavenumbers, 2850, 25, 0.6)  # CH2 symmetric stretch
    spectrum += gaussian(wavenumbers, 1740, 20, 0.5)  # C=O stretch
    spectrum += gaussian(wavenumbers, 1465, 15, 0.4)  # CH2 bend

    # Add random noise
    noise = np.random.normal(0, 0.02, n_wavenumbers)
    spectrum += noise

    spectra.append(spectrum)

# Create DataFrame in wide format
df = pd.DataFrame(spectra, columns=wavenumbers.astype(str))
df['label'] = [f'sample_{i}' for i in range(n_samples)]

# Reorder columns to have label first
cols = ['label'] + [c for c in df.columns if c != 'label']
df = df[cols]

print("=" * 80)
print("DENOISING METHOD EVALUATION WITH COMPOSITE SCORING")
print("=" * 80)
print(f"\nDataset: {n_samples} samples, {n_wavenumbers} wavenumbers")
print(f"Wavenumber range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")

# Evaluate denoising methods (use subset of methods for faster demo)
print("\n" + "-" * 80)
print("Step 1: Evaluating denoising methods...")
print("-" * 80)

methods_to_test = ['savgol', 'gaussian', 'median', 'whittaker', 'moving_average']
eval_results = evaluate_denoising_methods(
    df,
    methods=methods_to_test,
    n_samples=15,  # Use subset for faster evaluation
    sample_selection='random',
    random_state=42,
    n_jobs=-1
)

print(f"\nEvaluation complete! Generated {len(eval_results)} result entries")
print(f"Metrics: {eval_results.columns.tolist()}")

# Show sample of raw evaluation results
print("\n" + "-" * 80)
print("Sample raw evaluation results:")
print("-" * 80)
print(eval_results.groupby('method').agg({
    'snr_db': ['mean', 'std'],
    'smoothness': ['mean', 'std'],
    'fidelity': ['mean', 'std'],
    'time_ms': ['mean', 'std']
}).round(3))

# Find best methods using composite scoring
print("\n" + "=" * 80)
print("Step 2: Finding best methods with composite scoring...")
print("=" * 80)

recommendations = find_best_denoising_method(
    eval_results,
    snr_min=10.0,        # Minimum SNR threshold
    smoothness_min=1e3,  # Minimum smoothness threshold
    fidelity_min=0.9,    # Minimum fidelity threshold
    time_max_ms=50.0,    # Maximum computation time
    top_n=5
)

print("\nTOP 5 RECOMMENDED METHODS (by composite score):")
print("-" * 80)
print(recommendations.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
Composite Score Breakdown (0-1 scale, higher is better):
  - 30% weight: SNR (signal-to-noise ratio)
  - 25% weight: Smoothness (inverse of 2nd derivative variance)
  - 30% weight: Fidelity (correlation with original signal)
  - 5%  weight: Speed (inverse of computation time)
  - 10% weight: Pass rate (consistency across samples)

Key Metrics:
  - median_snr_db: Higher values indicate better noise reduction
  - median_smoothness: Higher values indicate smoother results
  - median_fidelity: Values close to 1.0 indicate minimal distortion
  - median_time_ms: Lower values indicate faster computation
  - pass_rate: Fraction of samples passing all thresholds (0-1)
  - composite_score: Overall ranking score (0-1)

Recommended Usage:
  1. Look at the top-ranked method (highest composite_score)
  2. Check if pass_rate is acceptable (> 0.7 is good)
  3. Consider trade-offs between fidelity and smoothness
  4. For production use, prioritize methods with high pass_rate
""")

print("=" * 80)
print("COMPARISON WITH BASELINE.PY APPROACH")
print("=" * 80)
print("""
Similarities:
  ✓ Both use composite scoring to rank methods
  ✓ Both normalize metrics to [0, 1] range
  ✓ Both include a pass_rate metric for consistency
  ✓ Both return top_n ranked methods

Differences:
  • Baseline uses: RFZN (30%), NAR (30%), SNR (30%), pass_rate (10%)
  • Denoising uses: SNR (30%), Smoothness (25%), Fidelity (30%),
                    Speed (5%), pass_rate (10%)

  • Baseline focuses on: noise reduction and artifact avoidance
  • Denoising focuses on: signal preservation and smoothness

Both approaches provide a unified metric to select the best method
for your specific data and requirements.
""")
