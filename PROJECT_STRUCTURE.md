# Xpectrass Project Structure

## Overview
This document describes the complete project structure for xpectrass v0.0.4, ready for PyPI, GitHub, and ReadTheDocs.

## Directory Structure

```
xpectrass_app/
├── .github/                      # GitHub-specific files
│   ├── workflows/
│   │   ├── tests.yml             # CI/CD testing workflow
│   │   └── publish.yml           # PyPI publishing workflow
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md         # Bug report template
│   │   └── feature_request.md    # Feature request template
│   └── PULL_REQUEST_TEMPLATE.md  # PR template
│
├── docs/                         # Sphinx documentation
│   ├── _static/                  # Static assets
│   │   └── custom.css
│   ├── api/                      # API reference
│   │   ├── index.md
│   │   ├── preprocessing_pipeline.md
│   │   └── utils.md
│   ├── user_guide/               # User guides
│   │   ├── index.md
│   │   ├── preprocessing_pipeline.md
│   │   ├── baseline_correction.md
│   │   ├── denoising.md
│   │   ├── normalization.md
│   │   ├── atmospheric_correction.md
│   │   ├── data_loading.md
│   │   ├── analysis.md
│   │   ├── machine_learning.md
│   │   ├── region_selection.md
│   │   ├── scatter_correction.md
│   │   ├── spectral_derivatives.md
│   │   └── data_validation.md
│   ├── changelog.md              # Version history for docs
│   ├── conf.py                   # Sphinx configuration
│   ├── examples.md               # Usage examples
│   ├── getting_started.md        # Getting started guide
│   ├── index.md                  # Documentation homepage
│   └── requirements.txt         # Documentation dependencies
│
├── notebooks/                    # Jupyter notebooks
│   ├── _1_select_denoising_method.ipynb
│   ├── _2_select_baseline_correction_method.ipynb
│   ├── _3_select_normalization_method.ipynb
│   ├── _4_get_processed_data_one.ipynb
│   ├── _5_get_processed_data_all.ipynb
│   └── _6_basic_data_analysis.ipynb
│
├── tests/                        # Test suite (at root level)
│   ├── test_denoise_composite.py
│   ├── test_ml.py
│   ├── test_normalization.py
│   ├── test_normalization_eval.py
│   ├── test_plotting_clus.py
│   ├── test_plotting_dim.py
│   ├── test_plotting_stats.py
│   ├── test_plottingx.py
│   ├── example_safe_evaluation.py
│   ├── PLOTTING_CLUS_ANALYSIS.md
│   ├── PLOTTING_DIM_ANALYSIS.md
│   ├── PLOTTING_STATS_ANALYSIS.md
│   ├── PLOTTINGX_ANALYSIS.md
│   ├── NORMALIZATION_SCORING_GUIDE.md
│   ├── NORMALIZATION_EVAL_ANALYSIS.md
│   ├── NORMALIZATION_ANALYSIS.md
│   └── ML_ANALYSIS.md
│
├── xpectrass/                    # Main package
│   ├── __init__.py               # Package initialization, version, public API
│   ├── main.py                   # FTIRdataprocessing & FTIRdataanalysis
│   ├── py.typed                  # PEP 561 type marker (inside package)
│   ├── data/                     # Bundled datasets
│   │   ├── __init__.py           # Dataset loaders (load_jung_2018, etc.)
│   │   ├── .gitignore            # Ignore large data; allow loaders
│   │   └── *.csv.xz              # Compressed datasets (when present)
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── atmospheric.py
│       ├── baseline.py
│       ├── data_validation.py
│       ├── denoise.py
│       ├── derivatives.py
│       ├── file_management.py
│       ├── interpolate.py
│       ├── ml.py
│       ├── normalization.py
│       ├── normalization_eval.py
│       ├── plotting.py
│       ├── plotting_clus.py
│       ├── plotting_dim.py
│       ├── plotting_stats.py
│       ├── plottingx.py
│       ├── region_selection.py
│       ├── scatter_correction.py
│       ├── spectral_utils.py
│       ├── trans_abs.py
│       └── warnings.py
│
├── .gitignore                    # Git ignore rules
├── CHANGELOG.md                  # Version history (Keep a Changelog)
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Package data rules
├── Makefile                      # Build commands
├── PROJECT_STRUCTURE.md         # This file
├── README.md                     # Project overview
├── pyproject.toml               # Modern Python packaging
├── readthedocs.yaml             # ReadTheDocs build configuration
└── setup.py                     # Legacy packaging support
```

## File Purposes

### Configuration Files

**pyproject.toml**
- Modern Python package configuration (PEP 518)
- Dependencies, metadata, build system
- Tool configurations (black, isort, mypy, pytest)
- Package data: `xpectrass` → `py.typed`; `xpectrass.data` → `*.csv.xz`

**setup.py**
- Legacy packaging support
- Fallback for older pip versions
- Same metadata as pyproject.toml

**readthedocs.yaml**
- ReadTheDocs build configuration
- Sphinx settings, Python version, dependencies

**MANIFEST.in**
- Specifies which non-Python files to include in distributions
- Includes LICENSE, README, pyproject.toml
- Recursive include: `xpectrass` (*.py, *.pyi, py.typed, *.csv.xz), docs (*.md, *.rst, *.py, *.txt)
- Prunes: docs/_build, __pycache__, *.egg-info, tests

### Documentation Files

**README.md**
- Project landing page
- Quick start guide
- Features overview
- Installation instructions
- Version badge and history

**CHANGELOG.md**
- Complete version history
- Breaking changes, new features, bug fixes
- Follows Keep a Changelog format

**CONTRIBUTING.md**
- Contribution guidelines
- Development setup
- Code style requirements
- Pull request process

**LICENSE**
- MIT License
- Copyright 2026 @KaziLab.se

### GitHub Integration

**.github/workflows/tests.yml**
- Automated testing on push/PR
- Multi-platform (Windows, macOS, Linux)
- Multi-version (Python 3.8–3.12)
- Lint (flake8), format check (black), pytest with coverage
- Codecov upload (optional)

**.github/workflows/publish.yml**
- Automated PyPI publishing on release
- Runs on GitHub release creation

**.github/ISSUE_TEMPLATE/**
- Standardized bug reports
- Feature request templates

**.github/PULL_REQUEST_TEMPLATE.md**
- PR description template
- Checklist for contributors

## Package Structure

### Main Module (`xpectrass/`)

**__init__.py**
- Package version (`__version__`), author, license
- Exports main classes and functions
- Provides clean public API

**main.py**
- `FTIRdataprocessing`: Complete preprocessing pipeline
- `FTIRdataanalysis`: Statistical analysis and ML

**py.typed**
- Empty PEP 561 marker file
- Must live inside the package directory (not project root) for type checkers

**data/**
- 6 bundled FTIR datasets (compressed *.csv.xz when included)
- Dataset loading functions: `load_jung_2018`, `load_kedzierski_2019`, `load_kedzierski_2019_u`, `load_frond_2021`, `load_villegas_camacho_2024_c4`, `load_villegas_camacho_2024_c8`, `load_all_datasets`, `load_datasets`, `get_data_info`
- `.gitignore` excludes large data; loaders and docs retained

**utils/**
- Modular preprocessing utilities
- Plotting and visualization
- Machine learning utilities
- Data validation, region selection, scatter correction

### Tests (`tests/`)

**Location**: Root-level `tests/` directory (excluded from PyPI distribution via MANIFEST.in prune)

**Test files:**
- `test_denoise_composite.py` – Composite denoising tests
- `test_ml.py` – Machine learning functionality tests
- `test_normalization.py` – Normalization methods tests
- `test_normalization_eval.py` – Normalization evaluation tests
- `test_plotting_clus.py` – Clustering visualization tests
- `test_plotting_dim.py` – Dimensionality reduction plot tests
- `test_plotting_stats.py` – Statistical plotting tests
- `test_plottingx.py` – Extended plotting tests
- `example_safe_evaluation.py` – Safe evaluation examples

**Test documentation (*.md):**
- PLOTTING_CLUS_ANALYSIS.md, PLOTTING_DIM_ANALYSIS.md, PLOTTING_STATS_ANALYSIS.md, PLOTTINGX_ANALYSIS.md
- NORMALIZATION_SCORING_GUIDE.md, NORMALIZATION_EVAL_ANALYSIS.md, NORMALIZATION_ANALYSIS.md
- ML_ANALYSIS.md

### Documentation (`docs/`)

**Sphinx-based documentation:**
- User guides for all features
- API reference with autodoc
- Examples and tutorials
- Installation and getting started
- `conf.py`: version/release set to package version

**Built and hosted on ReadTheDocs:**
- https://xpectrass.readthedocs.io/

## Dependencies

### Core
- numpy, scipy, pandas, polars
- pybaselines, PyWavelets

### Visualization
- matplotlib, seaborn, plotly

### Machine Learning
- scikit-learn, xgboost, lightgbm
- umap-learn, shap  
*(CatBoost removed as of v0.0.3)*

### Development
- pytest, pytest-cov, black, isort, flake8, mypy

## Build and Distribution

### Building for PyPI
```bash
python -m build
twine check dist/*
twine upload dist/*
```

### Building Documentation
```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

### Running Tests
```bash
pytest
pytest --cov=xpectrass --cov-report=html
```

## Recommended Next Steps

### High Priority
1. ✅ **Move tests to root level** – COMPLETED
2. **Add test data** (Optional): Create `tests/data/` with small test datasets; currently tests may use bundled datasets from `xpectrass/data/`

### Medium Priority
3. **Add GitHub badges to README** – Build status, coverage, PyPI version, ReadTheDocs
4. **Create release checklist** – Version update procedure, testing, documentation

### Low Priority
5. Add CODE_OF_CONDUCT.md
6. Add SECURITY.md for vulnerability reporting
7. Create issue labels in GitHub
8. Set up Codecov for coverage tracking

## Publication Checklist

Before publishing to PyPI:
- [x] CHANGELOG.md created
- [x] CONTRIBUTING.md created
- [x] GitHub Actions CI/CD configured
- [x] Documentation complete and builds successfully
- [x] All tests pass
- [x] Version numbers updated (e.g. 0.0.4)
- [x] Tests at root level
- [x] py.typed in `xpectrass/` (PEP 561)
- [ ] Final code review
- [ ] Tag release on GitHub
- [ ] Verify ReadTheDocs builds
- [ ] PyPI upload (manual or via GitHub Actions)

## Support and Contact

- **Email**: xpectrass@kazilab.se
- **GitHub**: https://github.com/kazilab/xpectrass
- **Documentation**: https://xpectrass.readthedocs.io/
- **Issues**: https://github.com/kazilab/xpectrass/issues
