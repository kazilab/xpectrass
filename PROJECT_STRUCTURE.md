# Xpectrass Project Structure

## Overview
This document describes the complete project structure for xpectrass v0.0.2, ready for PyPI, GitHub, and ReadTheDocs.

## Directory Structure

```
xpectrass_app/
├── .github/                      # GitHub-specific files
│   ├── workflows/
│   │   ├── tests.yml            # CI/CD testing workflow
│   │   └── publish.yml          # PyPI publishing workflow
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md        # Bug report template
│   │   └── feature_request.md   # Feature request template
│   └── PULL_REQUEST_TEMPLATE.md # PR template
│
├── docs/                         # Sphinx documentation
│   ├── _static/                 # Static assets
│   ├── _templates/              # Custom templates
│   ├── api/                     # API reference
│   │   ├── index.md
│   │   ├── preprocessing_pipeline.md
│   │   └── utils.md
│   ├── user_guide/              # User guides
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
│   ├── changelog.md             # Version history for docs
│   ├── conf.py                  # Sphinx configuration
│   ├── examples.md              # Usage examples
│   ├── getting_started.md       # Getting started guide
│   ├── index.md                 # Documentation homepage
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
│   └── *.md                     # Test documentation
│
├── xpectrass/                    # Main package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FTIRdataprocessing & FTIRdataanalysis
│   ├── data/                    # Bundled datasets
│   │   ├── __init__.py
│   │   └── *.csv.xz            # Compressed datasets
│   └── utils/                   # Utility modules
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
├── .readthedocs.yaml            # ReadTheDocs configuration
├── CHANGELOG.md                  # Version history (NEW)
├── CONTRIBUTING.md               # Contribution guidelines (NEW)
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Package data rules
├── Makefile                      # Build commands
├── PROJECT_STRUCTURE.md          # This file (NEW)
├── README.md                     # Project overview
├── py.typed                      # PEP 561 type marker
├── pyproject.toml               # Modern Python packaging
└── setup.py                     # Legacy packaging support
```

## File Purposes

### Configuration Files

**pyproject.toml**
- Modern Python package configuration (PEP 518)
- Dependencies, metadata, build system
- Tool configurations (black, isort, mypy, pytest)

**setup.py**
- Legacy packaging support
- Fallback for older pip versions
- Same metadata as pyproject.toml

**.readthedocs.yaml**
- ReadTheDocs build configuration
- Sphinx settings, Python version, dependencies

**MANIFEST.in**
- Specifies which non-Python files to include in distributions
- Includes docs, LICENSE, README

### Documentation Files

**README.md**
- Project landing page
- Quick start guide
- Features overview
- Installation instructions

**CHANGELOG.md** (NEW)
- Complete version history
- Breaking changes, new features, bug fixes
- Follows Keep a Changelog format

**CONTRIBUTING.md** (NEW)
- Contribution guidelines
- Development setup
- Code style requirements
- Pull request process

**LICENSE**
- MIT License
- Copyright 2026 @KaziLab.se

### GitHub Integration

**.github/workflows/tests.yml** (NEW)
- Automated testing on push/PR
- Multi-platform (Windows, macOS, Linux)
- Multi-version (Python 3.8-3.12)
- Code coverage reporting

**.github/workflows/publish.yml** (NEW)
- Automated PyPI publishing on release
- Runs on GitHub release creation

**.github/ISSUE_TEMPLATE/** (NEW)
- Standardized bug reports
- Feature request templates

**.github/PULL_REQUEST_TEMPLATE.md** (NEW)
- PR description template
- Checklist for contributors

## Package Structure

### Main Module (`xpectrass/`)

**__init__.py**
- Package version, author, license
- Exports main classes and functions
- Provides clean public API

**main.py**
- `FTIRdataprocessing`: Complete preprocessing pipeline
- `FTIRdataanalysis`: Statistical analysis and ML

**data/**
- 6 bundled FTIR datasets (compressed)
- Dataset loading functions
- Metadata and descriptions

**utils/**
- Modular preprocessing utilities
- Plotting and visualization
- Machine learning utilities
- Data validation

### Tests (`tests/`)

**Location**: Root-level `tests/` directory (properly excluded from PyPI distribution)

**Test files:**
- `test_denoise_composite.py` - Composite denoising tests
- `test_ml.py` - Machine learning functionality tests
- `test_normalization.py` - Normalization methods tests
- `test_normalization_eval.py` - Normalization evaluation tests
- `test_plotting_clus.py` - Clustering visualization tests
- `test_plotting_dim.py` - Dimensionality reduction plot tests
- `test_plotting_stats.py` - Statistical plotting tests
- `test_plottingx.py` - Extended plotting tests
- `example_safe_evaluation.py` - Safe evaluation examples
- Analysis documentation files (*.md) - Test methodology documentation

### Documentation (`docs/`)

**Sphinx-based documentation:**
- User guides for all features
- API reference with autodoc
- Examples and tutorials
- Installation and getting started

**Built and hosted on ReadTheDocs:**
- https://xpectrass.readthedocs.io/

## Dependencies

### Core
- numpy, scipy, pandas, polars
- pybaselines, PyWavelets

### Visualization
- matplotlib, seaborn

### Machine Learning
- scikit-learn, xgboost, lightgbm, catboost
- umap-learn, shap

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
1. ✅ **Move tests to root level** - COMPLETED
   - Tests successfully moved from `xpectrass/tests/` to `tests/`
   - Tests are now properly excluded from PyPI distribution
   - Import paths work correctly with pytest

2. **Add test data** (Optional)
   - Create `tests/data/` with small test datasets
   - Ensures tests can run independently
   - Currently tests use bundled datasets from `xpectrass/data/`

### Medium Priority
3. **Add GitHub badges to README**
   - Build status badge
   - Coverage badge
   - PyPI version badge
   - ReadTheDocs status

4. **Create release checklist**
   - Version update procedure
   - Testing requirements
   - Documentation updates

### Low Priority
5. **Add CODE_OF_CONDUCT.md**
6. **Add SECURITY.md** for vulnerability reporting
7. **Create issue labels** in GitHub
8. **Set up Codecov** for coverage tracking

## Publication Checklist

Before publishing to PyPI:
- [x] CHANGELOG.md created
- [x] CONTRIBUTING.md created
- [x] GitHub Actions CI/CD configured
- [x] Documentation complete and builds successfully
- [x] All tests pass
- [x] Version numbers updated
- [x] Tests moved to root level
- [ ] Final code review
- [ ] Tag release on GitHub
- [ ] Verify ReadTheDocs builds
- [ ] PyPI upload (manual or via GitHub Actions)

## Support and Contact

- **Email**: xpectrass@kazilab.se
- **GitHub**: https://github.com/kazilab/xpectrass
- **Documentation**: https://xpectrass.readthedocs.io/
- **Issues**: https://github.com/kazilab/xpectrass/issues
