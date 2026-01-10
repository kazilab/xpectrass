#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for xpectrass package.

Installation:
    pip install .                  # Regular install
    pip install -e .               # Editable/development install
    pip install .[dev]             # With development dependencies
    pip install .[docs]            # With documentation dependencies

Building for PyPI:
    python -m build
    twine upload dist/*
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
here = Path(__file__).parent.resolve()
long_description = ""
readme_path = here / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read version from package
version = "0.0.2"
try:
    with open(here / "xpectrass" / "__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
except FileNotFoundError:
    pass

setup(
    name="xpectrass",
    version=version,
    author="Data Analysis Team @KaziLab.se",
    author_email="xpectrass@kazilab.se",
    description="FTIR Spectral Analysis Suite - Preprocessing toolkit for spectral classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazilab/xpectrass",
    project_urls={
        "Bug Reports": "https://github.com/kazilab/xpectrass/issues",
        "Documentation": "https://xpectrass.readthedocs.io/",
        "Source": "https://github.com/kazilab/xpectrass",
    },
    license="MIT",
    
    # Package discovery - automatically find all packages
    packages=find_packages(include=["xpectrass", "xpectrass.*"]),
    
    # Package data (include non-Python files)
    package_data={
        "xpectrass": ["py.typed"],  # PEP 561 marker
        "xpectrass.data": ["*.csv.xz"],  # Include compressed datasets
    },
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        # Core scientific computing
        "numpy==2.3.5",
        "scipy==1.16.3",
        "pandas==2.3.3",
        "polars==1.37.0",

        # Signal processing and preprocessing
        "pybaselines==1.2.1",
        "PyWavelets==1.9.0",

        # Visualization
        "matplotlib==3.10.8",
        "seaborn==0.13.2",
        "plotly==6.5.1",

        # Machine learning - core
        "scikit-learn==1.8.0",

        # Machine learning - boosting
        "xgboost==3.1.3",
        "lightgbm==4.6.0",

        # Dimensionality reduction
        "umap-learn==0.5.9.post2",

        # Model explainability
        "shap==0.50.0",

        # Utilities
        "tqdm==4.67.1",
        "joblib==1.5.3",
        
        # Other
        "cloudpickle==3.1.2",
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "et-xmlfile==2.0.0",
        "fonttools==4.61.1",
        "graphviz==0.21",
        "kiwisolver==1.4.9",
        "llvmlite==0.46.0",
        "narwhals==2.15.0",
        "numba==0.63.1",
        "pillow==12.1.0",
        "polars-runtime-32==1.37.0",
        "pynndescent==0.6.0",
        "pyparsing==3.3.1",
        "pytz==2025.2",
        "openpyxl==3.1.5",
        "slicer==0.0.8",
        "threadpoolctl==3.6.0",
        "typing-extensions==4.15.0",
        "tzdata==2025.3",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
        "all": [
            # Include both dev and docs
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "FTIR",
        "spectroscopy",
        "preprocessing",
        "baseline correction",
        "plastic classification",
        "chemometrics",
        "machine learning",
        "PCA",
        "normalization",
    ],
    
    # Entry points (optional CLI commands)
    entry_points={
        "console_scripts": [
            # "xpectrass=xpectrass.cli:main",  # Uncomment if CLI is added
        ],
    },
    
    # ZIP safety
    zip_safe=False,
)
