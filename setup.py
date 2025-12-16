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
version = "0.0.1"
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
    author="FTIR-PLASTIC Team",
    author_email="xpectrass@kazilab.se",
    description="FTIR/ToF-SIMS Spectral Analysis Suite - Preprocessing toolkit for spectral classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazilab/xpectrass",
    project_urls={
        "Bug Reports": "https://github.com/kazilab/xpectrass/issues",
        "Documentation": "https://xpectrass.readthedocs.io/",
        "Source": "https://github.com/kazilab/xpectrass",
    },
    license="MIT",
    
    # Package discovery - current directory is the xpectrass package
    packages=["xpectrass", "xpectrass.utils"],
    package_dir={"xpectrass": "."},
    
    # Package data (include non-Python files)
    package_data={
        "xpectrass": ["py.typed"],  # PEP 561 marker
    },
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "polars>=0.15.0",
        "pybaselines>=1.0.0",
        "PyWavelets>=1.1.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "joblib>=1.0.0",
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
        "ToF-SIMS",
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
