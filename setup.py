"""
Q1Fold Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="q1fold",
    version="1.0.0",
    author="Zeyu Li",
    license="GPL-3.0-only",
    author_email="zeyu.li@latrobe.edu.au",
    description="A hybrid quantum-classical CNN for RNA secondary structure prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Robert-li1989/q1fold",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "pennylane>=0.30.0",
        "tqdm>=4.60.0",
        "grakel>=0.1.8",
    ],
    entry_points={
        "console_scripts": [
            "q1fold=q1fold:main",
        ],
    },
)
