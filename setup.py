"""
Setup script for Sequential VSA-OGM.
"""

from setuptools import setup, find_packages

setup(
    name="sequential_vsa_ogm",
    version="0.1.0",
    description="Vector Symbolic Architecture for Occupancy Grid Mapping",
    author="VSA-OGM Team",
    author_email="example@example.com",
    url="https://github.com/example/sequential-vsa-ogm",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
