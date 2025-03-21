from setuptools import setup, find_packages

setup(
    name="vsa_ogm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=[""],
    install_requires=[
        "matplotlib",
        "numpy",
        "omegaconf",
        "opencv-python",
        "pandas",
        "pyntcloud",
        "pyyaml",
        "scikit-learn",
        "scikit-image",
        "tqdm",
        "torch"
    ],
    entry_points={
        "console_scripts": [
            "vsa-ogm=main:cli_main",
        ],
    },
    python_requires=">=3.8",
    author="Shay Snyder",
    author_email="ssnyde9@gmu.edu",
    description="Vector Symbolic Architecture for Occupancy Grid Mapping",
    keywords="vector symbolic architectures, hyperdimensional computing",
    url="https://github.com/shaymeister/highfrost",
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
