from setuptools import setup

setup(
    name="chipseq_utils",
    version="0.1.0-dev0",
    description="Utilities for ChIP-seq analysis",
    author="Jakub Kaczmarzyk",
    py_modules=["chipseq_utils"],
    python_requires=">=3.6,<4",
    install_requires=["h5py", "matplotlib", "numpy", "pandas", "scipy"],
    extras_require={"dev": ["flake8", "mypy"]},
)
