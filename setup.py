#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='BioTorch',
    version='0.0.1',
    description='BioTorch is a PyTorch framework specialized in biologically plausible learning algorithms.',
    author='Albert Jimenez',
    author_email='albertjimenez@aip.ai',
    url='https://github.com/jsalbert/biotorch',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[],
    install_requires=[
        "setuptools"
        "torch>=1.0",
        "torchvision",
        "flake8",
        "autoflake",
        "pep8",
        "autopep8",
        "numpy",
        "matplotlib",
        "pandas",
        "ipykernel",
        "jupyter",
        "jsonschema",
        "Pillow",
        "pytest",
        "pytest-cov",
        "PyYAML",
        "scipy",
        "tensorboard",
        "torchattacks",
        "tqdm"
    ]
)
