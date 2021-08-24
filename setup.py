#!/usr/bin/env python
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
ROOT = pathlib.Path(__file__).parent

# The text of the README file
README = (ROOT/"README.md").read_text()

setup(
    name='biotorch',
    version='0.0.2',
    description='BioTorch is a PyTorch framework specialized in biologically plausible learning algorithms.',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Albert Jimenez',
    author_email='albertjimenez@aip.ai',
    url='https://github.com/jsalbert/biotorch',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
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
        "pyyaml",
        "scipy",
        "tensorboard",
        "torchattacks",
        "tqdm"
    ]
)
