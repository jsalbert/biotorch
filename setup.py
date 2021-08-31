#!/usr/bin/env python
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
ROOT = pathlib.Path(__file__).parent

# The text of the README file
README = (ROOT/"README.md").read_text()

setup(
    name='biotorch',
    version='0.0.9',
    description='BioTorch is a PyTorch framework specializing in biologically plausible learning algorithms.',
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    author='Albert Jimenez',
    author_email='albertjimenez@aip.ai',
    url='https://github.com/jsalbert/biotorch',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        "jsonschema>=3.0",
        "torch>=1.0",
        "torchvision",
        "numpy",
        "matplotlib",
        "pandas",
        "ipykernel",
        "jupyter",
        "Pillow",
        "pyyaml",
        "scipy",
        "tensorboard",
        "torchattacks",
        "tqdm"
    ]
)
