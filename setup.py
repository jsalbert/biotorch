#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='BioTorch',
    version='0.0.1',
    description='',
    author='Albert Jimenez',
    author_email='',
    url='',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[],
    install_requires=[
        'keras-model-specs',
        'Keras==2.2.4',
        'pandas',
        'matplotlib',
        'jupyter',
        'treelib==1.5.5',
        'jsonschema',
        'scipy',
        'sklearn',
        'plotly',
        'pyyaml',
        'freezegun==0.2.2',
        'GPUtil',
        'h5py==2.10.0',
        'gcloud',
        'httplib2==0.15.0'
    ]
)
