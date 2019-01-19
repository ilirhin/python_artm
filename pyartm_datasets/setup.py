#!/usr/bin/env python
import platform
import os
from setuptools import find_packages, setup
import pyartm_datasets


included_packages = ('pyartm_datasets*',)

is_windows = platform.system() == 'Windows'
setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, 'requirements.txt'), 'r') as req_file:
    requirements = [
        line.strip()
        for line in req_file
        if line.strip() and (is_windows or 'mysqlclient' not in line)
    ]

setup(
    name='pyartm-datasets',
    version=pyartm_datasets.__version__,
    author='Irkhin Ilya',
    author_email='ilirhin@gmail.com.com',
    description='Datasets for ARTM',
    long_description='',
    zip_safe=False,
    install_requires=requirements,
    packages=find_packages(include=included_packages)
)
