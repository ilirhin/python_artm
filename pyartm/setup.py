#!/usr/bin/env python
import os
from setuptools import find_packages, setup
import pyartm


included_packages = ('pyartm*',)
setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, 'requirements.txt'), 'r') as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

setup(
    name='pyartm',
    version=pyartm.__version__,
    author='Irkhin Ilya',
    author_email='ilirhin@gmail.com.com',
    description='Python implementation of ARTM algorithm',
    long_description='',
    zip_safe=False,
    install_requires=requirements,
    packages=find_packages(include=included_packages)
)
