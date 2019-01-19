#!/usr/bin/env python
import os
from setuptools import find_packages, setup
import pyartm_experiments


included_packages = ('pyartm_experiments*',)
setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, 'requirements.txt'), 'r') as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

setup(
    name='pyartm-experiments',
    version=pyartm_experiments.__version__,
    author='Irkhin Ilya',
    author_email='ilirhin@gmail.com.com',
    description='Experiments of pyartm',
    long_description='',
    zip_safe=False,
    install_requires=requirements,
    packages=find_packages(include=included_packages)
)
