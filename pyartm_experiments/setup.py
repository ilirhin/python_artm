#!/usr/bin/env python
from setuptools import find_packages, setup

import pyartm_experiments

included_packages = ('pyartm_experiments*',)

with open('requirements.txt', 'r') as req_file:
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
