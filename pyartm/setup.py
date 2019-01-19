#!/usr/bin/env python

from setuptools import find_packages, setup

import pyartm


included_packages = ('pyartm*',)

with open('requirements.txt', 'r') as req_file:
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
