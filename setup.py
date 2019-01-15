#!/usr/bin/env python
from __future__ import print_function

try:
    from Cython.Build import build_ext
    import numpy as np
    np_include_dirs = np.get_include()
except ImportError:
    from setuptools.command.build_ext import build_ext
    np_include_dirs = ''

import platform

from distutils.core import setup
from setuptools import Extension, find_packages

import pyartm


included_packages = ('pyartm*',)

is_windows = platform.system() == 'Windows'
with open('requirements.txt', 'r') as req_file:
    requirements = [
        line.strip()
        for line in req_file
        if line.strip() and (is_windows or 'mysqlclient' not in line)
    ]

setup(
    name='python-artm',
    version=pyartm.__version__,
    author='Irkhin Ilya',
    author_email='ilirhin@gmail.com.com',
    description='Python implementation (with optional cpp extensions)'
                ' of ARTM algorithm',
    long_description='',
    setup_requires=requirements[:2],
    ext_modules=[Extension(
        "pyartm.calculations.inner_product.cy_impl",
        sources=["pyartm/calculations/inner_product/cy_impl.pyx"],
        include_dirs=[np_include_dirs],
    )],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    install_requires=requirements,
    packages=find_packages(include=included_packages)
)
