#!/usr/bin/env python
from __future__ import print_function

try:
    from Cython.Build import build_ext
    import numpy as np
    np_include_dirs = np.get_include()
except ImportError:
    from setuptools.command.build_ext import build_ext
    np_include_dirs = ''

from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages

import pyartm


included_packages = ('pyartm*',)

with open('requirements.txt', 'r') as f:
    requirements = [x.strip() for x in f if x.strip()]

setup(
    name='python-artm',
    version=pyartm.__version__,
    author='Irkhin Ilya',
    author_email='ilirhin@gmail.com.com',
    description='Python implementation (with optional cpp extensions)'
                ' of ARTM algorithm',
    long_description='',
    setup_requires=['cython', 'numpy'],
    ext_modules=[Extension(
        "pyartm.calculations.inner_product.cy_impl",
        sources=["pyartm/calculations/inner_product/cy_impl.pyx"],
        include_dirs=[np_include_dirs],
    )],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    install_requires=['cython>=0.29'] + requirements,
    packages=find_packages(include=included_packages)
)
