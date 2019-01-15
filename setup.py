#!/usr/bin/env python
from __future__ import print_function

import platform
from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages

try:
    from Cython.Build import build_ext
    import numpy as np

    extra_setup_args = dict(
        ext_modules = [Extension(
            "pyartm.calculations.inner_product.cy_impl",
            sources=["pyartm/calculations/inner_product/cy_impl.pyx"],
            include_dirs=[np.get_include()],
        )],
        cmdclass={'build_ext': build_ext}
    )
except ImportError:
    extra_setup_args = dict()


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
    zip_safe=False,
    install_requires=requirements,
    packages=find_packages(include=included_packages),
    **extra_setup_args
)
