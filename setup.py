#!/usr/bin/env python
from __future__ import print_function

import os
import re
import sys
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import pyartm


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            print('\n\n\nERROR:\n'
                  'CMake must be installed to build the following extensions:\n\t' +
                  '\n\t'.join(e.name for e in self.extensions))
            return None

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            try:
                self.build_extension(ext)
            except subprocess.CalledProcessError:
                print('\n\n\nERROR: Failed to build C++ extension. '
                      'Use pure python\n\n\n')

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']

            plat = 'x64' if platform.architecture()[0] == '64bit' else 'Win32'
            # Assuming that Visual Studio and MinGW are supported compilers
            if self.compiler.compiler_type == 'msvc':
                cmake_args += [
                    '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                ]
            else:
                cmake_args += [
                    '-G', 'MinGW Makefiles',
                ]

            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


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
    ext_modules=[CMakeExtension('pyartm')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=requirements,
    packages=find_packages(include=included_packages)
)
