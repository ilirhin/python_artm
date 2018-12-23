#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import pyartm


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            print('\n\n\nERROR:\n'
                  'CMake must be installed to build the following extensions:\n\t' +
                  '\n\t'.join(e.name for e in self.extensions))
            return

        for ext in self.extensions:
            try:
                self.build_extension(ext)
            except subprocess.CalledProcessError:
                print('\n\n\nERROR: Failed to build C++ extension. '
                      'Use pure python\n\n\n')

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(extdir),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}'.format(self.build_temp),
            '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
        ]
        if platform.system() == 'Windows':
            plat = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
            cmake_args += [
                # These options are likely to be needed under Windows
                '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}'.format(extdir),
            ]
            # Assuming that Visual Studio and MinGW are supported compilers
            if self.compiler.compiler_type == 'msvc':
                cmake_args += [
                    '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                ]
            else:
                cmake_args += [
                    '-G', 'MinGW Makefiles',
                ]
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--config', cfg],
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
