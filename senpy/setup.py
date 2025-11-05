from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import glob

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path (imported lazily)"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# resolve directories relative to this file
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CPP_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'src'))

# explicit binding file in this directory
PROCESSING_SRC = os.path.join(CPP_DIR, 'sensor_processing_native.cpp')

# collect cpp sources from ../cpp (or list specific files)
cpp_files = glob.glob(os.path.join(CPP_DIR, '*.cpp'))
# If you only want specific files, replace above with e.g.:
# cpp_files = [os.path.join(CPP_DIR, 'sensor_processing.cpp')]

# ensure we compile only the processing TU which now contains the pybind11 module
sources = [PROCESSING_SRC]
print(f"Using processing source: {PROCESSING_SRC}")

# include dirs: pybind11 and your cpp headers
include_dirs = [get_pybind_include(), os.path.join(CPP_DIR)]
try:
    import numpy
    include_dirs.append(numpy.get_include())
except Exception:
    # numpy may be installed in the isolated build env; it's okay if not available here
    pass

ext_modules = [
    Extension(
        'senpy',
        sources=sources,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-march=native', '-DPYTHON'],
    ),
]

setup(
    name='senpy',
    version='0.2.0',
    author='Eric Canton and Franco Tavella',
    description='Fast sensor processing with FFT-based signal analysis',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.11',
)