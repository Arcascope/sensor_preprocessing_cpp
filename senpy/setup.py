from setuptools import setup, Extension, find_packages
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

# Add finufft configuration
finufft_include_dir = "/tmp/finufft/include"
finufft_lib_dir = "/tmp/finufft/build/src"
finufft_lib_shared = "/tmp/finufft/build/src/libfinufft.so"
fftw_include_dir_src = "/tmp/finufft/build/_deps/fftw3-src/api"  # fftw3.h in source
fftw_include_dir_build = "/tmp/finufft/build/_deps/fftw3-build"  # config.h in build

# Check if finufft is available
use_finufft = (os.path.exists(finufft_include_dir) and
               os.path.exists(finufft_lib_shared))

if use_finufft:
    print(f"Finufft found at {finufft_lib_shared}")
    include_dirs.append(finufft_include_dir)
    include_dirs.append(fftw_include_dir_src)
    include_dirs.append(fftw_include_dir_build)
else:
    print("Finufft not found, building without NUFFT support")

ext_modules = [
    Extension(
        'senpy._core',  # Full module path - creates senpy/_core.so
        sources=sources,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=(
            ['-std=c++17', '-O3', '-march=native', '-DPYTHON', '-DUSE_FINUFFT']
            if use_finufft
            else ['-std=c++17', '-O3', '-march=native', '-DPYTHON']
        ),
        extra_link_args=(
            [finufft_lib_shared, '-Wl,-rpath,' + finufft_lib_dir]
            if use_finufft
            else []
        ),
    ),
]

setup(
    name='senpy',
    version='0.2.0',
    author='Eric Canton and Franco Tavella',
    description='Fast sensor processing with FFT-based signal analysis',
    long_description='',
    ext_modules=ext_modules,
    packages=['senpy'],  # Define senpy as a package
    package_dir={'senpy': '.'},  # The senpy package is in the current directory
    py_modules=['senpy', 'senpy.api', "senpy._core"],
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.11',
)