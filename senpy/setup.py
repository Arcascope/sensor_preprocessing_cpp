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

# Add finufft configuration - check multiple locations
finufft_include_dirs = [
    "/usr/local/include",      # Standard install location
    "/usr/include",            # System-wide location
    "/opt/finufft/include",    # Docker build location
    "/tmp/finufft/include",    # Development location
]
finufft_lib_dirs = [
    "/usr/local/lib",          # Standard install location
    "/usr/lib",                # System-wide location
    "/opt/finufft/build/src",  # Docker build location
    "/tmp/finufft/build/src",  # Development location
]

# Find finufft headers and library
use_finufft = False
finufft_include_dir = None
finufft_lib_dir = None
finufft_lib_shared = None

# First, find the include directory
for inc_dir in finufft_include_dirs:
    header_path = os.path.join(inc_dir, "finufft.h")
    if os.path.exists(header_path):
        finufft_include_dir = inc_dir
        print(f"Found finufft.h at: {header_path}")
        break

# Then find the library in the same locations
# Check for shared library first (.so), then static (.a)
if finufft_include_dir:
    for lib_dir in finufft_lib_dirs:
        # Try shared library variants first
        for lib_name in ["libfinufft.so", "libfinufft.so.0"]:
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                finufft_lib_dir = lib_dir
                finufft_lib_shared = lib_path
                use_finufft = True
                print(f"Found {lib_name} at: {lib_path}")
                break
        if use_finufft:
            break

# If no shared library, try static library
if finufft_include_dir and not use_finufft:
    for lib_dir in finufft_lib_dirs:
        lib_path = os.path.join(lib_dir, "libfinufft.a")
        if os.path.exists(lib_path):
            finufft_lib_dir = lib_dir
            finufft_lib_shared = lib_path
            use_finufft = True
            print(f"Found libfinufft.a (static) at: {lib_path}")
            break

if not use_finufft:
    if finufft_include_dir:
        raise RuntimeError(f"Finufft header found but library not found in: {finufft_lib_dirs}")
    else:
        raise RuntimeError(f"Finufft is REQUIRED but not found. Searched in: {finufft_include_dirs}\n"
                          f"Please install finufft to /usr/local or set FINUFFT_PATH environment variable")

print(f"✓ Finufft found: include={finufft_include_dir}, lib={finufft_lib_shared}")
include_dirs.append(finufft_include_dir)

print(f"\n{'='*80}")
print(f"SETUP.PY DEBUG INFO")
print(f"{'='*80}")
print(f"finufft_lib_dir: {finufft_lib_dir}")
print(f"finufft_include_dir: {finufft_include_dir}")
print(f"finufft_lib_shared: {finufft_lib_shared}")
print(f"{'='*80}\n")

# When linking a static library into a shared library, we need --whole-archive
# to ensure all symbols are included (not just the ones we reference directly)
link_args = []
if finufft_lib_dir:
    # Use --whole-archive for finufft static lib to pull in all symbols
    if finufft_lib_shared.endswith('.a'):
        link_args = ['-Wl,--whole-archive', finufft_lib_shared, '-Wl,--no-whole-archive']
    link_args.append('-Wl,-rpath,' + finufft_lib_dir)

print(f"Link args (whole-archive aware): {link_args}")

ext_modules = [
    Extension(
        'senpy._core',  # Full module path - creates senpy/_core.so
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=[finufft_lib_dir] if finufft_lib_dir else [],
        libraries=['fftw3f_omp', 'fftw3_omp', 'fftw3f', 'fftw3', 'gomp'],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-march=native', '-DPYTHON', '-DUSE_FINUFFT', '-fopenmp'],
        extra_link_args=link_args,
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