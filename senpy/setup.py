from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import shutil
import sys
import setuptools
import os
import glob
import platform

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path (imported lazily)"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# resolve directories relative to this file
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CPP_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'src'))

# explicit binding file in this directory — must stay as a plain relative string
# (setuptools rejects absolute paths; os.path.abspath would break in pip's temp build dir)
PROCESSING_SRC = '../src/sensor_processing_native.cpp'

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

# Finufft built by the top-level CMake into build/_deps/
BUILD_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'build'))
finufft_include_dir = os.path.join(BUILD_DIR, "_deps/finufft-src/include")
finufft_lib_dir = os.path.join(BUILD_DIR, "_deps/finufft-build/src")
_lib_ext = "dylib" if platform.system() == "Darwin" else "so"
finufft_lib_so = os.path.join(finufft_lib_dir, f"libfinufft.{_lib_ext}")
fftw3_lib = os.path.join(BUILD_DIR, "_deps/fftw3-build/libfftw3.a")
fftw3f_lib = os.path.join(BUILD_DIR, "_deps/fftw3f-build/libfftw3f.a")

def required_artifacts():
    return [finufft_include_dir, finufft_lib_so, fftw3_lib, fftw3f_lib]

def artifacts_ready():
    return all(os.path.exists(path) for path in required_artifacts())

def ensure_native_artifacts():
    if artifacts_ready():
        print("senpy: reusing existing FINUFFT build artifacts", flush=True)
        return

    repo_root = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    print(f"senpy: configuring native build in {BUILD_DIR}", flush=True)
    subprocess.run(
        [
            "cmake",
            "-S", repo_root,
            "-B", BUILD_DIR,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFINUFFT_STATIC_LINKING=OFF",
        ],
        check=True,
    )
    print("senpy: building native artifacts with CMake", flush=True)
    subprocess.run(
        ["cmake", "--build", BUILD_DIR, "--parallel"],
        check=True,
    )
    print("senpy: native artifact build complete", flush=True)

    missing = [path for path in required_artifacts() if not os.path.exists(path)]
    if missing:
        raise RuntimeError(
            "Required finufft build artifact not found after CMake build: "
            + ", ".join(missing)
        )

class SenpyBuildExt(build_ext):
    def run(self):
        ensure_native_artifacts()
        super().run()
        self._copy_runtime_libraries()

    def _copy_runtime_libraries(self):
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            ext_dir = os.path.dirname(ext_path)
            os.makedirs(ext_dir, exist_ok=True)
            target_lib = os.path.join(ext_dir, os.path.basename(finufft_lib_so))
            shutil.copy2(finufft_lib_so, target_lib)
            print(f"senpy: copied runtime library to {target_lib}", flush=True)

include_dirs.append(finufft_include_dir)

ext_modules = [
    Extension(
        'senpy._core',  # Full module path - creates senpy/_core.so
        sources=sources,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-march=native', '-DPYTHON'],
        extra_link_args=[
            f'-L{finufft_lib_dir}',
            '-lfinufft',
            fftw3_lib, fftw3f_lib,
            '-Wl,-rpath,@loader_path' if platform.system() == 'Darwin' else '-Wl,-rpath,$ORIGIN',
            '-lomp' if platform.system() == 'Darwin' else '-lgomp',
        ],
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
    package_data={'senpy': ['*.so', '*.dylib', '*.dll']},
    include_package_data=True,
    py_modules=['senpy', 'senpy.api', "senpy._core"],
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': SenpyBuildExt},
    zip_safe=False,
    python_requires='>=3.11',
)
