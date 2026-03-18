# finufft Integration Notes

## What We Did Earlier

1. **Built finufft from source** at `/tmp/finufft/`
   - Cloned from GitHub: `https://github.com/flatironinstitute/finufft.git`
   - CMake build: `cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release`
   - Installed to `/tmp/finufft/` (not system-wide)

2. **Located Headers**
   - Main: `/tmp/finufft/include/finufft.h`
   - FFTW (source): `/tmp/finufft/build/_deps/fftw3-src/api/fftw3.h`
   - FFTW (build): `/tmp/finufft/build/_deps/fftw3-build/config.h`

3. **Located Libraries**
   - Shared library: `/tmp/finufft/build/src/libfinufft.so`
   - Used RPATH linking: `-Wl,-rpath,/tmp/finufft/build/src`

4. **Compile Flag**
   - Added `-DUSE_FINUFFT` to enable finufft code paths in C++

5. **setup.py Changes**
   - Check for finufft at hardcoded paths
   - Conditionally add include dirs and link flags
   - Set `-DUSE_FINUFFT` only when finufft found

6. **C++ Code Structure**
   - `computeSpectrogramNUFFT()` function at line ~302
   - Uses finufft 1D type-1 NUFFT (non-uniform to uniform)
   - Implements sliding window + cubic spline resampling

## For Docker Build

The challenge: finufft must be available during the senpy build phase.

Solution: Build finufft in Docker first, then link against it.

### Docker Build Strategy

1. **Install build dependencies** in Dockerfile (cmake, git, build-essential)
2. **Clone and build finufft** at `/tmp/finufft` (same as local machine)
   - `cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release`
   - Results in: `/tmp/finufft/include/finufft.h` and `/tmp/finufft/build/src/libfinufft.so`
3. **Run ldconfig** to register the library path
4. **setup.py automatically detects** finufft at `/tmp/finufft/include` and `/tmp/finufft/build/src`
5. **Compile flag** `-DUSE_FINUFFT` enables finufft code paths

### C++ Code Guards

Added `#ifdef USE_FINUFFT` / `#else` / `#endif` wrapper around:
- finufft header includes (line 11)
- finufft function calls in `computeSpectrogramNUFFT` (lines 397-447)
- pybind wrapper function (line 1778)
- pybind module registration (line 2080)

This allows the C++ to compile even when finufft isn't available (with empty fallback).
