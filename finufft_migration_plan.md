# Migration Plan: Self-contained finufft dependency in senpy

## Context
`senpy/setup.py` hardcodes `/tmp/finufft/` paths and silently builds without NUFFT support if those paths don't exist. Getting a working build requires undocumented manual steps (clone finufft, run cmake with specific flags, build). This caused extensive debugging of the Dockerfile. The fix: make `setup.py` auto-fetch and build finufft if it's not present, so `pip install senpy` just works.

**Constraint**: finufft must link dynamically (shared `.so`). Static linking causes undefined symbol errors at import time. Use `-DFINUFFT_STATIC_LINKING=OFF`.

## Critical Files
- [senpy/setup.py](../sensor_preprocessing_cpp/senpy/setup.py) — core rewrite
- [senpy/pyproject.toml](../sensor_preprocessing_cpp/senpy/pyproject.toml) — add cmake to build deps
- [senpy/.gitignore](../sensor_preprocessing_cpp/senpy/.gitignore) — add `_deps/`
- [Dockerfile](Dockerfile) — remove manual finufft build block

---

## Changes

### 1. `senpy/setup.py` — Add auto-fetch logic

Pin a version and cache inside the repo at `_deps/finufft/` (gitignored):

```python
FINUFFT_VERSION = "v2.5.0"
FINUFFT_REPO = "https://github.com/flatironinstitute/finufft.git"

# Cache location: inside the senpy directory (gitignored)
FINUFFT_BUILD_ROOT = os.path.join(ROOT_DIR, "_deps", f"finufft-{FINUFFT_VERSION}")
```

Add a `_ensure_finufft()` helper that:
1. Checks if `_deps/finufft-v2.5.0/build/src/libfinufft.so` already exists → skips build
2. Otherwise: shallow-clones the pinned tag, runs cmake with `-DFINUFFT_STATIC_LINKING=OFF -DBUILD_TESTING=OFF`, runs `make -j<nproc>`
3. Raises `RuntimeError` if the `.so` isn't present after build (no silent fallback)

Add a `BuildExtWithFinufft(build_ext)` subclass whose `run()`:
1. Calls `_ensure_finufft()` to get the build root
2. Patches `self.extensions[0]` with the correct include dirs, `-DUSE_FINUFFT`, and link args
3. Calls `super().run()`

Restructure the `Extension` object to be finufft-free at definition time (just pybind11 + numpy + source); `BuildExtWithFinufft.run()` augments it dynamically.

The `use_finufft` conditional is eliminated entirely — finufft is always fetched and always used. Remove the old module-level path checks.

```python
def _run(cmd, cwd=None):
    import subprocess
    print(f"[senpy] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

def _ensure_finufft():
    so_path = os.path.join(FINUFFT_BUILD_ROOT, "build", "src", "libfinufft.so")
    if os.path.isfile(so_path):
        print(f"[senpy] finufft already built at {so_path}")
        return FINUFFT_BUILD_ROOT

    print(f"[senpy] Fetching finufft {FINUFFT_VERSION} to {FINUFFT_BUILD_ROOT}")
    os.makedirs(FINUFFT_BUILD_ROOT, exist_ok=True)
    _run(["git", "clone", "--branch", FINUFFT_VERSION, "--depth", "1",
          FINUFFT_REPO, FINUFFT_BUILD_ROOT])

    cmake_build = os.path.join(FINUFFT_BUILD_ROOT, "build")
    os.makedirs(cmake_build, exist_ok=True)
    _run(["cmake", "-DCMAKE_BUILD_TYPE=Release",
          "-DFINUFFT_STATIC_LINKING=OFF", "-DBUILD_TESTING=OFF", ".."],
         cwd=cmake_build)

    import multiprocessing
    _run(["make", f"-j{multiprocessing.cpu_count()}"], cwd=cmake_build)

    if not os.path.isfile(so_path):
        raise RuntimeError(f"[senpy] finufft build failed: {so_path} not found")
    return FINUFFT_BUILD_ROOT


class BuildExtWithFinufft(build_ext):
    def run(self):
        root = _ensure_finufft()
        lib_dir = os.path.join(root, "build", "src")
        for ext in self.extensions:
            ext.include_dirs += [
                os.path.join(root, "include"),
                os.path.join(root, "build", "_deps", "fftw3-src", "api"),
                os.path.join(root, "build", "_deps", "fftw3-build"),
            ]
            ext.extra_compile_args += ["-DUSE_FINUFFT"]
            ext.extra_link_args += [
                os.path.join(lib_dir, "libfinufft.so"),
                f"-Wl,-rpath,{lib_dir}",
            ]
        super().run()
```

`setup()` call: change `cmdclass={'build_ext': build_ext}` → `cmdclass={'build_ext': BuildExtWithFinufft}`.

### 2. `senpy/pyproject.toml` — Add cmake

```toml
[build-system]
requires = ["setuptools>=42", "wheel", "pybind11>=2.6", "cmake>=3.18"]
build-backend = "setuptools.build_meta"
```

The `cmake` PyPI package installs the cmake binary into pip's isolated build environment, removing the system cmake requirement.

### 3. `senpy/.gitignore` (create if missing)

```
_deps/
*.egg-info/
build/
__pycache__/
*.so
```

### 4. `pisces2/Dockerfile` — Remove manual finufft block

Remove:
- The entire `RUN mkdir -p /tmp/finufft && ...` block (current lines 36–52)
- `ENV LD_LIBRARY_PATH=/tmp/finufft/build/src:/usr/local/lib` — the rpath is baked into `_core.so` by setup.py; no `LD_LIBRARY_PATH` needed
- `ENV PKG_CONFIG_PATH=...` — no longer needed

Simplify the senpy install step from the multi-line verification + `pip install -v .` to:

```dockerfile
RUN git clone https://github.com/Arcascope/sensor_preprocessing_cpp.git /opt/sensor_preprocessing_cpp
RUN pip install /opt/sensor_preprocessing_cpp/senpy
```

The `apt-get install git cmake build-essential` line stays (git is needed for the clone inside setup.py; cmake is a belt-and-suspenders fallback even though pyproject.toml now provides it).

---

## Developer Workflow After Migration

```bash
# Fresh clone — just works
pip install -e sensor_preprocessing_cpp/senpy
# → clones finufft v2.5.0 to senpy/_deps/finufft-v2.5.0/ (~3 min first time)
# → subsequent installs skip the build entirely

# Force finufft rebuild
rm -rf sensor_preprocessing_cpp/senpy/_deps/
pip install -e sensor_preprocessing_cpp/senpy

# Upgrade finufft: change FINUFFT_VERSION in setup.py, rebuild
```

## Verification

1. Delete `_deps/` and `pip install -e senpy/` — confirm output shows `[senpy] Fetching finufft v2.5.0...` followed by a successful build
2. Run again — confirm output shows `[senpy] finufft already built at ...` (skip)
3. `python -c "import senpy; print('ok')"` — no ImportError
4. Docker: `docker build --no-cache -t pisces2 .` — finufft is fetched inside the `pip install` step, no pre-steps needed
5. Confirm `ldd` on the installed `_core.so` shows `libfinufft.so` resolved via the rpath
