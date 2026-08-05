# Sensor Preprocessing C++ Library

This library provides a set of C++ classes and functions for preprocessing sensor data. We split it out as a separate code base to enable reuse across multiple projects. Include this as a git submodule in your project to take advantage of its functionality.

## Dart API
The library exposes a Dart API through FFI (Foreign Function Interface). This allows Dart applications to call the C++ functions for sensor data preprocessing seamlessly.

## Python API
In addition to the Dart API, the library also provides a Python package called `senpy` that provides acces to the C++ routines via Pybind11.

### Precompiled Python wheels

GitHub release wheels are built by `.github/workflows/release-wheel.yml` and attached to a release when it is published. The same workflow can be run manually to backfill an existing tag/release, such as `v1.0.0` or `v2.0.0`.

Use the release asset URL directly when installing a precompiled wheel:

```bash
python -m pip install https://github.com/<owner>/<repo>/releases/download/v2.0.0/senpy-2.0.0-cp311-cp311-linux_x86_64.whl
```

`pip install git+https://github.com/<owner>/<repo>.git@v2.0.0` installs from source and will still compile the native extension locally.

### JAX / NVIDIA CUDA NUFFT

The regular `senpy` API remains NumPy/C++ based. For a JAX-native NUFFT that
keeps sample arrays on the active JAX device, install `jax-finufft` with the
JAX build appropriate for the machine:

```bash
# First install JAX with its CUDA support using JAX's installation guidance.
# Then, from this repository's senpy/ directory:
python -m pip install '.[jax]'
```

```python
import jax
import jax.numpy as jnp
from senpy import jax as senpy_jax

timestamps = jnp.arange(3_000, dtype=jnp.float32) / 50.0
signal = jnp.sin(2 * jnp.pi * 3.0 * timestamps)
result = senpy_jax.compute_nustft(
    timestamps, signal, window_s=8.0, overlap_s=4.0, target_fs=16.0
)
print(jax.devices(), result.coefficients.shape)
```

`senpy.jax` uses `jax-finufft`'s type-1 transform; a CUDA-enabled
`jax-finufft` build dispatches it to cuFINUFFT. It returns JAX arrays rather
than `senpy.api.NUSTFTResult`, so subsequent JAX work stays device-resident.
The GPU default is `eps=1e-6`; enable JAX x64 before importing JAX if the
application requires float64 precision. For absolute microsecond timestamps,
either enable x64 before creating the JAX arrays or pass timestamps relative to
the first sample.
