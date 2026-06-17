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
