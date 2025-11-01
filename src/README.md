# Sensor Preprocessing C++ Library

This library provides a set of C++ classes and functions for preprocessing sensor data. We split it out as a separate code base to enable reuse across multiple projects. Include this as a git submodule in your project to take advantage of its functionality.

## Dart API
The library exposes a Dart API through FFI (Foreign Function Interface). This allows Dart applications to call the C++ functions for sensor data preprocessing seamlessly.

## Python API
In addition to the Dart API, the library also provides a Python package called `senpy` that provides acces to the C++ routines via Pybind11.