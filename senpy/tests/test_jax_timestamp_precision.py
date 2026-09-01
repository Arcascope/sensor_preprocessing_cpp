"""Tests for timestamp precision in the optional JAX NUFFT backend.

JAX materializes float64 host arrays as float32 unless ``jax_enable_x64`` is
set, and float32 cannot resolve millisecond sample spacing at epoch magnitude.
These tests stand NumPy in for ``jax.numpy`` with that same downcast so the
hazard is reproducible without the optional GPU packages installed.
"""

import numpy as np
import pytest

from senpy import jax_backend as senpy_jax


class _Float32Numpy:
    """NumPy behaving like ``jax.numpy`` without x64: ``asarray`` downcasts."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asarray(values, dtype=None):
        array = np.asarray(values, dtype=dtype)
        if dtype is None and array.dtype == np.float64:
            array = array.astype(np.float32)
        return array


class _HostJax:
    """A JAX stand-in whose ``Array`` type matches nothing the tests pass in."""

    class Array:
        pass

    @staticmethod
    def jit(function):
        return function

    @staticmethod
    def device_get(value):
        return value


class _DeviceJax(_HostJax):
    """A JAX stand-in that treats plain NumPy arrays as device arrays."""

    Array = np.ndarray


def _fake_nufft1(nfft, strengths, points, *, eps, iflag):
    """A dense type-1 transform: exact, and sensitive to every coordinate."""
    del eps
    modes = np.arange(nfft) - nfft // 2
    return np.exp(1j * iflag * np.outer(modes, points)) @ strengths


def _install(monkeypatch, jax_stub):
    monkeypatch.setattr(
        senpy_jax, "_dependencies", lambda: (jax_stub, _Float32Numpy(), _fake_nufft1)
    )


# 50 Hz for 20 s, expressed as offsets in milliseconds.
_OFFSETS_MS = np.arange(1000, dtype=np.float64) * 20.0
_EPOCH_MS = 1.7e12 + _OFFSETS_MS
_SIGNAL = np.sin(2.0 * np.pi * 3.0 * _OFFSETS_MS / 1000.0).astype(np.float32)


def _nustft(timestamps, ts_unit="ms"):
    return senpy_jax.compute_nustft(
        timestamps, _SIGNAL, window_s=8.0, overlap_s=4.0, ts_unit=ts_unit
    )


def test_host_epoch_timestamps_survive_the_device_float32_downcast(monkeypatch):
    _install(monkeypatch, _HostJax())

    absolute = _nustft(_EPOCH_MS)
    relative = _nustft(_OFFSETS_MS)

    assert absolute.coefficients.shape == relative.coefficients.shape
    assert absolute.coefficients.shape[0] > 0
    assert np.all(np.isfinite(absolute.coefficients))
    # Centering happens in float64, so only the float32 device representation of
    # the already-small offsets separates the two runs.
    assert np.allclose(absolute.coefficients, relative.coefficients, rtol=1e-3, atol=1e-6)


def test_host_centering_preserves_sample_spacing_exactly():
    t, gap = senpy_jax._to_centered_seconds(_HostJax(), np, _EPOCH_MS, "ms")

    assert gap == 0.0
    assert np.allclose(np.diff(t), 0.02, rtol=0.0, atol=1e-9)


def test_device_epoch_timestamps_are_rejected_with_a_precision_hint(monkeypatch):
    _install(monkeypatch, _DeviceJax())

    with pytest.raises(ValueError, match="jax_enable_x64"):
        _nustft(_EPOCH_MS.astype(np.float32))


def test_device_long_relative_timestamps_are_rejected_not_silently_rescaled(monkeypatch):
    _install(monkeypatch, _DeviceJax())
    # 30 h into a recording, float32 seconds: diffs survive but are quantized to
    # more than the sample spacing, which would otherwise skew median_fs.
    timestamps = (108_000.0 + _OFFSETS_MS / 1000.0).astype(np.float32)

    with pytest.raises(ValueError, match="cannot resolve its own sample spacing"):
        _nustft(timestamps, ts_unit="s")


def test_device_relative_timestamps_within_precision_are_accepted(monkeypatch):
    _install(monkeypatch, _DeviceJax())

    result = _nustft((_OFFSETS_MS / 1000.0).astype(np.float32), ts_unit="s")

    assert result.coefficients.shape[0] > 0
    assert np.all(np.isfinite(result.coefficients))


def test_device_integer_timestamps_are_exact_and_accepted(monkeypatch):
    _install(monkeypatch, _DeviceJax())

    result = _nustft((1_700_000_000_000 + _OFFSETS_MS).astype(np.int64))

    assert result.coefficients.shape[0] > 0
    assert np.all(np.isfinite(result.coefficients))
