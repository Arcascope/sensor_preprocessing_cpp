"""Optional integration tests for the JAX NUFFT API."""

import pytest


jax = pytest.importorskip("jax")
pytest.importorskip("jax_finufft")
jnp = pytest.importorskip("jax.numpy")

from senpy import jax_backend as senpy_jax


def test_compute_nustft_returns_device_arrays_with_known_tone():
    timestamps = jnp.arange(400, dtype=jnp.float32) / 50.0
    signal = jnp.sin(2.0 * jnp.pi * 3.0 * timestamps)

    result = senpy_jax.compute_nustft(
        timestamps,
        signal,
        window_s=4.0,
        overlap_s=2.0,
        target_fs=16.0,
    )

    assert result.axis_order == "time_frequency"
    assert result.coefficients.shape == (3, 33)
    assert result.frequencies.shape == (33,)
    assert result.times.shape == (3,)
    peak = result.frequencies[jnp.argmax(jnp.mean(result.power, axis=0))]
    assert float(peak) == pytest.approx(3.0)


def test_compute_nufft_spectrogram_and_welch_stay_jax_backed():
    timestamps = jnp.arange(400, dtype=jnp.float32) / 50.0
    signal = jnp.sin(2.0 * jnp.pi * 3.0 * timestamps)

    spectrogram = senpy_jax.compute_nufft_spectrogram(
        timestamps, signal, window_s=4.0, overlap_s=2.0, kind="power"
    )
    frequencies, welch = senpy_jax.compute_nufft_welch(
        timestamps, signal, window_s=4.0, overlap_s=2.0
    )

    assert spectrogram.Sxx.shape == (3, 129)
    assert frequencies.shape == welch.shape == (129,)
